"""FastAPI application: /health, /forecast, /inventory/{sku}, /segments, /model-info."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import date, timedelta
from typing import AsyncIterator

import numpy as np
import polars as pl
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from src.api.deps import ModelArtifacts, get_artifacts
from src.api.schemas import (
    ErrorResponse,
    ForecastPoint,
    ForecastResponse,
    HealthResponse,
    InventoryPolicyResponse,
    SegmentResponse,
)
from src.config import get_settings
from src.features.build import build_features
from src.inventory.segmentation import abc_classification, xyz_classification

logger = logging.getLogger(__name__)
logging.basicConfig(level=get_settings().log_level)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("Starting API lifespan: loading artifacts...")
    get_artifacts()
    logger.info("Artifacts loaded.")
    yield
    logger.info("Shutdown.")


app = FastAPI(
    title="Demand Forecast & Inventory Optimization API",
    version="0.1.0",
    lifespan=lifespan,
    responses={500: {"model": ErrorResponse}},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", version=app.version)


@app.get("/model-info")
def model_info(artifacts: ModelArtifacts = Depends(get_artifacts)) -> dict:
    if not artifacts.is_ready:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "model_type": type(artifacts.model).__name__,
        "n_features": len(artifacts.feature_cols),
        "features": artifacts.feature_cols,
        "last_training_date": artifacts.last_date,
        "metadata": artifacts.metadata,
    }


def _forecast_future(
    artifacts: ModelArtifacts,
    sku: str,
    horizon: int,
    start_date: date | None = None,
) -> pl.DataFrame:
    """Build synthetic future rows by extending history with structural features.

    For simplicity, future exogenous variables (price, promo, holiday, temperature)
    are projected from recent averages. Production systems should plug in
    planning data instead.
    """
    assert artifacts.demand_history is not None
    history = artifacts.demand_history
    sku_hist = history.filter(pl.col("sku") == sku).sort("date")
    if sku_hist.height == 0:
        raise HTTPException(status_code=404, detail=f"Unknown SKU: {sku}")

    last_date_str = str(sku_hist["date"].max())
    last_dt = date.fromisoformat(last_date_str)
    forecast_start = start_date or (last_dt + timedelta(days=1))

    # build future frame with structural columns
    future_dates = [forecast_start + timedelta(days=i) for i in range(horizon)]
    last_row = sku_hist.tail(1).to_dicts()[0]

    # approximate exogenous variables
    recent_price = float(sku_hist.tail(28)["price"].mean())
    recent_temp = float(sku_hist.tail(28)["temperature"].mean())

    future = pl.DataFrame(
        {
            "date": future_dates,
            "sku": [sku] * horizon,
            "region": [last_row["region"]] * horizon,
            "category": [last_row["category"]] * horizon,
            "demand": [0.0] * horizon,
            "price": [recent_price] * horizon,
            "promo": [0] * horizon,
            "holiday": [
                1 if d.month == 12 and d.day == 25 else 0 for d in future_dates
            ],
            "temperature": [recent_temp] * horizon,
            "is_weekend": [int(d.weekday() >= 5) for d in future_dates],
            "month": [d.month for d in future_dates],
            "day_of_week": [d.weekday() for d in future_dates],
        }
    )

    combined = pl.concat([sku_hist, future], how="vertical").sort("date")
    bundle = build_features(combined)
    feature_frame = bundle.frame.filter(pl.col("date") >= pl.lit(forecast_start))
    if feature_frame.height == 0:
        raise HTTPException(status_code=500, detail="Could not build features for forecast")
    return feature_frame


@app.get("/forecast", response_model=ForecastResponse)
def forecast(
    sku: str = Query(..., description="SKU identifier"),
    horizon: int = Query(28, ge=1, le=90),
    start_date: date | None = Query(None),
    artifacts: ModelArtifacts = Depends(get_artifacts),
) -> ForecastResponse:
    if not artifacts.is_ready:
        raise HTTPException(status_code=503, detail="Model not loaded")
    feature_frame = _forecast_future(artifacts, sku, horizon, start_date)
    try:
        preds = artifacts.model.predict(feature_frame, artifacts.feature_cols)
    except Exception as exc:  # pragma: no cover
        logger.exception("Prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
    preds = np.clip(np.asarray(preds, dtype=float), 0.0, None)
    dates = feature_frame["date"].to_list()
    points = [
        ForecastPoint(date=d, forecast=float(v)) for d, v in zip(dates, preds)
    ]
    return ForecastResponse(
        sku=sku,
        horizon=horizon,
        model=type(artifacts.model).__name__,
        points=points,
        total=float(preds.sum()),
    )


@app.get("/inventory/{sku}", response_model=InventoryPolicyResponse)
def inventory(
    sku: str,
    artifacts: ModelArtifacts = Depends(get_artifacts),
) -> InventoryPolicyResponse:
    if artifacts.policies is None:
        raise HTTPException(status_code=503, detail="Policies not loaded")
    row = artifacts.policies.filter(pl.col("sku") == sku)
    if row.height == 0:
        raise HTTPException(status_code=404, detail=f"No policy for SKU: {sku}")
    r = row.to_dicts()[0]
    return InventoryPolicyResponse(**r)


@app.get("/segments", response_model=list[SegmentResponse])
def segments(
    artifacts: ModelArtifacts = Depends(get_artifacts),
) -> list[SegmentResponse]:
    if artifacts.demand_history is None:
        raise HTTPException(status_code=503, detail="Demand history not loaded")
    abc = abc_classification(artifacts.demand_history)
    xyz = xyz_classification(artifacts.demand_history)
    combined = abc.join(xyz, on="sku", how="inner")
    return [
        SegmentResponse(sku=r["sku"], abc_class=r["abc_class"], xyz_class=r["xyz_class"])
        for r in combined.iter_rows(named=True)
    ]
