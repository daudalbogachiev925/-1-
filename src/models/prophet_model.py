"""Per-SKU Prophet baseline forecaster."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import polars as pl
from prophet import Prophet

from src.config import ProphetConfig
from src.models.base import BaseForecaster

logger = logging.getLogger(__name__)


class ProphetForecaster(BaseForecaster):
    name = "prophet"

    def __init__(self, config: ProphetConfig):
        self.config = config
        self.models: dict[str, Prophet] = {}

    def fit(self, train: pl.DataFrame, feature_cols: list[str], target: str = "demand") -> "ProphetForecaster":
        self.models = {}
        pdf = train.select(["date", "sku", "demand", "promo", "holiday", "temperature"]).to_pandas()
        for sku, group in pdf.groupby("sku"):
            df = group[["date", "demand", "promo", "holiday", "temperature"]].rename(
                columns={"date": "ds", "demand": "y"}
            )
            df = df.sort_values("ds").reset_index(drop=True)
            m = Prophet(
                changepoint_prior_scale=self.config.changepoint_prior_scale,
                seasonality_prior_scale=self.config.seasonality_prior_scale,
                weekly_seasonality=True,
                yearly_seasonality=True,
                daily_seasonality=False,
            )
            m.add_regressor("promo")
            m.add_regressor("holiday")
            m.add_regressor("temperature")
            m.fit(df)
            self.models[str(sku)] = m
        logger.info("Prophet fitted for %s SKUs", len(self.models))
        return self

    def predict(self, df: pl.DataFrame, feature_cols: list[str]) -> np.ndarray:
        out = np.zeros(df.height, dtype=np.float64)
        pdf = df.select(["date", "sku", "promo", "holiday", "temperature"]).to_pandas()
        for i, row in pdf.iterrows():
            sku = str(row["sku"])
            m = self.models.get(sku)
            if m is None:
                continue
            future = pd.DataFrame(
                {
                    "ds": [pd.Timestamp(row["date"])],
                    "promo": [row["promo"]],
                    "holiday": [row["holiday"]],
                    "temperature": [row["temperature"]],
                }
            )
            forecast = m.predict(future)
            out[i] = max(0.0, float(forecast["yhat"].iloc[0]))
        return out
