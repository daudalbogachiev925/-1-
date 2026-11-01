"""API tests using FastAPI TestClient with mocked artifacts."""
from __future__ import annotations

import pickle
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from fastapi.testclient import TestClient


class _DummyModel:
    """Deterministic dummy model for API smoke tests."""

    def predict(self, df: pl.DataFrame, cols: list[str]) -> np.ndarray:
        return np.full(df.height, 5.0)


def _make_history(n: int = 120) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    start = date(2023, 1, 1)
    dates = [start + timedelta(days=i) for i in range(n)]
    return pl.DataFrame(
        {
            "date": dates,
            "sku": ["SKU-A"] * n,
            "region": ["North"] * n,
            "category": ["Electronics"] * n,
            "demand": rng.uniform(1, 10, n),
            "price": [10.0] * n,
            "promo": [0] * n,
            "holiday": [0] * n,
            "temperature": [15.0] * n,
            "is_weekend": [int(d.weekday() >= 5) for d in dates],
            "month": [d.month for d in dates],
            "day_of_week": [d.weekday() for d in dates],
        }
    )


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    history = _make_history()
    history_path = tmp_path / "history.parquet"
    history.write_parquet(history_path)

    policies = pl.DataFrame(
        {
            "sku": ["SKU-A"],
            "mu_daily": [10.0],
            "sigma_daily": [2.0],
            "lead_time": [7],
            "review_period": [7],
            "safety_stock": [10.0],
            "reorder_point": [150.0],
            "order_up_to": [200.0],
            "service_level": [0.95],
        }
    )
    policies_path = tmp_path / "policies.parquet"
    policies.write_parquet(policies_path)

    feature_cols = [
        "promo", "holiday", "temperature", "is_weekend", "month", "day_of_week",
        "price", "cal_weekday", "cal_month", "cal_quarter", "cal_day_of_year",
        "cal_is_weekend", "region_code", "category_code", "sku_code",
        "promo_price", "promo_x_lag7", "holiday_x_weekday",
        "lag_demand_1", "lag_demand_7", "lag_demand_14", "lag_demand_28",
        "lag_price_1", "lag_price_7", "lag_price_14", "lag_price_28",
        "roll_mean_7", "roll_std_7", "roll_max_7",
        "roll_mean_14", "roll_std_14", "roll_max_14",
        "roll_mean_28", "roll_std_28", "roll_max_28",
        "roll_mean_56", "roll_std_56", "roll_max_56",
        "week_sin_1", "week_cos_1", "week_sin_2", "week_cos_2",
        "week_sin_3", "week_cos_3",
        "year_sin_1", "year_cos_1", "year_sin_2", "year_cos_2",
        "year_sin_3", "year_cos_3",
    ]

    with (tmp_path / "model.pkl").open("wb") as f:
        pickle.dump({"model": _DummyModel(), "feature_cols": feature_cols}, f)
    with (tmp_path / "metadata.pkl").open("wb") as f:
        pickle.dump({"last_date": str(history["date"].max()), "metrics": {}, "model_type": "_DummyModel"}, f)

    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path))
    monkeypatch.setenv("DATA_DIR", str(tmp_path))

    # reset caches so new env is picked up
    from src.config import get_settings, get_app_config
    from src.api import deps

    get_settings.cache_clear()
    get_app_config.cache_clear()
    deps.get_artifacts.cache_clear()

    from src.api.main import app

    with TestClient(app) as c:
        yield c


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_model_info(client: TestClient) -> None:
    r = client.get("/model-info")
    assert r.status_code == 200
    data = r.json()
    assert data["model_type"] == "_DummyModel"
    assert "last_training_date" in data


def test_forecast_ok(client: TestClient) -> None:
    r = client.get("/forecast", params={"sku": "SKU-A", "horizon": 7})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["sku"] == "SKU-A"
    assert data["horizon"] == 7
    assert len(data["points"]) == 7
    assert data["total"] == pytest.approx(35.0)


def test_forecast_unknown_sku(client: TestClient) -> None:
    r = client.get("/forecast", params={"sku": "UNKNOWN", "horizon": 7})
    assert r.status_code == 404


def test_forecast_invalid_horizon(client: TestClient) -> None:
    r = client.get("/forecast", params={"sku": "SKU-A", "horizon": 0})
    assert r.status_code == 422


def test_inventory_ok(client: TestClient) -> None:
    r = client.get("/inventory/SKU-A")
    assert r.status_code == 200
    data = r.json()
    assert data["sku"] == "SKU-A"
    assert data["reorder_point"] == 150.0


def test_inventory_unknown(client: TestClient) -> None:
    r = client.get("/inventory/MISSING")
    assert r.status_code == 404


def test_segments(client: TestClient) -> None:
    r = client.get("/segments")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)
    assert len(data) >= 1
    assert data[0]["sku"] == "SKU-A"
    assert data[0]["abc_class"] in {"A", "B", "C"}
    assert data[0]["xyz_class"] in {"X", "Y", "Z"}
