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
    def predict(self, df: pl.DataFrame, cols: list[str]) -> np.ndarray:
        return np.full(df.height, 5.0)


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    # write dummy artifacts
    history = pl.DataFrame(
        {
            "date": [date(2023, 1, 1) + timedelta(days=i) for i in range(60)],
            "sku": ["SKU-A"] * 60,
            "region": ["North"] * 60,
            "category": ["Electronics"] * 60,
            "demand": np.random.default_rng(0).uniform(1, 10, 60),
            "price": [10.0] * 60,
            "promo": [0] * 60,
            "holiday": [0] * 60,
            "temperature": [15.0] * 60,
            "is_weekend": [0] * 60,
            "month": [1] * 60,
            "day_of_week": [0] * 60,
        }
    )
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

    feature_cols =
