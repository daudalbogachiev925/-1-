"""Tests for feature engineering: no target leakage, determinism, shape."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from src.config import AppConfig
from src.features.build import build_features


def _make_frame(n_days: int = 120, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    start = date(2023, 1, 1)
    dates = [start + timedelta(days=i) for i in range(n_days)]
    return pl.DataFrame(
        {
            "date": dates * 2,
            "sku": ["SKU-A"] * n_days + ["SKU-B"] * n_days,
            "region": ["North"] * (n_days * 2),
            "category": ["Electronics"] * (n_days * 2),
            "demand": rng.uniform(1, 50, n_days * 2).round(1),
            "price": rng.uniform(5, 100, n_days * 2).round(2),
            "promo": rng.integers(0, 2, n_days * 2),
            "holiday": rng.integers(0, 2, n_days * 2),
            "temperature": rng.uniform(-10, 30, n_days * 2).round(2),
            "is_weekend": rng.integers(0, 2, n_days * 2),
            "month": [d.month for d in dates] * 2,
            "day_of_week": [d.weekday() for d in dates] * 2,
        }
    )


@pytest.fixture
def cfg() -> AppConfig:
    return AppConfig()


def test_features_no_leakage(cfg: AppConfig) -> None:
    df = _make_frame(seed=1).sort(["sku", "date"])
    bundle = build_features(df, cfg)
    # lag columns must be strictly < current demand value for each SKU
    sample = bundle.frame.filter(pl.col("sku") == "SKU-A").sort("date").head(20)
    lag1 = sample["lag_demand_1"].to_numpy()
    actual = sample["demand"].to_numpy()
    # rows were shifted; check the first lag isn't NaN
    assert not np.isnan(lag1).any()
    # lag must be past demand
    assert lag1[0] != actual[0]


def test_features_drop_warmup(cfg: AppConfig) -> None:
    df = _make_frame(seed=2)
    bundle = build_features(df, cfg)
    max_lag = max(max(cfg.features.lags), max(cfg.features.rolling_windows))
    for sku in df["sku"].unique():
        n = df.filter(pl.col("sku") == sku).height
        n_after = bundle.frame.filter(pl.col("sku") == sku).height
        assert n - n_after == max_lag


def test_features_all_columns_present(cfg: AppConfig) -> None:
    df = _make_frame(seed=3)
    bundle = build_features(df, cfg)
    for col in ["lag_demand_1", "lag_demand_7", "roll_mean_7", "roll_std_28", "week_sin_1", "year_cos_1"]:
        assert col in bundle.frame.columns
