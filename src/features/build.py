"""Feature engineering for demand forecasting.

Produces lag features, rolling stats, calendar features, Fourier terms,
price/promo interactions and holiday flags. Keeps rows for validation.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import polars as pl

from src.config import AppConfig, get_app_config

logger = logging.getLogger(__name__)


@dataclass
class FeatureBundle:
    frame: pl.DataFrame
    feature_cols: list[str]
    target_col: str = "demand"


def _fourier_terms(df: pl.DataFrame, orders: list[int], period: int, prefix: str) -> pl.DataFrame:
    t = df.select(pl.col("date").cast(pl.Int64) // 86_400_000_000_000).to_series().to_numpy()
    expr_list = []
    for k in orders:
        expr_list.append(
            pl.Series(f"{prefix}_sin_{k}", np.sin(2 * np.pi * k * t / period))
        )
        expr_list.append(
            pl.Series(f"{prefix}_cos_{k}", np.cos(2 * np.pi * k * t / period))
        )
    return pl.DataFrame(expr_list)


def build_features(df: pl.DataFrame, cfg: AppConfig | None = None) -> FeatureBundle:
    cfg = cfg or get_app_config()
    fc = cfg.features
    df = df.sort(["sku", "date"])

    # calendar
    df = df.with_columns(
        pl.col("date").dt.weekday().alias("cal_weekday"),
        pl.col("date").dt.month().alias("cal_month"),
        pl.col("date").dt.quarter().alias("cal_quarter"),
        pl.col("date").dt.ordinal_day().alias("cal_day_of_year"),
        (pl.col("date").dt.weekday() >= 6).cast(pl.Int8).alias("cal_is_weekend"),
    )

    # Fourier terms
    fourier = _fourier_terms(df, fc.fourier_orders, period=7, prefix="week")
    df = pl.concat([df, fourier], how="horizontal")
    fourier_year = _fourier_terms(df, fc.fourier_orders, period=365, prefix="year")
    df = pl.concat([df, fourier_year], how="horizontal")

    # lags on demand and price
    lag_exprs = []
    for L in fc.lags:
        lag_exprs.append(pl.col("demand").shift(L).over("sku").alias(f"lag_demand_{L}"))
        lag_exprs.append(pl.col("price").shift(L).over("sku").alias(f"lag_price_{L}"))
    df = df.with_columns(lag_exprs)

    # rolling stats on demand (past only)
    roll_exprs = []
    for W in fc.rolling_windows:
        roll_exprs.append(
            pl.col("demand").shift(1).rolling_mean(W).over("sku").alias(f"roll_mean_{W}")
        )
        roll_exprs.append(
            pl.col("demand").shift(1).rolling_std(W).over("sku").alias(f"roll_std_{W}")
        )
        roll_exprs.append(
            pl.col("demand").shift(1).rolling_max(W).over("sku").alias(f"roll_max_{W}")
        )
    df = df.with_columns(roll_exprs)

    # promo interactions
    df = df.with_columns(
        (pl.col("promo") * pl.col("price")).alias("promo_price"),
        (pl.col("promo") * pl.col("lag_demand_7")).alias("promo_x_lag7"),
        (pl.col("holiday") * pl.col("cal_weekday")).alias("holiday_x_weekday"),
    )

    # SKU static features (region/category as categorical codes)
    df = df.with_columns(
        pl.col("region").cast(pl.Categorical).to_physical().cast(pl.Int32).alias("region_code"),
        pl.col("category").cast(pl.Categorical).to_physical().cast(pl.Int32).alias("category_code"),
        pl.col("sku").cast(pl.Categorical).to_physical().cast(pl.Int32).alias("sku_code"),
    )

    base_cols = [
        "promo", "holiday", "temperature", "is_weekend", "month", "day_of_week",
        "price", "cal_weekday", "cal_month", "cal_quarter", "cal_day_of_year", "cal_is_weekend",
        "region_code", "category_code", "sku_code",
        "promo_price", "promo_x_lag7", "holiday_x_weekday",
    ]
    lag_cols = [c for c in df.columns if c.startswith(("lag_", "roll_"))]
    fourier_cols = [c for c in df.columns if c.startswith(("week_", "year_"))]
    feature_cols = base_cols + lag_cols + fourier_cols

    # drop rows with NaNs induced by lags/rolling
    max_lag = max(max(fc.lags), max(fc.rolling_windows))
    df = df.with_columns(
        pl.col("demand").is_not_null().alias("_ok")
    )
    # filter first max_lag rows per SKU
    df = df.with_columns(
        pl.int_range(pl.len()).over("sku").alias("_idx")
    ).filter(pl.col("_idx") >= max_lag).drop(["_idx", "_ok"])

    logger.info("Feature frame: %s rows, %s features", df.height, len(feature_cols))
    return FeatureBundle(frame=df, feature_cols=feature_cols)


def train_val_split(
    bundle: FeatureBundle, cutoff_date: str
) -> tuple[pl.DataFrame, pl.DataFrame]:
    cutoff = pl.lit(cutoff_date).str.strptime(pl.Date, "%Y-%m-%d")
    train = bundle.frame.filter(pl.col("date") < cutoff)
    val = bundle.frame.filter(pl.col("date") >= cutoff)
    return train, val
