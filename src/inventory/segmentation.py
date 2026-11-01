"""ABC / XYZ segmentation for inventory policies."""
from __future__ import annotations

import numpy as np
import polars as pl


def abc_classification(df: pl.DataFrame, value_col: str = "demand") -> pl.DataFrame:
    """Assign A/B/C based on cumulative share of total volume."""
    agg = (
        df.group_by("sku")
        .agg(pl.col(value_col).sum().alias("total_value"))
        .sort("total_value", descending=True)
    )
    total = float(agg["total_value"].sum())
    if total == 0:
        return agg.with_columns(pl.lit("C").alias("abc_class"))
    agg = agg.with_columns((pl.col("total_value").cum_sum() / total).alias("cum_share"))
    agg = agg.with_columns(
        pl.when(pl.col("cum_share") <= 0.8).then(pl.lit("A"))
        .when(pl.col("cum_share") <= 0.95).then(pl.lit("B"))
        .otherwise(pl.lit("C"))
        .alias("abc_class")
    )
    return agg.select(["sku", "total_value", "cum_share", "abc_class"])


def xyz_classification(df: pl.DataFrame, value_col: str = "demand") -> pl.DataFrame:
    """XYZ based on coefficient of variation of weekly demand."""
    weekly = (
        df.with_columns(pl.col("date").dt.week().alias("week"))
        .group_by(["sku", "week"])
        .agg(pl.col(value_col).sum().alias("weekly"))
        .group_by("sku")
        .agg(
            pl.col("weekly").mean().alias("mean_w"),
            pl.col("weekly").std().alias("std_w"),
        )
    )
    weekly = weekly.with_columns(
        pl.when(pl.col("mean_w") > 0)
        .then(pl.col("std_w") / pl.col("mean_w"))
        .otherwise(0.0)
        .alias("cv")
    )
    weekly = weekly.with_columns(
        pl.when(pl.col("cv") <= 0.25).then(pl.lit("X"))
        .when(pl.col("cv") <= 0.5).then(pl.lit("Y"))
        .otherwise(pl.lit("Z"))
        .alias("xyz_class")
    )
    return weekly.select(["sku", "mean_w", "std_w", "cv", "xyz_class"])


def abc_xyz_matrix(df: pl.DataFrame, value_col: str = "demand") -> pl.DataFrame:
    abc = abc_classification(df, value_col)
    xyz = xyz_classification(df, value_col)
    return abc.join(xyz, on="sku", how="inner")
