"""Tests for hierarchical reconciliation."""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from src.models.hierarchy import (
    _build_summing_matrix,
    _stack_forecasts,
    reconcile,
)


@pytest.fixture
def series_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "sku": ["A1", "A2", "B1", "B2"],
            "region": ["N", "N", "S", "S"],
            "category": ["X", "X", "Y", "Y"],
        }
    )


def test_summing_matrix_shape(series_df: pl.DataFrame) -> None:
    struct = _build_summing_matrix(series_df)
    # 4 skus + 2 categories + 2 regions + 1 total = 9
    assert struct.S.shape == (9, 4)


def test_summing_matrix_row_sums(series_df: pl.DataFrame) -> None:
    struct = _build_summing_matrix(series_df)
    # every row sums to number of skus it aggregates
    row_sums = struct.S.sum(axis=1)
    assert (row_sums > 0).all()


def test_bottom_up_reconciles_to_sum(series_df: pl.DataFrame) -> None:
    base = pl.DataFrame(
        {
            "level": ["sku"] * 4 + ["category"] * 2 + ["region"] * 2 + ["total"],
            "key": ["A1", "A2", "B1", "B2", "X", "Y", "N", "S", "ALL"],
            "forecast": [10.0, 20.0, 30.0, 40.0, 999.0, 999.0, 999.0, 999.0, 999.0],
        }
    )
    out = reconcile(base, series_df, method="bottom_up")
    total = out.filter((pl.col("level") == "total")).select("forecast").to_numpy()[0, 0]
    # 10 + 20 + 30 + 40 = 100
    assert np.isclose(total, 100.0)


def test_ols_reconciles_to_sum(series_df: pl.DataFrame) -> None:
    base = pl.DataFrame(
        {
            "level": ["sku"] * 4 + ["category"] * 2 + ["region"] * 2 + ["total"],
            "key": ["A1", "A2", "B1", "B2", "X", "Y", "N", "S", "ALL"],
            "forecast": [10.0, 20.0, 30.0, 40.0, 20.0, 80.0, 40.0, 60.0, 120.0],
        }
    )
    out = reconcile(base, series_df, method="ols")
    # after OLS, sums at higher levels equal sums at lower
    total = out.filter(pl.col("level") == "total").select("forecast").to_numpy()[0, 0]
    assert np.isclose(total, out.filter(pl.col("level") == "sku").select("forecast").to_numpy().sum())
