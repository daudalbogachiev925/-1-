"""Tests for inventory policies and simulation."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from src.config import InventoryConfig
from src.inventory.safety_stock import (
    baseline_policy,
    compute_policies,
    simulate_policy,
)


def _forecast_df(n_days: int = 60) -> pl.DataFrame:
    dates = [date(2023, 1, 1) + timedelta(days=i) for i in range(n_days)]
    rng = np.random.default_rng(0)
    return pl.DataFrame(
        {
            "sku": ["SKU-A"] * n_days,
            "date": dates,
            "forecast": rng.uniform(5, 15, n_days).round(2),
        }
    )


def _demand_df(n_days: int = 120) -> pl.DataFrame:
    dates = [date(2023, 1, 1) + timedelta(days=i) for i in range(n_days)]
    rng = np.random.default_rng(1)
    return pl.DataFrame(
        {
            "sku": ["SKU-A"] * n_days,
            "date": dates,
            "demand": rng.uniform(5, 15, n_days).round(2),
        }
    )


@pytest.fixture
def inventory_cfg() -> InventoryConfig:
    return InventoryConfig(
        service_level=0.95,
        lead_time_days=7,
        review_period_days=7,
        holding_cost_per_unit=0.5,
        stockout_cost_per_unit=5.0,
    )


def test_policies_computed(inventory_cfg: InventoryConfig) -> None:
    policies = compute_policies(_forecast_df(), inventory_cfg)
    assert policies.height == 1
    row = policies.to_dicts()[0]
    assert row["safety_stock"] > 0
    assert row["reorder_point"] > row["mu_daily"] * inventory_cfg.lead_time_days


def test_simulation_returns_metrics(inventory_cfg: InventoryConfig) -> None:
    policies = compute_policies(_forecast_df(), inventory_cfg)
    sim = simulate_policy(_demand_df(), policies, inventory_cfg)
    assert sim.height == 1
    row = sim.to_dicts()[0]
    assert 0.0 <= row["fill_rate"] <= 1.0
    assert row["total_cost"] >= 0.0


def test_baseline_runs(inventory_cfg: InventoryConfig) -> None:
    demand = _demand_df()
    policies = baseline_policy(demand, inventory_cfg)
    assert policies.height == 1
    sim = simulate_policy(demand, policies, inventory_cfg)
    assert sim.height == 1
