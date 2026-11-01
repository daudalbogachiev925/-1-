"""Safety stock and reorder point computation using forecast distributions.

For each SKU we compute:
- mean forecast per day (mu)
- std of daily forecast residuals (sigma)
- safety stock = z * sigma * sqrt(L + R)
- reorder point = mu * (L + R) + safety_stock
- expected fill rate via normal approximation (or negative-binomial)
- expected stockout cost and holding cost given base stock
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy.stats import norm

from src.config import InventoryConfig

logger = logging.getLogger(__name__)


@dataclass
class InventoryPolicy:
    sku: str
    mu_daily: float
    sigma_daily: float
    lead_time: int
    review_period: int
    safety_stock: float
    reorder_point: float
    order_up_to: float
    service_level: float


def _z_for_service(service_level: float) -> float:
    return float(norm.ppf(np.clip(service_level, 1e-6, 1 - 1e-6)))


def compute_policies(
    forecast_df: pl.DataFrame,
    cfg: InventoryConfig,
    residuals_std: dict[str, float] | None = None,
) -> pl.DataFrame:
    """Compute per-SKU inventory policies.

    forecast_df: sku, date, forecast (daily)
    residuals_std: optional dict sku -> std of residuals
    """
    z = _z_for_service(cfg.service_level)
    L = cfg.lead_time_days
    R = cfg.review_period_days
    agg = (
        forecast_df.group_by("sku")
        .agg(
            pl.col("forecast").mean().alias("mu_daily"),
            pl.col("forecast").std().alias("sigma_daily"),
        )
    )
    rows = []
    for row in agg.iter_rows(named=True):
        sku = row["sku"]
        mu = max(float(row["mu_daily"] or 0.0), 1e-6)
        sigma = residuals_std.get(sku, float(row["sigma_daily"] or mu * 0.3)) if residuals_std else float(row["sigma_daily"] or mu * 0.3)
        sigma = max(sigma, 1e-6)
        ss = z * sigma * np.sqrt(L + R)
        rop = mu * (L + R) + ss
        out = mu * (L + R) + ss
        rows.append(
            {
                "sku": sku,
                "mu_daily": mu,
                "sigma_daily": sigma,
                "lead_time": L,
                "review_period": R,
                "safety_stock": float(ss),
                "reorder_point": float(rop),
                "order_up_to": float(out),
                "service_level": cfg.service_level,
            }
        )
    return pl.DataFrame(rows)


def simulate_policy(
    demand_df: pl.DataFrame,
    policies: pl.DataFrame,
    cfg: InventoryConfig,
    seed: int = 42,
) -> pl.DataFrame:
    """Simulate weekly inventory operation per SKU given a policy.

    Returns per-SKU metrics: fill_rate, stockout_rate, avg_inventory, total_cost.
    """
    rng = np.random.default_rng(seed)
    L = cfg.lead_time_days
    R = cfg.review_period_days
    pol_map = {row["sku"]: row for row in policies.iter_rows(named=True)}
    metrics = []
    for sku, group in demand_df.group_by("sku"):
        sku = sku[0] if isinstance(sku, tuple) else sku
        pol = pol_map.get(str(sku))
        if pol is None:
            continue
        demand = group.sort("date")["demand"].to_numpy().astype(float)
        n = len(demand)
        inventory = float(pol["order_up_to"])
        pipeline: list[tuple[int, float]] = []  # (arrival_day, qty)
        served = 0.0
        total_demand = demand.sum()
        inventory_trace: list[float] = []
        stockouts = 0
        orders = 0
        for t in range(n):
            # receive orders
            pipeline = [(day, q) for (day, q) in pipeline if day > t] + [
                (day, q) for (day, q) in pipeline if day == t
            ]
            arrived = sum(q for (day, q) in pipeline if day <= t)
            pipeline = [(day, q) for (day, q) in pipeline if day > t]
            inventory += arrived
            # serve demand
            d = demand[t]
            if inventory >= d:
                inventory -= d
                served += d
            else:
                served += inventory
                stockouts += 1
                inventory = 0.0
            inventory_trace.append(inventory)
            # periodic review at end of each R days
            if (t + 1) % R == 0 and (t + 1) < n:
                position = inventory + sum(q for (_, q) in pipeline)
                if position <= pol["reorder_point"]:
                    order_qty = max(0.0, pol["order_up_to"] - position)
                    arrival = t + L
                    pipeline.append((arrival, order_qty))
                    orders += 1
        fill_rate = served / total_demand if total_demand > 0 else 1.0
        stockout_rate = stockouts / max(1, n)
        avg_inventory = float(np.mean(inventory_trace)) if inventory_trace else 0.0
        holding_cost = avg_inventory * cfg.holding_cost_per_unit * n
        stockout_cost = (total_demand - served) * cfg.stockout_cost_per_unit
        metrics.append(
            {
                "sku": str(sku),
                "fill_rate": float(fill_rate),
                "stockout_rate": float(stockout_rate),
                "avg_inventory": float(avg_inventory),
                "orders_placed": int(orders),
                "holding_cost": float(holding_cost),
                "stockout_cost": float(stockout_cost),
                "total_cost": float(holding_cost + stockout_cost),
            }
        )
    return pl.DataFrame(metrics)


def baseline_policy(
    demand_df: pl.DataFrame, cfg: InventoryConfig
) -> pl.DataFrame:
    """Naive policy: order_up_to = mean*lead_time * 1.5, no stats."""
    agg = demand_df.group_by("sku").agg(pl.col("demand").mean().alias("mu"))
    return agg.with_columns(
        (pl.col("mu") * cfg.lead_time_days * 1.5).alias("order_up_to"),
        (pl.col("mu") * cfg.lead_time_days * 1.5 * 0.6).alias("reorder_point"),
        (pl.col("mu") * 0.0 + 0.0).alias("safety_stock"),
        (pl.col("mu") * cfg.lead_time_days).alias("mu_daily"),
        (pl.col("mu") * 0.3).alias("sigma_daily"),
        pl.lit(cfg.lead_time_days).alias("lead_time"),
        pl.lit(cfg.review_period_days).alias("review_period"),
        pl.lit(cfg.service_level).alias("service_level"),
    ).select(
        ["sku", "mu_daily", "sigma_daily", "lead_time", "review_period",
         "safety_stock", "reorder_point", "order_up_to", "service_level"]
    )
