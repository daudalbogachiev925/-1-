

"""Hierarchical reconciliation: OLS, WLS, MinT-Sample.

Given base forecasts at multiple aggregation levels, reconciles them so
sums at higher levels equal sums of lower levels.

We implement:
- bottom-up (baseline)
- OLS (ordinary least squares)
- WLS (weighted least squares with variance weighting)
- MinT-Sample (sample covariance of residuals)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


@dataclass
class HierarchyStructure:
    """Summing matrix S for a given hierarchy.

    levels: list of level names bottom-up: [sku, category, region, total]
    mapping: mapping series that define S; each column is a leaf-level series (SKU).
    """
    levels: list[str]
    level_index: dict[str, int]
    S: np.ndarray  # shape (n_nodes, n_bottom)


def _build_summing_matrix(series_df: pl.DataFrame) -> HierarchyStructure:
    """Build S using available columns region/category/sku.

    Nodes are defined as (level, key) tuples:
      - level 'sku': per-SKU series
      - level 'category': sum over skus of a category
      - level 'region': sum over skus of a region
      - level 'total': grand total
    """
    skus = sorted(series_df["sku"].unique().to_list())
    sku_to_cat = dict(zip(series_df["sku"].to_list(), series_df["category"].to_list()))
    sku_to_region = dict(zip(series_df["sku"].to_list(), series_df["region"].to_list()))

    nodes: list[tuple[str, str]] = []
    for sku in skus:
        nodes.append(("sku", sku))
    for cat in sorted(set(sku_to_cat.values())):
        nodes.append(("category", cat))
    for reg in sorted(set(sku_to_region.values())):
        nodes.append(("region", reg))
    nodes.append(("total", "ALL"))

    n_nodes = len(nodes)
    n_bottom = len(skus)
    S = np.zeros((n_nodes, n_bottom), dtype=np.float64)
    for i, (level, key) in enumerate(nodes):
        if level == "sku":
            j = skus.index(key)
            S[i, j] = 1.0
        elif level == "category":
            for j, sku in enumerate(skus):
                if sku_to_cat[sku] == key:
                    S[i, j] = 1.0
        elif level == "region":
            for j, sku in enumerate(skus):
                if sku_to_region[sku] == key:
                    S[i, j] = 1.0
        elif level == "total":
            S[i, :] = 1.0
    level_index = {}
    for i, (level, _) in enumerate(nodes):
        level_index.setdefault(level, i)
    return HierarchyStructure(
        levels=["sku", "category", "region", "total"],
        level_index=level_index,
        S=S,
    )


def _stack_forecasts(base: pl.DataFrame) -> tuple[np.ndarray, list[tuple[str, str]]]:
    """Convert per-node forecast frame into ordered y_hat vector + node list."""
    nodes: list[tuple[str, str]] = []
    values: list[float] = []
    for level in ["sku", "category", "region", "total"]:
        sub = base.filter(pl.col("level") == level)
        for row in sub.iter_rows(named=True):
            nodes.append((level, row["key"]))
            values.append(float(row["forecast"]))
    return np.array(values, dtype=np.float64), nodes


def reconcile(
    base_forecast: pl.DataFrame,
    series_df: pl.DataFrame,
    method: str = "mint_sample",
    residuals: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Reconcile base forecasts across levels.

    base_forecast columns: level, key, forecast
    series_df columns: sku, region, category
    residuals (optional): level, key, residual — used for MinT-Sample
    Returns: level, key, forecast (reconciled)
    """
    struct = _build_summing_matrix(series_df)
    y_hat, nodes = _stack_forecasts(base_forecast)
    if len(nodes) != struct.S.shape[0]:
        raise ValueError(f"Node count mismatch: {len(nodes)} vs {struct.S.shape[0]}")
    # align column ordering of base_forecast with struct.S rows
    order = {node: i for i, node in enumerate(nodes)}
    reordered_rows = []
    for i, (level, key) in enumerate(
        [(struct.levels[0], sku) for sku in sorted(series_df["sku"].unique().to_list())]
        + [("category", c) for c in sorted(set(series_df["category"].to_list()))]
        + [("region", r) for r in sorted(set(series_df["region"].to_list()))]
        + [("total", "ALL")]
    ):
        reordered_rows.append(y_hat[order[(level, key)]])
    y_hat = np.array(reordered_rows)

    S = struct.S
    n = S.shape[0]
    m = S.shape[1]

    if method == "bottom_up":
        bottom = y_hat[-m:]
        reconciled = S @ bottom
    elif method == "ols":
        # y_tilde = S (S^T S)^-1 S^T y_hat
        P = S @ np.linalg.pinv(S.T @ S) @ S.T
        reconciled = P @ y_hat
    elif method == "wls":
        # W = diag(1/variance), variance approximated by y_hat (poisson-like)
        v = np.clip(y_hat, 1e-3, None)
        W = np.diag(1.0 / v)
        P = S @ np.linalg.pinv(S.T @ W @ S) @ S.T @ W
        reconciled = P @ y_hat
    elif method == "mint_sample":
        if residuals is None:
            logger.warning("mint_sample requested but residuals missing; falling back to WLS")
            v = np.clip(y_hat, 1e-3, None)
            W = np.diag(1.0 / v)
            P = S @ np.linalg.pinv(S.T @ W @ S) @ S.T @ W
            reconciled = P @ y_hat
        else:
            # residuals: level, key, residual, timestamp
            # pivot residuals to (timestamps x nodes) matrix
            res_pivot = (
                residuals.pivot(index="timestamp", columns=["level", "key"], values="residual")
                .sort("timestamp")
            )
            # build aligned array
            cols = []
            for level, key in nodes:
                colname = (level, key)
                if colname in res_pivot.columns:
                    cols.append(res_pivot[colname].to_numpy())
                else:
                    cols.append(np.zeros(res_pivot.height))
            R = np.stack(cols, axis=1)  # (T, n)
            cov = np.cov(R, rowvar=False)
            cov = cov + np.eye(n) * 1e-6
            W = np.linalg.pinv(cov)
            P = S @ np.linalg.pinv(S.T @ W @ S) @ S.T @ W
            reconciled = P @ y_hat
    else:
        raise ValueError(f"Unknown method: {method}")

    reconciled = np.clip(reconciled, 0.0, None)

    # rebuild output ordered by nodes list, per level
    out = []
    for level in ["sku", "category", "region", "total"]:
        keys = [k for (lvl, k) in nodes if lvl == level]
        idxs = [i for i, (lvl, _) in enumerate(nodes) if lvl == level]
        for k, i in zip(keys, idxs):
            out.append({"level": level, "key": k, "forecast": float(reconciled[i])})
    return pl.DataFrame(out)
