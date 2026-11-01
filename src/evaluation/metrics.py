"""Forecast metrics: WAPE, sMAPE, MASE, Bias."""
from __future__ import annotations

import numpy as np


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true).sum()
    if denom == 0:
        return 0.0
    return float(np.abs(y_true - y_pred).sum() / denom)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom = np.where(denom == 0, 1e-9, denom)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true).sum()
    if denom == 0:
        return 0.0
    return float((y_pred - y_true).sum() / denom)


def mase(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray, seasonality: int = 7) -> float:
    if len(y_train) <= seasonality:
        return float("nan")
    naive = np.abs(y_train[seasonality:] - y_train[:-seasonality]).mean()
    if naive == 0:
        return float("nan")
    return float(np.abs(y_true - y_pred).mean() / naive)


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray | None = None,
    seasonality: int = 7,
) -> dict[str, float]:
    out = {
        "wape": wape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "bias": bias(y_true, y_pred),
    }
    if y_train is not None:
        out["mase"] = mase(y_true, y_pred, y_train, seasonality=seasonality)
    return out
