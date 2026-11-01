"""Rolling-origin temporal cross-validation."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from src.config import CVConfig
from src.evaluation.metrics import compute_all_metrics
from src.models.base import BaseForecaster

logger = logging.getLogger(__name__)


@dataclass
class FoldResult:
    fold: int
    train_end: str
    test_start: str
    test_end: str
    metrics: dict[str, float] = field(default_factory=dict)
    y_true: np.ndarray | None = None
    y_pred: np.ndarray | None = None


def rolling_origin_folds(
    frame: pl.DataFrame, cfg: CVConfig
) -> list[tuple[pl.DataFrame, pl.DataFrame]]:
    """Return list of (train, test) frames by rolling origin."""
    df = frame.sort("date")
    dates = df.select("date").unique().sort("date")["date"].to_list()
    n = len(dates)
    if n < cfg.n_folds * cfg.horizon + 1:
        raise ValueError("Not enough data for requested CV config")
    folds: list[tuple[pl.DataFrame, pl.DataFrame]] = []
    # place last test fold at the end
    end = n
    for k in range(cfg.n_folds):
        test_end_idx = end - k * cfg.step
        test_start_idx = test_end_idx - cfg.horizon
        train_end_idx = test_start_idx
        train_dates = dates[:train_end_idx]
        test_dates = dates[test_start_idx:test_end_idx]
        train = df.filter(pl.col("date").is_in(train_dates))
        test = df.filter(pl.col("date").is_in(test_dates))
        folds.append((train, test))
    return list(reversed(folds))


def run_backtest(
    frame: pl.DataFrame,
    forecaster: BaseForecaster,
    feature_cols: list[str],
    cfg: CVConfig,
    by: list[str] | None = None,
) -> list[FoldResult]:
    """Run rolling-origin backtest; returns fold results."""
    folds = rolling_origin_folds(frame, cfg)
    results: list[FoldResult] = []
    for k, (train, test) in enumerate(folds):
        if train.height < 30 or test.height == 0:
            logger.warning("Fold %s skipped: empty train/test", k)
            continue
        logger.info(
            "Fold %s train[%s -> %s] test[%s -> %s]",
            k,
            train["date"].min(),
            train["date"].max(),
            test["date"].min(),
            test["date"].max(),
        )
        try:
            forecaster.fit(train, feature_cols)
            preds = forecaster.predict(test, feature_cols)
        except Exception as exc:  # pragma: no cover
            logger.exception("Fold %s failed: %s", k, exc)
            continue
        y_true = test.select("demand").to_numpy().ravel()
        y_train = train.select("demand").to_numpy().ravel()
        metrics = compute_all_metrics(y_true, preds, y_train=y_train)
        results.append(
            FoldResult(
                fold=k,
                train_end=str(train["date"].max()),
                test_start=str(test["date"].min()),
                test_end=str(test["date"].max()),
                metrics=metrics,
                y_true=y_true,
                y_pred=preds,
            )
        )
        logger.info("Fold %s metrics: %s", k, metrics)
    return results


def summarize(results: list[FoldResult]) -> pl.DataFrame:
    rows = []
    for r in results:
        rows.append({"fold": r.fold, **r.metrics})
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows)
