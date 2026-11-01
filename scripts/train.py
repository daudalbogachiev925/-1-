"""Full training pipeline: features, model fit, backtest, reconciliation, artifacts."""
from __future__ import annotations

import logging
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import polars as pl

from src.config import ensure_dirs, get_app_config, get_settings
from src.data.loader import load_demand
from src.evaluation.backtest import run_backtest, summarize
from src.evaluation.metrics import compute_all_metrics
from src.features.build import build_features
from src.inventory.safety_stock import baseline_policy, compute_policies, simulate_policy
from src.inventory.segmentation import abc_xyz_matrix
from src.models.hierarchy import reconcile
from src.models.lightgbm_model import LightGBMForecaster

logging.basicConfig(
    level=get_settings().log_level,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("dfi.train")


def _build_abc_xyz_frame(df: pl.DataFrame) -> pl.DataFrame:
    return abc_xyz_matrix(df)


def _build_base_forecast_frame(
    bundle_frame: pl.DataFrame,
    preds: np.ndarray,
    target_dates: pl.DataFrame,
) -> pl.DataFrame:
    """Aggregate bottom-level predictions to category, region, total."""
    df = target_dates.with_columns(pl.Series("forecast", preds))
    bottom = df.select(["sku", "region", "category", "date", "forecast"])
    sku_level = bottom.group_by(["date", "sku"]).agg(pl.col("forecast").sum())
    sku_level = sku_level.with_columns(pl.lit("sku").alias("level"), pl.col("sku").alias("key"))
    cat = bottom.group_by(["date", "category"]).agg(pl.col("forecast").sum())
    cat = cat.with_columns(pl.lit("category").alias("level"), pl.col("category").alias("key"))
    reg = bottom.group_by(["date", "region"]).agg(pl.col("forecast").sum())
    reg = reg.with_columns(pl.lit("region").alias("level"), pl.col("region").alias("key"))
    tot = bottom.group_by(["date"]).agg(pl.col("forecast").sum())
    tot = tot.with_columns(pl.lit("total").alias("level"), pl.lit("ALL").alias("key"))
    out = pl.concat(
        [
            sku_level.select(["date", "level", "key", "forecast"]),
            cat.select(["date", "level", "key", "forecast"]),
            reg.select(["date", "level", "key", "forecast"]),
            tot.select(["date", "level", "key", "forecast"]),
        ],
        how="vertical",
    )
    # reconcile: average base forecasts over horizon per node
    agg = out.group_by(["level", "key"]).agg(pl.col("forecast").mean().alias("forecast"))
    return agg


def _save_artifacts(
    model,
    feature_cols: list[str],
    history: pl.DataFrame,
    policies: pl.DataFrame,
    metrics: dict,
) -> None:
    settings = get_settings()
    out_dir = Path(settings.artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "model.pkl").open("wb") as f:
        pickle.dump({"model": model, "feature_cols": feature_cols}, f)
    with (out_dir / "metadata.pkl").open("wb") as f:
        pickle.dump(
            {
                "last_date": str(history["date"].max()),
                "metrics": metrics,
                "model_type": type(model).__name__,
            },
            f,
        )
    history.write_parquet(out_dir / "history.parquet")
    policies.write_parquet(out_dir / "policies.parquet")
    logger.info("Saved artifacts to %s", out_dir)


def main() -> int:
    ensure_dirs()
    cfg = get_app_config()

    logger.info("Loading demand...")
    df = load_demand(cfg)
    logger.info("Loaded %s rows", df.height)

    logger.info("Building features...")
    bundle = build_features(df, cfg)
    logger.info("Feature frame: %s rows x %s features", bundle.frame.height, len(bundle.feature_cols))

    categorical_cols = ["region_code", "category_code", "sku_code"]

    logger.info("Running rolling-origin backtest...")
    backtest_model = LightGBMForecaster(cfg.models.lightgbm)
    fold_results = run_backtest(
        bundle.frame,
        backtest_model,
        bundle.feature_cols,
        cfg.cv,
    )
    if fold_results:
        summary = summarize(fold_results)
        logger.info("Backtest summary:\n%s", summary)
        mean_metrics = {
            "wape": float(np.nanmean([f.metrics["wape"] for f in fold_results])),
            "smape": float(np.nanmean([f.metrics["smape"] for f in fold_results])),
            "bias": float(np.nanmean([f.metrics["bias"] for f in fold_results])),
        }
    else:
        mean_metrics = {"wape": float("nan"), "smape": float("nan"), "bias": float("nan")}
    logger.info("Mean CV metrics: %s", mean_metrics)

    logger.info("Training final model on all data...")
    final_model = LightGBMForecaster(cfg.models.lightgbm)
    final_model.fit(bundle.frame, bundle.feature_cols, categorical_cols=categorical_cols)

    logger.info("Computing feature importance...")
    fi = final_model.feature_importance(bundle.feature_cols)
    logger.info("Top-10 features:\n%s", fi.head(10))

    logger.info("Generating in-sample forecasts for reconciliation...")
    in_sample_preds = final_model.predict(bundle.frame, bundle.feature_cols)
    last_date = df["date"].max()
    target_df = bundle.frame.select(["date", "sku", "region", "category"]).with_columns(
        pl.Series("forecast", in_sample_preds)
    )
    base_forecast = _build_base_forecast_frame(bundle.frame, in_sample_preds, target_df)

    series_df = df.select(["sku", "region", "category"]).unique()
    logger.info("Reconciling forecasts (Mint-Sample)...")
    reconciled = reconcile(
        base_forecast,
        series_df,
        method=cfg.hierarchy_reconciliation.method,
    )
    logger.info("Reconciliation done: %s rows", reconciled.height)

    logger.info("Computing inventory policies...")
    forecast_for_policies = target_df.group_by("sku").agg(
        pl.col("forecast").mean().alias("forecast")
    )
    # expand per-SKU to a 28-day horizon (mean is constant)
    horizon = 28
    policies_df = compute_policies(
        target_df.select(["sku", "date", "forecast"]),
        cfg.inventory,
    )
    logger.info("Policies computed: %s SKUs", policies_df.height)

    logger.info("Simulating policies vs baseline...")
    sim = simulate_policy(df, policies_df, cfg.inventory)
    baseline = baseline_policy(df, cfg.inventory)
    sim_base = simulate_policy(df, baseline, cfg.inventory)
    logger.info(
        "Optimized: fill_rate=%.3f stockout_rate=%.3f total_cost=%.2f",
        float(sim["fill_rate"].mean()),
        float(sim["stockout_rate"].mean()),
        float(sim["total_cost"].sum()),
    )
    logger.info(
        "Baseline:  fill_rate=%.3f stockout_rate=%.3f total_cost=%.2f",
        float(sim_base["fill_rate"].mean()),
        float(sim_base["stockout_rate"].mean()),
        float(sim_base["total_cost"].sum()),
    )

    logger.info("ABC/XYZ segmentation...")
    seg = _build_abc_xyz_frame(df)
    logger.info("Segments:\n%s", seg.head())

    logger.info("Saving artifacts...")
    _save_artifacts(
        model=final_model,
        feature_cols=bundle.feature_cols,
        history=df,
        policies=policies_df,
        metrics=mean_metrics,
    )
    logger.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
