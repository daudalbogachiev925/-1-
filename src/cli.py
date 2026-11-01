"""Command-line entry: dfi generate|train|forecast|policies."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import polars as pl

from src.config import ensure_dirs, get_app_config, get_settings
from src.data.generator import generate_demand_frame, generate_exogenous_events
from src.data.loader import load_demand, persist_demand
from src.inventory.safety_stock import compute_policies, simulate_policy

logging.basicConfig(level=get_settings().log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("dfi.cli")


def cmd_generate(_: argparse.Namespace) -> int:
    cfg = get_app_config()
    ensure_dirs()
    df = generate_demand_frame(cfg)
    persist_demand(df, cfg)
    events = generate_exogenous_events(cfg)
    events_path = Path(cfg.data.raw_path) / "events.parquet"
    events.write_parquet(events_path)
    logger.info("Generated demand + events: %s rows, %s events", df.height, events.height)
    return 0


def cmd_train(_: argparse.Namespace) -> int:
    from scripts.train import main as train_main  # local import to avoid heavy deps at import
    return train_main()


def cmd_forecast(args: argparse.Namespace) -> int:
    df = load_demand()
    print(df.filter(pl.col("sku") == args.sku).tail(args.n))
    return 0


def cmd_policies(args: argparse.Namespace) -> int:
    cfg = get_app_config()
    df = load_demand()
    policies = compute_policies(
        df.rename({"demand": "forecast"}).select(["sku", "date", "forecast"]),
        cfg.inventory,
    )
    print(policies.head(args.n))
    metrics = simulate_policy(df, policies, cfg.inventory)
    print("\nSimulated metrics:")
    print(metrics.describe())
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dfi", description="Demand Forecast & Inventory CLI")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("generate", help="Generate synthetic demand data")
    sp.set_defaults(func=cmd_generate)

    sp = sub.add_parser("train", help="Train models and save artifacts")
    sp.set_defaults(func=cmd_train)

    sp = sub.add_parser("forecast", help="Show recent demand for a SKU")
    sp.add_argument("--sku", required=True)
    sp.add_argument("--n", type=int, default=10)
    sp.set_defaults(func=cmd_forecast)

    sp = sub.add_parser("policies", help="Compute and simulate inventory policies")
    sp.add_argument("--n", type=int, default=10)
    sp.set_defaults(func=cmd_policies)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
