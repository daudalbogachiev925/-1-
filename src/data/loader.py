"""Persist and load demand datasets via Parquet + DuckDB."""
from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import polars as pl

from src.config import AppConfig, get_app_config

logger = logging.getLogger(__name__)


def save_parquet(df: pl.DataFrame, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(p)
    logger.info("Saved %s rows to %s", df.height, p)
    return p


def load_parquet(path: str | Path) -> pl.DataFrame:
    return pl.read_parquet(path)


def build_duckdb_view(parquet_glob: str, db_path: str = ":memory:") -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(db_path)
    con.execute(f"CREATE OR REPLACE VIEW demand AS SELECT * FROM read_parquet('{parquet_glob}')")
    return con


def persist_demand(df: pl.DataFrame, cfg: AppConfig | None = None) -> Path:
    cfg = cfg or get_app_config()
    path = Path(cfg.data.raw_path) / "demand.parquet"
    return save_parquet(df, path)


def load_demand(cfg: AppConfig | None = None) -> pl.DataFrame:
    cfg = cfg or get_app_config()
    path = Path(cfg.data.raw_path) / "demand.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Demand not found at {path}. Run scripts/generate_data.py first.")
    return load_parquet(path)
