"""Dependency providers: lazy-loaded model artifacts."""
from __future__ import annotations

import logging
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any

import polars as pl

from src.config import get_app_config, get_settings

logger = logging.getLogger(__name__)


class ModelArtifacts:
    """Holds loaded artifacts: model, feature columns, last training date, metadata."""

    def __init__(self, artifacts_dir: Path):
        self.artifacts_dir = artifacts_dir
        self.model: Any | None = None
        self.feature_cols: list[str] = []
        self.metadata: dict[str, Any] = {}
        self.last_date: str | None = None
        self.demand_history: pl.DataFrame | None = None
        self.policies: pl.DataFrame | None = None

    def load(self) -> "ModelArtifacts":
        model_path = self.artifacts_dir / "model.pkl"
        meta_path = self.artifacts_dir / "metadata.pkl"
        history_path = self.artifacts_dir / "history.parquet"
        policies_path = self.artifacts_dir / "policies.parquet"

        if model_path.exists():
            with model_path.open("rb") as f:
                payload = pickle.load(f)
            self.model = payload["model"]
            self.feature_cols = payload["feature_cols"]
            logger.info("Loaded model from %s", model_path)
        else:
            logger.warning("Model file not found at %s", model_path)

        if meta_path.exists():
            with meta_path.open("rb") as f:
                self.metadata = pickle.load(f)
            self.last_date = self.metadata.get("last_date")

        if history_path.exists():
            self.demand_history = pl.read_parquet(history_path)
            logger.info("Loaded demand history: %s rows", self.demand_history.height)

        if policies_path.exists():
            self.policies = pl.read_parquet(policies_path)
            logger.info("Loaded inventory policies: %s rows", self.policies.height)

        return self

    @property
    def is_ready(self) -> bool:
        return self.model is not None and self.demand_history is not None


@lru_cache(maxsize=1)
def get_artifacts() -> ModelArtifacts:
    settings = get_settings()
    cfg = get_app_config()
    _ = cfg  # ensure config loaded (can be used for defaults)
    artifacts = ModelArtifacts(Path(settings.artifacts_dir)).load()
    return artifacts
