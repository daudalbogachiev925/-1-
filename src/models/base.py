"""Abstract base model interface."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import polars as pl


class BaseForecaster(ABC):
    """Common interface for all forecasters."""

    name: str = "base"

    @abstractmethod
    def fit(self, train: pl.DataFrame, feature_cols: list[str], target: str = "demand") -> "BaseForecaster":
        ...

    @abstractmethod
    def predict(self, df: pl.DataFrame, feature_cols: list[str]) -> np.ndarray:
        ...

    def fit_predict(
        self, train: pl.DataFrame, test: pl.DataFrame, feature_cols: list[str], target: str = "demand"
    ) -> np.ndarray:
        self.fit(train, feature_cols, target)
        return self.predict(test, feature_cols)
