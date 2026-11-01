"""Global LightGBM forecaster with categorical support."""
from __future__ import annotations

import logging

import lightgbm as lgb
import numpy as np
import polars as pl

from src.config import LightGBMConfig
from src.models.base import BaseForecaster

logger = logging.getLogger(__name__)


class LightGBMForecaster(BaseForecaster):
    name = "lightgbm"

    def __init__(self, config: LightGBMConfig):
        self.config = config
        self.model: lgb.LGBMRegressor | None = None
        self._feature_cols: list[str] = []

    def fit(
        self,
        train: pl.DataFrame,
        feature_cols: list[str],
        target: str = "demand",
        categorical_cols: list[str] | None = None,
    ) -> "LightGBMForecaster":
        X = train.select(feature_cols).to_pandas()
        y = train.select(target).to_numpy().ravel()
        cat_cols = categorical_cols or []
        for c in cat_cols:
            if c in X.columns:
                X[c] = X[c].astype("category")
        self._feature_cols = list(feature_cols)
        self.model = lgb.LGBMRegressor(
            objective="tweedie",
            tweedie_variance_power=1.3,
            n_estimators=self.config.n_estimators,
            learning_rate=self.config.learning_rate,
            num_leaves=self.config.num_leaves,
            min_child_samples=self.config.min_child_samples,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            reg_alpha=self.config.reg_alpha,
            reg_lambda=self.config.reg_lambda,
            n_jobs=-1,
            random_state=42,
            verbose=-1,
        )
        self.model.fit(X, y, categorical_feature=cat_cols or "auto")
        logger.info("LightGBM fitted on %s rows, %s features", X.shape[0], X.shape[1])
        return self

    def predict(self, df: pl.DataFrame, feature_cols: list[str]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fitted")
        X = df.select(feature_cols).to_pandas()
        for c in X.columns:
            if str(X[c].dtype) == "category":
                X[c] = X[c].astype("category")
        preds = self.model.predict(X)
        return np.clip(preds, 0.0, None)

    def feature_importance(self, feature_cols: list[str] | None = None) -> pl.DataFrame:
        if self.model is None:
            raise RuntimeError("Model not fitted")
        cols = feature_cols or self._feature_cols
        importances = self.model.feature_importances_
        return (
            pl.DataFrame({"feature": cols, "importance": importances})
            .sort("importance", descending=True)
        )
