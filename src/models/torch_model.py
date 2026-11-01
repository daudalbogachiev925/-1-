"""Global PyTorch MLP forecaster with categorical embeddings."""
from __future__ import annotations

import logging

import numpy as np
import polars as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import TorchConfig
from src.models.base import BaseForecaster

logger = logging.getLogger(__name__)


class _MLPWithEmbeddings(nn.Module):
    def __init__(self, n_numeric: int, cat_cardinalities: list[int], embedding_dim: int, hidden_dim: int):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(card, embedding_dim) for card in cat_cardinalities]
        )
        input_dim = n_numeric + embedding_dim * len(cat_cardinalities)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat([x_num] + embs, dim=1) if embs else x_num
        return self.net(x).squeeze(-1)


class TorchForecaster(BaseForecaster):
    name = "torch"

    def __init__(self, config: TorchConfig, categorical_cols: list[str] | None = None):
        self.config = config
        self.categorical_cols = categorical_cols or ["sku_code", "region_code", "category_code"]
        self.model: _MLPWithEmbeddings | None = None
        self.numeric_cols: list[str] = []
        self.cat_cardinalities: list[int] = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _split_cols(self, feature_cols: list[str]) -> tuple[list[str], list[str]]:
        cats = [c for c in self.categorical_cols if c in feature_cols]
        nums = [c for c in feature_cols if c not in cats]
        return nums, cats

    def fit(self, train: pl.DataFrame, feature_cols: list[str], target: str = "demand") -> "TorchForecaster":
        nums, cats = self._split_cols(feature_cols)
        self.numeric_cols = nums
        x_num = train.select(nums).to_numpy().astype(np.float32) if nums else np.zeros((train.height, 0), np.float32)
        x_cat = train.select(cats).to_numpy().astype(np.int64) if cats else np.zeros((train.height, 0), np.int64)
        y = train.select(target).to_numpy().astype(np.float32).ravel()
        self.cat_cardinalities = [int(x_cat[:, i].max()) + 1 for i in range(x_cat.shape[1])] if cats else []

        self.model = _MLPWithEmbeddings(
            n_numeric=len(nums),
            cat_cardinalities=self.cat_cardinalities,
            embedding_dim=self.config.embedding_dim,
            hidden_dim=self.config.hidden_dim,
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        loss_fn = nn.PoissonNLLLoss(log_input=False, full=False)

        ds = TensorDataset(torch.from_numpy(x_num), torch.from_numpy(x_cat), torch.from_numpy(y))
        loader = DataLoader(ds, batch_size=self.config.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.config.epochs):
            total = 0.0
            for xb_num, xb_cat, yb in loader:
                xb_num = xb_num.to(self.device)
                xb_cat = xb_cat.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad()
                pred = self.model(xb_num, xb_cat).clamp(min=1e-6)
                loss = loss_fn(pred, yb)
                loss.backward()
                optimizer.step()
                total += float(loss.item()) * len(yb)
            logger.info("Torch epoch %s/%s loss=%.4f", epoch + 1, self.config.epochs, total / len(ds))
        return self

    def predict(self, df: pl.DataFrame, feature_cols: list[str]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fitted")
        nums, cats = self._split_cols(feature_cols)
        x_num = df.select(nums).to_numpy().astype(np.float32) if nums else np.zeros((df.height, 0), np.float32)
        x_cat = df.select(cats).to_numpy().astype(np.int64) if cats else np.zeros((df.height, 0), np.int64)
        self.model.eval()
        with torch.no_grad():
            preds = self.model(
                torch.from_numpy(x_num).to(self.device),
                torch.from_numpy(x_cat).to(self.device),
            ).cpu().numpy()
        return np.clip(preds, 0.0, None)
