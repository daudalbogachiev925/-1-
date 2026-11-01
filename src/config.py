"""Central configuration loader using pydantic-settings + YAML."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DataConfig(BaseModel):
    n_sku: int = 40
    start_date: str = "2022-01-01"
    end_date: str = "2024-12-31"
    seed: int = 42
    raw_path: str = "./data/raw"
    processed_path: str = "./data/processed"


class HierarchyConfig(BaseModel):
    regions: list[str] = Field(default_factory=lambda: ["North", "South", "East", "West"])
    categories: list[str] = Field(default_factory=lambda: ["Electronics", "Apparel", "Grocery"])


class FeaturesConfig(BaseModel):
    lags: list[int] = Field(default_factory=lambda: [1, 7, 14, 28])
    rolling_windows: list[int] = Field(default_factory=lambda: [7, 14, 28, 56])
    fourier_orders: list[int] = Field(default_factory=lambda: [1, 2, 3])
    holiday_country: str = "US"


class CVConfig(BaseModel):
    n_folds: int = 4
    horizon: int = 28
    step: int = 28


class LightGBMConfig(BaseModel):
    n_estimators: int = 800
    learning_rate: float = 0.03
    num_leaves: int = 63
    min_child_samples: int = 20
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    reg_alpha: float = 0.1
    reg_lambda: float = 0.1


class ProphetConfig(BaseModel):
    changepoint_prior_scale: float = 0.05
    seasonality_prior_scale: float = 10.0


class TorchConfig(BaseModel):
    hidden_dim: int = 128
    embedding_dim: int = 16
    epochs: int = 30
    batch_size: int = 512
    lr: float = 0.001


class ModelsConfig(BaseModel):
    lightgbm: LightGBMConfig = Field(default_factory=LightGBMConfig)
    prophet: ProphetConfig = Field(default_factory=ProphetConfig)
    torch: TorchConfig = Field(default_factory=TorchConfig)


class ReconciliationConfig(BaseModel):
    method: str = "mint_sample"


class InventoryConfig(BaseModel):
    service_level: float = 0.95
    lead_time_days: int = 7
    holding_cost_per_unit: float = 0.5
    stockout_cost_per_unit: float = 5.0
    review_period_days: int = 7


class AppConfig(BaseModel):
    data: DataConfig = Field(default_factory=DataConfig)
    hierarchy: HierarchyConfig = Field(default_factory=HierarchyConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    cv: CVConfig = Field(default_factory=CVConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    hierarchy_reconciliation: ReconciliationConfig = Field(default_factory=ReconciliationConfig)
    inventory: InventoryConfig = Field(default_factory=InventoryConfig)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"
    data_dir: str = "./data"
    artifacts_dir: str = "./artifacts"
    config_path: str = "./configs/config.yaml"

    def load_app_config(self) -> AppConfig:
        path = Path(self.config_path)
        if not path.exists():
            return AppConfig()
        with path.open("r", encoding="utf-8") as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}
        return AppConfig(**raw)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


@lru_cache(maxsize=1)
def get_app_config() -> AppConfig:
    return get_settings().load_app_config()


def ensure_dirs() -> None:
    settings = get_settings()
    for p in (settings.data_dir, settings.artifacts_dir):
        Path(p).mkdir(parents=True, exist_ok=True)
    cfg = get_app_config()
    Path(cfg.data.raw_path).mkdir(parents=True, exist_ok=True)
    Path(cfg.data.processed_path).mkdir(parents=True, exist_ok=True)
