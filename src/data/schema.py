"""Data schema definitions and constants."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import polars as pl

DEMAND_SCHEMA = {
    "date": pl.Date,
    "sku": pl.Utf8,
    "region": pl.Utf8,
    "category": pl.Utf8,
    "demand": pl.Float64,
    "price": pl.Float64,
    "promo": pl.Int8,
    "holiday": pl.Int8,
    "temperature": pl.Float64,
    "is_weekend": pl.Int8,
    "month": pl.Int8,
    "day_of_week": pl.Int8,
}

HIERARCHY_LEVELS: tuple[Literal["total", "region", "category", "sku"], ...] = (
    "total",
    "region",
    "category",
    "sku",
)


@dataclass(frozen=True)
class HierarchySpec:
    """Mapping of aggregation level to key columns."""
    level: str
    keys: tuple[str, ...]

    @staticmethod
    def all_levels() -> list["HierarchySpec"]:
        return [
            HierarchySpec("total", ()),
            HierarchySpec("region", ("region",)),
            HierarchySpec("category", ("category",)),
            HierarchySpec("sku", ("sku",)),
        ]
