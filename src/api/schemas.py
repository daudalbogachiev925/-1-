"""Pydantic v2 schemas for API."""
from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    version: str


class ForecastRequest(BaseModel):
    sku: str = Field(..., description="SKU identifier, e.g. SKU-001")
    horizon: int = Field(28, ge=1, le=90, description="Forecast horizon in days")
    start_date: date | None = Field(
        None, description="Optional start date; defaults to day after last training date"
    )


class ForecastPoint(BaseModel):
    date: date
    forecast: float = Field(..., ge=0.0)


class ForecastResponse(BaseModel):
    sku: str
    horizon: int
    model: str
    points: list[ForecastPoint]
    total: float


class InventoryPolicyResponse(BaseModel):
    sku: str
    mu_daily: float
    sigma_daily: float
    lead_time: int
    review_period: int
    safety_stock: float
    reorder_point: float
    order_up_to: float
    service_level: float


class SegmentResponse(BaseModel):
    sku: str
    abc_class: str
    xyz_class: str


class MetricsResponse(BaseModel):
    wape: float
    smape: float
    bias: float
    mase: float | None = None


class ErrorResponse(BaseModel):
    detail: str
