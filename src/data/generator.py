"""Synthetic demand data generator with hierarchical structure, seasonality, promos, holidays."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import polars as pl

from src.config import AppConfig
from src.data.schema import DEMAND_SCHEMA

logger = logging.getLogger(__name__)


@dataclass
class _SkuProfile:
    sku_id: str
    region: str
    category: str
    base_level: float
    trend_slope: float
    weekly_amp: float
    annual_amp: float
    noise_std: float
    promo_lift: float
    holiday_lift: float
    price_base: float
    price_elasticity: float
    temperature_sensitivity: float


def _build_profiles(cfg: AppConfig) -> list[_SkuProfile]:
    rng = np.random.default_rng(cfg.data.seed)
    profiles: list[_SkuProfile] = []
    regions = cfg.hierarchy.regions
    categories = cfg.hierarchy.categories
    for i in range(cfg.data.n_sku):
        region = regions[i % len(regions)]
        category = categories[i % len(categories)]
        profiles.append(
            _SkuProfile(
                sku_id=f"SKU-{i + 1:03d}",
                region=region,
                category=category,
                base_level=float(rng.uniform(20, 200)),
                trend_slope=float(rng.uniform(-0.02, 0.05)),
                weekly_amp=float(rng.uniform(0.05, 0.25)),
                annual_amp=float(rng.uniform(0.1, 0.4)),
                noise_std=float(rng.uniform(1.5, 6.0)),
                promo_lift=float(rng.uniform(1.2, 2.2)),
                holiday_lift=float(rng.uniform(1.1, 1.8)),
                price_base=float(rng.uniform(5, 200)),
                price_elasticity=float(rng.uniform(-2.0, -0.5)),
                temperature_sensitivity=float(rng.uniform(-0.5, 1.5)),
            )
        )
    return profiles


def _us_holidays(year: int) -> set[date]:
    """Minimal fixed-date US holidays (no easter computation for simplicity)."""
    return {
        date(year, 1, 1),
        date(year, 7, 4),
        date(year, 11, 11),
        date(year, 12, 25),
        date(year, 12, 31),
    }


def _promo_flag(rng: np.random.Generator, n: int, promo_prob: float = 0.06) -> np.ndarray:
    return (rng.random(n) < promo_prob).astype(np.int8)


def generate_demand_frame(cfg: AppConfig) -> pl.DataFrame:
    """Generate long-format demand DataFrame (date x sku)."""
    profiles = _build_profiles(cfg)
    start = date.fromisoformat(cfg.data.start_date)
    end = date.fromisoformat(cfg.data.end_date)
    n_days = (end - start).days + 1
    dates = [start + timedelta(days=i) for i in range(n_days)]
    rng = np.random.default_rng(cfg.data.seed)

    holidays_by_year: dict[int, set[date]] = {}
    for y in {d.year for d in dates}:
        holidays_by_year[y] = _us_holidays(y)

    frames: list[pl.DataFrame] = []
    for p in profiles:
        t = np.arange(n_days, dtype=np.float64)
        dow = np.array([(start + timedelta(days=i)).weekday() for i in range(n_days)])
        month = np.array([(start + timedelta(days=i)).month for i in range(n_days)])
        day_of_year = np.array([(start + timedelta(days=i)).timetuple().tm_yday for i in range(n_days)])

        # temperature: annual sinusoid with noise
        temp = 15.0 + 12.0 * np.sin(2 * np.pi * (day_of_year - 100) / 365.0) + rng.normal(0, 3, n_days)

        # promo
        promo = _promo_flag(rng, n_days)

        # holidays
        holiday = np.array(
            [1 if (start + timedelta(days=i)) in holidays_by_year[(start + timedelta(days=i)).year] else 0
             for i in range(n_days)],
            dtype=np.int8,
        )

        # trend
        trend = p.base_level + p.trend_slope * t

        # seasonality
        weekly = 1.0 + p.weekly_amp * np.sin(2 * np.pi * dow / 7.0)
        annual = 1.0 + p.annual_amp * np.sin(2 * np.pi * day_of_year / 365.0)

        # price dynamics
        price = p.price_base * (1.0 + 0.1 * np.sin(2 * np.pi * day_of_year / 365.0))
        price *= 1.0 - 0.15 * promo  # discount during promo

        # elasticity
        price_rel = price / p.price_base
        elasticity_factor = np.power(np.clip(price_rel, 0.5, 2.0), p.price_elasticity)

        # temperature
        temp_factor = 1.0 + 0.005 * p.temperature_sensitivity * (temp - 15.0)

        # promo & holiday lift
        promo_factor = np.where(promo == 1, p.promo_lift, 1.0)
        holiday_factor = np.where(holiday == 1, p.holiday_lift, 1.0)

        # base demand
        mu = trend * weekly * annual * elasticity_factor * temp_factor * promo_factor * holiday_factor
        mu = np.clip(mu, 0.1, None)

        # negative binomial noise via poisson-gamma mixture
        dispersion = 0.15
        gamma_sample = rng.gamma(shape=1.0 / dispersion, scale=mu * dispersion, size=n_days)
        demand = rng.poisson(lam=gamma_sample).astype(np.float64)

        df = pl.DataFrame(
            {
                "date": dates,
                "sku": [p.sku_id] * n_days,
                "region": [p.region] * n_days,
                "category": [p.category] * n_days,
                "demand": demand,
                "price": price,
                "promo": promo,
                "holiday": holiday,
                "temperature": temp,
                "is_weekend": (dow >= 5).astype(np.int8),
                "month": month.astype(np.int8),
                "day_of_week": dow.astype(np.int8),
            }
        ).cast(DEMAND_SCHEMA)
        frames.append(df)

    out = pl.concat(frames, how="vertical").sort(["sku", "date"])
    logger.info("Generated demand frame: %s rows", out.height)
    return out


def generate_exogenous_events(cfg: AppConfig) -> pl.DataFrame:
    """Generate calendar of promo campaigns (aggregated events)."""
    rng = np.random.default_rng(cfg.data.seed + 7)
    start = date.fromisoformat(cfg.data.start_date)
    end = date.fromisoformat(cfg.data.end_date)
    events: list[dict] = []
    cur = start
    while cur <= end:
        if rng.random() < 0.02:
            dur = int(rng.integers(3, 8))
            for k in range(dur):
                d = cur + timedelta(days=k)
                if d > end:
                    break
                events.append({"date": d, "campaign": "flash_sale", "intensity": float(rng.uniform(0.5, 1.0))})
            cur = cur + timedelta(days=dur)
        else:
            cur = cur + timedelta(days=1)
    if not events:
        return pl.DataFrame({"date": [], "campaign": [], "intensity": []})
    return pl.DataFrame(events).with_columns(pl.col("date").cast(pl.Date))
