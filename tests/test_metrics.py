"""Metrics correctness tests."""
from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.metrics import bias, mase, smape, wape


def test_wape_perfect() -> None:
    y = np.array([1.0, 2.0, 3.0])
    assert wape(y, y) == 0.0


def test_wape_half_error() -> None:
    y = np.array([10.0, 10.0])
    p = np.array([15.0, 10.0])
    # abs error 5, denom 20 => 0.25
    assert wape(y, p) == pytest.approx(0.25)


def test_smape_symmetric() -> None:
    assert smape(np.array([1.0, 2.0]), np.array([1.0, 2.0])) == 0.0


def test_bias_positive_when_overpredict() -> None:
    assert bias(np.array([1.0, 1.0]), np.array([2.0, 2.0])) > 0


def test_mase_returns_nan_for_constant() -> None:
    y_train = np.array([5.0] * 30)
    val = mase(np.array([5.0]), np.array([5.0]), y_train)
    assert np.isnan(val)
