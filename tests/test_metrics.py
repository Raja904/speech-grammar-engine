"""Tests for regression metrics (RMSE, Pearson correlation, MAE)."""

import numpy as np
import pytest

from src.metrics import (
    compute_regression_metrics,
    mean_absolute_error,
    pearson_correlation,
    root_mean_squared_error,
)


def test_rmse_basic():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0])
    assert root_mean_squared_error(y_true, y_pred) == 0.0

    y_pred_off = np.array([2.0, 3.0, 4.0, 5.0])
    assert root_mean_squared_error(y_true, y_pred_off) == 1.0


def test_pearson_correlation_perfect():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([2.0, 4.0, 6.0, 8.0])
    assert pytest.approx(pearson_correlation(y_true, y_pred), 1e-6) == 1.0

    y_pred_inv = np.array([4.0, 3.0, 2.0, 1.0])
    assert pytest.approx(pearson_correlation(y_true, y_pred_inv), 1e-6) == -1.0


def test_pearson_correlation_constant_predictions():
    # Protection against division-by-zero when model predicts a constant value (e.g. mean baseline)
    y_true = np.array([1.0, 2.5, 3.0, 4.5])
    y_pred = np.array([2.75, 2.75, 2.75, 2.75])
    r = pearson_correlation(y_true, y_pred)
    assert r == 0.0
    assert not np.isnan(r)


def test_mae_basic():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 1.0, 4.0])
    assert pytest.approx(mean_absolute_error(y_true, y_pred), 1e-6) == 1.0


def test_compute_regression_metrics_dict():
    y_true = [1.0, 2.0, 3.0]
    y_pred = [1.0, 2.0, 3.0]
    res = compute_regression_metrics(y_true, y_pred)
    assert "rmse" in res
    assert "pearson_r" in res
    assert "mae" in res
    assert res["rmse"] == 0.0
    assert pytest.approx(res["pearson_r"], 1e-6) == 1.0
    assert res["n_samples"] == 3


def test_mismatched_shapes_raise_error():
    with pytest.raises(ValueError, match="Shape mismatch"):
        root_mean_squared_error([1.0, 2.0], [1.0])

    with pytest.raises(ValueError, match="Shape mismatch"):
        pearson_correlation([1.0, 2.0], [1.0])


def test_empty_arrays_raise_error():
    with pytest.raises(ValueError, match="empty arrays"):
        root_mean_squared_error([], [])
