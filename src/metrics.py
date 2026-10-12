"""Evaluation metrics for audio regression and grammar proficiency scoring.

Computes RMSE, Pearson Correlation, and MAE with robust numerical safeguards
against zero-variance predictions and edge cases.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Union

import numpy as np

logger = logging.getLogger(__name__)

ArrayLike = Union[np.ndarray, list]


def root_mean_squared_error(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Calculate Root Mean Squared Error (RMSE).

    Args:
        y_true: Ground truth continuous target values.
        y_pred: Predicted values.

    Returns:
        Float RMSE value.
    """
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)

    if yt.shape != yp.shape:
        raise ValueError(f"Shape mismatch: y_true {yt.shape} vs y_pred {yp.shape}")
    if len(yt) == 0:
        raise ValueError("Cannot calculate RMSE on empty arrays.")

    mse = np.mean((yt - yp) ** 2)
    return float(np.sqrt(mse))


def pearson_correlation(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Calculate Pearson correlation coefficient with zero-variance protection.

    If either vector has zero variance (e.g. constant prediction baseline),
    the correlation is theoretically undefined. Returns 0.0 with a warning
    instead of producing NaN or raising a division-by-zero error.

    Args:
        y_true: Ground truth continuous target values.
        y_pred: Predicted values.

    Returns:
        Float Pearson correlation coefficient in range [-1.0, 1.0].
    """
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)

    if yt.shape != yp.shape:
        raise ValueError(f"Shape mismatch: y_true {yt.shape} vs y_pred {yp.shape}")
    if len(yt) < 2:
        logger.warning("Pearson correlation requires at least 2 samples; returning 0.0.")
        return 0.0

    yt_diff = yt - np.mean(yt)
    yp_diff = yp - np.mean(yp)

    yt_std = np.sqrt(np.sum(yt_diff**2))
    yp_std = np.sqrt(np.sum(yp_diff**2))

    # Guard against zero variance
    if yt_std < 1e-12 or yp_std < 1e-12:
        logger.warning(
            "Zero variance detected in y_true or y_pred. Pearson correlation undefined, returning 0.0."
        )
        return 0.0

    r = float(np.sum(yt_diff * yp_diff) / (yt_std * yp_std))
    # Clip for floating point precision edge cases
    return float(np.clip(r, -1.0, 1.0))


def mean_absolute_error(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Calculate Mean Absolute Error (MAE).

    Args:
        y_true: Ground truth target values.
        y_pred: Predicted values.

    Returns:
        Float MAE value.
    """
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)

    if yt.shape != yp.shape:
        raise ValueError(f"Shape mismatch: y_true {yt.shape} vs y_pred {yp.shape}")
    if len(yt) == 0:
        raise ValueError("Cannot calculate MAE on empty arrays.")

    return float(np.mean(np.abs(yt - yp)))


def compute_regression_metrics(y_true: ArrayLike, y_pred: ArrayLike) -> Dict[str, float]:
    """Compute aggregate regression metrics: RMSE, Pearson r, and MAE.

    Args:
        y_true: Ground truth targets.
        y_pred: Model predictions.

    Returns:
        Dictionary containing 'rmse', 'pearson_r', 'mae', and 'n_samples'.
    """
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)

    return {
        "rmse": root_mean_squared_error(yt, yp),
        "pearson_r": pearson_correlation(yt, yp),
        "mae": mean_absolute_error(yt, yp),
        "n_samples": int(len(yt)),
    }
