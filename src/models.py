"""Model definitions, baselines, and cross-validation runners for Spoken Grammar Scoring.

Implements:
- MeanBaselineModel (empirical floor)
- RidgeGrammarModel (regularized linear model with internal StandardScaler)
- SVREngineModel (Support Vector Regression with RBF kernel)
- run_cv_experiment (leak-free 5-fold cross-validation loop and OOF metrics)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from src.metrics import mean_absolute_error, pearson_correlation, root_mean_squared_error

logger = logging.getLogger(__name__)


class BaseGrammarModel(ABC):
    """Abstract interface for all grammar scoring models."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> BaseGrammarModel:
        """Fit model on training features and targets."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate continuous grammar score predictions in range [0.0, 5.0]."""
        pass


class MeanBaselineModel(BaseGrammarModel):
    """Simple baseline model that predicts the training target mean (or median).

    Establishes the minimal performance floor against which all learned models are benchmarked.
    """

    def __init__(self, strategy: str = "mean"):
        if strategy not in ("mean", "median"):
            raise ValueError(f"Unknown baseline strategy: {strategy}. Must be 'mean' or 'median'.")
        self.strategy = strategy
        self.baseline_value: Optional[float] = None

    def fit(self, X: Optional[np.ndarray], y: np.ndarray) -> MeanBaselineModel:
        y_clean = np.asarray(y, dtype=np.float64)
        if len(y_clean) == 0:
            raise ValueError("Cannot fit MeanBaselineModel on empty targets.")

        if self.strategy == "mean":
            self.baseline_value = float(np.mean(y_clean))
        else:
            self.baseline_value = float(np.median(y_clean))

        logger.info(f"MeanBaselineModel fitted with strategy='{self.strategy}', value={self.baseline_value:.4f}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.baseline_value is None:
            raise RuntimeError("Model is not fitted. Call fit() before predict().")

        n_samples = len(X)
        preds = np.full(shape=(n_samples,), fill_value=self.baseline_value, dtype=np.float64)
        return np.clip(preds, 0.0, 5.0)


class RidgeGrammarModel(BaseGrammarModel):
    """Ridge regression with internal StandardScaler and cross-validated alpha search."""

    def __init__(self, alphas: Optional[List[float]] = None):
        self.alphas = alphas or [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=self.alphas)),
        ])

    def fit(self, X: np.ndarray, y: np.ndarray) -> RidgeGrammarModel:
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float64)
        self.pipeline.fit(X_arr, y_arr)
        best_alpha = self.pipeline.named_steps["ridge"].alpha_
        logger.info(f"RidgeGrammarModel fitted. Best alpha: {best_alpha}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=np.float32)
        raw_preds = self.pipeline.predict(X_arr)
        return np.clip(raw_preds, 0.0, 5.0)


class SVREngineModel(BaseGrammarModel):
    """Support Vector Regression (RBF kernel) with internal StandardScaler."""

    def __init__(self, C: float = 1.0, epsilon: float = 0.1, kernel: str = "rbf"):
        self.C = C
        self.epsilon = epsilon
        self.kernel = kernel
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("svr", SVR(C=self.C, epsilon=self.epsilon, kernel=self.kernel)),
        ])

    def fit(self, X: np.ndarray, y: np.ndarray) -> SVREngineModel:
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float64)
        self.pipeline.fit(X_arr, y_arr)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=np.float32)
        raw_preds = self.pipeline.predict(X_arr)
        return np.clip(raw_preds, 0.0, 5.0)


def run_cv_experiment(
    fold_df: pd.DataFrame,
    features: np.ndarray,
    target_col: str,
    model_factory: Callable[[], BaseGrammarModel],
    model_name: str = "Model",
    fold_col: str = "fold",
) -> Dict[str, Any]:
    """Execute a leak-free 5-fold cross-validation experiment.

    Args:
        fold_df: DataFrame containing samples with 'fold' column.
        features: 2D array of audio features aligned with fold_df rows.
        target_col: Target column name.
        model_factory: Callable returning an unfitted BaseGrammarModel instance.
        model_name: Human-readable model identifier.
        fold_col: Column storing fold assignments.

    Returns:
        Dictionary containing OOF RMSE, Pearson r, MAE, fold scores, and fitted models.
    """
    unique_folds = sorted(fold_df[fold_col].unique())
    n_splits = len(unique_folds)
    n_samples = len(fold_df)

    oof_predictions = np.zeros(n_samples, dtype=np.float64)
    fold_metrics: List[Dict[str, float]] = []
    fitted_models: List[BaseGrammarModel] = []

    y_all = fold_df[target_col].to_numpy(dtype=np.float64)

    logger.info(f"--- Running 5-Fold Cross-Validation for: {model_name} ---")

    for fold in unique_folds:
        val_mask = (fold_df[fold_col] == fold).to_numpy()
        train_mask = ~val_mask

        X_train, y_train = features[train_mask], y_all[train_mask]
        X_val, y_val = features[val_mask], y_all[val_mask]

        model = model_factory()
        model.fit(X_train, y_train)
        preds = model.predict(X_val)

        oof_predictions[val_mask] = preds
        fitted_models.append(model)

        fold_rmse = root_mean_squared_error(y_val, preds)
        fold_r = pearson_correlation(y_val, preds)
        fold_mae = mean_absolute_error(y_val, preds)

        fold_metrics.append({
            "fold": int(fold),
            "rmse": float(fold_rmse),
            "pearson_r": float(fold_r),
            "mae": float(fold_mae),
        })

    # Overall OOF Metrics
    oof_rmse = root_mean_squared_error(y_all, oof_predictions)
    oof_r = pearson_correlation(y_all, oof_predictions)
    oof_mae = mean_absolute_error(y_all, oof_predictions)

    logger.info(
        f"[{model_name}] OOF RMSE: {oof_rmse:.4f} | Pearson r: {oof_r:.4f} | MAE: {oof_mae:.4f}"
    )

    return {
        "model_name": model_name,
        "oof_rmse": float(oof_rmse),
        "oof_pearson_r": float(oof_r),
        "oof_mae": float(oof_mae),
        "fold_metrics": fold_metrics,
        "oof_predictions": oof_predictions,
        "models": fitted_models,
    }
