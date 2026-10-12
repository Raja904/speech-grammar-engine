"""Model definitions, baselines, tree regressors, MLPs, and ensembling for Spoken Grammar Scoring.

Implements:
- MeanBaselineModel (empirical floor)
- RidgeGrammarModel (regularized linear model with internal StandardScaler)
- SVREngineModel (Support Vector Regression with RBF kernel)
- LGBMGrammarModel (LightGBM / HistGradientBoosting decision trees)
- MLPGrammarModel (Multi-layer perceptron neural regressor)
- EnsembleBlender (Optimal constrained weighted blend of multiple models)
- run_cv_experiment (leak-free 5-fold cross-validation loop and OOF metrics)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor
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


class LGBMGrammarModel(BaseGrammarModel):
    """Gradient Boosted Decision Trees via LightGBM (with HistGradientBoosting fallback)."""

    def __init__(
        self,
        n_estimators: int = 150,
        learning_rate: float = 0.04,
        max_depth: int = 5,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.model: Any = None

        try:
            import lightgbm as lgb
            self.model = lgb.LGBMRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                num_leaves=2 ** self.max_depth - 1,
                colsample_bytree=0.6,
                subsample=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=self.random_state,
                verbose=-1,
            )
        except (ImportError, Exception):
            # Fallback to standard scikit-learn HistGradientBoostingRegressor
            self.model = HistGradientBoostingRegressor(
                max_iter=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                random_state=self.random_state,
                l2_regularization=1.0,
            )

    def fit(self, X: np.ndarray, y: np.ndarray) -> LGBMGrammarModel:
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float64)
        self.model.fit(X_arr, y_arr)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=np.float32)
        preds = self.model.predict(X_arr)
        return np.clip(preds, 0.0, 5.0)


class MLPGrammarModel(BaseGrammarModel):
    """Multi-Layer Perceptron Neural Regressor with internal standard scaler and early stopping."""

    def __init__(
        self,
        hidden_layer_sizes: Tuple[int, ...] = (256, 64),
        alpha: float = 0.01,
        learning_rate_init: float = 0.001,
        random_state: int = 42,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.random_state = random_state

        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPRegressor(
                    hidden_layer_sizes=self.hidden_layer_sizes,
                    activation="relu",
                    alpha=self.alpha,
                    learning_rate_init=self.learning_rate_init,
                    max_iter=300,
                    early_stopping=True,
                    n_iter_no_change=15,
                    random_state=self.random_state,
                ),
            ),
        ])

    def fit(self, X: np.ndarray, y: np.ndarray) -> MLPGrammarModel:
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float64)
        self.pipeline.fit(X_arr, y_arr)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=np.float32)
        preds = self.pipeline.predict(X_arr)
        return np.clip(preds, 0.0, 5.0)


class EnsembleBlender:
    """Optimal constrained weighted blending using scipy.optimize to minimize OOF RMSE.

    Finds non-negative weights that sum to 1.0, optimizing directly on out-of-fold predictions.
    """

    def __init__(self):
        self.weights: Dict[str, float] = {}
        self.model_names: List[str] = []

    def fit(self, oof_dict: Dict[str, np.ndarray], y_true: np.ndarray) -> EnsembleBlender:
        """Find optimal non-negative weights w_i sum(w_i)=1 minimizing RMSE."""
        self.model_names = list(oof_dict.keys())
        k = len(self.model_names)
        if k == 0:
            raise ValueError("oof_dict cannot be empty.")

        # Matrix of shape (N, k)
        OOF_mat = np.column_stack([oof_dict[m] for m in self.model_names])
        y_arr = np.asarray(y_true, dtype=np.float64)

        def loss(w: np.ndarray) -> float:
            blend = np.dot(OOF_mat, w)
            return float(np.sqrt(np.mean((y_arr - blend) ** 2)))

        # Uniform initial guess
        init_w = np.ones(k) / k
        bounds = [(0.0, 1.0) for _ in range(k)]
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

        res = minimize(loss, init_w, bounds=bounds, constraints=constraints, method="SLSQP")
        best_w = res.x if res.success else init_w

        # Normalize to strictly sum to 1.0
        best_w = np.clip(best_w, 0.0, 1.0)
        best_w /= np.sum(best_w)

        self.weights = {name: float(w) for name, w in zip(self.model_names, best_w)}
        logger.info(f"Ensemble blender fitted with weights: {self.weights}")
        return self

    def predict(self, test_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """Generate blended test predictions using fitted optimal weights."""
        if not self.weights:
            raise RuntimeError("EnsembleBlender is not fitted. Call fit() first.")

        blended = np.zeros_like(next(iter(test_dict.values())), dtype=np.float64)
        for name, w in self.weights.items():
            if name in test_dict:
                blended += w * test_dict[name]

        return np.clip(blended, 0.0, 5.0)


class CrossFittedEnsembleBlender:
    """Nested, leak-free cross-fitted ensemble blender.

    Solves for optimal non-negative weights on K-1 folds, evaluates strictly
    on the held-out fold, and averages the resulting fold weights for test inference.
    Guarantees that blended OOF predictions are 100% out-of-fold without in-sample optimism.
    """

    def __init__(self):
        self.fold_weights: List[Dict[str, float]] = []
        self.final_weights: Dict[str, float] = {}
        self.model_names: List[str] = []
        self.blended_oof: Optional[np.ndarray] = None

    def fit(
        self,
        oof_dict: Dict[str, np.ndarray],
        y_true: np.ndarray,
        fold_indices: np.ndarray,
    ) -> CrossFittedEnsembleBlender:
        """Fit weights on outer folds and evaluate strictly out-of-fold."""
        self.model_names = list(oof_dict.keys())
        k_models = len(self.model_names)
        if k_models == 0:
            raise ValueError("oof_dict cannot be empty.")

        n_samples = len(y_true)
        self.blended_oof = np.zeros(n_samples, dtype=np.float64)
        unique_folds = sorted(np.unique(fold_indices))
        self.fold_weights = []

        OOF_mat = np.column_stack([oof_dict[m] for m in self.model_names])
        y_arr = np.asarray(y_true, dtype=np.float64)

        for fold in unique_folds:
            val_mask = fold_indices == fold
            train_mask = ~val_mask

            X_blend_train = OOF_mat[train_mask]
            y_blend_train = y_arr[train_mask]

            def loss(w: np.ndarray) -> float:
                b = np.dot(X_blend_train, w)
                return float(np.sqrt(np.mean((y_blend_train - b) ** 2)))

            init_w = np.ones(k_models) / k_models
            bounds = [(0.0, 1.0) for _ in range(k_models)]
            constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

            res = minimize(loss, init_w, bounds=bounds, constraints=constraints, method="SLSQP")
            best_w = res.x if res.success else init_w
            best_w = np.clip(best_w, 0.0, 1.0)
            best_w /= np.sum(best_w)

            fold_w_dict = {name: float(w) for name, w in zip(self.model_names, best_w)}
            self.fold_weights.append(fold_w_dict)

            # Predict the held-out fold strictly out-of-fold
            X_blend_val = OOF_mat[val_mask]
            self.blended_oof[val_mask] = np.dot(X_blend_val, best_w)

        # Average weights across all folds for final test predictions
        avg_weights = {}
        for name in self.model_names:
            avg_weights[name] = float(np.mean([fw[name] for fw in self.fold_weights]))
        tot = sum(avg_weights.values())
        self.final_weights = {name: w / tot for name, w in avg_weights.items()}

        logger.info(f"Cross-fitted blender complete. Final average weights: {self.final_weights}")
        return self

    def predict(self, test_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """Generate blended test predictions using averaged cross-fitted weights."""
        if not self.final_weights:
            raise RuntimeError("CrossFittedEnsembleBlender is not fitted. Call fit() first.")

        blended = np.zeros_like(next(iter(test_dict.values())), dtype=np.float64)
        for name, w in self.final_weights.items():
            if name in test_dict:
                blended += w * test_dict[name]

        return np.clip(blended, 0.0, 5.0)


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
