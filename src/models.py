"""Model definitions and baselines for Spoken Grammar Scoring.

Implements the Phase 2 MeanBaselineModel and scaffolding for subsequent
Ridge and SVR regression models.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class BaseGrammarModel(ABC):
    """Abstract interface for all grammar scoring models."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> BaseGrammarModel:
        """Fit model on training features and targets."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate continuous grammar score predictions."""
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
        # Ensure clipping to valid domain [0.0, 5.0]
        return np.clip(preds, 0.0, 5.0)
