"""Tests for model training, Ridge, SVR, MeanBaseline, and cross-validation execution."""

import numpy as np
import pandas as pd
import pytest

from src.models import (
    MeanBaselineModel,
    RidgeGrammarModel,
    SVREngineModel,
    run_cv_experiment,
)


def test_mean_baseline_model():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    model = MeanBaselineModel(strategy="mean")
    model.fit(None, y)
    preds = model.predict(np.zeros((3, 2)))
    assert len(preds) == 3
    assert np.allclose(preds, 3.0)


def test_ridge_grammar_model():
    X = np.random.randn(20, 5).astype(np.float32)
    y = np.random.uniform(1.0, 4.0, size=20)
    model = RidgeGrammarModel(alphas=[0.1, 1.0])
    model.fit(X, y)
    preds = model.predict(X[:5])
    assert len(preds) == 5
    assert np.all((preds >= 0.0) & (preds <= 5.0))


def test_svr_engine_model():
    X = np.random.randn(20, 5).astype(np.float32)
    y = np.random.uniform(1.0, 4.0, size=20)
    model = SVREngineModel(C=1.0)
    model.fit(X, y)
    preds = model.predict(X[:5])
    assert len(preds) == 5
    assert np.all((preds >= 0.0) & (preds <= 5.0))


def test_run_cv_experiment():
    n = 20
    fold_df = pd.DataFrame({
        "filename": [f"file_{i}.wav" for i in range(n)],
        "label": np.random.uniform(1.0, 5.0, size=n),
        "fold": [i % 5 for i in range(n)],
    })
    X = np.random.randn(n, 4).astype(np.float32)

    res = run_cv_experiment(
        fold_df=fold_df,
        features=X,
        target_col="label",
        model_factory=lambda: RidgeGrammarModel(alphas=[1.0]),
        model_name="TestRidge",
    )

    assert "oof_rmse" in res
    assert "oof_pearson_r" in res
    assert len(res["fold_metrics"]) == 5
    assert len(res["oof_predictions"]) == n
    assert len(res["models"]) == 5
