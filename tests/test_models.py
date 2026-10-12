"""Tests for model training, Ridge, SVR, LightGBM, MLP, Ensembling, and cross-validation execution."""

import numpy as np
import pandas as pd
import pytest

from src.models import (
    EnsembleBlender,
    LGBMGrammarModel,
    MeanBaselineModel,
    MLPGrammarModel,
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


def test_lgbm_grammar_model():
    X = np.random.randn(25, 4).astype(np.float32)
    y = np.random.uniform(1.0, 4.0, size=25)
    model = LGBMGrammarModel(n_estimators=10)
    model.fit(X, y)
    preds = model.predict(X[:5])
    assert len(preds) == 5
    assert np.all((preds >= 0.0) & (preds <= 5.0))


def test_mlp_grammar_model():
    X = np.random.randn(25, 4).astype(np.float32)
    y = np.random.uniform(1.0, 4.0, size=25)
    model = MLPGrammarModel(hidden_layer_sizes=(32, 16))
    model.fit(X, y)
    preds = model.predict(X[:5])
    assert len(preds) == 5
    assert np.all((preds >= 0.0) & (preds <= 5.0))


def test_ensemble_blender():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    oof1 = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
    oof2 = np.array([0.9, 2.1, 2.9, 4.1, 4.9])

    blender = EnsembleBlender()
    blender.fit({"m1": oof1, "m2": oof2}, y)

    assert len(blender.weights) == 2
    assert pytest.approx(sum(blender.weights.values()), 0.01) == 1.0

    test_preds = blender.predict({"m1": oof1[:2], "m2": oof2[:2]})
    assert len(test_preds) == 2
    assert np.all((test_preds >= 0.0) & (test_preds <= 5.0))


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
