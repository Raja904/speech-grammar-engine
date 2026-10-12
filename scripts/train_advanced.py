"""Advanced Training & Ensembling Pipeline for Kaggle Spoken Grammar Scoring.

Trains and compares:
1. Mean Baseline Model
2. Ridge Regression (L2 linear model)
3. SVR (RBF non-linear kernel)
4. LightGBM / Gradient Boosted Trees
5. Multi-Layer Perceptron (MLP Neural Regressor)
6. Optimal Constrained Ensemble Blend (weights optimized via SLSQP to minimize OOF RMSE)

Generates final ensemble submission to outputs/submission.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.config import load_config
from src.data import load_dataset
from src.features import extract_dataset_features
from src.metrics import mean_absolute_error, pearson_correlation, root_mean_squared_error
from src.models import (
    EnsembleBlender,
    LGBMGrammarModel,
    MeanBaselineModel,
    MLPGrammarModel,
    RidgeGrammarModel,
    SVREngineModel,
    run_cv_experiment,
)
from src.validation import create_folds


def main():
    config = load_config()
    print("=" * 70)
    print("  Kaggle Spoken Grammar Scoring — Advanced Benchmark & Ensembling")
    print("=" * 70)

    train_csv = config.paths.train_csv_path
    test_csv = config.paths.test_csv_path
    output_dir = config.paths.output_dir
    cache_dir = config.paths.cache_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load datasets
    print("\n[1/6] Loading datasets & audio paths...")
    train_df, train_mapping = load_dataset(
        csv_path=train_csv,
        audio_dirs=[config.paths.train_audio_path, config.paths.data_dir],
    )
    test_df, test_mapping = load_dataset(
        csv_path=test_csv,
        audio_dirs=[config.paths.test_audio_path, config.paths.data_dir],
        is_test=True,
    )

    target_col = train_mapping.target_col or "label"
    print(f"Train samples: {len(train_df)} | Test samples: {len(test_df)}")

    # 2. Load or create 5-fold splits
    folds_path = output_dir / "folds.csv"
    if folds_path.exists():
        print(f"\n[2/6] Loading verified fold assignments from {folds_path}")
        fold_df = pd.read_csv(folds_path)
        if "resolved_audio_path" not in fold_df.columns:
            fold_df["resolved_audio_path"] = train_df["resolved_audio_path"]
        if target_col not in fold_df.columns:
            fold_df[target_col] = train_df[target_col]
    else:
        print("\n[2/6] Generating leak-free 5-fold cross-validation...")
        fold_df = create_folds(train_df, target_col=target_col, n_splits=5, seed=config.random_seed)
        fold_df.to_csv(folds_path, index=False)

    y_train = fold_df[target_col].to_numpy(dtype=np.float64)

    # 3. Audio Features (Cached)
    print("\n[3/6] Loading / extracting audio embeddings...")
    train_cache = cache_dir / "train_features.npy"
    test_cache = cache_dir / "test_features.npy"

    method = "wav2vec2" if is_torch_available() else "acoustic"
    X_train = extract_dataset_features(
        fold_df["resolved_audio_path"].tolist(),
        method=method,
        cache_path=train_cache,
    )
    X_test = extract_dataset_features(
        test_df["resolved_audio_path"].tolist(),
        method=method,
        cache_path=test_cache,
    )
    print(f"Features ready -> X_train: {X_train.shape} | X_test: {X_test.shape}")

    # 4. Benchmarking Models
    print("\n[4/6] Executing 5-Fold Cross-Validation for All Models...")

    # Model 1: Mean Baseline
    mean_oof = np.zeros(len(fold_df), dtype=np.float64)
    for fold in range(5):
        val_mask = fold_df["fold"] == fold
        baseline = MeanBaselineModel(strategy="mean")
        baseline.fit(None, y_train[~val_mask])
        mean_oof[val_mask] = baseline.predict(y_train[val_mask])
    mean_rmse = root_mean_squared_error(y_train, mean_oof)
    mean_r = pearson_correlation(y_train, mean_oof)
    mean_mae = mean_absolute_error(y_train, mean_oof)

    # Model 2: Ridge Regression
    ridge_res = run_cv_experiment(
        fold_df=fold_df,
        features=X_train,
        target_col=target_col,
        model_factory=lambda: RidgeGrammarModel(),
        model_name="Ridge Regression",
    )

    # Model 3: SVR
    svr_res = run_cv_experiment(
        fold_df=fold_df,
        features=X_train,
        target_col=target_col,
        model_factory=lambda: SVREngineModel(C=1.0, epsilon=0.1),
        model_name="SVR (RBF Kernel)",
    )

    # Model 4: LightGBM / GBDT
    lgbm_res = run_cv_experiment(
        fold_df=fold_df,
        features=X_train,
        target_col=target_col,
        model_factory=lambda: LGBMGrammarModel(n_estimators=150, learning_rate=0.03, max_depth=4),
        model_name="LightGBM Regressor",
    )

    # Model 5: MLP Neural Regressor
    mlp_res = run_cv_experiment(
        fold_df=fold_df,
        features=X_train,
        target_col=target_col,
        model_factory=lambda: MLPGrammarModel(hidden_layer_sizes=(256, 64), alpha=0.05),
        model_name="MLP Neural Regressor",
    )

    # 5. Generate Out-of-Fold & Test predictions for individual models
    models_dict = {
        "Ridge": ridge_res,
        "SVR": svr_res,
        "LightGBM": lgbm_res,
        "MLP": mlp_res,
    }

    oof_dict = {name: res["oof_predictions"] for name, res in models_dict.items()}

    test_preds_dict = {}
    for name, res in models_dict.items():
        fold_preds = [m.predict(X_test) for m in res["models"]]
        test_preds_dict[name] = np.mean(fold_preds, axis=0)

    # Model 6: Optimal Constrained Ensemble Blend
    print("\n[5/6] Optimizing Constrained Ensemble Blender...")
    blender = EnsembleBlender()
    blender.fit(oof_dict, y_train)

    ensemble_oof = np.zeros(len(fold_df), dtype=np.float64)
    for name, weight in blender.weights.items():
        ensemble_oof += weight * oof_dict[name]

    ens_rmse = root_mean_squared_error(y_train, ensemble_oof)
    ens_r = pearson_correlation(y_train, ensemble_oof)
    ens_mae = mean_absolute_error(y_train, ensemble_oof)

    # 6. Model Comparison Table
    print("\n[6/6] === Final Leaderboard of Evaluated Models ===")
    summary_data = [
        {"Model": "Mean Baseline", "OOF RMSE": mean_rmse, "Pearson r": mean_r, "MAE": mean_mae, "Blend Weight": "-"},
        {"Model": "Ridge Regression", "OOF RMSE": ridge_res["oof_rmse"], "Pearson r": ridge_res["oof_pearson_r"], "MAE": ridge_res["oof_mae"], "Blend Weight": f"{blender.weights.get('Ridge', 0.0):.3f}"},
        {"Model": "SVR (RBF Kernel)", "OOF RMSE": svr_res["oof_rmse"], "Pearson r": svr_res["oof_pearson_r"], "MAE": svr_res["oof_mae"], "Blend Weight": f"{blender.weights.get('SVR', 0.0):.3f}"},
        {"Model": "LightGBM Regressor", "OOF RMSE": lgbm_res["oof_rmse"], "Pearson r": lgbm_res["oof_pearson_r"], "MAE": lgbm_res["oof_mae"], "Blend Weight": f"{blender.weights.get('LightGBM', 0.0):.3f}"},
        {"Model": "MLP Neural Regressor", "OOF RMSE": mlp_res["oof_rmse"], "Pearson r": mlp_res["oof_pearson_r"], "MAE": mlp_res["oof_mae"], "Blend Weight": f"{blender.weights.get('MLP', 0.0):.3f}"},
        {"Model": "⭐ Optimal Ensemble Blend", "OOF RMSE": ens_rmse, "Pearson r": ens_r, "MAE": ens_mae, "Blend Weight": "1.000"},
    ]
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))

    # Generate test ensemble predictions
    final_test_preds = blender.predict(test_preds_dict)
    final_test_preds = np.clip(final_test_preds, 0.0, 5.0)

    # Persist final submission
    sub_df = pd.DataFrame({
        test_mapping.filename_col: test_df[test_mapping.filename_col],
        target_col: final_test_preds,
    })
    sub_path = output_dir / "submission.csv"
    sub_df.to_csv(sub_path, index=False)
    print(f"\nFinal Ensemble Submission written to: {sub_path}")
    print(f"Shape: {sub_df.shape} | Score Range: [{final_test_preds.min():.4f}, {final_test_preds.max():.4f}]")
    print(sub_df.head(10))


def is_torch_available() -> bool:
    try:
        import torch
        import transformers
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    main()
