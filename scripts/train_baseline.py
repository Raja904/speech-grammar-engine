"""Standalone training and evaluation script for Kaggle Spoken Grammar Scoring Engine.

Executes:
1. Mean Baseline Benchmark (empirical floor)
2. Audio Feature Extraction (with caching)
3. Ridge Regression Benchmark (with 5-fold cross-validation)
4. SVR Benchmark (RBF kernel)
5. Test Inference & Submission Generation (with bounds checking [0.0, 5.0])
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.config import load_config
from src.data import load_dataset
from src.features import extract_dataset_features
from src.metrics import root_mean_squared_error, pearson_correlation, mean_absolute_error
from src.models import (
    MeanBaselineModel,
    RidgeGrammarModel,
    SVREngineModel,
    run_cv_experiment,
)
from src.validation import create_folds


def main():
    config = load_config()
    print("=" * 65)
    print("Kaggle Spoken Grammar Scoring — Training & Benchmarking Pipeline")
    print("=" * 65)

    train_csv = config.paths.train_csv_path
    test_csv = config.paths.test_csv_path
    output_dir = config.paths.output_dir
    cache_dir = config.paths.cache_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load train dataset & resolve audio paths
    print("\n[1/5] Loading datasets...")
    train_df, train_mapping = load_dataset(
        csv_path=train_csv,
        audio_dirs=[config.paths.train_audio_path, config.paths.data_dir],
    )
    test_df, test_mapping = load_dataset(
        csv_path=test_csv,
        audio_dirs=[config.paths.test_audio_path, config.paths.data_dir],
        is_test=True,
    )

    target_col = train_mapping.target_col or "score"
    print(f"Loaded Train: {len(train_df)} samples | Test: {len(test_df)} samples")
    print(f"Target column: '{target_col}' | Filename column: '{train_mapping.filename_col}'")

    # 2. Check / create 5-fold splits
    folds_path = output_dir / "folds.csv"
    if folds_path.exists():
        print(f"\n[2/5] Loading verified fold assignments from {folds_path}")
        fold_df = pd.read_csv(folds_path)
        # Merge target and audio paths if needed
        if "resolved_audio_path" not in fold_df.columns:
            fold_df = train_df.merge(fold_df[[train_mapping.filename_col, "fold"]], on=train_mapping.filename_col, how="left")
    else:
        print("\n[2/5] Creating 5-fold cross-validation splits...")
        fold_df = create_folds(train_df, target_col=target_col, n_splits=5, seed=config.random_seed)
        fold_df.to_csv(folds_path, index=False)

    y_train = fold_df[target_col].to_numpy(dtype=np.float64)

    # 3. Benchmark 1: Mean Baseline Model
    print("\n[3/5] Benchmarking Mean Baseline Model (Empirical Floor)...")
    mean_oof = np.zeros(len(fold_df), dtype=np.float64)
    for fold in range(5):
        val_mask = fold_df["fold"] == fold
        train_y = y_train[~val_mask]
        baseline = MeanBaselineModel(strategy="mean")
        baseline.fit(None, train_y)
        mean_oof[val_mask] = baseline.predict(y_train[val_mask])

    mean_rmse = root_mean_squared_error(y_train, mean_oof)
    mean_r = pearson_correlation(y_train, mean_oof)
    mean_mae = mean_absolute_error(y_train, mean_oof)
    print(f"  -> Mean Baseline OOF RMSE: {mean_rmse:.4f} | Pearson r: {mean_r:.4f} | MAE: {mean_mae:.4f}")

    # 4. Feature Extraction (Train & Test)
    print("\n[4/5] Extracting Audio Features (Classical + Wav2Vec2 with disk caching)...")
    train_cache = cache_dir / "train_features.npy"
    test_cache = cache_dir / "test_features.npy"

    method = "wav2vec2" if is_torch_available() else "acoustic"
    print(f"Feature extraction method: {method}")

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
    print(f"X_train shape: {X_train.shape} | X_test shape: {X_test.shape}")

    # 5. Benchmark 2: Ridge Regression
    print("\n[5/5] Training Benchmark Models...")
    ridge_res = run_cv_experiment(
        fold_df=fold_df,
        features=X_train,
        target_col=target_col,
        model_factory=lambda: RidgeGrammarModel(),
        model_name="Ridge Regression",
    )
    print(f"  -> Ridge OOF RMSE: {ridge_res['oof_rmse']:.4f} | Pearson r: {ridge_res['oof_pearson_r']:.4f}")

    # Benchmark 3: Support Vector Regression (SVR)
    svr_res = run_cv_experiment(
        fold_df=fold_df,
        features=X_train,
        target_col=target_col,
        model_factory=lambda: SVREngineModel(C=1.0, epsilon=0.1),
        model_name="SVR (RBF Kernel)",
    )
    print(f"  -> SVR OOF RMSE:   {svr_res['oof_rmse']:.4f} | Pearson r: {svr_res['oof_pearson_r']:.4f}")

    # Pick best model for test submission
    best_res = ridge_res if ridge_res["oof_rmse"] <= svr_res["oof_rmse"] else svr_res
    print(f"\nBest Model: {best_res['model_name']} (OOF RMSE: {best_res['oof_rmse']:.4f})")

    # Generate ensemble/fold-averaged test predictions
    test_preds_folds = []
    for model in best_res["models"]:
        test_preds_folds.append(model.predict(X_test))
    final_test_preds = np.clip(np.mean(test_preds_folds, axis=0), 0.0, 5.0)

    # 6. Generate Submission
    sub_df = pd.DataFrame({
        test_mapping.filename_col: test_df[test_mapping.filename_col],
        target_col: final_test_preds,
    })
    sub_path = output_dir / "submission.csv"
    sub_df.to_csv(sub_path, index=False)
    print(f"\nSubmission generated and saved to: {sub_path}")
    print(f"Shape: {sub_df.shape} | Target range: [{final_test_preds.min():.4f}, {final_test_preds.max():.4f}]")
    print(sub_df.head())


def is_torch_available() -> bool:
    try:
        import torch
        import transformers
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    main()
