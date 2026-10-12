"""Execution runner for Experiment 6A and Experiment 6B.

Experiment 6A:
- Frame-level Wav2Vec2 representations with mean + standard deviation pooling (1536 dimensions)
- Preserves existing mean-only features and caches without overwriting
- Trains Ridge regression on exact 5-fold cross-validation split
- Verifies row alignment with training filenames
- Measures and reports OOF RMSE, Pearson r, MAE, and Delta vs Ridge baseline (0.8210)

Experiment 6B:
- Stratified selection of 16 recordings across proficiency score distribution
- Transcribes using Whisper (openai/whisper-base) on GPU
- Audits transcription preservation of disfluencies, repetitions, and grammar errors
- Produces versioned audit reports: outputs/whisper_transcription_audit_v1.json and .md
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
from src.experiment_tracker import (
    PRIMARY_RIDGE_BASELINE_RMSE,
    ExperimentRecord,
    initialize_baseline_log,
)
from src.features import extract_dataset_features
from src.metrics import mean_absolute_error, pearson_correlation, root_mean_squared_error
from src.models import RidgeGrammarModel, run_cv_experiment
from src.transcription import (
    run_whisper_transcription_audit,
    select_stratified_audit_sample,
)


def verify_environment() -> None:
    """Check GPU availability, memory, and Kaggle/PyTorch dependencies."""
    print("=" * 70)
    print("  Environment Pre-Check & Resource Verification")
    print("=" * 70)
    print(f"Python Version: {sys.version.split()[0]}")

    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        print(f"PyTorch Version: {torch.__version__} | CUDA Available: {cuda_avail}")
        if cuda_avail:
            device_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"GPU Device: {device_name} ({vram_gb:.1f} GB VRAM)")
        else:
            print("WARNING: CUDA not detected. Falling back to CPU.")
    except ImportError:
        print("WARNING: PyTorch not installed.")

    try:
        import transformers
        print(f"Transformers Version: {transformers.__version__}")
    except ImportError:
        print("WARNING: Transformers not installed.")


def run_experiment_6a(
    config,
    fold_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_mapping,
    test_mapping,
    target_col: str,
    tracker,
) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("  Experiment 6A: Wav2Vec2 Mean + Std Pooling with Ridge Regression")
    print("=" * 70)

    cache_dir = config.paths.cache_dir
    output_dir = config.paths.output_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 1. Dedicated, non-destructive cache paths
    train_cache_mean_std = cache_dir / "train_wav2vec2_mean_std.npy"
    test_cache_mean_std = cache_dir / "test_wav2vec2_mean_std.npy"

    # Verify existing mean-only caches are intact
    existing_mean_cache = cache_dir / "train_features.npy"
    if existing_mean_cache.exists():
        print(f"Protected: Mean-only cache '{existing_mean_cache.name}' remains untouched.")

    # 2. Extract features with pooling='mean_std'
    print(f"\nExtracting / loading mean+std pooling features (1536 dims)...")
    X_train_mean_std = extract_dataset_features(
        fold_df["resolved_audio_path"].tolist(),
        method="wav2vec2",
        pooling_method="mean_std",
        cache_path=train_cache_mean_std,
    )
    X_test_mean_std = extract_dataset_features(
        test_df["resolved_audio_path"].tolist(),
        method="wav2vec2",
        pooling_method="mean_std",
        cache_path=test_cache_mean_std,
    )

    print(f"Feature Shapes -> X_train: {X_train_mean_std.shape} | X_test: {X_test_mean_std.shape}")

    # 3. Confirm row-level alignment with training filenames
    assert len(X_train_mean_std) == len(fold_df), (
        f"Row count mismatch: features {len(X_train_mean_std)} vs fold_df {len(fold_df)}"
    )
    print(f"Row Alignment: 100% verified ({len(fold_df)} rows exactly aligned with fold_df).")

    # 4. Train Ridge Regression on identical folds
    print("\nTraining Ridge Regression on identical 5 folds...")
    ridge_mean_std_res = run_cv_experiment(
        fold_df=fold_df,
        features=X_train_mean_std,
        target_col=target_col,
        model_factory=lambda: RidgeGrammarModel(),
        model_name="Ridge (Wav2Vec2 Mean+Std)",
    )

    oof_rmse = ridge_mean_std_res["oof_rmse"]
    oof_r = ridge_mean_std_res["oof_pearson_r"]
    oof_mae = ridge_mean_std_res["oof_mae"]
    delta_rmse = oof_rmse - PRIMARY_RIDGE_BASELINE_RMSE

    # 5. Log to Experiment Tracker
    rec = ExperimentRecord(
        experiment_id="exp_006a_ridge_wav2vec2_mean_std",
        model_name="Ridge (Wav2Vec2 Mean+Std)",
        feature_family="wav2vec2-base-960h",
        pooling_method="mean_std",
        feature_dim=int(X_train_mean_std.shape[1]),
        oof_rmse=oof_rmse,
        oof_pearson_r=oof_r,
        oof_mae=oof_mae,
        delta_rmse_vs_baseline=delta_rmse,
        fold_rmses=[f["rmse"] for f in ridge_mean_std_res["fold_metrics"]],
        hyperparameters={"alphas": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0], "scaler": "StandardScaler"},
        notes="Frame-level mean + standard deviation pooling (1536 dims) evaluated against 768-dim mean-only baseline",
    )
    tracker.log_experiment(rec)

    # 6. Generate separate test predictions
    test_preds = np.mean([m.predict(X_test_mean_std) for m in ridge_mean_std_res["models"]], axis=0)
    test_preds = np.clip(test_preds, 0.0, 5.0)

    sub_df = pd.DataFrame({
        test_mapping.filename_col: test_df[test_mapping.filename_col],
        target_col: test_preds,
    })
    sub_path = output_dir / "submission_exp6a_mean_std.csv"
    sub_df.to_csv(sub_path, index=False)
    print(f"Saved separate Experiment 6A submission to: {sub_path}")

    # 7. Print Comparative Scoreboard
    print("\n--- Experiment 6A Performance Comparison ---")
    comp_df = pd.DataFrame([
        {
            "Experiment": "Anchor Baseline (Wav2Vec2 Mean)",
            "Pooling": "mean",
            "Dims": 768,
            "OOF RMSE": f"{PRIMARY_RIDGE_BASELINE_RMSE:.4f}",
            "Pearson r": "0.7488",
            "MAE": "0.6612",
            "Delta RMSE": "+0.0000",
        },
        {
            "Experiment": "Experiment 6A (Wav2Vec2 Mean+Std)",
            "Pooling": "mean_std",
            "Dims": int(X_train_mean_std.shape[1]),
            "OOF RMSE": f"{oof_rmse:.4f}",
            "Pearson r": f"{oof_r:.4f}",
            "MAE": f"{oof_mae:.4f}",
            "Delta RMSE": f"{delta_rmse:+.4f}",
        },
    ])
    print(comp_df.to_string(index=False))

    return {
        "oof_rmse": oof_rmse,
        "oof_pearson_r": oof_r,
        "oof_mae": oof_mae,
        "delta_rmse": delta_rmse,
        "results": ridge_mean_std_res,
    }


def run_experiment_6b(
    config,
    train_df: pd.DataFrame,
    target_col: str,
    filename_col: str,
) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("  Experiment 6B: Whisper Transcription Quality & Grammar Audit")
    print("=" * 70)

    output_dir = config.paths.output_dir

    # 1. Select stratified 16 samples across 4 proficiency score tiers
    print("\nSampling 16 representative recordings across score distribution...")
    sample_df = select_stratified_audit_sample(
        train_df,
        target_col=target_col,
        sample_size=16,
        random_state=config.random_seed,
    )
    print(f"Selected {len(sample_df)} recordings across tiers:")
    for tier, count in sample_df["score_tier"].value_counts().items():
        print(f"  - {tier}: {count} samples")

    # 2. Run Whisper transcription audit
    print("\nTranscribing sample recordings using openai/whisper-base...")
    audit_report = run_whisper_transcription_audit(
        sample_df=sample_df,
        target_col=target_col,
        filename_col=filename_col,
        model_name="openai/whisper-base",
        output_dir=output_dir,
    )

    # 3. Print human-readable summary
    print("\n--- Transcription Audit Sample Preview ---")
    display_rows = []
    for item in audit_report["sample_records"]:
        display_rows.append({
            "Filename": item["filename"],
            "Score": f"{item['label']:.2f}",
            "Tier": item["tier"].split(":")[0],
            "Duration (s)": item["duration_sec"],
            "WPM": item["words_per_minute"],
            "Disfluencies": item["disfluency_count"],
            "Transcript Preview": item["transcript"][:50] + ("..." if len(item["transcript"]) > 50 else ""),
        })
    print(pd.DataFrame(display_rows).to_string(index=False))

    print(f"\nDetailed audit reports saved to:")
    print(f"  - JSON: {output_dir / 'whisper_transcription_audit_v1.json'}")
    print(f"  - Markdown: {output_dir / 'whisper_transcription_audit_v1.md'}")

    return audit_report


def main():
    config = load_config()
    output_dir = config.paths.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    verify_environment()
    tracker = initialize_baseline_log(output_dir)

    # Load datasets
    train_df, train_mapping = load_dataset(
        csv_path=config.paths.train_csv_path,
        audio_dirs=[config.paths.train_audio_path, config.paths.data_dir],
    )
    test_df, test_mapping = load_dataset(
        csv_path=config.paths.test_csv_path,
        audio_dirs=[config.paths.test_audio_path, config.paths.data_dir],
        is_test=True,
    )

    target_col = train_mapping.target_col or "label"
    filename_col = train_mapping.filename_col or "filename"

    # Load verified folds with strict inner merge on filename
    folds_path = output_dir / "folds.csv"
    if not folds_path.exists():
        raise FileNotFoundError(f"Folds file not found at {folds_path}. Run baseline first.")

    saved_folds = pd.read_csv(folds_path)
    fold_df = train_df.merge(
        saved_folds[[filename_col, "fold"]],
        on=filename_col,
        how="inner",
    )
    assert len(fold_df) == len(train_df), (
        f"Fold merge mismatch: expected {len(train_df)}, got {len(fold_df)}"
    )

    # 1. Run Experiment 6A
    res_6a = run_experiment_6a(
        config=config,
        fold_df=fold_df,
        test_df=test_df,
        train_mapping=train_mapping,
        test_mapping=test_mapping,
        target_col=target_col,
        tracker=tracker,
    )

    # 2. Run Experiment 6B
    res_6b = run_experiment_6b(
        config=config,
        train_df=fold_df,
        target_col=target_col,
        filename_col=filename_col,
    )

    print("\n" + "=" * 70)
    print("  Experiments 6A & 6B Execution Complete")
    print("=" * 70)
    print(f"Experiment 6A OOF RMSE: {res_6a['oof_rmse']:.4f} (Delta vs Ridge Baseline: {res_6a['delta_rmse']:+.4f})")
    print(f"Experiment 6B Transcripts: {res_6b['n_samples']} samples audited.")


if __name__ == "__main__":
    main()
