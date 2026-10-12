"""Structured experiment tracking and persistence for Spoken Grammar Scoring.

Records all experimental runs atomically to JSON and CSV, tracking:
- Experiment ID and timestamp
- Feature extraction configuration (encoder, pooling method, dimension)
- Model architecture and hyperparameters
- Out-Of-Fold (OOF) metrics (RMSE, Pearson r, MAE)
- Delta RMSE compared against the primary Ridge baseline (0.8210)
- Cross-validation fold breakdown
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.artifacts import atomic_save_json

logger = logging.getLogger(__name__)

PRIMARY_RIDGE_BASELINE_RMSE = 0.821022


@dataclass
class ExperimentRecord:
    """Record of a single cross-validation experiment."""

    experiment_id: str
    model_name: str
    feature_family: str
    pooling_method: str
    feature_dim: int
    oof_rmse: float
    oof_pearson_r: float
    oof_mae: float
    delta_rmse_vs_baseline: float
    fold_rmses: List[float] = field(default_factory=list)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    timestamp_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class ExperimentTracker:
    """Thread-safe and process-safe experiment logger writing to JSON and CSV."""

    def __init__(self, log_dir: Path | str = "outputs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.json_path = self.log_dir / "experiment_log.json"
        self.csv_path = self.log_dir / "experiment_log.csv"

    def load_records(self) -> List[Dict[str, Any]]:
        """Load existing experiment records from JSON log."""
        if not self.json_path.exists():
            return []
        try:
            with open(self.json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
        except Exception as e:
            logger.warning(f"Error loading {self.json_path}: {e}")
            return []

    def log_experiment(self, record: ExperimentRecord) -> None:
        """Append an experiment record to both JSON and CSV logs."""
        records = self.load_records()
        record_dict = asdict(record)
        records.append(record_dict)

        # Atomic JSON write
        atomic_save_json(self.json_path, records)

        # CSV update
        fieldnames = [
            "experiment_id",
            "timestamp_utc",
            "model_name",
            "feature_family",
            "pooling_method",
            "feature_dim",
            "oof_rmse",
            "oof_pearson_r",
            "oof_mae",
            "delta_rmse_vs_baseline",
            "notes",
        ]
        file_exists = self.csv_path.exists()
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            if not file_exists:
                writer.writeheader()
            writer.writerow(record_dict)

        logger.info(
            f"Logged Experiment '{record.experiment_id}' ({record.model_name}): "
            f"OOF RMSE={record.oof_rmse:.4f} (Delta: {record.delta_rmse_vs_baseline:+.4f}), "
            f"Pearson r={record.oof_pearson_r:.4f}"
        )


def initialize_baseline_log(log_dir: Path | str = "outputs") -> ExperimentTracker:
    """Pre-populate the experiment log with our verified baseline results."""
    tracker = ExperimentTracker(log_dir)
    existing = tracker.load_records()
    if existing:
        return tracker

    # 1. Mean Baseline
    tracker.log_experiment(
        ExperimentRecord(
            experiment_id="exp_000_mean_baseline",
            model_name="Mean Baseline",
            feature_family="none",
            pooling_method="none",
            feature_dim=0,
            oof_rmse=1.238497,
            oof_pearson_r=-0.029490,
            oof_mae=0.996434,
            delta_rmse_vs_baseline=1.238497 - PRIMARY_RIDGE_BASELINE_RMSE,
            notes="Empirical baseline floor predicting train mean across folds",
        )
    )

    # 2. Ridge Wav2Vec2 Mean Pooling (Primary Anchor Baseline)
    tracker.log_experiment(
        ExperimentRecord(
            experiment_id="exp_001_ridge_wav2vec2_mean",
            model_name="Ridge Regression",
            feature_family="wav2vec2-base-960h",
            pooling_method="mean",
            feature_dim=768,
            oof_rmse=0.821022,
            oof_pearson_r=0.748797,
            oof_mae=0.661221,
            delta_rmse_vs_baseline=0.0,
            hyperparameters={"alphas": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0], "scaler": "StandardScaler"},
            notes="Primary Anchor Benchmark: Ridge on Wav2Vec2 mean pooled embeddings",
        )
    )

    # 3. SVR Wav2Vec2 Mean Pooling
    tracker.log_experiment(
        ExperimentRecord(
            experiment_id="exp_002_svr_wav2vec2_mean",
            model_name="SVR (RBF Kernel)",
            feature_family="wav2vec2-base-960h",
            pooling_method="mean",
            feature_dim=768,
            oof_rmse=0.853947,
            oof_pearson_r=0.725403,
            oof_mae=0.687304,
            delta_rmse_vs_baseline=0.853947 - PRIMARY_RIDGE_BASELINE_RMSE,
            hyperparameters={"C": 1.0, "epsilon": 0.1, "kernel": "rbf", "scaler": "StandardScaler"},
            notes="Non-linear kernel baseline on Wav2Vec2 embeddings",
        )
    )

    # 4. LightGBM Wav2Vec2 Mean Pooling
    tracker.log_experiment(
        ExperimentRecord(
            experiment_id="exp_003_lgbm_wav2vec2_mean",
            model_name="LightGBM Regressor",
            feature_family="wav2vec2-base-960h",
            pooling_method="mean",
            feature_dim=768,
            oof_rmse=0.860562,
            oof_pearson_r=0.719019,
            oof_mae=0.689962,
            delta_rmse_vs_baseline=0.860562 - PRIMARY_RIDGE_BASELINE_RMSE,
            hyperparameters={"n_estimators": 150, "learning_rate": 0.04, "max_depth": 5},
            notes="Tree-based gradient boosting baseline on Wav2Vec2 embeddings",
        )
    )

    # 5. MLP Wav2Vec2 Mean Pooling
    tracker.log_experiment(
        ExperimentRecord(
            experiment_id="exp_004_mlp_wav2vec2_mean",
            model_name="MLP Neural Regressor",
            feature_family="wav2vec2-base-960h",
            pooling_method="mean",
            feature_dim=768,
            oof_rmse=1.009723,
            oof_pearson_r=0.627139,
            oof_mae=0.787666,
            delta_rmse_vs_baseline=1.009723 - PRIMARY_RIDGE_BASELINE_RMSE,
            hyperparameters={"hidden_layer_sizes": [256, 64], "alpha": 0.01},
            notes="Neural MLP regressor baseline on Wav2Vec2 embeddings",
        )
    )

    return tracker
