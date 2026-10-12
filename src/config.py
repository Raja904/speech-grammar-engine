"""Configuration management for the Kaggle Spoken Grammar Scoring Engine.

Supports local development and automated Kaggle environment detection.
Provides typed, validated configuration parameters across all pipeline stages.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


def is_kaggle_environment() -> bool:
    """Detect if code is executing inside a Kaggle notebook/script session."""
    return bool(
        os.path.exists("/kaggle/input")
        or os.environ.get("KAGGLE_KERNEL_RUN_TYPE")
        or os.environ.get("KAGGLE_CONTAINER_NAME")
    )


def discover_kaggle_input_dir() -> Optional[Path]:
    """Search `/kaggle/input` for the competition dataset directory containing train.csv."""
    kaggle_input = Path("/kaggle/input")
    if not kaggle_input.exists():
        return None

    # Search subdirectories for train.csv
    for candidate in kaggle_input.glob("**/train.csv"):
        return candidate.parent

    # If train.csv not found directly, return the first directory in /kaggle/input if any
    subdirs = [p for p in kaggle_input.iterdir() if p.is_dir()]
    if subdirs:
        return subdirs[0]

    return kaggle_input


@dataclass
class PathsConfig:
    """File paths and directory structures for dataset inputs, caches, and outputs."""

    data_dir: Path = Path("data")
    output_dir: Path = Path("outputs")
    cache_dir: Path = Path("features_cache")
    checkpoints_dir: Path = Path("checkpoints")
    runs_dir: Path = Path("runs")

    train_csv_filename: str = "train.csv"
    test_csv_filename: str = "test.csv"
    sample_submission_filename: str = "sample_submission.csv"
    audio_dir_train: str = "train"
    audio_dir_test: str = "test"

    @property
    def train_csv_path(self) -> Path:
        return self.data_dir / self.train_csv_filename

    @property
    def test_csv_path(self) -> Path:
        return self.data_dir / self.test_csv_filename

    @property
    def sample_submission_path(self) -> Path:
        return self.data_dir / self.sample_submission_filename

    @property
    def train_audio_path(self) -> Path:
        # Check if audio_dir_train exists under data_dir, or if audio files are directly in data_dir
        candidate = self.data_dir / self.audio_dir_train
        return candidate if candidate.exists() else self.data_dir

    @property
    def test_audio_path(self) -> Path:
        candidate = self.data_dir / self.audio_dir_test
        return candidate if candidate.exists() else self.data_dir

    def ensure_directories(self) -> None:
        """Create output, cache, checkpoint and run directories safely."""
        for directory in [self.output_dir, self.cache_dir, self.checkpoints_dir, self.runs_dir]:
            directory.mkdir(parents=True, exist_ok=True)


@dataclass
class AudioConfig:
    """Audio signal processing and encoder compatibility settings."""

    target_sample_rate: int = 16000
    mono: bool = True
    max_duration_sec: float = 30.0
    silence_threshold_db: float = -40.0
    clipping_threshold: float = 0.999


@dataclass
class ValidationConfig:
    """Cross-validation and fold configuration."""

    n_splits: int = 5
    random_seed: int = 42
    target_col: Optional[str] = None
    id_col: Optional[str] = None
    audio_col: Optional[str] = None
    group_col: Optional[str] = None  # Speaker/Group column if available


@dataclass
class AuditConfig:
    """Settings controlling dataset and waveform auditing."""

    check_audio_signals: bool = True
    compute_signal_stats: bool = True
    max_samples: Optional[int] = None  # None indicates all samples


@dataclass
class FeatureConfig:
    """Audio/Text embedding feature extraction parameters."""

    encoder_name: str = "facebook/wav2vec2-base-960h"
    encoder_revision: str = "main"
    pooling_method: str = "mean"
    batch_size: int = 16


@dataclass
class ModelConfig:
    """Downstream regression models and hyperparameter search space."""

    baseline_strategy: str = "mean"
    ridge_alphas: List[float] = field(default_factory=lambda: [0.01, 0.1, 1.0, 10.0, 100.0])
    svr_c: float = 1.0
    svr_epsilon: float = 0.1


@dataclass
class ProjectConfig:
    """Top-level unified project configuration."""

    project_name: str = "spoken-grammar-scoring"
    random_seed: int = 42
    paths: PathsConfig = field(default_factory=PathsConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    audit: AuditConfig = field(default_factory=AuditConfig)
    feature_extraction: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    @classmethod
    def load(cls, config_path: Optional[str | Path] = None, auto_kaggle: bool = True) -> ProjectConfig:
        """Load configuration from a YAML file, applying environment overrides if on Kaggle."""
        data: Dict[str, Any] = {}
        if config_path:
            p = Path(config_path)
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}

        # Parse nested dictionaries
        paths_data = data.get("paths", {})
        audio_data = data.get("audio", {})
        val_data = data.get("validation", {})
        audit_data = data.get("audit", {})
        feat_data = data.get("feature_extraction", {})
        model_data = data.get("model", {})

        # Handle PathsConfig with string-to-Path conversion
        paths_cfg = PathsConfig(
            data_dir=Path(paths_data.get("data_dir", "data")),
            output_dir=Path(paths_data.get("output_dir", "outputs")),
            cache_dir=Path(paths_data.get("cache_dir", "features_cache")),
            checkpoints_dir=Path(paths_data.get("checkpoints_dir", "checkpoints")),
            runs_dir=Path(paths_data.get("runs_dir", "runs")),
            train_csv_filename=paths_data.get("train_csv_filename", "train.csv"),
            test_csv_filename=paths_data.get("test_csv_filename", "test.csv"),
            sample_submission_filename=paths_data.get("sample_submission_filename", "sample_submission.csv"),
            audio_dir_train=paths_data.get("audio_dir_train", "train"),
            audio_dir_test=paths_data.get("audio_dir_test", "test"),
        )

        cfg = cls(
            project_name=data.get("project_name", "spoken-grammar-scoring"),
            random_seed=data.get("random_seed", 42),
            paths=paths_cfg,
            audio=AudioConfig(**audio_data) if audio_data else AudioConfig(),
            validation=ValidationConfig(**val_data) if val_data else ValidationConfig(),
            audit=AuditConfig(**audit_data) if audit_data else AuditConfig(),
            feature_extraction=FeatureConfig(**feat_data) if feat_data else FeatureConfig(),
            model=ModelConfig(**model_data) if model_data else ModelConfig(),
        )

        # Automatic Kaggle path adaptation
        if auto_kaggle and is_kaggle_environment():
            kaggle_data = discover_kaggle_input_dir()
            if kaggle_data:
                cfg.paths.data_dir = kaggle_data
                logger.info(f"Kaggle environment detected. Adjusted data_dir to: {kaggle_data}")
            # Ensure working outputs reside in /kaggle/working
            cfg.paths.output_dir = Path("/kaggle/working/outputs")
            cfg.paths.cache_dir = Path("/kaggle/working/features_cache")
            cfg.paths.checkpoints_dir = Path("/kaggle/working/checkpoints")
            cfg.paths.runs_dir = Path("/kaggle/working/runs")

        return cfg

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary representation, serializing Paths to strings."""
        raw = asdict(self)

        def _clean_paths(d: Any) -> Any:
            if isinstance(d, dict):
                return {k: _clean_paths(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [_clean_paths(v) for v in d]
            elif isinstance(d, Path):
                return str(d)
            return d

        return _clean_paths(raw)


def setup_logging(
    log_level: int = logging.INFO,
    log_file: Optional[Path | str] = None,
) -> None:
    """Configure structured logging for stdout and optional file logging."""
    fmt = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_file:
        p = Path(log_file)
        p.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(str(p), encoding="utf-8"))

    logging.basicConfig(
        level=log_level,
        format=fmt,
        handlers=handlers,
        force=True,
    )
