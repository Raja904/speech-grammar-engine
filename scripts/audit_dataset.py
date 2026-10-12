"""CLI Script to audit Kaggle Spoken Grammar Scoring dataset.

Usage:
    python scripts/audit_dataset.py [--config configs/default.yaml] [--data-dir /path/to/data] [--output-dir /path/to/outputs]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.artifacts import atomic_save_json, atomic_write_text
from src.audio_audit import run_full_dataset_audit
from src.config import ProjectConfig, setup_logging

logger = logging.getLogger("audit_dataset")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Kaggle Spoken Grammar Scoring Dataset")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Optional override for root data directory containing train.csv/test.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional override for directory where audit report artifacts are written.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on number of audio files to inspect for quick checks (defaults to all).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(log_level=logging.INFO)

    logger.info("Initializing configuration...")
    config = ProjectConfig.load(config_path=args.config, auto_kaggle=True)

    # Apply CLI overrides if given
    if args.data_dir:
        config.paths.data_dir = Path(args.data_dir)
    if args.output_dir:
        config.paths.output_dir = Path(args.output_dir)
    if args.max_samples is not None:
        config.audit.max_samples = args.max_samples

    config.paths.ensure_directories()

    train_csv = config.paths.train_csv_path
    test_csv = config.paths.test_csv_path
    sample_sub = config.paths.sample_submission_path

    logger.info(f"Target data directory: {config.paths.data_dir}")
    logger.info(f"Checking for train CSV: {train_csv}")
    logger.info(f"Checking for test CSV: {test_csv}")

    if not train_csv.exists():
        logger.error(
            f"Train CSV not found at '{train_csv}'.\n"
            "If running locally, place dataset files in 'data/' or specify '--data-dir'.\n"
            "If running on Kaggle, ensure the competition dataset is attached in notebook input."
        )
        return 1

    if not test_csv.exists():
        logger.error(f"Test CSV not found at '{test_csv}'.")
        return 1

    sample_sub_path = sample_sub if sample_sub.exists() else None

    logger.info("Starting dataset and waveform audit...")
    try:
        report = run_full_dataset_audit(
            train_csv_path=train_csv,
            test_csv_path=test_csv,
            sample_sub_path=sample_sub_path,
            audio_dirs_train=[config.paths.train_audio_path, config.paths.data_dir],
            audio_dirs_test=[config.paths.test_audio_path, config.paths.data_dir],
            silence_threshold_db=config.audio.silence_threshold_db,
            clipping_threshold=config.audio.clipping_threshold,
            max_audio_samples=config.audit.max_samples,
        )
    except Exception as e:
        logger.exception(f"Fatal error during dataset audit: {e}")
        return 1

    # Persist JSON report and Markdown summary
    out_dir = config.paths.output_dir
    json_path = out_dir / "audit_report.json"
    md_path = out_dir / "audit_report.md"

    logger.info(f"Saving JSON audit report to {json_path}")
    atomic_save_json(json_path, report.to_dict())

    md_content = report.to_markdown()
    logger.info(f"Saving Markdown summary to {md_path}")
    atomic_write_text(md_path, md_content)

    # Print summary to console
    print("\n" + "=" * 80)
    print(md_content)
    print("=" * 80 + "\n")

    if report.review_items:
        logger.warning(
            f"Audit finished with {len(report.review_items)} items requiring review. Check {md_path}."
        )
    else:
        logger.info("Audit completed successfully with no critical anomalies detected.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
