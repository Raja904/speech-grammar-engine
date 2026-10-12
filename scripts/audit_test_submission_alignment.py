"""Diagnostic script to investigate row and ID discrepancy between test.csv and sample_submission.csv.

Analyzes:
1. Row counts and column schemas
2. Exact symmetric difference in IDs (missing in submission vs missing in test)
3. ID ordering alignment
4. Output schema contract to ensure predictions are never dropped, reordered, or fabricated
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.artifacts import atomic_save_json
from src.config import load_config
from src.data import detect_columns


def audit_test_submission_alignment(config_path: str = "configs/default.yaml") -> dict:
    config = load_config(config_path)
    output_dir = config.paths.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    test_csv = config.paths.test_csv_path
    sub_csv = config.paths.sample_submission_path

    print("=" * 70)
    print("  Test Dataset vs. Sample Submission Schema & ID Alignment Audit")
    print("=" * 70)

    if not test_csv.exists():
        print(f"ERROR: test.csv not found at {test_csv}")
        return {"error": f"test.csv not found at {test_csv}"}

    test_df = pd.read_csv(test_csv)
    test_mapping = detect_columns(test_df, is_test=True)
    test_col = test_mapping.filename_col
    test_ids = test_df[test_col].astype(str).tolist()

    sub_exists = sub_csv.exists()
    sub_col = None
    sub_ids = []
    sub_df = None

    if sub_exists:
        sub_df = pd.read_csv(sub_csv)
        sub_mapping = detect_columns(sub_df, is_test=True)
        sub_col = sub_mapping.filename_col
        sub_ids = sub_df[sub_col].astype(str).tolist()

    print(f"\n1. File Check:")
    print(f"   - test.csv path: {test_csv} (Exists: True)")
    print(f"   - sample_submission.csv path: {sub_csv} (Exists: {sub_exists})")

    print(f"\n2. Shapes & Columns:")
    print(f"   - test.csv shape: {test_df.shape} | Columns: {list(test_df.columns)}")
    if sub_df is not None:
        print(f"   - sample_submission.csv shape: {sub_df.shape} | Columns: {list(sub_df.columns)}")

    set_test = set(test_ids)
    set_sub = set(sub_ids) if sub_exists else set()

    only_in_test = sorted(list(set_test - set_sub))
    only_in_sub = sorted(list(set_sub - set_test))
    intersection = sorted(list(set_test.intersection(set_sub)))

    print(f"\n3. Set-Level Overlap & Discrepancy:")
    print(f"   - Unique IDs in test.csv: {len(set_test)}")
    print(f"   - Unique IDs in sample_submission.csv: {len(set_sub)}")
    print(f"   - IDs present in BOTH: {len(intersection)}")
    print(f"   - IDs in test.csv but MISSING from sample_submission.csv: {len(only_in_test)}")
    if only_in_test:
        print(f"     Examples ({min(5, len(only_in_test))} of {len(only_in_test)}): {only_in_test[:5]}")
    print(f"   - IDs in sample_submission.csv but MISSING from test.csv: {len(only_in_sub)}")
    if only_in_sub:
        print(f"     Examples ({min(5, len(only_in_sub))} of {len(only_in_sub)}): {only_in_sub[:5]}")

    # Check ordering
    is_ordered_match = False
    if len(test_ids) == len(sub_ids):
        is_ordered_match = test_ids == sub_ids
    print(f"\n4. Sequence Alignment:")
    print(f"   - Exact sequence and shape match: {is_ordered_match}")

    # Safety rules recommendations
    print(f"\n5. Submission Contract Recommendation:")
    if len(only_in_test) > 0 and len(only_in_sub) == 0:
        print(
            "   -> NOTE: sample_submission.csv appears to be a strict subset (204 rows) of test.csv (216 rows).\n"
            "   -> Safe Export Rule: When creating the competition submission, if the evaluator expects\n"
            "      the exact sample_submission.csv index, filter test predictions strictly on sample_sub IDs\n"
            "      in the exact sample_sub row order, preserving test predictions for all matching IDs without dropping."
        )

    report_payload = {
        "test_shape": list(test_df.shape),
        "test_columns": list(test_df.columns),
        "test_id_column": test_col,
        "sample_submission_shape": list(sub_df.shape) if sub_df is not None else None,
        "sample_submission_columns": list(sub_df.columns) if sub_df is not None else [],
        "sample_submission_id_column": sub_col,
        "total_test_ids": len(set_test),
        "total_sub_ids": len(set_sub),
        "intersection_count": len(intersection),
        "only_in_test_count": len(only_in_test),
        "only_in_test_ids": only_in_test,
        "only_in_sub_count": len(only_in_sub),
        "only_in_sub_ids": only_in_sub,
        "exact_sequence_match": is_ordered_match,
    }

    report_path = output_dir / "submission_alignment_audit.json"
    atomic_save_json(report_path, report_payload)
    print(f"\nSaved alignment diagnostic audit to: {report_path}")
    return report_payload


if __name__ == "__main__":
    audit_test_submission_alignment()
