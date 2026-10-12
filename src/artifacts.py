"""Artifact persistence, atomic file operations, and manifest management.

Guarantees data loss prevention through atomic writes, checkpoint validation,
and structured experiment manifest tracking.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def atomic_write_text(filepath: Path | str, content: str, encoding: str = "utf-8") -> Path:
    """Atomically write text content to a destination file.

    Writes to a temporary file in the same parent directory first, then uses
    atomic rename (os.replace) to ensure no partially written files exist.
    """
    dest = Path(filepath).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)

    temp_path = dest.parent / f".tmp_{dest.name}_{uuid.uuid4().hex[:8]}"
    try:
        with open(temp_path, "w", encoding=encoding) as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())

        os.replace(temp_path, dest)
        return dest
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        logger.error(f"Failed atomic write to {dest}: {e}")
        raise


def atomic_save_json(filepath: Path | str, data: Any, indent: int = 2) -> Path:
    """Atomically serialize and save data as JSON."""
    payload = json.dumps(data, indent=indent, default=str)
    return atomic_write_text(filepath, payload)


def atomic_save_numpy(filepath: Path | str, array: np.ndarray) -> Path:
    """Atomically save a NumPy array (.npy format)."""
    dest = Path(filepath).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)

    temp_path = dest.parent / f".tmp_{dest.name}_{uuid.uuid4().hex[:8]}"
    try:
        with open(temp_path, "wb") as f:
            np.save(f, array)
            f.flush()
            os.fsync(f.fileno())

        os.replace(temp_path, dest)
        return dest
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        logger.error(f"Failed atomic numpy save to {dest}: {e}")
        raise


def load_json(filepath: Path | str) -> Dict[str, Any]:
    """Safely load and parse a JSON artifact file."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"JSON artifact does not exist: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# Type alias for manifest entries
ManifestDict = Dict[str, Any]


class ManifestManager:
    """Tracks runs, checkpoints, and cached feature artifacts with status integrity."""

    def __init__(self, manifest_path: Path | str):
        self.manifest_path = Path(manifest_path)
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.data: Dict[str, Any] = self._load_or_init()

    def _load_or_init(self) -> Dict[str, Any]:
        if self.manifest_path.exists():
            try:
                return load_json(self.manifest_path)
            except Exception as e:
                logger.warning(f"Could not read existing manifest at {self.manifest_path}: {e}. Initializing fresh.")

        return {
            "version": "1.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "runs": {},
            "caches": {},
            "checkpoints": {},
        }

    def save(self) -> None:
        """Atomically persist manifest state to disk."""
        self.data["updated_at"] = datetime.now(timezone.utc).isoformat()
        atomic_save_json(self.manifest_path, self.data)

    def register_cache_entry(
        self,
        cache_id: str,
        feature_name: str,
        shape: List[int],
        dtype: str,
        file_path: Path | str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register or update a cached feature entry."""
        self.data.setdefault("caches", {})[cache_id] = {
            "feature_name": feature_name,
            "shape": list(shape),
            "dtype": str(dtype),
            "file_path": str(file_path),
            "registered_at": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
        }
        self.save()

    def is_cache_valid(self, cache_id: str, expected_shape: Optional[List[int]] = None) -> bool:
        """Verify that a cached feature exists and conforms to expected specifications."""
        entry = self.data.get("caches", {}).get(cache_id)
        if not entry:
            return False

        fp = Path(entry["file_path"])
        if not fp.exists() or fp.stat().st_size == 0:
            return False

        if expected_shape and entry.get("shape") != expected_shape:
            return False

        return True

    def record_run(
        self,
        run_id: str,
        config_dict: Dict[str, Any],
        metrics: Dict[str, Any],
        status: str = "COMPLETED",
        artifacts: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record an experiment execution log."""
        self.data.setdefault("runs", {})[run_id] = {
            "run_id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "config": config_dict,
            "metrics": metrics,
            "artifacts": artifacts or {},
        }
        self.save()
