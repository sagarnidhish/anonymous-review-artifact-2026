"""Storage-bounded evidence artifacts for the fresh matched benchmark."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np


COMPACT_NEXT_FRAME_MODE = "compact_samples_v1"


def representative_offsets(length: int, count: int = 16) -> np.ndarray:
    """Return deterministic endpoint-inclusive offsets into a sequence."""
    if length <= 0:
        raise ValueError("length must be positive")
    if count <= 0:
        raise ValueError("count must be positive")
    if length <= count:
        return np.arange(length, dtype=np.int64)
    offsets = np.rint(np.linspace(0, length - 1, count)).astype(np.int64)
    return np.unique(offsets)


def _per_step_errors(pred: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    axes = tuple(range(1, pred.ndim))
    difference = pred - target
    mae = np.mean(np.abs(difference), axis=axes).astype(np.float32)
    mse = np.mean(np.square(difference), axis=axes).astype(np.float32)
    return mae, mse


def write_compact_next_frame_artifact(
    path: Path,
    *,
    pred,
    naive,
    targets,
    target_steps,
    frame_times,
    metadata: dict,
    sample_count: int = 16,
) -> dict:
    """Atomically save full scalar errors and sampled next-frame images."""
    path = Path(path)
    pred = np.asarray(pred, dtype=np.float32)
    naive = np.asarray(naive, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    target_steps = np.asarray(target_steps, dtype=np.int64)
    frame_times = np.asarray(frame_times, dtype=np.float32)
    lengths = {
        len(pred),
        len(naive),
        len(targets),
        len(target_steps),
        len(frame_times),
    }
    if len(lengths) != 1 or pred.shape != naive.shape or pred.shape != targets.shape:
        raise ValueError("prediction, reference, index, and time arrays need aligned first dimensions")
    if pred.ndim < 2 or len(pred) == 0:
        raise ValueError("next-frame image arrays must be non-empty")
    if not np.isfinite(pred).all():
        raise ValueError("next-frame predictions must be finite")
    if not np.isfinite(naive).all() or not np.isfinite(targets).all():
        raise ValueError("next-frame references must be finite")
    if not np.isfinite(frame_times).all():
        raise ValueError("next-frame times must be finite")

    offsets = representative_offsets(len(pred), count=sample_count)
    model_mae, model_mse = _per_step_errors(pred, targets)
    naive_mae, naive_mse = _per_step_errors(naive, targets)
    arrays = {
        "artifact_mode": np.asarray(COMPACT_NEXT_FRAME_MODE),
        "sample_offsets": offsets,
        "pred_samples": pred[offsets],
        "naive_samples": naive[offsets],
        "target_samples": targets[offsets],
        "target_steps": target_steps,
        "frame_times": frame_times,
        "per_step_model_mae": model_mae,
        "per_step_naive_mae": naive_mae,
        "per_step_model_mse": model_mse,
        "per_step_naive_mse": naive_mse,
    }
    collisions = set(arrays).intersection(metadata)
    if collisions:
        raise ValueError(f"metadata collides with required artifact fields: {sorted(collisions)}")
    arrays.update({key: np.asarray(value) for key, value in metadata.items()})

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise

    return {
        "artifact_mode": COMPACT_NEXT_FRAME_MODE,
        "sequence_length": len(pred),
        "sample_count": len(offsets),
        "path": str(path),
    }
