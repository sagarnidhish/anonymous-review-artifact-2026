#!/usr/bin/env python3
"""Convert completed matched-grid next-frame artifacts to compact evidence."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.validate_matched_grid import (  # noqa: E402
    read_json,
    validate_payload_directory,
)
from train.matched_grid_artifacts import (  # noqa: E402
    COMPACT_NEXT_FRAME_MODE,
    write_compact_next_frame_artifact,
)
from train.matched_grid_payloads import payload_for_id  # noqa: E402


def _atomic_json_write(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def convert_payload(run_dir: Path, payload: dict, sample_count: int = 16) -> dict:
    """Convert one already-complete legacy payload and revalidate it."""
    run_dir = Path(run_dir)
    validate_payload_directory(run_dir, payload)
    manifest_path = run_dir / "completion_manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("next_frame_artifact_mode", "full_v1") != "full_v1":
        raise ValueError(f"payload is not a legacy full-v1 run: {run_dir}")
    rows = read_json(run_dir / "next_frame" / "next_frame_results.json")

    for row in rows:
        stem = row["stem"]
        artifact = run_dir / "next_frame" / stem / f"preds_{stem}.npz"
        with np.load(artifact) as data:
            required = {"pred", "naive", "targets", "target_steps", "frame_times"}
            if not required.issubset(data.files):
                raise ValueError(f"legacy next-frame fields incomplete for {stem}")
            pred = data["pred"].copy()
            naive = data["naive"].copy()
            targets = data["targets"].copy()
            target_steps = data["target_steps"].copy()
            frame_times = data["frame_times"].copy()
        write_compact_next_frame_artifact(
            artifact,
            pred=pred,
            naive=naive,
            targets=targets,
            target_steps=target_steps,
            frame_times=frame_times,
            metadata={
                "active_fields": np.asarray(row.get("active_fields", [])),
                "model_family": row.get("model_family", payload["model_family"]),
                "tag": row.get("tag", payload["tag"]),
                "prediction_form": row.get("prediction_form", "unknown"),
                "model_mae": np.asarray(row["model_mae"], dtype=np.float32),
                "naive_mae": np.asarray(row["naive_mae"], dtype=np.float32),
                "mae_ratio": np.asarray(row["mae_ratio"], dtype=np.float32),
                "model_mse": np.asarray(row["model_mse"], dtype=np.float32),
                "naive_mse": np.asarray(row["naive_mse"], dtype=np.float32),
                "mse_ratio": np.asarray(row["mse_ratio"], dtype=np.float32),
            },
            sample_count=sample_count,
        )
        study_path = run_dir / "next_frame" / stem / "study_manifest.json"
        study = read_json(study_path)
        study["artifact_mode"] = COMPACT_NEXT_FRAME_MODE
        study["evaluation_scope"] = "full_frame_metrics_with_sampled_images"
        _atomic_json_write(study_path, study)

    manifest["next_frame_artifact_mode"] = COMPACT_NEXT_FRAME_MODE
    manifest["next_frame_sample_count"] = int(sample_count)
    _atomic_json_write(manifest_path, manifest)
    validation = validate_payload_directory(run_dir, payload)
    return {
        **validation,
        "artifact_mode": COMPACT_NEXT_FRAME_MODE,
        "sample_count": int(sample_count),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--payload-id", type=int, required=True)
    parser.add_argument("--sample-count", type=int, default=16)
    args = parser.parse_args()
    payload = payload_for_id(args.payload_id)
    summary = convert_payload(
        Path(args.root) / payload["tag"], payload, sample_count=args.sample_count
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
