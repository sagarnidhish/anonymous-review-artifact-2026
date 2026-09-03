#!/usr/bin/env python3
"""Repair WALRUS rollout artifacts with a fixed observed-frame baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from analysis.evaluation_invariants import fixed_persistence
except ModuleNotFoundError:  # direct script execution
    from evaluation_invariants import fixed_persistence


def repair_artifact(
    source_path: Path | str,
    destination_path: Path | str,
    sequence_frames: np.ndarray,
) -> dict:
    source = Path(source_path).resolve()
    destination = Path(destination_path).resolve()
    if source == destination:
        raise ValueError("refusing to overwrite source artifact")

    with np.load(source) as loaded:
        arrays = {key: loaded[key].copy() for key in loaded.files}
    required = {"pred", "naive", "targets", "target_steps", "frame_times"}
    missing = sorted(required - arrays.keys())
    if missing:
        raise ValueError(f"missing required arrays: {missing}")

    steps = np.asarray(arrays["target_steps"], dtype=np.int64)
    if len(steps) == 0 or steps[0] < 1:
        raise ValueError("first target step must follow an observed frame")
    if steps[0] > len(sequence_frames):
        raise ValueError("target step exceeds source sequence")
    source_step = int(steps[0] - 1)
    corrected_naive = fixed_persistence(
        np.asarray(sequence_frames)[source_step], len(steps)
    )
    if corrected_naive.shape != arrays["targets"].shape:
        raise ValueError("corrected persistence shape differs from targets")

    previous_naive = arrays["naive"]
    arrays["naive"] = corrected_naive.astype(
        previous_naive.dtype, copy=False
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)

    pred_mae = float(np.abs(arrays["pred"] - arrays["targets"]).mean())
    old_naive_mae = float(
        np.abs(previous_naive - arrays["targets"]).mean()
    )
    corrected_naive_mae = float(
        np.abs(corrected_naive - arrays["targets"]).mean()
    )
    return {
        "source": str(source),
        "destination": str(destination),
        "persistence_source_step": source_step,
        "n_steps": int(len(steps)),
        "naive_changed": not np.array_equal(previous_naive, corrected_naive),
        "model_mae": pred_mae,
        "old_naive_mae": old_naive_mae,
        "corrected_naive_mae": corrected_naive_mae,
        "old_mae_ratio": pred_mae / max(old_naive_mae, 1e-12),
        "corrected_mae_ratio": (
            pred_mae / max(corrected_naive_mae, 1e-12)
        ),
    }


def repair_tree(
    source_root: Path | str,
    destination_root: Path | str,
    arrays_dir: Path | str,
) -> dict:
    source_root = Path(source_root)
    destination_root = Path(destination_root)
    arrays_dir = Path(arrays_dir)
    artifact_summaries = []
    for source in sorted(source_root.glob("*/rollout_*.npz")):
        stem = source.stem.removeprefix("rollout_")
        sequence_path = arrays_dir / f"{stem}.npz"
        if not sequence_path.is_file():
            raise FileNotFoundError(sequence_path)
        with np.load(sequence_path) as loaded:
            sequence = loaded["intensity"].astype(np.float32)
        destination = destination_root / source.parent.name / source.name
        summary = repair_artifact(source, destination, sequence)
        summary["stem"] = stem
        artifact_summaries.append(summary)
    if not artifact_summaries:
        raise ValueError(f"no rollout artifacts under {source_root}")

    aggregate = {
        "count": len(artifact_summaries),
        "mean_model_mae": float(np.mean([
            row["model_mae"] for row in artifact_summaries
        ])),
        "mean_naive_mae": float(np.mean([
            row["corrected_naive_mae"] for row in artifact_summaries
        ])),
        "mean_mae_ratio": float(np.mean([
            row["corrected_mae_ratio"] for row in artifact_summaries
        ])),
        "baseline": "fixed_last_observed_frame",
        "source_artifacts_preserved": True,
    }
    destination_root.mkdir(parents=True, exist_ok=True)
    (destination_root / "repair_summary.json").write_text(
        json.dumps(artifact_summaries, indent=2) + "\n"
    )
    (destination_root / "comparison_summary.json").write_text(
        json.dumps([aggregate], indent=2) + "\n"
    )
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--sequence", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--destination-root", type=Path)
    parser.add_argument("--arrays-dir", type=Path)
    args = parser.parse_args()

    if args.source_root:
        if not args.destination_root or not args.arrays_dir:
            parser.error("tree mode requires --destination-root and --arrays-dir")
        summary = repair_tree(
            args.source_root, args.destination_root, args.arrays_dir
        )
        print(
            f"repaired {summary['count']} artifacts; "
            f"mean ratio={summary['mean_mae_ratio']:.4f}"
        )
        return
    if not all((args.source, args.destination, args.sequence, args.summary)):
        parser.error(
            "single mode requires --source --destination --sequence --summary"
        )
    with np.load(args.sequence) as loaded:
        sequence = loaded["intensity"].astype(np.float32)
    summary = repair_artifact(args.source, args.destination, sequence)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {args.destination} and {args.summary}")


if __name__ == "__main__":
    main()
