#!/usr/bin/env python3
"""Fail-closed validator for the fresh 24-configuration benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train.matched_grid_payloads import build_payloads  # noqa: E402


TEST_STEMS = {f"GRA29_C20_45deg_particle{i}" for i in range(1, 5)}


def read_json(path: Path):
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"missing required JSON: {path}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _check_rows(rows, payload: dict, kind: str) -> None:
    if not isinstance(rows, list) or len(rows) != 4:
        raise ValueError(f"{kind} must contain four particle rows")
    stems = {row.get("stem") for row in rows}
    if stems != TEST_STEMS:
        raise ValueError(f"{kind} test particle set is incomplete: {sorted(stems)}")
    if any(row.get("payload_id") != payload["payload_id"] for row in rows):
        raise ValueError(f"{kind} payload ID mismatch")


def _close(actual, expected) -> bool:
    return bool(np.isclose(actual, expected, rtol=1e-5, atol=1e-6))


def _expected_frame_count(row: dict) -> int:
    """Return the full-sequence frame count recorded by compute_frame_metrics."""
    value = row.get("N")
    if not isinstance(value, (int, np.integer)) or int(value) <= 0:
        raise ValueError(f"invalid next-frame N for {row.get('stem')}: {value}")
    return int(value)


def _validate_next_frame_artifact(
    artifact: Path, row: dict, expected_mode: str
) -> None:
    if not artifact.is_file() or artifact.stat().st_size == 0:
        raise ValueError(f"missing next-frame artifact: {artifact}")
    with np.load(artifact) as data:
        mode = data["artifact_mode"].item() if "artifact_mode" in data else "full_v1"
        if mode != expected_mode:
            raise ValueError(
                f"next-frame artifact mode mismatch for {row['stem']}: {mode}"
            )
        if mode == "compact_samples_v1":
            required = {
                "sample_offsets",
                "pred_samples",
                "naive_samples",
                "target_samples",
                "target_steps",
                "frame_times",
                "per_step_model_mae",
                "per_step_naive_mae",
                "per_step_model_mse",
                "per_step_naive_mse",
            }
            if not required.issubset(data.files):
                raise ValueError(
                    f"compact next-frame artifact fields incomplete for {row['stem']}"
                )
            offsets = data["sample_offsets"]
            pred_samples = data["pred_samples"]
            naive_samples = data["naive_samples"]
            target_samples = data["target_samples"]
            target_steps = data["target_steps"]
            frame_times = data["frame_times"]
            model_mae = data["per_step_model_mae"]
            naive_mae = data["per_step_naive_mae"]
            model_mse = data["per_step_model_mse"]
            naive_mse = data["per_step_naive_mse"]
            lengths = {
                len(target_steps),
                len(frame_times),
                len(model_mae),
                len(naive_mae),
                len(model_mse),
                len(naive_mse),
            }
            if lengths != {_expected_frame_count(row)}:
                raise ValueError(f"compact next-frame sequence length mismatch for {row['stem']}")
            if offsets.ndim != 1 or len(offsets) == 0:
                raise ValueError(f"missing compact sample offsets for {row['stem']}")
            if not np.array_equal(offsets, np.unique(offsets)):
                raise ValueError(f"compact sample offsets are not unique for {row['stem']}")
            if int(offsets[0]) != 0 or int(offsets[-1]) != len(target_steps) - 1:
                raise ValueError(f"compact samples omit an endpoint for {row['stem']}")
            if pred_samples.shape != naive_samples.shape or pred_samples.shape != target_samples.shape:
                raise ValueError(f"compact sample images do not align for {row['stem']}")
            if len(pred_samples) != len(offsets):
                raise ValueError(f"compact sample count mismatch for {row['stem']}")
            if not np.isfinite(pred_samples).all():
                raise ValueError(f"non-finite next-frame sample for {row['stem']}")
            if not np.isfinite(naive_samples).all() or not np.isfinite(target_samples).all():
                raise ValueError(f"non-finite next-frame reference sample for {row['stem']}")
            if not np.isfinite(frame_times).all():
                raise ValueError(f"non-finite next-frame times for {row['stem']}")
            if len(target_steps) > 1 and not np.all(np.diff(target_steps) > 0):
                raise ValueError(f"next-frame target steps are not increasing for {row['stem']}")
            image_axes = tuple(range(1, pred_samples.ndim))
            sampled_model_mae = np.mean(
                np.abs(pred_samples - target_samples), axis=image_axes
            )
            sampled_naive_mae = np.mean(
                np.abs(naive_samples - target_samples), axis=image_axes
            )
            if not np.allclose(
                sampled_model_mae, model_mae[offsets], rtol=1e-5, atol=1e-6
            ):
                raise ValueError(f"sampled model MAE mismatch for {row['stem']}")
            if not np.allclose(
                sampled_naive_mae, naive_mae[offsets], rtol=1e-5, atol=1e-6
            ):
                raise ValueError(f"sampled persistence MAE mismatch for {row['stem']}")
        elif mode == "full_v1":
            required = {"pred", "naive", "targets", "target_steps", "frame_times"}
            if not required.issubset(data.files):
                raise ValueError(f"full next-frame artifact fields incomplete for {row['stem']}")
            pred = data["pred"]
            naive = data["naive"]
            target = data["targets"]
            if pred.shape != naive.shape or pred.shape != target.shape:
                raise ValueError(f"full next-frame arrays do not align for {row['stem']}")
            lengths = {len(pred), len(data["target_steps"]), len(data["frame_times"])}
            if lengths != {_expected_frame_count(row)}:
                raise ValueError(f"full next-frame sequence length mismatch for {row['stem']}")
            if not np.isfinite(pred).all() or not np.isfinite(naive).all() or not np.isfinite(target).all():
                raise ValueError(f"non-finite full next-frame array for {row['stem']}")
            image_axes = tuple(range(1, pred.ndim))
            model_mae = np.mean(np.abs(pred - target), axis=image_axes)
            naive_mae = np.mean(np.abs(naive - target), axis=image_axes)
            model_mse = np.mean(np.square(pred - target), axis=image_axes)
            naive_mse = np.mean(np.square(naive - target), axis=image_axes)
        else:
            raise ValueError(f"unknown next-frame artifact mode for {row['stem']}: {mode}")

        aggregates = {
            "model_mae": float(np.mean(model_mae)),
            "naive_mae": float(np.mean(naive_mae)),
            "model_mse": float(np.mean(model_mse)),
            "naive_mse": float(np.mean(naive_mse)),
        }
        labels = {
            "model_mae": "aggregate model MAE",
            "naive_mae": "aggregate persistence MAE",
            "model_mse": "aggregate model MSE",
            "naive_mse": "aggregate persistence MSE",
        }
        for key, actual in aggregates.items():
            if not _close(actual, row.get(key, np.nan)):
                raise ValueError(f"{labels[key]} mismatch for {row['stem']}")
        if not _close(aggregates["model_mae"] / aggregates["naive_mae"], row.get("mae_ratio", np.nan)):
            raise ValueError(f"aggregate MAE ratio mismatch for {row['stem']}")
        if not _close(aggregates["model_mse"] / aggregates["naive_mse"], row.get("mse_ratio", np.nan)):
            raise ValueError(f"aggregate MSE ratio mismatch for {row['stem']}")


def validate_payload_directory(run_dir: Path, payload: dict) -> dict:
    run_dir = Path(run_dir)
    manifest = read_json(run_dir / "completion_manifest.json")
    if manifest.get("status") != "complete":
        raise ValueError(f"payload is not complete: {run_dir}")
    if manifest.get("payload") != payload:
        raise ValueError(f"payload manifest mismatch: {run_dir}")
    checkpoint = run_dir / manifest.get("checkpoint", "")
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise ValueError(f"missing checkpoint: {checkpoint}")

    next_rows = read_json(run_dir / "next_frame" / "next_frame_results.json")
    rollout_rows = read_json(
        run_dir / "rollout_anchored" / "rollout_results.json"
    )
    _check_rows(next_rows, payload, "next-frame results")
    _check_rows(rollout_rows, payload, "rollout results")
    next_mode = manifest.get("next_frame_artifact_mode", "full_v1")
    for row in next_rows:
        if not np.isfinite(row.get("mae_ratio", np.nan)):
            raise ValueError(f"non-finite next-frame ratio for {row.get('stem')}")
        stem = row["stem"]
        _validate_next_frame_artifact(
            run_dir / "next_frame" / stem / f"preds_{stem}.npz",
            row,
            next_mode,
        )

    for row in rollout_rows:
        stem = row["stem"]
        if row.get("anchor_rule") != payload["anchor_rule"]:
            raise ValueError(f"anchor rule mismatch for {stem}")
        if row.get("rollout_steps") != payload["rollout_steps"]:
            raise ValueError(f"rollout length mismatch for {stem}")
        if row.get("onset_frame") - row.get("anchor_frame") != payload["context_len"]:
            raise ValueError(f"anchor/context mismatch for {stem}")
        horizons = row.get("horizons", {})
        expected_horizons = {str(value) for value in payload["report_horizons"]}
        if set(horizons) != expected_horizons:
            raise ValueError(f"horizon set mismatch for {stem}")
        status = row.get("status")
        first_nonfinite = row.get("first_nonfinite_step")
        if status == "complete" and first_nonfinite is not None:
            raise ValueError(f"complete rollout has a divergence step for {stem}")
        if status == "numerically_diverged" and not (
            isinstance(first_nonfinite, int) and 1 <= first_nonfinite <= 512
        ):
            raise ValueError(f"invalid divergence step for {stem}")
        if status not in {"complete", "numerically_diverged"}:
            raise ValueError(f"unknown rollout status for {stem}: {status}")

        artifact = (
            run_dir / "rollout_anchored" / stem / f"rollout_{stem}.npz"
        )
        if not artifact.is_file() or artifact.stat().st_size == 0:
            raise ValueError(f"missing rollout artifact: {artifact}")
        with np.load(artifact) as data:
            required = {
                "pred",
                "naive",
                "targets",
                "target_steps",
                "frame_times",
                "anchor_frame",
                "onset_frame",
                "first_nonfinite_step",
            }
            if not required.issubset(data.files):
                raise ValueError(f"rollout artifact fields incomplete for {stem}")
            pred = data["pred"]
            naive = data["naive"]
            targets = data["targets"]
            target_steps = data["target_steps"]
            if pred.shape != naive.shape or pred.shape != targets.shape:
                raise ValueError(f"rollout arrays do not align for {stem}")
            if len(pred) != payload["rollout_steps"]:
                raise ValueError(f"rollout array length mismatch for {stem}")
            if not np.isfinite(naive).all() or not np.isfinite(targets).all():
                raise ValueError(f"non-finite reference arrays for {stem}")
            if not np.array_equal(naive, np.repeat(naive[:1], len(naive), axis=0)):
                raise ValueError(f"fixed persistence invariant failed for {stem}")
            if not np.array_equal(np.diff(target_steps), np.ones(len(pred) - 1)):
                raise ValueError(f"non-consecutive target steps for {stem}")
            if int(target_steps[0]) != int(row["onset_frame"]):
                raise ValueError(f"first target is not the current transition for {stem}")
            if int(data["anchor_frame"]) != int(row["anchor_frame"]):
                raise ValueError(f"artifact anchor mismatch for {stem}")
            if int(data["onset_frame"]) != int(row["onset_frame"]):
                raise ValueError(f"artifact onset mismatch for {stem}")
            if status == "complete":
                if not np.isfinite(pred).all():
                    raise ValueError(f"complete rollout contains non-finite values for {stem}")
            else:
                offset = int(first_nonfinite) - 1
                if not np.isfinite(pred[:offset]).all() or not np.isnan(pred[offset:]).all():
                    raise ValueError(f"divergence encoding is inconsistent for {stem}")

    return {
        "status": "complete",
        "payload_id": payload["payload_id"],
        "tag": payload["tag"],
        "next_frame_particle_count": 4,
        "rollout_particle_count": 4,
        "diverged_particle_count": sum(
            row["status"] == "numerically_diverged" for row in rollout_rows
        ),
    }


def validate_grid(root: Path) -> dict:
    root = Path(root)
    summaries = []
    for payload in build_payloads():
        run_dir = root / payload["tag"]
        if not run_dir.is_dir():
            raise ValueError(f"missing payload directory: {run_dir}")
        summaries.append(validate_payload_directory(run_dir, payload))
    return {
        "status": "complete",
        "payload_count": len(summaries),
        "particle_rollout_count": sum(
            row["rollout_particle_count"] for row in summaries
        ),
        "diverged_particle_count": sum(
            row["diverged_particle_count"] for row in summaries
        ),
        "payloads": summaries,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--payload-id", type=int)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    if args.payload_id is None:
        summary = validate_grid(Path(args.root))
    else:
        payloads = build_payloads()
        if args.payload_id < 0 or args.payload_id >= len(payloads):
            parser.error(f"--payload-id must be in 0..{len(payloads) - 1}")
        payload = payloads[args.payload_id]
        summary = validate_payload_directory(
            Path(args.root) / payload["tag"], payload
        )
    rendered = json.dumps(summary, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
