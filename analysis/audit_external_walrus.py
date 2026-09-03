#!/usr/bin/env python3
"""Fail-closed audit of the recovered external Alice/WALRUS stress test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load_rows(paths) -> list[dict]:
    rows = []
    for path in paths:
        payload = json.loads(path.read_text())
        if not isinstance(payload, list):
            raise ValueError(f"expected list in {path}")
        rows.extend(payload)
    return rows


def audit_archive(archive_path: Path | str) -> dict:
    archive = Path(archive_path)
    result_root = archive / "results" / "walrus"
    script_path = archive / "walrus_native_alice.py"
    if not result_root.is_dir() or not script_path.is_file():
        raise FileNotFoundError("external archive lacks results or evaluator")

    rollout_rows = _load_rows(sorted(result_root.glob("*rollout*results.json")))
    next_frame_rows = _load_rows(
        sorted(result_root.glob("*next_frame*results.json"))
    )
    if not rollout_rows:
        raise ValueError("no external rollout result rows")

    script = script_path.read_text()
    script_declares_fixed = (
        "naive_next = naive_ctx[-1].copy()" in script
        and "naive context frozen" in script
    )

    artifact_checks = []
    fixed_arrays = True
    metrics_recomputed = True
    rollout_npzs = sorted(result_root.glob("rollout*.npz"))
    if not rollout_npzs:
        raise ValueError("no external rollout NPZ artifacts")
    for path in rollout_npzs:
        with np.load(path) as loaded:
            pred = loaded["pred"]
            naive = loaded["naive"]
            targets = loaded["targets"]
            row = json.loads(str(loaded["row_json"]))
        naive_is_fixed = bool(
            len(naive) > 0
            and np.array_equal(
                naive,
                np.repeat(naive[0][None, ...], len(naive), axis=0),
            )
        )
        model_mse = float(np.mean((pred - targets) ** 2))
        naive_mse = float(np.mean((naive - targets) ** 2))
        ratio = model_mse / max(naive_mse, 1e-12)
        numeric_match = bool(
            np.isclose(model_mse, row["model_mse"], rtol=1e-6, atol=1e-9)
            and np.isclose(naive_mse, row["naive_mse"], rtol=1e-6, atol=1e-9)
            and np.isclose(ratio, row["ratio"], rtol=1e-6, atol=1e-9)
        )
        fixed_arrays = fixed_arrays and naive_is_fixed
        metrics_recomputed = metrics_recomputed and numeric_match
        artifact_checks.append({
            "artifact": path.name,
            "stem": row["stem"],
            "n_steps": int(len(pred)),
            "fixed_persistence": naive_is_fixed,
            "metrics_recomputed": numeric_match,
            "model_mse": model_mse,
            "naive_mse": naive_mse,
            "mse_ratio": ratio,
        })

    fixed_verified = bool(script_declares_fixed and fixed_arrays)
    mse_better = all(float(row["ratio"]) < 1.0 for row in rollout_rows)
    ssim_worse = all(
        "ssim_model" in row
        and "ssim_naive" in row
        and float(row["ssim_model"]) < float(row["ssim_naive"])
        for row in rollout_rows
    )
    if mse_better and ssim_worse:
        metric_direction = "mse_better_ssim_worse"
    elif mse_better:
        metric_direction = "mse_better_ssim_not_worse"
    else:
        metric_direction = "mse_not_better"

    passed = fixed_verified and metrics_recomputed
    independent_movies = len({row["stem"] for row in rollout_rows})
    return {
        "schema_version": 1,
        "status": "PASS_WITH_LIMITATIONS" if passed else "FAIL",
        "dataset": "Alice Joule-paper iSCAT external stress test",
        "fixed_persistence_verified": fixed_verified,
        "metrics_recomputed": metrics_recomputed,
        "appendix_admissible": passed,
        "main_benchmark_admissible": False,
        "independent_rollout_movies": independent_movies,
        "metric_direction": metric_direction,
        "rollout_rows": rollout_rows,
        "next_frame_rows": next_frame_rows,
        "artifact_checks": artifact_checks,
        "limitations": [
            "One external rollout movie and one rollout origin.",
            "Rollout begins after the first four frames, not a protocol event.",
            "No model adaptation was performed; train/test labels describe file layout.",
            "Alice and GRA29 preprocessing and acquisition domains are not pooled.",
            "MSE and SSIM rank model versus persistence in opposite directions.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    audit = audit_archive(args.archive)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2) + "\n")
    print(f"wrote {args.output}: {audit['status']}")


if __name__ == "__main__":
    main()
