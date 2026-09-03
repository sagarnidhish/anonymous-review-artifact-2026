#!/usr/bin/env python3
"""Evaluate test-record versus paired-training normalization."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "ref"))

import common_sp_baselines as csb  # noqa: E402
import run_sp_baseline_study as rsb  # noqa: E402
from matched_grid_payloads import payload_for_id  # noqa: E402
from models import build_model  # noqa: E402
from protocol_evaluation_utils import first_current_transition_slice  # noqa: E402
from safe_anchored_rollout import (  # noqa: E402
    cumulative_horizon_metrics,
    run_safe_rollout,
)


PRIMARY_PAYLOAD_IDS = (0, 20)
TEST_STEMS = tuple(f"GRA29_C20_45deg_particle{i}" for i in range(1, 5))
NEXT_FRAME_REPRODUCTION_RTOL = 1e-5
ROLLOUT_REPRODUCTION_RTOL = 1e-2
REPRODUCTION_ATOL = 1e-5
DIVERGENT_RATIO_THRESHOLD = 1e3
DIVERGENT_LOG10_TOLERANCE = 0.05


def reproduction_diagnostic(
    current: float,
    original: float,
    *,
    rtol: float = NEXT_FRAME_REPRODUCTION_RTOL,
) -> dict:
    """Compare rerun ratios without treating chaotic divergence as precision.

    The caller sets the relative tolerance: next-frame values use 1e-5, while
    recursive rollouts use 1e-2 to allow bounded cross-GPU accumulation. Once
    both rollouts are already three orders of magnitude worse than persistence,
    the scientifically relevant check is whether they reproduce the same order
    of divergence; exact magnitudes are not stable under hundreds of recursive
    floating-point operations.
    """
    current = float(current)
    original = float(original)
    if not np.isfinite([current, original]).all():
        return {
            "reproduced": False,
            "strictly_close": False,
            "criterion": "nonfinite",
            "absolute_difference": None,
            "relative_difference": None,
            "log10_difference": None,
        }

    absolute_difference = abs(current - original)
    denominator = max(abs(original), np.finfo(np.float64).tiny)
    relative_difference = absolute_difference / denominator
    strictly_close = bool(
        np.isclose(
            current,
            original,
            rtol=rtol,
            atol=REPRODUCTION_ATOL,
        )
    )
    same_divergent_order = False
    log10_difference = None
    if current >= DIVERGENT_RATIO_THRESHOLD and original >= DIVERGENT_RATIO_THRESHOLD:
        log10_difference = abs(np.log10(current) - np.log10(original))
        same_divergent_order = log10_difference <= DIVERGENT_LOG10_TOLERANCE

    if strictly_close:
        criterion = "strict"
    elif same_divergent_order:
        criterion = "divergent_order"
    else:
        criterion = "mismatch"
    return {
        "reproduced": bool(strictly_close or same_divergent_order),
        "strictly_close": strictly_close,
        "criterion": criterion,
        "absolute_difference": float(absolute_difference),
        "relative_difference": float(relative_difference),
        "log10_difference": (
            float(log10_difference) if log10_difference is not None else None
        ),
    }


def summarize_reproduction_diagnostics(diagnostics: list[dict]) -> dict:
    if not diagnostics:
        raise ValueError("at least one reproduction comparison is required")

    finite_abs = [
        row["absolute_difference"]
        for row in diagnostics
        if row["absolute_difference"] is not None
    ]
    finite_rel = [
        row["relative_difference"]
        for row in diagnostics
        if row["relative_difference"] is not None
    ]
    finite_log = [
        row["log10_difference"]
        for row in diagnostics
        if row["log10_difference"] is not None
    ]
    return {
        "all_reproduced": all(row["reproduced"] for row in diagnostics),
        "comparison_count": len(diagnostics),
        "strict_count": sum(row["criterion"] == "strict" for row in diagnostics),
        "divergent_order_count": sum(
            row["criterion"] == "divergent_order" for row in diagnostics
        ),
        "mismatch_count": sum(row["criterion"] == "mismatch" for row in diagnostics),
        "nonfinite_count": sum(row["criterion"] == "nonfinite" for row in diagnostics),
        "max_absolute_difference": max(finite_abs, default=None),
        "max_relative_difference": max(finite_rel, default=None),
        "max_log10_difference_for_divergent_pairs": max(finite_log, default=None),
    }


def paired_training_reference_transform(
    standardized_test: np.ndarray,
    test_mu: float,
    test_std: float,
    train_mu: float,
    train_std: float,
) -> np.ndarray:
    values = np.asarray(standardized_test, dtype=np.float32)
    scalars = np.asarray([test_mu, test_std, train_mu, train_std], dtype=np.float64)
    if not np.isfinite(scalars).all() or test_std <= 0 or train_std <= 0:
        raise ValueError("normalization statistics must be finite with positive scales")
    raw = values * np.float32(test_std) + np.float32(test_mu)
    return ((raw - np.float32(train_mu)) / np.float32(train_std)).astype(np.float32)


def strict_json_dump(path: Path, value) -> None:
    def sanitize(item):
        if isinstance(item, dict):
            return {key: sanitize(val) for key, val in item.items()}
        if isinstance(item, (list, tuple)):
            return [sanitize(val) for val in item]
        if isinstance(item, (float, np.floating)) and not np.isfinite(item):
            return None
        if isinstance(item, np.integer):
            return int(item)
        if isinstance(item, np.floating):
            return float(item)
        return item

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(sanitize(value), handle, indent=2, allow_nan=False)


def load_test_sequence(arrays_dir: Path, stem: str, mode: str):
    particle = int(stem.rsplit("particle", 1)[1])
    train_path = arrays_dir / f"GRA29_C20_25deg_particle{particle}.npz"
    test_path = arrays_dir / f"{stem}.npz"
    with np.load(train_path) as train:
        train_mu = float(train["norm_mu"])
        train_std = float(train["norm_std"])
    with np.load(test_path) as test:
        intensity = test["intensity"].astype(np.float32)
        test_mu = float(test["norm_mu"])
        test_std = float(test["norm_std"])
        exogenous = {
            key: test[key].astype(np.float32)
            for key in ("voltage", "current", "time_norm")
        }
        frame_times = test["frame_times"].astype(np.float32)
    if mode == "paired_training_reference":
        intensity = paired_training_reference_transform(
            intensity, test_mu, test_std, train_mu, train_std
        )
    elif mode != "archived_test_record":
        raise ValueError(f"unknown normalization mode: {mode}")
    sequence = csb.LoadedSequence(
        stem=stem,
        role="test",
        intensity=intensity,
        exogenous_all=exogenous,
        frame_times=frame_times,
        raw_frame_indices=np.arange(len(intensity), dtype=np.int64),
        sequence_subsample_factor=1,
    )
    return sequence, {
        "particle": particle,
        "test_record_mu": test_mu,
        "test_record_std": test_std,
        "paired_25deg_training_mu": train_mu,
        "paired_25deg_training_std": train_std,
    }


def rows_by_stem(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        rows = json.load(handle)
    return {row["stem"]: row for row in rows}


def evaluate_payload(payload_id, arrays_dir, grid_root, device, batch_size):
    payload = payload_for_id(payload_id)
    if payload["input_mode"] != "image_only" or payload["target_mode"] != "delta":
        raise ValueError("normalization control requires image-only delta payloads")
    run_root = grid_root / payload["tag"]
    checkpoint_path = run_root / "models" / f"{payload['model_family']}_best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("payload") != payload:
        raise ValueError(f"checkpoint payload mismatch: {checkpoint_path}")
    model = build_model(
        model_family=payload["model_family"],
        in_fields=1,
        context_len=4,
        base_channels=32,
        hidden_layers=2,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device).eval()
    del checkpoint

    modes = {}
    for mode in ("archived_test_record", "paired_training_reference"):
        next_rows = []
        rollout_rows = []
        stats = {}
        for stem in TEST_STEMS:
            sequence, stem_stats = load_test_sequence(arrays_dir, stem, mode)
            stats[stem] = stem_stats
            pred, naive, targets, target_steps, _ = rsb.run_next_frame_eval_for_sequence(
                model=model,
                sequence=sequence,
                active_fields=["intensity"],
                context_len=4,
                window_stride=1,
                max_windows=3000,
                batch_size=batch_size,
                device=device,
                predict_delta=True,
            )
            next_metric = rsb.summarize_prediction_result(
                sequence=sequence,
                pred=pred,
                naive=naive,
                targets=targets,
                target_steps=target_steps,
                reversal_radius=5,
            )
            next_rows.append({"stem": stem, **next_metric})

            anchor_start, onset, anchored = first_current_transition_slice(
                sequence, context_len=4
            )
            rollout = run_safe_rollout(
                model=model,
                sequence=anchored,
                active_fields=["intensity"],
                context_len=4,
                device=device,
                predict_delta=True,
                max_rollout_steps=512,
            )
            horizons = cumulative_horizon_metrics(
                rollout["pred"],
                rollout["naive"],
                rollout["targets"],
                horizons=payload["report_horizons"],
                first_nonfinite_step=rollout["first_nonfinite_step"],
            )
            rollout_rows.append(
                {
                    "stem": stem,
                    "anchor_frame": anchor_start,
                    "onset_frame": onset,
                    "status": rollout["status"],
                    "first_nonfinite_step": rollout["first_nonfinite_step"],
                    "horizons": horizons,
                }
            )
        modes[mode] = {
            "next_frame": {
                "particle_rows": next_rows,
                "mean_particle_mae_ratio": float(
                    np.mean([row["mae_ratio"] for row in next_rows])
                ),
            },
            "rollout_anchored": {
                "particle_rows": rollout_rows,
                "mean_particle_mae_ratio": {
                    str(horizon): (
                        float(
                            np.mean(
                                [
                                    row["horizons"][str(horizon)]["mae_ratio"]
                                    for row in rollout_rows
                                ]
                            )
                        )
                    )
                    for horizon in payload["report_horizons"]
                },
            },
            "normalization_statistics": stats,
        }

    archived_next = rows_by_stem(run_root / "next_frame" / "next_frame_results.json")
    archived_rollout = rows_by_stem(
        run_root / "rollout_anchored" / "rollout_results.json"
    )
    next_diagnostics = []
    for row in modes["archived_test_record"]["next_frame"]["particle_rows"]:
        diagnostic = reproduction_diagnostic(
            row["mae_ratio"], archived_next[row["stem"]]["mae_ratio"]
        )
        next_diagnostics.append({"stem": row["stem"], **diagnostic})

    rollout_diagnostics = []
    for row in modes["archived_test_record"]["rollout_anchored"]["particle_rows"]:
        saved = archived_rollout[row["stem"]]
        if row["status"] != saved["status"]:
            raise ValueError(f"archived rollout status mismatch for {row['stem']}")
        for horizon in payload["report_horizons"]:
            current = row["horizons"][str(horizon)]["mae_ratio"]
            original = saved["horizons"][str(horizon)]["mae_ratio"]
            diagnostic = reproduction_diagnostic(
                np.nan if current is None else current,
                np.nan if original is None else original,
                rtol=ROLLOUT_REPRODUCTION_RTOL,
            )
            rollout_diagnostics.append(
                {"stem": row["stem"], "horizon": horizon, **diagnostic}
            )

    next_reproduction = summarize_reproduction_diagnostics(next_diagnostics)
    rollout_reproduction = summarize_reproduction_diagnostics(rollout_diagnostics)
    if not next_reproduction["all_reproduced"] or not rollout_reproduction[
        "all_reproduced"
    ]:
        raise RuntimeError(
            "archived evaluation did not reproduce: "
            f"next={next_reproduction}, rollout={rollout_reproduction}"
        )
    del model
    torch.cuda.empty_cache()
    gc.collect()
    return {
        "payload": payload,
        "checkpoint": str(checkpoint_path),
        "archived_reproduction": {
            "next_frame_rtol": NEXT_FRAME_REPRODUCTION_RTOL,
            "rollout_rtol": ROLLOUT_REPRODUCTION_RTOL,
            "strict_atol": REPRODUCTION_ATOL,
            "divergent_ratio_threshold": DIVERGENT_RATIO_THRESHOLD,
            "divergent_log10_tolerance": DIVERGENT_LOG10_TOLERANCE,
            "next_frame_summary": next_reproduction,
            "rollout_summary": rollout_reproduction,
            "next_frame_comparisons": next_diagnostics,
            "rollout_comparisons": rollout_diagnostics,
        },
        "modes": modes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arrays-dir", required=True, type=Path)
    parser.add_argument("--grid-root", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for normalization sensitivity inference")
    output = args.out_root.resolve()
    completion = output / "completion_manifest.json"
    if completion.is_file():
        raise RuntimeError(f"completion manifest already exists: {completion}")
    results = [
        evaluate_payload(
            payload_id,
            args.arrays_dir.resolve(),
            args.grid_root.resolve(),
            torch.device("cuda"),
            args.batch_size,
        )
        for payload_id in PRIMARY_PAYLOAD_IDS
    ]
    artifact = {
        "description": "Paired-training-statistics normalization sensitivity",
        "source_commit": args.source_commit,
        "same_particle_identity_assumption": True,
        "test_information_boundary": (
            "Final 45 C scaling uses only the paired 25 C training movie mean and "
            "standard deviation. Stored 45 C statistics are used only to invert the "
            "archived float16 standardization back to approximate raw intensity."
        ),
        "payload_results": results,
    }
    strict_json_dump(output / "normalization_sensitivity.json", artifact)
    strict_json_dump(
        completion,
        {
            "status": "complete",
            "source_commit": args.source_commit,
            "payload_ids": list(PRIMARY_PAYLOAD_IDS),
            "artifact": "normalization_sensitivity.json",
            "archived_evaluation_reproduced": True,
        },
    )
    print("NORMALIZATION SENSITIVITY COMPLETE", flush=True)


if __name__ == "__main__":
    main()
