#!/usr/bin/env python3
"""Inference-only forcing interventions for a conditioned GRA29 checkpoint."""

from __future__ import annotations

import argparse
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
from models import build_model  # noqa: E402
from protocol_interventions import CONDITIONS  # noqa: E402
from protocol_evaluation_utils import (  # noqa: E402
    aggregate_rows,
    anchored_slice,
    load_sequence,
    response_l1,
    with_intervention,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--arrays_dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--shift_frames", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_eval_windows", type=int, default=3000)
    parser.add_argument("--max_rollout_steps", type=int, default=512)
    parser.add_argument("--save_stride", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(
        args.checkpoint, map_location=device, weights_only=False
    )
    active_fields = list(checkpoint["active_fields"])
    if "voltage" not in active_fields or "current" not in active_fields:
        raise ValueError("checkpoint is not protocol-conditioned")
    context_len = int(checkpoint["context_len"])
    predict_delta = checkpoint["prediction_form"] == "delta_from_last_frame"
    model = build_model(
        model_family=checkpoint["model_family"],
        in_fields=len(active_fields),
        context_len=context_len,
        base_channels=checkpoint["base_channels"],
        hidden_layers=checkpoint["hidden_layers"],
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for stem in sorted(csb.TEST_STEMS):
        base = load_sequence(args.arrays_dir / f"{stem}.npz")
        changes = csb.current_sign_change_indices(
            base.exogenous_all["current"]
        )
        onset = int(changes[0]) if len(changes) else context_len
        anchor_start = max(0, onset - context_len)
        true_predictions = {}

        for condition in CONDITIONS:
            sequence = with_intervention(
                base, condition, seed=args.seed, shift=args.shift_frames
            )
            pred, naive, targets, steps, frame_times = (
                rsb.run_next_frame_eval_for_sequence(
                    model=model,
                    sequence=sequence,
                    active_fields=active_fields,
                    context_len=context_len,
                    window_stride=1,
                    max_windows=args.max_eval_windows,
                    batch_size=args.batch_size,
                    device=device,
                    predict_delta=predict_delta,
                )
            )
            metrics = rsb.summarize_prediction_result(
                sequence=sequence,
                pred=pred,
                naive=naive,
                targets=targets,
                target_steps=steps,
                reversal_radius=5,
            )
            if condition == "true":
                true_predictions["next_frame"] = pred.copy()
            row = {
                "stem": stem,
                "mode": "next_frame",
                "condition": condition,
                "seed": args.seed,
                "shift_frames": args.shift_frames,
                "prediction_change_l1": response_l1(
                    pred, true_predictions["next_frame"]
                ),
                **metrics,
            }
            rows.append(row)
            np.savez_compressed(
                args.output / f"{stem}_next_frame_{condition}_sparse.npz",
                pred=pred[::args.save_stride],
                targets=targets[::args.save_stride],
                target_steps=steps[::args.save_stride],
                frame_times=frame_times[::args.save_stride],
            )

            sliced = anchored_slice(sequence, anchor_start)
            pred, naive, targets, steps, frame_times = (
                rsb.run_rollout_eval_for_sequence(
                    model=model,
                    sequence=sliced,
                    active_fields=active_fields,
                    context_len=context_len,
                    device=device,
                    predict_delta=predict_delta,
                    max_rollout_steps=args.max_rollout_steps,
                )
            )
            metrics = rsb.summarize_prediction_result(
                sequence=sliced,
                pred=pred,
                naive=naive,
                targets=targets,
                target_steps=steps,
                reversal_radius=5,
                include_horizon=True,
            )
            if condition == "true":
                true_predictions["rollout_anchored"] = pred.copy()
            row = {
                "stem": stem,
                "mode": "rollout_anchored",
                "condition": condition,
                "seed": args.seed,
                "shift_frames": args.shift_frames,
                "anchor_frame": anchor_start,
                "onset_frame": onset,
                "prediction_change_l1": response_l1(
                    pred, true_predictions["rollout_anchored"]
                ),
                **metrics,
            }
            rows.append(row)
            np.savez_compressed(
                args.output / f"{stem}_rollout_{condition}_sparse.npz",
                pred=pred[::args.save_stride],
                targets=targets[::args.save_stride],
                target_steps=steps[::args.save_stride],
                frame_times=frame_times[::args.save_stride],
            )
            print(
                f"[done] {stem} {condition} "
                f"nf={rows[-2]['mae_ratio']:.4f} "
                f"rollout={row['mae_ratio']:.4f}",
                flush=True,
            )

    (args.output / "per_particle_results.json").write_text(
        json.dumps(rows, indent=2) + "\n"
    )
    summary = aggregate_rows(rows)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    manifest = {
        "mode": "inference_only_protocol_interventions",
        "checkpoint": args.checkpoint.name,
        "model_family": checkpoint["model_family"],
        "active_fields": active_fields,
        "conditions": list(CONDITIONS),
        "seed": args.seed,
        "shift_frames": args.shift_frames,
        "anchor_rule": "first_true_current_sign_change_minus_context",
        "statistical_unit": "particle_movie",
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print("PROTOCOL INTERVENTIONS DONE")


if __name__ == "__main__":
    main()
