#!/usr/bin/env python3
"""Frozen-checkpoint protocol delay sweep with measured transition windows."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "ref"))

import common_sp_baselines as csb  # noqa: E402
import run_sp_baseline_study as rsb  # noqa: E402
from analysis.protocol_timing_diagnostics import (  # noqa: E402
    channel_perturbation,
    measured_transition_mask,
    model_facing_channel_windows,
)
from models import build_model  # noqa: E402
from protocol_evaluation_utils import (  # noqa: E402
    delay_condition_specs,
    fixed_transition_summaries,
    load_sequence,
    response_l1,
    with_intervention,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _aggregate(rows: list[dict], labels: list[str]) -> dict:
    stems = sorted({row["stem"] for row in rows})
    output = {}
    measured = {
        row["stem"]: row
        for row in rows
        if row["condition"] == "measured"
    }
    for label in labels:
        selected = [row for row in rows if row["condition"] == label]
        if len(selected) != len(stems):
            raise ValueError(f"incomplete delay-sweep condition: {label}")
        output[label] = {
            "n_particle_movies": len(selected),
            "mean_mae_ratio": float(
                np.mean([row["mae_ratio"] for row in selected])
            ),
            "particle_mae_ratio": [row["mae_ratio"] for row in selected],
            "mean_paired_delta_mae_ratio": float(
                np.mean(
                    [
                        row["mae_ratio"] - measured[row["stem"]]["mae_ratio"]
                        for row in selected
                    ]
                )
            ),
            "mean_prediction_change_l1": float(
                np.mean([row["prediction_change_l1"] for row in selected])
            ),
            "mean_transition_mae_ratio": float(
                np.mean(
                    [row["fixed_transition"]["transition"]["mae_ratio"] for row in selected]
                )
            ),
            "transition_window_count_per_movie": [
                row["fixed_transition"]["transition"]["N"] for row in selected
            ],
            "input_perturbation": {
                channel: {
                    key: float(
                        np.mean(
                            [
                                row["input_perturbation"][channel][key]
                                for row in selected
                            ]
                        )
                    )
                    for key in (
                        "mean_abs_difference",
                        "measured_sigma",
                        "difference_over_sigma",
                        "correlation",
                    )
                }
                for channel in ("voltage", "current")
            },
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--arrays-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--delays", default="0,16,32,64,128,256,512")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-eval-windows", type=int, default=3000)
    parser.add_argument("--save-stride", type=int, default=64)
    parser.add_argument("--transition-radius", type=int, default=5)
    args = parser.parse_args()

    delays = tuple(int(value) for value in args.delays.split(","))
    specs = delay_condition_specs(delays)
    labels = [spec["label"] for spec in specs]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(
        args.checkpoint, map_location=device, weights_only=False
    )
    active_fields = list(checkpoint["active_fields"])
    if "voltage" not in active_fields or "current" not in active_fields:
        raise ValueError("checkpoint is not voltage/current conditioned")
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
    rows: list[dict] = []
    for stem in sorted(csb.TEST_STEMS):
        base = load_sequence(args.arrays_dir / f"{stem}.npz")
        measured_windows = {
            channel: model_facing_channel_windows(
                base.exogenous_all[channel],
                context_len=context_len,
                max_windows=args.max_eval_windows,
            )
            for channel in ("voltage", "current")
        }
        reference = None
        for spec in specs:
            sequence = with_intervention(
                base,
                spec["intervention"],
                seed=args.seed,
                shift=spec["shift_frames"],
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
            if reference is None:
                reference = {
                    "pred": pred.copy(),
                    "naive": naive.copy(),
                    "targets": targets.copy(),
                    "steps": steps.copy(),
                }
            else:
                if not np.array_equal(steps, reference["steps"]):
                    raise ValueError(f"target-step mismatch for {stem}/{spec['label']}")
                if not np.allclose(targets, reference["targets"], atol=0, rtol=0):
                    raise ValueError(f"target mismatch for {stem}/{spec['label']}")
                if not np.allclose(naive, reference["naive"], atol=0, rtol=0):
                    raise ValueError(f"persistence mismatch for {stem}/{spec['label']}")
            metrics = csb.compute_frame_metrics(pred, naive, targets)
            fixed = fixed_transition_summaries(
                pred,
                naive,
                targets,
                measured_current=base.exogenous_all["current"],
                target_steps=steps,
                radius=args.transition_radius,
            )
            input_perturbation = {
                channel: channel_perturbation(
                    measured_windows[channel],
                    model_facing_channel_windows(
                        sequence.exogenous_all[channel],
                        context_len=context_len,
                        max_windows=args.max_eval_windows,
                    ),
                )
                for channel in ("voltage", "current")
            }
            row = {
                "stem": stem,
                "condition": spec["label"],
                "intervention": spec["intervention"],
                "shift_frames": spec["shift_frames"],
                "seed": args.seed,
                "prediction_change_l1": response_l1(
                    pred, reference["pred"]
                ),
                "input_perturbation": input_perturbation,
                "fixed_transition": fixed,
                **metrics,
            }
            rows.append(row)
            selector = measured_transition_mask(
                base.exogenous_all["current"],
                steps,
                radius=args.transition_radius,
            )
            np.savez_compressed(
                args.output / f"{stem}_{spec['label']}_evidence.npz",
                sparse_pred=pred[:: args.save_stride],
                sparse_targets=targets[:: args.save_stride],
                sparse_target_steps=steps[:: args.save_stride],
                sparse_frame_times=frame_times[:: args.save_stride],
                transition_pred=pred[selector],
                transition_naive=naive[selector],
                transition_targets=targets[selector],
                transition_target_steps=steps[selector],
            )
            print(
                f"[done] {stem} {spec['label']} ratio={row['mae_ratio']:.4f} "
                f"transition={fixed['transition']['mae_ratio']:.4f}",
                flush=True,
            )
        del reference

    summary = {
        "description": "Frozen ConvLSTM protocol-delay sweep",
        "protocol_unit": "one shared cell-level trace",
        "optical_aggregation_unit": "particle_movie",
        "conditions": labels,
        "results": _aggregate(rows, labels),
    }
    (args.output / "per_particle_results.json").write_text(
        json.dumps(rows, indent=2) + "\n"
    )
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    manifest = {
        "mode": "inference_only_protocol_delay_sweep",
        "checkpoint": args.checkpoint.name,
        "checkpoint_sha256": _sha256(args.checkpoint),
        "model_family": checkpoint["model_family"],
        "active_fields": active_fields,
        "prediction_form": checkpoint["prediction_form"],
        "context_len": context_len,
        "conditions": specs,
        "seed": args.seed,
        "max_eval_windows": args.max_eval_windows,
        "transition_radius": args.transition_radius,
        "transition_definition": "absolute indices from original measured current",
        "device": str(device),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print("PROTOCOL DELAY SWEEP DONE")


if __name__ == "__main__":
    main()

