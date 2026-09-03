#!/usr/bin/env python3
"""Evaluation-only entry point for completed GRA29 checkpoints."""

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
from particle_splits import build_identity_holdout_fold  # noqa: E402
from protocol_evaluation_utils import (  # noqa: E402
    ROLLOUT_ANCHOR_RULE,
    first_current_transition_slice,
)
from result_metadata import build_result_row, rows_for_role  # noqa: E402


def split_stems(
    split: str, heldout_particle: int | None = None
) -> tuple[set[str], set[str]]:
    if split == "frozen":
        return set(csb.TRAIN_STEMS), set(csb.TEST_STEMS)
    if split == "lopo":
        train = {
            f"GRA29_C20_{temperature}_particle{particle}"
            for temperature in ("25deg", "45deg")
            for particle in (1, 2, 3)
        }
        test = {
            f"GRA29_C20_{temperature}_particle4"
            for temperature in ("25deg", "45deg")
        }
        return train, test
    if split == "identity_holdout":
        if heldout_particle is None:
            raise ValueError("heldout_particle is required for identity_holdout")
        fold = build_identity_holdout_fold(heldout_particle)
        return set(fold.train_stems), set(fold.all_test_stems)
    raise ValueError(f"unknown split: {split}")


def load_npz_sequence(path: Path, role: str):
    stem = path.stem
    with np.load(path) as loaded:
        intensity = loaded["intensity"].astype(np.float32)
        exogenous = {
            key: loaded[key]
            for key in ("voltage", "current", "time_norm")
        }
        frame_times = loaded["frame_times"]
    indices = np.arange(len(intensity), dtype=np.int64)
    return csb.LoadedSequence(
        stem=stem,
        role=role,
        intensity=intensity,
        exogenous_all=exogenous,
        frame_times=frame_times[indices],
        raw_frame_indices=indices,
        sequence_subsample_factor=1,
    )


def dump_mode_summaries(directory: Path, rows: list[dict], detail_name: str):
    rsb.dump_summary(str(directory / detail_name), rows)
    rsb.dump_summary(
        str(directory / "comparison_summary_all.json"),
        [rsb.summarize_rows(rows)],
    )
    rsb.dump_summary(
        str(directory / "comparison_summary.json"),
        [rsb.summarize_rows(rows_for_role(rows, "test"))],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--tag", required=True)
    parser.add_argument(
        "--split",
        choices=("frozen", "lopo", "identity_holdout"),
        required=True,
    )
    parser.add_argument("--heldout_particle", type=int, choices=(1, 2, 3, 4))
    parser.add_argument("--arrays_dir", required=True, type=Path)
    parser.add_argument("--out_root", required=True, type=Path)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_eval_windows", type=int, default=3000)
    parser.add_argument("--max_rollout_steps", type=int, default=512)
    parser.add_argument(
        "--include_train",
        action="store_true",
        help="also save training-particle diagnostics; summaries stay test-only",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(
        args.checkpoint, map_location=device, weights_only=False
    )
    model = build_model(
        model_family=checkpoint["model_family"],
        in_fields=len(checkpoint["active_fields"]),
        context_len=checkpoint["context_len"],
        base_channels=checkpoint["base_channels"],
        hidden_layers=checkpoint["hidden_layers"],
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    train_stems, test_stems = split_stems(
        args.split, heldout_particle=args.heldout_particle
    )
    identity_fold = (
        build_identity_holdout_fold(args.heldout_particle)
        if args.split == "identity_holdout"
        else None
    )
    selected = test_stems | (train_stems if args.include_train else set())
    sequences = []
    for stem in sorted(selected):
        role = "test" if stem in test_stems else "train"
        sequences.append(load_npz_sequence(args.arrays_dir / f"{stem}.npz", role))

    active_fields = list(checkpoint["active_fields"])
    context_len = int(checkpoint["context_len"])
    predict_delta = checkpoint["prediction_form"] == "delta_from_last_frame"
    seed = int(checkpoint.get("seed", -1))

    output = args.out_root / args.tag
    next_frame_dir = output / "next_frame"
    rollout_dir = output / "rollout"
    next_frame_dir.mkdir(parents=True, exist_ok=True)
    rollout_dir.mkdir(parents=True, exist_ok=True)

    next_rows = []
    rollout_rows = []
    for sequence in sequences:
        base_row = build_result_row(
            stem=sequence.stem,
            role=sequence.role,
            model_family=checkpoint["model_family"],
            tag=args.tag,
            active_fields=active_fields,
            context_len=context_len,
            split=args.split,
            seed=seed,
            prediction_form=checkpoint["prediction_form"],
            heldout_particle=args.heldout_particle,
            evaluation_group=(
                identity_fold.evaluation_group_for_stem(sequence.stem)
                if identity_fold is not None and sequence.role == "test"
                else None
            ),
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
        row = dict(base_row)
        row.update(
            rsb.summarize_prediction_result(
                sequence=sequence,
                pred=pred,
                naive=naive,
                targets=targets,
                target_steps=steps,
                reversal_radius=5,
            )
        )
        next_rows.append(row)
        rsb.save_per_stem_next_frame(
            str(next_frame_dir), row, pred, naive, targets, steps, frame_times
        )

        rollout_sequence = sequence
        anchor_start = None
        onset = None
        if identity_fold is not None:
            anchor_start, onset, rollout_sequence = (
                first_current_transition_slice(sequence, context_len=context_len)
            )
        pred, naive, targets, steps, frame_times = (
            rsb.run_rollout_eval_for_sequence(
                model=model,
                sequence=rollout_sequence,
                active_fields=active_fields,
                context_len=context_len,
                device=device,
                predict_delta=predict_delta,
                max_rollout_steps=args.max_rollout_steps,
            )
        )
        rollout_row = dict(base_row)
        rollout_row.update(
            rsb.summarize_prediction_result(
                sequence=rollout_sequence,
                pred=pred,
                naive=naive,
                targets=targets,
                target_steps=steps,
                reversal_radius=5,
                include_horizon=True,
            )
        )
        if identity_fold is not None:
            rollout_row.update(
                {
                    "anchor_rule": ROLLOUT_ANCHOR_RULE,
                    "anchor_frame": int(anchor_start),
                    "onset_frame": int(onset),
                }
            )
        rollout_rows.append(rollout_row)
        rsb.save_per_stem_rollout(
            str(rollout_dir),
            rollout_row,
            pred,
            naive,
            targets,
            steps,
            frame_times,
        )
        print(
            f"[eval done] {sequence.stem} "
            f"nf_ratio={row['mae_ratio']:.4f} "
            f"ro_ratio={rollout_row['mae_ratio']:.4f}",
            flush=True,
        )

    dump_mode_summaries(next_frame_dir, next_rows, "next_frame_results.json")
    dump_mode_summaries(rollout_dir, rollout_rows, "rollout_results.json")
    manifest = {
        "mode": "evaluation_only",
        "source_checkpoint": args.checkpoint.name,
        "tag": args.tag,
        "split": args.split,
        "heldout_particle": args.heldout_particle,
        "seed": seed,
        "roles_evaluated": sorted({sequence.role for sequence in sequences}),
        "test_only_comparison_summary": True,
    }
    (output / "evaluation_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(f"EVALUATION DONE {args.tag}")


if __name__ == "__main__":
    main()
