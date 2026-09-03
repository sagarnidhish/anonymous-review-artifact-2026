#!/usr/bin/env python3
"""Plain (non-Modal) training/eval entry point for HPC (CSD3 ampere).

Replicates modal_train.py run_experiment exactly: same splits, budgets,
reference-harness calls, artifact layout.  No modal dependency.

Example:
  python train_plain.py --model_family predrnnpp --tag predrnnpp_img_delta_lopo \
      --split lopo --epochs 26 --arrays_dir arrays --out_root out
"""

import argparse
import json
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "ref"))

import common_sp_baselines as csb  # noqa: E402
import run_sp_baseline_study as rsb  # noqa: E402
from models import build_model  # noqa: E402
from particle_splits import build_identity_holdout_fold  # noqa: E402
from protocol_evaluation_utils import (  # noqa: E402
    ROLLOUT_ANCHOR_RULE,
    first_current_transition_slice,
)
from result_metadata import build_result_row, rows_for_role  # noqa: E402


def make_evaluation_row(
        *, stem, role, model_family, tag, active_fields, context_len,
        split, seed, heldout_particle=None, evaluation_group=None):
    return build_result_row(
        stem=stem,
        role=role,
        model_family=model_family,
        tag=tag,
        active_fields=active_fields,
        context_len=context_len,
        split=split,
        seed=seed,
        prediction_form="delta_from_last_frame",
        heldout_particle=heldout_particle,
        evaluation_group=evaluation_group,
    )


def load_npz_sequence(path, role):
    stem = os.path.basename(path).replace(".npz", "")
    with np.load(path) as d:
        intensity = d["intensity"].astype(np.float32)
        exo = {k: d[k] for k in ("voltage", "current", "time_norm")}
        times = d["frame_times"]
    fi = np.arange(len(intensity), dtype=np.int64)
    return csb.LoadedSequence(
        stem=stem, role=role, intensity=intensity, exogenous_all=exo,
        frame_times=times[fi], raw_frame_indices=fi, sequence_subsample_factor=1,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_family", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument(
        "--split",
        default="frozen",
        choices=["frozen", "lopo", "identity_holdout"],
    )
    p.add_argument("--heldout_particle", type=int, choices=(1, 2, 3, 4))
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--epochs", type=int, default=26)
    p.add_argument("--arrays_dir", required=True)
    p.add_argument("--out_root", required=True)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--max_train_windows_per_stem", type=int, default=3000)
    p.add_argument("--max_eval_windows", type=int, default=3000)
    p.add_argument("--max_rollout_steps", type=int, default=512)
    args = p.parse_args()

    identity_fold = None
    if args.split == "frozen":
        train_stems = set(csb.TRAIN_STEMS)
        test_stems = set(csb.TEST_STEMS)
    elif args.split == "lopo":
        train_stems = {f"GRA29_C20_{t}_particle{i}"
                       for t in ("25deg", "45deg") for i in (1, 2, 3)}
        test_stems = {f"GRA29_C20_{t}_particle4" for t in ("25deg", "45deg")}
    else:
        if args.heldout_particle is None:
            p.error("--heldout_particle is required for identity_holdout")
        identity_fold = build_identity_holdout_fold(args.heldout_particle)
        train_stems = set(identity_fold.train_stems)
        test_stems = set(identity_fold.all_test_stems)

    device = torch.device("cuda")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    active_fields = ["intensity"]  # image-only configs for the controls
    all_sequences = []
    for s in sorted(train_stems | test_stems):
        role = "train" if s in train_stems else "test"
        all_sequences.append(load_npz_sequence(
            os.path.join(args.arrays_dir, f"{s}.npz"), role))
    train_sequences = [s for s in all_sequences if s.role == "train"]
    eval_sequences = sorted(
        (
            [sequence for sequence in all_sequences if sequence.role == "test"]
            if identity_fold is not None
            else all_sequences
        ),
        key=lambda sequence: (sequence.role, sequence.stem),
    )

    tr_idx, va_idx = [], []
    for si, seq in enumerate(train_sequences):
        starts = csb.build_window_starts(
            len(seq.intensity), 4, 1, args.max_train_windows_per_stem
        )
        tr, va = csb.split_train_val_starts(starts, 0.1)
        tr_idx.extend((si, s) for s in tr)
        va_idx.extend((si, s) for s in va)
    mk = lambda idx: csb.BaselineWindowDataset(
        train_sequences, idx, active_fields=active_fields, context_len=4,
        window_stride=1, predict_delta=True)
    train_loader = DataLoader(mk(tr_idx), batch_size=args.batch_size,
                              shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(mk(va_idx), batch_size=args.batch_size,
                            shuffle=False, num_workers=0, pin_memory=True)

    model = build_model(model_family=args.model_family,
                        in_fields=len(active_fields), context_len=4,
                        base_channels=32, hidden_layers=2).to(device)
    model, history, best = rsb.train_model(
        model=model, train_loader=train_loader, val_loader=val_loader,
        device=device, epochs=args.epochs, lr=args.lr,
        patience=args.patience, predict_delta=True, grad_clip_norm=1.0)

    out_root = os.path.join(args.out_root, args.tag)
    os.makedirs(os.path.join(out_root, "models"), exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "model_family": args.model_family,
                "active_fields": active_fields, "context_len": 4,
                "base_channels": 32, "hidden_layers": 2,
                "target_size": 128, "seed": args.seed, "mode": "train",
                "split": args.split,
                "heldout_particle": args.heldout_particle,
                "train_stems": sorted(train_stems),
                "test_stems": sorted(test_stems),
                "prediction_form": "delta_from_last_frame",
                "rollout_anchor_rule": (
                    ROLLOUT_ANCHOR_RULE if identity_fold is not None else None
                )},
               os.path.join(out_root, "models", f"{args.model_family}_best.pt"))
    with open(os.path.join(out_root, "models", "training_history.json"),
              "w") as f:
        json.dump({"cfg": vars(args), "best": best, "history": history}, f,
                  indent=2, default=float)

    nf_dir = os.path.join(out_root, "next_frame")
    ro_dir = os.path.join(out_root, "rollout")
    os.makedirs(nf_dir, exist_ok=True)
    os.makedirs(ro_dir, exist_ok=True)
    next_rows, rollout_rows = [], []
    for seq in eval_sequences:
        pred, naive, targets, steps, ftimes = rsb.run_next_frame_eval_for_sequence(
            model=model, sequence=seq, active_fields=active_fields,
            context_len=4, window_stride=1,
            max_windows=args.max_eval_windows,
            batch_size=args.batch_size, device=device, predict_delta=True)
        metrics = rsb.summarize_prediction_result(
            sequence=seq, pred=pred, naive=naive, targets=targets,
            target_steps=steps, reversal_radius=5)
        row = make_evaluation_row(
            stem=seq.stem,
            role=seq.role,
            model_family=args.model_family,
            tag=args.tag,
            active_fields=active_fields,
            context_len=4,
            split=args.split,
            seed=args.seed,
            heldout_particle=args.heldout_particle,
            evaluation_group=(
                identity_fold.evaluation_group_for_stem(seq.stem)
                if identity_fold is not None and seq.role == "test"
                else None
            ),
        )
        row.update(metrics)
        next_rows.append(row)
        rsb.save_per_stem_next_frame(nf_dir, row, pred, naive, targets, steps,
                                     ftimes)

        rollout_sequence = seq
        anchor_start = None
        onset = None
        if identity_fold is not None:
            anchor_start, onset, rollout_sequence = (
                first_current_transition_slice(seq, context_len=4)
            )
        rpred, rnaive, rtargets, rsteps, rftimes = rsb.run_rollout_eval_for_sequence(
            model=model, sequence=rollout_sequence, active_fields=active_fields,
            context_len=4, device=device, predict_delta=True,
            max_rollout_steps=args.max_rollout_steps)
        rmetrics = rsb.summarize_prediction_result(
            sequence=rollout_sequence, pred=rpred, naive=rnaive, targets=rtargets,
            target_steps=rsteps, reversal_radius=5, include_horizon=True)
        rrow = dict(row)
        rrow.update(rmetrics)
        if identity_fold is not None:
            rrow.update({
                "anchor_rule": ROLLOUT_ANCHOR_RULE,
                "anchor_frame": int(anchor_start),
                "onset_frame": int(onset),
            })
        rollout_rows.append(rrow)
        rsb.save_per_stem_rollout(ro_dir, rrow, rpred, rnaive, rtargets,
                                  rsteps, rftimes)
        print(f"[eval done] {seq.stem} nf_ratio={row['mae_ratio']:.4f} "
              f"ro_ratio={rrow['mae_ratio']:.4f}", flush=True)

    rsb.dump_summary(os.path.join(nf_dir, "next_frame_results.json"),
                     next_rows)
    rsb.dump_summary(os.path.join(nf_dir, "comparison_summary_all.json"),
                     [rsb.summarize_rows(next_rows)])
    rsb.dump_summary(os.path.join(nf_dir, "comparison_summary.json"),
                     [rsb.summarize_rows(rows_for_role(next_rows, "test"))])
    rsb.dump_summary(os.path.join(ro_dir, "rollout_results.json"),
                     rollout_rows)
    rsb.dump_summary(os.path.join(ro_dir, "comparison_summary_all.json"),
                     [rsb.summarize_rows(rollout_rows)])
    rsb.dump_summary(os.path.join(ro_dir, "comparison_summary.json"),
                     [rsb.summarize_rows(rows_for_role(rollout_rows, "test"))])
    if identity_fold is not None:
        for rows, directory in ((next_rows, nf_dir), (rollout_rows, ro_dir)):
            grouped = []
            for group_name in (
                    "same_temperature_unseen_particle",
                    "cross_temperature_unseen_particle"):
                group_rows = [
                    row for row in rows
                    if row.get("evaluation_group") == group_name
                ]
                summary = rsb.summarize_rows(group_rows)
                summary["evaluation_group"] = group_name
                grouped.append(summary)
            rsb.dump_summary(
                os.path.join(directory, "comparison_summary_by_group.json"),
                grouped,
            )
    print("ALL DONE", args.tag)


if __name__ == "__main__":
    main()
