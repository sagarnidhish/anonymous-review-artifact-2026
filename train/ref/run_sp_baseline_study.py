#!/usr/bin/env python3
import argparse
import json
import logging
import os
import random
import sys
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common_sp_baselines import (
    TEST_STEMS,
    TRAIN_STEMS,
    BaselineWindowDataset,
    build_active_fields,
    build_context_tensor,
    build_window_starts,
    compute_rollout_horizon_metrics,
    dump_summary,
    list_well_files,
    load_sp_sequence,
    role_of_stem,
    split_train_val_starts,
    stem_of,
    summarize_prediction_result,
    summarize_rows,
)
from models import build_model


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--well_data", required=True)
    p.add_argument("--model_out", required=True)
    p.add_argument("--next_frame_out", required=True)
    p.add_argument("--rollout_out", required=True)
    p.add_argument("--model_family", required=True, choices=["unet", "convlstm", "simvp", "residual_cnn", "predrnn", "predrnnpp"])
    p.add_argument("--tag", default="")
    p.add_argument("--target_size", type=int, default=128)
    p.add_argument("--context_len", type=int, default=4)
    p.add_argument("--window_stride", type=int, default=1)
    p.add_argument("--sequence_subsample_factor", type=int, default=1)
    p.add_argument("--use_voltage", action="store_true")
    p.add_argument("--use_current", action="store_true")
    p.add_argument("--use_time_norm", action="store_true")
    p.add_argument("--predict_delta", action="store_true")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--base_channels", type=int, default=32)
    p.add_argument("--hidden_layers", type=int, default=2)
    p.add_argument("--max_train_windows_per_stem", type=int, default=3000)
    p.add_argument("--max_eval_windows", type=int, default=3000)
    p.add_argument("--max_rollout_steps", type=int, default=0)
    p.add_argument("--reversal_radius", type=int, default=5)
    p.add_argument("--grad_clip_norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


def compute_mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(pred - target))


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    patience: int,
    predict_delta: bool,
    grad_clip_norm: float,
):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=min(lr * 1e-2, 1e-6))
    history: Dict[str, List[float]] = {
        "train_mae": [],
        "val_mae": [],
        "val_naive_mae": [],
        "val_ratio": [],
        "val_reversal_mae": [],
        "val_nonreversal_mae": [],
    }
    best = {"epoch": 0, "val_mae": float("inf")}
    best_state = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        tr_total = 0.0
        tr_n = 0
        for x, y, _, _, _, _ in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad()
            pred_raw = model(x)
            loss = compute_mae(pred_raw, y)
            loss.backward()
            if grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            opt.step()
            tr_total += loss.item() * len(x)
            tr_n += len(x)

        model.eval()
        val_total = 0.0
        val_naive_total = 0.0
        val_rev_sum = 0.0
        val_rev_n = 0
        val_nonrev_sum = 0.0
        val_nonrev_n = 0
        val_n = 0
        with torch.no_grad():
            for x, y, naive, _, _, reversal in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                naive = naive.to(device, non_blocking=True)
                reversal = reversal.to(device, non_blocking=True).bool()
                pred_raw = model(x)
                if predict_delta:
                    pred_frame = naive + pred_raw
                    target_frame = naive + y
                else:
                    pred_frame = pred_raw
                    target_frame = y
                mae_vals = torch.mean(torch.abs(pred_frame - target_frame), dim=(1, 2, 3))
                naive_mae_vals = torch.mean(torch.abs(naive - target_frame), dim=(1, 2, 3))
                val_total += mae_vals.sum().item()
                val_naive_total += naive_mae_vals.sum().item()
                val_n += len(x)
                if reversal.any():
                    val_rev_sum += mae_vals[reversal].sum().item()
                    val_rev_n += int(reversal.sum().item())
                if (~reversal).any():
                    val_nonrev_sum += mae_vals[~reversal].sum().item()
                    val_nonrev_n += int((~reversal).sum().item())

        sched.step()
        train_mae = tr_total / max(tr_n, 1)
        val_mae = val_total / max(val_n, 1)
        val_naive_mae = val_naive_total / max(val_n, 1)
        val_ratio = val_mae / max(val_naive_mae, 1e-12)
        val_reversal_mae = val_rev_sum / max(val_rev_n, 1) if val_rev_n else float("nan")
        val_nonreversal_mae = val_nonrev_sum / max(val_nonrev_n, 1) if val_nonrev_n else float("nan")
        history["train_mae"].append(train_mae)
        history["val_mae"].append(val_mae)
        history["val_naive_mae"].append(val_naive_mae)
        history["val_ratio"].append(val_ratio)
        history["val_reversal_mae"].append(val_reversal_mae)
        history["val_nonreversal_mae"].append(val_nonreversal_mae)

        improved = val_mae < best["val_mae"]
        if improved:
            best = {
                "epoch": epoch,
                "val_mae": float(val_mae),
                "val_ratio": float(val_ratio),
                "val_reversal_mae": float(val_reversal_mae) if not np.isnan(val_reversal_mae) else None,
                "val_nonreversal_mae": float(val_nonreversal_mae) if not np.isnan(val_nonreversal_mae) else None,
            }
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        log.info(
            "Epoch %3d/%d | train_mae=%.6f | val_mae=%.6f | val_naive_mae=%.6f | val_ratio=%.4f%s",
            epoch,
            epochs,
            train_mae,
            val_mae,
            val_naive_mae,
            val_ratio,
            " *" if improved else "",
        )
        if no_improve >= patience:
            log.info("Early stopping at epoch %d", epoch)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best


@torch.no_grad()
def run_next_frame_eval_for_sequence(
    model: nn.Module,
    sequence,
    active_fields,
    context_len: int,
    window_stride: int,
    max_windows: int,
    batch_size: int,
    device: torch.device,
    predict_delta: bool,
):
    starts = build_window_starts(len(sequence.intensity), context_len, window_stride, max_windows)
    dataset = BaselineWindowDataset(
        [sequence],
        [(0, s) for s in starts],
        active_fields=active_fields,
        context_len=context_len,
        window_stride=window_stride,
        predict_delta=predict_delta,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))
    preds, naives, targets, steps, frame_times = [], [], [], [], []
    model.eval()
    for x, y, naive, target_step, frame_time, _ in loader:
        x = x.to(device, non_blocking=True)
        pred_raw = model(x).cpu().numpy()[:, 0].astype(np.float32)
        naive_np = naive.numpy()[:, 0].astype(np.float32)
        if predict_delta:
            pred = naive_np + pred_raw
            target = naive_np + y.numpy()[:, 0].astype(np.float32)
        else:
            pred = pred_raw
            target = y.numpy()[:, 0].astype(np.float32)
        preds.append(pred)
        naives.append(naive_np)
        targets.append(target)
        steps.append(target_step.numpy().astype(np.int64))
        frame_times.append(frame_time.numpy().astype(np.float32))
    if not preds:
        s = sequence.intensity.shape[-1]
        empty = np.empty((0, s, s), dtype=np.float32)
        return empty, empty, empty, np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float32)
    return (
        np.concatenate(preds, axis=0),
        np.concatenate(naives, axis=0),
        np.concatenate(targets, axis=0),
        np.concatenate(steps, axis=0),
        np.concatenate(frame_times, axis=0),
    )


@torch.no_grad()
def run_rollout_eval_for_sequence(
    model: nn.Module,
    sequence,
    active_fields,
    context_len: int,
    device: torch.device,
    predict_delta: bool,
    max_rollout_steps: int,
):
    total_steps = len(sequence.intensity) - context_len
    if total_steps <= 0:
        s = sequence.intensity.shape[-1]
        empty = np.empty((0, s, s), dtype=np.float32)
        return empty, empty, empty, np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float32)
    if max_rollout_steps > 0:
        total_steps = min(total_steps, max_rollout_steps)

    pred_intensity_context = sequence.intensity[:context_len].copy()
    naive_intensity_context = sequence.intensity[:context_len].copy()
    preds, naives, targets, steps, frame_times = [], [], [], [], []
    model.eval()
    for offset in range(total_steps):
        target_index = context_len + offset
        context_indices = list(range(target_index - context_len, target_index))
        x_seq = build_context_tensor(
            sequence=sequence,
            intensity_context=pred_intensity_context,
            context_indices=context_indices,
            active_fields=active_fields,
        )
        x = torch.from_numpy(x_seq[None]).to(device, non_blocking=True)
        pred_raw = model(x).cpu().numpy()[0, 0].astype(np.float32)
        pred_next = pred_intensity_context[-1] + pred_raw if predict_delta else pred_raw
        naive_next = naive_intensity_context[-1].copy().astype(np.float32)
        target = sequence.intensity[target_index].astype(np.float32)
        preds.append(pred_next)
        naives.append(naive_next)
        targets.append(target)
        steps.append(target_index)
        frame_times.append(sequence.frame_times[target_index])
        if offset != total_steps - 1:
            pred_intensity_context = np.concatenate([pred_intensity_context[1:], pred_next[None]], axis=0)
            naive_intensity_context = np.concatenate([naive_intensity_context[1:], naive_next[None]], axis=0)
    return (
        np.stack(preds).astype(np.float32),
        np.stack(naives).astype(np.float32),
        np.stack(targets).astype(np.float32),
        np.asarray(steps, dtype=np.int64),
        np.asarray(frame_times, dtype=np.float32),
    )


def save_per_stem_next_frame(out_root, row, pred, naive, targets, target_steps, frame_times):
    stem_dir = os.path.join(out_root, row["stem"])
    os.makedirs(stem_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(stem_dir, f"preds_{row['stem']}.npz"),
        pred=pred,
        naive=naive,
        targets=targets,
        target_steps=target_steps,
        frame_times=frame_times,
        active_fields=np.asarray(row["active_fields"]),
        model_family=row["model_family"],
        tag=row["tag"],
        prediction_form=row["prediction_form"],
        model_mae=np.asarray(row["model_mae"], dtype=np.float32),
        naive_mae=np.asarray(row["naive_mae"], dtype=np.float32),
        mae_ratio=np.asarray(row["mae_ratio"], dtype=np.float32),
        model_mse=np.asarray(row["model_mse"], dtype=np.float32),
        naive_mse=np.asarray(row["naive_mse"], dtype=np.float32),
        mse_ratio=np.asarray(row["mse_ratio"], dtype=np.float32),
    )
    dump_summary(os.path.join(stem_dir, "next_frame_results.json"), [row])
    with open(os.path.join(stem_dir, "study_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "mode": "next_frame",
                "stem": row["stem"],
                "model_family": row["model_family"],
                "tag": row["tag"],
                "active_fields": row["active_fields"],
                "prediction_form": row["prediction_form"],
                "evaluation_scope": "full_frame_only",
            },
            f,
            indent=2,
        )


def save_per_stem_rollout(out_root, row, pred, naive, targets, target_steps, frame_times):
    stem_dir = os.path.join(out_root, row["stem"])
    os.makedirs(stem_dir, exist_ok=True)
    horizon = compute_rollout_horizon_metrics(pred, naive, targets)
    np.savez_compressed(
        os.path.join(stem_dir, f"rollout_{row['stem']}.npz"),
        pred=pred,
        naive=naive,
        targets=targets,
        target_steps=target_steps,
        frame_times=frame_times,
        active_fields=np.asarray(row["active_fields"]),
        model_family=row["model_family"],
        tag=row["tag"],
        prediction_form=row["prediction_form"],
        per_step_model_mae=np.asarray(horizon["per_step_model_mae"], dtype=np.float32),
        per_step_naive_mae=np.asarray(horizon["per_step_naive_mae"], dtype=np.float32),
        per_step_mae_ratio=np.asarray(horizon["per_step_mae_ratio"], dtype=np.float32),
        model_mae=np.asarray(row["model_mae"], dtype=np.float32),
        naive_mae=np.asarray(row["naive_mae"], dtype=np.float32),
        mae_ratio=np.asarray(row["mae_ratio"], dtype=np.float32),
        model_mse=np.asarray(row["model_mse"], dtype=np.float32),
        naive_mse=np.asarray(row["naive_mse"], dtype=np.float32),
        mse_ratio=np.asarray(row["mse_ratio"], dtype=np.float32),
    )
    dump_summary(os.path.join(stem_dir, "rollout_results.json"), [row])
    with open(os.path.join(stem_dir, "study_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "mode": "rollout",
                "stem": row["stem"],
                "model_family": row["model_family"],
                "tag": row["tag"],
                "active_fields": row["active_fields"],
                "prediction_form": row["prediction_form"],
                "evaluation_scope": "full_frame_only",
            },
            f,
            indent=2,
        )


def main():
    args = parse_args()
    os.makedirs(args.model_out, exist_ok=True)
    os.makedirs(args.next_frame_out, exist_ok=True)
    os.makedirs(args.rollout_out, exist_ok=True)
    active_fields = build_active_fields(args.use_voltage, args.use_current, args.use_time_norm)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    log.info("Model family: %s", args.model_family)
    log.info("Tag: %s", args.tag or "<none>")
    log.info("Active fields: %s", active_fields)
    log.info("Prediction form: %s", "delta_from_last_frame" if args.predict_delta else "direct_next_frame")
    log.info("Device: %s", device)
    log.info("Seed: %d", args.seed)
    log.info("Grad clip norm: %.4f", args.grad_clip_norm)

    all_sequences = []
    for well_path in list_well_files(args.well_data):
        stem = stem_of(well_path)
        if role_of_stem(stem) == "ignore":
            continue
        seq = load_sp_sequence(
            well_path,
            target_size=args.target_size,
            sequence_subsample_factor=args.sequence_subsample_factor,
        )
        all_sequences.append(seq)
        log.info("Loaded %s [%s] | frames=%d", seq.stem, seq.role, len(seq.intensity))

    train_sequences = [s for s in all_sequences if s.stem in TRAIN_STEMS]
    eval_sequences = sorted(all_sequences, key=lambda s: (s.role, s.stem))
    train_index = []
    val_index = []
    for seq_idx, seq in enumerate(train_sequences):
        starts = build_window_starts(
            len(seq.intensity),
            args.context_len,
            args.window_stride,
            args.max_train_windows_per_stem,
        )
        tr_starts, val_starts = split_train_val_starts(starts, args.val_fraction)
        train_index.extend((seq_idx, s) for s in tr_starts)
        val_index.extend((seq_idx, s) for s in val_starts)
        log.info("Train stem %s | windows=%d | train=%d | val=%d", seq.stem, len(starts), len(tr_starts), len(val_starts))

    train_ds = BaselineWindowDataset(
        train_sequences,
        train_index,
        active_fields=active_fields,
        context_len=args.context_len,
        window_stride=args.window_stride,
        predict_delta=args.predict_delta,
    )
    val_ds = BaselineWindowDataset(
        train_sequences,
        val_index,
        active_fields=active_fields,
        context_len=args.context_len,
        window_stride=args.window_stride,
        predict_delta=args.predict_delta,
    )
    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=pin)

    model = build_model(
        model_family=args.model_family,
        in_fields=len(active_fields),
        context_len=args.context_len,
        base_channels=args.base_channels,
        hidden_layers=args.hidden_layers,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("Model params: %s", f"{n_params:,}")

    model, history, best = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        predict_delta=args.predict_delta,
        grad_clip_norm=args.grad_clip_norm,
    )

    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_family": args.model_family,
            "active_fields": active_fields,
            "context_len": args.context_len,
            "base_channels": args.base_channels,
            "hidden_layers": args.hidden_layers,
            "target_size": args.target_size,
            "window_stride": args.window_stride,
            "sequence_subsample_factor": args.sequence_subsample_factor,
            "prediction_form": "delta_from_last_frame" if args.predict_delta else "direct_next_frame",
            "seed": args.seed,
            "grad_clip_norm": args.grad_clip_norm,
            "best": best,
        },
        os.path.join(args.model_out, f"{args.model_family}_best.pt"),
    )
    with open(os.path.join(args.model_out, "training_history.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_family": args.model_family,
                "tag": args.tag,
                "active_fields": active_fields,
                "n_params": n_params,
                "prediction_form": "delta_from_last_frame" if args.predict_delta else "direct_next_frame",
                "seed": args.seed,
                "grad_clip_norm": args.grad_clip_norm,
                "best": best,
                "history": history,
                "evaluation_scope": "full_frame_only",
            },
            f,
            indent=2,
        )

    next_rows = []
    for seq in eval_sequences:
        pred, naive, targets, target_steps, frame_times = run_next_frame_eval_for_sequence(
            model=model,
            sequence=seq,
            active_fields=active_fields,
            context_len=args.context_len,
            window_stride=args.window_stride,
            max_windows=args.max_eval_windows,
            batch_size=args.batch_size,
            device=device,
            predict_delta=args.predict_delta,
        )
        metrics = summarize_prediction_result(
            sequence=seq,
            pred=pred,
            naive=naive,
            targets=targets,
            target_steps=target_steps,
            reversal_radius=args.reversal_radius,
            include_horizon=False,
        )
        row = {
            "stem": seq.stem,
            "role": seq.role,
            "model_family": args.model_family,
            "tag": args.tag,
            "active_fields": active_fields,
            "context_len": args.context_len,
            "window_stride": args.window_stride,
            "sequence_subsample_factor": args.sequence_subsample_factor,
            "prediction_form": "delta_from_last_frame" if args.predict_delta else "direct_next_frame",
            "evaluation_scope": "full_frame_only",
            **metrics,
        }
        next_rows.append(row)
        save_per_stem_next_frame(args.next_frame_out, row, pred, naive, targets, target_steps, frame_times)
    dump_summary(os.path.join(args.next_frame_out, "next_frame_results.json"), next_rows)
    with open(os.path.join(args.next_frame_out, "study_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "mode": "next_frame",
                "model_family": args.model_family,
                "tag": args.tag,
                "active_fields": active_fields,
                "prediction_form": "delta_from_last_frame" if args.predict_delta else "direct_next_frame",
                "train_stems": sorted(TRAIN_STEMS),
                "test_stems": sorted(TEST_STEMS),
                "evaluation_scope": "full_frame_only",
            },
            f,
            indent=2,
        )
    with open(os.path.join(args.next_frame_out, "comparison_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summarize_rows(next_rows), f, indent=2)

    rollout_rows = []
    for seq in eval_sequences:
        pred, naive, targets, target_steps, frame_times = run_rollout_eval_for_sequence(
            model=model,
            sequence=seq,
            active_fields=active_fields,
            context_len=args.context_len,
            device=device,
            predict_delta=args.predict_delta,
            max_rollout_steps=args.max_rollout_steps,
        )
        metrics = summarize_prediction_result(
            sequence=seq,
            pred=pred,
            naive=naive,
            targets=targets,
            target_steps=target_steps,
            reversal_radius=args.reversal_radius,
            include_horizon=True,
        )
        row = {
            "stem": seq.stem,
            "role": seq.role,
            "model_family": args.model_family,
            "tag": args.tag,
            "active_fields": active_fields,
            "context_len": args.context_len,
            "sequence_subsample_factor": args.sequence_subsample_factor,
            "prediction_form": "delta_from_last_frame" if args.predict_delta else "direct_next_frame",
            "evaluation_scope": "full_frame_only",
            **metrics,
        }
        rollout_rows.append(row)
        save_per_stem_rollout(args.rollout_out, row, pred, naive, targets, target_steps, frame_times)
    dump_summary(os.path.join(args.rollout_out, "rollout_results.json"), rollout_rows)
    with open(os.path.join(args.rollout_out, "study_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "mode": "rollout",
                "model_family": args.model_family,
                "active_fields": active_fields,
                "prediction_form": "delta_from_last_frame" if args.predict_delta else "direct_next_frame",
                "train_stems": sorted(TRAIN_STEMS),
                "test_stems": sorted(TEST_STEMS),
                "evaluation_scope": "full_frame_only",
            },
            f,
            indent=2,
        )
    with open(os.path.join(args.rollout_out, "comparison_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summarize_rows(rollout_rows), f, indent=2)


if __name__ == "__main__":
    main()
