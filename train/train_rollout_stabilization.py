#!/usr/bin/env python3
"""Fine-tune one matched U-Net with a declared rollout-stabilization strategy."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "ref"))

from models import build_model
from protocol_evaluation_utils import ROLLOUT_ANCHOR_RULE, first_current_transition_slice
from safe_anchored_rollout import cumulative_horizon_metrics, run_safe_rollout
from stabilization_payloads import payload_for_id, teacher_forcing_probability
from stabilization_training import choose_feedback_frame, corrupt_intensity_context
from train_matched_grid import (
    TEST_STEMS,
    TRAIN_STEMS,
    load_sequence,
    strict_json_dump,
)

import common_sp_baselines as csb
import run_sp_baseline_study as rsb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload-id", type=int, required=True)
    parser.add_argument("--arrays-dir", required=True)
    parser.add_argument("--grid-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--source-commit", default="unknown")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_starts(sequences, context_len: int):
    training, validation, manifest = {}, {}, {}
    for sequence in sequences:
        starts = csb.build_window_starts(len(sequence.intensity), context_len, 1, 3000)
        train_starts, val_starts = csb.split_train_val_starts(starts, 0.1)
        training[sequence.stem] = np.asarray(train_starts, dtype=np.int64)
        validation[sequence.stem] = np.asarray(val_starts, dtype=np.int64)
        manifest[sequence.stem] = {
            "train_start_min": int(train_starts[0]),
            "train_start_max": int(train_starts[-1]),
            "train_count": len(train_starts),
            "validation_start_min": int(val_starts[0]),
            "validation_start_max": int(val_starts[-1]),
            "validation_count": len(val_starts),
        }
    return training, validation, manifest


def make_batch(items, context_len: int, device: torch.device):
    sequences = [item[0] for item in items]
    starts = [int(item[1]) for item in items]
    context = torch.from_numpy(
        np.stack(
            [sequence.intensity[start : start + context_len] for sequence, start in items]
        )
    ).float().to(device)
    return sequences, starts, context


def target_batch(sequences, starts, offset: int, device: torch.device):
    return torch.from_numpy(
        np.stack(
            [sequence.intensity[start + offset] for sequence, start in zip(sequences, starts)]
        )
    ).float().unsqueeze(1).to(device)


def validation_free_rollout_mae(
    model,
    sequences,
    starts_by_stem,
    payload,
    device,
):
    model.eval()
    context_len = int(payload["context_len"])
    horizon = int(payload["validation_horizon"])
    count = int(payload["validation_windows_per_stem"])
    errors = []
    with torch.no_grad():
        for sequence in sequences:
            candidates = starts_by_stem[sequence.stem]
            positions = np.linspace(0, len(candidates) - 1, num=min(count, len(candidates)))
            for position in np.unique(np.rint(positions).astype(np.int64)):
                start = int(candidates[position])
                context = torch.from_numpy(
                    sequence.intensity[start : start + context_len][None]
                ).float().to(device)
                for step in range(horizon):
                    raw = model(context.unsqueeze(2))
                    prediction = context[:, -1:].clone() + raw
                    target = torch.from_numpy(
                        sequence.intensity[start + context_len + step][None, None]
                    ).float().to(device)
                    errors.append(float(torch.mean(torch.abs(prediction - target))))
                    context = torch.cat([context[:, 1:], prediction], dim=1)
    return float(np.mean(errors))


def fine_tune(model, sequences, training_starts, validation_starts, payload, device):
    seed = int(payload["seed"])
    rng = np.random.default_rng(seed + 701 + int(payload["payload_id"]))
    torch.manual_seed(seed + 907 + int(payload["payload_id"]))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(payload["lr"]), weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(payload["epochs"])
    )
    context_len = int(payload["context_len"])
    horizon = int(payload["training_horizon"])
    batch_size = int(payload["batch_windows"])
    per_stem = int(payload["train_windows_per_stem_per_epoch"])
    strategy = payload["strategy"]
    history = []
    best_state = None
    best = {"epoch": 0, "validation_free_rollout_mae_h32": float("inf")}
    stale = 0

    for epoch in range(1, int(payload["epochs"]) + 1):
        pool = []
        for sequence in sequences:
            candidates = training_starts[sequence.stem]
            selected = rng.choice(
                candidates, size=min(per_stem, len(candidates)), replace=False
            )
            pool.extend((sequence, int(start)) for start in selected)
        rng.shuffle(pool)
        teacher_probability = teacher_forcing_probability(payload, epoch)
        model.train()
        total_loss = 0.0
        total_steps = 0
        for low in range(0, len(pool), batch_size):
            items = pool[low : low + batch_size]
            sequence_batch, starts, context = make_batch(items, context_len, device)
            if strategy == "input_noise":
                context = corrupt_intensity_context(context, float(payload["noise_std"]))
            optimizer.zero_grad(set_to_none=True)
            step_losses = []
            for step in range(horizon):
                raw = model(context.unsqueeze(2))
                prediction = context[:, -1:].clone() + raw
                target = target_batch(
                    sequence_batch, starts, context_len + step, device
                )
                step_loss = torch.mean(torch.abs(prediction - target))
                step_losses.append(step_loss)
                if step + 1 < horizon:
                    feedback = choose_feedback_frame(
                        truth=target,
                        prediction=prediction,
                        teacher_forcing_probability=teacher_probability,
                        detach_prediction=bool(payload["detach_feedback"]),
                        rng=rng,
                    )
                    context = torch.cat([context[:, 1:], feedback], dim=1)
            loss = torch.stack(step_losses).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(items)
            total_steps += len(items)

        validation_mae = validation_free_rollout_mae(
            model, sequences, validation_starts, payload, device
        )
        epoch_row = {
            "epoch": epoch,
            "train_mae": total_loss / max(total_steps, 1),
            "validation_free_rollout_mae_h32": validation_mae,
            "teacher_forcing_probability": teacher_probability,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(epoch_row)
        print(json.dumps(epoch_row, sort_keys=True), flush=True)
        if validation_mae < best["validation_free_rollout_mae_h32"]:
            best = deepcopy(epoch_row)
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        scheduler.step()
        if stale >= int(payload["patience"]):
            break

    if best_state is None:
        raise RuntimeError("fine-tuning produced no selectable checkpoint")
    model.load_state_dict(best_state)
    return history, best


def evaluate(model, test_sequences, payload, output_dir, device):
    context_len = int(payload["context_len"])
    next_dir = output_dir / "next_frame"
    rollout_dir = output_dir / "rollout_anchored"
    next_dir.mkdir(parents=True, exist_ok=True)
    rollout_dir.mkdir(parents=True, exist_ok=True)
    next_rows, rollout_rows = [], []
    for sequence in test_sequences:
        pred, naive, targets, target_steps, frame_times = (
            rsb.run_next_frame_eval_for_sequence(
                model=model,
                sequence=sequence,
                active_fields=["intensity"],
                context_len=context_len,
                window_stride=1,
                max_windows=3000,
                batch_size=8,
                device=device,
                predict_delta=True,
            )
        )
        metrics = rsb.summarize_prediction_result(
            sequence=sequence,
            pred=pred,
            naive=naive,
            targets=targets,
            target_steps=target_steps,
            reversal_radius=5,
        )
        row = {
            "stem": sequence.stem,
            "role": "test",
            "model_family": "unet",
            "tag": payload["tag"],
            "payload_id": payload["payload_id"],
            "strategy": payload["strategy"],
            "active_fields": ["intensity"],
            "context_len": context_len,
            "prediction_form": "delta_from_last_frame",
            **metrics,
        }
        next_rows.append(row)
        rsb.save_per_stem_next_frame(
            str(next_dir), row, pred, naive, targets, target_steps, frame_times
        )

        anchor_start, onset, anchored = first_current_transition_slice(
            sequence, context_len=context_len
        )
        result = run_safe_rollout(
            model=model,
            sequence=anchored,
            active_fields=["intensity"],
            context_len=context_len,
            device=device,
            predict_delta=True,
            max_rollout_steps=int(payload["rollout_steps"]),
        )
        horizons = cumulative_horizon_metrics(
            result["pred"],
            result["naive"],
            result["targets"],
            horizons=payload["report_horizons"],
            first_nonfinite_step=result["first_nonfinite_step"],
        )
        absolute_steps = anchored.raw_frame_indices[result["target_steps"]]
        full = horizons[str(payload["rollout_steps"])]
        rollout_row = {
            "stem": sequence.stem,
            "role": "test",
            "model_family": "unet",
            "tag": payload["tag"],
            "payload_id": payload["payload_id"],
            "strategy": payload["strategy"],
            "active_fields": ["intensity"],
            "prediction_form": "delta_from_last_frame",
            "anchor_rule": ROLLOUT_ANCHOR_RULE,
            "anchor_frame": int(anchor_start),
            "onset_frame": int(onset),
            "rollout_steps": int(payload["rollout_steps"]),
            "status": result["status"],
            "first_nonfinite_step": result["first_nonfinite_step"],
            "model_mae": full["model_mae"],
            "naive_mae": full["naive_mae"],
            "mae_ratio": full["mae_ratio"],
            "horizons": horizons,
        }
        rollout_rows.append(rollout_row)
        stem_dir = rollout_dir / sequence.stem
        stem_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            stem_dir / f"rollout_{sequence.stem}.npz",
            pred=result["pred"],
            naive=result["naive"],
            targets=result["targets"],
            target_steps=absolute_steps,
            frame_times=result["frame_times"],
            anchor_frame=np.asarray(anchor_start, dtype=np.int64),
            onset_frame=np.asarray(onset, dtype=np.int64),
            first_nonfinite_step=np.asarray(
                -1
                if result["first_nonfinite_step"] is None
                else result["first_nonfinite_step"],
                dtype=np.int64,
            ),
        )
        strict_json_dump(stem_dir / "rollout_results.json", [rollout_row])
        print(
            f"[evaluation] {sequence.stem} next={row['mae_ratio']:.4f} "
            f"rollout512={full['mae_ratio']}",
            flush=True,
        )
    strict_json_dump(next_dir / "next_frame_results.json", next_rows)
    strict_json_dump(rollout_dir / "rollout_results.json", rollout_rows)
    return next_rows, rollout_rows


def main():
    args = parse_args()
    payload = payload_for_id(args.payload_id)
    arrays_dir = Path(args.arrays_dir).resolve()
    grid_root = Path(args.grid_root).resolve()
    output_dir = Path(args.out_root).resolve() / payload["tag"]
    completion_path = output_dir / "completion_manifest.json"
    if completion_path.is_file():
        raise RuntimeError(f"complete payload already exists: {completion_path}")

    source_dir = grid_root / payload["source_tag"]
    source_completion = source_dir / "completion_manifest.json"
    source_checkpoint = source_dir / "models" / "unet_best.pt"
    if not source_completion.is_file() or not source_checkpoint.is_file():
        raise FileNotFoundError(f"fresh source payload is incomplete: {source_dir}")
    source_manifest = json.loads(source_completion.read_text())
    if source_manifest.get("status") != "complete":
        raise ValueError("fresh source payload did not complete")

    seed = int(payload["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for stabilization training")
    device = torch.device("cuda")
    train_sequences = [load_sequence(arrays_dir, stem, "train") for stem in TRAIN_STEMS]
    test_sequences = [load_sequence(arrays_dir, stem, "test") for stem in TEST_STEMS]
    training_starts, validation_starts, split_manifest = split_starts(
        train_sequences, int(payload["context_len"])
    )

    checkpoint = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    source_payload = checkpoint.get("payload", {})
    expected = {
        "model_family": "unet",
        "input_mode": "image_only",
        "target_mode": "delta",
        "seed": seed,
    }
    for key, value in expected.items():
        if source_payload.get(key) != value:
            raise ValueError(f"source checkpoint {key} mismatch: {source_payload.get(key)}")
    model = build_model(
        model_family="unet",
        in_fields=1,
        context_len=int(payload["context_len"]),
        base_channels=int(checkpoint["base_channels"]),
        hidden_layers=int(checkpoint["hidden_layers"]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    source_hash = file_sha256(source_checkpoint)
    history, best = fine_tune(
        model,
        train_sequences,
        training_starts,
        validation_starts,
        payload,
        device,
    )

    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    tuned_checkpoint = model_dir / "unet_best.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "payload": payload,
            "active_fields": ["intensity"],
            "prediction_form": "delta_from_last_frame",
            "base_channels": int(checkpoint["base_channels"]),
            "hidden_layers": int(checkpoint["hidden_layers"]),
            "source_checkpoint_sha256": source_hash,
            "source_commit": args.source_commit,
            "normalization": "archived_per_movie_full_record",
        },
        tuned_checkpoint,
    )
    strict_json_dump(
        model_dir / "training_history.json",
        {
            "payload": payload,
            "best": best,
            "history": history,
            "split_windows": split_manifest,
            "source_checkpoint": str(source_checkpoint),
            "source_checkpoint_sha256": source_hash,
            "source_commit": args.source_commit,
        },
    )
    next_rows, rollout_rows = evaluate(
        model, test_sequences, payload, output_dir, device
    )
    strict_json_dump(
        completion_path,
        {
            "status": "complete",
            "payload": payload,
            "source_commit": args.source_commit,
            "source_checkpoint_sha256": source_hash,
            "checkpoint": str(tuned_checkpoint.relative_to(output_dir)),
            "selected_epoch": int(best["epoch"]),
            "selection_metric": payload["selection_metric"],
            "next_frame_particle_count": len(next_rows),
            "rollout_particle_count": len(rollout_rows),
            "rollout_statuses": {row["stem"]: row["status"] for row in rollout_rows},
        },
    )
    print(f"STABILIZATION COMPLETE {payload['payload_id']} {payload['tag']}", flush=True)


if __name__ == "__main__":
    main()
