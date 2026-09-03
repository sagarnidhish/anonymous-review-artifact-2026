#!/usr/bin/env python3
"""Train one payload of the fresh 24-configuration GRA29 benchmark."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "ref"))

import common_sp_baselines as csb  # noqa: E402
import run_sp_baseline_study as rsb  # noqa: E402
from matched_grid_payloads import payload_for_id  # noqa: E402
from matched_grid_artifacts import (  # noqa: E402
    COMPACT_NEXT_FRAME_MODE,
    write_compact_next_frame_artifact,
)
from models import build_model  # noqa: E402
from protocol_evaluation_utils import (  # noqa: E402
    ROLLOUT_ANCHOR_RULE,
    first_current_transition_slice,
)
from safe_anchored_rollout import (  # noqa: E402
    cumulative_horizon_metrics,
    run_safe_rollout,
)


TRAIN_STEMS = tuple(f"GRA29_C20_25deg_particle{i}" for i in range(1, 5))
TEST_STEMS = tuple(f"GRA29_C20_45deg_particle{i}" for i in range(1, 5))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload-id", type=int, required=True)
    parser.add_argument("--arrays-dir", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-train-windows-per-stem", type=int, default=3000)
    parser.add_argument("--max-eval-windows", type=int, default=3000)
    parser.add_argument("--source-commit", default="unknown")
    return parser.parse_args()


def load_sequence(arrays_dir: Path, stem: str, role: str):
    path = arrays_dir / f"{stem}.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path) as data:
        required = {"intensity", "voltage", "current", "time_norm", "frame_times"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"{path} missing arrays: {sorted(missing)}")
        intensity = data["intensity"].astype(np.float32)
        exogenous = {
            key: data[key].astype(np.float32)
            for key in ("voltage", "current", "time_norm")
        }
        frame_times = data["frame_times"].astype(np.float32)
    if len(intensity) != len(frame_times):
        raise ValueError(f"frame/time length mismatch in {path}")
    indices = np.arange(len(intensity), dtype=np.int64)
    return csb.LoadedSequence(
        stem=stem,
        role=role,
        intensity=intensity,
        exogenous_all=exogenous,
        frame_times=frame_times,
        raw_frame_indices=indices,
        sequence_subsample_factor=1,
    )


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


def finite_summary(row: dict) -> None:
    for key in ("model_mae", "naive_mae", "mae_ratio"):
        if not np.isfinite(row[key]):
            raise RuntimeError(f"non-finite next-frame {key} for {row['stem']}")


def main():
    args = parse_args()
    payload = payload_for_id(args.payload_id)
    arrays_dir = Path(args.arrays_dir).resolve()
    output_dir = Path(args.out_root).resolve() / payload["tag"]
    completion_path = output_dir / "completion_manifest.json"
    if completion_path.is_file():
        raise RuntimeError(f"complete payload already exists: {completion_path}")

    seed = int(payload["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the matched benchmark")

    active_fields = (
        ["intensity"]
        if payload["input_mode"] == "image_only"
        else ["intensity", "voltage", "current", "time_norm"]
    )
    predict_delta = payload["target_mode"] == "delta"
    context_len = int(payload["context_len"])
    train_sequences = [load_sequence(arrays_dir, stem, "train") for stem in TRAIN_STEMS]
    test_sequences = [load_sequence(arrays_dir, stem, "test") for stem in TEST_STEMS]

    train_index, val_index = [], []
    split_manifest = {}
    for sequence_index, sequence in enumerate(train_sequences):
        starts = csb.build_window_starts(
            len(sequence.intensity),
            context_len,
            1,
            args.max_train_windows_per_stem,
        )
        train_starts, val_starts = csb.split_train_val_starts(starts, 0.1)
        train_index.extend((sequence_index, start) for start in train_starts)
        val_index.extend((sequence_index, start) for start in val_starts)
        split_manifest[sequence.stem] = {
            "all_start_min": int(starts[0]),
            "all_start_max": int(starts[-1]),
            "train_count": len(train_starts),
            "train_start_max": int(train_starts[-1]),
            "validation_count": len(val_starts),
            "validation_start_min": int(val_starts[0]),
        }

    def dataset(indices):
        return csb.BaselineWindowDataset(
            train_sequences,
            indices,
            active_fields=active_fields,
            context_len=context_len,
            window_stride=1,
            predict_delta=predict_delta,
        )

    train_loader = DataLoader(
        dataset(train_index),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        dataset(val_index),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    model = build_model(
        model_family=payload["model_family"],
        in_fields=len(active_fields),
        context_len=context_len,
        base_channels=32,
        hidden_layers=2,
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    model, history, best = rsb.train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        predict_delta=predict_delta,
        grad_clip_norm=1.0,
    )

    model_dir = output_dir / "models"
    next_dir = output_dir / "next_frame"
    rollout_dir = output_dir / "rollout_anchored"
    model_dir.mkdir(parents=True, exist_ok=True)
    next_dir.mkdir(parents=True, exist_ok=True)
    rollout_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = model_dir / f"{payload['model_family']}_best.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "payload": payload,
            "active_fields": active_fields,
            "prediction_form": (
                "delta_from_last_frame" if predict_delta else "direct_next_frame"
            ),
            "base_channels": 32,
            "hidden_layers": 2,
            "parameter_count": parameter_count,
            "source_commit": args.source_commit,
            "train_stems": list(TRAIN_STEMS),
            "test_stems": list(TEST_STEMS),
            "normalization": "archived_per_movie_full_record",
        },
        checkpoint_path,
    )
    strict_json_dump(
        model_dir / "training_history.json",
        {
            "payload": payload,
            "arguments": vars(args),
            "best": best,
            "history": history,
            "parameter_count": parameter_count,
            "active_fields": active_fields,
            "split_windows": split_manifest,
            "next_frame_artifact_mode": COMPACT_NEXT_FRAME_MODE,
            "normalization": "archived_per_movie_full_record",
        },
    )

    next_rows = []
    rollout_rows = []
    for sequence in test_sequences:
        pred, naive, targets, target_steps, frame_times = (
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
        row = {
            "stem": sequence.stem,
            "role": "test",
            "model_family": payload["model_family"],
            "tag": payload["tag"],
            "payload_id": payload["payload_id"],
            "active_fields": active_fields,
            "context_len": context_len,
            "prediction_form": (
                "delta_from_last_frame" if predict_delta else "direct_next_frame"
            ),
            **rsb.summarize_prediction_result(
                sequence=sequence,
                pred=pred,
                naive=naive,
                targets=targets,
                target_steps=target_steps,
                reversal_radius=5,
            ),
        }
        finite_summary(row)
        next_rows.append(row)
        next_stem_dir = next_dir / sequence.stem
        write_compact_next_frame_artifact(
            next_stem_dir / f"preds_{sequence.stem}.npz",
            pred=pred,
            naive=naive,
            targets=targets,
            target_steps=target_steps,
            frame_times=frame_times,
            metadata={
                "active_fields": np.asarray(row["active_fields"]),
                "model_family": row["model_family"],
                "tag": row["tag"],
                "prediction_form": row["prediction_form"],
                "model_mae": np.asarray(row["model_mae"], dtype=np.float32),
                "naive_mae": np.asarray(row["naive_mae"], dtype=np.float32),
                "mae_ratio": np.asarray(row["mae_ratio"], dtype=np.float32),
                "model_mse": np.asarray(row["model_mse"], dtype=np.float32),
                "naive_mse": np.asarray(row["naive_mse"], dtype=np.float32),
                "mse_ratio": np.asarray(row["mse_ratio"], dtype=np.float32),
            },
        )
        strict_json_dump(next_stem_dir / "next_frame_results.json", [row])
        strict_json_dump(
            next_stem_dir / "study_manifest.json",
            {
                "mode": "next_frame",
                "artifact_mode": COMPACT_NEXT_FRAME_MODE,
                "stem": row["stem"],
                "model_family": row["model_family"],
                "tag": row["tag"],
                "active_fields": row["active_fields"],
                "prediction_form": row["prediction_form"],
                "evaluation_scope": "full_frame_metrics_with_sampled_images",
                "source_commit": args.source_commit,
            },
        )

        anchor_start, onset, anchored = first_current_transition_slice(
            sequence, context_len=context_len
        )
        result = run_safe_rollout(
            model=model,
            sequence=anchored,
            active_fields=active_fields,
            context_len=context_len,
            device=device,
            predict_delta=predict_delta,
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
            "model_family": payload["model_family"],
            "tag": payload["tag"],
            "payload_id": payload["payload_id"],
            "active_fields": active_fields,
            "prediction_form": row["prediction_form"],
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
                -1 if result["first_nonfinite_step"] is None
                else result["first_nonfinite_step"],
                dtype=np.int64,
            ),
        )
        strict_json_dump(stem_dir / "rollout_results.json", [rollout_row])
        strict_json_dump(
            stem_dir / "study_manifest.json",
            {
                **payload,
                "stem": sequence.stem,
                "active_fields": active_fields,
                "prediction_form": row["prediction_form"],
                "anchor_frame": int(anchor_start),
                "onset_frame": int(onset),
                "status": result["status"],
                "first_nonfinite_step": result["first_nonfinite_step"],
                "source_commit": args.source_commit,
            },
        )
        print(
            f"[evaluation] {sequence.stem} next={row['mae_ratio']:.4f} "
            f"rollout_status={result['status']} rollout512={full['mae_ratio']}",
            flush=True,
        )

    strict_json_dump(next_dir / "next_frame_results.json", next_rows)
    strict_json_dump(rollout_dir / "rollout_results.json", rollout_rows)
    strict_json_dump(
        completion_path,
        {
            "status": "complete",
            "payload": payload,
            "source_commit": args.source_commit,
            "checkpoint": str(checkpoint_path.relative_to(output_dir)),
            "next_frame_particle_count": len(next_rows),
            "rollout_particle_count": len(rollout_rows),
            "rollout_statuses": {
                row["stem"]: row["status"] for row in rollout_rows
            },
            "next_frame_artifact_mode": COMPACT_NEXT_FRAME_MODE,
            "normalization": "archived_per_movie_full_record",
        },
    )
    print(f"PAYLOAD COMPLETE {payload['payload_id']} {payload['tag']}", flush=True)


if __name__ == "__main__":
    main()
