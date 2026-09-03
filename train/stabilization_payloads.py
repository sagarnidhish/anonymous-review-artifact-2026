"""Declared payloads for the focused U-Net rollout-stabilization campaign."""

from __future__ import annotations


COMMON = {
    "source_payload_id": 0,
    "source_tag": "fresh_unet_image_delta_s1337",
    "seed": 1337,
    "context_len": 4,
    "epochs": 12,
    "lr": 1e-4,
    "patience": 4,
    "rollout_steps": 512,
    "report_horizons": [32, 128, 256, 512],
    "anchor_rule": "first_current_sign_change_minus_context",
    "selection_metric": "heldout_free_rollout_mae_h32",
    "validation_horizon": 32,
    "validation_windows_per_stem": 16,
}


def build_payloads() -> list[dict]:
    rows = [
        {
            **COMMON,
            "payload_id": 0,
            "tag": "stabilized_unet_input_noise005_s1337",
            "strategy": "input_noise",
            "training_horizon": 1,
            "noise_std": 0.05,
            "batch_windows": 8,
            "train_windows_per_stem_per_epoch": 512,
            "teacher_forcing_start": 1.0,
            "teacher_forcing_end": 1.0,
            "detach_feedback": True,
        },
        {
            **COMMON,
            "payload_id": 1,
            "tag": "stabilized_unet_scheduled_sampling_h32_s1337",
            "strategy": "scheduled_sampling",
            "training_horizon": 32,
            "noise_std": 0.0,
            "batch_windows": 2,
            "train_windows_per_stem_per_epoch": 128,
            "teacher_forcing_start": 0.9,
            "teacher_forcing_end": 0.0,
            "detach_feedback": True,
        },
        {
            **COMMON,
            "payload_id": 2,
            "tag": "stabilized_unet_recursive_h32_s1337",
            "strategy": "recursive_unroll",
            "training_horizon": 32,
            "noise_std": 0.0,
            "batch_windows": 1,
            "train_windows_per_stem_per_epoch": 128,
            "teacher_forcing_start": 0.0,
            "teacher_forcing_end": 0.0,
            "detach_feedback": False,
        },
    ]
    validate_payloads(rows)
    return rows


def validate_payloads(rows: list[dict]) -> None:
    if len(rows) != 3:
        raise ValueError("expected exactly three stabilization payloads")
    if [row["payload_id"] for row in rows] != [0, 1, 2]:
        raise ValueError("stabilization payload IDs must be consecutive")
    if len({row["tag"] for row in rows}) != len(rows):
        raise ValueError("stabilization tags must be unique")
    if {row["strategy"] for row in rows} != {
        "input_noise",
        "scheduled_sampling",
        "recursive_unroll",
    }:
        raise ValueError("stabilization strategy set is incomplete")


def payload_for_id(payload_id: int) -> dict:
    rows = build_payloads()
    if payload_id < 0 or payload_id >= len(rows):
        raise ValueError(f"payload ID outside 0..{len(rows) - 1}: {payload_id}")
    return dict(rows[payload_id])


def teacher_forcing_probability(payload: dict, epoch: int) -> float:
    """Linear schedule with exact declared values at epochs 1 and N."""
    epochs = int(payload["epochs"])
    if epoch < 1 or epoch > epochs:
        raise ValueError(f"epoch outside 1..{epochs}: {epoch}")
    if epochs == 1:
        return float(payload["teacher_forcing_end"])
    fraction = (epoch - 1) / (epochs - 1)
    start = float(payload["teacher_forcing_start"])
    end = float(payload["teacher_forcing_end"])
    return start + fraction * (end - start)
