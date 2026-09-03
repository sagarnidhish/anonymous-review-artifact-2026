"""Declared payload grid for the fresh paired-temperature benchmark."""

from __future__ import annotations

from itertools import product


FAMILIES = (
    "unet",
    "convlstm",
    "simvp",
    "residual_cnn",
    "predrnn",
    "predrnnpp",
)
INPUT_MODES = ("image_only", "protocol_conditioned")
TARGET_MODES = ("delta", "direct")
REPORT_HORIZONS = (32, 128, 256, 512)
ANCHOR_RULE = "first_current_sign_change_minus_context"


def build_payloads() -> list[dict]:
    payloads = []
    for payload_id, (family, input_mode, target_mode) in enumerate(
        product(FAMILIES, INPUT_MODES, TARGET_MODES)
    ):
        short_input = "image" if input_mode == "image_only" else "protocol"
        payloads.append(
            {
                "payload_id": payload_id,
                "model_family": family,
                "input_mode": input_mode,
                "target_mode": target_mode,
                "tag": f"fresh_{family}_{short_input}_{target_mode}_s1337",
                "seed": 1337,
                "context_len": 4,
                "rollout_steps": 512,
                "report_horizons": list(REPORT_HORIZONS),
                "anchor_rule": ANCHOR_RULE,
            }
        )
    validate_payloads(payloads)
    return payloads


def validate_payloads(payloads: list[dict]) -> None:
    if len(payloads) != 24:
        raise ValueError(f"expected 24 payloads, found {len(payloads)}")
    ids = [int(row["payload_id"]) for row in payloads]
    if ids != list(range(24)):
        raise ValueError("payload IDs must be unique consecutive integers 0..23")
    combinations = [
        (row["model_family"], row["input_mode"], row["target_mode"])
        for row in payloads
    ]
    expected = list(product(FAMILIES, INPUT_MODES, TARGET_MODES))
    if combinations != expected:
        raise ValueError("payload combinations do not match the declared grid")
    tags = [row["tag"] for row in payloads]
    if len(set(tags)) != len(tags):
        raise ValueError("payload tags must be unique")


def payload_for_id(payload_id: int) -> dict:
    payloads = build_payloads()
    if payload_id < 0 or payload_id >= len(payloads):
        raise ValueError(f"payload_id must be in 0..{len(payloads) - 1}")
    return dict(payloads[payload_id])
