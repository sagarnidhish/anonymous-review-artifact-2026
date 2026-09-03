"""Tested data-alignment and aggregation helpers for forcing interventions."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE / "ref"))

import common_sp_baselines as csb  # noqa: E402

from analysis.protocol_timing_diagnostics import (  # noqa: E402
    measured_transition_mask,
    summarize_prediction_subset,
)


ROLLOUT_ANCHOR_RULE = "first_current_sign_change_minus_context"

try:
    from train.protocol_interventions import CONDITIONS, intervene_exogenous
except ModuleNotFoundError:  # direct script execution
    from protocol_interventions import CONDITIONS, intervene_exogenous


def load_sequence(path: Path):
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
        role="test",
        intensity=intensity,
        exogenous_all=exogenous,
        frame_times=frame_times,
        raw_frame_indices=indices,
        sequence_subsample_factor=1,
    )


def with_intervention(sequence, condition: str, seed: int, shift: int):
    return csb.LoadedSequence(
        stem=sequence.stem,
        role=sequence.role,
        intensity=sequence.intensity,
        exogenous_all=intervene_exogenous(
            sequence.exogenous_all, condition, seed=seed, shift=shift
        ),
        frame_times=sequence.frame_times,
        raw_frame_indices=sequence.raw_frame_indices,
        sequence_subsample_factor=sequence.sequence_subsample_factor,
    )


def anchored_slice(sequence, start: int):
    if start < 0 or start >= len(sequence.intensity):
        raise ValueError("anchor start outside sequence")
    return csb.LoadedSequence(
        stem=sequence.stem,
        role=sequence.role,
        intensity=sequence.intensity[start:],
        exogenous_all={
            key: value[start:]
            for key, value in sequence.exogenous_all.items()
        },
        frame_times=sequence.frame_times[start:],
        raw_frame_indices=sequence.raw_frame_indices[start:],
        sequence_subsample_factor=sequence.sequence_subsample_factor,
    )


def first_current_transition_slice(sequence, context_len: int):
    """Slice four context frames before the first current-sign transition."""
    if context_len < 1:
        raise ValueError("context_len must be positive")
    changes = csb.current_sign_change_indices(
        sequence.exogenous_all["current"]
    )
    if len(changes) == 0:
        raise ValueError("sequence has no current-sign transition")
    onset = int(changes[0])
    anchor_start = max(0, onset - int(context_len))
    return anchor_start, onset, anchored_slice(sequence, anchor_start)


def response_l1(prediction: np.ndarray, reference: np.ndarray) -> float:
    if prediction.shape != reference.shape:
        raise ValueError("prediction and reference shapes differ")
    return float(np.abs(prediction - reference).mean())


def delay_condition_specs(delays: tuple[int, ...] | list[int]) -> list[dict]:
    """Declare measured, zero, positive-delay, and shuffle interventions."""
    parsed = [int(value) for value in delays]
    if any(value < 0 for value in parsed):
        raise ValueError("delay values must be non-negative")
    positive = [value for value in parsed if value > 0]
    if len(set(positive)) != len(positive):
        raise ValueError("positive delay values must be unique")
    specs = [
        {"label": "measured", "intervention": "true", "shift_frames": 0},
        {"label": "zero", "intervention": "zero", "shift_frames": 0},
    ]
    specs.extend(
        {
            "label": f"delay_{value}",
            "intervention": "shift",
            "shift_frames": value,
        }
        for value in positive
    )
    specs.append(
        {"label": "shuffle", "intervention": "shuffle", "shift_frames": 0}
    )
    return specs


def fixed_transition_summaries(
    pred: np.ndarray,
    naive: np.ndarray,
    targets: np.ndarray,
    *,
    measured_current: np.ndarray,
    target_steps: np.ndarray,
    radius: int,
) -> dict:
    """Summarize all interventions on one measured-current transition mask."""
    selector = measured_transition_mask(
        measured_current, target_steps, radius=radius
    )
    return {
        "transition_radius": int(radius),
        "transition": summarize_prediction_subset(
            pred, naive, targets, selector
        ),
        "nontransition": summarize_prediction_subset(
            pred, naive, targets, ~selector
        ),
    }


def aggregate_rows(rows: list[dict]) -> list[dict]:
    output = []
    for mode in ("next_frame", "rollout_anchored"):
        for condition in CONDITIONS:
            selected = [
                row for row in rows
                if row["mode"] == mode and row["condition"] == condition
            ]
            if not selected:
                continue
            output.append({
                "mode": mode,
                "condition": condition,
                "n_particles": len(selected),
                "mean_model_mae": float(
                    np.mean([row["model_mae"] for row in selected])
                ),
                "mean_naive_mae": float(
                    np.mean([row["naive_mae"] for row in selected])
                ),
                "mean_mae_ratio": float(
                    np.mean([row["mae_ratio"] for row in selected])
                ),
                "mean_prediction_change_l1": float(
                    np.mean([
                        row["prediction_change_l1"] for row in selected
                    ])
                ),
            })
    return output
