"""Autoregressive rollout with fixed persistence and explicit divergence."""

from __future__ import annotations

import numpy as np
import torch

try:
    from train.ref.common_sp_baselines import build_context_tensor
except ModuleNotFoundError:  # direct execution from train/
    from ref.common_sp_baselines import build_context_tensor


@torch.no_grad()
def run_safe_rollout(
    *,
    model,
    sequence,
    active_fields,
    context_len: int,
    device: torch.device,
    predict_delta: bool,
    max_rollout_steps: int,
) -> dict:
    """Run a rollout and retain the target/persistence record after overflow.

    Once a model prediction becomes non-finite, subsequent model frames are
    stored as NaN. Ground truth and the single fixed persistence frame remain
    complete, so numerical divergence is evidence rather than a missing run.
    """
    total_steps = min(
        int(max_rollout_steps), len(sequence.intensity) - int(context_len)
    )
    if context_len < 1 or total_steps < 1:
        raise ValueError("sequence does not contain a valid rollout")

    pred_context = sequence.intensity[:context_len].astype(np.float32).copy()
    persistence_frame = pred_context[-1].copy()
    height, width = persistence_frame.shape
    preds = np.full((total_steps, height, width), np.nan, dtype=np.float32)
    targets = sequence.intensity[
        context_len : context_len + total_steps
    ].astype(np.float32)
    naives = np.repeat(persistence_frame[None], total_steps, axis=0)
    target_steps = np.arange(context_len, context_len + total_steps, dtype=np.int64)
    frame_times = sequence.frame_times[target_steps].astype(np.float32)

    first_nonfinite_step = None
    model.eval()
    for offset, target_index in enumerate(target_steps):
        context_indices = list(range(int(target_index) - context_len, int(target_index)))
        x_seq = build_context_tensor(
            sequence=sequence,
            intensity_context=pred_context,
            context_indices=context_indices,
            active_fields=active_fields,
        )
        x = torch.from_numpy(x_seq[None]).to(device, non_blocking=True)
        pred_raw = model(x).detach().cpu().numpy()[0, 0].astype(np.float32)
        pred_next = pred_context[-1] + pred_raw if predict_delta else pred_raw
        if not np.isfinite(pred_next).all():
            first_nonfinite_step = offset + 1
            break
        preds[offset] = pred_next
        if offset + 1 < total_steps:
            pred_context = np.concatenate(
                [pred_context[1:], pred_next[None]], axis=0
            )

    return {
        "pred": preds,
        "naive": naives,
        "targets": targets,
        "target_steps": target_steps,
        "frame_times": frame_times,
        "status": (
            "complete" if first_nonfinite_step is None else "numerically_diverged"
        ),
        "first_nonfinite_step": first_nonfinite_step,
    }


def cumulative_horizon_metrics(
    pred: np.ndarray,
    naive: np.ndarray,
    targets: np.ndarray,
    *,
    horizons=(32, 128, 256, 512),
    first_nonfinite_step: int | None = None,
) -> dict:
    if pred.shape != naive.shape or pred.shape != targets.shape:
        raise ValueError("prediction, persistence, and target arrays must align")
    output = {}
    for horizon in horizons:
        horizon = int(horizon)
        if horizon < 1 or horizon > len(targets):
            raise ValueError(f"horizon {horizon} outside rollout length {len(targets)}")
        naive_mae = float(np.mean(np.abs(naive[:horizon] - targets[:horizon])))
        diverged = (
            first_nonfinite_step is not None
            and int(first_nonfinite_step) <= horizon
        )
        if diverged:
            model_mae = float("inf")
            ratio = float("inf")
        else:
            model_mae = float(np.mean(np.abs(pred[:horizon] - targets[:horizon])))
            ratio = model_mae / max(naive_mae, 1e-12)
        output[str(horizon)] = {
            "model_mae": model_mae,
            "naive_mae": naive_mae,
            "mae_ratio": ratio,
        }
    return output
