"""Shared initialization for WALRUS autoregressive and persistence states."""

from __future__ import annotations

import numpy as np


def initialize_rollout_state(
    frames: np.ndarray,
    *,
    start: int,
    context_len: int,
) -> tuple[list[np.ndarray], np.ndarray]:
    values = np.asarray(frames)
    stop = start + context_len
    if start < 0 or context_len < 1 or stop > len(values):
        raise ValueError("context exceeds available frames")
    context = [values[index].copy() for index in range(start, stop)]
    persistence_frame = values[stop - 1].copy()
    return context, persistence_frame
