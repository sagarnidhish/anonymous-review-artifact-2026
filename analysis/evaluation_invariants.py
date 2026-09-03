"""Pure evaluation helpers that enforce shared scientific baselines."""

from __future__ import annotations

import numpy as np


def fixed_persistence(last_observed: np.ndarray, horizon: int) -> np.ndarray:
    """Repeat one observed frame without aliasing model or source memory."""
    if horizon < 0:
        raise ValueError("horizon must be non-negative")
    frame = np.asarray(last_observed)
    return np.repeat(frame[None, ...], horizon, axis=0).copy()


def calibrate_bright_threshold(
    context_frames: np.ndarray,
    mask: np.ndarray,
    percentile: float = 90.0,
) -> float:
    """Calibrate one threshold from observed context pixels only."""
    frames = np.asarray(context_frames)
    roi = np.asarray(mask, dtype=bool)
    if roi.sum() == 0:
        raise ValueError("mask contains no pixels")
    if frames.ndim != 3 or frames.shape[1:] != roi.shape:
        raise ValueError("context_frames and mask shapes are incompatible")
    return float(np.percentile(frames[:, roi], percentile))


def bright_fraction_trajectory(
    frames: np.ndarray,
    mask: np.ndarray,
    bright_threshold: float,
) -> np.ndarray:
    """Fraction of ROI pixels above a previously calibrated threshold."""
    values = np.asarray(frames)
    roi = np.asarray(mask, dtype=bool)
    if roi.sum() == 0:
        raise ValueError("mask contains no pixels")
    if values.ndim != 3 or values.shape[1:] != roi.shape:
        raise ValueError("frames and mask shapes are incompatible")
    return (values[:, roi] >= bright_threshold).mean(axis=1)
