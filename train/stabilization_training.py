"""Small, testable primitives used by stabilization fine-tuning."""

from __future__ import annotations

import numpy as np
import torch


def corrupt_intensity_context(context: torch.Tensor, noise_std: float) -> torch.Tensor:
    """Return a noisy copy in standardized-intensity units."""
    if noise_std < 0:
        raise ValueError("noise_std must be nonnegative")
    if noise_std == 0:
        return context.clone()
    return context + torch.randn_like(context) * float(noise_std)


def choose_feedback_frame(
    truth: torch.Tensor,
    prediction: torch.Tensor,
    teacher_forcing_probability: float,
    detach_prediction: bool,
    rng: np.random.Generator,
) -> torch.Tensor:
    """Choose truth or model feedback independently for each batch member."""
    probability = float(teacher_forcing_probability)
    if not 0.0 <= probability <= 1.0:
        raise ValueError("teacher-forcing probability must be within [0, 1]")
    predicted = prediction.detach() if detach_prediction else prediction
    if probability == 1.0:
        return truth
    if probability == 0.0:
        return predicted
    use_truth = torch.as_tensor(
        rng.random(len(truth)) < probability,
        dtype=torch.bool,
        device=truth.device,
    ).reshape(-1, 1, 1, 1)
    return torch.where(use_truth, truth, predicted)
