"""Pure payload construction for the paired-particle identity experiment."""

from __future__ import annotations


MODEL_FAMILIES = ("unet", "predrnnpp")
PARTICLES = (1, 2, 3, 4)


def _payload(model_family: str, particle: int, *, pilot: bool) -> dict:
    suffix = "_pilot" if pilot else ""
    cfg = {
        "model_family": model_family,
        "tag": f"identity_holdout_{model_family}_p{particle}{suffix}",
        "use_voltage": False,
        "use_current": False,
        "use_time_norm": False,
        "predict_delta": True,
        "target_size": 128,
        "context_len": 4,
        "window_stride": 1,
        "sequence_subsample_factor": 1,
        "epochs": 1 if pilot else 60,
        "batch_size": 8,
        "lr": 1e-3,
        "patience": 1 if pilot else 10,
        "val_fraction": 0.1,
        "base_channels": 32,
        "hidden_layers": 2,
        "max_train_windows_per_stem": 300 if pilot else 3000,
        "max_eval_windows": 300 if pilot else 3000,
        "max_rollout_steps": 32 if pilot else 512,
        "reversal_radius": 5,
        "grad_clip_norm": 1.0,
        "seed": 1337,
        "split": "identity_holdout",
        "heldout_particle": particle,
    }
    return {"mode": "train", "cfg": cfg}


def build_payloads(pilot: bool = False) -> list[dict]:
    """Return the two pilot jobs or the complete eight-job experiment grid."""
    particles = (1,) if pilot else PARTICLES
    return [
        _payload(model_family, particle, pilot=pilot)
        for particle in particles
        for model_family in MODEL_FAMILIES
    ]
