"""Intrinsic-dynamics and masked/full-frame rollout diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _mae(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.mean(np.abs(first - second)))


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else float("nan")


def evaluate_rollout_scale(
    targets: np.ndarray,
    naive: np.ndarray,
    pred: np.ndarray,
    mask: np.ndarray,
    *,
    horizons: tuple[int, ...] | list[int],
) -> dict:
    """Measure endpoint and cumulative error at declared one-based horizons."""
    truth = np.asarray(targets)
    persistence = np.asarray(naive)
    prediction = np.asarray(pred)
    particle_mask = np.asarray(mask, dtype=bool)
    if truth.shape != persistence.shape or truth.shape != prediction.shape:
        raise ValueError("target, persistence, and prediction shapes differ")
    if truth.ndim != 3 or particle_mask.shape != truth.shape[1:]:
        raise ValueError("rollouts must be T x H x W with an H x W mask")
    if not np.any(particle_mask):
        raise ValueError("particle mask is empty")
    requested = tuple(int(value) for value in horizons)
    if not requested or any(value < 1 or value > len(truth) for value in requested):
        raise ValueError("one-based horizon falls outside the rollout")

    output = {
        "n_steps": int(len(truth)),
        "mask_fraction": float(np.mean(particle_mask)),
        "horizons": {},
    }
    for horizon in requested:
        index = horizon - 1
        persistence_endpoint_full = _mae(persistence[index], truth[index])
        persistence_endpoint_masked = _mae(
            persistence[index][particle_mask], truth[index][particle_mask]
        )
        model_endpoint_full = _mae(prediction[index], truth[index])
        model_endpoint_masked = _mae(
            prediction[index][particle_mask], truth[index][particle_mask]
        )
        persistence_cumulative_full = _mae(
            persistence[:horizon], truth[:horizon]
        )
        persistence_cumulative_masked = _mae(
            persistence[:horizon, particle_mask],
            truth[:horizon, particle_mask],
        )
        model_cumulative_full = _mae(prediction[:horizon], truth[:horizon])
        model_cumulative_masked = _mae(
            prediction[:horizon, particle_mask], truth[:horizon, particle_mask]
        )
        output["horizons"][str(horizon)] = {
            "truth_displacement_full": persistence_endpoint_full,
            "truth_displacement_masked": persistence_endpoint_masked,
            "endpoint_model_mae_full": model_endpoint_full,
            "endpoint_persistence_mae_full": persistence_endpoint_full,
            "endpoint_mae_ratio_full": _ratio(
                model_endpoint_full, persistence_endpoint_full
            ),
            "endpoint_model_mae_masked": model_endpoint_masked,
            "endpoint_persistence_mae_masked": persistence_endpoint_masked,
            "endpoint_mae_ratio_masked": _ratio(
                model_endpoint_masked, persistence_endpoint_masked
            ),
            "cumulative_model_mae_full": model_cumulative_full,
            "cumulative_persistence_mae_full": persistence_cumulative_full,
            "cumulative_mae_ratio_full": _ratio(
                model_cumulative_full, persistence_cumulative_full
            ),
            "cumulative_model_mae_masked": model_cumulative_masked,
            "cumulative_persistence_mae_masked": persistence_cumulative_masked,
            "cumulative_mae_ratio_masked": _ratio(
                model_cumulative_masked, persistence_cumulative_masked
            ),
        }
    return output


def latex_table(payload: dict) -> str:
    """Render full/masked horizon diagnostics without hiding definitions."""
    lines = [
        r"\begin{tabular}{@{}lrrrrrrr@{}}",
        r"\toprule",
        r"& & \multicolumn{2}{c}{Truth displacement} & "
        r"\multicolumn{2}{c}{U-Net ratio} & "
        r"\multicolumn{2}{c}{RFT ratio} \\",
        r"Particle & Step & full & mask & full & mask & full & mask \\",
        r"\midrule",
    ]
    for particle in payload["particles"]:
        first = True
        for horizon in payload["horizons"]:
            unet = particle["models"]["unet"]["horizons"][str(horizon)]
            rft = particle["models"]["rft"]["horizons"][str(horizon)]
            label = f"p{particle['particle']}" if first else ""
            if first:
                label += f" (Mask area {100 * particle['mask_fraction']:.1f}\\%)"
            first = False
            lines.append(
                f"{label} & {horizon} & "
                f"{unet['truth_displacement_full']:.3f} & "
                f"{unet['truth_displacement_masked']:.3f} & "
                f"{unet['cumulative_mae_ratio_full']:.2f} & "
                f"{unet['cumulative_mae_ratio_masked']:.2f} & "
                f"{rft['cumulative_mae_ratio_full']:.2f} & "
                f"{rft['cumulative_mae_ratio_masked']:.2f} \\\\"
            )
        if particle is not payload["particles"][-1]:
            lines.append(r"\addlinespace")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def _artifact_path(root: Path, tag: str, stem: str) -> Path:
    candidates = sorted(
        (root / "results" / "out" / tag / "rollout_anchored" / stem).glob(
            "rollout_*.npz"
        )
    )
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"expected one anchored rollout for {tag}/{stem}; found {candidates}"
        )
    return candidates[0]


def repository_payload(root: Path, horizons: tuple[int, ...]) -> dict:
    """Evaluate the two aligned U-Net rollouts for all four test particles."""
    try:
        from analysis.physics_metrics import detect_roi
    except ModuleNotFoundError:
        from physics_metrics import detect_roi

    tags = {
        "unet": "unet_image_only_delta_anchored",
        "rft": "unet_image_only_delta_rft_anchored",
    }
    particles = []
    for particle in range(1, 5):
        stem = f"GRA29_C20_45deg_particle{particle}"
        source_path = root / "data_prep" / "arrays" / f"{stem}.npz"
        with np.load(source_path, allow_pickle=False) as loaded:
            intensity = loaded["intensity"].astype(np.float32)
        mask = detect_roi(intensity)
        del intensity
        model_payload = {}
        reference_targets = None
        reference_naive = None
        for label, tag in tags.items():
            with np.load(
                _artifact_path(root, tag, stem), allow_pickle=False
            ) as loaded:
                pred = loaded["pred"].astype(np.float32)
                naive = loaded["naive"].astype(np.float32)
                targets = loaded["targets"].astype(np.float32)
            if reference_targets is None:
                reference_targets = targets
                reference_naive = naive
            elif not np.allclose(targets, reference_targets, atol=5e-4, rtol=0):
                raise ValueError(f"target mismatch between aligned models: {stem}")
            elif not np.allclose(naive, reference_naive, atol=5e-6, rtol=0):
                raise ValueError(f"persistence mismatch between aligned models: {stem}")
            model_payload[label] = evaluate_rollout_scale(
                targets, naive, pred, mask, horizons=horizons
            )
        particles.append(
            {
                "particle": particle,
                "stem": stem,
                "mask_fraction": float(mask.mean()),
                "models": model_payload,
            }
        )
    return {
        "description": (
            "Endpoint truth displacement and cumulative model/persistence "
            "MAE on full frames and frozen ground-truth-derived masks"
        ),
        "normalization": "per-movie full-record standardized intensity",
        "aggregation_unit": "particle_movie",
        "horizons": list(horizons),
        "particles": particles,
    }


def main() -> None:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=default_root)
    parser.add_argument(
        "--output",
        type=Path,
        default=default_root / "artifacts" / "rollout_scale_diagnostics.json",
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=default_root / "paper" / "tables" / "rollout_scale.tex",
    )
    parser.add_argument("--horizons", default="1,32,128,256,512")
    args = parser.parse_args()
    horizons = tuple(int(value) for value in args.horizons.split(","))
    payload = repository_payload(args.root.resolve(), horizons)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text(latex_table(payload))
    print(f"wrote {args.output}")
    print(f"wrote {args.table}")


if __name__ == "__main__":
    main()
