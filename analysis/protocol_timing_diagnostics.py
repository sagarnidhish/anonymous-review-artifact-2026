"""Pure diagnostics for calibrated protocol-timing interventions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _finite_same_shape(first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    if left.shape != right.shape or left.size == 0:
        raise ValueError("protocol arrays must be nonempty and have equal shape")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("protocol arrays must contain only finite values")
    return left, right


def channel_perturbation(
    measured: np.ndarray, modified: np.ndarray
) -> dict[str, float]:
    """Describe an intervention in the normalized units seen by the model."""
    measured_values, modified_values = _finite_same_shape(measured, modified)
    mean_abs_difference = float(
        np.mean(np.abs(measured_values - modified_values))
    )
    sigma = float(np.std(measured_values))
    if sigma > 0:
        difference_over_sigma = mean_abs_difference / sigma
    else:
        difference_over_sigma = 0.0 if mean_abs_difference == 0 else float("inf")
    modified_sigma = float(np.std(modified_values))
    if sigma > 0 and modified_sigma > 0:
        correlation = float(
            np.corrcoef(measured_values.ravel(), modified_values.ravel())[0, 1]
        )
    else:
        correlation = float("nan")
    return {
        "mean_abs_difference": mean_abs_difference,
        "measured_sigma": sigma,
        "difference_over_sigma": float(difference_over_sigma),
        "correlation": correlation,
    }


def model_facing_channel_windows(
    values: np.ndarray, *, context_len: int, max_windows: int
) -> np.ndarray:
    """Return the stride-one channel contexts passed to next-frame inference."""
    channel = np.asarray(values)
    if channel.ndim != 1:
        raise ValueError("protocol channel must be one-dimensional")
    if context_len < 1 or max_windows < 1:
        raise ValueError("context_len and max_windows must be positive")
    available = len(channel) - context_len
    if available < 1:
        raise ValueError("protocol channel is shorter than one context/target")
    count = min(int(max_windows), available)
    return np.lib.stride_tricks.sliding_window_view(
        channel, int(context_len)
    )[:count].copy()


def _current_sign_change_indices(current: np.ndarray) -> np.ndarray:
    signs = np.sign(np.asarray(current, dtype=np.float64))
    changes: list[int] = []
    previous_nonzero = 0.0
    for index, sign in enumerate(signs):
        if sign != 0:
            if previous_nonzero != 0 and sign != previous_nonzero:
                changes.append(index)
            previous_nonzero = sign
    return np.asarray(changes, dtype=np.int64)


def measured_transition_mask(
    measured_current: np.ndarray,
    target_steps: np.ndarray,
    *,
    radius: int,
) -> np.ndarray:
    """Select targets near transitions defined once from measured current."""
    current = np.asarray(measured_current)
    steps = np.asarray(target_steps, dtype=np.int64)
    if current.ndim != 1 or steps.ndim != 1:
        raise ValueError("current and target_steps must be one-dimensional")
    if radius < 0:
        raise ValueError("transition radius must be non-negative")
    if len(steps) and (steps.min() < 0 or steps.max() >= len(current)):
        raise ValueError("target_steps fall outside measured current")
    changes = _current_sign_change_indices(current)
    if not len(changes) or not len(steps):
        return np.zeros(len(steps), dtype=bool)
    return np.any(
        np.abs(steps[:, None] - changes[None, :]) <= int(radius), axis=1
    )


def summarize_prediction_subset(
    pred: np.ndarray,
    naive: np.ndarray,
    targets: np.ndarray,
    frame_mask: np.ndarray,
) -> dict[str, float | int]:
    """Compute frame metrics on one externally supplied frame selector."""
    prediction = np.asarray(pred)
    persistence = np.asarray(naive)
    truth = np.asarray(targets)
    selector = np.asarray(frame_mask, dtype=bool)
    if prediction.shape != persistence.shape or prediction.shape != truth.shape:
        raise ValueError("prediction, persistence, and target shapes differ")
    if prediction.ndim < 1 or selector.shape != (len(prediction),):
        raise ValueError("frame mask must select the first array dimension")
    if not np.any(selector):
        return {
            "N": 0,
            "model_mae": float("nan"),
            "naive_mae": float("nan"),
            "mae_ratio": float("nan"),
        }
    model_mae = float(np.mean(np.abs(prediction[selector] - truth[selector])))
    naive_mae = float(np.mean(np.abs(persistence[selector] - truth[selector])))
    return {
        "N": int(selector.sum()),
        "model_mae": model_mae,
        "naive_mae": naive_mae,
        "mae_ratio": model_mae / naive_mae if naive_mae > 0 else float("nan"),
    }


def protocol_input_payload(
    arrays_dir: Path,
    *,
    delays: tuple[int, ...],
    context_len: int,
    max_windows: int,
    seed: int,
) -> dict:
    """Calibrate delay and shuffle magnitudes on exact model-facing contexts."""
    from train.protocol_interventions import intervene_exogenous

    particles = []
    for particle in range(1, 5):
        stem = f"GRA29_C20_45deg_particle{particle}"
        with np.load(arrays_dir / f"{stem}.npz", allow_pickle=False) as loaded:
            exogenous = {
                key: loaded[key].astype(np.float32)
                for key in ("voltage", "current", "time_norm")
            }
        measured_windows = {
            key: model_facing_channel_windows(
                exogenous[key],
                context_len=context_len,
                max_windows=max_windows,
            )
            for key in ("voltage", "current")
        }
        interventions = {}
        for label, condition, shift in (
            [(f"delay_{delay}", "shift", delay) for delay in delays]
            + [("shuffle", "shuffle", 0)]
        ):
            modified = intervene_exogenous(
                exogenous, condition, seed=seed, shift=shift
            )
            interventions[label] = {
                key: channel_perturbation(
                    measured_windows[key],
                    model_facing_channel_windows(
                        modified[key],
                        context_len=context_len,
                        max_windows=max_windows,
                    ),
                )
                for key in ("voltage", "current")
            }
        particles.append(
            {
                "particle": particle,
                "stem": stem,
                "interventions": interventions,
            }
        )
    return {
        "description": (
            "Protocol perturbations on exact stride-one four-frame input "
            "contexts used for next-frame evaluation"
        ),
        "normalization": "model-facing per-movie normalized channels",
        "protocol_unit": "one shared cell-level trace",
        "particle_movies": 4,
        "context_len": context_len,
        "max_eval_windows": max_windows,
        "delays_frames": list(delays),
        "seed": seed,
        "particles": particles,
    }


def latex_input_table(payload: dict) -> str:
    first = payload["particles"][0]["interventions"]
    labels = [f"delay_{value}" for value in payload["delays_frames"]] + [
        "shuffle"
    ]
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"& \multicolumn{3}{c}{Voltage} & \multicolumn{3}{c}{Current} \\",
        r"Input & mean $|\Delta|$ & $|\Delta|/\sigma$ & corr. & "
        r"mean $|\Delta|$ & $|\Delta|/\sigma$ & corr. \\",
        r"\midrule",
    ]
    for label in labels:
        voltage = first[label]["voltage"]
        current = first[label]["current"]
        display = (
            "shuffle"
            if label == "shuffle"
            else f"delay {int(label.split('_')[1])} frames"
        )
        lines.append(
            f"{display} & {voltage['mean_abs_difference']:.3f} & "
            f"{voltage['difference_over_sigma']:.3f} & "
            f"{voltage['correlation']:.3f} & "
            f"{current['mean_abs_difference']:.3f} & "
            f"{current['difference_over_sigma']:.3f} & "
            f"{current['correlation']:.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def main() -> None:
    root = ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arrays-dir", type=Path, default=root / "data_prep" / "arrays"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "artifacts" / "protocol_input_diagnostics.json",
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=root / "paper" / "tables" / "protocol_input_diagnostics.tex",
    )
    parser.add_argument("--delays", default="0,16,32,64,128,256,512")
    parser.add_argument("--context-len", type=int, default=4)
    parser.add_argument("--max-windows", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    delays = tuple(int(value) for value in args.delays.split(","))
    payload = protocol_input_payload(
        args.arrays_dir,
        delays=delays,
        context_len=args.context_len,
        max_windows=args.max_windows,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text(latex_input_table(payload))
    print(f"wrote {args.output}")
    print(f"wrote {args.table}")


if __name__ == "__main__":
    main()
