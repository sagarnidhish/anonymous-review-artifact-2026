"""Shared four-tap linear autoregressive baseline for GRA29 movies."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _design(contexts: np.ndarray) -> np.ndarray:
    values = np.asarray(contexts, dtype=np.float64)
    if values.ndim != 4 or values.shape[1] != 4:
        raise ValueError("contexts must have shape N x 4 x H x W")
    lag_values = values.transpose(0, 2, 3, 1).reshape(-1, 4)
    return np.column_stack(
        [lag_values, np.ones(len(lag_values), dtype=np.float64)]
    )


def accumulate_sufficient_statistics(
    frames: np.ndarray,
    starts: np.ndarray | list[int],
    *,
    chunk_size: int = 32,
) -> dict[str, np.ndarray | int]:
    """Accumulate X'X and X'y without materializing all pixel-time samples."""
    movie = np.asarray(frames, dtype=np.float64)
    indices = np.asarray(starts, dtype=np.int64)
    if movie.ndim != 3 or indices.ndim != 1 or not len(indices):
        raise ValueError("frames must be T x H x W and starts must be nonempty")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if indices.min() < 0 or indices.max() + 4 >= len(movie):
        raise ValueError("a four-frame context or target falls outside the movie")
    xtx = np.zeros((5, 5), dtype=np.float64)
    xty = np.zeros(5, dtype=np.float64)
    n_samples = 0
    for offset in range(0, len(indices), int(chunk_size)):
        chunk = indices[offset : offset + int(chunk_size)]
        contexts = np.stack([movie[start : start + 4] for start in chunk])
        design = _design(contexts)
        targets = movie[chunk + 4].reshape(-1)
        xtx += design.T @ design
        xty += design.T @ targets
        n_samples += len(targets)
    return {"xtx": xtx, "xty": xty, "n_samples": int(n_samples)}


def fit_ridge_from_statistics(
    statistics: dict[str, np.ndarray | int], *, ridge: float
) -> np.ndarray:
    if ridge < 0:
        raise ValueError("ridge penalty must be non-negative")
    n_samples = int(statistics["n_samples"])
    if n_samples < 1:
        raise ValueError("sufficient statistics contain no samples")
    covariance = np.asarray(statistics["xtx"], dtype=np.float64) / n_samples
    cross = np.asarray(statistics["xty"], dtype=np.float64) / n_samples
    if covariance.shape != (5, 5) or cross.shape != (5,):
        raise ValueError("sufficient statistics have incompatible shapes")
    penalty = np.diag([ridge, ridge, ridge, ridge, 0.0])
    return np.linalg.solve(covariance + penalty, cross)


def fit_ridge_ar(
    contexts: np.ndarray, targets: np.ndarray, *, ridge: float
) -> np.ndarray:
    """Fit four shared temporal coefficients and an unpenalized intercept."""
    if ridge < 0:
        raise ValueError("ridge penalty must be non-negative")
    truth = np.asarray(targets, dtype=np.float64)
    values = np.asarray(contexts, dtype=np.float64)
    if truth.shape != (values.shape[0], *values.shape[2:]):
        raise ValueError("targets must align with context movies and pixels")
    design = _design(values)
    target_vector = truth.reshape(-1)
    covariance = design.T @ design / len(design)
    cross = design.T @ target_vector / len(design)
    penalty = np.diag([ridge, ridge, ridge, ridge, 0.0])
    return np.linalg.solve(covariance + penalty, cross)


def predict_next(context: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """Predict one frame from four oldest-to-newest intensity frames."""
    values = np.asarray(context, dtype=np.float64)
    weights = np.asarray(coefficients, dtype=np.float64)
    if values.ndim != 3 or values.shape[0] != 4:
        raise ValueError("context must have shape 4 x H x W")
    if weights.shape != (5,):
        raise ValueError("coefficients must contain four lags and intercept")
    return np.einsum("khw,k->hw", values, weights[:4]) + weights[4]


def recursive_rollout(
    initial_context: np.ndarray, coefficients: np.ndarray, *, steps: int
) -> np.ndarray:
    """Roll the linear model forward using its own predicted frames."""
    if steps < 1:
        raise ValueError("steps must be positive")
    context = np.asarray(initial_context, dtype=np.float64).copy()
    predictions = []
    for _ in range(int(steps)):
        prediction = predict_next(context, coefficients)
        if not np.all(np.isfinite(prediction)):
            raise FloatingPointError("linear autoregressive rollout diverged")
        predictions.append(prediction)
        context = np.concatenate([context[1:], prediction[None]], axis=0)
    return np.stack(predictions)


def evaluate_next_frame(
    frames: np.ndarray,
    *,
    starts: np.ndarray | list[int],
    coefficients: np.ndarray,
    chunk_size: int = 32,
) -> dict[str, float | int]:
    """Evaluate shared AR and fixed last-frame persistence on declared starts."""
    movie = np.asarray(frames, dtype=np.float64)
    indices = np.asarray(starts, dtype=np.int64)
    weights = np.asarray(coefficients, dtype=np.float64)
    if movie.ndim != 3 or indices.ndim != 1 or not len(indices):
        raise ValueError("frames must be T x H x W and starts must be nonempty")
    if weights.shape != (5,) or chunk_size < 1:
        raise ValueError("five coefficients and a positive chunk size are required")
    if indices.min() < 0 or indices.max() + 4 >= len(movie):
        raise ValueError("a context or target falls outside the movie")
    model_absolute_error = 0.0
    persistence_absolute_error = 0.0
    count = 0
    for offset in range(0, len(indices), int(chunk_size)):
        chunk = indices[offset : offset + int(chunk_size)]
        contexts = np.stack([movie[start : start + 4] for start in chunk])
        targets = movie[chunk + 4]
        prediction = (
            np.einsum("nkhw,k->nhw", contexts, weights[:4]) + weights[4]
        )
        persistence = contexts[:, -1]
        model_absolute_error += float(np.sum(np.abs(prediction - targets)))
        persistence_absolute_error += float(
            np.sum(np.abs(persistence - targets))
        )
        count += int(targets.size)
    model_mae = model_absolute_error / count
    persistence_mae = persistence_absolute_error / count
    return {
        "n_windows": int(len(indices)),
        "n_pixel_targets": count,
        "model_mae": model_mae,
        "persistence_mae": persistence_mae,
        "mae_ratio": (
            model_mae / persistence_mae
            if persistence_mae > 0
            else float("nan")
        ),
    }


def latex_table(payload: dict) -> str:
    lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"& Next-frame MAE ratio & \multicolumn{2}{c}{512-step cumulative MAE ratio} \\",
        r"Test movie & full frame & full frame & masked \\",
        r"\midrule",
    ]
    for row in payload["test_particles"]:
        horizon = row["anchored_rollout"]["horizons"]["512"]
        lines.append(
            f"particle {row['particle']} & {row['next_frame']['mae_ratio']:.3f} & "
            f"{horizon['cumulative_mae_ratio_full']:.2f} & "
            f"{horizon['cumulative_mae_ratio_masked']:.2f} \\\\"
        )
    next_mean = float(
        np.mean([row["next_frame"]["mae_ratio"] for row in payload["test_particles"]])
    )
    full_mean = float(
        np.mean(
            [
                row["anchored_rollout"]["horizons"]["512"][
                    "cumulative_mae_ratio_full"
                ]
                for row in payload["test_particles"]
            ]
        )
    )
    masked_mean = float(
        np.mean(
            [
                row["anchored_rollout"]["horizons"]["512"][
                    "cumulative_mae_ratio_masked"
                ]
                for row in payload["test_particles"]
            ]
        )
    )
    lines.extend(
        [
            r"\midrule",
            f"mean over particles & {next_mean:.3f} & {full_mean:.2f} & {masked_mean:.2f} \\\\ ",
            r"\midrule",
            r"\multicolumn{4}{l}{Selected ridge $\lambda="
            + f"{payload['selected_ridge']:g}"
            + r"$; five fitted parameters (four lags plus intercept).} \\",
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
    return "\n".join(lines) + "\n"


def _merge_statistics(items: list[dict]) -> dict:
    if not items:
        raise ValueError("no sufficient statistics to merge")
    return {
        "xtx": np.sum([np.asarray(item["xtx"]) for item in items], axis=0),
        "xty": np.sum([np.asarray(item["xty"]) for item in items], axis=0),
        "n_samples": int(sum(int(item["n_samples"]) for item in items)),
    }


def _load_movie(arrays_dir: Path, stem: str) -> dict[str, np.ndarray]:
    with np.load(arrays_dir / f"{stem}.npz", allow_pickle=False) as loaded:
        return {
            key: loaded[key].astype(np.float32)
            for key in ("intensity", "current", "frame_times")
        }


def _window_split(n_frames: int, max_windows: int) -> tuple[np.ndarray, np.ndarray]:
    count = min(int(max_windows), n_frames - 4)
    if count < 2:
        raise ValueError("movie is too short for train/validation windows")
    starts = np.arange(count, dtype=np.int64)
    n_validation = max(1, int(round(0.1 * count)))
    return starts[:-n_validation], starts[-n_validation:]


def run_repository_baseline(
    arrays_dir: Path,
    rollout_root: Path,
    *,
    ridges: tuple[float, ...],
    max_windows: int,
    rollout_steps: int,
) -> dict:
    from analysis.physics_metrics import detect_roi
    from analysis.rollout_scale_diagnostics import evaluate_rollout_scale
    from train.ref.common_sp_baselines import current_sign_change_indices

    started = time.monotonic()
    train_stems = [f"GRA29_C20_25deg_particle{i}" for i in range(1, 5)]
    test_stems = [f"GRA29_C20_45deg_particle{i}" for i in range(1, 5)]
    statistics = []
    split_manifest = {}
    for stem in train_stems:
        movie = _load_movie(arrays_dir, stem)["intensity"]
        train_starts, validation_starts = _window_split(len(movie), max_windows)
        statistics.append(
            accumulate_sufficient_statistics(movie, train_starts, chunk_size=32)
        )
        split_manifest[stem] = {
            "train_start_min": int(train_starts.min()),
            "train_start_max": int(train_starts.max()),
            "n_train_windows": int(len(train_starts)),
            "validation_start_min": int(validation_starts.min()),
            "validation_start_max": int(validation_starts.max()),
            "n_validation_windows": int(len(validation_starts)),
        }
        del movie
    merged = _merge_statistics(statistics)
    candidates = {
        ridge: fit_ridge_from_statistics(merged, ridge=ridge) for ridge in ridges
    }
    validation = {ridge: [] for ridge in ridges}
    for stem in train_stems:
        movie = _load_movie(arrays_dir, stem)["intensity"]
        _, validation_starts = _window_split(len(movie), max_windows)
        for ridge, coefficients in candidates.items():
            validation[ridge].append(
                evaluate_next_frame(
                    movie,
                    starts=validation_starts,
                    coefficients=coefficients,
                    chunk_size=32,
                )
            )
        del movie
    validation_summary = []
    for ridge in ridges:
        rows = validation[ridge]
        validation_summary.append(
            {
                "ridge": ridge,
                "mean_particle_model_mae": float(
                    np.mean([row["model_mae"] for row in rows])
                ),
                "mean_particle_persistence_mae": float(
                    np.mean([row["persistence_mae"] for row in rows])
                ),
                "mean_particle_mae_ratio": float(
                    np.mean([row["mae_ratio"] for row in rows])
                ),
            }
        )
    selected = min(
        validation_summary, key=lambda row: row["mean_particle_model_mae"]
    )
    selected_ridge = float(selected["ridge"])
    coefficients = candidates[selected_ridge]

    rollout_root.mkdir(parents=True, exist_ok=True)
    test_particles = []
    for particle, stem in enumerate(test_stems, start=1):
        movie = _load_movie(arrays_dir, stem)
        intensity = movie["intensity"]
        test_starts = np.arange(
            min(max_windows, len(intensity) - 4), dtype=np.int64
        )
        next_frame = evaluate_next_frame(
            intensity,
            starts=test_starts,
            coefficients=coefficients,
            chunk_size=32,
        )
        changes = current_sign_change_indices(movie["current"])
        if not len(changes):
            raise ValueError(f"test movie has no current-sign transition: {stem}")
        onset = int(changes[0])
        anchor_start = onset - 4
        if anchor_start < 0 or onset + rollout_steps > len(intensity):
            raise ValueError(f"anchored rollout falls outside movie: {stem}")
        initial_context = intensity[anchor_start:onset]
        prediction = recursive_rollout(
            initial_context, coefficients, steps=rollout_steps
        )
        targets = intensity[onset : onset + rollout_steps].astype(np.float64)
        persistence = np.repeat(
            initial_context[-1][None].astype(np.float64), rollout_steps, axis=0
        )
        mask = detect_roi(intensity)
        rollout = evaluate_rollout_scale(
            targets,
            persistence,
            prediction,
            mask,
            horizons=tuple(
                value for value in (1, 32, 128, 256, 512) if value <= rollout_steps
            ),
        )
        rollout.update(
            {
                "anchor_start": anchor_start,
                "onset_frame": onset,
                "target_start": onset,
                "target_stop_exclusive": onset + rollout_steps,
            }
        )
        np.savez_compressed(
            rollout_root / f"rollout_{stem}.npz",
            pred=prediction.astype(np.float32),
            naive=persistence.astype(np.float32),
            targets=targets.astype(np.float32),
            target_steps=np.arange(
                onset, onset + rollout_steps, dtype=np.int64
            ),
            frame_times=movie["frame_times"][onset : onset + rollout_steps],
            mask=mask,
        )
        test_particles.append(
            {
                "particle": particle,
                "stem": stem,
                "next_frame": next_frame,
                "anchored_rollout": rollout,
            }
        )
        del movie, intensity, prediction, targets, persistence, mask

    return {
        "description": "Shared four-tap linear autoregressive intensity baseline",
        "fit_unit": "pixel-time sample with coefficients shared over pixels and particles",
        "train_stems": train_stems,
        "test_stems": test_stems,
        "split_manifest": split_manifest,
        "ridge_grid": list(ridges),
        "validation": validation_summary,
        "selected_ridge": selected_ridge,
        "coefficients_oldest_to_newest_then_intercept": coefficients.tolist(),
        "n_fitted_parameters": 5,
        "test_particles": test_particles,
        "runtime_seconds": float(time.monotonic() - started),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arrays-dir", type=Path, default=ROOT / "data_prep" / "arrays"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "linear_ar_baseline.json",
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=ROOT / "paper" / "tables" / "linear_ar_baseline.tex",
    )
    parser.add_argument(
        "--rollout-root",
        type=Path,
        default=ROOT / "results" / "out" / "linear_ar_baseline" / "rollout_anchored",
    )
    parser.add_argument("--ridges", default="0,1e-8,1e-6,1e-4,1e-2,1,100")
    parser.add_argument("--max-windows", type=int, default=3000)
    parser.add_argument("--rollout-steps", type=int, default=512)
    args = parser.parse_args()
    ridges = tuple(float(value) for value in args.ridges.split(","))
    payload = run_repository_baseline(
        args.arrays_dir,
        args.rollout_root,
        ridges=ridges,
        max_windows=args.max_windows,
        rollout_steps=args.rollout_steps,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text(latex_table(payload))
    print(f"wrote {args.output}")
    print(f"wrote {args.table}")
    print(
        f"selected ridge={payload['selected_ridge']:g}; "
        f"coefficients={payload['coefficients_oldest_to_newest_then_intercept']}"
    )


if __name__ == "__main__":
    main()
