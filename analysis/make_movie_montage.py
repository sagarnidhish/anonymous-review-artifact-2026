#!/usr/bin/env python3
"""Render a full-record GRA29 montage and a matching lightweight GIF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STEM = "GRA29_C20_45deg_particle1"


def select_frame_indices(n_frames: int, count: int = 24) -> np.ndarray:
    """Return strictly increasing, endpoint-inclusive full-record indices."""
    if n_frames < 1:
        raise ValueError("n_frames must be positive")
    if count < 2 or count > n_frames:
        raise ValueError("count must be between 2 and n_frames")
    indices = np.rint(np.linspace(0, n_frames - 1, count)).astype(np.int64)
    if len(np.unique(indices)) != count:
        raise ValueError("count does not permit unique evenly spaced indices")
    return indices


def display_limits(
    frames: np.ndarray, lower: float = 1.0, upper: float = 99.5
) -> tuple[float, float]:
    """Compute one robust intensity scale to share across every rendered frame."""
    if not 0 <= lower < upper <= 100:
        raise ValueError("percentiles must satisfy 0 <= lower < upper <= 100")
    vmin, vmax = np.percentile(np.asarray(frames), [lower, upper])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        raise ValueError("selected frames do not define finite display limits")
    return float(vmin), float(vmax)


def render_montage(
    frames: np.ndarray,
    times_s: np.ndarray,
    indices: np.ndarray,
    vmin: float,
    vmax: float,
    output: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = frames[indices]
    rows, columns = 4, 6
    if len(indices) != rows * columns:
        raise ValueError("the publication montage expects exactly 24 frames")
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(7.0, 4.55),
        layout="constrained",
    )
    start = float(times_s[0])
    for position, (ax, frame, index) in enumerate(
        zip(axes.ravel(), selected, indices, strict=True)
    ):
        ax.imshow(frame, cmap="gray", vmin=vmin, vmax=vmax, rasterized=True)
        elapsed_h = (float(times_s[index]) - start) / 3600
        ax.set_title(
            f"{elapsed_h:.1f} h  |  frame {int(index)}",
            fontsize=6.2,
            pad=1.8,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        if position == 0:
            ax.text(
                -0.05,
                1.19,
                "a",
                transform=ax.transAxes,
                fontsize=9,
                fontweight="bold",
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def render_gif(
    frames: np.ndarray,
    indices: np.ndarray,
    vmin: float,
    vmax: float,
    output: Path,
    duration_ms: int = 450,
) -> None:
    from PIL import Image

    selected = np.clip((frames[indices] - vmin) / (vmax - vmin), 0, 1)
    images = [Image.fromarray(np.rint(frame * 255).astype(np.uint8), mode="L")
              for frame in selected]
    output.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        output,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stem", default=DEFAULT_STEM)
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument(
        "--arrays-dir", type=Path, default=ROOT / "data_prep" / "arrays"
    )
    parser.add_argument(
        "--montage",
        type=Path,
        default=ROOT / "paper" / "figures" / "gra29_movie_montage.pdf",
    )
    parser.add_argument(
        "--gif",
        type=Path,
        default=(
            ROOT / "paper" / "supplementary" / "gra29_particle1_overview.gif"
        ),
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=(
            ROOT / "paper" / "supplementary" / "gra29_particle1_overview.json"
        ),
    )
    args = parser.parse_args()

    with np.load(args.arrays_dir / f"{args.stem}.npz") as loaded:
        frames = loaded["intensity"].astype(np.float32)
        times_s = loaded["frame_times"].astype(np.float64)
    indices = select_frame_indices(len(frames), args.count)
    selected = frames[indices]
    vmin, vmax = display_limits(selected)
    render_montage(frames, times_s, indices, vmin, vmax, args.montage)
    render_gif(frames, indices, vmin, vmax, args.gif)

    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_stem": args.stem,
        "n_source_frames": int(len(frames)),
        "selected_indices": indices.tolist(),
        "selected_elapsed_hours": (
            (times_s[indices] - times_s[0]) / 3600
        ).round(6).tolist(),
        "display_percentiles": [1.0, 99.5],
        "display_limits": [vmin, vmax],
        "montage": str(args.montage.relative_to(ROOT)),
        "gif": str(args.gif.relative_to(ROOT)),
    }
    args.metadata.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.montage}")
    print(f"wrote {args.gif}")
    print(f"wrote {args.metadata}")


if __name__ == "__main__":
    main()
