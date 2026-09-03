#!/usr/bin/env python3
"""Render an appendix overview of the complete fresh matched-grid campaign."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402


HORIZONS = (32, 128, 256, 512)
COLUMN_LABELS = ("Next", "+32", "+128", "+256", "+512")
MODEL_LABELS = {
    "unet": "U-Net",
    "convlstm": "ConvLSTM",
    "simvp": "SimVP",
    "residual_cnn": "Residual CNN",
    "predrnn": "PredRNN",
    "predrnnpp": "PredRNN++",
}


def _row_label(row: dict) -> str:
    inputs = "image" if row["input_mode"] == "image_only" else "protocol"
    target = "$\\Delta$" if row["target_mode"] == "delta" else "direct"
    divergence = " $\\dagger$" if row.get("rollout_diverged_particles", 0) else ""
    return f"{MODEL_LABELS.get(row['model_family'], row['model_family'])} | {inputs} | {target}{divergence}"


def _grid_values(rows: list[dict]) -> np.ndarray:
    def ratio(value):
        return float("inf") if value is None else float(value)

    values = []
    for row in rows:
        values.append(
            [ratio(row["next_frame_mean_mae_ratio"])]
            + [
                ratio(row[f"rollout_mean_mae_ratio_h{horizon}"])
                for horizon in HORIZONS
            ]
        )
    array = np.asarray(values, dtype=float)
    if array.shape != (24, 5) or np.any(array <= 0) or np.isnan(array).any():
        raise ValueError("matched-grid ratios must form a positive 24 by 5 matrix")
    return array


def make_matched_grid_figure(summary: dict, output: Path) -> None:
    rows = sorted(summary.get("payloads", []), key=lambda row: row["payload_id"])
    if summary.get("status") != "complete" or len(rows) != 24:
        raise ValueError("a complete summary with 24 payload rows is required")
    if [row["payload_id"] for row in rows] != list(range(24)):
        raise ValueError("the 24 payload IDs must be complete and ordered from 0 to 23")
    values = _grid_values(rows)
    finite_logs = np.log10(values[np.isfinite(values)])
    lower = min(-0.15, float(np.min(finite_logs)))
    upper = max(1.0, float(np.percentile(finite_logs, 90)))
    log_values = np.log10(values)
    display_values = np.where(np.isfinite(log_values), log_values, upper)

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure = plt.figure(figsize=(8.4, 8.2), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, width_ratios=(3.9, 1.7), height_ratios=(1, 1))
    heat = figure.add_subplot(grid[:, 0])
    image = heat.imshow(
        display_values,
        aspect="auto",
        interpolation="nearest",
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=lower, vcenter=0.0, vmax=upper),
    )
    heat.set_xticks(range(5), COLUMN_LABELS)
    heat.xaxis.tick_top()
    heat.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False, length=0)
    heat.set_yticks(range(24), [_row_label(row) for row in rows])
    heat.set_title("Particle-mean MAE ratio by forecast horizon", pad=22)
    for boundary in range(4, 24, 4):
        heat.axhline(boundary - 0.5, color="white", linewidth=1.5)
    for row_index, column_index in np.argwhere(~np.isfinite(values)):
        heat.text(column_index, row_index, "$\\infty$", ha="center", va="center", fontsize=7)
    heat.text(
        0,
        -1.25,
        "blue: better than fixed persistence     red: worse     $\\dagger$: at least one non-finite rollout",
        ha="left",
        va="center",
        fontsize=7,
        transform=heat.transData,
    )
    colorbar = figure.colorbar(image, ax=heat, location="bottom", shrink=0.75, pad=0.035)
    colorbar.set_label("$\\log_{10}$(model MAE / fixed-persistence MAE)")

    config_counts = [int(np.sum(values[:, column] < 1.0)) for column in range(5)]
    particle_keys = ["next_frame_particles_better"] + [
        f"rollout_particles_better_h{horizon}" for horizon in HORIZONS
    ]
    particle_counts = [sum(int(row[key]) for row in rows) for key in particle_keys]
    y = np.arange(5)

    configs = figure.add_subplot(grid[0, 1])
    configs.barh(y, config_counts, color="#32689b")
    configs.set_yticks(y, COLUMN_LABELS)
    configs.invert_yaxis()
    configs.set_xlim(0, 24)
    configs.set_xlabel("Configurations with ratio $<1$\n(out of 24)", fontsize=7)
    configs.set_title("Configuration-level skill")
    configs.spines[["top", "right"]].set_visible(False)
    for index, value in enumerate(config_counts):
        configs.text(value + 0.35, index, str(value), va="center", fontsize=7)

    particles = figure.add_subplot(grid[1, 1])
    particles.barh(y, particle_counts, color="#6b9e78")
    particles.set_yticks(y, COLUMN_LABELS)
    particles.invert_yaxis()
    particles.set_xlim(0, 96)
    particles.set_xlabel(
        "Particle comparisons with ratio $<1$\n(out of 96)", fontsize=7
    )
    particles.set_title("Particle-level skill")
    particles.spines[["top", "right"]].set_visible(False)
    for index, value in enumerate(particle_counts):
        particles.text(value + 1.2, index, str(value), va="center", fontsize=7)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.tmp{output.suffix}")
    try:
        figure.savefig(temporary, bbox_inches="tight")
        os.replace(temporary, output)
    finally:
        plt.close(figure)
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with Path(args.summary).open(encoding="utf-8") as handle:
        summary = json.load(handle)
    make_matched_grid_figure(summary, Path(args.output))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
