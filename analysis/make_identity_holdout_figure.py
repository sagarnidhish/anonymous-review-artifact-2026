#!/usr/bin/env python3
"""Render the validated four-fold identity-holdout results particle by particle."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODELS = ("unet", "predrnnpp")
MODEL_LABELS = {"unet": "U-Net", "predrnnpp": "PredRNN++"}
MODES = ("next_frame", "rollout")
MODE_LABELS = {"next_frame": "next frame", "rollout": "512-step rollout"}
GROUPS = (
    "same_temperature_unseen_particle",
    "cross_temperature_unseen_particle",
)
GROUP_LABELS = ("25 °C\nsame temperature", "45 °C\ncross temperature")
PARTICLE_COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
PARTICLE_MARKERS = ("o", "s", "^", "D")


def _grid(summary: dict) -> dict[tuple[str, str, str], list[tuple[int, float]]]:
    rows = summary.get("per_particle", [])
    expected = len(MODELS) * len(MODES) * len(GROUPS) * 4
    if len(rows) != expected:
        raise ValueError("identity result does not contain a complete four-particle grid")
    grid: dict[tuple[str, str, str], list[tuple[int, float]]] = {}
    for model in MODELS:
        for mode in MODES:
            for group in GROUPS:
                selected = sorted(
                    (
                        (int(row["heldout_particle"]), float(row["mae_ratio"]))
                        for row in rows
                        if row["model_family"] == model
                        and row["mode"] == mode
                        and row["evaluation_group"] == group
                    ),
                    key=lambda item: item[0],
                )
                if [particle for particle, _ in selected] != [1, 2, 3, 4]:
                    raise ValueError(
                        "identity result does not contain a complete four-particle grid"
                    )
                if not np.isfinite([value for _, value in selected]).all():
                    raise ValueError("identity result contains a non-finite MAE ratio")
                grid[(model, mode, group)] = selected
    return grid


def make_figure(summary: dict, output: Path) -> None:
    grid = _grid(summary)
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "pdf.fonttype": 42,
            "axes.linewidth": 0.7,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 4.6), sharex=True)
    x = np.arange(2)
    for row_index, mode in enumerate(MODES):
        row_values = []
        for column_index, model in enumerate(MODELS):
            ax = axes[row_index, column_index]
            values = np.asarray(
                [
                    [
                        dict(grid[(model, mode, group)])[particle]
                        for group in GROUPS
                    ]
                    for particle in range(1, 5)
                ],
                dtype=float,
            )
            row_values.extend(values.ravel().tolist())
            for particle_index in range(4):
                ax.plot(
                    x,
                    values[particle_index],
                    color=PARTICLE_COLORS[particle_index],
                    marker=PARTICLE_MARKERS[particle_index],
                    ms=4.0,
                    lw=0.9,
                    alpha=0.72,
                    label=f"particle {particle_index + 1}",
                )
            ax.plot(
                x,
                values.mean(axis=0),
                color="#111111",
                marker="P",
                ms=6.2,
                lw=2.0,
                label="particle mean",
                zorder=5,
            )
            ax.axhline(1.0, color="#777777", lw=0.9, ls="--")
            ax.grid(axis="y", color="#dddddd", lw=0.5)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_title(f"{MODEL_LABELS[model]} — {MODE_LABELS[mode]}")
            ax.set_xticks(x, GROUP_LABELS)
            if column_index == 0:
                ax.set_ylabel("MAE / fixed persistence")
            panel = chr(ord("a") + row_index * 2 + column_index)
            ax.text(
                -0.13,
                1.05,
                panel,
                transform=ax.transAxes,
                fontweight="bold",
                fontsize=9,
            )
        row_max = max(row_values)
        row_min = min(row_values)
        low = min(0.9, row_min * 0.92)
        high = max(1.08, row_max * 1.08)
        for ax in axes[row_index]:
            if row_min > 0 and row_max / row_min > 25:
                ax.set_yscale("log")
            ax.set_ylim(low, high)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 1.01),
        fontsize=7.2,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=1.3, w_pad=1.4)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    make_figure(summary, args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
