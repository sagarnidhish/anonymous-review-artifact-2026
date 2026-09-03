#!/usr/bin/env python3
"""Generate final-size main-paper and appendix figures.

All main panels expose the particle/movie as the independent unit. Optical
summaries are explicitly uncalibrated proxies and bright-fraction comparisons
share one threshold calibrated from observed ground-truth context.
"""

from __future__ import annotations

import csv
import json
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from analysis.evaluation_invariants import calibrate_bright_threshold
    from analysis.physics_metrics import detect_roi, observable_trajectories
except ModuleNotFoundError:
    from evaluation_invariants import calibrate_bright_threshold
    from physics_metrics import detect_roi, observable_trajectories


ROOT = Path(__file__).resolve().parents[1]
ARRAYS = ROOT / "data_prep" / "arrays"
FROZEN = ROOT / "results" / "frozen"
CORRECTED = ROOT / "results" / "analysis_corrected"
OUT = ROOT / "paper" / "figures"
TEST_STEMS = [f"GRA29_C20_45deg_particle{i}" for i in range(1, 5)]

# Okabe-Ito, with redundant marker/linestyle encodings.
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
GREY = "#777777"
LIGHT_GREY = "#D9D9D9"
BLACK = "#111111"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 7.3,
        "axes.titlesize": 8.2,
        "axes.labelsize": 7.3,
        "xtick.labelsize": 6.7,
        "ytick.labelsize": 6.7,
        "legend.fontsize": 6.2,
        "axes.linewidth": 0.65,
        "lines.linewidth": 1.15,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }
)


def _panel(ax, label: str) -> None:
    ax.text(
        -0.12,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def _clean(ax, grid: bool = True) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    if grid:
        ax.grid(axis="y", color=LIGHT_GREY, lw=0.45, zorder=0)


def _rows(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


@lru_cache(maxsize=2)
def _particle(stem: str) -> dict[str, np.ndarray]:
    with np.load(ARRAYS / f"{stem}.npz") as loaded:
        return {
            key: loaded[key].astype(np.float32)
            for key in ("intensity", "current", "voltage", "frame_times")
        }


def _artifact(tag: str, stem: str) -> dict[str, np.ndarray]:
    paths = sorted(
        (
            ROOT
            / "results"
            / "out"
            / tag
            / "rollout_anchored"
            / stem
        ).glob("*.npz")
    )
    if len(paths) != 1:
        raise FileNotFoundError(
            f"expected one anchored artifact for {tag}/{stem}"
        )
    with np.load(paths[0]) as loaded:
        return {key: loaded[key].copy() for key in loaded.files}


def fig_task_design() -> None:
    sequence = _particle(TEST_STEMS[0])
    anchor = 2848
    shown = np.r_[np.arange(anchor - 4, anchor), anchor, anchor + 255]
    frames = sequence["intensity"][shown]
    lo, hi = np.percentile(frames, [1, 99.5])

    fig = plt.figure(figsize=(6.8, 2.05), layout="constrained")
    grid = fig.add_gridspec(1, 4, width_ratios=(1.55, 1.35, 1.0, 1.25))

    ax = fig.add_subplot(grid[0, 0])
    strip = np.concatenate(frames, axis=1)
    ax.imshow(strip, cmap="gray", vmin=lo, vmax=hi, rasterized=True)
    ax.set_xticks([(index + 0.5) * 128 for index in range(len(shown))])
    ax.set_xticklabels(
        ["$t{-}4$", "$t{-}3$", "$t{-}2$", "$t{-}1$", "$t$", "$t{+}256$"]
    )
    ax.tick_params(axis="x", length=0, pad=2, labelsize=5.9)
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.axvline(4 * 128, color=ORANGE, lw=1.4)
    ax.axvline(5 * 128, color=GREEN, lw=1.4)
    ax.text(
        0.31,
        1.02,
        "context",
        transform=ax.transAxes,
        ha="center",
    )
    ax.text(0.83, 1.02, "targets", transform=ax.transAxes, ha="center")
    _panel(ax, "a")

    ax = fig.add_subplot(grid[0, 1])
    hours = sequence["frame_times"] / 3600
    ax.plot(hours, sequence["voltage"], color=ORANGE, label="voltage")
    ax.plot(hours, sequence["current"], color=BLUE, label="current")
    start_h = hours[anchor]
    stop_h = hours[anchor + 511]
    ax.axvline(start_h, color=BLACK, ls="--", lw=0.8)
    ax.axvspan(start_h, stop_h, color=GREEN, alpha=0.12, lw=0)
    ax.text(
        (start_h + stop_h) / 2,
        0.98,
        "anchored rollout",
        ha="center",
        va="top",
        fontsize=6.2,
    )
    ax.set(xlabel="elapsed time (h)", ylabel="normalized input")
    ax.legend(
        frameon=False,
        loc="lower left",
        ncol=2,
        handlelength=1.4,
    )
    _clean(ax)
    _panel(ax, "b")

    ax = fig.add_subplot(grid[0, 2])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    for index in range(4):
        y = 0.79 - index * 0.14
        ax.plot(
            [0.28, 0.72],
            [y, y],
            color=LIGHT_GREY,
            lw=1.0,
            zorder=0,
        )
        ax.scatter(
            0.22,
            y,
            s=45,
            color=BLUE,
            marker="o",
            edgecolor="white",
            lw=0.5,
        )
        ax.scatter(
            0.78,
            y,
            s=45,
            color=ORANGE,
            marker="s",
            edgecolor="white",
            lw=0.5,
        )
        ax.text(
            0.22,
            y,
            f"p{index + 1}",
            color="white",
            ha="center",
            va="center",
            fontsize=5.5,
        )
        ax.text(
            0.78,
            y,
            f"p{index + 1}",
            color="white",
            ha="center",
            va="center",
            fontsize=5.5,
        )
    ax.text(
        0.22,
        0.03,
        "train\n25 °C",
        color=BLUE,
        ha="center",
        fontweight="bold",
    )
    ax.text(
        0.78,
        0.03,
        "test\n45 °C",
        color=ORANGE,
        ha="center",
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.99,
        "$n=4$ paired physical particles",
        ha="center",
        fontsize=6.2,
    )
    ax.text(
        0.5,
        0.20,
        "paired split: same identity across temperature\n"
        "identity holdout: train on 3, test the fourth",
        ha="center",
        fontsize=5.9,
        color=GREY,
    )
    ax.axis("off")
    _panel(ax, "c")

    ax = fig.add_subplot(grid[0, 3])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    box = dict(
        boxstyle="round,pad=0.28",
        ec=LIGHT_GREY,
        fc="#FAFAFA",
        lw=0.7,
    )
    ax.text(
        0.5,
        0.82,
        "Pixel prediction error",
        ha="center",
        va="center",
        fontweight="bold",
        bbox=box,
    )
    ax.text(
        0.5,
        0.66,
        "MAE(model) / MAE(persistence)",
        ha="center",
        fontsize=6.3,
    )
    ax.annotate(
        "",
        xy=(0.5, 0.46),
        xytext=(0.5, 0.58),
        arrowprops=dict(arrowstyle="->", color=GREY),
    )
    ax.text(
        0.5,
        0.36,
        "Particle-region summaries",
        ha="center",
        va="center",
        fontweight="bold",
        bbox=box,
    )
    ax.text(
        0.5,
        0.16,
        "masked mean  •  P95\nshared-threshold bright fraction",
        ha="center",
        fontsize=6.2,
    )
    ax.text(
        0.5,
        0.02,
        "optical measurements, not calibrated state",
        ha="center",
        color=ORANGE,
        fontsize=6.1,
    )
    ax.axis("off")
    _panel(ax, "d")

    fig.savefig(OUT / "fig_task_design.pdf", dpi=300)
    plt.close(fig)


def fig_benchmark_controls() -> None:
    corrected = _rows(CORRECTED / "macro_trend.csv")
    matched = json.loads(
        (ROOT / "artifacts" / "v5_matched_grid_summary.json").read_text()
    )
    if matched.get("status") != "complete" or len(matched.get("payloads", [])) != 24:
        raise ValueError("complete fresh matched-grid summary is required")
    protocol = json.loads(
        (ROOT / "artifacts" / "protocol_interventions.json").read_text()
    )
    families = [
        "unet",
        "convlstm",
        "simvp",
        "residual_cnn",
        "predrnn",
        "predrnnpp",
    ]
    family_labels = [
        "U-Net",
        "ConvLSTM",
        "SimVP",
        "ResCNN",
        "PredRNN",
        "PredRNN++",
    ]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(6.8, 2.55),
        layout="constrained",
    )

    ax = axes[0]
    for family_index, family in enumerate(families):
        for input_mode, offset, color, marker, label in (
            (
                "image_only",
                -0.13,
                BLUE,
                "o",
                r"image-only $\Delta$",
            ),
            (
                "protocol_conditioned",
                0.13,
                ORANGE,
                "^",
                r"protocol $\Delta$",
            ),
        ):
            selected = [
                row
                for row in matched["payloads"]
                if row["model_family"] == family
                and row["input_mode"] == input_mode
                and row["target_mode"] == "delta"
            ]
            if len(selected) != 1:
                raise ValueError(
                    f"missing fresh delta result: {family}/{input_mode}"
                )
            value = float(selected[0]["next_frame_mean_mae_ratio"])
            x = family_index + offset
            ax.scatter(
                x,
                value,
                s=24,
                facecolor=color,
                edgecolor=color,
                lw=0.7,
                marker=marker,
                zorder=3,
                label=label if family_index == 0 else None,
            )
    ax.axhline(1, color=GREY, ls="--", lw=0.9)
    ax.text(
        0.02,
        1.01,
        "persistence",
        transform=ax.get_yaxis_transform(),
        color=GREY,
        fontsize=6.1,
        va="bottom",
    )
    ax.set_xticks(
        range(len(families)),
        family_labels,
        rotation=35,
        ha="right",
    )
    ax.set_ylim(0.77, 1.06)
    ax.set_ylabel("next-frame MAE ratio")
    ax.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        handletextpad=0.3,
        columnspacing=0.8,
    )
    _clean(ax)
    _panel(ax, "a")

    ax = axes[1]
    buckets = ["0-32", "32-128", "128-256", "256-512"]
    tag_specs = (
        ("unet_image_only_delta_anchored", BLUE, "o", "U-Net"),
        (
            "unet_image_only_delta_rft_anchored",
            ORANGE,
            "s",
            "U-Net + 8-step RFT",
        ),
    )
    for tag, color, marker, label in tag_specs:
        for stem in TEST_STEMS:
            values = []
            for bucket in buckets:
                selected = [
                    row
                    for row in corrected
                    if row["tag"] == tag
                    and row["mode"] == "rollout_anchored"
                    and row["stem"] == stem
                    and row["bucket"] == bucket
                ]
                if len(selected) != 1:
                    raise ValueError(
                        f"missing corrected horizon row: {tag}/{stem}/{bucket}"
                    )
                values.append(float(selected[0]["pixel_mae_ratio"]))
            ax.plot(
                range(4),
                values,
                color=color,
                alpha=0.22,
                lw=0.75,
                marker=marker,
                ms=2.0,
            )
        means = []
        for bucket in buckets:
            values = [
                float(row["pixel_mae_ratio"])
                for row in corrected
                if row["tag"] == tag
                and row["mode"] == "rollout_anchored"
                and row["stem"] in TEST_STEMS
                and row["bucket"] == bucket
            ]
            means.append(np.mean(values))
        ax.plot(
            range(4),
            means,
            color=color,
            marker=marker,
            ms=4,
            lw=1.8,
            label=label,
        )
    ax.axhline(1, color=GREY, ls="--", lw=0.9)
    ax.set_yscale("log")
    ax.set_xticks(
        range(4),
        ["0–32", "32–128", "128–256", "256–512"],
        rotation=30,
        ha="right",
    )
    ax.set_ylabel("anchored-rollout MAE ratio")
    ax.set_xlabel("forecast-step bucket")
    ax.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        columnspacing=0.8,
    )
    _clean(ax)
    _panel(ax, "b")

    ax = axes[2]
    conditions = ["true", "zero", "shift", "shuffle"]
    labels = ["measured", "zeroed", "delayed", "shuffled"]
    payload = protocol["modes"]["next_frame"]
    per_particle = np.asarray(
        [payload[condition]["mae_ratio"] for condition in conditions]
    ).T
    for values in per_particle:
        ax.plot(range(4), values, color=LIGHT_GREY, lw=0.8, zorder=1)
        ax.scatter(
            range(4),
            values,
            s=12,
            facecolor="white",
            edgecolor=GREY,
            lw=0.6,
            zorder=2,
        )
    means = per_particle.mean(axis=0)
    ax.plot(
        range(4),
        means,
        color=PURPLE,
        marker="D",
        ms=4,
        lw=1.8,
        zorder=3,
    )
    ax.axhline(1, color=GREY, ls="--", lw=0.9)
    ax.set_xticks(range(4), labels, rotation=30, ha="right")
    ax.set_ylim(0.75, 1.58)
    ax.set_ylabel("next-frame MAE ratio")
    ax.text(
        0.04,
        0.96,
        f"shuffled $-$ measured = +{means[-1] - means[0]:.2f}",
        transform=ax.transAxes,
        color=PURPLE,
        ha="left",
        va="top",
        fontweight="bold",
        fontsize=6.2,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1.5},
    )
    _clean(ax)
    _panel(ax, "c")

    fig.savefig(OUT / "fig_benchmark_controls.pdf", dpi=300)
    plt.close(fig)


def fig_rollout_fidelity() -> None:
    stem = TEST_STEMS[0]
    sequence = _particle(stem)
    unet = _artifact("unet_image_only_delta_anchored", stem)
    rft = _artifact("unet_image_only_delta_rft_anchored", stem)
    walrus = _artifact("walrus_native_corrected", stem)
    for candidate in (rft, walrus):
        if not np.allclose(
            candidate["frame_times"],
            unet["frame_times"],
            atol=0.05,
            rtol=0,
        ):
            raise ValueError(
                "qualitative rollout artifacts are not time aligned"
            )
        if not np.allclose(
            candidate["targets"],
            unet["targets"],
            atol=5e-4,
            rtol=0,
        ):
            raise ValueError("qualitative rollout targets do not match")

    truth = unet["targets"].astype(np.float32)
    persistence = unet["naive"].astype(np.float32)
    methods = [
        ("truth", truth, BLACK),
        ("persistence", persistence, GREY),
        ("U-Net", unet["pred"].astype(np.float32), BLUE),
        ("U-Net + RFT", rft["pred"].astype(np.float32), ORANGE),
        ("Walrus", walrus["pred"].astype(np.float32), GREEN),
    ]
    horizons = [31, 127, 255, 511]
    image_values = np.concatenate(
        [truth[index].ravel() for index in horizons]
    )
    vmin, vmax = np.percentile(image_values, [1, 99.7])
    final_errors = np.concatenate(
        [
            np.abs(frames[-1] - truth[-1]).ravel()
            for _, frames, _ in methods[1:]
        ]
    )
    error_vmax = max(float(np.percentile(final_errors, 98)), 1e-6)

    fig = plt.figure(figsize=(6.8, 3.65), layout="constrained")
    outer = fig.add_gridspec(1, 2, width_ratios=(1.42, 1.15))
    images_grid = outer[0, 0].subgridspec(
        5,
        5,
        wspace=0.035,
        hspace=0.035,
    )
    first_image_ax = None
    for row, (label, frames, _) in enumerate(methods):
        for column, horizon in enumerate(horizons):
            ax = fig.add_subplot(images_grid[row, column])
            if first_image_ax is None:
                first_image_ax = ax
            ax.imshow(
                frames[horizon],
                cmap="gray",
                vmin=vmin,
                vmax=vmax,
                rasterized=True,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                minutes = round((horizon + 1) * 30.06 / 60)
                ax.set_title(
                    f"+{horizon + 1}\n{minutes} min",
                    fontsize=5.8,
                    pad=1.5,
                )
            if column == 0:
                ax.set_ylabel(
                    label,
                    rotation=0,
                    ha="right",
                    va="center",
                    labelpad=25,
                    fontsize=6.5,
                )
        ax = fig.add_subplot(images_grid[row, 4])
        error = np.abs(frames[-1] - truth[-1])
        ax.imshow(
            error,
            cmap="magma",
            vmin=0,
            vmax=error_vmax,
            rasterized=True,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        if row == 0:
            ax.set_title("$|$error$|$\n+512 (257 min)", fontsize=5.8, pad=1.5)
    if first_image_ax is not None:
        _panel(first_image_ax, "a")

    right = outer[0, 1].subgridspec(3, 1, hspace=0.18)
    mask = detect_roi(sequence["intensity"])
    global_start = int(
        np.argmin(
            np.abs(sequence["frame_times"] - unet["frame_times"][0])
        )
    )
    context = sequence["intensity"][global_start - 4 : global_start]
    threshold = calibrate_bright_threshold(context, mask)
    proxy = {
        label: observable_trajectories(frames, mask, threshold)
        for label, frames, _ in methods
    }
    time_h = (
        unet["frame_times"] - unet["frame_times"][0]
    ) / 3600
    color_of = {label: color for label, _, color in methods}

    ax = fig.add_subplot(right[0, 0])
    for label, _, _ in methods:
        y = proxy[label]["mean"]
        ax.plot(
            time_h,
            y,
            color=color_of[label],
            ls="--" if label == "persistence" else "-",
            label=label,
        )
    ax.set_ylabel("masked mean\n(optical proxy)")
    ax.legend(
        frameon=False,
        ncol=3,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02),
        handlelength=1.3,
        handletextpad=0.3,
        columnspacing=0.65,
        borderaxespad=0,
    )
    _clean(ax)
    _panel(ax, "b")

    ax = fig.add_subplot(right[1, 0])
    for label, _, _ in methods:
        ax.plot(
            time_h,
            proxy[label]["bright_frac90"],
            color=color_of[label],
            ls="--" if label == "persistence" else "-",
        )
    ax.set_ylabel("shared-threshold\nbright fraction")
    _clean(ax)
    _panel(ax, "c")

    ax = fig.add_subplot(right[2, 0])
    denominator = np.cumsum(
        np.mean(np.abs(persistence - truth), axis=(1, 2))
    )
    denominator = np.maximum(denominator, 1e-12)
    ax.plot(time_h, np.ones_like(time_h), color=GREY, ls="--")
    for label, frames, color in methods[2:]:
        numerator = np.cumsum(
            np.mean(np.abs(frames - truth), axis=(1, 2))
        )
        ax.plot(
            time_h,
            numerator / denominator,
            color=color,
            label=label,
        )
    ax.set_yscale("log")
    ax.set_ylabel("cumulative\nMAE ratio")
    ax.set_xlabel("hours after protocol anchor")
    _clean(ax)
    _panel(ax, "d")

    fig.savefig(OUT / "fig_rollout_fidelity.pdf", dpi=300)
    plt.close(fig)


def fig_sign_reversal() -> None:
    rows = [
        row
        for row in _rows(CORRECTED / "phase_slopes.csv")
        if row["mode"] == "next_frame"
        and row["stem"] in TEST_STEMS
        and row["observable"] == "bright_frac90"
    ]
    tags = [
        tag
        for tag in (
            "unet_image_only_delta",
            "unet_multichannel_delta",
            "convlstm_multichannel_delta",
            "predrnnpp_image_only_delta",
            "unet_image_only_delta_rft",
        )
        if any(row["tag"] == tag for row in rows)
    ]
    labels = {
        "convlstm_multichannel_delta": r"ConvLSTM protocol $\Delta$",
        "predrnnpp_image_only_delta": r"PredRNN++ image $\Delta$",
        "unet_image_only_delta": r"U-Net image $\Delta$",
        "unet_image_only_delta_rft": "U-Net + RFT",
        "unet_multichannel_delta": r"U-Net protocol $\Delta$",
    }
    matrix = []
    for tag in tags:
        status = []
        for stem in TEST_STEMS:
            selected = [
                row
                for row in rows
                if row["tag"] == tag and row["stem"] == stem
            ]
            slopes = {
                row["phase"]: float(row["slope_pred_per_s"])
                for row in selected
            }
            status.append(
                int(
                    len(slopes) == 2
                    and np.sign(slopes["negative_current"])
                    != np.sign(slopes["positive_current"])
                )
            )
        matrix.append(status)
    if not matrix:
        raise ValueError("no corrected sign-reversal rows")

    truth_status = []
    for stem in TEST_STEMS:
        truth_slopes = {
            row["phase"]: float(row["slope_true_per_s"])
            for row in rows
            if row["stem"] == stem
        }
        truth_status.append(
            int(
                len(truth_slopes) == 2
                and np.sign(truth_slopes["negative_current"])
                != np.sign(truth_slopes["positive_current"])
            )
        )

    fig, ax = plt.subplots(
        figsize=(4.6, 0.46 * len(tags) + 1.05),
        layout="constrained",
    )
    for row in range(len(tags)):
        for column in range(4):
            agreement = matrix[row][column] == truth_status[column]
            ax.add_patch(
                plt.Rectangle(
                    (column - 0.5, row - 0.5),
                    1,
                    1,
                    facecolor="#DCEAF3" if agreement else "#F6E4D7",
                    edgecolor="white",
                    linewidth=1.0,
                )
            )
            ax.text(
                column,
                row,
                ("opposite" if matrix[row][column] else "same sign")
                + ("\nagrees" if agreement else "\ndiffers"),
                ha="center",
                va="center",
                fontsize=5.8,
            )
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(len(tags) - 0.5, -0.5)
    ax.set_xticks(
        range(4),
        [
            f"p{i}\ntruth: "
            f"{'opposite' if truth_status[i - 1] else 'same'}"
            for i in range(1, 5)
        ],
    )
    ax.set_yticks(range(len(tags)), [labels[tag] for tag in tags])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(
        "Bright-area slope sign under negative versus positive current",
        fontsize=7.0,
        pad=6,
    )
    fig.savefig(OUT / "fig_sign_reversal.pdf", dpi=300)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig_task_design()
    fig_benchmark_controls()
    fig_rollout_fidelity()
    fig_sign_reversal()
    print(f"figures written to {OUT}")


if __name__ == "__main__":
    main()
