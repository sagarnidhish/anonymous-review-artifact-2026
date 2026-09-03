#!/usr/bin/env python3
"""Build appendix robustness and seed tables from test-only summaries."""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_RESULTS = ROOT / "results" / "corrected_summaries"
DEFAULT_OUTPUT = ROOT / "paper" / "tables"

REQUIRED_TAGS = (
    "unet_image_only_delta",
    "unet_img_delta_lopo",
    "predrnnpp_image_only_delta",
    "predrnnpp_img_delta_lopo",
    "unet_img_delta_s2026",
    "predrnnpp_img_delta_s2026",
)


def summary_path(results_root: Path, tag: str) -> Path:
    return results_root / tag / "next_frame" / "comparison_summary.json"


def validate_required_runs(results_root: Path) -> None:
    missing = [
        tag for tag in REQUIRED_TAGS
        if not summary_path(results_root, tag).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "missing required manuscript runs: " + ", ".join(missing)
        )


def ratio(results_root: Path, tag: str) -> float:
    payload = json.loads(summary_path(results_root, tag).read_text())
    if len(payload) != 1 or "mean_mae_ratio" not in payload[0]:
        raise ValueError(f"invalid comparison summary for {tag}")
    return float(payload[0]["mean_mae_ratio"])


def _ratio_cell(value: float) -> str:
    return f"\\textbf{{{value:.3f}}}" if value < 1 else f"{value:.3f}"


def build_lopo_table(results_root: Path) -> str:
    rows = [
        (
            "U-Net image $\\Delta$ (fresh primary-split rerun)",
            "train 25\\,°C particles 1--4 $\\to$ test paired 45\\,°C particles 1--4",
            ratio(results_root, "unet_image_only_delta"),
        ),
        (
            "U-Net image $\\Delta$ (particle-4 robustness split)",
            "train particles 1--3 at both temperatures $\\to$ test particle 4 at both",
            ratio(results_root, "unet_img_delta_lopo"),
        ),
        (
            "PredRNN++ image $\\Delta$ (fresh primary-split rerun)",
            "train 25\\,°C particles 1--4 $\\to$ test paired 45\\,°C particles 1--4",
            ratio(results_root, "predrnnpp_image_only_delta"),
        ),
        (
            "PredRNN++ image $\\Delta$ (particle-4 robustness split)",
            "train particles 1--3 at both temperatures $\\to$ test particle 4 at both",
            ratio(results_root, "predrnnpp_img_delta_lopo"),
        ),
    ]
    lines = [
        r"\begin{tabular}{p{0.32\textwidth}p{0.46\textwidth}c}",
        r"\toprule",
        r"Configuration & Evaluation split & test MAE ratio \\",
        r"\midrule",
    ]
    lines.extend(
        f"{label} & {split} & {_ratio_cell(value)} \\\\"
        for label, split, value in rows
    )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def build_seed_table(results_root: Path) -> str:
    rows = [
        (
            "U-Net image $\\Delta$",
            ratio(results_root, "unet_image_only_delta"),
            ratio(results_root, "unet_img_delta_s2026"),
        ),
        (
            "PredRNN++ image $\\Delta$",
            ratio(results_root, "predrnnpp_image_only_delta"),
            ratio(results_root, "predrnnpp_img_delta_s2026"),
        ),
    ]
    lines = [
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Configuration & fresh seed 1337 & fresh seed 2026 \\",
        r"\midrule",
    ]
    lines.extend(
        f"{label} & {first:.3f} & {second:.3f} \\\\"
        for label, first, second in rows
    )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def build_tables(results_root: Path, output_dir: Path) -> None:
    results_root = Path(results_root)
    output_dir = Path(output_dir)
    validate_required_runs(results_root)
    lopo = build_lopo_table(results_root)
    seeds = build_seed_table(results_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "lopo.tex").write_text(lopo)
    (output_dir / "seeds.tex").write_text(seeds)
    print(f"wrote {output_dir / 'lopo.tex'}")
    print(f"wrote {output_dir / 'seeds.tex'}")


if __name__ == "__main__":
    build_tables(DEFAULT_RESULTS, DEFAULT_OUTPUT)
