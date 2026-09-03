#!/usr/bin/env python3
"""Render paper tables from the validated fresh matched-grid summary."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path


FAMILIES = (
    "unet",
    "convlstm",
    "simvp",
    "residual_cnn",
    "predrnn",
    "predrnnpp",
)
FAMILY_LABELS = {
    "unet": "U-Net",
    "convlstm": "ConvLSTM",
    "simvp": "SimVP",
    "residual_cnn": "Residual CNN",
    "predrnn": "PredRNN",
    "predrnnpp": "PredRNN++",
}
HORIZONS = (32, 128, 256, 512)


def _ordered_rows(summary: dict) -> list[dict]:
    rows = sorted(summary.get("payloads", []), key=lambda row: row["payload_id"])
    if summary.get("status") != "complete" or len(rows) != 24:
        raise ValueError("a complete summary with 24 payload rows is required")
    if [row["payload_id"] for row in rows] != list(range(24)):
        raise ValueError("the 24 payload IDs must be complete and ordered")
    expected = {
        (family, input_mode, target_mode)
        for family in FAMILIES
        for input_mode in ("image_only", "protocol_conditioned")
        for target_mode in ("delta", "direct")
    }
    observed = {
        (row["model_family"], row["input_mode"], row["target_mode"])
        for row in rows
    }
    if observed != expected:
        raise ValueError("the 24 payload rows do not form the declared factorial grid")
    return rows


def _format_ratio(value: float, *, bold_skill: bool = True) -> str:
    value = math.inf if value is None else float(value)
    if math.isnan(value) or value <= 0:
        raise ValueError(f"invalid MAE ratio: {value}")
    if math.isinf(value):
        rendered = r"$\infty$"
    elif value >= 1000:
        exponent = int(math.floor(math.log10(value)))
        coefficient = value / (10**exponent)
        rendered = rf"{coefficient:.1f}$\times10^{{{exponent}}}$"
    elif value < 10:
        rendered = f"{value:.3f}"
    else:
        rendered = f"{value:.1f}"
    if bold_skill and value < 1.0:
        return rf"\textbf{{{rendered}}}"
    return rendered


def render_family_table(summary: dict) -> str:
    rows = _ordered_rows(summary)
    indexed = {
        (row["model_family"], row["input_mode"], row["target_mode"]): row
        for row in rows
    }
    lines = [
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"& \multicolumn{4}{c}{One-frame MAE ratio} & Best 256-step ratio \\",
        r"\cmidrule(lr){2-5}\cmidrule(lr){6-6}",
        r"Family & image $\Delta$ & protocol $\Delta$ & image direct & protocol direct & best of four forms \\",
        r"\midrule",
    ]
    combinations = (
        ("image_only", "delta"),
        ("protocol_conditioned", "delta"),
        ("image_only", "direct"),
        ("protocol_conditioned", "direct"),
    )
    for family in FAMILIES:
        family_rows = [indexed[(family, *combination)] for combination in combinations]
        cells = [
            _format_ratio(row["next_frame_mean_mae_ratio"])
            for row in family_rows
        ]
        best_h256 = min(
            math.inf
            if row["rollout_mean_mae_ratio_h256"] is None
            else float(row["rollout_mean_mae_ratio_h256"])
            for row in family_rows
        )
        cells.append(_format_ratio(best_h256))
        lines.append(FAMILY_LABELS[family] + " & " + " & ".join(cells) + r" \\")
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    return "\n".join(lines) + "\n"


def render_full_grid_table(summary: dict) -> str:
    rows = _ordered_rows(summary)
    lines = [
        r"\begin{tabular}{lllccccc}",
        r"\toprule",
        r"& & & \multicolumn{5}{c}{Particle-mean MAE ratio by forecast horizon} \\",
        r"\cmidrule(lr){4-8}",
        r"Family & Input & Target & One frame & $+32$ & $+128$ & $+256$ & $+512$ \\",
        r"\midrule",
    ]
    for row in rows:
        family = FAMILY_LABELS[row["model_family"]]
        input_label = "image" if row["input_mode"] == "image_only" else "protocol"
        target_label = r"$\Delta$" if row["target_mode"] == "delta" else "direct"
        if row.get("rollout_diverged_particles", 0):
            target_label += r" $\dagger$"
        cells = [_format_ratio(row["next_frame_mean_mae_ratio"])]
        cells.extend(
            _format_ratio(row[f"rollout_mean_mae_ratio_h{horizon}"])
            for horizon in HORIZONS
        )
        lines.append(
            " & ".join((family, input_label, target_label, *cells)) + r" \\"
        )
        if row["payload_id"] % 4 == 3 and row["payload_id"] != 23:
            lines.append(r"\addlinespace[1.5pt]")
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    return "\n".join(lines) + "\n"


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--family-output", required=True, type=Path)
    parser.add_argument("--full-output", required=True, type=Path)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    _write_atomic(args.family_output, render_family_table(summary))
    _write_atomic(args.full_output, render_full_grid_table(summary))
    print(f"wrote {args.family_output} and {args.full_output}")


if __name__ == "__main__":
    main()
