#!/usr/bin/env python3
"""Render Version-5 control tables from validated JSON artifacts."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path


MODEL_LABELS = {"unet": "U-Net", "predrnnpp": "PredRNN++"}
STABILIZATION_LABELS = (
    "Fresh one-step U-Net",
    "Input noise",
    "Scheduled sampling (32 steps)",
    "Recursive loss (32 steps)",
)


def _number(value: float, digits: int = 3) -> str:
    value = math.inf if value is None else float(value)
    if not math.isfinite(value):
        return r"$\infty$"
    return f"{value:.{digits}f}"


def _ratio(value: float) -> str:
    value = math.inf if value is None else float(value)
    rendered = _number(value)
    return rf"\textbf{{{rendered}}}" if value < 1.0 else rendered


def render_walrus_probe_table(probe: dict) -> str:
    if probe.get("status") != "passed" or not probe.get("validation_improved"):
        raise ValueError("Walrus projection probe did not pass its validation gate")
    rows = (
        (
            "Unfitted graft",
            probe["baseline_validation"]["mean_model_mae"],
            probe["baseline_test"]["mean_model_mae"],
            probe["baseline_test"]["mean_particle_mae_ratio"],
        ),
        (
            r"Fitted on 25\,$^\circ$C",
            probe["fitted_validation"]["mean_model_mae"],
            probe["fitted_test"]["mean_model_mae"],
            probe["fitted_test"]["mean_particle_mae_ratio"],
        ),
    )
    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"One-step projection & 25\,$^\circ$C validation MAE & 45\,$^\circ$C test MAE & Test MAE ratio \\",
        r"\midrule",
    ]
    for label, validation_mae, test_mae, test_ratio in rows:
        lines.append(
            " & ".join(
                (
                    label,
                    _number(validation_mae, 5),
                    _number(test_mae, 5),
                    _ratio(test_ratio),
                )
            )
            + r" \\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    return "\n".join(lines) + "\n"


def render_normalization_table(artifact: dict) -> str:
    payloads = artifact.get("payload_results", [])
    if len(payloads) != 2:
        raise ValueError("normalization artifact must contain two payload results")
    lines = [
        r"\begin{tabular}{llcc}",
        r"\toprule",
        r"Model & Test scaling reference & One-frame ratio & $+512$ ratio \\",
        r"\midrule",
    ]
    modes = (
        ("archived_test_record", "Full test record"),
        (
            "paired_training_reference",
            r"Paired 25\,$^\circ$C training record",
        ),
    )
    for payload_index, payload_result in enumerate(payloads):
        family = payload_result["payload"]["model_family"]
        if family not in MODEL_LABELS:
            raise ValueError(f"unexpected normalization model family: {family}")
        for mode, mode_label in modes:
            result = payload_result["modes"][mode]
            one_frame = result["next_frame"]["mean_particle_mae_ratio"]
            h512 = result["rollout_anchored"]["mean_particle_mae_ratio"]["512"]
            lines.append(
                " & ".join(
                    (MODEL_LABELS[family], mode_label, _ratio(one_frame), _ratio(h512))
                )
                + r" \\"
            )
        if payload_index == 0:
            lines.append(r"\addlinespace[1.5pt]")
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    return "\n".join(lines) + "\n"


def render_stabilization_table(rows: list[dict]) -> str:
    if len(rows) != 4 or tuple(row.get("label") for row in rows) != STABILIZATION_LABELS:
        raise ValueError("stabilization table requires the baseline and three declared strategies")
    lines = [
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Training strategy & One frame & $+32$ & $+128$ & $+256$ & $+512$ \\",
        r"\midrule",
    ]
    for row in rows:
        label = row["label"]
        if row.get("rollout_diverged_particles", 0):
            label += r" $\dagger$"
        values = [_ratio(row["next_frame_mean_mae_ratio"])]
        values.extend(
            _ratio(row[f"rollout_mean_mae_ratio_h{horizon}"])
            for horizon in (32, 128, 256, 512)
        )
        lines.append(" & ".join((label, *values)) + r" \\")
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    return "\n".join(lines) + "\n"


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--walrus-probe", required=True, type=Path)
    parser.add_argument("--walrus-output", required=True, type=Path)
    parser.add_argument("--normalization", required=True, type=Path)
    parser.add_argument("--normalization-output", required=True, type=Path)
    parser.add_argument("--stabilization-summary", required=True, type=Path)
    parser.add_argument("--stabilization-output", required=True, type=Path)
    args = parser.parse_args()
    _write_atomic(args.walrus_output, render_walrus_probe_table(_load(args.walrus_probe)))
    _write_atomic(
        args.normalization_output,
        render_normalization_table(_load(args.normalization)),
    )
    stabilization = _load(args.stabilization_summary)
    _write_atomic(
        args.stabilization_output,
        render_stabilization_table(stabilization["rows"]),
    )
    print("wrote Version-5 control tables")


if __name__ == "__main__":
    main()
