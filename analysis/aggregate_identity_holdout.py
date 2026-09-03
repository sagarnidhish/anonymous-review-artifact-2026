#!/usr/bin/env python3
"""Aggregate validated GRA29 identity-holdout runs at particle level."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analysis.validate_identity_holdout import (
    MODEL_FAMILIES,
    validate_identity_holdout,
)


GROUPS = (
    "same_temperature_unseen_particle",
    "cross_temperature_unseen_particle",
)


def aggregate_identity_holdout(root: Path) -> dict:
    records = validate_identity_holdout(Path(root))
    output = {
        "aggregation_unit": "physical_particle",
        "shared_cell_level_protocol": True,
        "particle_count": 4,
        "models": {},
        "per_particle": [],
    }
    for record in records:
        output["per_particle"].append(
            {
                "model_family": record.model_family,
                "heldout_particle": record.heldout_particle,
                "evaluation_group": record.evaluation_group,
                "mode": record.mode,
                "stem": record.stem,
                "mae_ratio": record.mae_ratio,
            }
        )
    for model_family in MODEL_FAMILIES:
        output["models"][model_family] = {}
        for group in GROUPS:
            selected = [
                record
                for record in records
                if record.model_family == model_family
                and record.evaluation_group == group
            ]
            by_mode = {
                mode: [record.mae_ratio for record in selected if record.mode == mode]
                for mode in ("next_frame", "rollout")
            }
            if any(len(values) != 4 for values in by_mode.values()):
                raise AssertionError(
                    f"validated aggregation lost a particle for {model_family}/{group}"
                )
            output["models"][model_family][group] = {
                "next_frame_mean_mae_ratio": float(np.mean(by_mode["next_frame"])),
                "next_frame_particles_better_than_persistence": int(
                    np.count_nonzero(np.asarray(by_mode["next_frame"]) < 1.0)
                ),
                "rollout_mean_mae_ratio": float(np.mean(by_mode["rollout"])),
                "rollout_particles_better_than_persistence": int(
                    np.count_nonzero(np.asarray(by_mode["rollout"]) < 1.0)
                ),
            }
    return output


def render_latex_table(summary: dict) -> str:
    labels = {"unet": "U-Net", "predrnnpp": "PredRNN++"}

    def cell(result: dict, mode: str) -> str:
        return (
            f"{result[f'{mode}_mean_mae_ratio']:.3f} "
            f"({result[f'{mode}_particles_better_than_persistence']}/4)"
        )

    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"& \multicolumn{2}{c}{Same temperature, unseen particle} & \multicolumn{2}{c}{Cross temperature, unseen particle} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"Model & next frame & 512-step rollout & next frame & 512-step rollout \\",
        r"\midrule",
    ]
    for model in MODEL_FAMILIES:
        same = summary["models"][model]["same_temperature_unseen_particle"]
        cross = summary["models"][model]["cross_temperature_unseen_particle"]
        lines.append(
            f"{labels[model]} & "
            f"{cell(same, 'next_frame')} & "
            f"{cell(same, 'rollout')} & "
            f"{cell(cross, 'next_frame')} & "
            f"{cell(cross, 'rollout')} " + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--table", required=True, type=Path)
    args = parser.parse_args()
    summary = aggregate_identity_holdout(args.root)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    args.table.write_text(render_latex_table(summary))
    print(f"wrote {args.summary} and {args.table}")


if __name__ == "__main__":
    main()
