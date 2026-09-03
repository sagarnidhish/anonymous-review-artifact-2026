#!/usr/bin/env python3
"""Aggregate protocol-forcing controls with particle/movie as the unit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


CONDITIONS = ("true", "zero", "shuffle", "shift")
MODES = ("next_frame", "rollout_anchored")


def _load_rows(results_root: Path) -> list[dict]:
    path = Path(results_root) / "per_particle_results.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = json.loads(path.read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError("protocol intervention rows are empty")
    return rows


def summarize(results_root: Path) -> dict:
    rows = _load_rows(Path(results_root))
    stems = sorted({str(row["stem"]) for row in rows})
    by_key = {
        (str(row["stem"]), str(row["mode"]), str(row["condition"])): row
        for row in rows
    }
    expected = {
        (stem, mode, condition)
        for stem in stems
        for mode in MODES
        for condition in CONDITIONS
    }
    missing = sorted(expected - set(by_key))
    extras = sorted(set(by_key) - expected)
    if missing or extras or len(by_key) != len(rows):
        raise ValueError(
            "incomplete intervention grid: "
            f"missing={missing}; unexpected_or_duplicate={extras}"
        )

    payload = {
        "statistical_unit": "particle_movie",
        "n_particles": len(stems),
        "stems": stems,
        "conditions": list(CONDITIONS),
        "modes": {},
    }
    for mode in MODES:
        mode_payload = {}
        true = {
            stem: float(by_key[(stem, mode, "true")]["mae_ratio"])
            for stem in stems
        }
        for condition in CONDITIONS:
            selected = [by_key[(stem, mode, condition)] for stem in stems]
            ratios = [float(row["mae_ratio"]) for row in selected]
            deltas = [ratio - true[stem] for stem, ratio in zip(stems, ratios)]
            response = [
                float(row["prediction_change_l1"]) for row in selected
            ]
            mode_payload[condition] = {
                "n_particles": len(stems),
                "mean_mae_ratio": mean(ratios),
                "mae_ratio": ratios,
                "mean_paired_delta_mae_ratio": mean(deltas),
                "paired_delta_mae_ratio": deltas,
                "mean_prediction_change_l1": mean(response),
                "prediction_change_l1": response,
            }
        payload["modes"][mode] = mode_payload
    return payload


def latex_table(payload: dict) -> str:
    labels = {
        "true": "Measured protocol",
        "zero": "Zero voltage/current",
        "shuffle": "Shuffled voltage/current",
        "shift": "Delayed voltage/current",
    }
    lines = [
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Mode & Protocol input & $n$ & MAE ratio & paired $\Delta$ \\",
        r"\midrule",
    ]
    for mode in MODES:
        mode_label = "Next frame" if mode == "next_frame" else "Anchored rollout"
        for index, condition in enumerate(CONDITIONS):
            entry = payload["modes"][mode][condition]
            prefix = mode_label if index == 0 else ""
            lines.append(
                f"{prefix} & {labels[condition]} & {entry['n_particles']} & "
                f"{entry['mean_mae_ratio']:.3f} & "
                f"{entry['mean_paired_delta_mae_ratio']:+.3f} \\\\"
            )
        if mode != MODES[-1]:
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--table", type=Path)
    args = parser.parse_args()

    payload = summarize(args.results_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.output}")
    if args.table:
        args.table.parent.mkdir(parents=True, exist_ok=True)
        args.table.write_text(latex_table(payload))
        print(f"wrote {args.table}")


if __name__ == "__main__":
    main()
