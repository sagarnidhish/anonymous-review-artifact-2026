#!/usr/bin/env python3
"""Rebuild particle-level comparison summaries using test rows only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


DETAIL_FILES = {
    "next_frame": "next_frame_results.json",
    "rollout": "rollout_results.json",
    "rollout_anchored": "rollout_results.json",
}


def summarize_rows(rows: list[dict]) -> dict:
    summary = {
        "count": len(rows),
        "mean_model_mae": float(np.mean([row["model_mae"] for row in rows])),
        "mean_naive_mae": float(np.mean([row["naive_mae"] for row in rows])),
        "mean_mae_ratio": float(np.mean([row["mae_ratio"] for row in rows])),
        "mean_model_rmse": float(
            np.mean([row["model_rmse"] for row in rows])
        ),
        "mean_naive_rmse": float(
            np.mean([row["naive_rmse"] for row in rows])
        ),
        "mean_rmse_ratio": float(
            np.mean([row["rmse_ratio"] for row in rows])
        ),
    }
    reversal = [
        row["reversal_metrics"]["mae_ratio"]
        for row in rows
        if row["reversal_metrics"]["count"] > 0
    ]
    if reversal:
        summary["mean_reversal_mae_ratio"] = float(np.mean(reversal))
    nonreversal = [
        row["nonreversal_metrics"]["mae_ratio"]
        for row in rows
        if row["nonreversal_metrics"]["count"] > 0
    ]
    if nonreversal:
        summary["mean_nonreversal_mae_ratio"] = float(np.mean(nonreversal))
    first_step = [
        row["first_step_mae_ratio"]
        for row in rows
        if "first_step_mae_ratio" in row
    ]
    if first_step:
        summary["mean_first_step_mae_ratio"] = float(np.mean(first_step))
    last_step = [
        row["last_step_mae_ratio"]
        for row in rows
        if "last_step_mae_ratio" in row
    ]
    if last_step:
        summary["mean_last_step_mae_ratio"] = float(np.mean(last_step))
    return summary


def rebuild_summary(detail_path: Path | str, output_path: Path | str) -> dict:
    detail = Path(detail_path)
    output = Path(output_path)
    rows = json.loads(detail.read_text())
    test_rows = [row for row in rows if row.get("role") == "test"]
    if not test_rows:
        raise ValueError(f"no test rows in {detail}")
    summary = summarize_rows(test_rows)
    summary.update({
        "role": "test",
        "aggregation_unit": "particle_movie",
        "source_detail_file": detail.name,
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps([summary], indent=2) + "\n")
    return summary


def rebuild_tree(results_root: Path, output_root: Path) -> list[dict]:
    rebuilt = []
    for tag_directory in sorted(path for path in results_root.iterdir()
                                if path.is_dir()):
        for mode, detail_name in DETAIL_FILES.items():
            detail = tag_directory / mode / detail_name
            if not detail.is_file():
                continue
            destination = (
                output_root / tag_directory.name / mode
                / "comparison_summary.json"
            )
            summary = rebuild_summary(detail, destination)
            rebuilt.append({
                "tag": tag_directory.name,
                "mode": mode,
                "count": summary["count"],
                "mean_mae_ratio": summary["mean_mae_ratio"],
            })
    return rebuilt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()

    rebuilt = rebuild_tree(args.results_root, args.output_root)
    inventory = args.output_root / "inventory.json"
    inventory.parent.mkdir(parents=True, exist_ok=True)
    inventory.write_text(json.dumps(rebuilt, indent=2) + "\n")
    print(f"rebuilt {len(rebuilt)} test-only summaries")


if __name__ == "__main__":
    main()
