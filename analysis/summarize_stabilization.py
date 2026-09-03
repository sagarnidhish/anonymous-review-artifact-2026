#!/usr/bin/env python3
"""Validate and summarize the focused rollout-stabilization campaign."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

from analysis.validate_stabilization import validate_campaign
from analysis.summarize_matched_grid import strict_json_text
from train.stabilization_payloads import build_payloads


LABELS = {
    "input_noise": "Input noise",
    "scheduled_sampling": "Scheduled sampling (32 steps)",
    "recursive_unroll": "Recursive loss (32 steps)",
}
METRIC_KEYS = (
    "next_frame_mean_mae_ratio",
    "rollout_mean_mae_ratio_h32",
    "rollout_mean_mae_ratio_h128",
    "rollout_mean_mae_ratio_h256",
    "rollout_mean_mae_ratio_h512",
    "rollout_diverged_particles",
)


def assemble_stabilization_rows(
    grid_summary: dict, strategy_summaries: list[dict]
) -> list[dict]:
    grid_rows = grid_summary.get("payloads", [])
    if grid_summary.get("status") != "complete" or len(grid_rows) != 24:
        raise ValueError("fresh grid summary must contain 24 payload rows")
    baseline_candidates = [row for row in grid_rows if row.get("payload_id") == 0]
    if len(baseline_candidates) != 1:
        raise ValueError("fresh grid payload 0 is missing or duplicated")
    if len(strategy_summaries) != 3:
        raise ValueError("exactly three stabilization strategy summaries are required")
    strategies = {row.get("strategy"): row for row in strategy_summaries}
    if set(strategies) != set(LABELS):
        raise ValueError("stabilization strategy set is incomplete")

    baseline = baseline_candidates[0]
    rows = [
        {
            "label": "Fresh one-step U-Net",
            "source": "fresh_matched_grid_payload_0",
            **{key: baseline[key] for key in METRIC_KEYS},
        }
    ]
    for strategy in ("input_noise", "scheduled_sampling", "recursive_unroll"):
        summary = strategies[strategy]
        rows.append(
            {
                "label": LABELS[strategy],
                "source": "v5_stabilization",
                "strategy": strategy,
                **{key: summary[key] for key in METRIC_KEYS},
            }
        )
    return rows


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _mean_ratio(values) -> float:
    normalized = [math.inf if value is None else float(value) for value in values]
    if any(math.isnan(value) or value <= 0 for value in normalized):
        raise ValueError("stabilization ratios must be positive or divergent")
    return float(np.mean(normalized))


def summarize_strategy(root: Path, payload: dict) -> dict:
    run = root / payload["tag"]
    next_rows = _read_json(run / "next_frame" / "next_frame_results.json")
    rollout_rows = _read_json(run / "rollout_anchored" / "rollout_results.json")
    if len(next_rows) != 4 or len(rollout_rows) != 4:
        raise ValueError(f"strategy does not contain four particle rows: {run}")
    summary = {
        "payload_id": payload["payload_id"],
        "tag": payload["tag"],
        "strategy": payload["strategy"],
        "next_frame_mean_mae_ratio": _mean_ratio(
            row["mae_ratio"] for row in next_rows
        ),
        "rollout_diverged_particles": sum(
            row.get("status") == "numerically_diverged" for row in rollout_rows
        ),
    }
    for horizon in payload["report_horizons"]:
        summary[f"rollout_mean_mae_ratio_h{horizon}"] = _mean_ratio(
            row["horizons"][str(horizon)]["mae_ratio"] for row in rollout_rows
        )
    return summary


def _write_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(strict_json_text(value), encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--grid-summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    validation = validate_campaign(args.root)
    strategies = [summarize_strategy(args.root, payload) for payload in build_payloads()]
    grid_summary = _read_json(args.grid_summary)
    artifact = {
        "status": "complete",
        "validation": validation,
        "rows": assemble_stabilization_rows(grid_summary, strategies),
    }
    _write_atomic(args.output, artifact)
    print(json.dumps({"status": "complete", "row_count": len(artifact["rows"])}, indent=2))


if __name__ == "__main__":
    main()
