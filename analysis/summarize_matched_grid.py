#!/usr/bin/env python3
"""Reduce a validated 24-payload grid to figure- and table-ready records."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.validate_matched_grid import TEST_STEMS, read_json, validate_grid  # noqa: E402
from train.matched_grid_payloads import build_payloads  # noqa: E402


def _validate_particle_rows(payload: dict, rows: list[dict], kind: str) -> None:
    if len(rows) != 4 or {row.get("stem") for row in rows} != TEST_STEMS:
        raise ValueError(f"{kind} must contain the four declared test particles")
    if any(row.get("payload_id") != payload["payload_id"] for row in rows):
        raise ValueError(f"{kind} payload ID mismatch")


def summarize_payload_rows(
    payload: dict, next_rows: list[dict], rollout_rows: list[dict]
) -> dict:
    """Summarize one payload without pooling correlated image windows."""
    _validate_particle_rows(payload, next_rows, "next-frame rows")
    _validate_particle_rows(payload, rollout_rows, "rollout rows")
    next_by_stem = {row["stem"]: row for row in next_rows}
    rollout_by_stem = {row["stem"]: row for row in rollout_rows}

    next_ratios = np.asarray(
        [next_by_stem[stem]["mae_ratio"] for stem in sorted(TEST_STEMS)],
        dtype=float,
    )
    if not np.isfinite(next_ratios).all():
        raise ValueError("non-finite next-frame particle ratio")

    summary = {
        "payload_id": payload["payload_id"],
        "tag": payload["tag"],
        "model_family": payload["model_family"],
        "input_mode": payload["input_mode"],
        "target_mode": payload["target_mode"],
        "next_frame_mean_mae_ratio": float(np.mean(next_ratios)),
        "next_frame_particles_better": int(np.sum(next_ratios < 1.0)),
        "rollout_complete_particles": sum(
            rollout_by_stem[stem].get("status") == "complete"
            for stem in sorted(TEST_STEMS)
        ),
        "rollout_diverged_particles": sum(
            rollout_by_stem[stem].get("status") == "numerically_diverged"
            for stem in sorted(TEST_STEMS)
        ),
    }

    for horizon in payload["report_horizons"]:
        ratios = []
        for stem in sorted(TEST_STEMS):
            horizons = rollout_by_stem[stem].get("horizons", {})
            if str(horizon) not in horizons:
                raise ValueError(
                    f"missing rollout horizon {horizon} for {payload['tag']}:{stem}"
                )
            raw_ratio = horizons[str(horizon)].get("mae_ratio", math.nan)
            if raw_ratio is None:
                if rollout_by_stem[stem].get("status") != "numerically_diverged":
                    raise ValueError(
                        f"null rollout ratio without divergence for "
                        f"{payload['tag']}:{stem}:h{horizon}"
                    )
                ratio = math.inf
            else:
                ratio = float(raw_ratio)
            if math.isnan(ratio):
                raise ValueError(
                    f"NaN rollout horizon {horizon} for {payload['tag']}:{stem}"
                )
            ratios.append(ratio)
        summary[f"rollout_mean_mae_ratio_h{horizon}"] = float(np.mean(ratios))
        summary[f"rollout_particles_better_h{horizon}"] = sum(
            ratio < 1.0 for ratio in ratios
        )
    return summary


def summarize_grid(root: Path) -> dict:
    root = Path(root)
    validation = validate_grid(root)
    payload_summaries = []
    for payload in build_payloads():
        run = root / payload["tag"]
        payload_summaries.append(
            summarize_payload_rows(
                payload,
                read_json(run / "next_frame" / "next_frame_results.json"),
                read_json(run / "rollout_anchored" / "rollout_results.json"),
            )
        )
    return {
        "status": "complete",
        "validated_payload_count": validation["payload_count"],
        "next_frame_skilled_configurations": sum(
            row["next_frame_mean_mae_ratio"] < 1.0 for row in payload_summaries
        ),
        "rollout_h512_skilled_configurations": sum(
            row["rollout_mean_mae_ratio_h512"] < 1.0 for row in payload_summaries
        ),
        "rollout_h512_particle_comparisons_better": sum(
            row["rollout_particles_better_h512"] for row in payload_summaries
        ),
        "diverged_particle_rollouts": sum(
            row["rollout_diverged_particles"] for row in payload_summaries
        ),
        "payloads": payload_summaries,
    }


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _csv_text(rows: list[dict]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def strict_json_text(value) -> str:
    """Render RFC-compliant JSON while preserving divergence as null."""

    def sanitize(item):
        if isinstance(item, dict):
            return {key: sanitize(val) for key, val in item.items()}
        if isinstance(item, (list, tuple)):
            return [sanitize(val) for val in item]
        if isinstance(item, (float, np.floating)):
            return None if not np.isfinite(item) else float(item)
        if isinstance(item, np.integer):
            return int(item)
        return item

    return json.dumps(sanitize(value), indent=2, allow_nan=False) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--json-output", required=True)
    parser.add_argument("--csv-output", required=True)
    args = parser.parse_args()
    summary = summarize_grid(Path(args.root))
    _write_text_atomic(
        Path(args.json_output), strict_json_text(summary)
    )
    _write_text_atomic(Path(args.csv_output), _csv_text(summary["payloads"]))
    print(json.dumps({key: value for key, value in summary.items() if key != "payloads"}, indent=2))


if __name__ == "__main__":
    main()
