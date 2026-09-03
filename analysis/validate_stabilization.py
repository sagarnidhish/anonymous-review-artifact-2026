#!/usr/bin/env python3
"""Fail-closed structural validator for the stabilization campaign."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.validate_matched_grid import validate_payload_directory  # noqa: E402
from train.stabilization_payloads import build_payloads  # noqa: E402


def validate_campaign(root: Path, payload_id: int | None = None) -> dict:
    payloads = build_payloads()
    if payload_id is not None:
        if payload_id < 0 or payload_id >= len(payloads):
            raise ValueError(f"payload ID outside 0..{len(payloads) - 1}")
        payloads = [payloads[payload_id]]
    summaries = []
    for payload in payloads:
        run_dir = Path(root) / payload["tag"]
        summary = validate_payload_directory(run_dir, payload)
        completion = json.loads(
            (run_dir / "completion_manifest.json").read_text(encoding="utf-8")
        )
        source_hash = completion.get("source_checkpoint_sha256", "")
        if not re.fullmatch(r"[0-9a-f]{64}", source_hash):
            raise ValueError(f"invalid source checkpoint hash: {run_dir}")
        if completion.get("selection_metric") != payload["selection_metric"]:
            raise ValueError(f"selection metric mismatch: {run_dir}")
        selected_epoch = completion.get("selected_epoch")
        if not isinstance(selected_epoch, int) or not 1 <= selected_epoch <= payload["epochs"]:
            raise ValueError(f"selected epoch is invalid: {run_dir}")
        history_path = run_dir / "models" / "training_history.json"
        if not history_path.is_file():
            raise ValueError(f"missing training history: {history_path}")
        history = json.loads(history_path.read_text(encoding="utf-8"))
        if history.get("source_checkpoint_sha256") != source_hash:
            raise ValueError(f"source checkpoint hash mismatch: {run_dir}")
        best = history.get("best", {})
        if best.get("epoch") != selected_epoch or not math.isfinite(
            best.get("validation_free_rollout_mae_h32", float("nan"))
        ):
            raise ValueError(f"selection record is invalid: {run_dir}")
        summaries.append(summary)
    return {
        "status": "complete",
        "payload_count": len(summaries),
        "particle_rollout_count": sum(
            row["rollout_particle_count"] for row in summaries
        ),
        "diverged_particle_count": sum(
            row["diverged_particle_count"] for row in summaries
        ),
        "payloads": summaries,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--payload-id", type=int)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    summary = validate_campaign(Path(args.root), args.payload_id)
    rendered = json.dumps(summary, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
