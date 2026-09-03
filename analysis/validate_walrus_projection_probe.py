#!/usr/bin/env python3
"""Validate the gated Walrus intensity-projection experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


TRAIN_STEMS = {f"GRA29_C20_25deg_particle{i}" for i in range(1, 5)}
TEST_STEMS = {f"GRA29_C20_45deg_particle{i}" for i in range(1, 5)}


def read_json(path: Path):
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"missing required JSON: {path}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def validate_probe(root: Path) -> dict:
    root = Path(root)
    completion = read_json(root / "completion_manifest.json")
    selector = read_json(root / "selector_manifest.json")
    tiny = read_json(root / "tiny_overfit_gate.json")
    inventory = selector.get("inventory", {})
    if len(inventory) != 6:
        raise ValueError(f"expected six projection containers, found {len(inventory)}")
    if not selector.get("all_other_parameters_frozen"):
        raise ValueError("non-projection parameters were not all frozen")
    if set(selector.get("pretrained_projection_hashes_before", {})) != set(inventory):
        raise ValueError("pretrained projection hashes are incomplete")

    status = completion.get("status")
    if status == "tiny_overfit_gate_failed":
        if tiny.get("status") != "failed" or completion.get("tiny_gate_passed"):
            raise ValueError("tiny-gate failure manifests disagree")
        return {
            "status": status,
            "projection_container_count": len(inventory),
            "validation_improved": False,
        }

    if tiny.get("status") != "passed" or not completion.get("tiny_gate_passed"):
        raise ValueError("full probe exists without a passed tiny-overfit gate")
    before = tiny.get("before", {}).get("mean_model_mae")
    after = tiny.get("after", {}).get("mean_model_mae")
    if not all(np.isfinite(value) for value in (before, after)) or not after < before:
        raise ValueError("tiny-overfit loss did not decrease")
    if not tiny.get("pretrained_projection_hashes_unchanged"):
        raise ValueError("tiny gate changed pretrained projection entries")

    training = read_json(root / "training_curve.json")
    split = training.get("split", {})
    if set(split.get("train_stems", [])) != TRAIN_STEMS:
        raise ValueError("training particle set is incorrect")
    if set(split.get("test_stems", [])) != TEST_STEMS:
        raise ValueError("test particle set is incorrect")
    if split.get("train_start_range") != [0, 2699]:
        raise ValueError("training start range is incorrect")
    if split.get("validation_start_range") != [2700, 2999]:
        raise ValueError("validation start range is incorrect")
    gradient_names = set(training.get("gradient_parameter_names", []))
    if not gradient_names.issubset(inventory):
        raise ValueError("gradient inventory contains an undeclared parameter")
    if not any("embed." in name for name in gradient_names):
        raise ValueError("no encoder projection received a gradient")
    if not any("debed." in name for name in gradient_names):
        raise ValueError("no decoder projection received a gradient")
    for evaluation_name in ("baseline_test", "fitted_test"):
        rows = training.get(evaluation_name, {}).get("particle_rows", [])
        if {row.get("stem") for row in rows} != TEST_STEMS:
            raise ValueError(f"{evaluation_name} particle rows are incomplete")

    if status not in {"passed", "validation_gate_failed"}:
        raise ValueError(f"unexpected completion status: {status}")
    baseline = completion.get("baseline_validation", {}).get("mean_model_mae")
    fitted = completion.get("fitted_validation", {}).get("mean_model_mae")
    if not all(np.isfinite(value) for value in (baseline, fitted)):
        raise ValueError("validation metrics are non-finite")
    improved = fitted < baseline
    if status == "passed" and not improved:
        raise ValueError("passed probe did not improve validation error")
    if status == "validation_gate_failed" and improved:
        raise ValueError("failed probe improved validation error")
    if completion.get("validation_improved") != improved:
        raise ValueError("validation-improvement flag disagrees with metrics")
    if not completion.get("pretrained_projection_hashes_unchanged"):
        raise ValueError("full probe changed pretrained projection entries")

    checkpoint_path = root / completion.get("projection_checkpoint", "")
    if not checkpoint_path.is_file() or checkpoint_path.stat().st_size == 0:
        raise ValueError("projection checkpoint is missing")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("inventory") != inventory:
        raise ValueError("checkpoint projection inventory differs from manifest")
    if checkpoint.get("intensity_index") != selector.get("intensity_index"):
        raise ValueError("checkpoint intensity index differs from manifest")
    entries = checkpoint.get("projection_entries", {})
    if set(entries) != set(inventory):
        raise ValueError("checkpoint projection entries are incomplete")
    for name, tensor in entries.items():
        if tensor.numel() != inventory[name]["active_entries"]:
            raise ValueError(f"projection entry size mismatch for {name}")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"non-finite projection entry for {name}")

    return {
        "status": status,
        "projection_container_count": len(inventory),
        "active_entry_count": sum(
            metadata["active_entries"] for metadata in inventory.values()
        ),
        "validation_improved": improved,
        "baseline_validation_mae": baseline,
        "fitted_validation_mae": fitted,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    summary = validate_probe(Path(args.root))
    rendered = json.dumps(summary, indent=2)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
