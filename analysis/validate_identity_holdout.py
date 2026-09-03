#!/usr/bin/env python3
"""Fail-closed validation for the four-fold GRA29 identity holdout study."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from train.particle_splits import build_identity_holdout_fold
from train.protocol_evaluation_utils import ROLLOUT_ANCHOR_RULE


MODEL_FAMILIES = ("unet", "predrnnpp")
MODES = ("next_frame", "rollout")


class IdentityHoldoutValidationError(ValueError):
    """Raised when a saved run cannot support the declared fold result."""


@dataclass(frozen=True)
class ValidatedResult:
    model_family: str
    heldout_particle: int
    evaluation_group: str
    mode: str
    stem: str
    model_mae: float
    persistence_mae: float
    mae_ratio: float
    artifact_path: str
    result_path: str


def _fail(message: str) -> None:
    raise IdentityHoldoutValidationError(message)


def _load_single_checkpoint(run_dir: Path, model_family: str) -> dict:
    checkpoints = sorted((run_dir / "models").glob("*_best.pt"))
    if len(checkpoints) != 1:
        _fail(
            f"expected one checkpoint for {run_dir.name}; found {len(checkpoints)}"
        )
    return torch.load(checkpoints[0], map_location="cpu", weights_only=False)


def _validate_checkpoint(
    run_dir: Path, model_family: str, heldout_particle: int
) -> None:
    metadata = _load_single_checkpoint(run_dir, model_family)
    fold = build_identity_holdout_fold(heldout_particle)
    if metadata.get("model_family") != model_family:
        _fail(f"checkpoint model family mismatch in {run_dir.name}")
    if metadata.get("split") != "identity_holdout":
        _fail(f"checkpoint split is not identity_holdout in {run_dir.name}")
    if metadata.get("heldout_particle") != heldout_particle:
        _fail(f"checkpoint held-out particle mismatch in {run_dir.name}")
    if metadata.get("rollout_anchor_rule") != ROLLOUT_ANCHOR_RULE:
        _fail(f"checkpoint rollout anchor is missing in {run_dir.name}")

    train_stems = set(metadata.get("train_stems", []))
    if any(
        stem.endswith(f"particle{heldout_particle}") for stem in train_stems
    ):
        _fail(f"held-out particle {heldout_particle} appears in training")
    if train_stems != set(fold.train_stems):
        _fail(f"training stems do not match fold {heldout_particle}")
    if set(metadata.get("test_stems", [])) != set(fold.all_test_stems):
        _fail(f"test stems do not match fold {heldout_particle}")


def _scalar(array: np.ndarray, name: str, artifact: Path) -> float:
    if np.asarray(array).size != 1:
        _fail(f"{name} is not scalar in {artifact}")
    return float(np.asarray(array).reshape(()))


def _validate_saved_result(
    run_dir: Path,
    model_family: str,
    heldout_particle: int,
    mode: str,
    stem: str,
    expected_group: str,
) -> ValidatedResult:
    stem_dir = run_dir / mode / stem
    if mode == "next_frame":
        result_path = stem_dir / "next_frame_results.json"
        artifact_path = stem_dir / f"preds_{stem}.npz"
    else:
        result_path = stem_dir / "rollout_results.json"
        artifact_path = stem_dir / f"rollout_{stem}.npz"
    if not result_path.is_file() or not artifact_path.is_file():
        _fail(f"missing {mode} artifact for {run_dir.name}/{stem}")

    rows = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or len(rows) != 1:
        _fail(f"expected one result row in {result_path}")
    row = rows[0]
    if row.get("evaluation_group") != expected_group:
        _fail(f"missing or incorrect evaluation_group in {result_path}")
    expected_metadata = {
        "stem": stem,
        "role": "test",
        "model_family": model_family,
        "tag": run_dir.name,
        "split": "identity_holdout",
        "seed": 1337,
        "prediction_form": "delta_from_last_frame",
        "heldout_particle": heldout_particle,
    }
    for key, value in expected_metadata.items():
        if row.get(key) != value:
            _fail(f"incorrect {key} in {result_path}")
    if mode == "rollout":
        if row.get("anchor_rule") != ROLLOUT_ANCHOR_RULE:
            _fail(f"rollout anchor rule is missing in {result_path}")
        anchor_frame = row.get("anchor_frame")
        onset_frame = row.get("onset_frame")
        if (
            not isinstance(anchor_frame, int)
            or not isinstance(onset_frame, int)
            or anchor_frame < 0
            or onset_frame - anchor_frame != 4
        ):
            _fail(f"rollout anchor frames are invalid in {result_path}")

    with np.load(artifact_path, allow_pickle=False) as loaded:
        pred = loaded["pred"].astype(np.float64)
        naive = loaded["naive"].astype(np.float64)
        targets = loaded["targets"].astype(np.float64)
        target_steps = loaded["target_steps"].astype(np.int64)
        saved_npz_model_mae = _scalar(
            loaded["model_mae"], "model_mae", artifact_path
        )
        saved_npz_naive_mae = _scalar(
            loaded["naive_mae"], "naive_mae", artifact_path
        )
        saved_npz_ratio = _scalar(
            loaded["mae_ratio"], "mae_ratio", artifact_path
        )

    if pred.shape != naive.shape or pred.shape != targets.shape or pred.ndim != 3:
        _fail(f"prediction, persistence, and target shapes differ in {artifact_path}")
    if pred.shape[0] == 0 or not all(
        np.isfinite(array).all() for array in (pred, naive, targets)
    ):
        _fail(f"empty or non-finite arrays in {artifact_path}")
    if mode == "rollout" and not np.array_equal(
        naive, np.broadcast_to(naive[0], naive.shape)
    ):
        _fail(f"rollout persistence is not fixed in {artifact_path}")
    if mode == "rollout" and (
        len(target_steps) != len(pred) or int(target_steps[0]) != 4
    ):
        _fail(f"rollout target steps do not follow the anchor in {artifact_path}")

    model_mae = float(np.mean(np.abs(pred - targets)))
    persistence_mae = float(np.mean(np.abs(naive - targets)))
    if persistence_mae <= 0:
        _fail(f"persistence MAE is not positive in {artifact_path}")
    ratio = model_mae / persistence_mae
    comparisons = (
        (saved_npz_model_mae, model_mae, "NPZ model MAE"),
        (saved_npz_naive_mae, persistence_mae, "NPZ persistence MAE"),
        (saved_npz_ratio, ratio, "NPZ MAE ratio"),
        (row.get("model_mae"), model_mae, "saved model MAE"),
        (row.get("naive_mae"), persistence_mae, "saved persistence MAE"),
        (row.get("mae_ratio"), ratio, "saved MAE ratio"),
    )
    for saved, recomputed, label in comparisons:
        if saved is None or not np.isclose(
            float(saved), recomputed, rtol=1e-5, atol=1e-7
        ):
            _fail(f"{label} does not match arrays in {artifact_path}")

    return ValidatedResult(
        model_family=model_family,
        heldout_particle=heldout_particle,
        evaluation_group=expected_group,
        mode=mode,
        stem=stem,
        model_mae=model_mae,
        persistence_mae=persistence_mae,
        mae_ratio=ratio,
        artifact_path=str(artifact_path),
        result_path=str(result_path),
    )


def validate_identity_holdout(root: Path) -> list[ValidatedResult]:
    """Validate all eight runs and return 32 per-condition result records."""
    root = Path(root)
    records: list[ValidatedResult] = []
    for model_family in MODEL_FAMILIES:
        for heldout_particle in range(1, 5):
            tag = f"identity_holdout_{model_family}_p{heldout_particle}"
            run_dir = root / tag
            if not run_dir.is_dir():
                _fail(f"missing run {tag}")
            _validate_checkpoint(run_dir, model_family, heldout_particle)
            fold = build_identity_holdout_fold(heldout_particle)
            grouped_stems = (
                (
                    "same_temperature_unseen_particle",
                    fold.same_temperature_test_stems,
                ),
                (
                    "cross_temperature_unseen_particle",
                    fold.cross_temperature_test_stems,
                ),
            )
            for group, stems in grouped_stems:
                for stem in stems:
                    for mode in MODES:
                        records.append(
                            _validate_saved_result(
                                run_dir,
                                model_family,
                                heldout_particle,
                                mode,
                                stem,
                                group,
                            )
                        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    records = validate_identity_holdout(args.root)
    payload = [asdict(record) for record in records]
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"validated {len(records)} identity-holdout results")


if __name__ == "__main__":
    main()
