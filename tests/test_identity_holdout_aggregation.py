import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from analysis.aggregate_identity_holdout import (
    aggregate_identity_holdout,
    render_latex_table,
)
from analysis.validate_identity_holdout import IdentityHoldoutValidationError


MODELS = ("unet", "predrnnpp")
GROUPS = {
    "25deg": "same_temperature_unseen_particle",
    "45deg": "cross_temperature_unseen_particle",
}


def _write_mode_artifact(
    run_dir: Path,
    mode: str,
    stem: str,
    model: str,
    particle: int,
    group: str,
) -> None:
    directory = run_dir / mode / stem
    directory.mkdir(parents=True, exist_ok=True)
    targets = np.zeros((3, 2, 2), dtype=np.float32)
    naive = np.ones_like(targets)
    if mode == "next_frame":
        pred_value = 0.5 if group.startswith("same_") else 1.5
        filename = f"preds_{stem}.npz"
    else:
        pred_value = 2.0 if group.startswith("same_") else 3.0
        filename = f"rollout_{stem}.npz"
    pred = np.full_like(targets, pred_value)
    ratio = pred_value
    np.savez_compressed(
        directory / filename,
        pred=pred,
        naive=naive,
        targets=targets,
        target_steps=(
            np.arange(4, 7, dtype=np.int64)
            if mode == "rollout"
            else np.arange(3, dtype=np.int64)
        ),
        frame_times=np.arange(3, dtype=np.float32),
        active_fields=np.asarray(["intensity"]),
        model_family=model,
        tag=f"identity_holdout_{model}_p{particle}",
        prediction_form="delta_from_last_frame",
        model_mae=np.asarray(pred_value, dtype=np.float32),
        naive_mae=np.asarray(1.0, dtype=np.float32),
        mae_ratio=np.asarray(ratio, dtype=np.float32),
        model_mse=np.asarray(pred_value**2, dtype=np.float32),
        naive_mse=np.asarray(1.0, dtype=np.float32),
        mse_ratio=np.asarray(pred_value**2, dtype=np.float32),
    )
    row = {
        "stem": stem,
        "role": "test",
        "model_family": model,
        "tag": f"identity_holdout_{model}_p{particle}",
        "active_fields": ["intensity"],
        "context_len": 4,
        "split": "identity_holdout",
        "seed": 1337,
        "prediction_form": "delta_from_last_frame",
        "heldout_particle": particle,
        "evaluation_group": group,
        "model_mae": pred_value,
        "naive_mae": 1.0,
        "mae_ratio": ratio,
        "model_mse": pred_value**2,
        "naive_mse": 1.0,
        "mse_ratio": pred_value**2,
        "model_rmse": pred_value,
        "naive_rmse": 1.0,
        "rmse_ratio": pred_value,
    }
    if mode == "rollout":
        row.update(
            {
                "anchor_rule": "first_current_sign_change_minus_context",
                "anchor_frame": 1,
                "onset_frame": 5,
            }
        )
    result_name = (
        "next_frame_results.json" if mode == "next_frame" else "rollout_results.json"
    )
    (directory / result_name).write_text(json.dumps([row]), encoding="utf-8")


def write_complete_fixture(root: Path) -> None:
    for model in MODELS:
        for particle in range(1, 5):
            tag = f"identity_holdout_{model}_p{particle}"
            run_dir = root / tag
            train_stems = [
                f"GRA29_C20_25deg_particle{i}"
                for i in range(1, 5)
                if i != particle
            ]
            test_stems = [
                f"GRA29_C20_25deg_particle{particle}",
                f"GRA29_C20_45deg_particle{particle}",
            ]
            model_dir = run_dir / "models"
            model_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_family": model,
                    "split": "identity_holdout",
                    "heldout_particle": particle,
                    "train_stems": train_stems,
                    "test_stems": test_stems,
                    "seed": 1337,
                    "active_fields": ["intensity"],
                    "prediction_form": "delta_from_last_frame",
                    "rollout_anchor_rule": (
                        "first_current_sign_change_minus_context"
                    ),
                },
                model_dir / f"{model}_best.pt",
            )
            for temperature, group in GROUPS.items():
                stem = f"GRA29_C20_{temperature}_particle{particle}"
                _write_mode_artifact(
                    run_dir, "next_frame", stem, model, particle, group
                )
                _write_mode_artifact(
                    run_dir, "rollout", stem, model, particle, group
                )


class IdentityHoldoutAggregationTest(unittest.TestCase):
    def test_complete_grid_aggregates_over_four_physical_particles(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_complete_fixture(root)

            summary = aggregate_identity_holdout(root)

            self.assertEqual(summary["aggregation_unit"], "physical_particle")
            self.assertTrue(summary["shared_cell_level_protocol"])
            self.assertEqual(summary["particle_count"], 4)
            self.assertEqual(
                summary["models"]["unet"]["same_temperature_unseen_particle"],
                {
                    "next_frame_mean_mae_ratio": 0.5,
                    "next_frame_particles_better_than_persistence": 4,
                    "rollout_mean_mae_ratio": 2.0,
                    "rollout_particles_better_than_persistence": 0,
                },
            )
            self.assertEqual(
                summary["models"]["predrnnpp"]
                ["cross_temperature_unseen_particle"],
                {
                    "next_frame_mean_mae_ratio": 1.5,
                    "next_frame_particles_better_than_persistence": 0,
                    "rollout_mean_mae_ratio": 3.0,
                    "rollout_particles_better_than_persistence": 0,
                },
            )

    def test_missing_fold_is_rejected_instead_of_partially_aggregated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_complete_fixture(root)
            missing = root / "identity_holdout_predrnnpp_p4"
            for path in sorted(missing.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                else:
                    path.rmdir()
            missing.rmdir()

            with self.assertRaisesRegex(
                IdentityHoldoutValidationError, "missing run.*predrnnpp_p4"
            ):
                aggregate_identity_holdout(root)

    def test_training_test_particle_overlap_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_complete_fixture(root)
            checkpoint = root / "identity_holdout_unet_p2/models/unet_best.pt"
            metadata = torch.load(checkpoint, weights_only=False)
            metadata["train_stems"].append("GRA29_C20_25deg_particle2")
            torch.save(metadata, checkpoint)

            with self.assertRaisesRegex(
                IdentityHoldoutValidationError, "held-out particle 2 appears in training"
            ):
                aggregate_identity_holdout(root)

    def test_nonfixed_rollout_persistence_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_complete_fixture(root)
            artifact = root / (
                "identity_holdout_unet_p1/rollout/"
                "GRA29_C20_25deg_particle1/"
                "rollout_GRA29_C20_25deg_particle1.npz"
            )
            with np.load(artifact) as loaded:
                arrays = {key: loaded[key] for key in loaded.files}
            arrays["naive"] = arrays["naive"].copy()
            arrays["naive"][1] = 2.0
            np.savez_compressed(artifact, **arrays)

            with self.assertRaisesRegex(
                IdentityHoldoutValidationError, "rollout persistence is not fixed"
            ):
                aggregate_identity_holdout(root)

    def test_unanchored_rollout_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_complete_fixture(root)
            result = root / (
                "identity_holdout_unet_p1/rollout/"
                "GRA29_C20_25deg_particle1/rollout_results.json"
            )
            rows = json.loads(result.read_text(encoding="utf-8"))
            rows[0].pop("anchor_rule")
            result.write_text(json.dumps(rows), encoding="utf-8")

            with self.assertRaisesRegex(
                IdentityHoldoutValidationError, "rollout anchor"
            ):
                aggregate_identity_holdout(root)

    def test_missing_evaluation_group_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_complete_fixture(root)
            result = root / (
                "identity_holdout_unet_p3/next_frame/"
                "GRA29_C20_45deg_particle3/next_frame_results.json"
            )
            rows = json.loads(result.read_text(encoding="utf-8"))
            rows[0].pop("evaluation_group")
            result.write_text(json.dumps(rows), encoding="utf-8")

            with self.assertRaisesRegex(
                IdentityHoldoutValidationError, "evaluation_group"
            ):
                aggregate_identity_holdout(root)

    def test_saved_mae_ratio_must_match_prediction_arrays(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_complete_fixture(root)
            result = root / (
                "identity_holdout_predrnnpp_p1/next_frame/"
                "GRA29_C20_25deg_particle1/next_frame_results.json"
            )
            rows = json.loads(result.read_text(encoding="utf-8"))
            rows[0]["mae_ratio"] = 0.25
            result.write_text(json.dumps(rows), encoding="utf-8")

            with self.assertRaisesRegex(
                IdentityHoldoutValidationError, "saved MAE ratio"
            ):
                aggregate_identity_holdout(root)

    def test_latex_table_defines_both_test_conditions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_complete_fixture(root)
            summary = aggregate_identity_holdout(root)

            latex = render_latex_table(summary)

            self.assertIn("Same temperature", latex)
            self.assertIn("Cross temperature", latex)
            self.assertIn("next frame", latex)
            self.assertIn("512-step rollout", latex)
            self.assertIn("0.500 (4/4)", latex)
            self.assertIn("2.000 (0/4)", latex)
            self.assertNotIn("confidence interval", latex.lower())


if __name__ == "__main__":
    unittest.main()
