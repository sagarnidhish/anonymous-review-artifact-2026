import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from analysis.validate_matched_grid import (
    validate_grid,
    validate_payload_directory,
)
from train.matched_grid_payloads import payload_for_id


STEMS = [f"GRA29_C20_45deg_particle{i}" for i in range(1, 5)]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def make_payload_tree(root: Path, payload_id: int = 0):
    payload = payload_for_id(payload_id)
    run = root / payload["tag"]
    checkpoint = run / "models" / f"{payload['model_family']}_best.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"checkpoint")
    next_rows, rollout_rows = [], []
    for stem in STEMS:
        next_rows.append(
            {
                "stem": stem,
                "payload_id": payload_id,
                "N": 3,
                "model_mae": 1.0,
                "naive_mae": 2.0,
                "mae_ratio": 0.5,
                "model_mse": 1.0,
                "naive_mse": 4.0,
                "mse_ratio": 0.25,
            }
        )
        rollout_rows.append(
            {
                "stem": stem,
                "payload_id": payload_id,
                "anchor_rule": payload["anchor_rule"],
                "anchor_frame": 96,
                "onset_frame": 100,
                "rollout_steps": 512,
                "status": "complete",
                "first_nonfinite_step": None,
                "horizons": {
                    str(h): {"model_mae": 1.0, "naive_mae": 1.0, "mae_ratio": 1.0}
                    for h in payload["report_horizons"]
                },
            }
        )
        npz = run / "rollout_anchored" / stem / f"rollout_{stem}.npz"
        npz.parent.mkdir(parents=True, exist_ok=True)
        pred = np.zeros((512, 2, 2), dtype=np.float32)
        naive = np.ones_like(pred)
        target = np.full_like(pred, 2)
        np.savez_compressed(
            npz,
            pred=pred,
            naive=naive,
            targets=target,
            target_steps=np.arange(100, 612, dtype=np.int64),
            frame_times=np.arange(512, dtype=np.float32),
            anchor_frame=np.asarray(96),
            onset_frame=np.asarray(100),
            first_nonfinite_step=np.asarray(-1),
        )
        next_npz = run / "next_frame" / stem / f"preds_{stem}.npz"
        next_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            next_npz,
            artifact_mode=np.asarray("compact_samples_v1"),
            sample_offsets=np.asarray([0, 2], dtype=np.int64),
            pred_samples=np.ones((2, 2, 2), dtype=np.float32),
            naive_samples=np.full((2, 2, 2), 2, dtype=np.float32),
            target_samples=np.zeros((2, 2, 2), dtype=np.float32),
            target_steps=np.asarray([10, 11, 12], dtype=np.int64),
            frame_times=np.asarray([0, 1, 2], dtype=np.float32),
            per_step_model_mae=np.ones(3, dtype=np.float32),
            per_step_naive_mae=np.full(3, 2, dtype=np.float32),
            per_step_model_mse=np.ones(3, dtype=np.float32),
            per_step_naive_mse=np.full(3, 4, dtype=np.float32),
        )
    write_json(run / "next_frame" / "next_frame_results.json", next_rows)
    write_json(run / "rollout_anchored" / "rollout_results.json", rollout_rows)
    write_json(
        run / "completion_manifest.json",
        {
            "status": "complete",
            "payload": payload,
            "checkpoint": str(checkpoint.relative_to(run)),
            "next_frame_particle_count": 4,
            "rollout_particle_count": 4,
            "next_frame_artifact_mode": "compact_samples_v1",
        },
    )
    return payload, run


class ValidateMatchedGridTest(unittest.TestCase):
    def test_complete_payload_with_fixed_persistence_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload, run = make_payload_tree(Path(tmp))
            summary = validate_payload_directory(run, payload)
            self.assertEqual("complete", summary["status"])
            self.assertEqual(4, summary["rollout_particle_count"])

    def test_updated_naive_frame_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload, run = make_payload_tree(Path(tmp))
            npz = run / "rollout_anchored" / STEMS[0] / f"rollout_{STEMS[0]}.npz"
            with np.load(npz) as data:
                arrays = {key: data[key] for key in data.files}
            arrays["naive"] = arrays["naive"].copy()
            arrays["naive"][1] += 1
            np.savez_compressed(npz, **arrays)
            with self.assertRaisesRegex(ValueError, "fixed persistence"):
                validate_payload_directory(run, payload)

    def test_tampered_compact_sample_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload, run = make_payload_tree(Path(tmp))
            npz = run / "next_frame" / STEMS[0] / f"preds_{STEMS[0]}.npz"
            with np.load(npz) as data:
                arrays = {key: data[key] for key in data.files}
            arrays["pred_samples"] = arrays["pred_samples"].copy()
            arrays["pred_samples"][0] = 3
            np.savez_compressed(npz, **arrays)
            with self.assertRaisesRegex(ValueError, "sampled model MAE"):
                validate_payload_directory(run, payload)

    def test_inconsistent_aggregate_next_frame_mae_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload, run = make_payload_tree(Path(tmp))
            path = run / "next_frame" / "next_frame_results.json"
            rows = json.loads(path.read_text(encoding="utf-8"))
            rows[0]["model_mae"] = 1.5
            write_json(path, rows)
            with self.assertRaisesRegex(ValueError, "aggregate model MAE"):
                validate_payload_directory(run, payload)

    def test_grid_rejects_missing_payloads(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_payload_tree(Path(tmp), payload_id=0)
            with self.assertRaisesRegex(ValueError, "missing payload"):
                validate_grid(Path(tmp))


if __name__ == "__main__":
    unittest.main()
