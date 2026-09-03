import tempfile
import unittest
from pathlib import Path

import numpy as np

from train.matched_grid_artifacts import (
    representative_offsets,
    write_compact_next_frame_artifact,
)


class MatchedGridArtifactsTest(unittest.TestCase):
    def test_representative_offsets_are_deterministic_and_include_endpoints(self):
        self.assertEqual([0, 2, 4], representative_offsets(5, count=3).tolist())
        self.assertEqual([0, 1], representative_offsets(2, count=16).tolist())

    def test_compact_artifact_keeps_full_errors_and_sampled_images(self):
        targets = np.zeros((5, 2, 2), dtype=np.float32)
        pred = np.stack(
            [np.full((2, 2), value, dtype=np.float32) for value in range(5)]
        )
        naive = np.ones_like(pred)
        target_steps = np.arange(10, 15, dtype=np.int64)
        frame_times = np.arange(5, dtype=np.float32) * 2.5

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "artifact.npz"
            summary = write_compact_next_frame_artifact(
                path,
                pred=pred,
                naive=naive,
                targets=targets,
                target_steps=target_steps,
                frame_times=frame_times,
                metadata={"model_family": "literal_model", "tag": "literal_tag"},
                sample_count=3,
            )

            self.assertEqual("compact_samples_v1", summary["artifact_mode"])
            self.assertEqual(3, summary["sample_count"])
            with np.load(path) as data:
                self.assertEqual("compact_samples_v1", data["artifact_mode"].item())
                self.assertEqual([0, 2, 4], data["sample_offsets"].tolist())
                self.assertTrue(
                    np.array_equal(data["pred_samples"], pred[[0, 2, 4]])
                )
                self.assertTrue(
                    np.array_equal(data["target_steps"], target_steps)
                )
                self.assertTrue(
                    np.allclose(
                        data["per_step_model_mae"],
                        np.asarray([0, 1, 2, 3, 4], dtype=np.float32),
                    )
                )
                self.assertTrue(
                    np.allclose(
                        data["per_step_model_mse"],
                        np.asarray([0, 1, 4, 9, 16], dtype=np.float32),
                    )
                )
                self.assertTrue(
                    np.allclose(data["per_step_naive_mae"], np.ones(5))
                )
                self.assertEqual("literal_model", data["model_family"].item())
            self.assertEqual([], list(Path(tmp).glob("*.partial")))

    def test_misaligned_arrays_fail_without_creating_an_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "artifact.npz"
            with self.assertRaisesRegex(ValueError, "aligned first dimensions"):
                write_compact_next_frame_artifact(
                    path,
                    pred=np.zeros((4, 2, 2), dtype=np.float32),
                    naive=np.zeros((5, 2, 2), dtype=np.float32),
                    targets=np.zeros((5, 2, 2), dtype=np.float32),
                    target_steps=np.arange(5),
                    frame_times=np.arange(5),
                    metadata={},
                )
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
