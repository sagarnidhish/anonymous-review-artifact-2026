import tempfile
import unittest
from pathlib import Path

import numpy as np

from analysis.repair_walrus_rollout_baseline import repair_artifact, repair_tree


class RepairWalrusRolloutTest(unittest.TestCase):
    def test_repair_replaces_only_naive_with_fixed_persistence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.npz"
            destination = root / "corrected.npz"
            sequence = np.arange(16, dtype=np.float32).reshape(4, 2, 2)
            pred = np.stack([sequence[1] + 10, sequence[1] + 20])
            incorrect_naive = pred.copy()
            targets = np.stack([sequence[2], sequence[3]])
            steps = np.array([2, 3], dtype=np.int64)
            frame_times = np.array([20.0, 30.0], dtype=np.float32)
            np.savez_compressed(
                source,
                pred=pred,
                naive=incorrect_naive,
                targets=targets,
                target_steps=steps,
                frame_times=frame_times,
            )

            summary = repair_artifact(source, destination, sequence)

            with np.load(destination) as repaired:
                np.testing.assert_array_equal(repaired["pred"], pred)
                np.testing.assert_array_equal(repaired["targets"], targets)
                np.testing.assert_array_equal(repaired["target_steps"], steps)
                np.testing.assert_array_equal(
                    repaired["frame_times"], frame_times
                )
                np.testing.assert_array_equal(
                    repaired["naive"],
                    np.repeat(sequence[1][None, ...], 2, axis=0),
                )
            self.assertTrue(summary["naive_changed"])
            self.assertEqual(summary["persistence_source_step"], 1)

    def test_repair_refuses_to_overwrite_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.npz"
            np.savez_compressed(
                source,
                pred=np.zeros((1, 1, 1)),
                naive=np.zeros((1, 1, 1)),
                targets=np.zeros((1, 1, 1)),
                target_steps=np.array([1]),
                frame_times=np.array([1.0]),
            )

            with self.assertRaisesRegex(ValueError, "overwrite source"):
                repair_artifact(source, source, np.zeros((2, 1, 1)))

    def test_repair_tree_writes_particle_and_aggregate_summaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_root = root / "source"
            destination_root = root / "destination"
            arrays = root / "arrays"
            particle = source_root / "particle1"
            particle.mkdir(parents=True)
            arrays.mkdir()
            sequence = np.arange(16, dtype=np.float32).reshape(4, 2, 2)
            np.savez_compressed(arrays / "particle1.npz", intensity=sequence)
            np.savez_compressed(
                particle / "rollout_particle1.npz",
                pred=np.stack([sequence[1] + 1, sequence[1] + 2]),
                naive=np.stack([sequence[1] + 1, sequence[1] + 2]),
                targets=np.stack([sequence[2], sequence[3]]),
                target_steps=np.array([2, 3]),
                frame_times=np.array([2.0, 3.0]),
            )

            summary = repair_tree(source_root, destination_root, arrays)

            self.assertEqual(summary["count"], 1)
            self.assertTrue(
                (destination_root / "particle1" / "rollout_particle1.npz").is_file()
            )
            self.assertTrue((destination_root / "repair_summary.json").is_file())
            self.assertTrue(
                (destination_root / "comparison_summary.json").is_file()
            )


if __name__ == "__main__":
    unittest.main()
