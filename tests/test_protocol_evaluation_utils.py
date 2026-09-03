import tempfile
import unittest
from pathlib import Path

import numpy as np

from train.protocol_evaluation_utils import (
    aggregate_rows,
    anchored_slice,
    delay_condition_specs,
    first_current_transition_slice,
    fixed_transition_summaries,
    load_sequence,
    response_l1,
    with_intervention,
)


class ProtocolEvaluationUtilsTest(unittest.TestCase):
    def _write_sequence(self, root: Path) -> Path:
        path = root / "particle.npz"
        np.savez_compressed(
            path,
            intensity=np.arange(24, dtype=np.float32).reshape(6, 2, 2),
            voltage=np.arange(6, dtype=np.float32),
            current=-np.arange(6, dtype=np.float32),
            time_norm=np.linspace(0, 1, 6, dtype=np.float32),
            frame_times=np.arange(6, dtype=np.float32) * 30,
        )
        return path

    def test_first_current_transition_slice_ends_context_at_transition(self):
        with tempfile.TemporaryDirectory() as tmp:
            sequence = load_sequence(self._write_sequence(Path(tmp)))
        sequence.exogenous_all["current"] = np.array(
            [-1, -1, -1, -1, -1, 1], dtype=np.float32
        )

        anchor_start, onset, sliced = first_current_transition_slice(
            sequence, context_len=4
        )

        self.assertEqual(onset, 5)
        self.assertEqual(anchor_start, 1)
        np.testing.assert_array_equal(sliced.raw_frame_indices, [1, 2, 3, 4, 5])
        np.testing.assert_array_equal(sliced.intensity[0], sequence.intensity[1])

    def test_first_current_transition_slice_rejects_sequence_without_reversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            sequence = load_sequence(self._write_sequence(Path(tmp)))
        sequence.exogenous_all["current"] = np.ones(6, dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "current-sign transition"):
            first_current_transition_slice(sequence, context_len=4)

    def test_load_and_anchor_preserve_aligned_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            sequence = load_sequence(self._write_sequence(Path(tmp)))

        sliced = anchored_slice(sequence, 2)

        np.testing.assert_array_equal(sliced.intensity, sequence.intensity[2:])
        np.testing.assert_array_equal(
            sliced.exogenous_all["current"],
            sequence.exogenous_all["current"][2:],
        )
        np.testing.assert_array_equal(sliced.frame_times, sequence.frame_times[2:])

    def test_intervention_does_not_change_images_or_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            sequence = load_sequence(self._write_sequence(Path(tmp)))

        modified = with_intervention(sequence, "zero", seed=1, shift=2)

        np.testing.assert_array_equal(modified.intensity, sequence.intensity)
        np.testing.assert_array_equal(modified.frame_times, sequence.frame_times)
        np.testing.assert_array_equal(modified.exogenous_all["current"], 0)

    def test_response_l1_and_particle_aggregation(self):
        change = response_l1(np.array([0.0, 2.0]), np.array([0.0, 0.0]))
        rows = [
            {
                "mode": "next_frame",
                "condition": "true",
                "model_mae": 1.0,
                "naive_mae": 2.0,
                "mae_ratio": 0.5,
                "prediction_change_l1": 0.0,
            },
            {
                "mode": "next_frame",
                "condition": "true",
                "model_mae": 3.0,
                "naive_mae": 2.0,
                "mae_ratio": 1.5,
                "prediction_change_l1": 0.0,
            },
        ]

        summary = aggregate_rows(rows)

        self.assertEqual(change, 1.0)
        self.assertEqual(summary[0]["n_particles"], 2)
        self.assertEqual(summary[0]["mean_mae_ratio"], 1.0)

    def test_delay_condition_specs_are_complete_and_unique(self):
        specs = delay_condition_specs((0, 16, 128))

        self.assertEqual(
            [spec["label"] for spec in specs],
            ["measured", "zero", "delay_16", "delay_128", "shuffle"],
        )
        self.assertEqual(specs[2]["shift_frames"], 16)
        self.assertEqual(specs[-1]["intervention"], "shuffle")

    def test_fixed_transition_summary_uses_measured_current_indices(self):
        targets = np.zeros((5, 1, 1), dtype=np.float32)
        naive = np.ones_like(targets)
        pred = np.arange(5, dtype=np.float32).reshape(5, 1, 1)
        measured_current = np.array([-1, -1, 1, 1, 1], dtype=np.float32)
        target_steps = np.arange(5, dtype=np.int64)

        result = fixed_transition_summaries(
            pred,
            naive,
            targets,
            measured_current=measured_current,
            target_steps=target_steps,
            radius=0,
        )

        self.assertEqual(result["transition"]["N"], 1)
        self.assertEqual(result["transition"]["model_mae"], 2.0)
        self.assertEqual(result["nontransition"]["N"], 4)


if __name__ == "__main__":
    unittest.main()
