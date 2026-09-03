import unittest
from pathlib import Path

import numpy as np

from train.walrus_rollout_state import initialize_rollout_state


class WalrusRolloutStateTest(unittest.TestCase):
    def test_persistence_frame_remains_last_observation(self):
        frames = np.arange(6 * 2 * 2, dtype=np.float32).reshape(6, 2, 2)

        context, persistence_frame = initialize_rollout_state(
            frames, start=1, context_len=3
        )
        context[-1][:] = 99.0

        np.testing.assert_array_equal(persistence_frame, frames[3])
        self.assertEqual(len(context), 3)

    def test_invalid_context_bounds_fail(self):
        frames = np.zeros((3, 2, 2), dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "context exceeds"):
            initialize_rollout_state(frames, start=1, context_len=3)

    def test_walrus_evaluators_use_fixed_persistence_frame(self):
        repo = Path(__file__).resolve().parents[1]
        for relative in (
            "train/walrus_eval_plain.py",
            "train/modal_walrus_eval.py",
        ):
            source = (repo / relative).read_text()
            self.assertIn("initialize_rollout_state", source)
            self.assertIn("naives.append(persistence_frame.copy())", source)
            self.assertNotIn("naives.append(ctx[-1].copy())", source)


if __name__ == "__main__":
    unittest.main()
