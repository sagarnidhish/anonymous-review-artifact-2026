import unittest

import numpy as np

from analysis.evaluation_invariants import (
    bright_fraction_trajectory,
    calibrate_bright_threshold,
    fixed_persistence,
)


class EvaluationInvariantsTest(unittest.TestCase):
    def test_fixed_persistence_repeats_last_observed_frame(self):
        last_observed = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

        baseline = fixed_persistence(last_observed, horizon=3)

        self.assertEqual(baseline.shape, (3, 2, 2))
        np.testing.assert_array_equal(baseline[0], last_observed)
        np.testing.assert_array_equal(baseline[2], last_observed)

    def test_fixed_persistence_does_not_share_memory_with_context(self):
        last_observed = np.ones((2, 2), dtype=np.float32)

        baseline = fixed_persistence(last_observed, horizon=2)
        baseline[0, 0, 0] = 99.0

        self.assertEqual(last_observed[0, 0], 1.0)
        self.assertEqual(baseline[1, 0, 0], 1.0)

    def test_one_context_threshold_is_reused_for_all_trajectories(self):
        mask = np.ones((2, 2), dtype=bool)
        context = np.array([
            [[0.0, 1.0], [2.0, 3.0]],
            [[0.0, 1.0], [2.0, 3.0]],
        ])
        threshold = calibrate_bright_threshold(
            context, mask, percentile=50.0
        )
        target = np.full((2, 2, 2), 3.0)
        attenuated_prediction = np.full((2, 2, 2), 1.0)
        persistence = np.full((2, 2, 2), 2.0)

        target_fraction = bright_fraction_trajectory(target, mask, threshold)
        prediction_fraction = bright_fraction_trajectory(
            attenuated_prediction, mask, threshold
        )
        persistence_fraction = bright_fraction_trajectory(
            persistence, mask, threshold
        )

        np.testing.assert_array_equal(target_fraction, [1.0, 1.0])
        np.testing.assert_array_equal(prediction_fraction, [0.0, 0.0])
        np.testing.assert_array_equal(persistence_fraction, [1.0, 1.0])

    def test_calibration_rejects_empty_mask(self):
        context = np.zeros((2, 2, 2), dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "mask contains no pixels"):
            calibrate_bright_threshold(
                context, np.zeros((2, 2), dtype=bool)
            )


if __name__ == "__main__":
    unittest.main()
