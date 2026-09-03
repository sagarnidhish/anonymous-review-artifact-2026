import unittest

import numpy as np

from analysis.physics_metrics import (
    align_artifact_steps,
    calibration_context,
    observable_trajectories,
)


class PhysicsMetricsSharedThresholdTest(unittest.TestCase):
    def test_calibration_context_uses_frames_before_first_target(self):
        frames = np.arange(8 * 2 * 2).reshape(8, 2, 2)

        context, start, stop = calibration_context(
            frames, np.array([5, 6, 7]), context_len=4
        )

        np.testing.assert_array_equal(context, frames[1:5])
        self.assertEqual((start, stop), (1, 5))

    def test_observable_uses_caller_supplied_bright_threshold(self):
        frames = np.array([
            [[0.0, 2.0], [4.0, 6.0]],
            [[10.0, 10.0], [10.0, 10.0]],
        ])
        mask = np.ones((2, 2), dtype=bool)

        observable = observable_trajectories(
            frames, mask, bright_threshold=5.0
        )

        np.testing.assert_array_equal(
            observable["bright_frac90"], [0.25, 1.0]
        )

    def test_artifact_steps_are_aligned_from_frame_times_when_local(self):
        sequence_times = np.arange(10, dtype=np.float64) * 10.0

        steps, source = align_artifact_steps(
            sequence_times,
            artifact_steps=np.array([4, 5, 6]),
            artifact_frame_times=np.array([50.0, 60.0, 70.0]),
        )

        np.testing.assert_array_equal(steps, [5, 6, 7])
        self.assertEqual(source, "frame_times")

    def test_artifact_step_alignment_fails_if_times_do_not_match(self):
        with self.assertRaisesRegex(ValueError, "cannot align artifact"):
            align_artifact_steps(
                np.arange(6, dtype=np.float64) * 10.0,
                artifact_steps=np.array([1, 2]),
                artifact_frame_times=np.array([15.0, 25.0]),
            )


if __name__ == "__main__":
    unittest.main()
