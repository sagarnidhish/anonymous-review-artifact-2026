import unittest

import numpy as np

from analysis.make_movie_montage import display_limits, select_frame_indices


class MovieMontageTest(unittest.TestCase):
    def test_indices_span_complete_movie_without_duplicates(self):
        indices = select_frame_indices(n_frames=5333, count=24)

        self.assertEqual(len(indices), 24)
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 5332)
        self.assertTrue(np.all(np.diff(indices) > 0))

    def test_invalid_frame_count_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "count"):
            select_frame_indices(n_frames=8, count=9)

    def test_display_limits_are_shared_robust_percentiles(self):
        frames = np.arange(3 * 2 * 2, dtype=np.float32).reshape(3, 2, 2)

        vmin, vmax = display_limits(frames, lower=10, upper=90)

        self.assertAlmostEqual(vmin, float(np.percentile(frames, 10)))
        self.assertAlmostEqual(vmax, float(np.percentile(frames, 90)))
        self.assertLess(vmin, vmax)


if __name__ == "__main__":
    unittest.main()
