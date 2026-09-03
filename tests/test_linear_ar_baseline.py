import unittest

import numpy as np

from analysis.linear_ar_baseline import (
    accumulate_sufficient_statistics,
    evaluate_next_frame,
    fit_ridge_ar,
    fit_ridge_from_statistics,
    latex_table,
    predict_next,
    recursive_rollout,
)


class LinearArBaselineTest(unittest.TestCase):
    def test_recovers_shared_four_tap_coefficients(self):
        rng = np.random.default_rng(7)
        contexts = rng.normal(size=(20, 4, 3, 2)).astype(np.float64)
        expected = np.array([0.1, -0.2, 0.3, 0.7, 0.05])
        targets = (
            np.einsum("nkhw,k->nhw", contexts, expected[:4])
            + expected[4]
        )

        fitted = fit_ridge_ar(contexts, targets, ridge=0.0)

        np.testing.assert_allclose(fitted, expected, atol=1e-10, rtol=0)

    def test_predict_next_applies_oldest_to_newest_lag_order(self):
        context = np.stack(
            [np.full((2, 2), value, dtype=np.float64) for value in (1, 2, 3, 4)]
        )
        coefficients = np.array([0.0, 0.0, 0.25, 0.75, 1.0])

        prediction = predict_next(context, coefficients)

        np.testing.assert_array_equal(prediction, np.full((2, 2), 4.75))

    def test_recursive_rollout_consumes_its_own_predictions(self):
        context = np.stack(
            [np.full((1, 1), value, dtype=np.float64) for value in (0, 1, 2, 3)]
        )
        coefficients = np.array([0.0, 0.0, 0.0, 1.0, 1.0])

        rollout = recursive_rollout(context, coefficients, steps=3)

        np.testing.assert_array_equal(rollout[:, 0, 0], [4, 5, 6])

    def test_streaming_statistics_match_direct_fit(self):
        rng = np.random.default_rng(11)
        frames = rng.normal(size=(12, 2, 3)).astype(np.float64)
        starts = np.arange(6, dtype=np.int64)
        contexts = np.stack([frames[start : start + 4] for start in starts])
        targets = frames[starts + 4]

        direct = fit_ridge_ar(contexts, targets, ridge=0.01)
        statistics = accumulate_sufficient_statistics(
            frames, starts, chunk_size=2
        )
        streaming = fit_ridge_from_statistics(statistics, ridge=0.01)

        np.testing.assert_allclose(streaming, direct, atol=1e-12, rtol=0)

    def test_next_frame_metrics_use_fixed_last_frame_persistence(self):
        frames = np.arange(8, dtype=np.float64)[:, None, None]
        coefficients = np.array([0.0, 0.0, 0.0, 1.0, 1.0])

        result = evaluate_next_frame(
            frames, starts=np.array([0, 1, 2]), coefficients=coefficients
        )

        self.assertEqual(result["model_mae"], 0.0)
        self.assertEqual(result["persistence_mae"], 1.0)
        self.assertEqual(result["mae_ratio"], 0.0)

    def test_latex_table_defines_next_frame_and_masked_rollout(self):
        payload = {
            "selected_ridge": 0.01,
            "coefficients_oldest_to_newest_then_intercept": [0, 0, 0, 1, 0],
            "test_particles": [
                {
                    "particle": 1,
                    "next_frame": {"mae_ratio": 0.9},
                    "anchored_rollout": {
                        "horizons": {
                            "512": {
                                "cumulative_mae_ratio_full": 1.2,
                                "cumulative_mae_ratio_masked": 1.3,
                            }
                        }
                    },
                }
            ],
        }

        latex = latex_table(payload)

        self.assertIn("Next-frame MAE ratio", latex)
        self.assertIn("512-step cumulative MAE ratio", latex)
        self.assertIn("masked", latex)
        self.assertIn("0.01", latex)


if __name__ == "__main__":
    unittest.main()
