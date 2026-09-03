import unittest

import numpy as np

from analysis.rollout_scale_diagnostics import evaluate_rollout_scale, latex_table


class RolloutScaleDiagnosticsTest(unittest.TestCase):
    def test_reports_masked_endpoint_and_cumulative_metrics(self):
        targets = np.stack(
            [np.full((2, 2), value, dtype=np.float32) for value in (1, 2, 4)]
        )
        naive = np.zeros_like(targets)
        pred = targets * 0.5
        mask = np.array([[True, False], [False, True]])

        result = evaluate_rollout_scale(
            targets, naive, pred, mask, horizons=(1, 3)
        )

        self.assertEqual(result["mask_fraction"], 0.5)
        first = result["horizons"]["1"]
        self.assertEqual(first["truth_displacement_full"], 1.0)
        self.assertEqual(first["truth_displacement_masked"], 1.0)
        self.assertEqual(first["endpoint_mae_ratio_full"], 0.5)
        self.assertEqual(first["endpoint_mae_ratio_masked"], 0.5)
        third = result["horizons"]["3"]
        self.assertEqual(third["truth_displacement_full"], 4.0)
        self.assertAlmostEqual(
            third["cumulative_persistence_mae_full"], 7 / 3, places=6
        )
        self.assertAlmostEqual(
            third["cumulative_model_mae_masked"], 7 / 6, places=6
        )
        self.assertEqual(third["cumulative_mae_ratio_masked"], 0.5)

    def test_rejects_out_of_range_one_based_horizon(self):
        values = np.zeros((2, 2, 2), dtype=np.float32)
        mask = np.ones((2, 2), dtype=bool)

        with self.assertRaisesRegex(ValueError, "one-based horizon"):
            evaluate_rollout_scale(values, values, values, mask, horizons=(0,))

    def test_latex_table_labels_masked_and_full_frame_quantities(self):
        payload = {
            "horizons": [1],
            "particles": [
                {
                    "particle": 1,
                    "mask_fraction": 0.25,
                    "models": {
                        "unet": {
                            "horizons": {
                                "1": {
                                    "truth_displacement_full": 0.1,
                                    "truth_displacement_masked": 0.2,
                                    "cumulative_mae_ratio_full": 1.5,
                                    "cumulative_mae_ratio_masked": 1.7,
                                }
                            }
                        },
                        "rft": {
                            "horizons": {
                                "1": {
                                    "cumulative_mae_ratio_full": 1.1,
                                    "cumulative_mae_ratio_masked": 1.2,
                                }
                            }
                        },
                    },
                }
            ],
        }

        latex = latex_table(payload)

        self.assertIn("Mask area", latex)
        self.assertIn("Truth displacement", latex)
        self.assertIn("full & mask", latex)
        self.assertIn("p1", latex)
        self.assertIn(r"\begin{tabular}{@{}lrrrrrrr@{}}", latex)


if __name__ == "__main__":
    unittest.main()
