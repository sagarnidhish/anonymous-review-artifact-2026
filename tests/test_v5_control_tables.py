import unittest

from analysis.make_v5_control_tables import (
    render_normalization_table,
    render_stabilization_table,
    render_walrus_probe_table,
)


class V5ControlTablesTest(unittest.TestCase):
    def test_walrus_probe_is_explicitly_one_step(self):
        probe = {
            "status": "passed",
            "validation_improved": True,
            "baseline_validation": {"mean_model_mae": 0.015435},
            "fitted_validation": {"mean_model_mae": 0.014850},
            "baseline_test": {
                "mean_model_mae": 0.017871,
                "mean_particle_mae_ratio": 1.2145,
            },
            "fitted_test": {
                "mean_model_mae": 0.018004,
                "mean_particle_mae_ratio": 1.2196,
            },
        }

        latex = render_walrus_probe_table(probe)

        self.assertIn("One-step projection", latex)
        self.assertIn("Unfitted graft", latex)
        self.assertIn("Fitted on 25", latex)
        self.assertIn("1.214", latex)
        self.assertIn("1.220", latex)
        self.assertNotIn("rollout", latex.lower())

    def test_normalization_table_names_information_boundary(self):
        artifact = {
            "payload_results": [
                {
                    "payload": {"model_family": family},
                    "modes": {
                        "archived_test_record": {
                            "next_frame": {"mean_particle_mae_ratio": 0.8},
                            "rollout_anchored": {
                                "mean_particle_mae_ratio": {"512": 2.0}
                            },
                        },
                        "paired_training_reference": {
                            "next_frame": {"mean_particle_mae_ratio": 0.9},
                            "rollout_anchored": {
                                "mean_particle_mae_ratio": {"512": 2.5}
                            },
                        },
                    },
                }
                for family in ("unet", "predrnnpp")
            ]
        }

        latex = render_normalization_table(artifact)

        self.assertIn("U-Net", latex)
        self.assertIn("PredRNN++", latex)
        self.assertIn("Full test record", latex)
        self.assertIn(r"Paired 25\,$^\circ$C training record", latex)
        self.assertIn("One-frame ratio", latex)
        self.assertIn(r"$+512$ ratio", latex)

    def test_stabilization_table_keeps_all_declared_strategies(self):
        rows = [
            {
                "label": label,
                "next_frame_mean_mae_ratio": 0.8,
                "rollout_mean_mae_ratio_h32": 1.0,
                "rollout_mean_mae_ratio_h128": 1.2,
                "rollout_mean_mae_ratio_h256": 1.5,
                "rollout_mean_mae_ratio_h512": 2.0,
                "rollout_diverged_particles": 0,
            }
            for label in (
                "Fresh one-step U-Net",
                "Input noise",
                "Scheduled sampling (32 steps)",
                "Recursive loss (32 steps)",
            )
        ]
        rows[-1]["rollout_mean_mae_ratio_h512"] = None

        latex = render_stabilization_table(rows)

        for row in rows:
            self.assertIn(row["label"], latex)
        self.assertIn("One frame", latex)
        self.assertIn(r"$+512$", latex)
        self.assertIn(r"$\infty$", latex)


if __name__ == "__main__":
    unittest.main()
