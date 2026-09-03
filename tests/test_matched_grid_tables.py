import unittest

from analysis.make_matched_grid_tables import (
    render_family_table,
    render_full_grid_table,
)
from train.matched_grid_payloads import build_payloads


def complete_summary():
    rows = []
    for payload in build_payloads():
        row = dict(payload)
        row.update(
            {
                "next_frame_mean_mae_ratio": 0.8 + 0.03 * payload["payload_id"],
                "next_frame_particles_better": payload["payload_id"] % 5,
                "rollout_diverged_particles": int(payload["payload_id"] == 23),
            }
        )
        for horizon in payload["report_horizons"]:
            row[f"rollout_mean_mae_ratio_h{horizon}"] = (
                None
                if payload["payload_id"] == 23 and horizon == 512
                else 1.0 + payload["payload_id"] / 10 + horizon / 512
            )
            row[f"rollout_particles_better_h{horizon}"] = 0
        rows.append(row)
    return {"status": "complete", "payloads": rows}


class MatchedGridTablesTest(unittest.TestCase):
    def test_family_table_names_all_forms_and_complete_rollout_scope(self):
        latex = render_family_table(complete_summary())

        self.assertIn("One-frame MAE ratio", latex)
        self.assertIn(r"image $\Delta$ & protocol $\Delta$", latex)
        self.assertIn("image direct & protocol direct", latex)
        self.assertIn("Best 256-step ratio", latex)
        self.assertIn("best of four forms", latex)
        self.assertNotIn("available", latex.lower())
        for label in (
            "U-Net",
            "ConvLSTM",
            "SimVP",
            "Residual CNN",
            "PredRNN",
            "PredRNN++",
        ):
            self.assertIn(label, latex)

    def test_full_table_contains_every_payload_and_horizon(self):
        latex = render_full_grid_table(complete_summary())

        self.assertEqual(latex.count(r" \\"), 26)
        self.assertIn("One frame", latex)
        self.assertIn(r"$+32$", latex)
        self.assertIn(r"$+512$", latex)
        self.assertIn(r"protocol & direct $\dagger$", latex)

    def test_incomplete_summary_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "24 payload"):
            render_family_table({"status": "complete", "payloads": []})


if __name__ == "__main__":
    unittest.main()
