import unittest

from analysis.summarize_matched_grid import strict_json_text, summarize_payload_rows
from train.matched_grid_payloads import payload_for_id


class SummarizeMatchedGridTest(unittest.TestCase):
    def test_particle_means_counts_and_divergence_are_retained(self):
        payload = payload_for_id(0)
        next_ratios = (0.8, 1.0, 1.2, 0.6)
        rollout_ratios = (1.2, 1.3, float("inf"), 0.9)
        next_rows = []
        rollout_rows = []
        for index, (next_ratio, rollout_ratio) in enumerate(
            zip(next_ratios, rollout_ratios), start=1
        ):
            stem = f"GRA29_C20_45deg_particle{index}"
            next_rows.append(
                {
                    "stem": stem,
                    "payload_id": payload["payload_id"],
                    "mae_ratio": next_ratio,
                }
            )
            rollout_rows.append(
                {
                    "stem": stem,
                    "payload_id": payload["payload_id"],
                    "status": "numerically_diverged" if index == 3 else "complete",
                    "horizons": {
                        str(horizon): {"mae_ratio": rollout_ratio}
                        for horizon in payload["report_horizons"]
                    },
                }
            )

        summary = summarize_payload_rows(payload, next_rows, rollout_rows)

        self.assertAlmostEqual(0.9, summary["next_frame_mean_mae_ratio"])
        self.assertEqual(2, summary["next_frame_particles_better"])
        self.assertEqual(1, summary["rollout_diverged_particles"])
        self.assertEqual(3, summary["rollout_complete_particles"])
        self.assertEqual(1, summary["rollout_particles_better_h512"])
        self.assertEqual(float("inf"), summary["rollout_mean_mae_ratio_h512"])

    def test_missing_horizon_fails_loudly(self):
        payload = payload_for_id(0)
        next_rows = [
            {
                "stem": f"GRA29_C20_45deg_particle{i}",
                "payload_id": 0,
                "mae_ratio": 0.9,
            }
            for i in range(1, 5)
        ]
        rollout_rows = [
            {
                "stem": row["stem"],
                "payload_id": 0,
                "status": "complete",
                "horizons": {"32": {"mae_ratio": 1.1}},
            }
            for row in next_rows
        ]
        with self.assertRaisesRegex(ValueError, "missing rollout horizon"):
            summarize_payload_rows(payload, next_rows, rollout_rows)

    def test_json_null_divergence_is_promoted_to_infinite_error(self):
        payload = payload_for_id(0)
        next_rows = []
        rollout_rows = []
        for index in range(1, 5):
            stem = f"GRA29_C20_45deg_particle{index}"
            next_rows.append(
                {"stem": stem, "payload_id": 0, "mae_ratio": 0.9}
            )
            rollout_rows.append(
                {
                    "stem": stem,
                    "payload_id": 0,
                    "status": "numerically_diverged" if index == 1 else "complete",
                    "horizons": {
                        str(horizon): {
                            "mae_ratio": None if index == 1 else 1.2
                        }
                        for horizon in payload["report_horizons"]
                    },
                }
            )

        summary = summarize_payload_rows(payload, next_rows, rollout_rows)

        self.assertEqual(float("inf"), summary["rollout_mean_mae_ratio_h512"])
        self.assertEqual(0, summary["rollout_particles_better_h512"])

    def test_summary_json_serializes_divergence_as_standard_null(self):
        text = strict_json_text({"status": "complete", "ratio": float("inf")})

        self.assertIn('"ratio": null', text)
        self.assertNotIn("Infinity", text)


if __name__ == "__main__":
    unittest.main()
