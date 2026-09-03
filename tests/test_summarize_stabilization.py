import unittest

from analysis.summarize_stabilization import assemble_stabilization_rows


def metric_row(payload_id, **extra):
    row = {
        "payload_id": payload_id,
        "next_frame_mean_mae_ratio": 0.8 + payload_id / 100,
        "rollout_mean_mae_ratio_h32": 1.0 + payload_id / 10,
        "rollout_mean_mae_ratio_h128": 1.2 + payload_id / 10,
        "rollout_mean_mae_ratio_h256": 1.5 + payload_id / 10,
        "rollout_mean_mae_ratio_h512": 2.0 + payload_id / 10,
        "rollout_diverged_particles": 0,
    }
    row.update(extra)
    return row


class SummarizeStabilizationTest(unittest.TestCase):
    def test_assembles_baseline_and_all_three_strategies_in_declared_order(self):
        grid = {
            "status": "complete",
            "payloads": [metric_row(0)] + [metric_row(i) for i in range(1, 24)],
        }
        strategies = [
            metric_row(0, strategy="input_noise"),
            metric_row(1, strategy="scheduled_sampling"),
            metric_row(2, strategy="recursive_unroll"),
        ]

        rows = assemble_stabilization_rows(grid, strategies)

        self.assertEqual(
            [row["label"] for row in rows],
            [
                "Fresh one-step U-Net",
                "Input noise",
                "Scheduled sampling (32 steps)",
                "Recursive loss (32 steps)",
            ],
        )
        self.assertEqual(rows[0]["source"], "fresh_matched_grid_payload_0")
        self.assertEqual(rows[3]["strategy"], "recursive_unroll")

    def test_rejects_missing_strategy(self):
        grid = {"status": "complete", "payloads": [metric_row(i) for i in range(24)]}
        with self.assertRaisesRegex(ValueError, "three stabilization"):
            assemble_stabilization_rows(grid, [metric_row(0)])


if __name__ == "__main__":
    unittest.main()
