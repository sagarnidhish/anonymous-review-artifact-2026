import unittest

from train.stabilization_payloads import build_payloads, teacher_forcing_probability


class StabilizationPayloadsTest(unittest.TestCase):
    def test_campaign_contains_three_declared_standard_fixes(self):
        payloads = build_payloads()

        self.assertEqual([0, 1, 2], [row["payload_id"] for row in payloads])
        self.assertEqual(
            ["input_noise", "scheduled_sampling", "recursive_unroll"],
            [row["strategy"] for row in payloads],
        )
        self.assertEqual(3, len({row["tag"] for row in payloads}))

    def test_all_runs_share_source_split_and_evaluation_protocol(self):
        for row in build_payloads():
            self.assertEqual(0, row["source_payload_id"])
            self.assertEqual("fresh_unet_image_delta_s1337", row["source_tag"])
            self.assertEqual(1337, row["seed"])
            self.assertEqual(4, row["context_len"])
            self.assertEqual(12, row["epochs"])
            self.assertEqual(512, row["rollout_steps"])
            self.assertEqual([32, 128, 256, 512], row["report_horizons"])
            self.assertEqual(
                "first_current_sign_change_minus_context", row["anchor_rule"]
            )
            self.assertEqual("heldout_free_rollout_mae_h32", row["selection_metric"])

    def test_scheduled_sampling_probability_reaches_declared_endpoints(self):
        row = build_payloads()[1]

        self.assertAlmostEqual(
            row["teacher_forcing_start"], teacher_forcing_probability(row, 1)
        )
        self.assertAlmostEqual(
            row["teacher_forcing_end"],
            teacher_forcing_probability(row, row["epochs"]),
        )


if __name__ == "__main__":
    unittest.main()
