import unittest

from train.matched_grid_payloads import (
    FAMILIES,
    INPUT_MODES,
    TARGET_MODES,
    build_payloads,
    validate_payloads,
)


class MatchedGridPayloadsTest(unittest.TestCase):
    def test_grid_has_every_family_input_and_target_combination(self):
        payloads = build_payloads()

        self.assertEqual(24, len(payloads))
        observed = {
            (row["model_family"], row["input_mode"], row["target_mode"])
            for row in payloads
        }
        expected = {
            (family, input_mode, target_mode)
            for family in FAMILIES
            for input_mode in INPUT_MODES
            for target_mode in TARGET_MODES
        }
        self.assertEqual(expected, observed)
        self.assertEqual(list(range(24)), [row["payload_id"] for row in payloads])
        self.assertEqual(24, len({row["tag"] for row in payloads}))

    def test_payloads_fix_the_primary_protocol(self):
        for row in build_payloads():
            self.assertEqual(1337, row["seed"])
            self.assertEqual(4, row["context_len"])
            self.assertEqual(512, row["rollout_steps"])
            self.assertEqual([32, 128, 256, 512], row["report_horizons"])
            self.assertEqual(
                "first_current_sign_change_minus_context",
                row["anchor_rule"],
            )

    def test_validation_rejects_a_missing_or_duplicate_payload(self):
        payloads = build_payloads()
        with self.assertRaisesRegex(ValueError, "24 payloads"):
            validate_payloads(payloads[:-1])

        duplicate = [dict(row) for row in payloads]
        duplicate[-1] = dict(duplicate[0])
        with self.assertRaisesRegex(ValueError, "payload IDs"):
            validate_payloads(duplicate)


if __name__ == "__main__":
    unittest.main()
