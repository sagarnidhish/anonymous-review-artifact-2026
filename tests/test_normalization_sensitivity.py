import unittest

import numpy as np

from train.eval_normalization_sensitivity import (
    PRIMARY_PAYLOAD_IDS,
    reproduction_diagnostic,
    summarize_reproduction_diagnostics,
    paired_training_reference_transform,
)


class NormalizationSensitivityTest(unittest.TestCase):
    def test_transform_recovers_raw_then_applies_training_statistics(self):
        raw = np.asarray([[10.0, 14.0], [18.0, 22.0]], dtype=np.float32)
        encoded = (raw - 16.0) / 4.0
        transformed = paired_training_reference_transform(
            encoded,
            test_mu=16.0,
            test_std=4.0,
            train_mu=10.0,
            train_std=2.0,
        )
        np.testing.assert_allclose(transformed, (raw - 10.0) / 2.0)

    def test_equivalent_test_encodings_give_the_same_training_scale(self):
        raw = np.linspace(3, 9, 12, dtype=np.float32).reshape(3, 2, 2)
        first = paired_training_reference_transform(
            (raw - 5.0) / 2.0, 5.0, 2.0, 4.0, 3.0
        )
        second = paired_training_reference_transform(
            (raw + 7.0) / 5.0, -7.0, 5.0, 4.0, 3.0
        )
        np.testing.assert_allclose(first, second, rtol=1e-6, atol=1e-6)

    def test_control_targets_the_two_representative_image_delta_models(self):
        self.assertEqual((0, 20), PRIMARY_PAYLOAD_IDS)

    def test_catastrophic_rollout_reproduces_at_the_same_order_of_magnitude(self):
        diagnostic = reproduction_diagnostic(
            current=1.527216e14,
            original=1.585000e14,
        )
        self.assertTrue(diagnostic["reproduced"])
        self.assertFalse(diagnostic["strictly_close"])
        self.assertEqual("divergent_order", diagnostic["criterion"])

    def test_skill_scale_mismatch_is_not_hidden_by_divergence_handling(self):
        diagnostic = reproduction_diagnostic(current=1.02, original=1.00)
        self.assertFalse(diagnostic["reproduced"])
        self.assertEqual("mismatch", diagnostic["criterion"])

    def test_recursive_rollout_accepts_subpercent_cross_gpu_drift(self):
        diagnostic = reproduction_diagnostic(
            current=1.009,
            original=1.000,
            rtol=1e-2,
        )
        self.assertTrue(diagnostic["reproduced"])
        self.assertTrue(diagnostic["strictly_close"])

    def test_recursive_rollout_rejects_two_percent_drift(self):
        diagnostic = reproduction_diagnostic(
            current=1.02,
            original=1.00,
            rtol=1e-2,
        )
        self.assertFalse(diagnostic["reproduced"])

    def test_different_orders_of_catastrophic_divergence_do_not_reproduce(self):
        diagnostic = reproduction_diagnostic(current=2.0e14, original=1.0e14)
        self.assertFalse(diagnostic["reproduced"])
        self.assertEqual("mismatch", diagnostic["criterion"])

    def test_reproduction_summary_exposes_any_meaningful_mismatch(self):
        diagnostics = [
            reproduction_diagnostic(current=1.527216e14, original=1.585000e14),
            reproduction_diagnostic(current=1.02, original=1.00),
        ]
        summary = summarize_reproduction_diagnostics(diagnostics)
        self.assertFalse(summary["all_reproduced"])
        self.assertEqual(2, summary["comparison_count"])
        self.assertEqual(1, summary["divergent_order_count"])
        self.assertEqual(1, summary["mismatch_count"])


if __name__ == "__main__":
    unittest.main()
