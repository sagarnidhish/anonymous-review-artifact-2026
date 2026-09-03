import unittest

import numpy as np

from train.protocol_interventions import intervene_exogenous


class ProtocolInterventionsTest(unittest.TestCase):
    def setUp(self):
        self.exogenous = {
            "voltage": np.arange(8, dtype=np.float32),
            "current": np.arange(8, dtype=np.float32) * -2,
            "time_norm": np.linspace(0, 1, 8, dtype=np.float32),
        }

    def test_zero_changes_only_voltage_and_current(self):
        modified = intervene_exogenous(self.exogenous, "zero", seed=7, shift=2)

        np.testing.assert_array_equal(modified["voltage"], 0)
        np.testing.assert_array_equal(modified["current"], 0)
        np.testing.assert_array_equal(
            modified["time_norm"], self.exogenous["time_norm"]
        )

    def test_shuffle_is_deterministic_and_keeps_voltage_current_paired(self):
        first = intervene_exogenous(self.exogenous, "shuffle", seed=7, shift=2)
        second = intervene_exogenous(self.exogenous, "shuffle", seed=7, shift=2)

        np.testing.assert_array_equal(first["voltage"], second["voltage"])
        np.testing.assert_array_equal(first["current"], second["current"])
        np.testing.assert_array_equal(first["current"], first["voltage"] * -2)

    def test_shift_delays_protocol_without_wrapping_future_values(self):
        modified = intervene_exogenous(self.exogenous, "shift", seed=7, shift=2)

        np.testing.assert_array_equal(
            modified["voltage"], [0, 0, 0, 1, 2, 3, 4, 5]
        )
        np.testing.assert_array_equal(
            modified["current"], [0, 0, 0, -2, -4, -6, -8, -10]
        )

    def test_true_returns_copies(self):
        modified = intervene_exogenous(self.exogenous, "true", seed=7, shift=2)
        modified["voltage"][0] = 99

        self.assertEqual(self.exogenous["voltage"][0], 0)

    def test_unknown_intervention_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "unknown intervention"):
            intervene_exogenous(self.exogenous, "invented", seed=7, shift=2)


if __name__ == "__main__":
    unittest.main()
