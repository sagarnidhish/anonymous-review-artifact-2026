import unittest

from train.eval_protocol_delay_sweep import _aggregate


def _row(stem, condition, ratio, transition_ratio, response):
    perturbation = {
        channel: {
            "mean_abs_difference": response,
            "measured_sigma": 1.0,
            "difference_over_sigma": response,
            "correlation": 1.0 - response,
        }
        for channel in ("voltage", "current")
    }
    return {
        "stem": stem,
        "condition": condition,
        "mae_ratio": ratio,
        "prediction_change_l1": response,
        "fixed_transition": {
            "transition": {"N": 1, "mae_ratio": transition_ratio}
        },
        "input_perturbation": perturbation,
    }


class ProtocolDelaySweepTest(unittest.TestCase):
    def test_aggregate_reports_paired_particle_delta(self):
        rows = [
            _row("p1", "measured", 0.8, 0.9, 0.0),
            _row("p2", "measured", 1.0, 1.1, 0.0),
            _row("p1", "delay_16", 0.9, 1.0, 0.1),
            _row("p2", "delay_16", 1.3, 1.4, 0.2),
        ]

        result = _aggregate(rows, ["measured", "delay_16"])

        self.assertAlmostEqual(
            result["delay_16"]["mean_paired_delta_mae_ratio"], 0.2
        )
        self.assertAlmostEqual(
            result["delay_16"]["mean_transition_mae_ratio"], 1.2
        )
        self.assertAlmostEqual(
            result["delay_16"]["input_perturbation"]["voltage"][
                "difference_over_sigma"
            ],
            0.15,
        )

    def test_aggregate_rejects_incomplete_particle_grid(self):
        rows = [
            _row("p1", "measured", 0.8, 0.9, 0.0),
            _row("p2", "measured", 1.0, 1.1, 0.0),
            _row("p1", "delay_16", 0.9, 1.0, 0.1),
        ]

        with self.assertRaisesRegex(ValueError, "incomplete"):
            _aggregate(rows, ["measured", "delay_16"])


if __name__ == "__main__":
    unittest.main()

