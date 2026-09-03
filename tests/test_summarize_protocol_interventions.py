import json
import tempfile
import unittest
from pathlib import Path

from analysis.summarize_protocol_interventions import summarize


class SummarizeProtocolInterventionsTest(unittest.TestCase):
    def test_summary_uses_paired_particle_deltas_and_requires_full_grid(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = []
            for stem, offset in (("particle1", 0.0), ("particle2", 0.2)):
                for mode in ("next_frame", "rollout_anchored"):
                    for condition, effect in (
                        ("true", 0.0),
                        ("zero", 0.1),
                        ("shuffle", 0.5),
                        ("shift", 0.2),
                    ):
                        rows.append(
                            {
                                "stem": stem,
                                "mode": mode,
                                "condition": condition,
                                "mae_ratio": 1.0 + offset + effect,
                                "prediction_change_l1": effect,
                            }
                        )
            (root / "per_particle_results.json").write_text(
                json.dumps(rows)
            )

            payload = summarize(root)

            shuffle = payload["modes"]["next_frame"]["shuffle"]
            self.assertEqual(shuffle["n_particles"], 2)
            self.assertAlmostEqual(shuffle["mean_mae_ratio"], 1.6)
            self.assertAlmostEqual(shuffle["mean_paired_delta_mae_ratio"], 0.5)
            self.assertEqual(shuffle["paired_delta_mae_ratio"], [0.5, 0.5])

            rows.pop()
            (root / "per_particle_results.json").write_text(
                json.dumps(rows)
            )
            with self.assertRaisesRegex(ValueError, "incomplete intervention grid"):
                summarize(root)


if __name__ == "__main__":
    unittest.main()
