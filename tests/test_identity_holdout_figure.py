import tempfile
import unittest
from pathlib import Path

from analysis.make_identity_holdout_figure import make_figure


def synthetic_summary() -> dict:
    rows = []
    for model_index, model in enumerate(("unet", "predrnnpp")):
        for particle in range(1, 5):
            for group_index, group in enumerate(
                (
                    "same_temperature_unseen_particle",
                    "cross_temperature_unseen_particle",
                )
            ):
                for mode_index, mode in enumerate(("next_frame", "rollout")):
                    rows.append(
                        {
                            "model_family": model,
                            "heldout_particle": particle,
                            "evaluation_group": group,
                            "mode": mode,
                            "stem": f"synthetic_particle{particle}",
                            "mae_ratio": (
                                0.7
                                + 0.1 * model_index
                                + 0.15 * group_index
                                + 0.5 * mode_index
                                + 0.02 * particle
                            ),
                        }
                    )
    return {
        "aggregation_unit": "physical_particle",
        "shared_cell_level_protocol": True,
        "particle_count": 4,
        "per_particle": rows,
    }


class IdentityHoldoutFigureTest(unittest.TestCase):
    def test_complete_particle_grid_renders_nonempty_pdf(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "identity.pdf"
            make_figure(synthetic_summary(), output)

            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 5_000)

    def test_missing_particle_record_is_rejected(self):
        summary = synthetic_summary()
        summary["per_particle"].pop()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "complete four-particle grid"):
                make_figure(summary, Path(tmp) / "identity.pdf")


if __name__ == "__main__":
    unittest.main()
