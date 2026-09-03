import tempfile
import unittest
from pathlib import Path

from analysis.make_matched_grid_figure import make_matched_grid_figure
from train.matched_grid_payloads import build_payloads


class MakeMatchedGridFigureTest(unittest.TestCase):
    def test_complete_grid_renders_nonempty_pdf(self):
        rows = []
        for payload in build_payloads():
            row = dict(payload)
            row.update(
                {
                    "next_frame_mean_mae_ratio": 0.8 + 0.03 * payload["payload_id"],
                    "next_frame_particles_better": payload["payload_id"] % 5,
                    "rollout_diverged_particles": payload["payload_id"] % 2,
                }
            )
            for horizon in payload["report_horizons"]:
                row[f"rollout_mean_mae_ratio_h{horizon}"] = (
                    None
                    if payload["payload_id"] == 23 and horizon == 512
                    else 1.0 + horizon / 256 + payload["payload_id"] / 10
                )
                row[f"rollout_particles_better_h{horizon}"] = (
                    payload["payload_id"] + horizon
                ) % 5
            rows.append(row)
        summary = {"status": "complete", "payloads": rows}

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "grid.pdf"
            make_matched_grid_figure(summary, output)
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 5_000)

    def test_incomplete_grid_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "24 payload"):
                make_matched_grid_figure(
                    {"status": "complete", "payloads": []},
                    Path(tmp) / "grid.pdf",
                )


if __name__ == "__main__":
    unittest.main()
