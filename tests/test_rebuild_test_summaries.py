import json
import tempfile
import unittest
from pathlib import Path

from analysis.rebuild_test_summaries import rebuild_summary


class RebuildTestSummariesTest(unittest.TestCase):
    def test_rebuild_uses_only_test_particles(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            detail = root / "next_frame_results.json"
            output = root / "comparison_summary.json"
            rows = [
                self._row("train1", "train", 0.1),
                self._row("test1", "test", 1.2),
                self._row("test2", "test", 1.4),
            ]
            detail.write_text(json.dumps(rows))

            summary = rebuild_summary(detail, output)

            self.assertEqual(summary["count"], 2)
            self.assertAlmostEqual(summary["mean_mae_ratio"], 1.3)
            self.assertEqual(summary["role"], "test")
            self.assertAlmostEqual(
                json.loads(output.read_text())[0]["mean_mae_ratio"], 1.3
            )

    def test_rebuild_fails_when_no_test_rows_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            detail = root / "next_frame_results.json"
            detail.write_text(json.dumps([self._row("train1", "train", 0.1)]))

            with self.assertRaisesRegex(ValueError, "no test rows"):
                rebuild_summary(detail, root / "comparison_summary.json")

    @staticmethod
    def _row(stem: str, role: str, ratio: float) -> dict:
        return {
            "stem": stem,
            "role": role,
            "model_mae": ratio,
            "naive_mae": 1.0,
            "mae_ratio": ratio,
            "model_rmse": ratio,
            "naive_rmse": 1.0,
            "rmse_ratio": ratio,
            "reversal_metrics": {"count": 0, "mae_ratio": float("nan")},
            "nonreversal_metrics": {"count": 1, "mae_ratio": ratio},
        }


if __name__ == "__main__":
    unittest.main()
