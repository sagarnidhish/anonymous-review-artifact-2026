import csv
import unittest
from pathlib import Path

from analysis.make_tables import render_benchmark


ROOT = Path(__file__).resolve().parents[1]


class MakeTablesTest(unittest.TestCase):
    def test_benchmark_header_names_all_prediction_forms_and_rollout_scope(self):
        source = ROOT / "results" / "frozen" / "baseline_suite_test_summary.csv"
        with source.open(newline="") as handle:
            rows = list(csv.DictReader(handle))

        latex = render_benchmark(rows)

        self.assertIn("Next-frame MAE ratio", latex)
        self.assertIn("256-step rollout", latex)
        self.assertIn(r"image $\Delta$ & protocol $\Delta$", latex)
        self.assertIn("image direct & protocol direct", latex)
        self.assertIn("best ratio (available)", latex)
        self.assertNotIn("best direct", latex)
        self.assertIn("SimVP", latex)
        self.assertRegex(latex, r"SimVP .* \(1/4\) \\")
        self.assertRegex(latex, r"U-Net .* \(4/4\) \\")


if __name__ == "__main__":
    unittest.main()
