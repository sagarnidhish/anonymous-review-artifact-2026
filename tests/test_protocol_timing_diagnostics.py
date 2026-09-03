import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from analysis.protocol_timing_diagnostics import (
    channel_perturbation,
    measured_transition_mask,
    model_facing_channel_windows,
    summarize_prediction_subset,
)


class ProtocolTimingDiagnosticsTest(unittest.TestCase):
    def test_zero_delay_has_zero_difference_and_unit_correlation(self):
        measured = np.array([-1.0, -0.5, 0.25, 1.0], dtype=np.float32)

        result = channel_perturbation(measured, measured.copy())

        self.assertEqual(result["mean_abs_difference"], 0.0)
        self.assertEqual(result["difference_over_sigma"], 0.0)
        self.assertAlmostEqual(result["correlation"], 1.0)

    def test_transition_mask_uses_absolute_measured_current_indices(self):
        measured_current = np.array(
            [-1, -1, 0, 1, 1, -1, -1], dtype=np.float32
        )
        target_steps = np.array([1, 2, 3, 4, 5, 6], dtype=np.int64)

        mask = measured_transition_mask(
            measured_current, target_steps, radius=0
        )

        np.testing.assert_array_equal(
            mask, np.array([False, False, True, False, True, False])
        )

    def test_subset_summary_applies_one_supplied_frame_mask(self):
        targets = np.zeros((3, 2, 2), dtype=np.float32)
        naive = np.ones_like(targets)
        pred = np.stack(
            [
                np.zeros((2, 2), dtype=np.float32),
                np.full((2, 2), 2.0, dtype=np.float32),
                np.full((2, 2), 4.0, dtype=np.float32),
            ]
        )
        measured_mask = np.array([False, True, False])

        result = summarize_prediction_subset(
            pred, naive, targets, measured_mask
        )

        self.assertEqual(result["N"], 1)
        self.assertEqual(result["model_mae"], 2.0)
        self.assertEqual(result["naive_mae"], 1.0)
        self.assertEqual(result["mae_ratio"], 2.0)

    def test_model_facing_windows_match_stride_one_contexts(self):
        values = np.arange(7, dtype=np.float32)

        windows = model_facing_channel_windows(
            values, context_len=3, max_windows=2
        )

        np.testing.assert_array_equal(windows, [[0, 1, 2], [1, 2, 3]])

    def test_direct_script_entrypoint_resolves_repository_imports(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            temp = Path(tmp)
            arrays = temp / "arrays"
            arrays.mkdir()
            for particle in range(1, 5):
                np.savez_compressed(
                    arrays / f"GRA29_C20_45deg_particle{particle}.npz",
                    voltage=np.linspace(0, 1, 8, dtype=np.float32),
                    current=np.array([-1, -1, -1, -1, 1, 1, 1, 1], np.float32),
                    time_norm=np.linspace(0, 1, 8, dtype=np.float32),
                )
            output = temp / "out.json"
            table = temp / "out.tex"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(root / "analysis" / "protocol_timing_diagnostics.py"),
                    "--arrays-dir",
                    str(arrays),
                    "--output",
                    str(output),
                    "--table",
                    str(table),
                    "--delays",
                    "0",
                    "--max-windows",
                    "2",
                ],
                cwd=temp,
                text=True,
                capture_output=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(output.is_file())
            self.assertTrue(table.is_file())


if __name__ == "__main__":
    unittest.main()
