import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from analysis.audit_external_walrus import audit_archive


class AuditExternalWalrusTest(unittest.TestCase):
    def _archive(self, root: Path, fixed_persistence: bool) -> Path:
        archive = root / "archive"
        results = archive / "results" / "walrus"
        results.mkdir(parents=True)
        pred = np.array([[[1.0]], [[2.0]]], dtype=np.float32)
        targets = np.array([[[2.0]], [[4.0]]], dtype=np.float32)
        if fixed_persistence:
            naive = np.array([[[1.0]], [[1.0]]], dtype=np.float32)
        else:
            naive = np.array([[[1.0]], [[2.0]]], dtype=np.float32)
        row = {
            "stem": "alice_2C",
            "role": "test",
            "mode": "rollout",
            "target_size": 128,
            "context_len": 4,
            "window_stride": 1,
            "subsample": 5,
            "max_windows": 0,
            "N": 2,
            "model_mse": float(np.mean((pred - targets) ** 2)),
            "naive_mse": float(np.mean((naive - targets) ** 2)),
            "ratio": float(
                np.mean((pred - targets) ** 2)
                / np.mean((naive - targets) ** 2)
            ),
            "ssim_model": 0.8,
            "ssim_naive": 0.9,
        }
        (results / "walrus_alice_rollout_results.json").write_text(
            json.dumps([row])
        )
        np.savez_compressed(
            results / "rollout_fixture.npz",
            pred=pred,
            naive=naive,
            targets=targets,
            row_json=np.array(json.dumps(row)),
        )
        (archive / "walrus_native_alice.py").write_text(
            "naive_next = naive_ctx[-1].copy()\n"
            "# naive context frozen: persistence keeps predicting the same frame\n"
        )
        return archive

    def test_valid_external_rollout_passes_with_appendix_limitations(self):
        with tempfile.TemporaryDirectory() as tmp:
            audit = audit_archive(self._archive(Path(tmp), True))

        self.assertEqual(audit["status"], "PASS_WITH_LIMITATIONS")
        self.assertTrue(audit["fixed_persistence_verified"])
        self.assertTrue(audit["appendix_admissible"])
        self.assertFalse(audit["main_benchmark_admissible"])
        self.assertEqual(audit["independent_rollout_movies"], 1)
        self.assertEqual(audit["metric_direction"], "mse_better_ssim_worse")

    def test_nonfixed_persistence_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            audit = audit_archive(self._archive(Path(tmp), False))

        self.assertEqual(audit["status"], "FAIL")
        self.assertFalse(audit["fixed_persistence_verified"])
        self.assertFalse(audit["appendix_admissible"])


if __name__ == "__main__":
    unittest.main()
