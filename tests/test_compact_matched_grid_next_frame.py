import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from analysis.compact_matched_grid_next_frame import convert_payload
from analysis.validate_matched_grid import validate_payload_directory
from tests.test_validate_matched_grid import STEMS, make_payload_tree, write_json


def make_legacy_full_payload(root: Path):
    payload, run = make_payload_tree(root)
    manifest_path = run / "completion_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("next_frame_artifact_mode")
    write_json(manifest_path, manifest)
    for stem in STEMS:
        path = run / "next_frame" / stem / f"preds_{stem}.npz"
        pred = np.ones((3, 2, 2), dtype=np.float32)
        naive = np.full_like(pred, 2)
        target = np.zeros_like(pred)
        np.savez_compressed(
            path,
            pred=pred,
            naive=naive,
            targets=target,
            target_steps=np.asarray([10, 11, 12], dtype=np.int64),
            frame_times=np.asarray([0, 1, 2], dtype=np.float32),
        )
        write_json(
            run / "next_frame" / stem / "study_manifest.json",
            {
                "mode": "next_frame",
                "stem": stem,
                "evaluation_scope": "full_frame_only",
            },
        )
    validate_payload_directory(run, payload)
    return payload, run


class CompactMatchedGridNextFrameTest(unittest.TestCase):
    def test_completed_dense_payload_is_converted_without_changing_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload, run = make_legacy_full_payload(Path(tmp))
            results_path = run / "next_frame" / "next_frame_results.json"
            original_results = results_path.read_bytes()

            summary = convert_payload(run, payload, sample_count=2)

            self.assertEqual("complete", summary["status"])
            self.assertEqual("compact_samples_v1", summary["artifact_mode"])
            self.assertEqual(original_results, results_path.read_bytes())
            manifest = json.loads(
                (run / "completion_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                "compact_samples_v1", manifest["next_frame_artifact_mode"]
            )
            for stem in STEMS:
                path = run / "next_frame" / stem / f"preds_{stem}.npz"
                with np.load(path) as data:
                    self.assertEqual(
                        "compact_samples_v1", data["artifact_mode"].item()
                    )
                    self.assertEqual([0, 2], data["sample_offsets"].tolist())
            validate_payload_directory(run, payload)

    def test_payload_without_completion_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload, _ = make_legacy_full_payload(Path(tmp))
            incomplete = Path(tmp) / "incomplete"
            incomplete.mkdir()
            with self.assertRaisesRegex(ValueError, "missing required JSON"):
                convert_payload(incomplete, payload)


if __name__ == "__main__":
    unittest.main()
