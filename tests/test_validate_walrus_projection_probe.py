import json
import tempfile
import unittest
from pathlib import Path

import torch

from analysis.validate_walrus_projection_probe import validate_probe


def write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def make_probe(root: Path):
    inventory = {
        "embed.2.proj1.weight": {"axis": 1, "shape": [2, 3, 1, 1, 1], "active_entries": 2},
        "embed.3.proj1.weight": {"axis": 1, "shape": [2, 3, 1, 1, 1], "active_entries": 2},
        "debed.2.proj2.weight": {"axis": 1, "shape": [2, 3, 1, 1, 1], "active_entries": 2},
        "debed.2.proj2.bias": {"axis": 0, "shape": [3], "active_entries": 1},
        "debed.3.proj2.weight": {"axis": 1, "shape": [2, 3, 1, 1, 1], "active_entries": 2},
        "debed.3.proj2.bias": {"axis": 0, "shape": [3], "active_entries": 1},
    }
    write(
        root / "selector_manifest.json",
        {
            "inventory": inventory,
            "intensity_index": 2,
            "all_other_parameters_frozen": True,
            "pretrained_projection_hashes_before": {name: "a" for name in inventory},
        },
    )
    write(
        root / "tiny_overfit_gate.json",
        {
            "status": "passed",
            "before": {"mean_model_mae": 2.0},
            "after": {"mean_model_mae": 1.0},
            "gradient_parameter_names": list(inventory),
            "pretrained_projection_hashes_unchanged": True,
        },
    )
    rows = [
        {"stem": f"GRA29_C20_45deg_particle{i}", "model_mae": 1.0}
        for i in range(1, 5)
    ]
    write(
        root / "training_curve.json",
        {
            "split": {
                "train_stems": [f"GRA29_C20_25deg_particle{i}" for i in range(1, 5)],
                "test_stems": [f"GRA29_C20_45deg_particle{i}" for i in range(1, 5)],
                "train_start_range": [0, 2699],
                "validation_start_range": [2700, 2999],
            },
            "gradient_parameter_names": list(inventory),
            "baseline_test": {"particle_rows": rows},
            "fitted_test": {"particle_rows": rows},
        },
    )
    checkpoint = root / "intensity_projection_probe.pt"
    torch.save(
        {
            "projection_entries": {
                name: torch.zeros(meta["active_entries"])
                for name, meta in inventory.items()
            },
            "inventory": inventory,
            "intensity_index": 2,
        },
        checkpoint,
    )
    write(
        root / "completion_manifest.json",
        {
            "status": "passed",
            "tiny_gate_passed": True,
            "validation_improved": True,
            "baseline_validation": {"mean_model_mae": 2.0},
            "fitted_validation": {"mean_model_mae": 1.0},
            "projection_checkpoint": checkpoint.name,
            "pretrained_projection_hashes_unchanged": True,
        },
    )


class ValidateWalrusProjectionProbeTest(unittest.TestCase):
    def test_complete_passed_probe_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_probe(root)
            result = validate_probe(root)
            self.assertEqual("passed", result["status"])
            self.assertEqual(6, result["projection_container_count"])

    def test_false_validation_improvement_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_probe(root)
            path = root / "completion_manifest.json"
            data = json.loads(path.read_text())
            data["fitted_validation"]["mean_model_mae"] = 3.0
            write(path, data)
            with self.assertRaisesRegex(ValueError, "did not improve"):
                validate_probe(root)


if __name__ == "__main__":
    unittest.main()
