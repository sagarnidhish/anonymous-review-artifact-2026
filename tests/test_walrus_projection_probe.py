import json
import tempfile
import unittest
from pathlib import Path

import torch

from train.walrus_projection_probe import (
    freeze_for_intensity_projection,
    load_accepted_tiny_gate,
    projection_entry_state,
    resolve_tiny_learning_rate,
    restore_projection_entry_state,
    zero_disallowed_projection_gradients,
)


class ToyProjection(torch.nn.Module):
    def __init__(self, in_fields, out_fields):
        super().__init__()
        self.proj1 = torch.nn.Conv3d(in_fields, 2, kernel_size=1, bias=False)
        self.proj2 = torch.nn.ConvTranspose3d(
            2, out_fields, kernel_size=1, bias=True
        )


class ToyWalrus(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.ModuleList(
            [torch.nn.Identity(), torch.nn.Identity(), ToyProjection(3, 3), ToyProjection(3, 3)]
        )
        self.debed = torch.nn.ModuleList(
            [torch.nn.Identity(), torch.nn.Identity(), ToyProjection(3, 3), ToyProjection(3, 3)]
        )
        self.backbone = torch.nn.Linear(4, 4)


class WalrusProjectionProbeTest(unittest.TestCase):
    def test_accepted_tiny_gate_is_bound_to_checkpoint_and_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "selector_manifest.json").write_text(
                json.dumps({"checkpoint_sha256": "a" * 64, "seed": 1337})
            )
            gate = {
                "status": "passed",
                "before": {"mean_model_mae": 1.0},
                "after": {"mean_model_mae": 0.7},
                "required_relative_loss": 0.9,
                "steps_completed": 128,
                "learning_rate": 1e-4,
                "gradient_parameter_names": ["embed.2.proj1.weight", "debed.2.proj2.weight"],
                "pretrained_projection_hashes_unchanged": True,
            }
            (root / "tiny_overfit_gate.json").write_text(json.dumps(gate))

            accepted = load_accepted_tiny_gate(root, "a" * 64, 1337)

            self.assertEqual(128, accepted["steps_completed"])
            with self.assertRaisesRegex(ValueError, "checkpoint"):
                load_accepted_tiny_gate(root, "b" * 64, 1337)

    def test_tiny_gate_learning_rate_can_be_lower_than_full_fit(self):
        self.assertEqual(5e-4, resolve_tiny_learning_rate(None, 5e-4))
        self.assertEqual(1e-4, resolve_tiny_learning_rate(1e-4, 5e-4))
        with self.assertRaisesRegex(ValueError, "positive"):
            resolve_tiny_learning_rate(0.0, 5e-4)

    def test_selector_enables_only_six_projection_containers(self):
        model = ToyWalrus()
        inventory = freeze_for_intensity_projection(model, intensity_idx=2)

        self.assertEqual(
            {
                "embed.2.proj1.weight",
                "embed.3.proj1.weight",
                "debed.2.proj2.weight",
                "debed.2.proj2.bias",
                "debed.3.proj2.weight",
                "debed.3.proj2.bias",
            },
            set(inventory),
        )
        self.assertFalse(model.backbone.weight.requires_grad)
        self.assertFalse(model.backbone.bias.requires_grad)
        self.assertTrue(model.embed[2].proj1.weight.requires_grad)

    def test_masked_optimizer_step_changes_only_the_new_field_entries(self):
        torch.manual_seed(4)
        model = ToyWalrus()
        inventory = freeze_for_intensity_projection(model, intensity_idx=2)
        before_all = {name: value.detach().clone() for name, value in model.named_parameters()}
        before_entries = projection_entry_state(model, inventory, intensity_idx=2)
        optimizer = torch.optim.SGD(
            [value for value in model.parameters() if value.requires_grad], lr=0.1
        )
        for name, value in model.named_parameters():
            if value.requires_grad:
                value.grad = torch.ones_like(value)
        observed = zero_disallowed_projection_gradients(
            model, inventory, intensity_idx=2
        )
        optimizer.step()

        self.assertEqual(set(inventory), set(observed))
        for name, value in model.named_parameters():
            old = before_all[name]
            if name.endswith("weight") and name in inventory:
                torch.testing.assert_close(value[:, :2], old[:, :2])
                self.assertFalse(torch.equal(value[:, 2], old[:, 2]))
            elif name.endswith("bias") and name in inventory:
                torch.testing.assert_close(value[:2], old[:2])
                self.assertNotEqual(value[2].item(), old[2].item())
            else:
                torch.testing.assert_close(value, old)

        restore_projection_entry_state(model, before_entries, intensity_idx=2)
        restored = projection_entry_state(model, inventory, intensity_idx=2)
        for name in before_entries:
            torch.testing.assert_close(restored[name], before_entries[name])


if __name__ == "__main__":
    unittest.main()
