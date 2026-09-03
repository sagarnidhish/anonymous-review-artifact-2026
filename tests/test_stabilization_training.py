import unittest

import numpy as np
import torch

from train.ref.common_sp_baselines import LoadedSequence
from train.stabilization_training import choose_feedback_frame, corrupt_intensity_context
from train.train_rollout_stabilization import fine_tune


class TinyDeltaModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, values):
        return values[:, -1, :1] * self.scale


class StabilizationTrainingTest(unittest.TestCase):
    def test_scheduled_sampling_endpoint_selects_truth_or_prediction(self):
        truth = torch.ones((3, 1, 2, 2))
        prediction = torch.zeros_like(truth, requires_grad=True)
        rng = np.random.default_rng(7)

        teacher = choose_feedback_frame(truth, prediction, 1.0, True, rng)
        free = choose_feedback_frame(truth, prediction, 0.0, True, rng)

        torch.testing.assert_close(teacher, truth)
        torch.testing.assert_close(free, prediction.detach())
        self.assertFalse(free.requires_grad)

    def test_recursive_feedback_keeps_gradient_path(self):
        truth = torch.ones((1, 1, 2, 2))
        prediction = torch.zeros_like(truth, requires_grad=True)

        selected = choose_feedback_frame(
            truth, prediction, 0.0, False, np.random.default_rng(1)
        )

        self.assertTrue(selected.requires_grad)
        selected.sum().backward()
        torch.testing.assert_close(prediction.grad, torch.ones_like(prediction))

    def test_noise_augmentation_changes_context_without_mutating_source(self):
        context = torch.zeros((2, 4, 8, 8))
        original = context.clone()
        torch.manual_seed(4)

        corrupted = corrupt_intensity_context(context, noise_std=0.05)

        torch.testing.assert_close(context, original)
        self.assertGreater(float(torch.std(corrupted)), 0.0)
        self.assertEqual(context.shape, corrupted.shape)

    def test_fine_tune_executes_declared_free_rollout_selection(self):
        intensity = np.linspace(0, 1, 20, dtype=np.float32)[:, None, None]
        intensity = np.broadcast_to(intensity, (20, 4, 4)).copy()
        zeros = np.zeros(20, dtype=np.float32)
        sequence = LoadedSequence(
            stem="toy",
            role="train",
            intensity=intensity,
            exogenous_all={"voltage": zeros, "current": zeros, "time_norm": zeros},
            frame_times=np.arange(20, dtype=np.float32),
            raw_frame_indices=np.arange(20, dtype=np.int64),
            sequence_subsample_factor=1,
        )
        payload = {
            "payload_id": 9,
            "seed": 3,
            "strategy": "scheduled_sampling",
            "context_len": 2,
            "training_horizon": 2,
            "validation_horizon": 2,
            "validation_windows_per_stem": 1,
            "teacher_forcing_start": 0.0,
            "teacher_forcing_end": 0.0,
            "detach_feedback": True,
            "noise_std": 0.0,
            "epochs": 1,
            "lr": 1e-3,
            "patience": 1,
            "batch_windows": 1,
            "train_windows_per_stem_per_epoch": 2,
        }

        history, best = fine_tune(
            TinyDeltaModel(),
            [sequence],
            {"toy": np.asarray([0, 1])},
            {"toy": np.asarray([2])},
            payload,
            torch.device("cpu"),
        )

        self.assertEqual(1, len(history))
        self.assertEqual(1, best["epoch"])
        self.assertTrue(np.isfinite(best["validation_free_rollout_mae_h32"]))


if __name__ == "__main__":
    unittest.main()
