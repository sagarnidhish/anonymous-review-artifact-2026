import unittest

import numpy as np
import torch

from train.ref.common_sp_baselines import LoadedSequence
from train.safe_anchored_rollout import (
    cumulative_horizon_metrics,
    run_safe_rollout,
)


class ZeroModel(torch.nn.Module):
    def forward(self, x):
        return torch.zeros((len(x), 1, x.shape[-2], x.shape[-1]), device=x.device)


class InfiniteModel(torch.nn.Module):
    def forward(self, x):
        return torch.full(
            (len(x), 1, x.shape[-2], x.shape[-1]),
            float("inf"),
            device=x.device,
        )


def make_sequence():
    intensity = np.arange(8 * 4, dtype=np.float32).reshape(8, 2, 2) / 10
    zeros = np.zeros(8, dtype=np.float32)
    return LoadedSequence(
        stem="synthetic",
        role="test",
        intensity=intensity,
        exogenous_all={"voltage": zeros, "current": zeros, "time_norm": zeros},
        frame_times=np.arange(8, dtype=np.float32),
        raw_frame_indices=np.arange(8, dtype=np.int64),
        sequence_subsample_factor=1,
    )


class SafeAnchoredRolloutTest(unittest.TestCase):
    def test_zero_delta_model_uses_one_fixed_persistence_frame(self):
        result = run_safe_rollout(
            model=ZeroModel(),
            sequence=make_sequence(),
            active_fields=["intensity"],
            context_len=2,
            device=torch.device("cpu"),
            predict_delta=True,
            max_rollout_steps=4,
        )

        self.assertEqual("complete", result["status"])
        self.assertIsNone(result["first_nonfinite_step"])
        np.testing.assert_array_equal(result["target_steps"], [2, 3, 4, 5])
        np.testing.assert_allclose(result["pred"], result["naive"])
        np.testing.assert_allclose(
            result["naive"],
            np.repeat(make_sequence().intensity[1][None], 4, axis=0),
        )

    def test_nonfinite_prediction_is_recorded_without_fabricating_later_frames(self):
        result = run_safe_rollout(
            model=InfiniteModel(),
            sequence=make_sequence(),
            active_fields=["intensity"],
            context_len=2,
            device=torch.device("cpu"),
            predict_delta=False,
            max_rollout_steps=4,
        )

        self.assertEqual("numerically_diverged", result["status"])
        self.assertEqual(1, result["first_nonfinite_step"])
        self.assertTrue(np.isnan(result["pred"]).all())
        self.assertTrue(np.isfinite(result["targets"]).all())
        self.assertTrue(np.isfinite(result["naive"]).all())

    def test_horizon_metrics_mark_post_divergence_ratios_as_infinite(self):
        targets = np.ones((4, 2, 2), dtype=np.float32)
        naive = np.zeros_like(targets)
        pred = np.stack(
            [np.zeros((2, 2), np.float32), np.full((2, 2), np.nan, np.float32),
             np.full((2, 2), np.nan, np.float32), np.full((2, 2), np.nan, np.float32)]
        )
        metrics = cumulative_horizon_metrics(
            pred, naive, targets, horizons=(1, 2, 4), first_nonfinite_step=2
        )

        self.assertEqual(1.0, metrics["1"]["mae_ratio"])
        self.assertEqual(float("inf"), metrics["2"]["mae_ratio"])
        self.assertEqual(float("inf"), metrics["4"]["mae_ratio"])


if __name__ == "__main__":
    unittest.main()
