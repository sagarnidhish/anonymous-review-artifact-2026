import unittest

from train.identity_fold_payloads import build_payloads


class IdentityFoldPayloadsTest(unittest.TestCase):
    def test_full_grid_contains_two_models_and_four_particle_folds(self):
        payloads = build_payloads(pilot=False)

        self.assertEqual(len(payloads), 8)
        tags = {payload["cfg"]["tag"] for payload in payloads}
        self.assertEqual(len(tags), 8)
        self.assertEqual(
            {(payload["cfg"]["model_family"], payload["cfg"]["heldout_particle"])
             for payload in payloads},
            {(family, particle)
             for family in ("unet", "predrnnpp")
             for particle in range(1, 5)},
        )

        for payload in payloads:
            cfg = payload["cfg"]
            self.assertEqual(payload["mode"], "train")
            self.assertEqual(cfg["split"], "identity_holdout")
            self.assertEqual(cfg["seed"], 1337)
            self.assertTrue(cfg["predict_delta"])
            self.assertFalse(cfg["use_voltage"])
            self.assertFalse(cfg["use_current"])
            self.assertFalse(cfg["use_time_norm"])
            self.assertEqual(cfg["epochs"], 60)
            self.assertEqual(cfg["max_train_windows_per_stem"], 3000)
            self.assertEqual(cfg["max_eval_windows"], 3000)
            self.assertEqual(cfg["max_rollout_steps"], 512)

    def test_pilot_payloads_are_small_and_cannot_collide_with_full_runs(self):
        pilot_payloads = build_payloads(pilot=True)
        full_tags = {payload["cfg"]["tag"] for payload in build_payloads(False)}

        self.assertEqual(len(pilot_payloads), 2)
        for payload in pilot_payloads:
            cfg = payload["cfg"]
            self.assertTrue(cfg["tag"].endswith("_pilot"))
            self.assertNotIn(cfg["tag"], full_tags)
            self.assertEqual(cfg["heldout_particle"], 1)
            self.assertEqual(cfg["epochs"], 1)
            self.assertEqual(cfg["max_train_windows_per_stem"], 300)
            self.assertEqual(cfg["max_eval_windows"], 300)
            self.assertEqual(cfg["max_rollout_steps"], 32)


if __name__ == "__main__":
    unittest.main()
