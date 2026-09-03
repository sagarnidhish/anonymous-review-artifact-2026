import unittest

from train.particle_splits import (
    build_identity_holdout_fold,
    particle_identity_from_stem,
)


class ParticleSplitsTest(unittest.TestCase):
    def test_every_identity_holdout_fold_excludes_test_particle_from_training(self):
        for heldout_particle in range(1, 5):
            with self.subTest(heldout_particle=heldout_particle):
                fold = build_identity_holdout_fold(heldout_particle)
                expected_train = {
                    f"GRA29_C20_25deg_particle{particle}"
                    for particle in range(1, 5)
                    if particle != heldout_particle
                }

                self.assertEqual(fold.train_stems, expected_train)
                self.assertEqual(
                    fold.same_temperature_test_stems,
                    {f"GRA29_C20_25deg_particle{heldout_particle}"},
                )
                self.assertEqual(
                    fold.cross_temperature_test_stems,
                    {f"GRA29_C20_45deg_particle{heldout_particle}"},
                )
                self.assertFalse(fold.train_stems & fold.all_test_stems)

    def test_test_pair_has_one_shared_physical_particle_identity(self):
        for heldout_particle in range(1, 5):
            fold = build_identity_holdout_fold(heldout_particle)
            identities = {
                particle_identity_from_stem(stem)
                for stem in fold.all_test_stems
            }

            self.assertEqual(identities, {heldout_particle})

    def test_evaluation_groups_are_explicit(self):
        fold = build_identity_holdout_fold(3)

        self.assertEqual(
            fold.evaluation_group_for_stem("GRA29_C20_25deg_particle3"),
            "same_temperature_unseen_particle",
        )
        self.assertEqual(
            fold.evaluation_group_for_stem("GRA29_C20_45deg_particle3"),
            "cross_temperature_unseen_particle",
        )
        with self.assertRaisesRegex(ValueError, "outside identity-holdout test"):
            fold.evaluation_group_for_stem("GRA29_C20_25deg_particle1")

    def test_invalid_particle_or_stem_fails_closed(self):
        for particle in (0, 5):
            with self.subTest(particle=particle):
                with self.assertRaisesRegex(ValueError, "heldout_particle"):
                    build_identity_holdout_fold(particle)

        with self.assertRaisesRegex(ValueError, "GRA29 particle stem"):
            particle_identity_from_stem("particle1")


if __name__ == "__main__":
    unittest.main()
