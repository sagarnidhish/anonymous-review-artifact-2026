import unittest

from train.eval_plain_checkpoint import split_stems


class EvalPlainCheckpointTest(unittest.TestCase):
    def test_frozen_split_uses_25_degree_train_and_45_degree_test(self):
        train_stems, test_stems = split_stems("frozen")

        self.assertEqual(len(train_stems), 4)
        self.assertEqual(len(test_stems), 4)
        self.assertTrue(all("25deg" in stem for stem in train_stems))
        self.assertTrue(all("45deg" in stem for stem in test_stems))

    def test_lopo_split_holds_out_particle_four_at_both_temperatures(self):
        train_stems, test_stems = split_stems("lopo")

        self.assertEqual(len(train_stems), 6)
        self.assertEqual(
            test_stems,
            {
                "GRA29_C20_25deg_particle4",
                "GRA29_C20_45deg_particle4",
            },
        )

    def test_identity_holdout_requires_and_excludes_declared_particle(self):
        train_stems, test_stems = split_stems(
            "identity_holdout", heldout_particle=2
        )

        self.assertEqual(
            train_stems,
            {
                "GRA29_C20_25deg_particle1",
                "GRA29_C20_25deg_particle3",
                "GRA29_C20_25deg_particle4",
            },
        )
        self.assertEqual(
            test_stems,
            {
                "GRA29_C20_25deg_particle2",
                "GRA29_C20_45deg_particle2",
            },
        )

        with self.assertRaisesRegex(ValueError, "heldout_particle"):
            split_stems("identity_holdout")

    def test_unknown_split_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "unknown split"):
            split_stems("invented")


if __name__ == "__main__":
    unittest.main()
