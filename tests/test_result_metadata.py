import unittest

from train.result_metadata import build_result_row, rows_for_role
from train.train_plain import make_evaluation_row


class ResultMetadataTest(unittest.TestCase):
    def test_result_row_contains_reference_saver_contract(self):
        row = build_result_row(
            stem="particle4",
            role="test",
            model_family="predrnnpp",
            tag="predrnnpp_img_delta_lopo",
            active_fields=["intensity"],
            context_len=4,
            split="lopo",
            seed=1337,
            prediction_form="delta_from_last_frame",
        )

        self.assertEqual(row["active_fields"], ["intensity"])
        self.assertEqual(row["context_len"], 4)
        self.assertEqual(row["split"], "lopo")
        self.assertEqual(row["seed"], 1337)
        self.assertEqual(row["prediction_form"], "delta_from_last_frame")
        self.assertEqual(row["stem"], "particle4")

    def test_result_row_copies_active_fields(self):
        fields = ["intensity"]
        row = build_result_row(
            stem="particle4",
            role="test",
            model_family="unet",
            tag="tag",
            active_fields=fields,
            context_len=4,
            split="frozen",
            seed=2026,
            prediction_form="direct_frame",
        )

        fields.append("current")

        self.assertEqual(row["active_fields"], ["intensity"])

    def test_plain_evaluator_uses_complete_result_metadata(self):
        row = make_evaluation_row(
            stem="particle4",
            role="test",
            model_family="predrnnpp",
            tag="predrnnpp_img_delta_lopo",
            active_fields=["intensity"],
            context_len=4,
            split="lopo",
            seed=1337,
        )

        self.assertEqual(row["active_fields"], ["intensity"])
        self.assertEqual(row["context_len"], 4)
        self.assertEqual(row["split"], "lopo")
        self.assertEqual(row["seed"], 1337)

    def test_identity_holdout_metadata_records_fold_and_evaluation_group(self):
        row = build_result_row(
            stem="GRA29_C20_45deg_particle2",
            role="test",
            model_family="unet",
            tag="unet_identity_holdout_p2",
            active_fields=["intensity"],
            context_len=4,
            split="identity_holdout",
            seed=1337,
            prediction_form="delta_from_last_frame",
            heldout_particle=2,
            evaluation_group="cross_temperature_unseen_particle",
        )

        self.assertEqual(row["heldout_particle"], 2)
        self.assertEqual(
            row["evaluation_group"],
            "cross_temperature_unseen_particle",
        )

    def test_test_summary_excludes_training_particles(self):
        rows = [
            {"stem": "train1", "role": "train", "mae_ratio": 0.1},
            {"stem": "test1", "role": "test", "mae_ratio": 1.2},
            {"stem": "test2", "role": "test", "mae_ratio": 1.4},
        ]

        selected = rows_for_role(rows, "test")

        self.assertEqual([row["stem"] for row in selected],
                         ["test1", "test2"])


if __name__ == "__main__":
    unittest.main()
