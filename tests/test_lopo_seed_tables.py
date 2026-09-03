import json
import tempfile
import unittest
from pathlib import Path

from analysis.make_lopo_seed_tables import REQUIRED_TAGS, build_tables


class LopoSeedTablesTest(unittest.TestCase):
    def _write_summary(self, root: Path, tag: str, ratio: float) -> None:
        directory = root / tag / "next_frame"
        directory.mkdir(parents=True)
        (directory / "comparison_summary.json").write_text(
            json.dumps([{"mean_mae_ratio": ratio}])
        )

    def test_missing_required_runs_fail_loudly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "tables"
            self._write_summary(root, "unet_image_only_delta", 0.8)

            with self.assertRaisesRegex(
                FileNotFoundError, "predrnnpp_img_delta_lopo"
            ):
                build_tables(root, output)

            self.assertFalse((output / "lopo.tex").exists())
            self.assertFalse((output / "seeds.tex").exists())

    def test_complete_run_set_produces_nonempty_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "tables"
            for index, tag in enumerate(REQUIRED_TAGS):
                self._write_summary(root, tag, 0.8 + 0.01 * index)

            build_tables(root, output)

            lopo = (output / "lopo.tex").read_text()
            seeds = (output / "seeds.tex").read_text()
            self.assertIn("particle-4 robustness split", lopo)
            self.assertGreaterEqual(lopo.count("\\\\"), 5)
            self.assertGreaterEqual(seeds.count("\\\\"), 3)


if __name__ == "__main__":
    unittest.main()
