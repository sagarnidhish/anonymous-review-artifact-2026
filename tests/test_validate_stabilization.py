import tempfile
import unittest
from pathlib import Path

from analysis.validate_stabilization import validate_campaign


class ValidateStabilizationTest(unittest.TestCase):
    def test_complete_campaign_requires_every_declared_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "missing required JSON"):
                validate_campaign(Path(tmp))

    def test_payload_id_is_bounded(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "outside"):
                validate_campaign(Path(tmp), payload_id=3)


if __name__ == "__main__":
    unittest.main()
