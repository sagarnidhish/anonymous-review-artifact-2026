import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "prepare_source_data",
    ROOT / "scripts" / "prepare_source_data.py",
)
PREPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREPARE)


class PrepareSourceDataTest(unittest.TestCase):
    def test_valid_archive_is_not_downloaded_again(self):
        payload = b"valid compressed payload"
        digest = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            source_dir = Path(tmp)
            (source_dir / "movie.h5.zst").write_bytes(payload)
            with mock.patch.object(
                PREPARE.urllib.request,
                "urlretrieve",
                side_effect=AssertionError("unexpected download"),
            ):
                PREPARE.download(source_dir, [(digest, "movie.h5.zst")])

    def test_lfs_pointer_is_replaced_by_verified_archive(self):
        payload = b"valid compressed payload"
        digest = hashlib.sha256(payload).hexdigest()

        def fake_download(url, destination, reporthook=None):
            Path(destination).write_bytes(payload)
            if reporthook is not None:
                reporthook(1, len(payload), len(payload))
            return str(destination), None

        with tempfile.TemporaryDirectory() as tmp:
            source_dir = Path(tmp)
            destination = source_dir / "movie.h5.zst"
            destination.write_text(
                "version https://git-lfs.github.com/spec/v1\n"
                "oid sha256:placeholder\nsize 123\n",
                encoding="utf-8",
            )
            with mock.patch.object(
                PREPARE.urllib.request,
                "urlretrieve",
                side_effect=fake_download,
            ):
                PREPARE.download(source_dir, [(digest, destination.name)])
            self.assertEqual(destination.read_bytes(), payload)


if __name__ == "__main__":
    unittest.main()
