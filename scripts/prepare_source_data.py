#!/usr/bin/env python3
"""Download, verify, and decompress the anonymous source-data release."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import urllib.request
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CHECKSUM_FILE = REPOSITORY_ROOT / "data" / "SHA256SUMS"
DEFAULT_SOURCE_DIR = REPOSITORY_ROOT / "data" / "source"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "data" / "source_uncompressed"
BASE_URL = (
    "https://anonymous.4open.science/api/repo/"
    "anonymous-review-artifact-2026-A03A/file/data/source"
)


def checksum_entries() -> list[tuple[str, str]]:
    entries = []
    for line in CHECKSUM_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, relative_path = line.split(maxsplit=1)
        entries.append((digest, Path(relative_path).name))
    return entries


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def progress_hook(filename: str):
    last_report = -1

    def report(block_count: int, block_size: int, total_size: int) -> None:
        nonlocal last_report
        downloaded = block_count * block_size
        report_number = downloaded // (64 * 1024 * 1024)
        if (
            report_number == last_report
            and (total_size <= 0 or downloaded < total_size)
        ):
            return
        last_report = report_number
        downloaded_mib = downloaded / (1024 * 1024)
        if total_size > 0:
            total_mib = total_size / (1024 * 1024)
            print(
                f"  {filename}: {downloaded_mib:.0f}/{total_mib:.0f} MiB",
                end="\r",
                flush=True,
            )
        else:
            print(
                f"  {filename}: {downloaded_mib:.0f} MiB",
                end="\r",
                flush=True,
            )

    return report


def download(source_dir: Path, expected: list[tuple[str, str]]) -> None:
    source_dir.mkdir(parents=True, exist_ok=True)
    for expected_digest, filename in expected:
        destination = source_dir / filename
        if destination.exists() and sha256(destination) == expected_digest:
            print(f"Already downloaded and verified {filename}")
            continue
        if destination.exists():
            print(f"Replacing Git LFS pointer or incomplete file: {filename}")
        url = f"{BASE_URL}/{filename}?download=true"
        partial = destination.with_suffix(destination.suffix + ".part")
        print(f"Downloading {filename}")
        partial.unlink(missing_ok=True)
        try:
            urllib.request.urlretrieve(
                url,
                partial,
                reporthook=progress_hook(filename),
            )
            print()
            actual_digest = sha256(partial)
            if actual_digest != expected_digest:
                raise ValueError(
                    f"SHA-256 mismatch for downloaded {filename}: "
                    f"expected {expected_digest}, found {actual_digest}"
                )
            partial.replace(destination)
            print(f"Downloaded and verified {filename}")
        except Exception:
            partial.unlink(missing_ok=True)
            raise


def verify(source_dir: Path, expected: list[tuple[str, str]]) -> None:
    for expected_digest, filename in expected:
        path = source_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing source archive: {path}")
        actual_digest = sha256(path)
        if actual_digest != expected_digest:
            raise ValueError(
                f"SHA-256 mismatch for {filename}: "
                f"expected {expected_digest}, found {actual_digest}"
            )
        print(f"Verified {filename}")


def decompress(
    source_dir: Path,
    output_dir: Path,
    expected: list[tuple[str, str]],
) -> None:
    zstd = shutil.which("zstd")
    if zstd is None:
        raise RuntimeError(
            "The 'zstd' executable is required for decompression. "
            "Install Zstandard using your operating system package manager."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    for _, filename in expected:
        archive = source_dir / filename
        if not filename.endswith(".zst"):
            raise ValueError(f"Unexpected archive name: {filename}")
        destination = output_dir / filename[:-4]
        print(f"Decompressing {filename}")
        subprocess.run(
            [zstd, "-d", "-f", str(archive), "-o", str(destination)],
            check=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir", type=Path, default=DEFAULT_SOURCE_DIR,
        help="Directory containing compressed source archives",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Directory for decompressed HDF5 files",
    )
    parser.add_argument(
        "--download", action="store_true",
        help="Download missing archives from the anonymous mirror",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Verify existing archives without downloading",
    )
    parser.add_argument(
        "--decompress", action="store_true",
        help="Decompress verified archives after download or verification",
    )
    args = parser.parse_args()

    expected = checksum_entries()
    if not expected:
        raise RuntimeError(f"No checksums found in {CHECKSUM_FILE}")
    if not (args.download or args.verify):
        parser.error("choose --download or --verify")

    if args.download:
        download(args.source_dir, expected)
    verify(args.source_dir, expected)
    if args.decompress:
        decompress(args.source_dir, args.output_dir, expected)


if __name__ == "__main__":
    main()
