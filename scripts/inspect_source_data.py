#!/usr/bin/env python3
"""Check that the eight decompressed source HDF5 files are readable."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = REPOSITORY_ROOT / "data" / "source_uncompressed"
CHECKSUM_FILE = REPOSITORY_ROOT / "data" / "SHA256SUMS"
REQUIRED_DATASETS = ("movie", "camera_timing", "potentiostat_value")


def expected_hdf5_names() -> list[str]:
    names = []
    for line in CHECKSUM_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        _, relative_path = line.split(maxsplit=1)
        archive_name = Path(relative_path).name
        if not archive_name.endswith(".h5.zst"):
            raise ValueError(f"Unexpected archive name: {archive_name}")
        names.append(archive_name[:-4])
    return names


def inspect_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing decompressed source file: {path}")
    with h5py.File(path, "r") as handle:
        missing = [name for name in REQUIRED_DATASETS if name not in handle]
        if missing:
            raise ValueError(f"{path.name}: missing datasets {missing}")
        movie = handle["movie"]
        timing = handle["camera_timing"]
        potentiostat = handle["potentiostat_value"]
        if movie.ndim != 3:
            raise ValueError(f"{path.name}: movie must have three dimensions")
        print(
            f"{path.name}: movie={movie.shape} {movie.dtype}; "
            f"camera_timing={timing.shape}; "
            f"potentiostat_value={potentiostat.shape}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="Directory containing the eight decompressed HDF5 files",
    )
    args = parser.parse_args()

    names = expected_hdf5_names()
    if len(names) != 8:
        raise ValueError(f"Expected eight source files, found {len(names)}")
    for name in names:
        inspect_file(args.source_dir / name)
    print("Validated 8 readable source HDF5 files.")


if __name__ == "__main__":
    main()
