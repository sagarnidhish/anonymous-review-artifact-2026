# Source data

This directory contains eight source HDF5 movies grouped by recording
temperature (25 °C or 45 °C) and physical particle (1–4). The files are stored
as losslessly compressed `.h5.zst` archives.

Each HDF5 file contains:

- `movie`: registered image frames stored as unsigned 16-bit integers, with
  dimensions `(frame, height, width)`;
- `camera_timing`: camera timestamps in seconds; and
- `potentiostat_value`: rows containing time, voltage, and current.

The cropped movies were image-registered before this release. The files contain
no people, clinical records, or personal data.

Use `python3 scripts/prepare_source_data.py --download --decompress` from the
repository root to retrieve the Git LFS content, verify the compressed-file
hashes listed in `SHA256SUMS`, and create ordinary HDF5 files under
`data/source_uncompressed/`.
