#!/usr/bin/env python3
"""Regenerate single-particle analysis arrays from immutable source HDF5.

Replicates the exact numeric path of the frozen pipeline:
  1. convert_to_well_sp.py: front-trim timing/movie to equal length
     ("fixt"), global per-file z-score of the uint16 movie,
     spatial transpose to (T, W, H) well order.
  2. common_sp_baselines.load_sp_sequence: per-frame bilinear resize
     W x H -> (target, target) with align_corners=False, EC channel
     normalization (voltage range01, current maxabs over the full
     potentiostat record), linear interpolation onto camera frame times,
     time_norm range01.

Output: one .npz per particle with
  intensity  (T, S, S) float16   z-scored, resized
  voltage    (T,)      float32   range01, interpolated to frame times
  current    (T,)      float32   maxabs, interpolated to frame times
  time_norm  (T,)      float32   range01 of frame times
  frame_times(T,)      float32   seconds (after fixt)
plus norm_mu / norm_std scalars for reproducibility.
"""

import argparse
import json
import os

import h5py
import numpy as np
import torch
import torch.nn.functional as F

DEFAULT_OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "arrays")

PARTICLES = [
    "GRA29_C20_25deg_particle1",
    "GRA29_C20_25deg_particle2",
    "GRA29_C20_25deg_particle3",
    "GRA29_C20_25deg_particle4",
    "GRA29_C20_45deg_particle1",
    "GRA29_C20_45deg_particle2",
    "GRA29_C20_45deg_particle3",
    "GRA29_C20_45deg_particle4",
]


def fixt(tt, mm):
    """Trim the longer array from the front until lengths match (frozen logic)."""
    loops = 0
    while tt.shape[0] != mm.shape[0]:
        if tt.shape[0] > mm.shape[0]:
            tt = tt[1:]
        else:
            mm = mm[1:]
        loops += 1
        if loops > 100:
            raise RuntimeError("fixt: too many trim loops")
    return tt, mm


def normalize_trace(values, mode):
    values = values.astype(np.float64)
    if mode == "range01":
        lo, hi = values.min(), values.max()
        scale = max(hi - lo, 1e-8)
        return ((values - lo) / scale).astype(np.float32)
    if mode == "maxabs":
        scale = max(np.abs(values).max(), 1e-8)
        return (values / scale).astype(np.float32)
    raise ValueError(mode)


def resize_frames(movie_thw, target):
    """Per-frame bilinear resize, identical to _resize_frame in the frozen code."""
    out = np.empty((movie_thw.shape[0], target, target), dtype=np.float32)
    for i in range(movie_thw.shape[0]):
        t = torch.tensor(movie_thw[i][np.newaxis, np.newaxis], dtype=torch.float32)
        out[i] = F.interpolate(t, size=(target, target), mode="bilinear",
                               align_corners=False)[0, 0].numpy()
    return out


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True,
                        help="Directory containing the eight source HDF5 files")
    parser.add_argument("--output-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--target-size", type=int, default=128)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.set_num_threads(max(1, os.cpu_count() - 2))
    summary = {}
    for stem in PARTICLES:
        src = os.path.join(args.source_dir, f"{stem}.h5")
        with h5py.File(src, "r") as f:
            mov0 = f["movie"][:]                      # (T, H, W) uint16
            camtime0 = f["camera_timing"][:]
            if camtime0.ndim > 1:
                camtime0 = camtime0[:, 0]
            ec = None
            if "potentiostat_value" in f:
                ec = f["potentiostat_value"][:].astype(np.float64)  # (3, N)
        camtime0, mov0 = fixt(camtime0.astype(np.float64), mov0)

        mov_f = mov0.astype(np.float32)
        mu = float(mov_f.mean())
        sigma = float(mov_f.std()) + 1e-8
        mov_z = (mov_f - mu) / sigma

        # Well order is (T, W, H); loader transposes back to (H, W) before resize.
        mov_well = np.swapaxes(mov_z, 1, 2)
        intensity = resize_frames(mov_well, args.target_size).astype(np.float16)
        del mov_f, mov_z, mov_well

        sel_times = camtime0.astype(np.float32)
        if ec is not None:
            if ec.shape[0] == 3 and ec.shape[1] == 3:
                ec = ec.T
            ec_time = ec[0]
            voltage = normalize_trace(ec[1], "range01")
            current = normalize_trace(ec[2], "maxabs")
            voltage_i = np.interp(sel_times, ec_time, voltage).astype(np.float32)
            current_i = np.interp(sel_times, ec_time, current).astype(np.float32)
        else:
            raise RuntimeError(f"{stem}: no potentiostat record")

        time_norm = normalize_trace(camtime0, "range01").astype(np.float32)
        out_path = os.path.join(args.output_dir, f"{stem}.npz")
        np.savez(
            out_path,
            intensity=intensity,
            voltage=voltage_i,
            current=current_i,
            time_norm=time_norm,
            frame_times=sel_times,
            norm_mu=np.float32(mu),
            norm_std=np.float32(sigma),
        )
        summary[stem] = {
            "T": int(intensity.shape[0]),
            "S": int(intensity.shape[1]),
            "mu": mu,
            "sigma": sigma,
            "dt_mean_s": float(np.diff(camtime0).mean()),
            "voltage_range": [float(voltage_i.min()), float(voltage_i.max())],
            "current_range": [float(current_i.min()), float(current_i.max())],
            "bytes_on_disk": int(os.path.getsize(out_path)),
        }
        print(stem, summary[stem], flush=True)

    with open(os.path.join(args.output_dir, "prep_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print("done")


if __name__ == "__main__":
    main()
