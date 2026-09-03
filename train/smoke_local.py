#!/usr/bin/env python3
"""CPU smoke test of the full pipeline on this machine before GPU spend.

Runs ref.main() with tiny budgets against the locally regenerated npz
arrays using the same loader shim as the Modal app. Verifies: loading,
window building, one-epoch training, checkpoint save, next-frame eval,
rollout eval, and artifact writing.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "ref"))

import common_sp_baselines as csb  # noqa: E402
import run_sp_baseline_study as rsb  # noqa: E402

ARRAYS = os.path.abspath(os.path.join(HERE, "..", "data_prep", "arrays"))
OUT = "/tmp/gra29_smoke"


def npz_loader(well_path, target_size=128, sequence_subsample_factor=1):
    stem = os.path.basename(well_path).replace(".npz", "")
    role = csb.role_of_stem(stem)
    with np.load(well_path) as d:
        intensity = d["intensity"].astype(np.float32)
        exo = {k: d[k] for k in ("voltage", "current", "time_norm")}
        times = d["frame_times"]
    fi = np.arange(len(intensity), dtype=np.int64)
    return csb.LoadedSequence(
        stem=stem, role=role, intensity=intensity,
        exogenous_all=exo, frame_times=times[fi], raw_frame_indices=fi,
        sequence_subsample_factor=1,
    )


rsb.list_well_files = lambda root: sorted(
    os.path.join(root, f"{s}.npz") for s in sorted(csb.TRAIN_STEMS | csb.TEST_STEMS)
)
rsb.stem_of = lambda p: os.path.basename(p).replace(".npz", "")
rsb.load_sp_sequence = npz_loader

sys.argv = [
    "smoke",
    "--well_data", ARRAYS,
    "--model_family", "unet",
    "--tag", "smoke",
    "--model_out", f"{OUT}/models",
    "--next_frame_out", f"{OUT}/next_frame",
    "--rollout_out", f"{OUT}/rollout",
    "--epochs", "1",
    "--max_train_windows_per_stem", "60",
    "--max_eval_windows", "80",
    "--max_rollout_steps", "40",
]
rsb.main()
print("SMOKE OK")
