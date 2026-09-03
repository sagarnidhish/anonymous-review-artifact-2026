#!/usr/bin/env python3
"""Read every NPZ on the Modal volume inside the cloud and report which are
corrupt there (distinguishes volume-side vs transfer-side corruption)."""

import glob
import os

import modal
IMAGE = modal.Image.debian_slim(python_version="3.11").pip_install("numpy<2")

VOL = modal.Volume.from_name("gra29sp-v1", create_if_missing=False)

app = modal.App("gra29-verify-volume", image=IMAGE)


@app.function(volumes={"/vol": VOL}, timeout=1800)
def verify() -> str:
    import numpy as np

    bad, ok = [], 0
    for p in sorted(glob.glob("/vol/out/*/*/*/*.npz")):
        try:
            with np.load(p) as d:
                for k in d.files:
                    _ = d[k]
            ok += 1
        except Exception as e:
            bad.append(f"{p}: {str(e)[:60]}")
    report = f"OK={ok} BAD={len(bad)}\n" + "\n".join(bad)
    print(report)
    return report


@app.local_entrypoint()
def main():
    print(verify.remote())
