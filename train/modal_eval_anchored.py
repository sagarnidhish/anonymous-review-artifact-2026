#!/usr/bin/env python3
"""Anchored-rollout evaluation for existing GRA29 checkpoints.

Rollouts launched from t=0 spend the entire horizon inside the initial rest
phase, where the true optical state is flat and physics-facing metrics are
undefined.  This app instead anchors rollouts at the first current sign
change (rest -> charge onset): the model receives the `context_len` frames
before onset, then unrolls `max_rollout_steps` into the charge leg, where the
optical state genuinely evolves.

Uses the frozen reference evaluation functions unmodified; only the sequence
objects are sliced.  Artifacts are written to
/vol/out/<out_tag>/rollout_anchored/ in the same layout as rollout/.

Payload JSON: {"src_tag": "...", "src_model": "unet", "out_tag": "..."}
"""

import json
import os
import sys

import modal

IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "numpy<2", "h5py")
    .add_local_dir(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "ref"),
        remote_path="/root/ref",
    )
)

VOL = modal.Volume.from_name("gra29sp-v1", create_if_missing=True)

app = modal.App("gra29-eval-anchored", image=IMAGE)


def _load_npz_sequence(well_path, target_size=128, sequence_subsample_factor=1):
    """Same numeric path as the training app's loader (see modal_train.py)."""
    import numpy as np

    import common_sp_baselines as csb

    stem = os.path.basename(well_path).replace(".npz", "")
    role = csb.role_of_stem(stem)
    if role == "ignore":
        raise ValueError(f"unexpected stem outside configured split: {stem}")
    with np.load(well_path) as d:
        intensity_full = d["intensity"].astype(np.float32)
        voltage = d["voltage"]
        current = d["current"]
        time_norm = d["time_norm"]
        frame_times = d["frame_times"]
    fi = np.arange(0, len(intensity_full), 1, dtype=np.int64)
    return csb.LoadedSequence(
        stem=stem, role=role, intensity=intensity_full[fi],
        exogenous_all={"voltage": voltage[fi], "current": current[fi],
                       "time_norm": time_norm[fi]},
        frame_times=frame_times[fi], raw_frame_indices=fi,
        sequence_subsample_factor=1,
    )


@app.function(gpu="A100-40GB", volumes={"/vol": VOL}, timeout=60 * 60 * 3)
def run_eval(payload: str) -> str:
    import numpy as np
    import torch

    sys.path.insert(0, "/root/ref")
    import common_sp_baselines as csb
    import run_sp_baseline_study as rsb
    from models import build_model

    req = json.loads(payload)
    src_tag = req["src_tag"]
    src_model = req.get("src_model", "unet")
    out_tag = req.get("out_tag", f"{src_tag}_anchored")

    device = torch.device("cuda")
    ckpt_path = os.path.join("/vol/out", src_tag, "models", f"{src_model}_best.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(
        model_family=ckpt["model_family"],
        in_fields=len(ckpt["active_fields"]),
        context_len=ckpt["context_len"],
        base_channels=ckpt["base_channels"],
        hidden_layers=ckpt["hidden_layers"],
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    active_fields = list(ckpt["active_fields"])
    ctx = ckpt["context_len"]
    predict_delta = ckpt["prediction_form"] == "delta_from_last_frame"

    stems = sorted(csb.TRAIN_STEMS | csb.TEST_STEMS)
    out_dir = os.path.join("/vol/out", out_tag, "rollout_anchored")
    os.makedirs(out_dir, exist_ok=True)

    rollout_rows = []
    for stem in stems:
        seq = _load_npz_sequence(f"/vol/arrays/{stem}.npz")
        current = seq.exogenous_all["current"]
        changes = csb.current_sign_change_indices(current)
        onset = int(changes[0]) if len(changes) else 0
        start = max(0, onset - ctx)

        sliced = csb.LoadedSequence(
            stem=seq.stem,
            role=seq.role,
            intensity=seq.intensity[start:],
            exogenous_all={k: v[start:] for k, v in seq.exogenous_all.items()},
            frame_times=seq.frame_times[start:],
            raw_frame_indices=seq.raw_frame_indices[start:],
            sequence_subsample_factor=1,
        )
        pred, naive, targets, steps, ftimes = rsb.run_rollout_eval_for_sequence(
            model=model, sequence=sliced, active_fields=active_fields,
            context_len=ctx, device=device, predict_delta=predict_delta,
            max_rollout_steps=512,
        )
        metrics = rsb.summarize_prediction_result(
            sequence=sliced, pred=pred, naive=naive, targets=targets,
            target_steps=steps, reversal_radius=5, include_horizon=True,
        )
        row = {
            "stem": seq.stem, "role": seq.role,
            "model_family": ckpt["model_family"], "tag": out_tag,
            "active_fields": active_fields, "context_len": ctx,
            "anchor_frame": start, "onset_frame": onset,
            "prediction_form": ckpt["prediction_form"],
        }
        row.update(metrics)
        rollout_rows.append(row)
        rsb.save_per_stem_rollout(out_dir, row, pred, naive, targets, steps, ftimes)
        print(f"[done] {stem} anchor@{start}", flush=True)

    rsb.dump_summary(os.path.join(out_dir, "rollout_results.json"), rollout_rows)
    rsb.dump_summary(os.path.join(out_dir, "comparison_summary.json"),
                     [rsb.summarize_rows(rollout_rows)])
    with open(os.path.join(out_dir, "study_manifest.json"), "w") as f:
        json.dump({"mode": "rollout_anchored", "src_tag": src_tag,
                   "src_model": src_model,
                   "anchor_rule": "first_current_sign_change_minus_ctx",
                   "max_rollout_steps": 512}, f, indent=2)
    VOL.commit()
    return f"done anchored eval {out_tag}"


@app.local_entrypoint()
def main(payload_arg: str = ""):
    payload = payload_arg or os.environ.get("PAYLOAD", "")
    if not payload:
        raise SystemExit("pass payload JSON via --payload-arg")
    print(run_eval.remote(payload))
