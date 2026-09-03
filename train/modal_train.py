#!/usr/bin/env python3
"""Modal entry point for GRA29 SP emulator experiments.

Runs the frozen reference implementation (train/ref/*.py, unmodified) against
npz arrays on a Modal volume, and adds a recursive-horizon fine-tuning mode.

Payload JSON:
  {"mode": "train",    "cfg": {...}}                 vanilla training + eval
  {"mode": "finetune", "cfg": {...}, "src_tag": ..., "src_model": "unet"}
                                                     recursive h-step FT + eval
"""

import json
import os
import random
import sys

import modal

IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "numpy<2", "h5py")
    .add_local_dir(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "ref"),
        remote_path="/root/ref",
    )
    .add_local_file(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "particle_splits.py"
        ),
        remote_path="/root/particle_splits.py",
    )
)

VOL = modal.Volume.from_name("gra29sp-v1", create_if_missing=True)

app = modal.App("gra29-sp-emulator", image=IMAGE)

CFG_DEFAULTS = dict(
    model_family="unet",
    tag="unet_image_only_delta",
    use_voltage=False,
    use_current=False,
    use_time_norm=False,
    predict_delta=True,
    target_size=128,
    context_len=4,
    window_stride=1,
    sequence_subsample_factor=1,
    epochs=60,
    batch_size=8,
    lr=1e-3,
    patience=10,
    val_fraction=0.1,
    base_channels=32,
    hidden_layers=2,
    max_train_windows_per_stem=3000,
    max_eval_windows=3000,
    max_rollout_steps=512,
    reversal_radius=5,
    grad_clip_norm=1.0,
    seed=1337,
    split="frozen",
    heldout_particle=None,
    ft_epochs=12,
    ft_lr=1e-4,
    ft_horizon=8,
    ft_batch_windows=8,
)


def _load_npz_sequence(well_path, target_size=128, sequence_subsample_factor=1,
                       role_override=None):
    """Drop-in replacement for common_sp_baselines.load_sp_sequence.

    Local preprocessing used the identical numeric path (front-trim timing,
    global per-file z-score, bilinear resize W,H -> 128, EC normalization and
    interpolation onto camera frame times)."""
    import numpy as np

    import common_sp_baselines as csb

    stem = os.path.basename(well_path).replace(".npz", "")
    role = role_override or csb.role_of_stem(stem)
    if role == "ignore":
        raise ValueError(f"unexpected stem outside configured split: {stem}")
    with np.load(well_path) as d:
        intensity_full = d["intensity"].astype(np.float32)
        voltage = d["voltage"]
        current = d["current"]
        time_norm = d["time_norm"]
        frame_times = d["frame_times"]
    subsample = csb.VIDEO_SUBSAMPLE.get(stem, 1) * sequence_subsample_factor
    fi = np.arange(0, len(intensity_full), subsample, dtype=np.int64)
    return csb.LoadedSequence(
        stem=stem,
        role=role,
        intensity=intensity_full[fi],
        exogenous_all={
            "voltage": voltage[fi],
            "current": current[fi],
            "time_norm": time_norm[fi],
        },
        frame_times=frame_times[fi],
        raw_frame_indices=fi,
        sequence_subsample_factor=sequence_subsample_factor,
    )


@app.function(gpu="A100-40GB", volumes={"/vol": VOL}, timeout=60 * 60 * 8,
              retries=3)
def run_experiment(payload: str) -> str:
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    sys.path.insert(0, "/root/ref")
    sys.path.insert(0, "/root")
    import common_sp_baselines as csb
    import run_sp_baseline_study as rsb
    from models import build_model
    from particle_splits import build_identity_holdout_fold

    req = json.loads(payload)
    cfg = dict(CFG_DEFAULTS)
    cfg.update(req.get("cfg", {}))
    mode = req.get("mode", "train")
    out_root = f"/vol/out/{cfg['tag']}"
    active_fields = csb.build_active_fields(
        cfg["use_voltage"], cfg["use_current"], cfg["use_time_norm"]
    )
    device = torch.device("cuda")
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])

    split = cfg.get("split", "frozen")
    identity_fold = None
    if split == "frozen":
        train_stems = set(csb.TRAIN_STEMS)
        test_stems = set(csb.TEST_STEMS)
    elif split == "lopo":
        # Particle-only transfer control: particles 1-3 of BOTH temperatures
        # train; particle 4 of BOTH temperatures tests.  Decomposes the
        # frozen 25->45 split into particle vs temperature shift.
        train_stems = {f"GRA29_C20_{t}_particle{i}"
                       for t in ("25deg", "45deg") for i in (1, 2, 3)}
        test_stems = {f"GRA29_C20_{t}_particle4"
                      for t in ("25deg", "45deg")}
    elif split == "identity_holdout":
        if cfg.get("heldout_particle") is None:
            raise ValueError("heldout_particle is required for identity_holdout")
        identity_fold = build_identity_holdout_fold(
            int(cfg["heldout_particle"])
        )
        train_stems = set(identity_fold.train_stems)
        test_stems = set(identity_fold.all_test_stems)
    else:
        raise ValueError(f"unknown split {split}")
    stems = sorted(train_stems | test_stems)
    all_sequences = [
        _load_npz_sequence(f"/vol/arrays/{s}.npz",
                           role_override="train" if s in train_stems else "test")
        for s in stems
    ]
    train_sequences = [s for s in all_sequences if s.stem in train_stems]
    eval_sequences = sorted(
        (
            [sequence for sequence in all_sequences if sequence.role == "test"]
            if identity_fold is not None
            else all_sequences
        ),
        key=lambda sequence: (sequence.role, sequence.stem),
    )

    def make_loaders():
        tr_idx, va_idx = [], []
        for si, seq in enumerate(train_sequences):
            starts = csb.build_window_starts(
                len(seq.intensity), cfg["context_len"], cfg["window_stride"],
                cfg["max_train_windows_per_stem"],
            )
            tr, va = csb.split_train_val_starts(starts, cfg["val_fraction"])
            tr_idx.extend((si, s) for s in tr)
            va_idx.extend((si, s) for s in va)
        mk = lambda idx: csb.BaselineWindowDataset(
            train_sequences, idx, active_fields=active_fields,
            context_len=cfg["context_len"], window_stride=cfg["window_stride"],
            predict_delta=cfg["predict_delta"],
        )
        return (
            DataLoader(mk(tr_idx), batch_size=cfg["batch_size"], shuffle=True,
                       num_workers=0, pin_memory=True),
            DataLoader(mk(va_idx), batch_size=cfg["batch_size"], shuffle=False,
                       num_workers=0, pin_memory=True),
        )

    history = {"train_mae": [], "val_mae": []}
    best = {}
    if mode == "train":
        train_loader, val_loader = make_loaders()
        model = build_model(
            model_family=cfg["model_family"], in_fields=len(active_fields),
            context_len=cfg["context_len"], base_channels=cfg["base_channels"],
            hidden_layers=cfg["hidden_layers"],
        ).to(device)
        model, history, best = rsb.train_model(
            model=model, train_loader=train_loader, val_loader=val_loader,
            device=device, epochs=cfg["epochs"], lr=cfg["lr"],
            patience=cfg["patience"], predict_delta=cfg["predict_delta"],
            grad_clip_norm=cfg["grad_clip_norm"],
        )
    elif mode == "finetune":
        src_model_name = req.get("src_model", cfg["model_family"])
        ckpt_path = os.path.join("/vol/out", req["src_tag"], "models",
                                 f"{src_model_name}_best.pt")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = build_model(
            model_family=ckpt["model_family"],
            in_fields=len(ckpt["active_fields"]),
            context_len=ckpt["context_len"],
            base_channels=ckpt["base_channels"],
            hidden_layers=ckpt["hidden_layers"],
        ).to(device)
        model.load_state_dict(ckpt["state_dict"])
        predict_delta = ckpt["prediction_form"] == "delta_from_last_frame"
        history, best = _recursive_finetune(
            model=model, train_sequences=train_sequences,
            active_fields=ckpt["active_fields"], cfg=cfg, device=device,
            predict_delta=predict_delta,
        )
    else:
        raise ValueError(f"unknown mode {mode}")

    os.makedirs(os.path.join(out_root, "models"), exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "model_family": cfg["model_family"],
         "active_fields": active_fields, "context_len": cfg["context_len"],
         "base_channels": cfg["base_channels"], "hidden_layers": cfg["hidden_layers"],
         "target_size": cfg["target_size"], "seed": cfg["seed"],
         "mode": mode, "split": split,
         "heldout_particle": cfg.get("heldout_particle"),
         "train_stems": sorted(train_stems),
         "test_stems": sorted(test_stems),
         "prediction_form": "delta_from_last_frame" if cfg["predict_delta"]
         else "direct_next_frame"},
        os.path.join(out_root, "models", f"{cfg['model_family']}_best.pt"),
    )
    with open(os.path.join(out_root, "models", "training_history.json"), "w") as f:
        json.dump({"cfg": cfg, "mode": mode, "best": best, "history": history},
                  f, indent=2, default=float)

    _evaluate_and_save(
        rsb, csb, model, eval_sequences, active_fields, cfg,
        out_root, device, identity_fold=identity_fold,
    )
    VOL.commit()
    return f"done {mode} {cfg['tag']}"


def _evaluate_and_save(rsb, csb, model, eval_sequences, active_fields, cfg,
                       out_root, device, identity_fold=None):
    nf_dir = os.path.join(out_root, "next_frame")
    ro_dir = os.path.join(out_root, "rollout")
    os.makedirs(nf_dir, exist_ok=True)
    os.makedirs(ro_dir, exist_ok=True)
    pred_form = ("delta_from_last_frame" if cfg["predict_delta"]
                 else "direct_next_frame")

    def base_row(seq):
        row = {"stem": seq.stem, "role": seq.role,
                "model_family": cfg["model_family"], "tag": cfg["tag"],
                "active_fields": list(active_fields),
                "context_len": cfg["context_len"],
                "prediction_form": pred_form,
                "split": cfg.get("split", "frozen"),
                "seed": int(cfg["seed"])}
        if cfg.get("heldout_particle") is not None:
            row["heldout_particle"] = int(cfg["heldout_particle"])
        if identity_fold is not None and seq.role == "test":
            row["evaluation_group"] = identity_fold.evaluation_group_for_stem(
                seq.stem
            )
        return row

    next_rows, rollout_rows = [], []
    for seq in eval_sequences:
        pred, naive, targets, steps, ftimes = rsb.run_next_frame_eval_for_sequence(
            model=model, sequence=seq, active_fields=list(active_fields),
            context_len=cfg["context_len"], window_stride=cfg["window_stride"],
            max_windows=cfg["max_eval_windows"], batch_size=cfg["batch_size"],
            device=device, predict_delta=cfg["predict_delta"],
        )
        metrics = rsb.summarize_prediction_result(
            sequence=seq, pred=pred, naive=naive, targets=targets,
            target_steps=steps, reversal_radius=cfg["reversal_radius"],
        )
        row = base_row(seq)
        row.update(metrics)
        next_rows.append(row)
        rsb.save_per_stem_next_frame(nf_dir, row, pred, naive, targets, steps, ftimes)

        rpred, rnaive, rtargets, rsteps, rftimes = rsb.run_rollout_eval_for_sequence(
            model=model, sequence=seq, active_fields=list(active_fields),
            context_len=cfg["context_len"], device=device,
            predict_delta=cfg["predict_delta"],
            max_rollout_steps=cfg["max_rollout_steps"],
        )
        rmetrics = rsb.summarize_prediction_result(
            sequence=seq, pred=rpred, naive=rnaive, targets=rtargets,
            target_steps=rsteps, reversal_radius=cfg["reversal_radius"],
            include_horizon=True,
        )
        rrow = base_row(seq)
        rrow.update(rmetrics)
        rollout_rows.append(rrow)
        rsb.save_per_stem_rollout(ro_dir, rrow, rpred, rnaive, rtargets, rsteps,
                                  rftimes)

    rsb.dump_summary(os.path.join(nf_dir, "next_frame_results.json"), next_rows)
    test_next_rows = [row for row in next_rows if row.get("role") == "test"]
    test_rollout_rows = [
        row for row in rollout_rows if row.get("role") == "test"
    ]
    rsb.dump_summary(os.path.join(nf_dir, "comparison_summary.json"),
                     [rsb.summarize_rows(test_next_rows)])
    rsb.dump_summary(os.path.join(ro_dir, "rollout_results.json"), rollout_rows)
    rsb.dump_summary(os.path.join(ro_dir, "comparison_summary.json"),
                     [rsb.summarize_rows(test_rollout_rows)])
    if identity_fold is not None:
        for rows, directory in (
            (test_next_rows, nf_dir),
            (test_rollout_rows, ro_dir),
        ):
            grouped = []
            for group_name in (
                "same_temperature_unseen_particle",
                "cross_temperature_unseen_particle",
            ):
                group_rows = [
                    row for row in rows
                    if row.get("evaluation_group") == group_name
                ]
                summary = rsb.summarize_rows(group_rows)
                summary["evaluation_group"] = group_name
                grouped.append(summary)
            rsb.dump_summary(
                os.path.join(directory, "comparison_summary_by_group.json"),
                grouped,
            )


def _recursive_finetune(model, train_sequences, active_fields, cfg, device,
                        predict_delta):
    """Unroll ft_horizon steps through the model, feeding predicted intensity
    back into the context; exogenous channels stay teacher-forced at true
    indices (identical to frozen rollout evaluation semantics); MAE loss on
    accumulated frames vs ground truth averaged over steps, one backward per
    batch through the full unroll."""
    import numpy as np
    import torch
    import torch.nn as nn

    ctx = cfg["context_len"]
    h = cfg["ft_horizon"]

    pool = []
    for seq in train_sequences:
        limit = len(seq.intensity) - (ctx + h) + 1
        starts = list(range(limit))[: cfg["max_train_windows_per_stem"]]
        pool.extend((seq, s) for s in starts)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["ft_lr"], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["ft_epochs"])
    rng = np.random.default_rng(cfg["seed"] + 101)
    bs = cfg["ft_batch_windows"]
    fields = list(active_fields)
    history = {"train_mae": []}
    best = {"epoch": 0, "train_mae": float("inf")}

    for epoch in range(1, cfg["ft_epochs"] + 1):
        order = rng.permutation(len(pool))
        totals, n_units = 0.0, 0
        for lo in range(0, len(order), bs):
            sel = [pool[i] for i in order[lo : lo + bs]]
            B = len(sel)
            ctx_int = torch.from_numpy(np.stack([
                np.stack([seq.intensity[s + k] for k in range(ctx)])
                for seq, s in sel
            ])).float().to(device)                              # (B, ctx, S, S)
            starts_idx = [s for _, s in sel]
            seq_refs = [seq for seq, _ in sel]

            opt.zero_grad()
            step_losses = []
            for step in range(h):
                tgt_off = ctx + step                            # target = start+ctx+step
                x_seq = torch.empty(
                    (B, ctx, len(fields), cfg["target_size"], cfg["target_size"]),
                    dtype=torch.float32, device=device)
                for b in range(B):
                    seq = seq_refs[b]
                    base_t = starts_idx[b] + tgt_off
                    for k in range(ctx):
                        abs_i = base_t - ctx + k
                        x_seq[b, k, 0] = ctx_int[b, k]
                        for ci, fname in enumerate(fields[1:], start=1):
                            x_seq[b, k, ci] = float(seq.exogenous_all[fname][abs_i])
                raw = model(x_seq)                              # (B, 1, S, S)
                last = ctx_int[:, -1:].clone()
                pred_next = last + raw if predict_delta else raw
                tgt = torch.from_numpy(np.stack([
                    seq_ref.intensity[starts_idx[b] + tgt_off]
                    for b, seq_ref in enumerate(seq_refs)
                ])).float().unsqueeze(1).to(device)
                step_losses.append(torch.mean(torch.abs(pred_next - tgt)))
                totals += float(step_losses[-1].detach()) * B
                ctx_int = torch.cat([ctx_int[:, 1:], pred_next], dim=1)
            n_units += B
            torch.stack(step_losses).mean().backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip_norm"])
            opt.step()

        sched.step()
        epoch_mae = totals / max(n_units * h, 1)
        history["train_mae"].append(epoch_mae)
        if epoch_mae < best["train_mae"]:
            best = {"epoch": epoch, "train_mae": epoch_mae}
        print(f"ft epoch {epoch}: mae={epoch_mae:.6f}", flush=True)

    return history, best


@app.local_entrypoint()
def main(payload_arg: str = ""):
    payload = payload_arg or os.environ.get("PAYLOAD", "")
    if not payload:
        raise SystemExit("pass payload JSON via --payload-arg")
    result = run_experiment.remote(payload)
    print(result)
