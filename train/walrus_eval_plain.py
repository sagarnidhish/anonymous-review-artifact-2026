#!/usr/bin/env python3
"""Plain (non-Modal) WALRUS evaluation for HPC (CSD3 ampere).

Replicates modal_walrus_eval.py exactly: colab-identical checkpoint
initialization with field-index alignment, delta interpretation of the decoder
output, 4-frame context at 128x128, time-first inputs, OPEN boundary codes.

Artifacts (same layout as the Modal runs, consumable by physics_metrics.py):
  <out_root>/next_frame/preds_<stem>.npz
  <out_root>/rollout_anchored/rollout_<stem>.npz
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate
from the_well.data.datasets import WellMetadata

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from walrus_rollout_state import initialize_rollout_state  # noqa: E402


def load_npz_sequence(path):
    import common_sp_baselines as csb

    stem = os.path.basename(path).replace(".npz", "")
    role = csb.role_of_stem(stem)
    with np.load(path) as d:
        intensity = d["intensity"].astype(np.float32)
        times = d["frame_times"]
    fi = np.arange(len(intensity), dtype=np.int64)
    seq = csb.LoadedSequence(
        stem=stem, role=role, intensity=intensity,
        exogenous_all={"voltage": np.zeros(len(intensity), np.float32),
                       "current": np.zeros(len(intensity), np.float32),
                       "time_norm": np.zeros(len(intensity), np.float32)},
        frame_times=times[fi], raw_frame_indices=fi,
        sequence_subsample_factor=1,
    )
    return seq, intensity


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arrays_dir", required=True)
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--walrus_repo", required=True)
    p.add_argument("--ref_dir", required=True)
    p.add_argument("--out_root", required=True)
    p.add_argument("--max_eval_windows", type=int, default=3000)
    p.add_argument("--eval_stride", type=int, default=6)
    p.add_argument("--rollout_steps", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--stems", default="",
                   help="comma-separated subset; default = all test particles")
    args = p.parse_args()

    sys.path.insert(0, args.walrus_repo)
    import walrus  # noqa: F401
    import walrus.models  # noqa: F401
    import walrus.models.isotropic_model  # noqa: F401
    import walrus.models.encoders  # noqa: F401
    import walrus.models.decoders  # noqa: F401
    import walrus.models.shared_utils  # noqa: F401
    import walrus.models.spatial_blocks  # noqa: F401
    import walrus.models.spatiotemporal_blocks  # noqa: F401
    import walrus.models.temporal_blocks  # noqa: F401

    sys.path.insert(0, args.ref_dir)
    import common_sp_baselines as csb  # noqa: F401

    device = torch.device("cuda")

    ckpt_path = os.path.join(args.ckpt_dir, "walrus.pt")
    cfg_path = os.path.join(args.ckpt_dir, "extended_config.yaml")
    checkpoint = torch.load(ckpt_path, map_location="cpu",
                            weights_only=True)["app"]["model"]
    config = OmegaConf.load(cfg_path)

    def align_checkpoint_with_field_to_index_map(
            checkpoint_state_dict, model_state_dict,
            checkpoint_field_to_index_map, model_field_to_index_map,
            embed_string="embed", embed_weight_name="proj1.weight",
            debed_string="debed", debed_weight_name="proj2.weight",
            debed_bias_name="proj2.bias"):
        checkpoint_num_dims = max(checkpoint_field_to_index_map.values()) + 1
        model_num_dims = max(model_field_to_index_map.values()) + 1
        scale_factor = (checkpoint_num_dims / model_num_dims) ** 0.5
        for param_name in model_state_dict:
            if ((embed_string in param_name and embed_weight_name in param_name)
                    or (debed_string in param_name
                        and debed_weight_name in param_name)
                    or (debed_string in param_name
                        and debed_bias_name in param_name)):
                replacement = model_state_dict[param_name].clone()
                ckpt_param = checkpoint_state_dict[param_name]
                for field in model_field_to_index_map:
                    if field in checkpoint_field_to_index_map:
                        if debed_bias_name in param_name:
                            replacement[model_field_to_index_map[field]] = (
                                ckpt_param[checkpoint_field_to_index_map[field]])
                        elif embed_weight_name in param_name:
                            replacement[:, model_field_to_index_map[field]] = (
                                ckpt_param[:, checkpoint_field_to_index_map[field]]
                                * scale_factor)
                        else:
                            replacement[:, model_field_to_index_map[field]] = (
                                ckpt_param[:, checkpoint_field_to_index_map[field]])
                checkpoint_state_dict[param_name] = replacement.clone()
        return checkpoint_state_dict

    original_field_map = dict(config.data.get("field_index_map_override", {}))
    intensity_idx = max(original_field_map.values()) + 1
    new_field_map = {**original_field_map, "intensity": intensity_idx}
    total_fields = intensity_idx + 1
    print(f"field map: {len(original_field_map)} pretrained fields, "
          f"'intensity' -> {intensity_idx}", flush=True)

    model = instantiate(config.model, n_states=total_fields)
    aligned = align_checkpoint_with_field_to_index_map(
        checkpoint_state_dict=checkpoint,
        model_state_dict=model.state_dict(),
        checkpoint_field_to_index_map=original_field_map,
        model_field_to_index_map=new_field_map,
    )
    model.load_state_dict(aligned)
    model = model.to(device).eval()
    for prm in model.parameters():
        prm.requires_grad_(False)
    print(f"model loaded | hidden_dim={model.hidden_dim}", flush=True)

    field_idx_tensor = torch.tensor([intensity_idx], device=device)
    bcs_base = torch.tensor([[[1, 1], [1, 1], [2, 2]]])
    S = 128

    def make_metadata():
        return WellMetadata(
            dataset_name="battery_lithiation_iSCAT",
            n_spatial_dims=2,
            field_names={0: ["intensity"], 1: [], 2: []},
            spatial_resolution=(S, S, 1),
            scalar_names=[],
            constant_field_names={0: [], 1: [], 2: []},
            constant_scalar_names=[],
            boundary_condition_types=[],
            n_files=[], n_trajectories_per_file=[], n_steps_per_trajectory=[],
        )

    @torch.no_grad()
    def predict_next_batch(context_batch):
        """(B, L, S, S) contexts -> (B, S, S) decoder deltas."""
        x = torch.tensor(context_batch, dtype=torch.float32, device=device)
        x = x.permute(1, 0, 2, 3).unsqueeze(2)  # (L, B, C=1, S, S)
        bcs = bcs_base.expand(x.shape[1], -1, -1).tolist()
        out = model(x, field_idx_tensor, bcs, metadata=make_metadata(),
                    train=False)
        if isinstance(out, (tuple, list)):
            out = out[0]
        out = out.detach().cpu().numpy()
        if out.ndim == 5:
            delta = out[-1, :, 0]
        elif out.ndim == 4:
            delta = out[-1] if out.shape[0] == x.shape[0] else out[0]
        elif out.ndim == 3:
            delta = out
        else:
            raise ValueError(f"unexpected model output shape {out.shape}")
        return np.asarray(delta, dtype=np.float32)

    @torch.no_grad()
    def predict_next(context_int):
        return predict_next_batch(context_int[None])[0]

    if args.stems:
        stems = sorted(args.stems.split(","))
    else:
        stems = sorted(f"GRA29_C20_45deg_particle{i}" for i in (1, 2, 3, 4))
    out_root = args.out_root
    suffix = "_" + "_".join(s[-1] for s in stems) if args.stems else ""

    # ---- next-frame (strided) ----
    nf_dir = os.path.join(out_root, "next_frame")
    os.makedirs(nf_dir, exist_ok=True)
    nf_rows = []
    for stem in stems:
        seq, inten = load_npz_sequence(
            os.path.join(args.arrays_dir, f"{stem}.npz"))
        n = min(args.max_eval_windows, len(inten) - 4)
        starts_all = np.arange(0, n, args.eval_stride)
        preds, naives, targets, steps, ftimes = [], [], [], [], []
        for lo in range(0, len(starts_all), args.batch_size):
            starts = starts_all[lo:lo + args.batch_size]
            batch = np.stack([inten[s:s + 4] for s in starts])
            raws = predict_next_batch(batch)
            last = batch[:, -1]
            preds.append(last + raws)
            naives.append(last.copy())
            targets.append(inten[starts + 4])
            steps.extend((starts + 4).tolist())
            ftimes.extend(seq.frame_times[starts + 4].tolist())
        preds = np.concatenate(preds)
        naives = np.concatenate(naives)
        targets = np.concatenate(targets)
        steps_a = np.asarray(steps, dtype=np.int64)
        ftimes_a = np.asarray(ftimes, dtype=np.float32)
        mae = float(np.abs(preds - targets).mean())
        nmae = float(np.abs(naives - targets).mean())
        nf_rows.append({"stem": stem, "model_family": "walrus",
                        "tag": "walrus_native", "model_mae": mae,
                        "naive_mae": nmae,
                        "mae_ratio": mae / max(nmae, 1e-12)})
        stem_dir = os.path.join(nf_dir, stem)
        os.makedirs(stem_dir, exist_ok=True)
        np.savez_compressed(os.path.join(stem_dir, f"preds_{stem}.npz"),
                            pred=preds, naive=naives, targets=targets,
                            target_steps=steps_a, frame_times=ftimes_a)
        print(f"[nf done] {stem}: ratio {nf_rows[-1]['mae_ratio']:.4f}",
              flush=True)
    with open(os.path.join(nf_dir, f"comparison_summary{suffix}.json"),
              "w") as f:
        json.dump(nf_rows, f, indent=2)

    # ---- anchored rollout (sequential) ----
    ro_dir = os.path.join(out_root, "rollout_anchored")
    os.makedirs(ro_dir, exist_ok=True)
    ro_rows = []
    for stem in stems:
        seq, inten = load_npz_sequence(
            os.path.join(args.arrays_dir, f"{stem}.npz"))
        with np.load(os.path.join(args.arrays_dir, f"{stem}.npz")) as d:
            current = d["current"]
        signs = np.sign(current.astype(np.float64))
        changes = [i for i in range(1, len(signs))
                   if signs[i] != 0 and signs[i - 1] != 0
                   and signs[i] != signs[i - 1]]
        onset = changes[0] if changes else 0
        start = max(0, onset - 4)
        ctx, persistence_frame = initialize_rollout_state(
            inten, start=start, context_len=4
        )
        preds, naives, targets, steps, ftimes = [], [], [], [], []
        horizon = min(args.rollout_steps, len(inten) - start - 4)
        for step in range(horizon):
            tgt_idx = start + 4 + step
            raw = predict_next(np.stack(ctx))
            pred = ctx[-1] + raw
            preds.append(pred)
            naives.append(persistence_frame.copy())
            targets.append(inten[tgt_idx])
            steps.append(tgt_idx)
            ftimes.append(seq.frame_times[tgt_idx])
            ctx = ctx[1:] + [pred]
        preds = np.stack(preds)
        naives = np.stack(naives)
        targets = np.stack(targets)
        steps_a = np.asarray(steps, dtype=np.int64)
        ftimes_a = np.asarray(ftimes, dtype=np.float32)
        mae = float(np.abs(preds - targets).mean())
        nmae = float(np.abs(naives - targets).mean())
        ro_rows.append({"stem": stem, "model_family": "walrus",
                        "tag": "walrus_native", "anchor_frame": start,
                        "model_mae": mae, "naive_mae": nmae,
                        "mae_ratio": mae / max(nmae, 1e-12)})
        stem_dir = os.path.join(ro_dir, stem)
        os.makedirs(stem_dir, exist_ok=True)
        np.savez_compressed(os.path.join(stem_dir, f"rollout_{stem}.npz"),
                            pred=preds, naive=naives, targets=targets,
                            target_steps=steps_a, frame_times=ftimes_a)
        print(f"[rollout done] {stem}: ratio {ro_rows[-1]['mae_ratio']:.4f}",
              flush=True)
    with open(os.path.join(ro_dir, f"comparison_summary{suffix}.json"),
              "w") as f:
        json.dump(ro_rows, f, indent=2)

    print("WALRUS EVAL DONE")


if __name__ == "__main__":
    main()
