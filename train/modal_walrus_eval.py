#!/usr/bin/env python3
"""WALRUS foundation-model evaluation on GRA29 test particles.

Initialization replicates the project's working linprobe colab exactly:
checkpoint field-index alignment for a new 'intensity' field, delta
interpretation of the decoder output (setup doc: "reading the decoder output
as the full frame is wrong for this checkpoint"), 4-frame context at 128x128,
time-first inputs, OPEN boundary codes.

Outputs NPZ artifacts compatible with analysis/physics_metrics.py:
  /vol/out/walrus_native/next_frame/preds_<stem>.npz
  /vol/out/walrus_native/rollout_anchored/rollout_<stem>.npz
"""

import json
import os
import sys

import modal

IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "wget")
    .pip_install(
        "torch==2.5.1", "numpy<2", "h5py", "omegaconf", "hydra-core",
        "einops", "timm", "pillow",
    )
    .pip_install("git+https://github.com/PolymathicAI/the_well.git")
    .run_commands(
        "git clone --depth 1 https://github.com/PolymathicAI/walrus.git /opt/walrus"
    )
    .add_local_dir(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "ref"),
        remote_path="/root/ref",
    )
    .add_local_file(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "walrus_rollout_state.py",
        ),
        remote_path="/root/walrus_rollout_state.py",
    )
)

VOL = modal.Volume.from_name("gra29sp-v1", create_if_missing=True)

app = modal.App("gra29-walrus-eval", image=IMAGE)

CKPT_URL = "https://huggingface.co/polymathic-ai/walrus/resolve/main"
CONTEXT_LEN = 4
MAX_EVAL_WINDOWS = 3000
ROLLOUT_STEPS = 512


def _load_npz_sequence(well_path):
    import numpy as np

    import common_sp_baselines as csb

    stem = os.path.basename(well_path).replace(".npz", "")
    role = csb.role_of_stem(stem)
    with np.load(well_path) as d:
        intensity = d["intensity"].astype(np.float32)
        exo = {k: d[k] for k in ("voltage", "current", "time_norm")}
        times = d["frame_times"]
    fi = np.arange(len(intensity), dtype=np.int64)
    return csb.LoadedSequence(
        stem=stem, role=role, intensity=intensity, exogenous_all=exo,
        frame_times=times[fi], raw_frame_indices=fi, sequence_subsample_factor=1,
    ), intensity


@app.function(gpu="A100-40GB", volumes={"/vol": VOL}, timeout=60 * 60 * 3)
def run_eval(payload: str) -> str:
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from hydra.utils import instantiate
    from the_well.data.datasets import WellMetadata

    sys.path.insert(0, "/opt/walrus")
    import walrus  # noqa: F401
    import walrus.models  # noqa: F401
    import walrus.models.isotropic_model  # noqa: F401
    import walrus.models.encoders  # noqa: F401
    import walrus.models.decoders  # noqa: F401
    import walrus.models.shared_utils  # noqa: F401
    import walrus.models.spatial_blocks  # noqa: F401
    import walrus.models.spatiotemporal_blocks  # noqa: F401
    import walrus.models.temporal_blocks  # noqa: F401

    sys.path.insert(0, "/root/ref")
    import common_sp_baselines as csb  # noqa: F401
    sys.path.insert(0, "/root")
    from walrus_rollout_state import initialize_rollout_state

    req = json.loads(payload) if payload else {}
    evals = req.get("evals", ["next_frame", "rollout_anchored"])
    stems = req.get("stems", sorted(csb.TEST_STEMS))

    device = torch.device("cuda")

    ckpt_dir = "/vol/walrus_ckpt"
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "walrus.pt")
    cfg_path = os.path.join(ckpt_dir, "extended_config.yaml")
    if not os.path.exists(ckpt_path):
        import urllib.request
        for url, dst in [(f"{CKPT_URL}/walrus.pt", ckpt_path),
                         (f"{CKPT_URL}/extended_config.yaml", cfg_path)]:
            print(f"downloading {url}", flush=True)
            urllib.request.urlretrieve(url, dst)
        VOL.commit()
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
    for p in model.parameters():
        p.requires_grad_(False)
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
        """(B, L, S, S) contexts -> (B, S, S) decoder deltas (time-first
        internally: (L, B, C, S, S))."""
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
        """(L, S, S) context -> (S, S) decoder delta."""
        return predict_next_batch(context_int[None])[0]


        bcs = bcs_base.expand(1, -1, -1).tolist()
        out = model(x, field_idx_tensor, bcs, metadata=make_metadata(),
                    train=False)
        if isinstance(out, (tuple, list)):
            out = out[0]
        out = out.detach().cpu().numpy()
        if out.ndim == 5:
            delta = out[-1, 0, 0]
        elif out.ndim == 4:
            delta = out[-1, 0] if out.shape[0] == CONTEXT_LEN else out[0, 0]
        elif out.ndim == 3:
            delta = out[0]
        else:
            raise ValueError(f"unexpected model output shape {out.shape}")
        return np.asarray(delta, dtype=np.float32)

    out_root = "/vol/out/walrus_native"

    if "next_frame" in evals:
        nf_dir = os.path.join(out_root, "next_frame")
        os.makedirs(nf_dir, exist_ok=True)
        nf_rows = []
        for stem in stems:
            seq, inten = _load_npz_sequence(f"/vol/arrays/{stem}.npz")
            n = min(MAX_EVAL_WINDOWS, len(inten) - CONTEXT_LEN)
            # stride 6: ~3 min effective sampling; the optical state evolves
            # over hours, so trend/sign statistics are unaffected
            starts_all = np.arange(0, n, 6)
            preds, naives, targets, steps, ftimes = [], [], [], [], []
            BS = 32
            for lo in range(0, len(starts_all), BS):
                starts = starts_all[lo:lo + BS]
                batch = np.stack([inten[s:s + CONTEXT_LEN] for s in starts])
                raws = predict_next_batch(batch)
                last = batch[:, -1]
                pr = last + raws
                tg = inten[starts + CONTEXT_LEN]
                preds.append(pr); naives.append(last.copy())
                targets.append(tg); steps.extend((starts + CONTEXT_LEN).tolist())
                ftimes.extend(seq.frame_times[starts + CONTEXT_LEN].tolist())
                if lo == 0:
                    print(f"  batched forward OK, raw shape {raws.shape}",
                          flush=True)
            preds = np.stack(preds)
            naives = np.stack(naives)
            targets = np.stack(targets)
            steps_a = np.asarray(steps, dtype=np.int64)
            ftimes_a = np.asarray(ftimes, dtype=np.float32)
            mae = float(np.abs(preds - targets).mean())
            nmae = float(np.abs(naives - targets).mean())
            row = {"stem": stem, "role": seq.role, "model_family": "walrus",
                   "tag": "walrus_native", "mode": "next_frame",
                   "model_mae": mae, "naive_mae": nmae,
                   "mae_ratio": mae / max(nmae, 1e-12)}
            nf_rows.append(row)
            stem_dir = os.path.join(nf_dir, stem)
            os.makedirs(stem_dir, exist_ok=True)
            np.savez_compressed(
                os.path.join(stem_dir, f"preds_{stem}.npz"),
                pred=preds, naive=naives, targets=targets,
                target_steps=steps_a, frame_times=ftimes_a)
            print(f"[nf done] {stem}: ratio {row['mae_ratio']:.4f}", flush=True)
        with open(os.path.join(nf_dir, "comparison_summary.json"), "w") as f:
            json.dump(nf_rows, f, indent=2)

    if "rollout_anchored" in evals:
        ro_dir = os.path.join(out_root, "rollout_anchored")
        os.makedirs(ro_dir, exist_ok=True)
        ro_rows = []
        for stem in stems:
            seq, inten = _load_npz_sequence(f"/vol/arrays/{stem}.npz")
            current = seq.exogenous_all["current"]
            signs = np.sign(current.astype(np.float64))
            changes = [i for i in range(1, len(signs))
                       if signs[i] != 0 and signs[i - 1] != 0
                       and signs[i] != signs[i - 1]]
            onset = changes[0] if changes else 0
            start = max(0, onset - CONTEXT_LEN)
            ctx, persistence_frame = initialize_rollout_state(
                inten, start=start, context_len=CONTEXT_LEN
            )
            preds, naives, targets, steps, ftimes = [], [], [], [], []
            horizon = min(ROLLOUT_STEPS, len(inten) - start - CONTEXT_LEN)
            for step in range(horizon):
                tgt_idx = start + CONTEXT_LEN + step
                raw = predict_next(np.stack(ctx))
                pred = ctx[-1] + raw
                tgt = inten[tgt_idx]
                preds.append(pred)
                naives.append(persistence_frame.copy())
                targets.append(tgt)
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
            row = {"stem": stem, "role": seq.role, "model_family": "walrus",
                   "tag": "walrus_native", "mode": "rollout_anchored",
                   "anchor_frame": start, "model_mae": mae, "naive_mae": nmae,
                   "mae_ratio": mae / max(nmae, 1e-12)}
            ro_rows.append(row)
            stem_dir = os.path.join(ro_dir, stem)
            os.makedirs(stem_dir, exist_ok=True)
            np.savez_compressed(
                os.path.join(stem_dir, f"rollout_{stem}.npz"),
                pred=preds, naive=naives, targets=targets,
                target_steps=steps_a, frame_times=ftimes_a)
            print(f"[rollout done] {stem}: ratio {row['mae_ratio']:.4f}",
                  flush=True)
        with open(os.path.join(ro_dir, "comparison_summary.json"), "w") as f:
            json.dump(ro_rows, f, indent=2)

    VOL.commit()
    return "walrus eval done"


@app.local_entrypoint()
def main(payload_arg: str = ""):
    print(run_eval.remote(payload_arg or "{}"))
