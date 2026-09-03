#!/usr/bin/env python3
"""Debug: inspect WALRUS encoder embed structure and forward expectations."""

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
)

VOL = modal.Volume.from_name("gra29sp-v1", create_if_missing=True)

app = modal.App("gra29-walrus-debug", image=IMAGE)


@app.function(gpu="A100-40GB", volumes={"/vol": VOL}, timeout=1800)
def probe() -> str:
    import os
    import sys

    import torch
    from omegaconf import OmegaConf
    from hydra.utils import instantiate

    sys.path.insert(0, "/opt/walrus")
    import walrus  # noqa: F401
    import walrus.models  # noqa: F401
    import walrus.models.isotropic_model  # noqa: F401
    import walrus.models.encoders  # noqa: F401
    import walrus.models.decoders  # noqa: F401

    src = open("/opt/walrus/walrus/models/isotropic_model.py").read()
    lines = src.split("\n")
    print("=== isotropic_model.py lines 195-240 ===")
    for i in range(194, 240):
        print(f"{i + 1}: {lines[i]}")

    ckpt_path = "/vol/walrus_ckpt/walrus.pt"
    cfg_path = "/vol/walrus_ckpt/extended_config.yaml"
    if not os.path.exists(ckpt_path):
        return "no checkpoint on volume"
    checkpoint = torch.load(ckpt_path, map_location="cpu",
                            weights_only=True)["app"]["model"]
    config = OmegaConf.load(cfg_path)
    print("=== config.model ===")
    print(OmegaConf.to_yaml(config.model)[:2000])
    print("=== config.data keys ===")
    print(OmegaConf.to_yaml(config.data)[:800])

    original_field_map = dict(config.data.get("field_index_map_override", {}))
    total_fields = max(original_field_map.values()) + 2
    model = instantiate(config.model, n_states=total_fields)
    print("=== model top-level modules ===")
    for k, v in model.named_children():
        print(f"  {k}: {type(v).__name__}")
    enc = getattr(model, "encoder", None)
    if enc is not None and hasattr(enc, "embed"):
        print("=== encoder.embed ===")
        print(type(enc.embed), list(enc.embed.keys()))
    if hasattr(model, "embed"):
        print("=== model.embed ===")
        print(type(model.embed), list(model.embed.keys()))
    print("=== state_dict embed-ish keys (first 15) ===")
    for k in list(model.state_dict().keys()):
        if "embed" in k:
            print(" ", k)
    return "probe done"


@app.local_entrypoint()
def main():
    print(probe.remote())
