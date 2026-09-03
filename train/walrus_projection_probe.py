#!/usr/bin/env python3
"""Fit only the new Walrus intensity projection entries on 25 C GRA29."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch


def _projection_axis(name: str) -> int | None:
    if "embed." in name and name.endswith("proj1.weight"):
        return 1
    if "debed." in name and name.endswith("proj2.weight"):
        return 1
    if "debed." in name and name.endswith("proj2.bias"):
        return 0
    return None


def freeze_for_intensity_projection(model, intensity_idx: int) -> dict:
    """Freeze the model and expose containers holding the new field entries."""
    inventory = {}
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for name, parameter in model.named_parameters():
        axis = _projection_axis(name)
        if axis is None:
            continue
        if parameter.ndim <= axis or parameter.shape[axis] <= intensity_idx:
            raise ValueError(f"intensity index {intensity_idx} outside {name}")
        parameter.requires_grad_(True)
        active_entries = parameter.numel() // parameter.shape[axis]
        inventory[name] = {
            "axis": axis,
            "shape": list(parameter.shape),
            "container_entries": parameter.numel(),
            "active_entries": active_entries,
        }
    if not inventory:
        raise ValueError("no Walrus projection parameters matched")
    return inventory


def projection_entry_state(model, inventory: dict, intensity_idx: int) -> dict:
    named = dict(model.named_parameters())
    state = {}
    for name, metadata in inventory.items():
        parameter = named[name].detach()
        if metadata["axis"] == 1:
            value = parameter[:, intensity_idx]
        else:
            value = parameter[intensity_idx]
        state[name] = value.cpu().clone()
    return state


def restore_projection_entry_state(
    model, state: dict, intensity_idx: int
) -> None:
    named = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in state.items():
            parameter = named[name]
            restored = value.to(device=parameter.device, dtype=parameter.dtype)
            if _projection_axis(name) == 1:
                parameter[:, intensity_idx].copy_(restored)
            else:
                parameter[intensity_idx].copy_(restored)


def zero_disallowed_projection_gradients(
    model, inventory: dict, intensity_idx: int
) -> dict:
    """Zero every projection gradient except the newly appended field entry."""
    named = dict(model.named_parameters())
    observed = {}
    for name, metadata in inventory.items():
        gradient = named[name].grad
        if gradient is None:
            continue
        axis = metadata["axis"]
        if axis == 1:
            active = gradient[:, intensity_idx]
            observed[name] = float(torch.linalg.vector_norm(active).detach().cpu())
            gradient[:, :intensity_idx].zero_()
            gradient[:, intensity_idx + 1 :].zero_()
        else:
            active = gradient[intensity_idx]
            observed[name] = float(torch.abs(active).detach().cpu())
            gradient[:intensity_idx].zero_()
            gradient[intensity_idx + 1 :].zero_()
    return observed


def hash_pretrained_projection_entries(
    model, inventory: dict, intensity_idx: int
) -> dict:
    """Hash every projection value except the newly appended field entry."""
    named = dict(model.named_parameters())
    hashes = {}
    for name, metadata in inventory.items():
        parameter = named[name].detach()
        digest = hashlib.sha256()
        if metadata["axis"] == 1:
            pieces = (parameter[:, :intensity_idx], parameter[:, intensity_idx + 1 :])
        else:
            pieces = (parameter[:intensity_idx], parameter[intensity_idx + 1 :])
        for piece in pieces:
            digest.update(piece.cpu().contiguous().numpy().tobytes())
        hashes[name] = digest.hexdigest()
    return hashes


def align_checkpoint_with_field_to_index_map(
    checkpoint_state_dict,
    model_state_dict,
    checkpoint_field_to_index_map,
    model_field_to_index_map,
):
    checkpoint_num_dims = max(checkpoint_field_to_index_map.values()) + 1
    model_num_dims = max(model_field_to_index_map.values()) + 1
    scale_factor = (checkpoint_num_dims / model_num_dims) ** 0.5
    for name in model_state_dict:
        axis = _projection_axis(name)
        if axis is None:
            continue
        replacement = model_state_dict[name].clone()
        checkpoint_value = checkpoint_state_dict[name]
        for field, model_index in model_field_to_index_map.items():
            if field not in checkpoint_field_to_index_map:
                continue
            checkpoint_index = checkpoint_field_to_index_map[field]
            if name.endswith("proj2.bias"):
                replacement[model_index] = checkpoint_value[checkpoint_index]
            elif name.endswith("proj1.weight"):
                replacement[:, model_index] = (
                    checkpoint_value[:, checkpoint_index] * scale_factor
                )
            else:
                replacement[:, model_index] = checkpoint_value[:, checkpoint_index]
        checkpoint_state_dict[name] = replacement
    return checkpoint_state_dict


def strict_json_dump(path: Path, value) -> None:
    def sanitize(item):
        if isinstance(item, dict):
            return {key: sanitize(val) for key, val in item.items()}
        if isinstance(item, (list, tuple)):
            return [sanitize(val) for val in item]
        if isinstance(item, (float, np.floating)) and not np.isfinite(item):
            return None
        if isinstance(item, np.integer):
            return int(item)
        if isinstance(item, np.floating):
            return float(item)
        return item

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(sanitize(value), handle, indent=2, allow_nan=False)


def load_accepted_tiny_gate(
    root: Path, expected_checkpoint_sha256: str, expected_seed: int
) -> dict:
    """Validate a completed gate before reusing it in a shorter full-fit job."""
    root = Path(root).resolve()
    try:
        selector = json.loads((root / "selector_manifest.json").read_text())
        gate = json.loads((root / "tiny_overfit_gate.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise ValueError(f"accepted tiny gate is incomplete: {root}") from error
    if selector.get("checkpoint_sha256") != expected_checkpoint_sha256:
        raise ValueError("accepted tiny gate checkpoint does not match")
    if selector.get("seed") != int(expected_seed):
        raise ValueError("accepted tiny gate seed does not match")
    if gate.get("status") != "passed":
        raise ValueError("accepted tiny gate did not pass")
    if not gate.get("pretrained_projection_hashes_unchanged"):
        raise ValueError("accepted tiny gate changed pretrained entries")
    required = float(gate.get("required_relative_loss", 0.0))
    before = float(gate.get("before", {}).get("mean_model_mae", float("nan")))
    after = float(gate.get("after", {}).get("mean_model_mae", float("nan")))
    if not np.isfinite(before) or not np.isfinite(after) or not after < required * before:
        raise ValueError("accepted tiny gate loss criterion is invalid")
    names = gate.get("gradient_parameter_names", [])
    if not any("embed." in name for name in names) or not any(
        "debed." in name for name in names
    ):
        raise ValueError("accepted tiny gate lacks encoder/decoder gradients")
    if int(gate.get("steps_completed", 0)) < 1:
        raise ValueError("accepted tiny gate has no optimization steps")
    return gate


def evenly_spaced_starts(first: int, last: int, count: int) -> list[int]:
    if count < 1 or first < 0 or last < first:
        raise ValueError("invalid window sampling request")
    values = np.linspace(first, last, num=count, dtype=np.int64).tolist()
    if len(set(values)) != len(values):
        raise ValueError("requested more unique starts than the interval contains")
    return values


def build_records(stems, first: int, last: int, per_stem: int) -> list[tuple[str, int]]:
    starts = evenly_spaced_starts(first, last, per_stem)
    return [(stem, start) for stem in stems for start in starts]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arrays-dir", required=True)
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--walrus-repo", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--tiny-steps", type=int, default=64)
    parser.add_argument("--train-steps", type=int, default=256)
    parser.add_argument("--train-per-stem", type=int, default=64)
    parser.add_argument("--val-per-stem", type=int, default=16)
    parser.add_argument("--test-per-stem", type=int, default=32)
    parser.add_argument("--eval-interval", type=int, default=32)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--tiny-lr", type=float)
    parser.add_argument("--accepted-tiny-gate-dir")
    return parser.parse_args()


def resolve_tiny_learning_rate(tiny_lr: float | None, full_lr: float) -> float:
    value = float(full_lr if tiny_lr is None else tiny_lr)
    if value <= 0:
        raise ValueError("tiny-gate learning rate must be positive")
    return value


class IntensityAdapter:
    def __init__(self, model, intensity_idx, metadata, device):
        self.model = model
        self.intensity_index = torch.tensor([intensity_idx], device=device)
        self.metadata = metadata
        self.device = device
        self.boundaries = torch.tensor([[[1, 1], [1, 1], [2, 2]]])

    def predict_delta(self, contexts: np.ndarray) -> torch.Tensor:
        inputs = torch.as_tensor(contexts, dtype=torch.float32, device=self.device)
        inputs = inputs.permute(1, 0, 2, 3).unsqueeze(2)
        boundaries = self.boundaries.expand(inputs.shape[1], -1, -1).tolist()
        output = self.model(
            inputs,
            self.intensity_index,
            boundaries,
            metadata=self.metadata,
            train=False,
        )
        if isinstance(output, (tuple, list)):
            output = output[0]
        if output.ndim == 5:
            return output[-1, :, 0]
        if output.ndim == 4:
            return output[-1] if output.shape[0] == inputs.shape[0] else output[0]
        if output.ndim == 3:
            return output
        raise ValueError(f"unexpected Walrus output shape {tuple(output.shape)}")


def load_intensity_arrays(arrays_dir: Path, stems) -> dict[str, np.ndarray]:
    arrays = {}
    for stem in stems:
        path = arrays_dir / f"{stem}.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path) as data:
            arrays[stem] = data["intensity"].astype(np.float32)
    return arrays


def make_batch(arrays, records):
    contexts = np.stack([arrays[stem][start : start + 4] for stem, start in records])
    targets = np.stack([arrays[stem][start + 4] for stem, start in records])
    return contexts, targets


@torch.no_grad()
def evaluate_records(adapter, arrays, records, batch_size: int) -> dict:
    by_stem = {}
    for offset in range(0, len(records), batch_size):
        batch_records = records[offset : offset + batch_size]
        contexts, targets = make_batch(arrays, batch_records)
        raw = adapter.predict_delta(contexts)
        last = torch.as_tensor(contexts[:, -1], device=adapter.device)
        target = torch.as_tensor(targets, device=adapter.device)
        prediction = last + raw
        model_errors = torch.mean(torch.abs(prediction - target), dim=(1, 2)).cpu().numpy()
        naive_errors = torch.mean(torch.abs(last - target), dim=(1, 2)).cpu().numpy()
        for (stem, _), model_error, naive_error in zip(
            batch_records, model_errors, naive_errors
        ):
            bucket = by_stem.setdefault(stem, {"model": [], "naive": []})
            bucket["model"].append(float(model_error))
            bucket["naive"].append(float(naive_error))
    rows = []
    for stem in sorted(by_stem):
        model_mae = float(np.mean(by_stem[stem]["model"]))
        naive_mae = float(np.mean(by_stem[stem]["naive"]))
        rows.append(
            {
                "stem": stem,
                "count": len(by_stem[stem]["model"]),
                "model_mae": model_mae,
                "naive_mae": naive_mae,
                "mae_ratio": model_mae / max(naive_mae, 1e-12),
            }
        )
    return {
        "particle_rows": rows,
        "mean_model_mae": float(np.mean([row["model_mae"] for row in rows])),
        "mean_naive_mae": float(np.mean([row["naive_mae"] for row in rows])),
        "mean_particle_mae_ratio": float(np.mean([row["mae_ratio"] for row in rows])),
    }


def train_steps(
    *,
    adapter,
    arrays,
    records,
    optimizer,
    inventory,
    intensity_idx,
    steps,
    batch_size,
    rng,
    validation_records=None,
    eval_interval=0,
    patience=0,
):
    curve = []
    gradient_names = set()
    best_validation = float("inf")
    best_state = None
    stale = 0
    order = list(records)
    cursor = len(order)
    for step in range(1, steps + 1):
        if cursor + batch_size > len(order):
            rng.shuffle(order)
            cursor = 0
        batch_records = order[cursor : cursor + batch_size]
        cursor += batch_size
        contexts, targets = make_batch(arrays, batch_records)
        raw = adapter.predict_delta(contexts)
        last = torch.as_tensor(contexts[:, -1], device=adapter.device)
        target = torch.as_tensor(targets, device=adapter.device)
        loss = torch.mean(torch.abs(last + raw - target))
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite projection loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        observed = zero_disallowed_projection_gradients(
            adapter.model, inventory, intensity_idx
        )
        gradient_names.update(name for name, norm in observed.items() if norm > 0)
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in adapter.model.parameters() if parameter.requires_grad],
            max_norm=1.0,
        )
        optimizer.step()
        point = {"step": step, "train_loss": float(loss.detach().cpu())}
        if validation_records and (step % eval_interval == 0 or step == steps):
            validation = evaluate_records(
                adapter, arrays, validation_records, batch_size=1
            )
            point["validation"] = validation
            value = validation["mean_model_mae"]
            if value < best_validation:
                best_validation = value
                best_state = projection_entry_state(
                    adapter.model, inventory, intensity_idx
                )
                stale = 0
            else:
                stale += 1
        curve.append(point)
        if validation_records and patience > 0 and stale >= patience:
            break
    return {
        "curve": curve,
        "gradient_parameter_names": sorted(gradient_names),
        "best_validation_mae": best_validation,
        "best_state": best_state,
        "steps_completed": len(curve),
    }


def main():
    args = parse_args()
    started = time.time()
    output = Path(args.out_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    completion = output / "completion_manifest.json"
    if completion.is_file():
        raise RuntimeError(f"completion manifest already exists: {completion}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Walrus projection probe")
    device = torch.device("cuda")

    sys.path.insert(0, args.walrus_repo)
    import walrus  # noqa: F401
    import walrus.models  # noqa: F401
    import walrus.models.decoders  # noqa: F401
    import walrus.models.encoders  # noqa: F401
    import walrus.models.isotropic_model  # noqa: F401
    import walrus.models.shared_utils  # noqa: F401
    import walrus.models.spatial_blocks  # noqa: F401
    import walrus.models.spatiotemporal_blocks  # noqa: F401
    import walrus.models.temporal_blocks  # noqa: F401
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from the_well.data.datasets import WellMetadata

    config = OmegaConf.load(Path(args.ckpt_dir) / "extended_config.yaml")
    checkpoint = torch.load(
        Path(args.ckpt_dir) / "walrus.pt", map_location="cpu", weights_only=True
    )["app"]["model"]
    original_field_map = dict(config.data.get("field_index_map_override", {}))
    intensity_idx = max(original_field_map.values()) + 1
    field_map = {**original_field_map, "intensity": intensity_idx}
    model = instantiate(config.model, n_states=intensity_idx + 1)
    aligned = align_checkpoint_with_field_to_index_map(
        checkpoint,
        model.state_dict(),
        original_field_map,
        field_map,
    )
    model.load_state_dict(aligned)
    del checkpoint, aligned
    gc.collect()
    model = model.to(device).eval()

    inventory = freeze_for_intensity_projection(model, intensity_idx)
    if len(inventory) != 6:
        raise RuntimeError(f"expected six projection containers, found {len(inventory)}")
    before_hashes = hash_pretrained_projection_entries(
        model, inventory, intensity_idx
    )
    initial_entries = projection_entry_state(model, inventory, intensity_idx)
    strict_json_dump(
        output / "selector_manifest.json",
        {
            "source_commit": args.source_commit,
            "checkpoint_sha256": args.checkpoint_sha256,
            "seed": args.seed,
            "original_field_count": len(original_field_map),
            "intensity_index": intensity_idx,
            "inventory": inventory,
            "active_entry_count": sum(
                metadata["active_entries"] for metadata in inventory.values()
            ),
            "container_entry_count": sum(
                metadata["container_entries"] for metadata in inventory.values()
            ),
            "pretrained_projection_hashes_before": before_hashes,
            "all_other_parameters_frozen": all(
                (name in inventory) == parameter.requires_grad
                for name, parameter in model.named_parameters()
            ),
        },
    )

    metadata = WellMetadata(
        dataset_name="battery_lithiation_iSCAT",
        n_spatial_dims=2,
        field_names={0: ["intensity"], 1: [], 2: []},
        spatial_resolution=(128, 128, 1),
        scalar_names=[],
        constant_field_names={0: [], 1: [], 2: []},
        constant_scalar_names=[],
        boundary_condition_types=[],
        n_files=[],
        n_trajectories_per_file=[],
        n_steps_per_trajectory=[],
    )
    adapter = IntensityAdapter(model, intensity_idx, metadata, device)
    train_stems = [f"GRA29_C20_25deg_particle{i}" for i in range(1, 5)]
    test_stems = [f"GRA29_C20_45deg_particle{i}" for i in range(1, 5)]
    arrays = load_intensity_arrays(Path(args.arrays_dir), train_stems + test_stems)

    tiny_records = [(train_stems[0], start) for start in evenly_spaced_starts(0, 224, 8)]
    train_records = build_records(train_stems, 0, 2699, args.train_per_stem)
    validation_records = build_records(train_stems, 2700, 2999, args.val_per_stem)
    test_records = build_records(test_stems, 0, 2999, args.test_per_stem)
    baseline_validation = evaluate_records(adapter, arrays, validation_records, 1)
    baseline_test = evaluate_records(adapter, arrays, test_records, 1)

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if args.accepted_tiny_gate_dir:
        accepted_gate_dir = Path(args.accepted_tiny_gate_dir).resolve()
        accepted_gate = load_accepted_tiny_gate(
            accepted_gate_dir, args.checkpoint_sha256, args.seed
        )
        strict_json_dump(
            output / "tiny_overfit_gate.json",
            {
                **accepted_gate,
                "accepted_gate_directory": str(accepted_gate_dir),
                "reused_without_refitting": True,
            },
        )
    else:
        tiny_before = evaluate_records(adapter, arrays, tiny_records, 1)
        tiny_lr = resolve_tiny_learning_rate(args.tiny_lr, args.lr)
        optimizer = torch.optim.AdamW(trainable, lr=tiny_lr, weight_decay=0.0)
        tiny_run = train_steps(
            adapter=adapter,
            arrays=arrays,
            records=tiny_records,
            optimizer=optimizer,
            inventory=inventory,
            intensity_idx=intensity_idx,
            steps=args.tiny_steps,
            batch_size=args.batch_size,
            rng=random.Random(args.seed + 1),
        )
        tiny_after = evaluate_records(adapter, arrays, tiny_records, 1)
        tiny_pass = (
            np.isfinite(tiny_after["mean_model_mae"])
            and tiny_after["mean_model_mae"] < 0.9 * tiny_before["mean_model_mae"]
            and any("embed." in name for name in tiny_run["gradient_parameter_names"])
            and any("debed." in name for name in tiny_run["gradient_parameter_names"])
        )
        hashes_after_tiny = hash_pretrained_projection_entries(
            model, inventory, intensity_idx
        )
        if hashes_after_tiny != before_hashes:
            raise RuntimeError("pretrained projection entries changed during tiny gate")
        strict_json_dump(
            output / "tiny_overfit_gate.json",
            {
                "status": "passed" if tiny_pass else "failed",
                "records": tiny_records,
                "before": tiny_before,
                "after": tiny_after,
                "steps_completed": tiny_run["steps_completed"],
                "learning_rate": tiny_lr,
                "curve": tiny_run["curve"],
                "gradient_parameter_names": tiny_run["gradient_parameter_names"],
                "required_relative_loss": 0.9,
                "pretrained_projection_hashes_unchanged": True,
            },
        )
        if not tiny_pass:
            strict_json_dump(
                completion,
                {
                    "status": "tiny_overfit_gate_failed",
                    "elapsed_seconds": time.time() - started,
                    "source_commit": args.source_commit,
                },
            )
            raise SystemExit(4)

    restore_projection_entry_state(model, initial_entries, intensity_idx)
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    full_run = train_steps(
        adapter=adapter,
        arrays=arrays,
        records=train_records,
        optimizer=optimizer,
        inventory=inventory,
        intensity_idx=intensity_idx,
        steps=args.train_steps,
        batch_size=args.batch_size,
        rng=random.Random(args.seed + 2),
        validation_records=validation_records,
        eval_interval=args.eval_interval,
        patience=args.patience,
    )
    if full_run["best_state"] is None:
        raise RuntimeError("projection pilot did not produce a validation checkpoint")
    restore_projection_entry_state(model, full_run["best_state"], intensity_idx)
    fitted_validation = evaluate_records(adapter, arrays, validation_records, 1)
    validation_pass = (
        np.isfinite(fitted_validation["mean_model_mae"])
        and fitted_validation["mean_model_mae"]
        < baseline_validation["mean_model_mae"]
    )
    fitted_test = evaluate_records(adapter, arrays, test_records, 1)
    after_hashes = hash_pretrained_projection_entries(model, inventory, intensity_idx)
    if after_hashes != before_hashes:
        raise RuntimeError("pretrained projection entries changed during full probe")

    strict_json_dump(
        output / "training_curve.json",
        {
            "arguments": vars(args),
            "split": {
                "train_stems": train_stems,
                "test_stems": test_stems,
                "train_start_range": [0, 2699],
                "validation_start_range": [2700, 2999],
                "test_start_range": [0, 2999],
                "normalization": "archived_per_movie_full_record",
            },
            "baseline_validation": baseline_validation,
            "baseline_test": baseline_test,
            "curve": full_run["curve"],
            "steps_completed": full_run["steps_completed"],
            "gradient_parameter_names": full_run["gradient_parameter_names"],
            "fitted_validation": fitted_validation,
            "fitted_test": fitted_test,
        },
    )
    projection_checkpoint = output / "intensity_projection_probe.pt"
    torch.save(
        {
            "projection_entries": projection_entry_state(
                model, inventory, intensity_idx
            ),
            "inventory": inventory,
            "intensity_index": intensity_idx,
            "field_map": field_map,
            "seed": args.seed,
            "source_commit": args.source_commit,
            "walrus_checkpoint_sha256": args.checkpoint_sha256,
            "validation_pass": validation_pass,
        },
        projection_checkpoint,
    )
    strict_json_dump(
        completion,
        {
            "status": "passed" if validation_pass else "validation_gate_failed",
            "source_commit": args.source_commit,
            "walrus_checkpoint_sha256": args.checkpoint_sha256,
            "seed": args.seed,
            "tiny_gate_passed": True,
            "validation_improved": validation_pass,
            "baseline_validation": baseline_validation,
            "fitted_validation": fitted_validation,
            "baseline_test": baseline_test,
            "fitted_test": fitted_test,
            "steps_completed": full_run["steps_completed"],
            "projection_checkpoint": projection_checkpoint.name,
            "pretrained_projection_hashes_unchanged": True,
            "elapsed_seconds": time.time() - started,
        },
    )
    print(
        f"WALRUS PROJECTION PROBE status={'passed' if validation_pass else 'validation_gate_failed'} "
        f"validation={baseline_validation['mean_model_mae']:.6g}->{fitted_validation['mean_model_mae']:.6g}",
        flush=True,
    )


if __name__ == "__main__":
    main()
