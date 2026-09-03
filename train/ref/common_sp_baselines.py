#!/usr/bin/env python3
import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


log = logging.getLogger(__name__)


VIDEO_SUBSAMPLE = {
    "GRA29_C20_25deg_particle1": 1,
    "GRA29_C20_25deg_particle2": 1,
    "GRA29_C20_25deg_particle3": 1,
    "GRA29_C20_25deg_particle4": 1,
    "GRA29_C20_45deg_particle1": 1,
    "GRA29_C20_45deg_particle2": 1,
    "GRA29_C20_45deg_particle3": 1,
    "GRA29_C20_45deg_particle4": 1,
}

TRAIN_STEMS = {
    "GRA29_C20_25deg_particle1",
    "GRA29_C20_25deg_particle2",
    "GRA29_C20_25deg_particle3",
    "GRA29_C20_25deg_particle4",
}

TEST_STEMS = {
    "GRA29_C20_45deg_particle1",
    "GRA29_C20_45deg_particle2",
    "GRA29_C20_45deg_particle3",
    "GRA29_C20_45deg_particle4",
}


@dataclass
class LoadedSequence:
    stem: str
    role: str
    intensity: np.ndarray
    exogenous_all: Dict[str, np.ndarray]
    frame_times: np.ndarray
    raw_frame_indices: np.ndarray
    sequence_subsample_factor: int


def stem_of(path: str) -> str:
    return os.path.basename(path).replace("_well.hdf5", "")


def role_of_stem(stem: str) -> str:
    if stem in TRAIN_STEMS:
        return "train"
    if stem in TEST_STEMS:
        return "test"
    return "ignore"


def list_well_files(root: str) -> List[str]:
    out = []
    for split in ["train", "valid", "test"]:
        split_dir = os.path.join(root, split)
        if not os.path.isdir(split_dir):
            continue
        for name in sorted(os.listdir(split_dir)):
            if name.endswith("_well.hdf5"):
                out.append(os.path.join(split_dir, name))
    return out


def build_active_fields(use_voltage: bool, use_current: bool, use_time_norm: bool) -> List[str]:
    active = ["intensity"]
    if use_voltage:
        active.append("voltage")
    if use_current:
        active.append("current")
    if use_time_norm:
        active.append("time_norm")
    return active


def normalize_trace(values: np.ndarray, mode: str) -> np.ndarray:
    values = values.astype(np.float64)
    if mode == "range01":
        lo, hi = values.min(), values.max()
        scale = max(hi - lo, 1e-8)
        return ((values - lo) / scale).astype(np.float32)
    if mode == "maxabs":
        scale = max(np.abs(values).max(), 1e-8)
        return (values / scale).astype(np.float32)
    raise ValueError(f"Unsupported normalization mode: {mode}")


def _resize_frame(frame_wx: np.ndarray, target_size: int) -> np.ndarray:
    frame_hw = frame_wx.T
    t = torch.tensor(frame_hw[np.newaxis, np.newaxis], dtype=torch.float32)
    t = F.interpolate(t, size=(target_size, target_size), mode="bilinear", align_corners=False)
    return t[0, 0].numpy()


def load_sp_sequence(well_path: str, target_size: int, sequence_subsample_factor: int = 1) -> LoadedSequence:
    stem = stem_of(well_path)
    role = role_of_stem(stem)
    if role == "ignore":
        raise ValueError(f"Unexpected stem outside configured split: {stem}")

    subsample = VIDEO_SUBSAMPLE.get(stem, 1) * sequence_subsample_factor
    with h5py.File(well_path, "r") as f:
        ds = f["t0_fields/intensity"]
        cam_times = f["dimensions/time"][:].astype(np.float64)
        frame_indices = np.arange(0, ds.shape[1], subsample, dtype=np.int64)
        intensity_frames = np.stack(
            [_resize_frame(ds[0, idx], target_size) for idx in frame_indices],
            axis=0,
        ).astype(np.float32)
        sel_times = cam_times[frame_indices]
        if "metadata/potentiostat_value" in f:
            ec = f["metadata/potentiostat_value"][:]
            if ec.shape[0] == 3 and ec.shape[1] != 3:
                ec = ec.T
            ec_time = ec[:, 0].astype(np.float64)
            voltage = normalize_trace(ec[:, 1], mode="range01")
            current = normalize_trace(ec[:, 2], mode="maxabs")
            voltage_interp = np.interp(sel_times, ec_time, voltage).astype(np.float32)
            current_interp = np.interp(sel_times, ec_time, current).astype(np.float32)
        else:
            voltage_interp = np.zeros(len(sel_times), dtype=np.float32)
            current_interp = np.zeros(len(sel_times), dtype=np.float32)
    time_norm = normalize_trace(sel_times, mode="range01")
    return LoadedSequence(
        stem=stem,
        role=role,
        intensity=intensity_frames,
        exogenous_all={
            "voltage": voltage_interp,
            "current": current_interp,
            "time_norm": time_norm,
        },
        frame_times=sel_times.astype(np.float32),
        raw_frame_indices=frame_indices,
        sequence_subsample_factor=sequence_subsample_factor,
    )


def build_window_starts(sequence_len: int, context_len: int, window_stride: int, max_windows: int) -> List[int]:
    need = context_len * window_stride + window_stride
    starts = []
    for i in range(0, sequence_len - need + 1):
        starts.append(i)
        if max_windows > 0 and len(starts) >= max_windows:
            break
    return starts


def split_train_val_starts(starts: Sequence[int], val_fraction: float) -> Tuple[List[int], List[int]]:
    starts = list(starts)
    if not starts:
        return [], []
    n_val = max(1, int(round(len(starts) * val_fraction)))
    if len(starts) <= n_val:
        return starts, starts[-1:]
    return starts[:-n_val], starts[-n_val:]


def build_context_tensor(
    sequence: LoadedSequence,
    intensity_context: np.ndarray,
    context_indices: Sequence[int],
    active_fields: Sequence[str],
) -> np.ndarray:
    frames = []
    for pos, idx in enumerate(context_indices):
        channels = [intensity_context[pos]]
        for field_name in active_fields[1:]:
            scalar = float(sequence.exogenous_all[field_name][idx])
            channels.append(np.full_like(intensity_context[pos], scalar, dtype=np.float32))
        frames.append(np.stack(channels, axis=0))
    return np.stack(frames, axis=0).astype(np.float32)


class BaselineWindowDataset(Dataset):
    def __init__(
        self,
        sequences: Sequence[LoadedSequence],
        sample_index: Sequence[Tuple[int, int]],
        active_fields: Sequence[str],
        context_len: int,
        window_stride: int,
        predict_delta: bool,
    ):
        self.sequences = list(sequences)
        self.sample_index = list(sample_index)
        self.active_fields = list(active_fields)
        self.context_len = context_len
        self.window_stride = window_stride
        self.predict_delta = predict_delta

    def __len__(self):
        return len(self.sample_index)

    def __getitem__(self, idx):
        seq_idx, start = self.sample_index[idx]
        seq = self.sequences[seq_idx]
        ctx_indices = [start + k * self.window_stride for k in range(self.context_len)]
        target_index = start + self.context_len * self.window_stride
        x_seq = build_context_tensor(
            sequence=seq,
            intensity_context=seq.intensity[ctx_indices],
            context_indices=ctx_indices,
            active_fields=self.active_fields,
        )
        naive = seq.intensity[ctx_indices[-1]][None].astype(np.float32)
        target = seq.intensity[target_index][None].astype(np.float32)
        y = (target - naive).astype(np.float32) if self.predict_delta else target
        frame_time = np.float32(seq.frame_times[target_index])
        reversal = np.int64(is_reversal_step(seq, target_index))
        return (
            torch.from_numpy(x_seq),
            torch.from_numpy(y),
            torch.from_numpy(naive),
            torch.tensor(target_index, dtype=torch.int64),
            torch.tensor(frame_time, dtype=torch.float32),
            torch.tensor(reversal, dtype=torch.int64),
        )


def model_input_channels(active_fields: Sequence[str], context_len: int) -> int:
    return len(active_fields) * context_len


def current_sign_change_indices(current_values: np.ndarray) -> np.ndarray:
    signs = np.sign(current_values.astype(np.float64))
    change_indices = []
    prev = 0.0
    for idx, sign in enumerate(signs):
        if sign != 0:
            if prev != 0 and sign != prev:
                change_indices.append(idx)
            prev = sign
    return np.asarray(change_indices, dtype=np.int64)


def is_reversal_step(sequence: LoadedSequence, target_index: int, radius: int = 5) -> bool:
    current = sequence.exogenous_all["current"]
    if current.size == 0:
        return False
    changes = current_sign_change_indices(current)
    if changes.size == 0:
        return False
    return bool(np.any(np.abs(changes - int(target_index)) <= radius))


def reversal_mask(sequence: LoadedSequence, target_steps: np.ndarray, radius: int = 5) -> np.ndarray:
    current = sequence.exogenous_all["current"]
    changes = current_sign_change_indices(current)
    if changes.size == 0 or len(target_steps) == 0:
        return np.zeros(len(target_steps), dtype=bool)
    return np.any(np.abs(target_steps[:, None] - changes[None, :]) <= radius, axis=1)


def compute_frame_metrics(pred: np.ndarray, naive: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    if len(pred) == 0:
        return {
            "N": 0,
            "model_mae": float("nan"),
            "naive_mae": float("nan"),
            "mae_ratio": float("nan"),
            "model_mse": float("nan"),
            "naive_mse": float("nan"),
            "mse_ratio": float("nan"),
            "model_rmse": float("nan"),
            "naive_rmse": float("nan"),
            "rmse_ratio": float("nan"),
        }
    model_mae = float(np.mean(np.abs(pred - targets)))
    naive_mae = float(np.mean(np.abs(naive - targets)))
    model_mse = float(np.mean((pred - targets) ** 2))
    naive_mse = float(np.mean((naive - targets) ** 2))
    model_rmse = float(np.sqrt(model_mse))
    naive_rmse = float(np.sqrt(naive_mse))
    return {
        "N": int(len(pred)),
        "model_mae": model_mae,
        "naive_mae": naive_mae,
        "mae_ratio": model_mae / naive_mae if naive_mae > 0 else float("nan"),
        "model_mse": model_mse,
        "naive_mse": naive_mse,
        "mse_ratio": model_mse / naive_mse if naive_mse > 0 else float("nan"),
        "model_rmse": model_rmse,
        "naive_rmse": naive_rmse,
        "rmse_ratio": model_rmse / naive_rmse if naive_rmse > 0 else float("nan"),
    }


def metrics_for_mask(pred: np.ndarray, naive: np.ndarray, targets: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    if mask.size == 0 or not np.any(mask):
        return {
            "count": 0,
            "model_mae": float("nan"),
            "naive_mae": float("nan"),
            "mae_ratio": float("nan"),
            "model_rmse": float("nan"),
            "naive_rmse": float("nan"),
            "rmse_ratio": float("nan"),
        }
    sub_pred = pred[mask]
    sub_naive = naive[mask]
    sub_targets = targets[mask]
    base = compute_frame_metrics(sub_pred, sub_naive, sub_targets)
    return {
        "count": int(mask.sum()),
        "model_mae": base["model_mae"],
        "naive_mae": base["naive_mae"],
        "mae_ratio": base["mae_ratio"],
        "model_rmse": base["model_rmse"],
        "naive_rmse": base["naive_rmse"],
        "rmse_ratio": base["rmse_ratio"],
    }


def compute_rollout_horizon_metrics(pred: np.ndarray, naive: np.ndarray, targets: np.ndarray) -> Dict[str, List[float] | float]:
    if len(pred) == 0:
        return {
            "per_step_model_mae": [],
            "per_step_naive_mae": [],
            "per_step_mae_ratio": [],
            "first_step_mae_ratio": float("nan"),
            "last_step_mae_ratio": float("nan"),
        }
    per_step_model_mae = np.mean(np.abs(pred - targets), axis=(1, 2))
    per_step_naive_mae = np.mean(np.abs(naive - targets), axis=(1, 2))
    per_step_ratio = per_step_model_mae / np.maximum(per_step_naive_mae, 1e-12)
    return {
        "per_step_model_mae": per_step_model_mae.astype(np.float32).tolist(),
        "per_step_naive_mae": per_step_naive_mae.astype(np.float32).tolist(),
        "per_step_mae_ratio": per_step_ratio.astype(np.float32).tolist(),
        "first_step_mae_ratio": float(per_step_ratio[0]),
        "last_step_mae_ratio": float(per_step_ratio[-1]),
    }


def summarize_prediction_result(
    sequence: LoadedSequence,
    pred: np.ndarray,
    naive: np.ndarray,
    targets: np.ndarray,
    target_steps: np.ndarray,
    reversal_radius: int = 5,
    include_horizon: bool = False,
) -> Dict:
    row = compute_frame_metrics(pred, naive, targets)
    rev_mask = reversal_mask(sequence, target_steps, radius=reversal_radius)
    nonrev_mask = ~rev_mask if len(rev_mask) else rev_mask
    row.update(
        {
            "reversal_radius": int(reversal_radius),
            "reversal_metrics": metrics_for_mask(pred, naive, targets, rev_mask),
            "nonreversal_metrics": metrics_for_mask(pred, naive, targets, nonrev_mask),
        }
    )
    if include_horizon:
        row.update(compute_rollout_horizon_metrics(pred, naive, targets))
    return row


def dump_summary(out_path: str, rows: Sequence[Dict]):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(list(rows), f, indent=2)


def summarize_rows(rows: Sequence[Dict]) -> Dict[str, float]:
    if not rows:
        return {"count": 0}
    out = {
        "count": len(rows),
        "mean_model_mae": float(np.mean([r["model_mae"] for r in rows])),
        "mean_naive_mae": float(np.mean([r["naive_mae"] for r in rows])),
        "mean_mae_ratio": float(np.mean([r["mae_ratio"] for r in rows])),
        "mean_model_rmse": float(np.mean([r["model_rmse"] for r in rows])),
        "mean_naive_rmse": float(np.mean([r["naive_rmse"] for r in rows])),
        "mean_rmse_ratio": float(np.mean([r["rmse_ratio"] for r in rows])),
    }
    rev_rows = [r["reversal_metrics"]["mae_ratio"] for r in rows if r["reversal_metrics"]["count"] > 0]
    if rev_rows:
        out["mean_reversal_mae_ratio"] = float(np.mean(rev_rows))
    nonrev_rows = [r["nonreversal_metrics"]["mae_ratio"] for r in rows if r["nonreversal_metrics"]["count"] > 0]
    if nonrev_rows:
        out["mean_nonreversal_mae_ratio"] = float(np.mean(nonrev_rows))
    horizon_last = [r["last_step_mae_ratio"] for r in rows if "last_step_mae_ratio" in r]
    if horizon_last:
        out["mean_last_step_mae_ratio"] = float(np.mean(horizon_last))
    return out

