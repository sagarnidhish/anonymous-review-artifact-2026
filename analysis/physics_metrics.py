#!/usr/bin/env python3
"""Physics-facing evaluation of learned emulators vs persistence.

Consumes per-stem NPZ artifacts (preds_<stem>.npz under next_frame/,
rollout_<stem>.npz under rollout/) written by run_sp_baseline_study.save_*,
plus locally regenerated particle arrays for EC channels, and produces:

  macro_trend.csv   per tag x mode x horizon bucket: pixel MAE ratio (ratio of
                    means, frozen convention) and correlation / OLS slope of
                    predicted vs true optical-state trajectories (masked mean,
                    P95, bright_frac90), plus identical persistence stats.
  phase_slopes.csv  per tag x mode x observable x current phase: OLS slope of
                    each observable vs time within positive/negative current
                    phases, per particle.
  summary.json      sign-flip coherence across particles per tag/mode/observable.

Observable and phase-label definitions replicate
project_scientific_progression_may2026/scripts/extract_sp_observables_and_phases.py.
"""

import argparse
import csv
import glob
import json
import os

import numpy as np
from scipy import ndimage as ndi

try:
    from analysis.evaluation_invariants import (
        bright_fraction_trajectory,
        calibrate_bright_threshold,
    )
except ModuleNotFoundError:  # direct script execution
    from evaluation_invariants import (
        bright_fraction_trajectory,
        calibrate_bright_threshold,
    )

HERE = os.path.dirname(os.path.abspath(__file__))
ARRAYS = os.path.abspath(os.path.join(HERE, "..", "data_prep", "arrays"))
OBSERVABLES = ["mean", "p95", "bright_frac90"]
BUCKETS = [("0-32", 0, 32), ("32-128", 32, 128), ("128-256", 128, 256),
           ("256-512", 256, 512)]


def load_sequence(stem):
    path = os.path.join(ARRAYS, f"{stem}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with np.load(path) as d:
        return {
            "intensity": d["intensity"].astype(np.float32),
            "current": d["current"],
            "frame_times": d["frame_times"],
        }


def largest_component(mask):
    lab, n = ndi.label(mask)
    if n == 0:
        return mask
    counts = np.bincount(lab.ravel())[1:]
    out = lab == int(np.argmax(counts)) + 1
    return ndi.binary_fill_holes(out)


def detect_roi(frames):
    """Frozen ROI detection (temporal projections + border MAD thresholds)."""
    p05 = np.percentile(frames, 5, axis=0)
    p95 = np.percentile(frames, 95, axis=0)
    dyn = p95 - p05
    bri = p95
    b = max(4, min(frames.shape[1:]) // 12)
    bd = np.concatenate([dyn[:b].ravel(), dyn[-b:].ravel(),
                         dyn[:, :b].ravel(), dyn[:, -b:].ravel()])
    bb = np.concatenate([bri[:b].ravel(), bri[-b:].ravel(),
                         bri[:, :b].ravel(), bri[:, -b:].ravel()])
    dyn_thr = np.median(bd) + 8 * max(np.median(np.abs(bd - np.median(bd))), 1e-8)
    bri_thr = np.median(bb) + 8 * max(np.median(np.abs(bb - np.median(bb))), 1e-8)
    mask = largest_component((dyn > dyn_thr) | (bri > bri_thr))
    if mask.mean() < 0.01 or mask.mean() > 0.95:
        score = dyn + np.maximum(bri - np.median(bb), 0)
        mask = largest_component(score > np.percentile(score, 70))
    return mask.astype(bool)


def observable_trajectories(frames, mask, bright_threshold):
    pix = frames[:, mask]
    return {
        "mean": pix.mean(axis=1),
        "p95": np.percentile(pix, 95, axis=1),
        "bright_frac90": bright_fraction_trajectory(
            frames, mask, bright_threshold
        ),
    }


def calibration_context(frames, target_steps, context_len=4):
    """Return observed frames immediately before the first target step."""
    steps = np.asarray(target_steps, dtype=np.int64)
    if len(steps) == 0:
        raise ValueError("target_steps is empty")
    stop = int(steps[0])
    start = stop - int(context_len)
    if start < 0 or stop > len(frames):
        raise ValueError("insufficient observed context for calibration")
    return np.asarray(frames)[start:stop], start, stop


def align_artifact_steps(
    sequence_frame_times, artifact_steps, artifact_frame_times=None
):
    """Map saved target indices to the unsliced source sequence.

    Some anchored evaluators saved indices relative to an anchor slice while
    preserving absolute camera timestamps.  Trust a saved index only when its
    timestamp agrees with the source sequence; otherwise recover the global
    index from the timestamp and fail if no unique close match exists.
    """
    sequence_times = np.asarray(sequence_frame_times, dtype=np.float64)
    saved_steps = np.asarray(artifact_steps, dtype=np.int64)
    if saved_steps.ndim != 1 or len(saved_steps) == 0:
        raise ValueError("artifact target_steps must be a nonempty vector")
    if artifact_frame_times is None:
        if saved_steps.min() < 0 or saved_steps.max() >= len(sequence_times):
            raise ValueError("artifact target_steps are outside source sequence")
        return saved_steps, "target_steps_unverified"

    saved_times = np.asarray(artifact_frame_times, dtype=np.float64)
    if saved_times.shape != saved_steps.shape:
        raise ValueError("artifact frame_times and target_steps differ in length")
    cadence = (
        float(np.nanmedian(np.diff(sequence_times)))
        if len(sequence_times) > 1
        else 1.0
    )
    tolerance = max(0.05, abs(cadence) * 1e-3)
    indices_valid = (
        saved_steps.min() >= 0 and saved_steps.max() < len(sequence_times)
    )
    if indices_valid and np.allclose(
        sequence_times[saved_steps], saved_times, atol=tolerance, rtol=0
    ):
        return saved_steps, "target_steps"

    right = np.searchsorted(sequence_times, saved_times, side="left")
    right = np.clip(right, 0, len(sequence_times) - 1)
    left = np.clip(right - 1, 0, len(sequence_times) - 1)
    choose_left = (
        np.abs(sequence_times[left] - saved_times)
        <= np.abs(sequence_times[right] - saved_times)
    )
    recovered = np.where(choose_left, left, right).astype(np.int64)
    error = np.abs(sequence_times[recovered] - saved_times)
    if np.any(error > tolerance) or len(np.unique(recovered)) != len(recovered):
        raise ValueError(
            "cannot align artifact target steps to source frame times; "
            f"max_time_error={float(error.max()):.6g}s"
        )
    return recovered, "frame_times"


def phase_labels(current, rest_fraction=0.02, transition_buffer=25):
    thr = rest_fraction * max(float(np.nanmax(np.abs(current))), 1e-12)
    base = np.full(len(current), "rest", dtype=object)
    base[current > thr] = "positive_current"
    base[current < -thr] = "negative_current"
    labels = base.copy()
    changes = np.where(base[1:] != base[:-1])[0] + 1
    for c in changes:
        lo = max(0, c - transition_buffer)
        hi = min(len(labels), c + transition_buffer + 1)
        labels[lo:hi] = "transition"
    return labels


def corr(a, b):
    good = np.isfinite(a) & np.isfinite(b)
    if good.sum() < 10 or np.std(a[good]) < 1e-12 or np.std(b[good]) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a[good], b[good])[0, 1])


def slope_of_scatter(x_traj, y_traj):
    good = np.isfinite(x_traj) & np.isfinite(y_traj)
    if good.sum() < 10 or np.std(x_traj[good]) < 1e-12:
        return float("nan")
    return float(np.polyfit(x_traj[good], y_traj[good], 1)[0])


def ols_slope(t, y):
    good = np.isfinite(y)
    if good.sum() < 5 or np.std(t[good]) <= 0:
        return float("nan")
    return float(np.polyfit(t[good], y[good], 1)[0])


def stem_of(fname):
    return (fname.replace("rollout_", "").replace("preds_", "")
            .replace(".npz", ""))


def process_artifact(art_path, seq, mode, tag):
    with np.load(art_path) as d:
        pred = d["pred"]
        naive = d["naive"]
        targets = d["targets"]
        saved_steps = np.asarray(d["target_steps"], dtype=np.int64)
        artifact_frame_times = d["frame_times"] if "frame_times" in d else None

    steps, step_source = align_artifact_steps(
        seq["frame_times"], saved_steps, artifact_frame_times
    )
    if not (len(pred) == len(naive) == len(targets) == len(steps)):
        raise ValueError("artifact arrays and aligned target steps differ in length")

    mask = detect_roi(seq["intensity"])
    context, calibration_start, calibration_stop = calibration_context(
        seq["intensity"], steps
    )
    bright_threshold = calibrate_bright_threshold(context, mask)
    # Observables evaluated at the predicted step positions only. One
    # context-calibrated threshold is shared by truth, model, and persistence.
    obs_true = observable_trajectories(targets, mask, bright_threshold)
    obs_pred = observable_trajectories(pred, mask, bright_threshold)
    obs_naive = observable_trajectories(naive, mask, bright_threshold)
    times_s = seq["frame_times"][steps]

    abs_pred = np.abs(pred - targets).mean(axis=(1, 2))
    abs_naive = np.abs(naive - targets).mean(axis=(1, 2))

    spans = ([("all", 0, len(steps))] if mode == "next_frame" else
             [(name, a, min(b, len(steps))) for name, a, b in BUCKETS])

    macro_rows = []
    for name, a, b in spans:
        idx = np.arange(a, b)
        if len(idx) < 10:
            continue
        row = {
            "tag": tag,
            "stem": stem_of(os.path.basename(art_path)),
            "mode": mode,
            "bucket": name,
            "n_steps": int(len(idx)),
            "bright_threshold": bright_threshold,
            "calibration_start": calibration_start,
            "calibration_stop": calibration_stop,
            "target_step_source": step_source,
            "pixel_mae_ratio": float(abs_pred[idx].mean()
                                     / max(abs_naive[idx].mean(), 1e-12)),
        }
        for key in OBSERVABLES:
            row[f"corr_{key}"] = corr(obs_pred[key][idx], obs_true[key][idx])
            row[f"slope_{key}"] = slope_of_scatter(obs_pred[key][idx],
                                                   obs_true[key][idx])
            row[f"corr_naive_{key}"] = corr(obs_naive[key][idx],
                                            obs_true[key][idx])
        macro_rows.append(row)

    phases = phase_labels(seq["current"])[steps]
    phase_rows = []
    for key in OBSERVABLES:
        for phase in ("positive_current", "negative_current"):
            sel = phases == phase
            if sel.sum() < 10:
                continue
            t = times_s[sel] - times_s[sel].min()
            phase_rows.append({
                "tag": tag,
                "stem": stem_of(os.path.basename(art_path)),
                "mode": mode,
                "observable": key,
                "phase": phase,
                "n_steps": int(sel.sum()),
                "slope_true_per_s": ols_slope(t, obs_true[key][sel]),
                "slope_pred_per_s": ols_slope(t, obs_pred[key][sel]),
                "slope_naive_per_s": ols_slope(t, obs_naive[key][sel]),
            })
    return macro_rows, phase_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", required=True,
                    help="directory containing <tag>/{next_frame,rollout}/")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    all_macro, all_phase = [], []
    for tag in sorted(os.listdir(args.results_root)):
        tag_dir = os.path.join(args.results_root, tag)
        if not os.path.isdir(tag_dir):
            continue
        for mode in ("next_frame", "rollout", "rollout_anchored"):
            mode_dir = os.path.join(tag_dir, mode)
            if not os.path.isdir(mode_dir):
                continue
            for art_path in sorted(
                    glob.glob(os.path.join(mode_dir, "*", "*.npz"))):
                stem = stem_of(os.path.basename(art_path))
                try:
                    seq = load_sequence(stem)
                except FileNotFoundError:
                    print(f"[skip] unknown stem {stem}")
                    continue
                try:
                    mrows, prows = process_artifact(art_path, seq, mode, tag)
                except Exception as e:
                    import traceback
                    print(f"[ARTIFACT FAIL] {art_path}: {e}", flush=True)
                    traceback.print_exc()
                    raise
                all_macro.extend(mrows)
                all_phase.extend(prows)
        print(f"[done] {tag}", flush=True)

    def dump(rows, name, fields):
        path = os.path.join(args.out_dir, name)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {path} ({len(rows)} rows)")

    keys = ["tag", "stem", "mode", "bucket", "n_steps",
            "bright_threshold", "calibration_start", "calibration_stop",
            "target_step_source",
            "pixel_mae_ratio"]
    for k in OBSERVABLES:
        keys += [f"corr_{k}", f"slope_{k}", f"corr_naive_{k}"]
    dump(all_macro, "macro_trend.csv", keys)

    pkeys = ["tag", "stem", "mode", "observable", "phase", "n_steps",
             "slope_true_per_s", "slope_pred_per_s", "slope_naive_per_s"]
    dump(all_phase, "phase_slopes.csv", pkeys)

    summary = {}
    for tag in sorted({r["tag"] for r in all_phase}):
        for mode in ("next_frame", "rollout"):
            for key in OBSERVABLES:
                sub = [r for r in all_phase if r["tag"] == tag
                       and r["mode"] == mode and r["observable"] == key]
                stems = sorted({r["stem"] for r in sub})
                entry = {}
                for who, col in (("truth", "slope_true_per_s"),
                                 ("emulator", "slope_pred_per_s"),
                                 ("persistence", "slope_naive_per_s")):
                    by_stem = {}
                    for st in stems:
                        d = {r["phase"]: float(r[col]) for r in sub
                             if r["stem"] == st
                             and r["phase"] in ("positive_current",
                                                "negative_current")
                             and np.isfinite(float(r[col]))}
                        if len(d) == 2:
                            by_stem[st] = d
                    flips = sum(
                        1 for d in by_stem.values()
                        if np.sign(d["negative_current"])
                        != np.sign(d["positive_current"]))
                    entry[who] = {
                        "n_particles": len(by_stem),
                        "n_sign_flips": flips,
                        "coherence": (flips / len(by_stem)) if by_stem
                        else float("nan"),
                    }
                summary.setdefault(tag, {}).setdefault(mode, {})[key] = entry
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {os.path.join(args.out_dir, 'summary.json')}")


if __name__ == "__main__":
    main()
