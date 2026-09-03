#!/usr/bin/env python3
"""Generate appendix tables from frozen CSVs, fresh-run analysis outputs,
and locally computed constants.  Writes paper/tables/*.tex."""

import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
FROZEN = os.path.join(ROOT, "results", "frozen")
ANALYSIS = os.path.join(ROOT, "results", "analysis_corrected")
CORRECTED_WALRUS = os.path.join(
    ROOT, "results", "out", "walrus_native_corrected", "rollout_anchored"
)
OUT = os.path.join(ROOT, "paper", "tables")

FAMILIES = ["unet", "convlstm", "simvp", "residual_cnn", "predrnn", "predrnnpp"]
FAMILY_LABEL = {
    "unet": "U-Net", "convlstm": "ConvLSTM", "simvp": "SimVP",
    "residual_cnn": "Residual CNN", "predrnn": "PredRNN", "predrnnpp": "PredRNN++",
}
PARAMS = {  # computed locally via build_model (context_len=4, base=32, hidden=2)
    "unet": (7763329, 7766785), "convlstm": (112033, 115489),
    "simvp": (391937, 392801), "residual_cnn": (89057, 92513),
    "predrnn": (237569, 243617), "predrnnpp": (275457, 281505),
}


def rows_of(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def esc(s):
    return str(s).replace("_", "\\_")

def write(name, lines):
    path = os.path.join(OUT, name)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", path)


def benchmark_full():
    rows = rows_of(os.path.join(FROZEN, "baseline_suite_test_summary.csv"))
    nf = {(r["model_family"], r["tag"]): r for r in rows if r["mode"] == "next_frame"}
    ro = {(r["model_family"], r["tag"]): r for r in rows if r["mode"] == "rollout"}
    forms = [("image_only_delta",
              "image-only, delta-formulation"),
             ("multichannel_delta",
              "protocol-conditioned, delta-formulation"),
             ("image_only_direct", "image-only, direct"),
             ("multichannel_direct", "protocol-conditioned, direct")]
    lines = [r"\begin{tabular}{llccccc}", r"\toprule",
             r"Family & Form & MAE ratio & rev.\ ratio & non-rev.\ ratio & RMSE ratio & tag \\",
             r"\midrule"]
    for fam in FAMILIES:
        first = True
        for tag, label in forms:
            r = nf.get((fam, tag))
            if not r:
                continue
            v = float(r["mean_mae_ratio"])
            rev = float(r["mean_reversal_mae_ratio"])
            non = float(r["mean_nonreversal_mae_ratio"])
            rm = float(r["mean_rmse_ratio"])
            fam_cell = (FAMILY_LABEL[fam] if first else "")
            first = False
            cell = lambda x: (f"\\textbf{{{x:.3f}}}" if x < 1 else f"{x:.3f}")
            lines.append(f"{fam_cell} & {label} & {cell(v)} & {rev:.3f} & "
                         f"{non:.3f} & {cell(rm)} & {esc(tag)} \\\\")
    lines.append(r"\midrule")
    for fam in FAMILIES:
        cand = [(t, r) for (f, t), r in ro.items() if f == fam]
        if not cand:
            continue
        tag, r = min(cand, key=lambda kv: float(kv[1]["mean_mae_ratio"]))
        lines.append(f"{FAMILY_LABEL[fam]} & rollout (best) & "
                     f"{float(r['mean_mae_ratio']):.3f} & "
                     f"{float(r['mean_reversal_mae_ratio']):.3f} & "
                     f"{float(r['mean_nonreversal_mae_ratio']):.3f} & "
                     f"{float(r['mean_rmse_ratio']):.3f} & {esc(tag)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("benchmark_full.tex", lines)


def walrus():
    rows = rows_of(os.path.join(FROZEN, "walrus_variant_summary.csv"))
    desc = {
        "sp_native": "unmodified decoder, 128$\\times$128 input",
        "sp_diffnorm":
            "difference-normalized input movies",
        "sp_diffnorm_ws10":
            "difference normalization + context stride 10",
        "sp_ws10": "context stride 10",
        "sp_down128": "input downsampled to 128$\\times$128",
        "sp_down32": "input downsampled to 32$\\times$32",
        "fov_native": "field-of-view movies, unmodified decoder",
        "fov_diffnorm": "field-of-view, difference-normalized",
        "fov_diffnorm_ws10":
            "field-of-view, diff-norm + stride 10",
        "fov_ws10": "field-of-view, context stride 10",
        "fov_down128": "field-of-view, downsampled 128",
        "fov_down32": "field-of-view, downsampled 32",
    }
    lines = [r"\begin{tabular}{p{2.6cm}p{4.2cm}cccc}", r"\toprule",
             r"Variant & Meaning & $n$ test & mean & min & max \\",
             r"\midrule"]
    for r in sorted(rows, key=lambda r: float(r["mean_test_ratio"])):
        lines.append(f"{esc(r['variant'])} & {desc.get(r['variant'], '')} & "
                     f"{r['n_test']} & "
                     f"{float(r['mean_test_ratio']):.3f} & "
                     f"{float(r['min_test_ratio']):.3f} & "
                     f"{float(r['max_test_ratio']):.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("walrus.tex", lines)


def signflip():
    rows = rows_of(os.path.join(ANALYSIS, "phase_slopes.csv"))
    test = {f"GRA29_C20_45deg_particle{i}" for i in (1, 2, 3, 4)}
    sub = [r for r in rows if r["mode"] == "next_frame" and r["stem"] in test
           and r["observable"] == "bright_frac90"]
    lines = [r"\begin{tabular}{llccc}", r"\toprule",
             r"Configuration & Particle & slope (neg.\ curr.) & slope (pos.\ curr.) & flip \\",
             r"\midrule"]
    plain = {"unet_image_only_delta": "U-Net (image-only inputs)",
             "unet_multichannel_delta":
                 "U-Net (voltage/current-conditioned)",
             "predrnnpp_image_only_delta": "PredRNN++ (image-only inputs)",
             "convlstm_multichannel_delta":
                 "ConvLSTM (voltage/current-conditioned)",
             "unet_image_only_delta_rft":
                 "U-Net (image-only, recursive-horizon fine-tuned)"}
    tags = [tag for tag in plain if any(r["tag"] == tag for r in sub)]
    for tag in tags:
        for stem in sorted(test):
            d = {r["phase"]: r for r in sub if r["tag"] == tag and r["stem"] == stem}
            if len(d) < 2:
                continue
            sn = float(d["negative_current"]["slope_pred_per_s"])
            sp = float(d["positive_current"]["slope_pred_per_s"])
            stn = float(d["negative_current"]["slope_true_per_s"])
            stp = float(d["positive_current"]["slope_true_per_s"])
            flip = "$\\checkmark$" if np.sign(sn) != np.sign(sp) else "$\\times$"
            tflip = "$\\checkmark$" if np.sign(stn) != np.sign(stp) else "$\\times$"
            lines.append(f"{plain.get(tag, esc(tag))} & particle "
                         f"{stem[-1]} & {sn:.2e} & {sp:.2e} & "
                         f"{flip} (truth {tflip}) \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("signflip.tex", lines)


def hyperparams():
    lines = [r"\begin{tabular}{ll}", r"\toprule"]
    shared = [
        ("Input resolution", "128$\\times$128 (bilinear from native crop)"),
        ("Context length / stride", "4 frames / 1"),
        ("Train windows per stem", "$\\leq$3000 (first $N$ starts)"),
        ("Validation split", "last 10\\% of train windows (early stopping)"),
        ("Optimizer", "AdamW, weight decay $10^{-4}$"),
        ("Learning rate / schedule", "$10^{-3}$, cosine to $10^{-2}\\cdot$lr"),
        ("Epochs / early stopping", "60 / patience 10 on validation MAE"),
        ("Recovered PredRNN++ cap", "26 epochs / same validation rule"),
        (r"Batch size / grad.\ clip", "8 / norm 1.0"),
        ("Seeds", "1337 (frozen suite + fresh), 2026 (replicates)"),
        ("Reversal window", "$\\pm5$ frames around current sign change"),
        ("Rollout horizons", "256 (frozen), 512 (fresh + anchored)"),
        ("Anchored rollout start", "first current sign change $-$ context"),
        ("Hardware", "1 GPU/run; A100-40GB, RTX 3090, or RTX 4090"),
    ]
    for k, v in shared:
        lines.append(f"{k} & {v} \\\\")
    lines.append(r"\midrule")
    for fam in FAMILIES:
        p1, p4 = PARAMS[fam]
        lines.append(f"{FAMILY_LABEL[fam]} parameters & "
                     f"{p1:,} (image-only) / {p4:,} (4-channel) \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("hyperparams.tex", lines)


def walrus_rollout_corrected():
    path = os.path.join(CORRECTED_WALRUS, "repair_summary.json")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    rows = json.load(open(path))
    if len(rows) != 4 or not all(r.get("naive_changed") for r in rows):
        raise ValueError("corrected WALRUS rollout audit is incomplete")
    lines = [r"\begin{tabular}{lrrr}", r"\toprule",
             r"Held-out movie & model MAE & persistence MAE & ratio \\",
             r"\midrule"]
    for row in rows:
        lines.append(
            f"particle {row['stem'][-1]} & {row['model_mae']:.3f} & "
            f"{row['corrected_naive_mae']:.3f} & "
            f"{row['corrected_mae_ratio']:.2f} \\\\"
        )
    lines.append(r"\midrule")
    lines.append(
        "mean over particles & "
        f"{np.mean([r['model_mae'] for r in rows]):.3f} & "
        f"{np.mean([r['corrected_naive_mae'] for r in rows]):.3f} & "
        f"{np.mean([r['corrected_mae_ratio'] for r in rows]):.2f} \\\\"
    )
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("walrus_rollout_corrected.tex", lines)


import numpy as np  # noqa: E402

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    benchmark_full()
    walrus()
    signflip()
    hyperparams()
    walrus_rollout_corrected()
