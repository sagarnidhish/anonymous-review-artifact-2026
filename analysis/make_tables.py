#!/usr/bin/env python3
"""Generate paper/tables/benchmark.tex from the frozen suite CSV."""

import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
SRC = os.path.join(ROOT, "results", "frozen", "baseline_suite_test_summary.csv")
OUT_DIR = os.path.join(ROOT, "paper", "tables")

FAMILIES = ["unet", "convlstm", "simvp", "residual_cnn", "predrnn", "predrnnpp"]
FAMILY_LABEL = {
    "unet": "U-Net", "convlstm": "ConvLSTM", "simvp": "SimVP",
    "residual_cnn": "Residual CNN", "predrnn": "PredRNN", "predrnnpp": "PredRNN++",
}
DELTA_FORMS = [
    ("image_only_delta", "img\\,$\\Delta$"),
    ("multichannel_delta", "prot\\,$\\Delta$"),
]
DIRECT_FORMS = (
    ("image_only_direct", "image direct"),
    ("multichannel_direct", "protocol direct"),
)


def fmt(value):
    if value >= 1000:
        exponent = int(f"{value:.0e}".split("e")[1])
        coefficient = value / (10 ** exponent)
        return f"{coefficient:.1f}$\\times10^{{{exponent}}}$"
    return f"{value:.3f}" if value < 10 else f"{value:.1f}"


def render_benchmark(rows):
    """Render the manuscript table from parsed frozen-suite summary rows."""
    nf = {(r["model_family"], r["tag"]): r for r in rows if r["mode"] == "next_frame"}
    ro = [r for r in rows if r["mode"] == "rollout"]
    best_rollout = {}
    rollout_count = {}
    for fam in FAMILIES:
        cand = [r for r in ro if r["model_family"] == fam]
        rollout_count[fam] = len(cand)
        if cand:
            best = min(cand, key=lambda r: float(r["mean_mae_ratio"]))
            best_rollout[fam] = float(best["mean_mae_ratio"])

    lines = [
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"& \multicolumn{4}{c}{Next-frame MAE ratio} & 256-step rollout \\",
        r"\cmidrule(lr){2-5}\cmidrule(lr){6-6}",
        r"Family & image $\Delta$ & protocol $\Delta$ & image direct & protocol direct & best ratio (available) \\",
        r"\midrule",
    ]
    for fam in FAMILIES:
        cells = []
        for tag, _ in DELTA_FORMS:
            r = nf.get((fam, tag))
            v = float(r["mean_mae_ratio"]) if r else float("nan")
            beat = r is not None and v < 1.0
            cell = fmt(v)
            cells.append(rf"\textbf{{{cell}}}" if beat else cell)
        for tag, _ in DIRECT_FORMS:
            row = nf.get((fam, tag))
            value = float(row["mean_mae_ratio"]) if row else float("nan")
            direct_cell = fmt(value)
            cells.append(
                rf"\textbf{{{direct_cell}}}" if row and value < 1 else direct_cell
            )
        roll = best_rollout.get(fam)
        rollout_cell = (
            f"{fmt(roll)} ({rollout_count[fam]}/4)"
            if roll is not None
            else "-- (0/4)"
        )
        cells.append(rollout_cell)
        lines.append(FAMILY_LABEL[fam] + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main():
    with open(SRC, newline="") as handle:
        rows = list(csv.DictReader(handle))
    latex = render_benchmark(rows)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "benchmark.tex"), "w") as f:
        f.write(latex)
    n_delta_beating = sum(
        float(row["mean_mae_ratio"]) < 1
        for row in rows
        if row["mode"] == "next_frame" and row["tag"] in dict(DELTA_FORMS)
    )
    print(f"wrote table ({n_delta_beating}/12 delta next-frame cells beat persistence)")


if __name__ == "__main__":
    main()
