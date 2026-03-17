#!/usr/bin/env python3
"""Phase 1.3.4 legacy simplified-baseline comparison visualization.

Compares NeuroHorizon v2 results with the legacy simplified baselines from the
original 1.8.3 experiment. These are NOT faithful reproductions of NDT2,
Neuroformer, or IBL-MtM.

Usage:
    python scripts/analysis/neurohorizon/phase1_benchmark_compare.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path("/root/autodl-tmp/NeuroHorizon/results/logs")
OUT_DIR = Path("/root/autodl-tmp/NeuroHorizon/results/figures/phase1_v2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOWS = [250, 500, 1000]

# Load NeuroHorizon v2 results
nh_results = {}
for w in WINDOWS:
    p = BASE / f"phase1_v2_{w}ms_cont" / "lightning_logs" / "eval_v2_results.json"
    if p.exists():
        with open(p) as f:
            d = json.load(f)
            nh_results[w] = {
                "fp_bps": d["continuous"]["fp_bps"],
                "r2": d["continuous"]["r2"],
            }

# Load benchmark results
BENCHMARKS = ["ndt2", "neuroformer", "ibl_mtm"]
BENCH_LABELS = {
    "ndt2": "Legacy NDT2-like",
    "neuroformer": "Legacy Neuroformer-like",
    "ibl_mtm": "Legacy IBL-MtM-like",
}
BENCH_COLORS = {"ndt2": "#d62728", "neuroformer": "#9467bd", "ibl_mtm": "#8c564b"}

bench_results = {}
for model in BENCHMARKS:
    bench_results[model] = {}
    for w in WINDOWS:
        p = BASE / f"phase1_benchmark_{model}_{w}ms" / "results.json"
        if p.exists():
            with open(p) as f:
                d = json.load(f)
                bench_results[model][w] = {
                    "fp_bps": d["best_val_fp_bps"],
                    "n_params": d["n_params"],
                }

# === Figure 6: Multi-model fp-bps comparison bar chart ===
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel A: Grouped bar chart
models = ["NeuroHorizon"] + [BENCH_LABELS[m] for m in BENCHMARKS]
colors = ["#1f77b4", "#d62728", "#9467bd", "#8c564b"]

x = np.arange(len(WINDOWS))
width = 0.2

for i, (model_key, label) in enumerate(
    [("neurohorizon", "NeuroHorizon")] + [(m, BENCH_LABELS[m]) for m in BENCHMARKS]
):
    vals = []
    for w in WINDOWS:
        if model_key == "neurohorizon":
            vals.append(nh_results.get(w, {}).get("fp_bps", 0))
        else:
            vals.append(bench_results.get(model_key, {}).get(w, {}).get("fp_bps", 0))

    bars = axes[0].bar(x + i * width - 1.5 * width, vals, width, label=label,
                       color=colors[i], alpha=0.85)
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            axes[0].text(bar.get_x() + bar.get_width() / 2., h,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=8)

axes[0].set_xlabel("Prediction Window (ms)", fontsize=12)
axes[0].set_ylabel("fp-bps", fontsize=12)
axes[0].set_title(
    "fp-bps Comparison: NeuroHorizon vs Legacy Simplified Baselines",
    fontsize=13,
)
axes[0].set_xticks(x)
axes[0].set_xticklabels([f"{w}ms" for w in WINDOWS])
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3, axis="y")

# Panel B: Line chart with relative improvement
for i, (model_key, label) in enumerate(
    [("neurohorizon", "NeuroHorizon")] + [(m, BENCH_LABELS[m]) for m in BENCHMARKS]
):
    vals = []
    for w in WINDOWS:
        if model_key == "neurohorizon":
            vals.append(nh_results.get(w, {}).get("fp_bps", 0))
        else:
            vals.append(bench_results.get(model_key, {}).get(w, {}).get("fp_bps", 0))

    marker = "o" if model_key == "neurohorizon" else "s"
    lw = 2.5 if model_key == "neurohorizon" else 1.5
    axes[1].plot(WINDOWS, vals, f"-{marker}", label=label, color=colors[i],
                linewidth=lw, markersize=8 if model_key == "neurohorizon" else 6)

axes[1].set_xlabel("Prediction Window (ms)", fontsize=12)
axes[1].set_ylabel("fp-bps", fontsize=12)
axes[1].set_title("fp-bps vs Prediction Window (Legacy Baselines)", fontsize=13)
axes[1].set_xticks(WINDOWS)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / "06_benchmark_comparison.png", dpi=150)
plt.close()
print("Figure 6 saved: 06_benchmark_comparison.png (legacy simplified baselines)")

# === Print summary ===
print("\n" + "=" * 75)
print("LEGACY SIMPLIFIED BASELINE COMPARISON SUMMARY (1.3.4 vs 1.8.3)")
print("=" * 75)
print(f"{'Model':15s} | {'250ms fp-bps':>12s} | {'500ms fp-bps':>12s} | {'1000ms fp-bps':>13s} | {'Params':>10s}")
print("-" * 75)
for model_key, label in [("neurohorizon", "NeuroHorizon")] + [(m, BENCH_LABELS[m]) for m in BENCHMARKS]:
    row = []
    for w in WINDOWS:
        if model_key == "neurohorizon":
            row.append(nh_results.get(w, {}).get("fp_bps", 0))
        else:
            row.append(bench_results.get(model_key, {}).get(w, {}).get("fp_bps", 0))
    if model_key == "neurohorizon":
        params = "~2.1M"
    else:
        p = bench_results.get(model_key, {}).get(250, {}).get("n_params", 0)
        params = f"~{p/1e6:.1f}M"
    print(f"{label:15s} | {row[0]:12.4f} | {row[1]:12.4f} | {row[2]:13.4f} | {params:>10s}")
print("=" * 75)

# Relative improvements
print("\nRelative improvement of NeuroHorizon over legacy simplified baselines:")
for m in BENCHMARKS:
    for w in WINDOWS:
        nh_val = nh_results.get(w, {}).get("fp_bps", 0)
        b_val = bench_results.get(m, {}).get(w, {}).get("fp_bps", 0)
        if b_val > 0:
            pct = (nh_val - b_val) / b_val * 100
            print(f"  vs {BENCH_LABELS[m]:15s} @ {w}ms: +{pct:.1f}%")
