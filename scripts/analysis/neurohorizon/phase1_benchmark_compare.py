#!/usr/bin/env python3
"""Phase 1.3.4 supplemental benchmark reference visualization.

Rebuilds ``results/figures/phase1_v2/06_benchmark_comparison.png`` with the
latest benchmark references that currently exist in the repo:

- NeuroHorizon baseline_v2 uses evalfix held-out test when available
- IBL-MtM uses faithful 250ms e10 / e50 / e300 runs
- Neuroformer uses faithful 50ms reference / 250ms canonical rollout, plus the
  latest 20260321 250ms +session-cond rerun rollout
- Legacy NDT2-like uses protocol-fix held-out test as a historical internal
  reference because no matching newer sweep exists

Usage:
    python scripts/analysis/neurohorizon/phase1_benchmark_compare.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/root/autodl-tmp/NeuroHorizon")
BASE = ROOT / "results" / "logs"
OUT_DIR = ROOT / "results" / "figures" / "phase1_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOWS = [250, 500, 1000]
ALL_PLOT_WINDOWS = [50, 250, 500, 1000]


def load_json(path: Path):
    with path.open() as handle:
        return json.load(handle)


def load_first_json(paths):
    for path in paths:
        if path.exists():
            return load_json(path), path
    return None, None


def annotate_series(ax, xs, ys, color):
    for x_val, y_val in zip(xs, ys):
        offset = 8 if y_val >= 0 else -16
        ax.annotate(
            f"{y_val:.3f}",
            (x_val, y_val),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=color,
        )


def annotate_bars(ax, bars):
    for bar in bars:
        height = bar.get_height()
        offset = 0.06 if height >= 0 else -0.18
        va = "bottom" if height >= 0 else "top"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + offset,
            f"{height:.3f}",
            ha="center",
            va=va,
            fontsize=8,
        )


def load_neurohorizon_results():
    results = {}
    source_label = "legacy valid"
    for window_ms in WINDOWS:
        payload, payload_path = load_first_json(
            [
                BASE / f"phase1_v2_evalfix_{window_ms}ms_cont" / "lightning_logs" / "version_0" / "eval_v2_test_results.json",
                BASE / f"phase1_v2_evalfix_{window_ms}ms_cont" / "lightning_logs" / "eval_v2_test_results.json",
                BASE / f"phase1_v2_{window_ms}ms_cont" / "lightning_logs" / "version_0" / "eval_v2_results.json",
                BASE / f"phase1_v2_{window_ms}ms_cont" / "lightning_logs" / "eval_v2_results.json",
            ]
        )
        if payload is None:
            continue
        if "phase1_v2_evalfix_" in str(payload_path):
            source_label = "evalfix held-out test"
        results[window_ms] = payload["continuous"]["fp_bps"]
    return results, source_label


def load_ndt2_protocolfix_results():
    results = {}
    for window_ms in WINDOWS:
        path = BASE / f"phase1_benchmark_protocolfix_ndt2_{window_ms}ms" / "results.json"
        if not path.exists():
            continue
        payload = load_json(path)
        results[window_ms] = payload["test_metrics"]["continuous"]["fp_bps"]
    return results


def load_ibl_results():
    configs = [
        ("IBL e10", 10, BASE / "phase1_benchmark_repro_faithful_ibl_mtm_250ms_combined_e10" / "results.json"),
        ("IBL e50", 50, BASE / "phase1_benchmark_repro_faithful_ibl_mtm_250ms_combined_e50_aligned" / "results.json"),
        (
            "IBL e300",
            300,
            BASE
            / "phase1-autoregressive-1.8-benchmark_model"
            / "20260321_benchmark_ibl_e300_neuroformer_session_conditioning"
            / "ibl_mtm_combined_e300_aligned"
            / "results.json",
        ),
    ]
    rows = []
    for label, epochs, path in configs:
        payload = load_json(path)
        rows.append(
            {
                "label": label,
                "epochs": epochs,
                "fp_bps": payload["test_metrics"]["fp_bps"],
                "best_valid_fp_bps": payload["best_valid_metrics"]["fp_bps"],
            }
        )
    return rows


def load_neuroformer_results():
    configs = [
        (
            "NF ref 50ms",
            50,
            BASE
            / "phase1_benchmark_repro_faithful_neuroformer_50ms_reference_e50_aligned"
            / "formal_eval"
            / "eval_results.json",
            False,
        ),
        (
            "NF can 250ms",
            250,
            BASE
            / "phase1_benchmark_repro_faithful_neuroformer_250ms_canonical_e50_aligned"
            / "formal_eval"
            / "eval_results.json",
            False,
        ),
        (
            "NF +SC 250ms",
            250,
            BASE
            / "phase1-autoregressive-1.8-benchmark_model"
            / "20260321_benchmark_ibl_e300_neuroformer_session_conditioning"
            / "neuroformer_250ms_session_conditioning_e50"
            / "results.json",
            True,
        ),
    ]
    rows = []
    for label, pred_window_ms, path, has_bug_note in configs:
        payload = load_json(path)
        if "continuous_metrics" in payload:
            test_rollout = payload["continuous_metrics"]["test"]["rollout"]["fp_bps"]
            test_true_past = payload["continuous_metrics"]["test"]["true_past"]["fp_bps"]
        else:
            test_rollout = payload["test_metrics"]["rollout"]["fp_bps"]
            test_true_past = payload["test_metrics"]["true_past"]["fp_bps"]
        rows.append(
            {
                "label": label,
                "pred_window_ms": pred_window_ms,
                "rollout_fp_bps": test_rollout,
                "true_past_fp_bps": test_true_past,
                "session_bug_note": has_bug_note,
            }
        )
    return rows


nh_results, nh_source = load_neurohorizon_results()
ndt2_results = load_ndt2_protocolfix_results()
ibl_results = load_ibl_results()
nf_results = load_neuroformer_results()

fig, (ax_windows, ax_reference, ax_focus_250) = plt.subplots(1, 3, figsize=(21, 6.8))
fig.suptitle("Phase1_v2 vs Current Benchmark References", fontsize=14, fontweight="bold")

# Panel A: available prediction-window view
ax_windows.axhline(0, color="gray", linestyle=":", alpha=0.6)

nh_x = sorted(nh_results)
nh_y = [nh_results[window_ms] for window_ms in nh_x]
ax_windows.plot(
    nh_x,
    nh_y,
    "-o",
    color="#1f77b4",
    linewidth=2.5,
    markersize=8,
    label=f"NeuroHorizon ({nh_source})",
)
annotate_series(ax_windows, nh_x, nh_y, "#1f77b4")

if ndt2_results:
    ndt2_x = sorted(ndt2_results)
    ndt2_y = [ndt2_results[window_ms] for window_ms in ndt2_x]
    ax_windows.plot(
        ndt2_x,
        ndt2_y,
        "--^",
        color="#7f7f7f",
        linewidth=1.8,
        markersize=7,
        label="Legacy NDT2-like (protocol-fix test)",
    )
    annotate_series(ax_windows, ndt2_x, ndt2_y, "#7f7f7f")

nf_main = [row for row in nf_results if row["label"] != "NF +SC 250ms"]
nf_x = [row["pred_window_ms"] for row in nf_main]
nf_y = [row["rollout_fp_bps"] for row in nf_main]
ax_windows.plot(
    nf_x,
    nf_y,
    "--s",
    color="#7b1fa2",
    linewidth=2.2,
    markersize=7,
    label="Neuroformer faithful (rollout)",
)
annotate_series(ax_windows, nf_x, nf_y, "#7b1fa2")

nf_sc = next(row for row in nf_results if row["label"] == "NF +SC 250ms")
ax_windows.scatter(
    [nf_sc["pred_window_ms"]],
    [nf_sc["rollout_fp_bps"]],
    color="#ce5db7",
    s=90,
    marker="D",
    label="Neuroformer +SC 250ms (rollout)",
    zorder=5,
)
annotate_series(ax_windows, [nf_sc["pred_window_ms"]], [nf_sc["rollout_fp_bps"]], "#ce5db7")

ibl_e300 = next(row for row in ibl_results if row["label"] == "IBL e300")
ax_windows.scatter(
    [250],
    [ibl_e300["fp_bps"]],
    color="#2e7d32",
    s=90,
    marker="D",
    label="IBL-MtM faithful e300 @ 250ms",
    zorder=5,
)
annotate_series(ax_windows, [250], [ibl_e300["fp_bps"]], "#2e7d32")

ax_windows.set_xlabel("Prediction Window (ms)")
ax_windows.set_ylabel("fp-bps")
ax_windows.set_title("Current baseline + latest available benchmark points")
ax_windows.set_xticks(ALL_PLOT_WINDOWS)
ax_windows.grid(True, alpha=0.3)
ax_windows.legend(fontsize=8.5, loc="lower left")

# Panel B: latest benchmark reference bars
bar_labels = [
    "NH\n250",
    "NDT2-like\n250",
    "NF ref\n50",
    "NF can\n250",
    "NF +SC\n250",
    "IBL e10\n250",
    "IBL e50\n250",
    "IBL e300\n250",
]
bar_values = [
    nh_results.get(250, np.nan),
    ndt2_results.get(250, np.nan),
    next(row["rollout_fp_bps"] for row in nf_results if row["label"] == "NF ref 50ms"),
    next(row["rollout_fp_bps"] for row in nf_results if row["label"] == "NF can 250ms"),
    nf_sc["rollout_fp_bps"],
    next(row["fp_bps"] for row in ibl_results if row["label"] == "IBL e10"),
    next(row["fp_bps"] for row in ibl_results if row["label"] == "IBL e50"),
    ibl_e300["fp_bps"],
]
bar_colors = [
    "#1f77b4",
    "#7f7f7f",
    "#b39ddb",
    "#7b1fa2",
    "#ce5db7",
    "#a5d6a7",
    "#4caf50",
    "#2e7d32",
]
bars = ax_reference.bar(np.arange(len(bar_labels)), bar_values, color=bar_colors, alpha=0.9)
annotate_bars(ax_reference, bars)
ax_reference.axhline(0, color="gray", linestyle=":", alpha=0.6)
ax_reference.set_xticks(np.arange(len(bar_labels)))
ax_reference.set_xticklabels(bar_labels)
ax_reference.set_ylabel("fp-bps")
ax_reference.set_title("250ms / available current benchmark references")
ax_reference.grid(True, alpha=0.3, axis="y")

# Panel C: focused 250ms comparison requested by user
nf_can_250 = next(row for row in nf_results if row["label"] == "NF can 250ms")
focus_labels = ["NH\n250", "NF faithful\n250", "IBL e300\n250"]
focus_values = [
    nh_results.get(250, np.nan),
    nf_can_250["rollout_fp_bps"],
    ibl_e300["fp_bps"],
]
focus_colors = ["#1f77b4", "#7b1fa2", "#2e7d32"]
focus_bars = ax_focus_250.bar(np.arange(len(focus_labels)), focus_values, color=focus_colors, alpha=0.9)
annotate_bars(ax_focus_250, focus_bars)
ax_focus_250.axhline(0, color="gray", linestyle=":", alpha=0.6)
ax_focus_250.set_xticks(np.arange(len(focus_labels)))
ax_focus_250.set_xticklabels(focus_labels)
ax_focus_250.set_ylabel("fp-bps")
ax_focus_250.set_title("250ms focus: NH vs NF faithful vs IBL e300")
ax_focus_250.grid(True, alpha=0.3, axis="y")

fig.text(
    0.5,
    0.01,
    "Neuroformer uses rollout test fp-bps. The 20260321 +SC run is kept as the latest recorded reference only; "
    "session_idx was not actually injected during that run, so it is not valid evidence for conditioning gains.",
    ha="center",
    fontsize=8.5,
)

plt.tight_layout(rect=(0, 0.04, 1, 0.96))
plt.savefig(OUT_DIR / "06_benchmark_comparison.png", dpi=150)
plt.close()
print("Figure 6 saved: 06_benchmark_comparison.png (current benchmark references)")

print("\n" + "=" * 92)
print("PHASE1_V2 BENCHMARK REFERENCE SUMMARY")
print("=" * 92)
print(f"NeuroHorizon source: {nh_source}")
for window_ms in WINDOWS:
    if window_ms in nh_results:
        print(f"  NeuroHorizon {window_ms}ms: {nh_results[window_ms]:.4f}")
if ndt2_results:
    print("  Legacy NDT2-like (protocol-fix test):")
    for window_ms in WINDOWS:
        if window_ms in ndt2_results:
            print(f"    {window_ms}ms: {ndt2_results[window_ms]:.4f}")
print("  Neuroformer faithful rollout:")
for row in nf_results:
    note = " [session bug note]" if row["session_bug_note"] else ""
    print(
        f"    {row['label']}: rollout={row['rollout_fp_bps']:.4f}, "
        f"true_past={row['true_past_fp_bps']:.4f}{note}"
    )
print("  IBL-MtM faithful test fp-bps:")
for row in ibl_results:
    print(
        f"    {row['label']}: test={row['fp_bps']:.4f}, "
        f"best_valid={row['best_valid_fp_bps']:.4f}"
    )
print("=" * 92)
