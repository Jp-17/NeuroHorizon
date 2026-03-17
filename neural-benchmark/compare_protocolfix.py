#!/usr/bin/env python3
"""Compare legacy 1.8.3 outputs against protocol-fixed reevaluation results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/root/autodl-tmp/NeuroHorizon/results/logs")
FIG_ROOT = Path("/root/autodl-tmp/NeuroHorizon/results/figures/phase1_benchmark_protocolfix")
SUMMARY_ROOT = ROOT / "phase1_benchmark_protocolfix_comparison"
FIG_ROOT.mkdir(parents=True, exist_ok=True)
SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)

MODELS = ["ndt2", "ibl_mtm", "neuroformer"]
MODEL_LABELS = {
    "ndt2": "Legacy NDT2-like",
    "ibl_mtm": "Legacy IBL-MtM-like",
    "neuroformer": "Legacy Neuroformer-like",
}
WINDOWS = [250, 500, 1000]
COLORS = {
    "legacy": "#B0BEC5",
    "valid": "#1976D2",
    "test": "#D32F2F",
}


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def collect_results():
    collected = {}
    for model in MODELS:
        collected[model] = {}
        for window in WINDOWS:
            legacy_path = ROOT / f"phase1_benchmark_{model}_{window}ms" / "results.json"
            fixed_path = ROOT / f"phase1_benchmark_protocolfix_{model}_{window}ms" / "results.json"
            if not legacy_path.exists() or not fixed_path.exists():
                continue
            legacy = load_json(legacy_path)
            fixed = load_json(fixed_path)
            collected[model][window] = {
                "legacy_best_val_fp_bps": legacy.get("best_val_fp_bps"),
                "legacy_best_checkpoint_r2": fixed.get("legacy_reference", {})
                .get("legacy_best_checkpoint_metrics", {})
                .get("r2"),
                "fixed_valid_fp_bps": fixed["best_valid_metrics"]["continuous"]["fp_bps"],
                "fixed_valid_r2": fixed["best_valid_metrics"]["continuous"]["r2"],
                "fixed_test_fp_bps": fixed["test_metrics"]["continuous"]["fp_bps"],
                "fixed_test_r2": fixed["test_metrics"]["continuous"]["r2"],
                "fixed_test_psth_r2": fixed["test_metrics"]
                .get("trial_aligned", {})
                .get("per_neuron_psth_r2"),
                "delta_test_vs_legacy": fixed["test_metrics"]["continuous"]["fp_bps"]
                - legacy.get("best_val_fp_bps", 0.0),
            }
    return collected


def write_summary(collected):
    summary_path = SUMMARY_ROOT / "comparison.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(collected, handle, indent=2)

    lines = [
        "# Legacy 1.8.3 vs Protocol-Fix Reevaluation",
        "",
        "## Continuous fp-bps",
        "",
        "| 模型 | 窗口 | legacy best-val fp-bps | protocol-fix valid fp-bps | protocol-fix test fp-bps | test - legacy |",
        "|------|------|------------------------|---------------------------|--------------------------|---------------|",
    ]
    for model in MODELS:
        for window in WINDOWS:
            if window not in collected[model]:
                continue
            row = collected[model][window]
            lines.append(
                f"| {MODEL_LABELS[model]} | {window}ms | "
                f"{row['legacy_best_val_fp_bps']:.4f} | {row['fixed_valid_fp_bps']:.4f} | "
                f"{row['fixed_test_fp_bps']:.4f} | {row['delta_test_vs_legacy']:+.4f} |"
            )

    lines.extend(
        [
            "",
            "## Protocol-fix test metrics",
            "",
            "| 模型 | 窗口 | test fp-bps | test R² | test PSTH-R² |",
            "|------|------|-------------|---------|--------------|",
        ]
    )
    for model in MODELS:
        for window in WINDOWS:
            if window not in collected[model]:
                continue
            row = collected[model][window]
            psth = row["fixed_test_psth_r2"]
            psth_str = "N/A" if psth is None else f"{psth:.4f}"
            lines.append(
                f"| {MODEL_LABELS[model]} | {window}ms | {row['fixed_test_fp_bps']:.4f} | "
                f"{row['fixed_test_r2']:.4f} | {psth_str} |"
            )

    markdown_path = SUMMARY_ROOT / "comparison.md"
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_path, markdown_path


def plot_fpbps(collected):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    x = np.arange(len(MODELS))
    width = 0.24
    for axis, window in zip(axes, WINDOWS):
        legacy_vals = [collected[m][window]["legacy_best_val_fp_bps"] for m in MODELS]
        valid_vals = [collected[m][window]["fixed_valid_fp_bps"] for m in MODELS]
        test_vals = [collected[m][window]["fixed_test_fp_bps"] for m in MODELS]
        axis.bar(x - width, legacy_vals, width, label="legacy best-valid", color=COLORS["legacy"])
        axis.bar(x, valid_vals, width, label="protocol-fix valid", color=COLORS["valid"])
        axis.bar(x + width, test_vals, width, label="protocol-fix test", color=COLORS["test"])
        axis.set_title(f"{window}ms")
        axis.set_xticks(x)
        axis.set_xticklabels([MODEL_LABELS[m].replace("Legacy ", "") for m in MODELS], rotation=12)
        axis.grid(axis="y", alpha=0.25)
        if window == WINDOWS[0]:
            axis.set_ylabel("fp-bps")
    axes[0].legend(fontsize=8)
    plt.suptitle("Legacy 1.8.3 Validation Results vs Protocol-Fix Reevaluation", y=1.02)
    plt.tight_layout()
    out_path = FIG_ROOT / "legacy_vs_protocolfix_fpbps.png"
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()
    return out_path


def plot_psth(collected):
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(WINDOWS))
    width = 0.22
    for idx, model in enumerate(MODELS):
        vals = []
        for window in WINDOWS:
            vals.append(collected[model][window]["fixed_test_psth_r2"])
        ax.bar(x + (idx - 1) * width, vals, width, label=MODEL_LABELS[model])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{window}ms" for window in WINDOWS])
    ax.set_ylabel("test PSTH-R²")
    ax.set_title("Protocol-Fix Trial-Aligned Test PSTH-R²")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    plt.tight_layout()
    out_path = FIG_ROOT / "protocolfix_test_psth_r2.png"
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()
    return out_path


def main():
    collected = collect_results()
    summary_path, markdown_path = write_summary(collected)
    fp_path = plot_fpbps(collected)
    psth_path = plot_psth(collected)
    print("Saved:")
    print(f"  {summary_path}")
    print(f"  {markdown_path}")
    print(f"  {fp_path}")
    print(f"  {psth_path}")


if __name__ == "__main__":
    main()
