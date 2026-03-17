#!/usr/bin/env python3
"""Phase 1 v2 visualization: 5 figures for experiment analysis.

Generates:
1. fp-bps vs pred_window (continuous vs trial-aligned)
2. Per-bin fp-bps decay curves (6 conditions)
3. per-neuron PSTH-R2 heatmap (target_id x condition)
4. Continuous vs trial-aligned comparison bar chart
5. Training curves (val_loss / val_fp_bps vs epoch)

Usage:
    python scripts/analysis/neurohorizon/phase1_v2_visualize.py
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path("/root/autodl-tmp/NeuroHorizon/results/logs")
OUT_DIR = Path("/root/autodl-tmp/NeuroHorizon/results/figures/phase1_v2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONDITIONS = [
    ("phase1_v2_250ms_cont", "250ms-cont", 250, "continuous"),
    ("phase1_v2_250ms_trial", "250ms-trial", 250, "trial-aligned"),
    ("phase1_v2_500ms_cont", "500ms-cont", 500, "continuous"),
    ("phase1_v2_500ms_trial", "500ms-trial", 500, "trial-aligned"),
    ("phase1_v2_1000ms_cont", "1000ms-cont", 1000, "continuous"),
    ("phase1_v2_1000ms_trial", "1000ms-trial", 1000, "trial-aligned"),
]

COLORS = {
    "250ms-cont": "#1f77b4",
    "250ms-trial": "#1f77b4",
    "500ms-cont": "#ff7f0e",
    "500ms-trial": "#ff7f0e",
    "1000ms-cont": "#2ca02c",
    "1000ms-trial": "#2ca02c",
}

LINE_STYLES = {
    "continuous": "-",
    "trial-aligned": "--",
}


def candidate_result_dirs(dir_name: str):
    evalfix_dir = dir_name.replace("phase1_v2_", "phase1_v2_evalfix_", 1)
    return [evalfix_dir, dir_name]


def load_eval_results():
    """Load valid-split eval results for all conditions."""
    results = {}
    for dir_name, label, window_ms, mode in CONDITIONS:
        candidate_paths = []
        for result_dir in candidate_result_dirs(dir_name):
            candidate_paths.extend(
                [
                    BASE / result_dir / "lightning_logs" / "version_0" / "eval_v2_valid_results.json",
                    BASE / result_dir / "lightning_logs" / "eval_v2_valid_results.json",
                    BASE / result_dir / "lightning_logs" / "version_0" / "eval_v2_results.json",
                    BASE / result_dir / "lightning_logs" / "eval_v2_results.json",
                ]
            )
        eval_path = None
        for candidate in candidate_paths:
            if candidate.exists():
                eval_path = candidate
                break
        if eval_path is not None:
            with open(eval_path) as f:
                results[label] = json.load(f)
                results[label]["window_ms"] = window_ms
                results[label]["mode"] = mode
    return results


def load_training_curves():
    """Load metrics.csv for all conditions."""
    curves = {}
    for dir_name, label, window_ms, mode in CONDITIONS:
        metrics_path = None
        for result_dir in candidate_result_dirs(dir_name):
            candidate = BASE / result_dir / "lightning_logs" / "version_0" / "metrics.csv"
            if candidate.exists():
                metrics_path = candidate
                break
        if metrics_path is None:
            continue

        epochs = []
        val_loss = []
        val_fp_bps = []
        train_loss_epochs = []
        train_loss_vals = []

        with open(metrics_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Validation rows (have val/fp_bps)
                if row.get("val/fp_bps", ""):
                    epochs.append(int(row["epoch"]))
                    val_loss.append(float(row["val_loss"]))
                    val_fp_bps.append(float(row["val/fp_bps"]))

                # Training rows (have train_loss)
                if row.get("train_loss", ""):
                    ep = int(row["epoch"])
                    tl = float(row["train_loss"])
                    train_loss_epochs.append(ep)
                    train_loss_vals.append(tl)

        curves[label] = {
            "epochs": epochs,
            "val_loss": val_loss,
            "val_fp_bps": val_fp_bps,
            "train_loss_epochs": train_loss_epochs,
            "train_loss_vals": train_loss_vals,
        }
    return curves


def fig1_fpbps_vs_window(results):
    """Figure 1: fp-bps vs prediction window (continuous vs trial-aligned)."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    for mode, ls in [("continuous", "-o"), ("trial-aligned", "--s")]:
        windows = []
        bps_vals = []
        for label, res in sorted(results.items()):
            if res["mode"] == mode and "continuous" in res:
                windows.append(res["window_ms"])
                bps_vals.append(res["continuous"]["fp_bps"])

        if windows:
            ax.plot(windows, bps_vals, ls, label=mode, markersize=8, linewidth=2)

            # Annotate values
            for w, b in zip(windows, bps_vals):
                ax.annotate(f"{b:.3f}", (w, b), textcoords="offset points",
                           xytext=(0, 10), ha="center", fontsize=9)

    ax.set_xlabel("Prediction Window (ms)", fontsize=12)
    ax.set_ylabel("fp-bps", fontsize=12)
    ax.set_title("fp-bps vs Prediction Window", fontsize=14)
    ax.legend(fontsize=11)
    ax.set_xticks([250, 500, 1000])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "01_fpbps_vs_window.png", dpi=150)
    plt.close()
    print("Figure 1 saved: 01_fpbps_vs_window.png")


def fig2_perbin_decay(results):
    """Figure 2: Per-bin fp-bps decay curves."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for label, res in sorted(results.items()):
        if "continuous" not in res:
            continue
        per_bin = res["continuous"]["per_bin_fp_bps"]
        if not per_bin:
            continue

        bins = sorted(per_bin.keys(), key=int)
        bin_ms = [int(b) * 20 for b in bins]  # 20ms per bin
        bps_vals = [per_bin[b] for b in bins]

        mode = res["mode"]
        color = COLORS[label]
        ls = LINE_STYLES[mode]

        ax.plot(bin_ms, bps_vals, ls, color=color, label=label,
                linewidth=2, marker="o" if mode == "continuous" else "s",
                markersize=4, alpha=0.8)

    ax.set_xlabel("Time from prediction start (ms)", fontsize=12)
    ax.set_ylabel("fp-bps per bin", fontsize=12)
    ax.set_title("Per-bin fp-bps Decay Analysis", fontsize=14)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "02_perbin_fpbps_decay.png", dpi=150)
    plt.close()
    print("Figure 2 saved: 02_perbin_fpbps_decay.png")


def fig3_psth_heatmap(results):
    """Figure 3: per-neuron PSTH-R2 heatmap (target_id x condition)."""
    labels = []
    per_target_data = []

    for label in ["250ms-cont", "250ms-trial", "500ms-cont", "500ms-trial",
                   "1000ms-cont", "1000ms-trial"]:
        if label not in results:
            continue
        res = results[label]
        if "trial_aligned" not in res:
            continue
        trial = res["trial_aligned"]
        pt = (
            trial.get("per_target_per_neuron_psth_r2")
            or trial.get("per_target_psth_r2")
        )
        if not pt:
            continue

        labels.append(label)
        row = [pt.get(str(i), 0) for i in range(8)]
        per_target_data.append(row)

    if not per_target_data:
        print("Figure 3 skipped: no per-neuron PSTH-R2 data")
        return

    data = np.array(per_target_data)  # [n_conditions, 8]

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    im = ax.imshow(data.T, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(8))
    ax.set_yticklabels([f"Dir {i}" for i in range(8)])
    ax.set_title("per-neuron PSTH-R2 by Target Direction and Condition", fontsize=14)

    # Add text annotations
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            text_color = "white" if data[i, j] > 0.5 else "black"
            ax.text(i, j, f"{data[i, j]:.2f}", ha="center", va="center",
                   color=text_color, fontsize=9)

    plt.colorbar(im, ax=ax, label="per-neuron PSTH-R2")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "03_psth_r2_heatmap.png", dpi=150)
    plt.close()
    print("Figure 3 saved: 03_psth_r2_heatmap.png")


def fig4_cont_vs_trial(results):
    """Figure 4: Continuous vs trial-aligned comparison bar chart."""
    windows = [250, 500, 1000]
    metrics = ["fp_bps", "r2"]
    metric_labels = ["fp-bps", "R-squared"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, metric, mlabel in zip(axes, metrics, metric_labels):
        cont_vals = []
        trial_vals = []
        window_labels = []

        for w in windows:
            cont_label = f"{w}ms-cont"
            trial_label = f"{w}ms-trial"
            if cont_label in results and "continuous" in results[cont_label]:
                cont_vals.append(results[cont_label]["continuous"][metric])
            else:
                cont_vals.append(0)

            if trial_label in results and "continuous" in results[trial_label]:
                trial_vals.append(results[trial_label]["continuous"][metric])
            else:
                trial_vals.append(0)

            window_labels.append(f"{w}ms")

        x = np.arange(len(window_labels))
        width = 0.35

        bars1 = ax.bar(x - width/2, cont_vals, width, label="Continuous",
                       color="#1f77b4", alpha=0.8)
        bars2 = ax.bar(x + width/2, trial_vals, width, label="Trial-aligned",
                       color="#ff7f0e", alpha=0.8)

        # Value labels
        for bar in bars1:
            if bar.get_height() > 0:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                       f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
        for bar in bars2:
            if bar.get_height() > 0:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                       f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

        ax.set_xlabel("Prediction Window")
        ax.set_ylabel(mlabel)
        ax.set_title(f"{mlabel}: Continuous vs Trial-aligned")
        ax.set_xticks(x)
        ax.set_xticklabels(window_labels)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "04_cont_vs_trial.png", dpi=150)
    plt.close()
    print("Figure 4 saved: 04_cont_vs_trial.png")


def fig5_training_curves(curves):
    """Figure 5: Training curves for all 6 models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: val_loss
    for label, data in sorted(curves.items()):
        if data["epochs"]:
            mode = "trial-aligned" if "trial" in label else "continuous"
            color = COLORS[label]
            ls = LINE_STYLES[mode]
            axes[0].plot(data["epochs"], data["val_loss"], ls, color=color,
                        label=label, linewidth=1.5, alpha=0.8)

    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Validation Loss", fontsize=12)
    axes[0].set_title("Validation Loss vs Epoch", fontsize=14)
    axes[0].legend(fontsize=8, ncol=2)
    axes[0].grid(True, alpha=0.3)

    # Panel B: val_fp_bps
    for label, data in sorted(curves.items()):
        if data["epochs"]:
            mode = "trial-aligned" if "trial" in label else "continuous"
            color = COLORS[label]
            ls = LINE_STYLES[mode]
            axes[1].plot(data["epochs"], data["val_fp_bps"], ls, color=color,
                        label=label, linewidth=1.5, alpha=0.8)

    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("val/fp_bps", fontsize=12)
    axes[1].set_title("fp-bps vs Epoch", fontsize=14)
    axes[1].legend(fontsize=8, ncol=2)
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color="gray", linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "05_training_curves.png", dpi=150)
    plt.close()
    print("Figure 5 saved: 05_training_curves.png")


def main():
    print("Loading evaluation results...")
    results = load_eval_results()
    print(f"  Loaded {len(results)} conditions: {list(results.keys())}")

    print("Loading training curves...")
    curves = load_training_curves()
    print(f"  Loaded {len(curves)} conditions: {list(curves.keys())}")

    if not results:
        print("No evaluation results found. Run eval_phase1_v2.py first.")
        return

    print("\nGenerating figures...")
    fig1_fpbps_vs_window(results)
    fig2_perbin_decay(results)
    fig3_psth_heatmap(results)
    fig4_cont_vs_trial(results)
    fig5_training_curves(curves)

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Condition':15s} | {'fp-bps':>8s} | {'R2':>8s} | {'per-neuron PSTH-R2':>18s} | {'trial fp-bps':>12s}")
    print("-" * 65)
    for label in ["250ms-cont", "250ms-trial", "500ms-cont", "500ms-trial",
                   "1000ms-cont", "1000ms-trial"]:
        if label in results:
            r = results[label]
            c = r.get("continuous", {})
            t = r.get("trial_aligned", {})
            print(f"{label:15s} | {c.get('fp_bps', 0):8.4f} | {c.get('r2', 0):8.4f} | "
                  f"{t.get('per_neuron_psth_r2', t.get('psth_r2', 0)):18.4f} | {t.get('trial_fp_bps', 0):12.4f}")
        else:
            print(f"{label:15s} | {'--':>8s} | {'--':>8s} | {'--':>8s} | {'--':>12s}")
    print("=" * 80)

    print(f"\nAll figures saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
