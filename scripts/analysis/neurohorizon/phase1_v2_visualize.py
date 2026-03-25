#!/usr/bin/env python3
"""Phase 1 v2 visualization utility.

Generates the 5 canonical 1.3.4 figures for either the legacy protocol or
the 2026-03-17 evalfix reruns.

Examples:
    python scripts/analysis/neurohorizon/phase1_v2_visualize.py --protocol legacy
    python scripts/analysis/neurohorizon/phase1_v2_visualize.py --protocol evalfix --split valid \
        --out-dir /root/autodl-tmp/NeuroHorizon/results/figures/phase1_v2_evalfix
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/root/autodl-tmp/NeuroHorizon")
BASE = ROOT / "results" / "logs"
DEFAULT_OUT_DIRS = {
    "legacy": ROOT / "results" / "figures" / "phase1_v2",
    "evalfix": ROOT / "results" / "figures" / "phase1_v2_evalfix",
}
CONDITIONS = [
    {
        "legacy_dir": "phase1_v2_250ms_cont",
        "evalfix_dir": "phase1_v2_evalfix_250ms_cont",
        "label": "250ms-cont",
        "window_ms": 250,
        "mode": "continuous",
    },
    {
        "legacy_dir": "phase1_v2_250ms_trial",
        "evalfix_dir": "phase1_v2_evalfix_250ms_trial",
        "label": "250ms-trial",
        "window_ms": 250,
        "mode": "trial-aligned",
    },
    {
        "legacy_dir": "phase1_v2_500ms_cont",
        "evalfix_dir": "phase1_v2_evalfix_500ms_cont",
        "label": "500ms-cont",
        "window_ms": 500,
        "mode": "continuous",
    },
    {
        "legacy_dir": "phase1_v2_500ms_trial",
        "evalfix_dir": "phase1_v2_evalfix_500ms_trial",
        "label": "500ms-trial",
        "window_ms": 500,
        "mode": "trial-aligned",
    },
    {
        "legacy_dir": "phase1_v2_1000ms_cont",
        "evalfix_dir": "phase1_v2_evalfix_1000ms_cont",
        "label": "1000ms-cont",
        "window_ms": 1000,
        "mode": "continuous",
    },
    {
        "legacy_dir": "phase1_v2_1000ms_trial",
        "evalfix_dir": "phase1_v2_evalfix_1000ms_trial",
        "label": "1000ms-trial",
        "window_ms": 1000,
        "mode": "trial-aligned",
    },
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
MARKERS = {
    "continuous": "o",
    "trial-aligned": "s",
}
MODE_DISPLAY_NAMES = {
    "continuous": "continuous-trained",
    "trial-aligned": "trial-aligned-trained",
}
MODE_SERIES_COLORS = {
    "continuous": "#1f77b4",
    "trial-aligned": "#ff7f0e",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        choices=["legacy", "evalfix"],
        default="legacy",
        help="Which result family to visualize.",
    )
    parser.add_argument(
        "--split",
        choices=["valid", "test"],
        default="valid",
        help="Evalfix split to visualize. Ignored for legacy results.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Optional output directory. Defaults to the protocol-specific figure directory.",
    )
    return parser.parse_args()


def protocol_tag(protocol: str, split: str) -> str:
    if protocol == "legacy":
        return "Legacy Protocol"
    return f"Evalfix ({split})"


def resolve_out_dir(args) -> Path:
    out_dir = args.out_dir or DEFAULT_OUT_DIRS[args.protocol]
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def result_json_candidates(condition, protocol: str, split: str):
    dir_name = condition["legacy_dir"] if protocol == "legacy" else condition["evalfix_dir"]
    if protocol == "legacy":
        return [
            BASE / dir_name / "lightning_logs" / "version_0" / "eval_v2_results.json",
            BASE / dir_name / "lightning_logs" / "eval_v2_results.json",
        ]
    return [
        BASE / dir_name / "lightning_logs" / "version_0" / f"eval_v2_{split}_results.json",
        BASE / dir_name / "lightning_logs" / f"eval_v2_{split}_results.json",
    ]


def metrics_candidates(condition, protocol: str):
    dir_name = condition["legacy_dir"] if protocol == "legacy" else condition["evalfix_dir"]
    return [
        BASE / dir_name / "lightning_logs" / "version_0" / "metrics.csv",
        BASE / dir_name / "lightning_logs" / "metrics.csv",
    ]


def load_first_json(candidates):
    for candidate in candidates:
        if candidate.exists():
            with candidate.open() as handle:
                return json.load(handle)
    return None


def load_first_path(candidates):
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_eval_results(protocol: str, split: str):
    results = {}
    for condition in CONDITIONS:
        payload = load_first_json(result_json_candidates(condition, protocol, split))
        if payload is None:
            continue
        payload["window_ms"] = condition["window_ms"]
        payload["mode"] = condition["mode"]
        results[condition["label"]] = payload
    return results


def load_training_curves(protocol: str):
    curves = {}
    for condition in CONDITIONS:
        metrics_path = load_first_path(metrics_candidates(condition, protocol))
        if metrics_path is None:
            continue

        epochs = []
        val_loss = []
        val_fp_bps = []
        train_loss_epochs = []
        train_loss_vals = []

        with metrics_path.open() as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                epoch_value = row.get("epoch")
                if not epoch_value:
                    continue
                epoch = int(float(epoch_value))

                if row.get("val/fp_bps"):
                    epochs.append(epoch)
                    val_loss.append(float(row["val_loss"]))
                    val_fp_bps.append(float(row["val/fp_bps"]))

                if row.get("train_loss"):
                    train_loss_epochs.append(epoch)
                    train_loss_vals.append(float(row["train_loss"]))

        curves[condition["label"]] = {
            "epochs": epochs,
            "val_loss": val_loss,
            "val_fp_bps": val_fp_bps,
            "train_loss_epochs": train_loss_epochs,
            "train_loss_vals": train_loss_vals,
        }
    return curves


def mode_output_path(out_dir: Path, base_name: str, mode: str) -> Path:
    path = Path(base_name)
    if mode == "continuous":
        return out_dir / path.name
    return out_dir / f"{path.stem}_trial_aligned{path.suffix}"


def filter_results_by_mode(results, mode: str):
    return {
        label: payload
        for label, payload in results.items()
        if payload.get("mode") == mode
    }


def filter_curves_by_mode(curves, mode: str):
    mode_token = "trial" if mode == "trial-aligned" else "cont"
    return {
        label: payload
        for label, payload in curves.items()
        if label.endswith(mode_token)
    }


def annotate_point_series(ax, xs, ys):
    for x_val, y_val in zip(xs, ys):
        offset = 10 if y_val >= 0 else -16
        ax.annotate(
            f"{y_val:.3f}",
            (x_val, y_val),
            textcoords="offset points",
            xytext=(0, offset),
            ha="center",
            fontsize=9,
        )


def annotate_bars(ax, bars):
    for bar in bars:
        height = bar.get_height()
        if abs(height) < 1e-8:
            continue
        offset = 0.01 if height >= 0 else -0.02
        va = "bottom" if height >= 0 else "top"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + offset,
            f"{height:.3f}",
            ha="center",
            va=va,
            fontsize=9,
        )


def fig1_fpbps_vs_window(results, out_dir: Path, tag: str, mode: str):
    mode_results = filter_results_by_mode(results, mode)
    if not mode_results:
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    windows = []
    values = []
    for _, payload in sorted(mode_results.items(), key=lambda item: item[1]["window_ms"]):
        if "continuous" not in payload:
            continue
        windows.append(payload["window_ms"])
        values.append(payload["continuous"].get("fp_bps", 0.0))

    if windows:
        ax.plot(
            windows,
            values,
            color=MODE_SERIES_COLORS[mode],
            linestyle=LINE_STYLES[mode],
            marker=MARKERS[mode],
            label=MODE_DISPLAY_NAMES[mode],
            markersize=8,
            linewidth=2.5,
        )
        annotate_point_series(ax, windows, values)

    ax.set_xlabel("Prediction Window (ms)", fontsize=12)
    ax.set_ylabel("fp-bps", fontsize=12)
    ax.set_title(f"fp-bps vs Prediction Window ({MODE_DISPLAY_NAMES[mode]}, {tag})", fontsize=14)
    ax.legend(fontsize=11)
    ax.set_xticks([250, 500, 1000])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(mode_output_path(out_dir, "01_fpbps_vs_window.png", mode), dpi=150)
    plt.close()


def fig2_perbin_decay(results, out_dir: Path, tag: str, mode: str):
    mode_results = filter_results_by_mode(results, mode)
    if not mode_results:
        return

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for label, payload in sorted(mode_results.items(), key=lambda item: item[1]["window_ms"]):
        if "continuous" not in payload:
            continue
        per_bin = payload["continuous"].get("per_bin_fp_bps") or {}
        if not per_bin:
            continue

        bins = sorted(per_bin.keys(), key=int)
        bin_ms = [int(bin_key) * 20 for bin_key in bins]
        values = [per_bin[bin_key] for bin_key in bins]
        ax.plot(
            bin_ms,
            values,
            LINE_STYLES[payload["mode"]],
            color=COLORS[label],
            label=label,
            linewidth=2,
            marker=MARKERS[payload["mode"]],
            markersize=4,
            alpha=0.85,
        )

    ax.set_xlabel("Time from prediction start (ms)", fontsize=12)
    ax.set_ylabel("fp-bps per bin", fontsize=12)
    ax.set_title(f"Per-bin fp-bps Decay ({MODE_DISPLAY_NAMES[mode]}, {tag})", fontsize=14)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(mode_output_path(out_dir, "02_perbin_fpbps_decay.png", mode), dpi=150)
    plt.close()


def fig3_psth_heatmap(results, out_dir: Path, tag: str, mode: str):
    labels = []
    per_target_rows = []

    for label in [
        "250ms-cont",
        "250ms-trial",
        "500ms-cont",
        "500ms-trial",
        "1000ms-cont",
        "1000ms-trial",
    ]:
        payload = results.get(label)
        if payload is None or payload.get("mode") != mode:
            continue
        trial_payload = payload.get("trial_aligned") or {}
        per_target = trial_payload.get("per_target_per_neuron_psth_r2") or trial_payload.get("per_target_psth_r2")
        if not per_target:
            continue
        labels.append(label)
        per_target_rows.append([per_target.get(str(target_id), 0.0) for target_id in range(8)])

    if not per_target_rows:
        print("Figure 3 skipped: no per-target PSTH data found")
        return

    data = np.array(per_target_rows)
    vmax = max(0.7, float(np.max(data)))

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    image = ax.imshow(data.T, aspect="auto", cmap="YlOrRd", vmin=0.0, vmax=vmax)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(8))
    ax.set_yticklabels([f"Dir {idx}" for idx in range(8)])
    ax.set_title(
        f"per-neuron PSTH-R2 by Direction ({MODE_DISPLAY_NAMES[mode]}, {tag})",
        fontsize=14,
    )

    for row_idx in range(data.shape[0]):
        for col_idx in range(data.shape[1]):
            value = data[row_idx, col_idx]
            text_color = "white" if value > (0.55 * vmax) else "black"
            ax.text(row_idx, col_idx, f"{value:.2f}", ha="center", va="center", color=text_color, fontsize=9)

    plt.colorbar(image, ax=ax, label="per-neuron PSTH-R2")
    plt.tight_layout()
    plt.savefig(mode_output_path(out_dir, "03_psth_r2_heatmap.png", mode), dpi=150)
    plt.close()


def fig4_cont_vs_trial(results, out_dir: Path, tag: str, mode: str):
    windows = [250, 500, 1000]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    metric_specs = [("fp_bps", "fp-bps"), ("r2", "R-squared")]

    for axis, (metric_key, metric_label) in zip(axes, metric_specs):
        values = []
        for window_ms in windows:
            label = f"{window_ms}ms-{'cont' if mode == 'continuous' else 'trial'}"
            values.append(results.get(label, {}).get("continuous", {}).get(metric_key, 0.0))

        axis.plot(
            windows,
            values,
            color=MODE_SERIES_COLORS[mode],
            linestyle=LINE_STYLES[mode],
            marker=MARKERS[mode],
            linewidth=2.5,
            markersize=8,
            label=MODE_DISPLAY_NAMES[mode],
        )
        annotate_point_series(axis, windows, values)

        axis.set_xlabel("Prediction Window (ms)")
        axis.set_ylabel(metric_label)
        axis.set_title(f"{metric_label} vs Prediction Window")
        axis.set_xticks(windows)
        axis.legend()
        axis.grid(True, alpha=0.3)
        axis.axhline(y=0, color="gray", linestyle=":", alpha=0.5)

    fig.suptitle(f"Window Sweep Summary ({MODE_DISPLAY_NAMES[mode]}, {tag})", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(mode_output_path(out_dir, "04_cont_vs_trial.png", mode), dpi=150)
    plt.close()


def fig5_training_curves(curves, out_dir: Path, tag: str, mode: str):
    mode_curves = filter_curves_by_mode(curves, mode)
    if not mode_curves:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for label, curve in sorted(mode_curves.items(), key=lambda item: item[0]):
        if not curve["epochs"]:
            continue
        axes[0].plot(
            curve["epochs"],
            curve["val_loss"],
            LINE_STYLES[mode],
            color=COLORS[label],
            label=label,
            linewidth=1.5,
            alpha=0.8,
        )
        axes[1].plot(
            curve["epochs"],
            curve["val_fp_bps"],
            LINE_STYLES[mode],
            color=COLORS[label],
            label=label,
            linewidth=1.5,
            alpha=0.8,
        )

    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Validation Loss", fontsize=12)
    axes[0].set_title("Validation Loss vs Epoch", fontsize=14)
    axes[0].legend(fontsize=8, ncol=2)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("val/fp_bps", fontsize=12)
    axes[1].set_title("fp-bps vs Epoch", fontsize=14)
    axes[1].legend(fontsize=8, ncol=2)
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color="gray", linestyle=":", alpha=0.5)

    fig.suptitle(f"Training Curves ({MODE_DISPLAY_NAMES[mode]}, {tag})", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(mode_output_path(out_dir, "05_training_curves.png", mode), dpi=150)
    plt.close()


def print_summary(results):
    print("\n" + "=" * 88)
    print("SUMMARY TABLE")
    print("=" * 88)
    print(f"{'Condition':15s} | {'fp-bps':>8s} | {'R2':>8s} | {'per-neuron PSTH-R2':>18s} | {'trial fp-bps':>12s}")
    print("-" * 88)
    ordered_labels = [
        "250ms-cont",
        "250ms-trial",
        "500ms-cont",
        "500ms-trial",
        "1000ms-cont",
        "1000ms-trial",
    ]
    for label in ordered_labels:
        payload = results.get(label)
        if payload is None:
            print(f"{label:15s} | {'--':>8s} | {'--':>8s} | {'--':>18s} | {'--':>12s}")
            continue
        cont = payload.get("continuous") or {}
        trial = payload.get("trial_aligned") or {}
        psth_r2 = trial.get("per_neuron_psth_r2", trial.get("psth_r2", 0.0))
        print(
            f"{label:15s} | {cont.get('fp_bps', 0.0):8.4f} | {cont.get('r2', 0.0):8.4f} | {psth_r2:18.4f} | {trial.get('trial_fp_bps', 0.0):12.4f}"
        )
    print("=" * 88)


def main():
    args = parse_args()
    out_dir = resolve_out_dir(args)
    tag = protocol_tag(args.protocol, args.split)

    print(f"Loading evaluation results for {tag}...")
    results = load_eval_results(args.protocol, args.split)
    print(f"  Loaded {len(results)} conditions: {list(results.keys())}")

    print("Loading training curves...")
    curves = load_training_curves(args.protocol)
    print(f"  Loaded {len(curves)} conditions: {list(curves.keys())}")

    if not results:
        raise SystemExit("No evaluation results found. Run eval_phase1_v2.py first.")

    for mode in ("continuous", "trial-aligned"):
        fig1_fpbps_vs_window(results, out_dir, tag, mode)
        fig2_perbin_decay(results, out_dir, tag, mode)
        fig3_psth_heatmap(results, out_dir, tag, mode)
        fig4_cont_vs_trial(results, out_dir, tag, mode)
        fig5_training_curves(curves, out_dir, tag, mode)
    print_summary(results)
    print(f"\nAll figures saved to: {out_dir}")


if __name__ == "__main__":
    main()
