"""Comprehensive evaluation and comparison of NeuroHorizon models.

Runs evaluation on all trained models and generates comparison tables/plots.
Designed to be run after training completes.

Usage:
    conda run -n poyo python scripts/compare_models.py \
        --v1-ckpt logs/neurohorizon/lightning_logs/version_5/checkpoints/last.ckpt \
        --v2-ckpt logs/neurohorizon_v2_ibl/lightning_logs/version_0/checkpoints/last.ckpt \
        --output results/

    conda run -n poyo python scripts/compare_models.py \
        --v1-metrics logs/neurohorizon/lightning_logs/version_5/metrics.csv \
        --v2-metrics logs/neurohorizon_v2_ibl/lightning_logs/version_0/metrics.csv \
        --poyo-metrics logs/poyo_baseline/lightning_logs/version_20/metrics.csv \
        --output results/
"""

import argparse
import csv
import json
import logging
import math
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def load_metrics(csv_path):
    """Load training metrics from Lightning CSV log."""
    train_losses = {}
    val_metrics = {}

    with open(csv_path) as f:
        for row in csv.DictReader(f):
            epoch = int(row["epoch"]) if row.get("epoch") and row["epoch"] else None
            if row.get("train_loss") and row["train_loss"]:
                v = float(row["train_loss"])
                if epoch is not None and not math.isnan(v):
                    train_losses.setdefault(epoch, []).append(v)
            for k in [
                "val_loss",
                "val_bits_per_spike",
                "val_r2",
                "val_fr_corr",
                "val_co_bps",
            ]:
                if row.get(k) and row[k]:
                    v = float(row[k])
                    if not math.isnan(v):
                        val_metrics.setdefault(k, {})[epoch] = v

    avg_train = {e: sum(v) / len(v) for e, v in train_losses.items()}
    return avg_train, val_metrics


def compare_training_curves(metrics_dict, output_dir):
    """Plot training curves for multiple models."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping plots")
        return

    colors = ["steelblue", "coral", "green", "purple", "orange"]

    # Plot 1: Validation loss comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, (name, (train_losses, val_metrics)) in enumerate(metrics_dict.items()):
        color = colors[idx % len(colors)]

        if "val_loss" in val_metrics:
            epochs = sorted(val_metrics["val_loss"].keys())
            vals = [val_metrics["val_loss"][e] for e in epochs]
            axes[0].plot(epochs, vals, "o-", color=color, label=name, markersize=5)

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Validation Loss")
    axes[0].set_title("Validation Loss Comparison")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Bits per spike
    for idx, (name, (train_losses, val_metrics)) in enumerate(metrics_dict.items()):
        color = colors[idx % len(colors)]
        if "val_bits_per_spike" in val_metrics:
            epochs = sorted(val_metrics["val_bits_per_spike"].keys())
            vals = [val_metrics["val_bits_per_spike"][e] for e in epochs]
            axes[1].plot(epochs, vals, "o-", color=color, label=name, markersize=5)

    axes[1].axhline(y=0, color="k", linestyle="--", alpha=0.5, label="Null model")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Bits per spike")
    axes[1].set_title("Bits per Spike Comparison")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Plot 3: R² (for models that have it)
    has_r2 = False
    for idx, (name, (train_losses, val_metrics)) in enumerate(metrics_dict.items()):
        color = colors[idx % len(colors)]
        if "val_r2" in val_metrics:
            epochs = sorted(val_metrics["val_r2"].keys())
            vals = [val_metrics["val_r2"][e] for e in epochs]
            axes[2].plot(epochs, vals, "o-", color=color, label=name, markersize=5)
            has_r2 = True

    if has_r2:
        axes[2].axhline(y=0, color="k", linestyle="--", alpha=0.5)
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("R²")
        axes[2].set_title("R² Comparison")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].text(
            0.5,
            0.5,
            "R² not available\nfor these models",
            ha="center",
            va="center",
            transform=axes[2].transAxes,
            fontsize=12,
            color="gray",
        )
        axes[2].set_title("R²")

    plt.tight_layout()
    fig.savefig(output_dir / "model_comparison.png", dpi=150)
    plt.close()
    logger.info("Saved model comparison plot")


def generate_summary_table(metrics_dict, output_dir):
    """Generate a summary table of best metrics for each model."""
    rows = []
    for name, (train_losses, val_metrics) in metrics_dict.items():
        row = {"Model": name}

        # Best validation loss
        if "val_loss" in val_metrics:
            best_epoch = min(val_metrics["val_loss"], key=val_metrics["val_loss"].get)
            row["Best val_loss"] = f"{val_metrics['val_loss'][best_epoch]:.4f}"
            row["Best epoch (loss)"] = best_epoch

        # Best bits/spike
        if "val_bits_per_spike" in val_metrics:
            best_epoch = max(
                val_metrics["val_bits_per_spike"],
                key=val_metrics["val_bits_per_spike"].get,
            )
            row["Best bits/spike"] = (
                f"{val_metrics['val_bits_per_spike'][best_epoch]:.4f}"
            )
            row["Best epoch (bps)"] = best_epoch

        # Best R²
        if "val_r2" in val_metrics:
            best_epoch = max(val_metrics["val_r2"], key=val_metrics["val_r2"].get)
            row["Best R²"] = f"{val_metrics['val_r2'][best_epoch]:.4f}"
            row["Best epoch (R²)"] = best_epoch

        # Last epoch metrics
        if "val_loss" in val_metrics:
            last_epoch = max(val_metrics["val_loss"].keys())
            row["Final val_loss"] = f"{val_metrics['val_loss'][last_epoch]:.4f}"
            if "val_bits_per_spike" in val_metrics and last_epoch in val_metrics["val_bits_per_spike"]:
                row["Final bits/spike"] = f"{val_metrics['val_bits_per_spike'][last_epoch]:.4f}"

        rows.append(row)

    # Print table
    if rows:
        headers = list(rows[0].keys())
        col_widths = {h: max(len(h), max(len(str(r.get(h, ""))) for r in rows)) for h in headers}

        header_line = " | ".join(h.ljust(col_widths[h]) for h in headers)
        separator = "-+-".join("-" * col_widths[h] for h in headers)

        logger.info("\n" + "=" * len(header_line))
        logger.info("MODEL COMPARISON SUMMARY")
        logger.info("=" * len(header_line))
        logger.info(header_line)
        logger.info(separator)
        for row in rows:
            logger.info(
                " | ".join(str(row.get(h, "")).ljust(col_widths[h]) for h in headers)
            )
        logger.info("")

    # Save as JSON
    with open(output_dir / "comparison_summary.json", "w") as f:
        json.dump(rows, f, indent=2)
    logger.info(f"Saved comparison summary to {output_dir / 'comparison_summary.json'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1-metrics", type=str, help="NH v1 metrics.csv path")
    parser.add_argument("--v2-metrics", type=str, help="NH v2 metrics.csv path")
    parser.add_argument("--v2-beh-metrics", type=str, help="NH v2 behavior metrics.csv")
    parser.add_argument("--v2-mm-metrics", type=str, help="NH v2 multimodal metrics.csv")
    parser.add_argument("--poyo-metrics", type=str, help="POYO baseline metrics.csv")
    parser.add_argument("--output", type=str, default="./results")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_dict = {}

    metric_args = {
        "NH v1 (unnormalized)": args.v1_metrics,
        "NH v2 (normalized)": args.v2_metrics,
        "NH v2 + behavior": args.v2_beh_metrics,
        "NH v2 + multimodal": args.v2_mm_metrics,
        "POYO baseline": args.poyo_metrics,
    }

    for name, path in metric_args.items():
        if path and Path(path).exists():
            logger.info(f"Loading {name} from {path}")
            metrics_dict[name] = load_metrics(path)

    if not metrics_dict:
        # Try auto-detecting from common paths
        auto_paths = {
            "NH v1": "logs/neurohorizon/lightning_logs/version_5/metrics.csv",
            "POYO baseline": "logs/poyo_baseline/lightning_logs/version_20/metrics.csv",
        }
        for name, path in auto_paths.items():
            if Path(path).exists():
                logger.info(f"Auto-detected {name} at {path}")
                metrics_dict[name] = load_metrics(path)

    if not metrics_dict:
        logger.error("No metrics files found")
        return

    compare_training_curves(metrics_dict, output_dir)
    generate_summary_table(metrics_dict, output_dir)


if __name__ == "__main__":
    main()
