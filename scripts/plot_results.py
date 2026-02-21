"""Generate publication-quality figures for NeuroHorizon results.

Creates:
1. Training curves (loss, bits/spike vs epoch)
2. Cross-session generalization comparison
3. Ablation results table
4. Prediction quality visualization (predicted rates vs ground truth)

Usage:
    conda run -n poyo python scripts/plot_results.py \
        --metrics logs/neurohorizon/lightning_logs/version_X/metrics.csv \
        --output figures/
"""

import argparse
import csv
import json
import logging
import math
from pathlib import Path

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
                "val_loss", "val_bits_per_spike", "val_r2",
                "val_fr_corr", "val_co_bps",
            ]:
                if row.get(k) and row[k]:
                    v = float(row[k])
                    if not math.isnan(v):
                        val_metrics.setdefault(k, {})[epoch] = v

    # Average train losses per epoch
    avg_train = {e: sum(v) / len(v) for e, v in train_losses.items()}
    return avg_train, val_metrics


def plot_training_curves(train_losses, val_metrics, output_dir, name="NH"):
    """Plot training loss and validation metrics."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping plots")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Plot 1: Training loss
    epochs = sorted(train_losses.keys())
    losses = [train_losses[e] for e in epochs]
    axes[0].plot(epochs, losses, "b-", linewidth=1.5, label="Train")
    if "val_loss" in val_metrics:
        val_epochs = sorted(val_metrics["val_loss"].keys())
        val_losses = [val_metrics["val_loss"][e] for e in val_epochs]
        axes[0].plot(val_epochs, val_losses, "ro-", markersize=6, label="Validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{name} Training Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Bits per spike
    if "val_bits_per_spike" in val_metrics:
        val_epochs = sorted(val_metrics["val_bits_per_spike"].keys())
        bps = [val_metrics["val_bits_per_spike"][e] for e in val_epochs]
        axes[1].plot(val_epochs, bps, "go-", markersize=6, linewidth=2)
        axes[1].axhline(y=0, color="k", linestyle="--", alpha=0.5, label="Null model")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Bits per spike")
        axes[1].set_title(f"{name} Bits per Spike")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    elif "val_r2" in val_metrics:
        val_epochs = sorted(val_metrics["val_r2"].keys())
        r2 = [val_metrics["val_r2"][e] for e in val_epochs]
        axes[1].plot(val_epochs, r2, "go-", markersize=6, linewidth=2)
        axes[1].axhline(y=0, color="k", linestyle="--", alpha=0.5)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("R²")
        axes[1].set_title(f"{name} Validation R²")
        axes[1].grid(True, alpha=0.3)

    # Plot 3: Learning rate
    axes[2].text(
        0.5, 0.5, "LR schedule\n(not plotted)",
        ha="center", va="center", transform=axes[2].transAxes,
        fontsize=12, color="gray",
    )
    axes[2].set_title("Learning Rate")

    plt.tight_layout()
    fig.savefig(output_dir / f"{name.lower()}_training_curves.png", dpi=150)
    plt.close()
    logger.info(f"Saved {name} training curves")


def plot_cross_session_comparison(results_path, output_dir):
    """Plot cross-session generalization results."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    with open(results_path) as f:
        results = json.load(f)

    sessions = results["sessions"]
    train_sessions = {k: v for k, v in sessions.items() if v["is_train_session"]}
    test_sessions = {k: v for k, v in sessions.items() if not v["is_train_session"]}

    fig, ax = plt.subplots(figsize=(10, 5))

    # Bar chart
    all_names = []
    all_bps = []
    all_colors = []

    for sid, metrics in sorted(train_sessions.items()):
        all_names.append(sid[:8])
        all_bps.append(metrics["bits_per_spike_mean"])
        all_colors.append("steelblue")

    for sid, metrics in sorted(test_sessions.items()):
        all_names.append(sid[:8])
        all_bps.append(metrics["bits_per_spike_mean"])
        all_colors.append("coral")

    ax.bar(range(len(all_names)), all_bps, color=all_colors)
    ax.set_xticks(range(len(all_names)))
    ax.set_xticklabels(all_names, rotation=45, ha="right", fontsize=8)
    ax.axhline(y=0, color="k", linestyle="--", alpha=0.5)
    ax.set_ylabel("Bits per spike")
    ax.set_title("Cross-Session Generalization")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="steelblue", label="Train sessions"),
        Patch(facecolor="coral", label="Test sessions (zero-shot)"),
    ]
    ax.legend(handles=legend_elements)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(output_dir / "cross_session_comparison.png", dpi=150)
    plt.close()
    logger.info("Saved cross-session comparison plot")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        help="Paths to metrics.csv files",
    )
    parser.add_argument(
        "--cross-session-results",
        type=str,
        default=None,
        help="Path to cross_session_results.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./figures",
    )
    parser.add_argument(
        "--names",
        type=str,
        nargs="+",
        default=None,
        help="Names for each metrics file",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.metrics:
        names = args.names or [f"Model{i}" for i in range(len(args.metrics))]
        for metrics_path, name in zip(args.metrics, names):
            logger.info(f"Loading {name} from {metrics_path}")
            train_losses, val_metrics = load_metrics(metrics_path)
            if train_losses:
                plot_training_curves(train_losses, val_metrics, output_dir, name)

    if args.cross_session_results:
        plot_cross_session_comparison(args.cross_session_results, output_dir)

    logger.info(f"All figures saved to {output_dir}")


if __name__ == "__main__":
    main()
