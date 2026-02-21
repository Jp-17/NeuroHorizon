#!/usr/bin/env python3
"""Quick training status check for NeuroHorizon and POYO baseline."""

import csv
import math
import os
import sys

import numpy as np


def analyze_metrics(csv_path, name):
    """Analyze a metrics CSV file and print summary."""
    if not os.path.exists(csv_path):
        print(f"  {name}: No metrics file found at {csv_path}")
        return

    losses = {}
    val_losses = {}
    val_metrics = {}
    epoch_times = {}

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            epoch = int(row["epoch"]) if row.get("epoch") and row["epoch"] else None

            if row.get("train_loss") and row["train_loss"]:
                v = float(row["train_loss"])
                if epoch is not None and not math.isnan(v):
                    losses.setdefault(epoch, []).append(v)

            if row.get("epoch_time") and row["epoch_time"]:
                if epoch is not None:
                    epoch_times[epoch] = float(row["epoch_time"])

            # Validation metrics
            for key in ["val_loss", "val_bits_per_spike", "val_fr_corr", "val_r2"]:
                if row.get(key) and row[key]:
                    v = float(row[key])
                    if not math.isnan(v):
                        val_metrics.setdefault(key, {})[epoch] = v

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    if not losses:
        print("  No training loss data found.")
        return

    max_epoch = max(losses.keys())
    total_steps = sum(len(v) for v in losses.values())
    nan_count = 0

    # Re-scan for NaN count
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("train_loss") and row["train_loss"]:
                v = float(row["train_loss"])
                if math.isnan(v) or math.isinf(v):
                    nan_count += 1

    print(f"  Epochs: 0-{max_epoch}, Total steps: {total_steps}")
    if nan_count > 0:
        print(f"  WARNING: {nan_count} NaN/inf losses ({nan_count/(total_steps+nan_count)*100:.1f}%)")

    print(f"\n  Per-epoch training loss:")
    for e in sorted(losses.keys()):
        d = np.array(losses[e])
        time_str = f", time={epoch_times[e]:.0f}s" if e in epoch_times else ""
        done_str = f" ({len(d)} steps)" if len(d) < 1746 else ""
        print(f"    Epoch {e:3d}: avg={np.mean(d):.4f}, median={np.median(d):.4f}, min={np.min(d):.4f}{time_str}{done_str}")

    if val_metrics:
        print(f"\n  Validation metrics:")
        for key, vals in val_metrics.items():
            for e in sorted(vals.keys()):
                print(f"    Epoch {e:3d}: {key}={vals[e]:.4f}")

    # ETA
    if epoch_times:
        avg_time = np.mean(list(epoch_times.values()))
        target_epochs = 100 if "neurohorizon" in name.lower() else 200
        remaining = target_epochs - max_epoch - 1
        if remaining > 0:
            eta_h = (remaining * avg_time) / 3600
            print(f"\n  ETA: ~{eta_h:.1f} hours ({remaining} epochs remaining, ~{avg_time:.0f}s/epoch)")


if __name__ == "__main__":
    base = "/root/autodl-tmp/NeuroHorizon"

    analyze_metrics(
        f"{base}/logs/neurohorizon/lightning_logs/version_5/metrics.csv",
        "NeuroHorizon (100 epochs)"
    )

    # Find latest POYO baseline version
    poyo_dir = f"{base}/logs/poyo_baseline/lightning_logs"
    if os.path.exists(poyo_dir):
        versions = sorted([d for d in os.listdir(poyo_dir) if d.startswith("version_")],
                         key=lambda x: int(x.split("_")[1]))
        if versions:
            latest = versions[-1]
            analyze_metrics(
                f"{poyo_dir}/{latest}/metrics.csv",
                f"POYO Baseline (200 epochs, {latest})"
            )

    # GPU status
    print(f"\n{'='*60}")
    print("  GPU Status")
    print(f"{'='*60}")
    os.system("nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader")

    # Process count
    import subprocess
    nh = subprocess.run(["bash", "-c", "ps aux | grep 'train.py.*neurohorizon\\|train_neurohorizon' | grep -v grep | wc -l"],
                       capture_output=True, text=True)
    pb = subprocess.run(["bash", "-c", "ps aux | grep train_baseline | grep -v grep | wc -l"],
                       capture_output=True, text=True)
    print(f"  NeuroHorizon processes: {nh.stdout.strip()}")
    print(f"  POYO baseline processes: {pb.stdout.strip()}")
