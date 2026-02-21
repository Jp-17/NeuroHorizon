#!/usr/bin/env python3
"""Post-training evaluation for NeuroHorizon.

Loads a trained checkpoint, runs evaluation on validation data,
computes detailed metrics, and generates visualizations.

Usage:
    conda run -n poyo python scripts/evaluate_neurohorizon.py \
        --ckpt logs/neurohorizon/lightning_logs/version_5/checkpoints/last.ckpt \
        --data-dir /path/to/ibl_processed \
        --output-dir results/neurohorizon_eval
"""

import argparse
import json
import logging
import os
from collections import defaultdict
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from temporaldata import Data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# Add project root to path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from torch_brain.models import NeuroHorizon
from torch_brain.utils.neurohorizon_metrics import (
    bits_per_spike,
    firing_rate_correlation,
    poisson_log_likelihood,
    r2_binned_counts,
)


def load_model(ckpt_path, device="cuda"):
    """Load NeuroHorizon model from Lightning checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Extract model config from hyperparameters
    hparams = ckpt.get("hyper_parameters", {})
    model_cfg = hparams.get("model", {})

    # Remove non-model keys
    model_cfg.pop("_target_", None)

    # Create model
    model = NeuroHorizon(**model_cfg)

    # Load state dict (strip 'model.' prefix from Lightning wrapper)
    state_dict = {}
    for k, v in ckpt["state_dict"].items():
        if k.startswith("model."):
            state_dict[k[6:]] = v
        else:
            state_dict[k] = v

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Loaded model with {n_params:,} parameters from {ckpt_path}")
    return model


def load_hdf5_data(data_dir, use_raw_features=False):
    """Load all HDF5 sessions eagerly."""
    data_dir = Path(data_dir)
    sessions = {}
    for fpath in sorted(data_dir.glob("*.h5")):
        rid = fpath.stem
        with h5py.File(fpath, "r") as f:
            data = Data.from_hdf5(f, lazy=False)
            # Optionally swap normalized features for raw features
            if use_raw_features and "reference_features_raw" in f["units"]:
                raw = f["units"]["reference_features_raw"][:]
                data.units.reference_features = raw
        sessions[rid] = data
        logger.info(f"  Loaded {rid}: {len(data.spikes.timestamps)} spikes, "
                     f"{len(data.units.id)} units"
                     + (" [raw features]" if use_raw_features else ""))
    return sessions


def get_validation_windows(data, model, n_windows=50, seed=42):
    """Sample fixed windows from validation domain."""
    rng = np.random.RandomState(seed)
    window_length = model.sequence_length + model.pred_length

    # Get validation domain
    if hasattr(data, "valid_domain"):
        domain = data.valid_domain
    else:
        domain = data.domain

    # Sample windows
    windows = []
    for start, end in zip(domain.start, domain.end):
        if end - start < window_length:
            continue
        max_start = end - window_length
        n_from_interval = max(1, int(n_windows * (end - start) / sum(
            e - s for s, e in zip(domain.start, domain.end) if e - s >= window_length
        )))
        for _ in range(n_from_interval):
            t = rng.uniform(start, max_start)
            windows.append((t, t + window_length))

    # Limit to n_windows
    if len(windows) > n_windows:
        rng.shuffle(windows)
        windows = windows[:n_windows]

    return windows


@torch.no_grad()
def evaluate_session(model, data, session_id, device="cuda", n_windows=50):
    """Evaluate model on a single session's validation data."""
    windows = get_validation_windows(data, model, n_windows=n_windows)

    if not windows:
        logger.warning(f"  {session_id}: No valid windows found")
        return None

    all_log_rates = []
    all_target_counts = []
    all_unit_masks = []

    for t_start, t_end in windows:
        sample = data.slice(t_start, t_end)
        tokenized = model.tokenize(sample)

        # Prepare batch (single sample)
        model_inputs = {}
        for k, v in tokenized["model_inputs"].items():
            if hasattr(v, "obj"):
                # Padded8Object (namedtuple with .obj field)
                inner = v.obj
                if isinstance(inner, torch.Tensor):
                    model_inputs[k] = inner.unsqueeze(0).to(device)
                else:
                    model_inputs[k] = torch.tensor(np.asarray(inner)).unsqueeze(0).to(device)
            elif isinstance(v, np.ndarray):
                model_inputs[k] = torch.tensor(v).unsqueeze(0).to(device)
            elif isinstance(v, torch.Tensor):
                model_inputs[k] = v.unsqueeze(0).to(device)
            elif isinstance(v, (int, float)):
                model_inputs[k] = torch.tensor([v]).to(device)
            else:
                model_inputs[k] = torch.tensor(np.asarray(v)).unsqueeze(0).to(device)

        n_units = tokenized["n_units"]
        ref_features = model_inputs["reference_features"]
        unit_mask = torch.zeros(1, ref_features.shape[1], dtype=torch.bool, device=device)
        unit_mask[0, :n_units] = True
        model_inputs["unit_mask"] = unit_mask

        # Pad target counts
        tc = tokenized["target_counts"]
        n_bins = tc.shape[0]
        tc_padded = np.zeros((1, n_bins, ref_features.shape[1]), dtype=np.float32)
        tc_padded[0, :, :n_units] = tc
        target_counts = torch.tensor(tc_padded).to(device)

        # Forward pass
        log_rates = model(**model_inputs)

        all_log_rates.append(log_rates.cpu())
        all_target_counts.append(target_counts.cpu())
        all_unit_masks.append(unit_mask.cpu())

    # Concatenate
    log_rates = torch.cat(all_log_rates, dim=0)
    target_counts = torch.cat(all_target_counts, dim=0)
    unit_mask = torch.cat(all_unit_masks, dim=0)

    # Compute metrics
    metrics = {
        "session_id": session_id,
        "n_windows": len(windows),
        "n_units": int(len(data.units.id)),
        "poisson_ll": poisson_log_likelihood(log_rates, target_counts, unit_mask).item(),
        "bits_per_spike": bits_per_spike(log_rates, target_counts, unit_mask).item(),
        "fr_correlation": firing_rate_correlation(log_rates, target_counts, unit_mask),
        "r2": r2_binned_counts(log_rates, target_counts, unit_mask),
    }

    # Per-unit metrics
    pred_rates = torch.exp(log_rates)
    per_unit_bps = []
    for u in range(len(data.units.id)):
        u_log_rates = log_rates[:, :, u:u+1]
        u_targets = target_counts[:, :, u:u+1]
        bps = bits_per_spike(u_log_rates, u_targets).item()
        per_unit_bps.append(bps)
    metrics["per_unit_bps"] = per_unit_bps

    return metrics, log_rates, target_counts, unit_mask


def plot_predictions(log_rates, target_counts, unit_mask, session_id, output_dir, n_examples=3):
    """Plot example predictions vs ground truth."""
    output_dir = Path(output_dir)
    pred_rates = torch.exp(log_rates).numpy()
    targets = target_counts.numpy()
    mask = unit_mask.numpy()

    for ex_idx in range(min(n_examples, log_rates.shape[0])):
        n_units_valid = int(mask[ex_idx].sum())
        n_to_show = min(6, n_units_valid)

        fig, axes = plt.subplots(n_to_show, 1, figsize=(12, 2.5 * n_to_show), sharex=True)
        if n_to_show == 1:
            axes = [axes]

        for u_idx in range(n_to_show):
            ax = axes[u_idx]
            pred = pred_rates[ex_idx, :, u_idx]
            true = targets[ex_idx, :, u_idx]

            x = np.arange(len(pred))
            ax.bar(x, true, alpha=0.4, color="steelblue", label="True counts")
            ax.plot(x, pred, "r-", linewidth=1.5, label="Predicted rate")
            ax.set_ylabel(f"Unit {u_idx}")
            if u_idx == 0:
                ax.legend(loc="upper right", fontsize=8)

        axes[-1].set_xlabel("Time bin (20ms)")
        fig.suptitle(f"Session {session_id[:8]} - Window {ex_idx}", fontsize=12)
        plt.tight_layout()
        plt.savefig(output_dir / f"pred_{session_id[:8]}_w{ex_idx}.png", dpi=150, bbox_inches="tight")
        plt.close()


def plot_summary(all_metrics, output_dir):
    """Plot summary metrics across sessions."""
    output_dir = Path(output_dir)

    sessions = [m["session_id"][:8] for m in all_metrics]
    bps = [m["bits_per_spike"] for m in all_metrics]
    frc = [m["fr_correlation"] for m in all_metrics]
    r2s = [m["r2"] for m in all_metrics]
    n_units = [m["n_units"] for m in all_metrics]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Bits per spike
    ax = axes[0, 0]
    colors = ["green" if b > 0 else "red" for b in bps]
    ax.barh(sessions, bps, color=colors, alpha=0.7)
    ax.axvline(x=0, color="k", linestyle="--", linewidth=0.5)
    ax.set_xlabel("Bits/spike")
    ax.set_title("Bits per Spike (> 0 = better than null model)")

    # Firing rate correlation
    ax = axes[0, 1]
    ax.barh(sessions, frc, color="steelblue", alpha=0.7)
    ax.set_xlabel("Pearson r")
    ax.set_title("Firing Rate Correlation")

    # R² of binned counts
    ax = axes[1, 0]
    colors = ["green" if r > 0 else "red" for r in r2s]
    ax.barh(sessions, r2s, color=colors, alpha=0.7)
    ax.axvline(x=0, color="k", linestyle="--", linewidth=0.5)
    ax.set_xlabel("R²")
    ax.set_title("R² of Binned Counts")

    # Per-unit bits/spike distribution
    ax = axes[1, 1]
    all_unit_bps = []
    for m in all_metrics:
        all_unit_bps.extend(m["per_unit_bps"])
    all_unit_bps = [b for b in all_unit_bps if not np.isnan(b) and not np.isinf(b)]
    if all_unit_bps:
        ax.hist(all_unit_bps, bins=50, color="steelblue", alpha=0.7, edgecolor="white")
        ax.axvline(x=0, color="k", linestyle="--", linewidth=0.5)
        median_bps = np.median(all_unit_bps)
        ax.axvline(x=median_bps, color="red", linestyle="-", linewidth=1.5,
                    label=f"Median: {median_bps:.3f}")
        ax.legend()
    ax.set_xlabel("Bits/spike")
    ax.set_ylabel("Count")
    ax.set_title("Per-unit Bits/spike Distribution")

    plt.suptitle("NeuroHorizon Evaluation Summary", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Summary plot saved to {output_dir / 'summary.png'}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate NeuroHorizon model")
    parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--data-dir", type=str,
                        default="/root/autodl-tmp/datasets/ibl_processed",
                        help="Path to HDF5 data directory")
    parser.add_argument("--output-dir", type=str, default="results/neurohorizon_eval",
                        help="Output directory for results")
    parser.add_argument("--n-windows", type=int, default=50,
                        help="Number of validation windows per session")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use-raw-features", action="store_true",
                        help="Use unnormalized reference features (for v1 model)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    device = args.device if torch.cuda.is_available() else "cpu"
    model = load_model(args.ckpt, device=device)

    # Load data
    logger.info("Loading data...")
    sessions = load_hdf5_data(args.data_dir, use_raw_features=args.use_raw_features)

    # Evaluate each session
    all_metrics = []
    for session_id, data in sessions.items():
        logger.info(f"Evaluating {session_id}...")
        result = evaluate_session(
            model, data, session_id, device=device, n_windows=args.n_windows
        )
        if result is None:
            continue

        metrics, log_rates, target_counts, unit_mask = result
        all_metrics.append(metrics)

        logger.info(f"  bits/spike={metrics['bits_per_spike']:.4f}, "
                     f"fr_corr={metrics['fr_correlation']:.4f}, "
                     f"r2={metrics['r2']:.4f}")

        # Plot predictions
        plot_predictions(log_rates, target_counts, unit_mask, session_id, output_dir)

    # Summary
    if all_metrics:
        avg_bps = np.mean([m["bits_per_spike"] for m in all_metrics])
        avg_frc = np.mean([m["fr_correlation"] for m in all_metrics])
        avg_r2 = np.mean([m["r2"] for m in all_metrics])

        summary = {
            "checkpoint": str(args.ckpt),
            "n_sessions": len(all_metrics),
            "avg_bits_per_spike": avg_bps,
            "avg_fr_correlation": avg_frc,
            "avg_r2": avg_r2,
            "per_session": [{k: v for k, v in m.items() if k != "per_unit_bps"}
                           for m in all_metrics],
        }

        # Save JSON
        with open(output_dir / "metrics.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

        # Plot summary
        plot_summary(all_metrics, output_dir)

        logger.info("\n" + "=" * 60)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Sessions evaluated: {len(all_metrics)}")
        logger.info(f"Avg bits/spike:     {avg_bps:.4f}")
        logger.info(f"Avg FR correlation: {avg_frc:.4f}")
        logger.info(f"Avg R²:             {avg_r2:.4f}")
        logger.info(f"Results saved to:   {output_dir}")


if __name__ == "__main__":
    main()
