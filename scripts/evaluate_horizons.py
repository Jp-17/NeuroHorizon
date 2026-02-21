"""Evaluate NeuroHorizon at different prediction horizons.

Tests how prediction quality degrades as the prediction window extends
further into the future. This is Experiment 3 from the plan.

Evaluates at: 100ms, 200ms, 500ms, 1000ms prediction horizons.
For each horizon, re-tokenizes the data with the specified pred_length
and evaluates bits/spike and other metrics.

Usage:
    conda run -n poyo python scripts/evaluate_horizons.py \
        --ckpt logs/neurohorizon/lightning_logs/version_5/checkpoints/last.ckpt \
        --data-dir /path/to/ibl_processed \
        --output results/horizon_eval
"""

import argparse
import json
import logging
from pathlib import Path

import h5py
import numpy as np
import torch
from temporaldata import Data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from torch_brain.data import collate
from torch_brain.models import NeuroHorizon
from torch_brain.utils.neurohorizon_metrics import (
    bits_per_spike,
    firing_rate_correlation,
    r2_binned_counts,
)

HORIZONS = [0.1, 0.2, 0.5, 1.0]  # seconds


def load_model(ckpt_path, device="cuda"):
    """Load NeuroHorizon model from Lightning checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hparams = ckpt.get("hyper_parameters", {})
    model_cfg = hparams.get("model", {})
    model_cfg.pop("_target_", None)
    model = NeuroHorizon(**model_cfg)

    state_dict = {}
    for k, v in ckpt["state_dict"].items():
        if k.startswith("model."):
            state_dict[k[6:]] = v
        else:
            state_dict[k] = v
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Loaded model with {n_params:,} params, "
                f"trained pred_length={model.pred_length}s")
    return model


def tokenize_with_horizon(model, sample, pred_length):
    """Re-tokenize a sample with a specific prediction horizon.

    Overrides the model's pred_length temporarily to generate
    target counts for a different prediction window.
    """
    original_pred_length = model.pred_length
    original_num_pred_bins = model.num_pred_bins

    model.pred_length = pred_length
    model.num_pred_bins = round(pred_length / model.bin_size)

    try:
        tokenized = model.tokenize(sample)
    finally:
        model.pred_length = original_pred_length
        model.num_pred_bins = original_num_pred_bins

    return tokenized


@torch.no_grad()
def evaluate_horizon(model, sessions, pred_length, device="cuda",
                     n_windows_per_session=30, seed=42):
    """Evaluate model at a specific prediction horizon."""
    rng = np.random.RandomState(seed)
    window_length = model.sequence_length + pred_length

    all_bps = []
    all_fr_corr = []
    all_r2 = []
    n_evaluated = 0

    for session_id, data in sessions.items():
        if hasattr(data, "valid_domain"):
            domain = data.valid_domain
        else:
            domain = data.domain

        # Sample windows
        windows = []
        for start, end in zip(domain.start, domain.end):
            usable = end - start - window_length
            if usable <= 0:
                continue
            n_from = max(1, min(n_windows_per_session,
                                int(n_windows_per_session * (end - start) /
                                    sum(max(0, e - s - window_length)
                                        for s, e in zip(domain.start, domain.end)
                                        if e - s > window_length))))
            for _ in range(n_from):
                t = rng.uniform(start, start + usable)
                windows.append((t, t + window_length))

        if len(windows) > n_windows_per_session:
            rng.shuffle(windows)
            windows = windows[:n_windows_per_session]

        if not windows:
            continue

        session_bps = []
        for t_start, t_end in windows:
            sample = data.slice(t_start, t_end)

            try:
                tokenized = tokenize_with_horizon(model, sample, pred_length)
            except Exception as e:
                continue

            # Build model inputs
            model_inputs = {}
            for k, v in tokenized["model_inputs"].items():
                if isinstance(v, np.ndarray):
                    model_inputs[k] = torch.tensor(v).unsqueeze(0).to(device)
                elif hasattr(v, "data"):
                    model_inputs[k] = collate([v]).to(device)
                else:
                    model_inputs[k] = torch.tensor(v).unsqueeze(0).to(device)

            n_units = tokenized["n_units"]
            ref_features = model_inputs["reference_features"]
            unit_mask = torch.zeros(1, ref_features.shape[1],
                                    dtype=torch.bool, device=device)
            unit_mask[0, :n_units] = True
            model_inputs["unit_mask"] = unit_mask

            tc = tokenized["target_counts"]
            n_bins = tc.shape[0]
            tc_padded = np.zeros((1, n_bins, ref_features.shape[1]),
                                dtype=np.float32)
            tc_padded[0, :, :n_units] = tc
            target_counts = torch.tensor(tc_padded).to(device)

            # Override decoder bins for different horizon
            num_bins = round(pred_length / model.bin_size)
            bin_times = torch.linspace(
                model.sequence_length,
                model.sequence_length + pred_length,
                num_bins + 1, device=device
            )
            bin_centers = (bin_times[:-1] + bin_times[1:]) / 2

            # Forward pass - need to handle different num bins
            # The model's decoder creates bin tokens based on num_pred_bins
            # We temporarily modify it
            original_num_bins = model.num_pred_bins
            model.num_pred_bins = num_bins
            try:
                log_rates = model(**model_inputs)
            finally:
                model.num_pred_bins = original_num_bins

            bps = bits_per_spike(log_rates, target_counts, unit_mask).item()
            if not np.isnan(bps) and not np.isinf(bps):
                session_bps.append(bps)
                all_bps.append(bps)

            fr_c = firing_rate_correlation(log_rates, target_counts, unit_mask)
            if not np.isnan(fr_c):
                all_fr_corr.append(fr_c)

            r2 = r2_binned_counts(log_rates, target_counts, unit_mask)
            if not np.isnan(r2):
                all_r2.append(r2)

            n_evaluated += 1

        if session_bps:
            logger.info(f"  {session_id[:8]}: bps={np.mean(session_bps):.4f} "
                        f"({len(session_bps)} windows)")

    if not all_bps:
        return None

    return {
        "pred_length": pred_length,
        "n_windows": n_evaluated,
        "bits_per_spike_mean": float(np.mean(all_bps)),
        "bits_per_spike_std": float(np.std(all_bps)),
        "fr_corr_mean": float(np.mean(all_fr_corr)) if all_fr_corr else None,
        "r2_mean": float(np.mean(all_r2)) if all_r2 else None,
    }


def plot_results(results, output_dir):
    """Plot bits/spike vs prediction horizon."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping plots")
        return

    horizons = [r["pred_length"] * 1000 for r in results]  # ms
    bps_mean = [r["bits_per_spike_mean"] for r in results]
    bps_std = [r["bits_per_spike_std"] for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # BPS vs horizon
    ax = axes[0]
    ax.errorbar(horizons, bps_mean, yerr=bps_std, fmt="o-",
                color="steelblue", capsize=5, markersize=8)
    ax.axhline(y=0, color="k", linestyle="--", alpha=0.5, label="Null model")
    ax.set_xlabel("Prediction Horizon (ms)")
    ax.set_ylabel("Bits per Spike")
    ax.set_title("Prediction Quality vs Horizon")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # FR correlation vs horizon
    ax = axes[1]
    fr_corrs = [r["fr_corr_mean"] for r in results if r["fr_corr_mean"] is not None]
    if fr_corrs:
        ax.plot([r["pred_length"] * 1000 for r in results if r["fr_corr_mean"] is not None],
                fr_corrs, "o-", color="coral", markersize=8)
    ax.set_xlabel("Prediction Horizon (ms)")
    ax.set_ylabel("Firing Rate Correlation")
    ax.set_title("FR Correlation vs Horizon")
    ax.grid(True, alpha=0.3)

    # R2 vs horizon
    ax = axes[2]
    r2s = [r["r2_mean"] for r in results if r["r2_mean"] is not None]
    if r2s:
        ax.plot([r["pred_length"] * 1000 for r in results if r["r2_mean"] is not None],
                r2s, "o-", color="green", markersize=8)
    ax.axhline(y=0, color="k", linestyle="--", alpha=0.5)
    ax.set_xlabel("Prediction Horizon (ms)")
    ax.set_ylabel("R²")
    ax.set_title("R² vs Horizon")
    ax.grid(True, alpha=0.3)

    plt.suptitle("NeuroHorizon: Prediction Quality vs Time Horizon",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "horizon_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Plot saved to {output_dir / 'horizon_comparison.png'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--data-dir", type=str,
                        default="/root/"
                                "autodl-tmp/datasets/ibl_processed")
    parser.add_argument("--horizons", nargs="+", type=float, default=HORIZONS,
                        help="Prediction horizons in seconds")
    parser.add_argument("--n-windows", type=int, default=30,
                        help="Windows per session")
    parser.add_argument("--output", type=str, default="results/horizon_eval")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device if torch.cuda.is_available() else "cpu"
    model = load_model(args.ckpt, device=device)

    # Load data
    logger.info("Loading data...")
    data_dir = Path(args.data_dir)
    sessions = {}
    for fpath in sorted(data_dir.glob("*.h5")):
        rid = fpath.stem
        with h5py.File(fpath, "r") as f:
            data = Data.from_hdf5(f, lazy=False)
        sessions[rid] = data
        logger.info(f"  Loaded {rid}: {len(data.units.id)} units")
    logger.info(f"Total: {len(sessions)} sessions")

    # Evaluate at each horizon
    results = []
    for horizon in sorted(args.horizons):
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating at horizon = {horizon*1000:.0f}ms")
        logger.info(f"{'='*60}")

        result = evaluate_horizon(
            model, sessions, pred_length=horizon,
            device=device, n_windows_per_session=args.n_windows
        )
        if result:
            results.append(result)
            logger.info(f"  >> BPS: {result['bits_per_spike_mean']:.4f} "
                        f"+/- {result['bits_per_spike_std']:.4f}")

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("HORIZON EVALUATION SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"{'Horizon (ms)':>15} {'BPS Mean':>10} {'BPS Std':>10} "
                f"{'FR Corr':>10} {'R²':>10}")
    logger.info("-" * 60)
    for r in results:
        logger.info(f"{r['pred_length']*1000:>12.0f}ms "
                    f"{r['bits_per_spike_mean']:>10.4f} "
                    f"{r['bits_per_spike_std']:>10.4f} "
                    f"{r['fr_corr_mean'] or 0:>10.4f} "
                    f"{r['r2_mean'] or 0:>10.4f}")

    # Save results
    with open(output_dir / "horizon_results.json", "w") as f:
        json.dump({
            "checkpoint": args.ckpt,
            "trained_pred_length": model.pred_length,
            "results": results,
        }, f, indent=2)

    # Plot
    if results:
        plot_results(results, output_dir)

    logger.info(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
