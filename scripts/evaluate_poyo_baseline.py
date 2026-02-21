#!/usr/bin/env python3
"""Post-training evaluation for POYO baseline (wheel velocity decoding).

Loads a trained POYO checkpoint, runs evaluation on validation data,
computes R², MSE, and correlation metrics.

Usage:
    conda run -n poyo python scripts/evaluate_poyo_baseline.py \
        --ckpt logs/poyo_baseline/lightning_logs/version_20/checkpoints/last.ckpt \
        --data-dir /path/to/ibl_processed \
        --output-dir results/poyo_eval
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
from temporaldata import Data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from torch_brain.data import collate
from torch_brain.models.poyo import POYO
from torch_brain.registry import MODALITY_REGISTRY

IBL_READOUT_CONFIG = {
    "readout": {
        "readout_id": "wheel_velocity",
    }
}


def load_model(ckpt_path, data_dir, device="cuda"):
    """Load POYO model from Lightning checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    hparams = ckpt.get("hyper_parameters", {})
    model_cfg = hparams.get("model", {})
    model_cfg.pop("_target_", None)

    readout_spec = MODALITY_REGISTRY["wheel_velocity"]
    model = POYO(readout_spec=readout_spec, **model_cfg)

    # Initialize vocabularies from data
    data_dir = Path(data_dir)
    recording_ids = sorted([x.stem for x in data_dir.glob("*.h5")])
    all_unit_ids = []
    for rid in recording_ids:
        with h5py.File(data_dir / f"{rid}.h5", "r") as f:
            data = Data.from_hdf5(f, lazy=False)
            if hasattr(data, "units") and hasattr(data.units, "id"):
                all_unit_ids.extend(data.units.id.tolist())
    unit_ids = sorted(set(all_unit_ids))

    model.unit_emb.initialize_vocab(unit_ids)
    model.session_emb.initialize_vocab(recording_ids)

    # Load state dict
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
    logger.info(f"Loaded POYO model with {n_params:,} parameters from {ckpt_path}")
    return model, readout_spec


def load_hdf5_data(data_dir):
    """Load all HDF5 sessions eagerly."""
    data_dir = Path(data_dir)
    sessions = {}
    for fpath in sorted(data_dir.glob("*.h5")):
        rid = fpath.stem
        with h5py.File(fpath, "r") as f:
            data = Data.from_hdf5(f, lazy=False)
        data.config = IBL_READOUT_CONFIG
        sessions[rid] = data
    return sessions


@torch.no_grad()
def evaluate_session(model, data, session_id, device="cuda", n_windows=50, seed=42):
    """Evaluate POYO on a single session's validation data."""
    rng = np.random.RandomState(seed)
    window_length = model.sequence_length

    if hasattr(data, "valid_domain"):
        domain = data.valid_domain
    else:
        domain = data.domain

    windows = []
    for start, end in zip(domain.start, domain.end):
        if end - start < window_length:
            continue
        max_start = end - window_length
        total_valid = sum(e - s for s, e in zip(domain.start, domain.end) if e - s >= window_length)
        n_from = max(1, int(n_windows * (end - start) / max(total_valid, 1e-6)))
        for _ in range(n_from):
            t = rng.uniform(start, max_start)
            windows.append((t, t + window_length))

    if len(windows) > n_windows:
        rng.shuffle(windows)
        windows = windows[:n_windows]

    if not windows:
        return None

    all_preds = []
    all_targets = []

    for t_start, t_end in windows:
        sample = data.slice(t_start, t_end)
        sample.config = IBL_READOUT_CONFIG

        try:
            tokenized = model.tokenize(sample)
        except Exception as e:
            continue

        # Collate single sample
        batch = collate([tokenized])

        # Move to device
        model_inputs = {}
        for k, v in batch["model_inputs"].items():
            if isinstance(v, torch.Tensor):
                model_inputs[k] = v.to(device)
            else:
                model_inputs[k] = v

        output = model(**model_inputs)

        mask = model_inputs.get("output_mask")
        if mask is not None:
            output_masked = output[mask].cpu()
            target_masked = batch["target_values"][mask].cpu()
        else:
            output_masked = output.cpu()
            target_masked = batch["target_values"].cpu()

        # Flatten
        all_preds.append(output_masked.view(-1))
        all_targets.append(target_masked.view(-1))

    if not all_preds:
        return None

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)

    # Metrics
    mse = ((preds - targets) ** 2).mean().item()
    ss_res = ((preds - targets) ** 2).sum()
    ss_tot = ((targets - targets.mean()) ** 2).sum()
    r2 = (1 - ss_res / ss_tot.clamp(min=1e-8)).item()

    # Correlation
    p_c = preds - preds.mean()
    t_c = targets - targets.mean()
    corr = (p_c * t_c).sum() / (p_c.norm() * t_c.norm()).clamp(min=1e-8)
    corr = corr.item()

    return {
        "session_id": session_id,
        "n_windows": len(windows),
        "n_predictions": int(preds.shape[0]),
        "mse": mse,
        "r2": r2,
        "correlation": corr,
        "pred_mean": preds.mean().item(),
        "pred_std": preds.std().item(),
        "target_mean": targets.mean().item(),
        "target_std": targets.std().item(),
    }


def plot_summary(all_metrics, output_dir):
    """Plot summary metrics."""
    output_dir = Path(output_dir)

    sessions = [m["session_id"][:8] for m in all_metrics]
    r2s = [m["r2"] for m in all_metrics]
    corrs = [m["correlation"] for m in all_metrics]
    mses = [m["mse"] for m in all_metrics]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    colors = ["green" if r > 0 else "red" for r in r2s]
    ax.barh(sessions, r2s, color=colors, alpha=0.7)
    ax.axvline(x=0, color="k", linestyle="--", linewidth=0.5)
    ax.set_xlabel("R²")
    ax.set_title("R² (Wheel Velocity Decoding)")

    ax = axes[1]
    ax.barh(sessions, corrs, color="steelblue", alpha=0.7)
    ax.set_xlabel("Pearson r")
    ax.set_title("Prediction-Target Correlation")

    ax = axes[2]
    ax.barh(sessions, mses, color="orange", alpha=0.7)
    ax.set_xlabel("MSE")
    ax.set_title("Mean Squared Error")

    plt.suptitle("POYO Baseline Evaluation Summary", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "summary.png", dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate POYO baseline")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--data-dir", type=str,
                        default="/root/autodl-tmp/datasets/ibl_processed")
    parser.add_argument("--output-dir", type=str, default="results/poyo_eval")
    parser.add_argument("--n-windows", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device if torch.cuda.is_available() else "cpu"
    model, readout_spec = load_model(args.ckpt, args.data_dir, device=device)

    logger.info("Loading data...")
    sessions = load_hdf5_data(args.data_dir)

    all_metrics = []
    for session_id, data in sessions.items():
        logger.info(f"Evaluating {session_id}...")
        result = evaluate_session(model, data, session_id, device=device, n_windows=args.n_windows)
        if result is None:
            logger.warning(f"  Skipped {session_id}")
            continue
        all_metrics.append(result)
        logger.info(f"  R²={result['r2']:.4f}, corr={result['correlation']:.4f}, MSE={result['mse']:.4f}")

    if all_metrics:
        avg_r2 = np.mean([m["r2"] for m in all_metrics])
        avg_corr = np.mean([m["correlation"] for m in all_metrics])
        avg_mse = np.mean([m["mse"] for m in all_metrics])

        summary = {
            "checkpoint": str(args.ckpt),
            "n_sessions": len(all_metrics),
            "avg_r2": avg_r2,
            "avg_correlation": avg_corr,
            "avg_mse": avg_mse,
            "per_session": all_metrics,
        }

        with open(output_dir / "metrics.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

        plot_summary(all_metrics, output_dir)

        logger.info("\n" + "=" * 60)
        logger.info("POYO BASELINE EVALUATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Sessions evaluated: {len(all_metrics)}")
        logger.info(f"Avg R²:         {avg_r2:.4f}")
        logger.info(f"Avg Correlation: {avg_corr:.4f}")
        logger.info(f"Avg MSE:        {avg_mse:.4f}")
        logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
