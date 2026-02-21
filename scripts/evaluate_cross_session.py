"""Cross-session generalization evaluation for NeuroHorizon.

Evaluates how well the model generalizes to completely unseen sessions.
This is the core experiment testing IDEncoder's cross-session capability.

Setup:
- Train on N sessions, hold out M sessions
- At evaluation, load model checkpoint and run on held-out sessions
- No fine-tuning: IDEncoder generates unit embeddings from reference features alone

Usage:
    conda run -n poyo python scripts/evaluate_cross_session.py \
        --ckpt logs/neurohorizon/lightning_logs/version_X/checkpoints/last.ckpt \
        --train-sessions session1 session2 ... \
        --test-sessions sessionA sessionB ...
"""

import argparse
import json
import logging
from pathlib import Path

import h5py
import numpy as np
import torch
from temporaldata import Data

from torch_brain.models.neurohorizon import NeuroHorizon
from torch_brain.utils.neurohorizon_metrics import (
    bits_per_spike,
    firing_rate_correlation,
    r2_binned_counts,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def load_model(ckpt_path, device="cuda"):
    """Load NeuroHorizon model from Lightning checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
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
    return model


def evaluate_session(model, data, n_windows=50, device="cuda"):
    """Evaluate model on a single session using sliding windows.

    Returns per-window metrics.
    """
    window_length = model.sequence_length + model.pred_length
    domain_start = float(data.domain.start[0])
    domain_end = float(data.domain.end[0])
    usable_length = domain_end - domain_start - window_length

    if usable_length <= 0:
        logger.warning("Session too short for evaluation")
        return None

    # Sample window start times evenly
    rng = np.random.RandomState(42)
    starts = rng.uniform(domain_start, domain_start + usable_length, size=n_windows)

    all_metrics = []
    for t0 in starts:
        sample = data.slice(t0, t0 + window_length)
        token_dict = model.tokenize(sample)

        # Build model inputs
        from torch_brain.data import collate

        model_inputs = {}
        for k, v in token_dict["model_inputs"].items():
            if isinstance(v, np.ndarray):
                model_inputs[k] = torch.tensor(v).unsqueeze(0).to(device)
            elif isinstance(v, torch.Tensor):
                model_inputs[k] = v.unsqueeze(0).to(device)
            elif hasattr(v, "obj"):
                # Padded8Object (namedtuple with .obj field)
                inner = v.obj
                if isinstance(inner, torch.Tensor):
                    model_inputs[k] = inner.unsqueeze(0).to(device)
                else:
                    model_inputs[k] = torch.tensor(np.asarray(inner)).unsqueeze(0).to(device)
            else:
                model_inputs[k] = torch.tensor(np.asarray(v)).unsqueeze(0).to(device)

        target_counts = torch.tensor(token_dict["target_counts"]).unsqueeze(0).to(device)
        n_units = token_dict["n_units"]
        unit_mask = torch.ones(1, n_units, dtype=torch.bool, device=device)
        model_inputs["unit_mask"] = unit_mask

        with torch.no_grad():
            log_rates = model(**model_inputs)

        # Compute metrics
        bps = bits_per_spike(log_rates, target_counts, unit_mask).item()
        fr_corr = firing_rate_correlation(log_rates, target_counts, unit_mask).item()
        r2 = r2_binned_counts(log_rates, target_counts, unit_mask).item()

        all_metrics.append({
            "bits_per_spike": bps,
            "fr_corr": fr_corr,
            "r2": r2,
        })

    return all_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--data-dir", type=str,
                        default="/root/autodl-tmp/datasets/ibl_processed")
    parser.add_argument("--train-sessions", nargs="+", default=None,
                        help="Session IDs used for training (for reference)")
    parser.add_argument("--test-sessions", nargs="+", default=None,
                        help="Session IDs for cross-session evaluation. If None, use all sessions not in train-sessions")
    parser.add_argument("--n-windows", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    all_sessions = sorted([p.stem for p in data_dir.glob("*.h5")])
    logger.info(f"Found {len(all_sessions)} sessions")

    if args.test_sessions:
        test_sessions = args.test_sessions
    elif args.train_sessions:
        test_sessions = [s for s in all_sessions if s not in args.train_sessions]
    else:
        test_sessions = all_sessions

    logger.info(f"Evaluating on {len(test_sessions)} sessions")
    if args.train_sessions:
        in_train = [s for s in test_sessions if s in (args.train_sessions or [])]
        out_of_train = [s for s in test_sessions if s not in (args.train_sessions or [])]
        logger.info(f"  In-distribution: {len(in_train)}, Cross-session: {len(out_of_train)}")

    # Load model
    model = load_model(args.ckpt, device=args.device)
    logger.info(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

    results = {}
    for sid in test_sessions:
        logger.info(f"\nEvaluating {sid}...")
        fpath = data_dir / f"{sid}.h5"
        if not fpath.exists():
            logger.warning(f"  File not found: {fpath}")
            continue

        with h5py.File(fpath, "r") as f:
            data = Data.from_hdf5(f, lazy=False)

        metrics = evaluate_session(model, data, n_windows=args.n_windows, device=args.device)
        if metrics is None:
            continue

        # Aggregate
        mean_bps = np.mean([m["bits_per_spike"] for m in metrics])
        mean_corr = np.mean([m["fr_corr"] for m in metrics])
        mean_r2 = np.mean([m["r2"] for m in metrics])
        std_bps = np.std([m["bits_per_spike"] for m in metrics])

        is_train = sid in (args.train_sessions or [])
        label = "TRAIN" if is_train else "TEST"
        logger.info(f"  [{label}] bits/spike: {mean_bps:.4f} +/- {std_bps:.4f}")
        logger.info(f"  [{label}] FR corr: {mean_corr:.4f}")
        logger.info(f"  [{label}] R²: {mean_r2:.4f}")

        results[sid] = {
            "bits_per_spike_mean": mean_bps,
            "bits_per_spike_std": std_bps,
            "fr_corr_mean": mean_corr,
            "r2_mean": mean_r2,
            "n_windows": len(metrics),
            "is_train_session": is_train,
        }

    # Summary
    train_results = {k: v for k, v in results.items() if v["is_train_session"]}
    test_results = {k: v for k, v in results.items() if not v["is_train_session"]}

    logger.info("\n" + "=" * 60)
    logger.info("CROSS-SESSION GENERALIZATION SUMMARY")
    logger.info("=" * 60)

    if train_results:
        train_bps = np.mean([v["bits_per_spike_mean"] for v in train_results.values()])
        logger.info(f"In-distribution ({len(train_results)} sessions):")
        logger.info(f"  Mean bits/spike: {train_bps:.4f}")

    if test_results:
        test_bps = np.mean([v["bits_per_spike_mean"] for v in test_results.values()])
        logger.info(f"Cross-session ({len(test_results)} sessions):")
        logger.info(f"  Mean bits/spike: {test_bps:.4f}")

        if train_results:
            gap = train_bps - test_bps
            logger.info(f"Generalization gap: {gap:.4f}")

    # Save
    output_path = args.output or "cross_session_results.json"
    with open(output_path, "w") as f:
        json.dump({
            "checkpoint": args.ckpt,
            "sessions": results,
            "summary": {
                "n_train": len(train_results),
                "n_test": len(test_results),
                "train_bps_mean": float(np.mean([v["bits_per_spike_mean"] for v in train_results.values()])) if train_results else None,
                "test_bps_mean": float(np.mean([v["bits_per_spike_mean"] for v in test_results.values()])) if test_results else None,
            },
        }, f, indent=2, default=float)
    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
