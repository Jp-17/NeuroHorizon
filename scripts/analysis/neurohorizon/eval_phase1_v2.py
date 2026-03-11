#!/usr/bin/env python3
"""Comprehensive evaluation script for Phase 1 v2 experiments.

Evaluates a trained NeuroHorizon model with:
1. Continuous mode: fp-bps (overall + per-bin), R-squared
2. Trial-aligned mode: PSTH-R-squared (8 directions)

Usage:
    python scripts/analysis/neurohorizon/eval_phase1_v2.py \
        --checkpoint results/logs/phase1_v2_250ms_cont/.../last.ckpt \
        --config-name train_v2_250ms

    # Or auto-find best checkpoint:
    python scripts/analysis/neurohorizon/eval_phase1_v2.py \
        --log-dir results/logs/phase1_v2_250ms_cont \
        --config-name train_v2_250ms
"""

import argparse
import json
import logging
import math
from collections import defaultdict
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf, DictConfig
from torch.utils.data import DataLoader

from torch_brain.data import Dataset, collate
from torch_brain.data.sampler import RandomFixedWindowSampler
from torch_brain.data.trial_sampler import TrialAlignedSampler
from torch_brain.models import NeuroHorizon
from torch_brain.transforms import Compose
from torch_brain.utils.neurohorizon_metrics import (
    fp_bps,
    fp_bps_per_bin,
    r2_score,
    psth_r2,
    compute_null_rates,
    build_null_rate_lookup,
)

logger = logging.getLogger(__name__)


def find_best_checkpoint(log_dir: str) -> str:
    """Find the best (last.ckpt) checkpoint in a log directory."""
    log_path = Path(log_dir)
    # Look for last.ckpt in lightning_logs/version_*/checkpoints/
    candidates = list(log_path.glob("lightning_logs/version_*/checkpoints/last.ckpt"))
    if not candidates:
        # Try direct checkpoints dir
        candidates = list(log_path.glob("**/last.ckpt"))
    if not candidates:
        raise FileNotFoundError(f"No last.ckpt found in {log_dir}")
    # Return the most recent
    return str(sorted(candidates, key=lambda p: p.stat().st_mtime)[-1])


def load_model_from_checkpoint(ckpt_path: str, device: torch.device):
    """Load model and config from a checkpoint."""
    logger.info(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(ckpt["hyper_parameters"])

    model = hydra.utils.instantiate(cfg.model)

    # Setup train dataset for vocab initialization
    train_dataset = Dataset(
        root=cfg.data_root,
        config=cfg.dataset,
        split="train",
    )
    train_dataset.disable_data_leakage_check()

    model.unit_emb.initialize_vocab(train_dataset.get_unit_ids())
    model.session_emb.initialize_vocab(train_dataset.get_session_ids())

    # Load weights
    state_dict = {
        k.replace("model.", ""): v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("model.")
    }
    model.load_state_dict(state_dict)
    model.eval()
    model = model.to(device)

    return model, cfg, train_dataset


def evaluate_continuous(model, cfg, train_dataset, null_lookup, device, batch_size=64):
    """Continuous mode evaluation: fp-bps (overall + per-bin), R-squared."""
    logger.info("=== Continuous Mode Evaluation ===")

    eval_transforms = (
        hydra.utils.instantiate(cfg.eval_transforms) if cfg.get("eval_transforms") else []
    )
    eval_dataset = Dataset(
        root=cfg.data_root,
        config=cfg.dataset,
        split="valid",
        transform=Compose([*eval_transforms, model.tokenize]),
    )
    eval_dataset.disable_data_leakage_check()

    sampler = RandomFixedWindowSampler(
        sampling_intervals=eval_dataset.get_sampling_intervals(),
        window_length=model.sequence_length,
        generator=torch.Generator().manual_seed(cfg.seed + 2),
    )

    loader = DataLoader(
        eval_dataset,
        sampler=sampler,
        collate_fn=collate,
        batch_size=batch_size,
        num_workers=0,
    )

    logger.info(f"Continuous eval: {len(sampler)} samples")

    all_bps = []
    all_r2 = []
    per_bin_bps_accum = defaultdict(list)
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            inputs = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch["model_inputs"].items()
            }

            log_rate = model(**inputs)
            target = batch["target_spike_counts"].to(device)
            unit_mask = batch["model_inputs"]["target_unit_mask"].to(device)
            T = log_rate.shape[1]

            # Loss
            mask_3d = unit_mask.unsqueeze(1).expand(-1, T, -1)
            loss = (torch.exp(log_rate[mask_3d].clamp(-10, 10))
                    - target[mask_3d] * log_rate[mask_3d].clamp(-10, 10)).mean()
            total_loss += loss.item()

            # R-squared
            pred_rate = torch.exp(log_rate.clamp(-10, 10))
            batch_r2 = r2_score(pred_rate, target, unit_mask)
            all_r2.append(batch_r2.item())

            # fp-bps
            target_unit_index = batch["model_inputs"]["target_unit_index"].to(device)
            max_idx = null_lookup.shape[0] - 1
            null_log_rates = null_lookup[target_unit_index.clamp(0, max_idx)]

            batch_bps = fp_bps(log_rate, target, null_log_rates, unit_mask)
            all_bps.append(batch_bps.item())

            # Per-bin fp-bps
            pb_bps = fp_bps_per_bin(log_rate, target, null_log_rates, unit_mask)
            for t in range(T):
                per_bin_bps_accum[t].append(pb_bps[t].item())

            n_batches += 1

    mean_bps = np.mean(all_bps)
    mean_r2 = np.mean(all_r2)
    mean_loss = total_loss / max(n_batches, 1)
    per_bin_mean = {t: np.mean(vals) for t, vals in sorted(per_bin_bps_accum.items())}

    logger.info(f"  fp-bps = {mean_bps:.4f}")
    logger.info(f"  R2 = {mean_r2:.4f}")
    logger.info(f"  val_loss = {mean_loss:.4f}")
    logger.info(f"  per-bin fp-bps: {[f'{v:.3f}' for v in per_bin_mean.values()]}")

    return {
        "fp_bps": float(mean_bps),
        "r2": float(mean_r2),
        "val_loss": float(mean_loss),
        "per_bin_fp_bps": {str(k): float(v) for k, v in per_bin_mean.items()},
        "n_samples": len(sampler),
    }


def evaluate_trial_aligned(model, cfg, train_dataset, null_lookup, device,
                           batch_size=16, sigma_bins=1):
    """Trial-aligned evaluation: PSTH-R-squared by target direction."""
    logger.info("=== Trial-Aligned Evaluation (PSTH-R2) ===")

    eval_transforms = (
        hydra.utils.instantiate(cfg.eval_transforms) if cfg.get("eval_transforms") else []
    )
    eval_dataset = Dataset(
        root=cfg.data_root,
        config=cfg.dataset,
        split="valid",
        transform=Compose([*eval_transforms, model.tokenize]),
    )
    eval_dataset.disable_data_leakage_check()

    trial_info = eval_dataset.get_trial_intervals(split="valid")
    sampler = TrialAlignedSampler(
        trial_info=trial_info,
        obs_window=model.hist_window,
        pred_window=model.pred_window,
        shuffle=False,
    )

    loader = DataLoader(
        eval_dataset,
        sampler=sampler,
        collate_fn=collate,
        batch_size=batch_size,
        num_workers=0,
    )

    logger.info(f"Trial-aligned eval: {len(sampler)} trials")

    # Store per-trial masked-mean rates: [T] per trial (averaged over valid neurons)
    pred_by_target = defaultdict(list)
    true_by_target = defaultdict(list)
    trial_fp_bps = []

    with torch.no_grad():
        for batch in loader:
            inputs = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch["model_inputs"].items()
            }

            log_rate = model(**inputs)
            pred_rate = torch.exp(log_rate.clamp(-10, 10)).cpu()
            target = batch["target_spike_counts"].cpu()
            unit_mask = batch["model_inputs"]["target_unit_mask"].cpu()

            # fp-bps for this batch
            target_unit_index = batch["model_inputs"]["target_unit_index"].to(device)
            max_idx = null_lookup.shape[0] - 1
            null_log_rates = null_lookup[target_unit_index.clamp(0, max_idx)].cpu()
            batch_bps = fp_bps(log_rate.cpu(), target, null_log_rates, unit_mask)
            trial_fp_bps.append(batch_bps.item())

            # Group by target_id: use masked mean over neurons -> [T] per trial
            target_ids = batch.get("trial_target_id", None)
            if target_ids is None:
                continue

            B, T, N = pred_rate.shape
            for i in range(B):
                tid = int(target_ids[i])
                mask_i = unit_mask[i]  # [N]
                n_valid = mask_i.sum().item()
                if n_valid == 0:
                    continue
                # Average over valid neurons: [T, N] * [N] -> [T]
                pred_mean = (pred_rate[i] * mask_i.unsqueeze(0)).sum(dim=1) / n_valid
                true_mean = (target[i] * mask_i.unsqueeze(0)).sum(dim=1) / n_valid
                pred_by_target[tid].append(pred_mean)
                true_by_target[tid].append(true_mean)

    results = {
        "trial_fp_bps": float(np.mean(trial_fp_bps)) if trial_fp_bps else 0.0,
        "n_trials": len(sampler),
    }

    if pred_by_target:
        pred_stacked = {}
        true_stacked = {}
        per_target_r2 = {}

        for tid in sorted(pred_by_target.keys()):
            # Now all tensors are [T], can safely stack -> [n_trials, T]
            pred_stacked[tid] = torch.stack(pred_by_target[tid])
            true_stacked[tid] = torch.stack(true_by_target[tid])
            n_trials = pred_stacked[tid].shape[0]
            logger.info(f"  Target {tid}: {n_trials} trials")

            # Per-target PSTH R2 (on population-mean rates)
            # PSTH: mean across trials -> [T]
            psth_pred = pred_stacked[tid].mean(dim=0)
            psth_true = true_stacked[tid].mean(dim=0)
            ss_res = ((psth_pred - psth_true) ** 2).sum()
            ss_tot = ((psth_true - psth_true.mean()) ** 2).sum()
            tid_r2 = float(1 - ss_res / (ss_tot + 1e-8))
            per_target_r2[str(tid)] = tid_r2

        # Overall PSTH R2: concatenate all directions' PSTHs
        all_psth_pred = []
        all_psth_true = []
        for tid in sorted(pred_stacked.keys()):
            all_psth_pred.append(pred_stacked[tid].mean(dim=0))
            all_psth_true.append(true_stacked[tid].mean(dim=0))
        all_pred = torch.cat(all_psth_pred)
        all_true = torch.cat(all_psth_true)
        ss_res = ((all_pred - all_true) ** 2).sum()
        ss_tot = ((all_true - all_true.mean()) ** 2).sum()
        overall_r2 = float(1 - ss_res / (ss_tot + 1e-8))

        results["psth_r2"] = overall_r2
        results["per_target_psth_r2"] = per_target_r2
        results["trials_per_target"] = {
            str(k): len(v) for k, v in pred_by_target.items()
        }
        logger.info(f"  Overall PSTH-R2 = {overall_r2:.4f}")
        logger.info(f"  Per-target PSTH-R2: {per_target_r2}")
    else:
        results["psth_r2"] = float("nan")
        logger.warning("No trial metadata, PSTH-R2 not computed")

    return results


def main():
    parser = argparse.ArgumentParser(description="Phase 1 v2 comprehensive evaluation")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint", type=str, help="Direct checkpoint path")
    group.add_argument("--log-dir", type=str, help="Log directory (auto-finds last.ckpt)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--trial-batch-size", type=int, default=16)
    parser.add_argument("--sigma-bins", type=int, default=1)
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument("--skip-continuous", action="store_true")
    parser.add_argument("--skip-trial", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Find checkpoint
    if args.checkpoint:
        ckpt_path = args.checkpoint
    else:
        ckpt_path = find_best_checkpoint(args.log_dir)
    logger.info(f"Checkpoint: {ckpt_path}")

    # Load model
    model, cfg, train_dataset = load_model_from_checkpoint(ckpt_path, device)
    logger.info(
        f"Model: pred_window={model.pred_window}s, "
        f"hist_window={model.hist_window}s, "
        f"T_pred_bins={model.T_pred_bins}, "
        f"bin_size={model.bin_size}s"
    )

    # Compute null rates
    logger.info("Computing null rates from training data...")
    null_rates = compute_null_rates(train_dataset, model, model.bin_size)
    null_lookup = build_null_rate_lookup(null_rates, device=device)
    logger.info(f"Null model: {len(null_rates)} neurons")

    results = {
        "checkpoint": ckpt_path,
        "pred_window_ms": int(model.pred_window * 1000),
        "hist_window_ms": int(model.hist_window * 1000),
        "n_pred_bins": model.T_pred_bins,
        "bin_size_ms": int(model.bin_size * 1000),
        "trial_aligned_training": bool(getattr(cfg, "trial_aligned", False)),
    }

    # Continuous evaluation
    if not args.skip_continuous:
        cont_results = evaluate_continuous(
            model, cfg, train_dataset, null_lookup, device,
            batch_size=args.batch_size
        )
        results["continuous"] = cont_results

    # Trial-aligned evaluation
    if not args.skip_trial:
        trial_results = evaluate_trial_aligned(
            model, cfg, train_dataset, null_lookup, device,
            batch_size=args.trial_batch_size,
            sigma_bins=args.sigma_bins
        )
        results["trial_aligned"] = trial_results

    # Print summary
    print("\n" + "=" * 60)
    print(f"EVALUATION SUMMARY")
    print(f"  Model: pred={results['pred_window_ms']}ms, "
          f"obs={results['hist_window_ms']}ms, "
          f"training={'trial-aligned' if results['trial_aligned_training'] else 'continuous'}")
    print("=" * 60)

    if "continuous" in results:
        c = results["continuous"]
        print(f"\n  [Continuous Mode]")
        print(f"    fp-bps:   {c['fp_bps']:.4f}")
        print(f"    R2:       {c['r2']:.4f}")
        print(f"    val_loss: {c['val_loss']:.4f}")
        bins = c["per_bin_fp_bps"]
        if bins:
            print(f"    per-bin fp-bps (first 5): "
                  f"{', '.join(f'{bins[str(i)]:.3f}' for i in range(min(5, len(bins))))}")

    if "trial_aligned" in results:
        t = results["trial_aligned"]
        print(f"\n  [Trial-Aligned Mode]")
        print(f"    PSTH-R2:      {t.get('psth_r2', 'N/A')}")
        print(f"    trial fp-bps: {t.get('trial_fp_bps', 'N/A')}")
        if "per_target_psth_r2" in t:
            print(f"    per-target PSTH-R2:")
            for tid, r2 in sorted(t["per_target_psth_r2"].items()):
                n = t["trials_per_target"].get(tid, "?")
                print(f"      dir {tid}: R2={r2:.4f} (n={n})")

    print("=" * 60)

    # Save results
    if args.output:
        output_path = args.output
    else:
        ckpt_dir = Path(ckpt_path).parent.parent.parent
        output_path = str(ckpt_dir / "eval_v2_results.json")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
