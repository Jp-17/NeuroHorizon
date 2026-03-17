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
from collections import defaultdict
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf, DictConfig
from torch.utils.data import DataLoader

from torch_brain.data import Dataset, collate
from torch_brain.data.sampler import SequentialFixedWindowSampler
from torch_brain.data.trial_sampler import TrialAlignedSampler
from torch_brain.models import NeuroHorizon
from torch_brain.transforms import Compose
from torch_brain.utils.neurohorizon_metrics import (
    fp_bps_per_bin_stats,
    fp_bps_stats,
    finalize_fp_bps_from_stats,
    finalize_fp_bps_per_bin_from_stats,
    finalize_r2_from_stats,
    psth_r2,
    r2_stats,
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


def run_model(model, batch, device, rollout: bool = False):
    """Run the model in teacher-forced or free-running mode."""
    inputs = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch["model_inputs"].items()
    }
    if rollout:
        return model.generate(**inputs)
    if getattr(model, "requires_target_counts", False):
        inputs["target_counts"] = batch["target_spike_counts"].to(device)
    return model(**inputs)


def evaluate_continuous(
    model,
    cfg,
    train_dataset,
    null_lookup,
    device,
    batch_size=64,
    rollout=False,
    split="valid",
):
    """Continuous mode evaluation: fp-bps (overall + per-bin), R-squared."""
    logger.info("=== Continuous Mode Evaluation ===")

    eval_transforms = (
        hydra.utils.instantiate(cfg.eval_transforms) if cfg.get("eval_transforms") else []
    )
    eval_dataset = Dataset(
        root=cfg.data_root,
        config=cfg.dataset,
        split=split,
        transform=Compose([*eval_transforms, model.tokenize]),
    )
    eval_dataset.disable_data_leakage_check()

    sampler = SequentialFixedWindowSampler(
        sampling_intervals=eval_dataset.get_sampling_intervals(),
        window_length=model.sequence_length,
    )

    loader = DataLoader(
        eval_dataset,
        sampler=sampler,
        collate_fn=collate,
        batch_size=batch_size,
        num_workers=0,
    )

    logger.info(f"Continuous eval ({split}): {len(sampler)} samples")

    total_ss_res = torch.zeros((), device=device, dtype=torch.float64)
    total_target_sum = torch.zeros((), device=device, dtype=torch.float64)
    total_target_sq_sum = torch.zeros((), device=device, dtype=torch.float64)
    total_count = torch.zeros((), device=device, dtype=torch.float64)
    total_nll_model = torch.zeros((), device=device, dtype=torch.float64)
    total_nll_null = torch.zeros((), device=device, dtype=torch.float64)
    total_spikes = torch.zeros((), device=device, dtype=torch.float64)
    per_bin_model = None
    per_bin_null = None
    per_bin_spikes = None
    loss_sum = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            log_rate = run_model(model, batch, device, rollout=rollout)
            target = batch["target_spike_counts"].to(device)
            unit_mask = batch["model_inputs"]["target_unit_mask"].to(device)
            T = log_rate.shape[1]

            # Loss
            mask_3d = unit_mask.unsqueeze(1).expand(-1, T, -1)
            loss = (torch.exp(log_rate[mask_3d].clamp(-10, 10))
                    - target[mask_3d] * log_rate[mask_3d].clamp(-10, 10)).mean()
            loss_sum += loss.item()

            pred_rate = torch.exp(log_rate.clamp(-10, 10))
            ss_res, target_sum, target_sq_sum, count = r2_stats(pred_rate, target, unit_mask)
            total_ss_res += ss_res
            total_target_sum += target_sum
            total_target_sq_sum += target_sq_sum
            total_count += count

            target_unit_index = batch["model_inputs"]["target_unit_index"].to(device)
            max_idx = null_lookup.shape[0] - 1
            null_log_rates = null_lookup[target_unit_index.clamp(0, max_idx)]

            nll_model_sum, nll_null_sum, spikes_sum = fp_bps_stats(
                log_rate,
                target,
                null_log_rates,
                unit_mask,
            )
            total_nll_model += nll_model_sum
            total_nll_null += nll_null_sum
            total_spikes += spikes_sum

            pb_model, pb_null, pb_spikes = fp_bps_per_bin_stats(
                log_rate,
                target,
                null_log_rates,
                unit_mask,
            )
            if per_bin_model is None:
                per_bin_model = pb_model
                per_bin_null = pb_null
                per_bin_spikes = pb_spikes
            else:
                per_bin_model += pb_model
                per_bin_null += pb_null
                per_bin_spikes += pb_spikes

            n_batches += 1

    mean_bps = finalize_fp_bps_from_stats(total_nll_model, total_nll_null, total_spikes)
    mean_r2 = finalize_r2_from_stats(
        total_ss_res,
        total_target_sum,
        total_target_sq_sum,
        total_count,
    )
    mean_loss = loss_sum / max(n_batches, 1)
    per_bin_bps = finalize_fp_bps_per_bin_from_stats(
        per_bin_model,
        per_bin_null,
        per_bin_spikes,
    )
    per_bin_mean = {t: float(per_bin_bps[t]) for t in range(per_bin_bps.shape[0])}

    logger.info(f"  fp-bps = {float(mean_bps):.4f}")
    logger.info(f"  R2 = {float(mean_r2):.4f}")
    logger.info(f"  val_loss = {mean_loss:.4f}")
    logger.info(f"  per-bin fp-bps: {[f'{v:.3f}' for v in per_bin_mean.values()]}")

    return {
        "fp_bps": float(mean_bps),
        "r2": float(mean_r2),
        "val_loss": float(mean_loss),
        "per_bin_fp_bps": {str(k): float(v) for k, v in per_bin_mean.items()},
        "n_samples": len(sampler),
        "split": split,
    }


def evaluate_trial_aligned(model, cfg, train_dataset, null_lookup, device,
                           batch_size=16, sigma_bins=1, rollout=False, split="valid"):
    """Trial-aligned evaluation: per-neuron PSTH-R-squared by target direction."""
    logger.info("=== Trial-Aligned Evaluation (per-neuron PSTH-R2) ===")

    eval_transforms = (
        hydra.utils.instantiate(cfg.eval_transforms) if cfg.get("eval_transforms") else []
    )
    eval_dataset = Dataset(
        root=cfg.data_root,
        config=cfg.dataset,
        split=split,
        transform=Compose([*eval_transforms, model.tokenize]),
    )
    eval_dataset.disable_data_leakage_check()

    trial_info = eval_dataset.get_trial_intervals(split=split)
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

    logger.info(f"Trial-aligned eval ({split}): {len(sampler)} trials")

    pred_by_group = defaultdict(list)
    true_by_group = defaultdict(list)
    trials_per_target = defaultdict(int)
    total_nll_model = torch.zeros((), device=device, dtype=torch.float64)
    total_nll_null = torch.zeros((), device=device, dtype=torch.float64)
    total_spikes = torch.zeros((), device=device, dtype=torch.float64)

    with torch.no_grad():
        for batch in loader:
            log_rate = run_model(model, batch, device, rollout=rollout)
            pred_rate = torch.exp(log_rate.clamp(-10, 10)).cpu()
            target = batch["target_spike_counts"].cpu()
            unit_mask = batch["model_inputs"]["target_unit_mask"].cpu()

            target_unit_index = batch["model_inputs"]["target_unit_index"].to(device)
            max_idx = null_lookup.shape[0] - 1
            null_log_rates = null_lookup[target_unit_index.clamp(0, max_idx)]
            nll_model_sum, nll_null_sum, spikes_sum = fp_bps_stats(
                log_rate,
                batch["target_spike_counts"].to(device),
                null_log_rates,
                batch["model_inputs"]["target_unit_mask"].to(device),
            )
            total_nll_model += nll_model_sum
            total_nll_null += nll_null_sum
            total_spikes += spikes_sum

            target_ids = batch.get("trial_target_id", None)
            session_ids = batch.get("session_id", None)
            if target_ids is None:
                continue
            if session_ids is None:
                logger.warning("No session_id in batch, skipping PSTH grouping")
                continue

            B, T, N = pred_rate.shape
            for i in range(B):
                tid = int(target_ids[i])
                session_id = session_ids[i]
                mask_i = unit_mask[i]  # [N]
                if mask_i.sum().item() == 0:
                    continue
                group_key = (session_id, tid)
                pred_by_group[group_key].append(pred_rate[i][:, mask_i])
                true_by_group[group_key].append(target[i][:, mask_i])
                trials_per_target[str(tid)] += 1

    results = {
        "trial_fp_bps": float(
            finalize_fp_bps_from_stats(total_nll_model, total_nll_null, total_spikes)
        ) if total_spikes > 0 else 0.0,
        "n_trials": len(sampler),
        "split": split,
    }

    if pred_by_group:
        pred_stacked = {}
        true_stacked = {}
        per_target_r2 = {}

        for group_key in sorted(pred_by_group.keys()):
            pred_stacked[group_key] = torch.stack(pred_by_group[group_key])
            true_stacked[group_key] = torch.stack(true_by_group[group_key])
            session_id, tid = group_key
            logger.info(
                f"  Session {session_id}, Target {tid}: "
                f"{pred_stacked[group_key].shape[0]} trials"
            )

        overall_r2 = float(psth_r2(pred_stacked, true_stacked, sigma_bins=sigma_bins))

        target_ids = sorted({tid for _, tid in pred_stacked.keys()})
        for tid in target_ids:
            pred_sub = {
                key: value
                for key, value in pred_stacked.items()
                if key[1] == tid
            }
            true_sub = {
                key: value
                for key, value in true_stacked.items()
                if key[1] == tid
            }
            per_target_r2[str(tid)] = float(
                psth_r2(pred_sub, true_sub, sigma_bins=sigma_bins)
            )

        results["per_neuron_psth_r2"] = overall_r2
        results["per_target_per_neuron_psth_r2"] = per_target_r2
        results["trials_per_target"] = dict(trials_per_target)
        logger.info(f"  Overall per-neuron PSTH-R2 = {overall_r2:.4f}")
        logger.info(f"  Per-target per-neuron PSTH-R2: {per_target_r2}")
    else:
        results["per_neuron_psth_r2"] = float("nan")
        logger.warning("No trial metadata, per-neuron PSTH-R2 not computed")

    return results


def main():
    parser = argparse.ArgumentParser(description="Phase 1 v2 comprehensive evaluation")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint", type=str, help="Direct checkpoint path")
    group.add_argument("--log-dir", type=str, help="Log directory (auto-finds last.ckpt)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--trial-batch-size", type=int, default=16)
    parser.add_argument("--sigma-bins", type=int, default=1)
    parser.add_argument("--split", type=str, default="valid", choices=["valid", "test"])
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument("--skip-continuous", action="store_true")
    parser.add_argument("--skip-trial", action="store_true")
    parser.add_argument("--rollout", action="store_true", help="Use free-running generate() instead of teacher forcing")
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
        "rollout": bool(args.rollout),
        "split": args.split,
    }

    # Continuous evaluation
    if not args.skip_continuous:
        cont_results = evaluate_continuous(
            model, cfg, train_dataset, null_lookup, device,
            batch_size=args.batch_size,
            rollout=args.rollout,
            split=args.split,
        )
        results["continuous"] = cont_results

    # Trial-aligned evaluation
    if not args.skip_trial:
        trial_results = evaluate_trial_aligned(
            model, cfg, train_dataset, null_lookup, device,
            batch_size=args.trial_batch_size,
            sigma_bins=args.sigma_bins,
            rollout=args.rollout,
            split=args.split,
        )
        results["trial_aligned"] = trial_results

    # Print summary
    print("\n" + "=" * 60)
    print(f"EVALUATION SUMMARY")
    print(f"  Model: pred={results['pred_window_ms']}ms, "
          f"obs={results['hist_window_ms']}ms, "
          f"training={'trial-aligned' if results['trial_aligned_training'] else 'continuous'}, "
          f"eval={'rollout' if results['rollout'] else 'teacher-forced'}, "
          f"split={results['split']}")
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
        print(f"    per-neuron PSTH-R2: {t.get('per_neuron_psth_r2', 'N/A')}")
        print(f"    trial fp-bps: {t.get('trial_fp_bps', 'N/A')}")
        if "per_target_per_neuron_psth_r2" in t:
            print(f"    per-target per-neuron PSTH-R2:")
            for tid, r2 in sorted(t["per_target_per_neuron_psth_r2"].items()):
                n = t["trials_per_target"].get(tid, "?")
                print(f"      dir {tid}: R2={r2:.4f} (n={n})")

    print("=" * 60)

    # Save results
    if args.output:
        output_path = args.output
    else:
        ckpt_path_obj = Path(ckpt_path)
        if ckpt_path_obj.parent.name == "checkpoints" and ckpt_path_obj.parent.parent.name == "lightning_logs":
            ckpt_dir = ckpt_path_obj.parent.parent.parent
        elif ckpt_path_obj.parent.name == "checkpoints":
            ckpt_dir = ckpt_path_obj.parent.parent
        else:
            ckpt_dir = ckpt_path_obj.parent
        output_path = str(ckpt_dir / f"eval_v2_{args.split}_results.json")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
