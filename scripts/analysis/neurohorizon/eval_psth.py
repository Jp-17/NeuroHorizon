"""PSTH-R-squared evaluation script for NeuroHorizon.

Loads a trained model, runs trial-aligned inference, computes PSTH-R-squared
by grouping predictions by target direction (8 directions).

Usage:
    python scripts/analysis/neurohorizon/eval_psth.py \
        --checkpoint results/logs/phase1_small_250ms/.../last.ckpt \
        --split valid
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from torch_brain.data import Dataset, collate
from torch_brain.data.trial_sampler import TrialAlignedSampler
from torch_brain.models import NeuroHorizon
from torch_brain.transforms import Compose
from torch_brain.utils.neurohorizon_metrics import (
    psth_r2,
    fp_bps,
    build_null_rate_lookup,
    compute_null_rates,
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="PSTH-R2 evaluation for NeuroHorizon")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="valid", choices=["valid", "test"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--sigma_bins", type=int, default=1, help="Gaussian smoothing sigma for PSTH"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Load checkpoint and config
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = OmegaConf.create(ckpt["hyper_parameters"])

    # Build model
    model = hydra.utils.instantiate(cfg.model)

    # Setup train dataset (for vocab initialization and null rates)
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Setup eval dataset
    eval_transforms = (
        hydra.utils.instantiate(cfg.eval_transforms) if cfg.get("eval_transforms") else []
    )
    eval_dataset = Dataset(
        root=cfg.data_root,
        config=cfg.dataset,
        split=args.split,
        transform=Compose([*eval_transforms, model.tokenize]),
    )
    eval_dataset.disable_data_leakage_check()

    # Trial-aligned sampling
    trial_info = eval_dataset.get_trial_intervals(split=args.split)
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
        batch_size=args.batch_size,
        num_workers=0,
    )

    logger.info(
        f"Evaluating {len(sampler)} trials from {args.split} split"
    )

    # Compute null rates for fp-bps
    null_rates = compute_null_rates(train_dataset, model, model.bin_size)
    null_lookup = build_null_rate_lookup(null_rates, device=device)

    # Collect predictions grouped by target_id
    pred_by_target = defaultdict(list)
    true_by_target = defaultdict(list)
    all_fp_bps = []

    with torch.no_grad():
        for batch in loader:
            # Move to device
            inputs = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch["model_inputs"].items()
            }

            log_rate = model(**inputs)
            pred_rate = torch.exp(log_rate.clamp(-10, 10)).cpu()
            target = batch["target_spike_counts"].cpu()
            unit_mask = batch["model_inputs"]["target_unit_mask"].cpu()

            # fp-bps for this batch
            target_unit_index = batch["model_inputs"]["target_unit_index"]
            max_idx = null_lookup.shape[0] - 1
            null_log_rates = null_lookup[target_unit_index.clamp(0, max_idx).to(device)].cpu()
            batch_bps = fp_bps(log_rate.cpu(), target, null_log_rates, unit_mask)
            all_fp_bps.append(batch_bps.item())

            # Get trial metadata
            target_ids = batch.get("trial_target_id", None)
            if target_ids is None:
                logger.warning("No trial_target_id in batch, skipping PSTH grouping")
                continue

            # Group by target_id
            B = pred_rate.shape[0]
            for i in range(B):
                tid = int(target_ids[i])
                pred_by_target[tid].append(pred_rate[i])
                true_by_target[tid].append(target[i])

    # Compute PSTH-R-squared
    if pred_by_target:
        pred_stacked = {}
        true_stacked = {}
        for tid in sorted(pred_by_target.keys()):
            pred_stacked[tid] = torch.stack(pred_by_target[tid])
            true_stacked[tid] = torch.stack(true_by_target[tid])
            logger.info(f"  Target {tid}: {len(pred_by_target[tid])} trials")

        r2 = psth_r2(pred_stacked, true_stacked, sigma_bins=args.sigma_bins)
        logger.info(f"PSTH-R2 = {r2:.4f}")
    else:
        r2 = torch.tensor(float("nan"))
        logger.warning("No trial metadata available, PSTH-R2 not computed")

    # Mean fp-bps
    mean_bps = sum(all_fp_bps) / max(len(all_fp_bps), 1)
    logger.info(f"Mean fp-bps = {mean_bps:.4f}")

    # Save results
    output_dir = args.output_dir or str(Path(args.checkpoint).parent)
    results = {
        "psth_r2": float(r2),
        "fp_bps": mean_bps,
        "split": args.split,
        "sigma_bins": args.sigma_bins,
        "n_trials_per_target": {
            str(k): len(v) for k, v in pred_by_target.items()
        },
        "checkpoint": args.checkpoint,
    }

    output_path = Path(output_dir) / f"psth_r2_{args.split}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
