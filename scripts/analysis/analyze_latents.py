"""
Phase 0.3.2: POYO encoder latent representation quality analysis
- Extract encoder latents from validation set
- PCA visualization (colored by session / movement phase)
- Linear decoding probe (cursor velocity R²)
Output: results/figures/baseline/latent_pca.png, latent_linear_probe.txt
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
import hydra
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from temporaldata import Data

# ---- repo root ----
repo_root = Path("/root/autodl-tmp/NeuroHorizon")
sys.path.insert(0, str(repo_root))

from torch_brain.data import Dataset, collate
from torch_brain.data.sampler import DistributedStitchingFixedWindowSampler
from torch_brain.models import POYOPlus
from torch_brain.transforms import Compose
from torch_brain.registry import MODALITY_REGISTRY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model_and_cfg(ckpt_path: str, config_dir: str, config_name: str):
    """Load trained POYOPlus model from checkpoint."""
    with hydra.initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        cfg = hydra.compose(config_name=config_name)

    # Build model
    model = hydra.utils.instantiate(cfg.model, readout_specs=MODALITY_REGISTRY)

    # Load dataset to initialize vocab
    val_dataset = Dataset(
        root=cfg.data_root,
        config=cfg.dataset,
        split="valid",
        transform=Compose([model.tokenize]),
    )
    val_dataset.disable_data_leakage_check()
    model.unit_emb.initialize_vocab(val_dataset.get_unit_ids())
    model.session_emb.initialize_vocab(val_dataset.get_session_ids())

    # Load checkpoint weights
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    # Lightning wraps model under 'model.' prefix
    new_sd = {}
    for k, v in state_dict.items():
        new_k = k[len("model."):] if k.startswith("model.") else k
        new_sd[new_k] = v
    model.load_state_dict(new_sd, strict=False)
    model.eval()
    return model, cfg, val_dataset


@torch.no_grad()
def extract_latents(model, val_dataset, batch_size=32, max_batches=100):
    """
    Run encoder on validation set windows; collect latent representations.
    Returns latents [N_windows, num_latents, dim] and metadata.
    """
    sampler = DistributedStitchingFixedWindowSampler(
        sampling_intervals=val_dataset.get_sampling_intervals(),
        window_length=model.sequence_length,
        step=model.sequence_length / 2,
        batch_size=batch_size,
        num_replicas=1,
        rank=0,
    )
    loader = DataLoader(
        val_dataset,
        sampler=sampler,
        shuffle=False,
        batch_size=batch_size,
        collate_fn=collate,
        num_workers=0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    all_latents = []       # mean-pooled latent per window [dim]
    all_session_ids = []
    all_targets = []       # cursor velocity targets

    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        inputs = batch["model_inputs"]
        # Move inputs to device
        inputs_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                      for k, v in inputs.items()}

        # Run encoder only (call model.encoder)
        # POYOPlus stores latents after perceiver encode step
        # We hook into the forward to extract latents
        # Simpler: run the full model and grab intermediate latent via hook

        latents_captured = {}

        def hook_fn(module, inp, out):
            latents_captured["latents"] = out.detach().cpu()

        # Register hook on the perceiver (encoder output)
        hook = model.perceiver.register_forward_hook(hook_fn)

        _ = model(**inputs_dev, unpack_output=False)
        hook.remove()

        if "latents" in latents_captured:
            lat = latents_captured["latents"]  # [B, num_latents*T_latent, dim]
            # Mean pool over latent sequence
            pooled = lat.mean(dim=1)  # [B, dim]
            all_latents.append(pooled)
            all_session_ids.extend(batch.get("session_id", ["unknown"] * len(pooled)))
            # Target cursor velocity (if available)
            if "cursor_velocity_2d" in batch.get("target_values", {}):
                tgt = batch["target_values"]["cursor_velocity_2d"]
                # Mean over time for this window
                all_targets.append(tgt.mean(dim=1).cpu())

    if not all_latents:
        logger.error("No latents captured - check hook target module name")
        return None, None, None

    latents = torch.cat(all_latents, dim=0).numpy()   # [N, dim]
    targets = torch.cat(all_targets, dim=0).numpy() if all_targets else None  # [N, 2]
    return latents, all_session_ids, targets


def pca_and_plot(latents, session_ids, out_dir):
    """PCA of latents, colored by session."""
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    coords = pca.fit_transform(latents)
    var_ratio = pca.explained_variance_ratio_

    unique_sessions = sorted(set(session_ids))
    colors = cm.tab10(np.linspace(0, 1, len(unique_sessions)))
    sess2color = {s: c for s, c in zip(unique_sessions, colors)}

    fig, ax = plt.subplots(figsize=(10, 8))
    for sess in unique_sessions:
        idx = [i for i, s in enumerate(session_ids) if s == sess]
        short = sess.split("/")[-1][:30]
        ax.scatter(coords[idx, 0], coords[idx, 1],
                   color=sess2color[sess], label=short, alpha=0.5, s=10)

    ax.set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}% var)")
    ax.set_title("POYO+ Encoder Latent PCA (colored by session)")
    ax.legend(fontsize=6, loc="upper right", markerscale=2)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "latent_pca.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info(f"Saved PCA plot to {out_path}")
    return var_ratio


def linear_probe(latents, targets, out_dir):
    """Linear regression from latents to cursor velocity; report R²."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_score

    if targets is None or len(targets) == 0:
        logger.warning("No cursor velocity targets available for linear probe")
        return

    # Cross-validated Ridge regression
    clf = Ridge(alpha=1.0)
    scores = cross_val_score(clf, latents, targets, cv=5, scoring="r2")
    mean_r2 = scores.mean()
    std_r2 = scores.std()

    result = {
        "linear_probe_r2_mean": float(mean_r2),
        "linear_probe_r2_std": float(std_r2),
        "cv_scores": scores.tolist(),
        "note": "5-fold CV Ridge regression from encoder latents to cursor velocity"
    }

    out_path = os.path.join(out_dir, "latent_linear_probe.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"Linear probe R² = {mean_r2:.3f} ± {std_r2:.3f}")
    logger.info(f"Saved probe results to {out_path}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint (.ckpt)")
    parser.add_argument("--config_dir",
                        default="/root/autodl-tmp/NeuroHorizon/examples/poyo_plus/configs")
    parser.add_argument("--config_name", default="train_baseline_10sessions")
    parser.add_argument("--out_dir",
                        default="/root/autodl-tmp/NeuroHorizon/results/figures/baseline")
    parser.add_argument("--max_batches", type=int, default=80)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    logger.info("Loading model from checkpoint...")
    model, cfg, val_dataset = load_model_and_cfg(
        args.ckpt, args.config_dir, args.config_name)

    logger.info("Extracting latents from validation set...")
    latents, session_ids, targets = extract_latents(
        model, val_dataset, max_batches=args.max_batches)

    if latents is None:
        logger.error("Failed to extract latents")
        sys.exit(1)

    logger.info(f"Extracted {len(latents)} windows, latent dim={latents.shape[1]}")

    logger.info("Running PCA...")
    var_ratio = pca_and_plot(latents, session_ids, args.out_dir)
    logger.info(f"PCA variance explained: PC1={var_ratio[0]*100:.1f}%, PC2={var_ratio[1]*100:.1f}%")

    logger.info("Running linear probe...")
    probe_result = linear_probe(latents, targets, args.out_dir)

    # Summary
    summary = {
        "n_windows": len(latents),
        "latent_dim": latents.shape[1],
        "pca_var_pc1": float(var_ratio[0]),
        "pca_var_pc2": float(var_ratio[1]),
    }
    if probe_result:
        summary.update(probe_result)

    with open(os.path.join(args.out_dir, "latent_analysis_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Done! Summary:")
    for k, v in summary.items():
        logger.info(f"  {k}: {v}")


if __name__ == "__main__":
    main()
