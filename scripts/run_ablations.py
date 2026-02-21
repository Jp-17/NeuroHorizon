"""Run ablation experiments for NeuroHorizon.

Trains multiple model variants to understand the contribution of each component:
1. IDEncoder vs random embeddings (controls for cross-session capability)
2. Different prediction horizons (100ms, 200ms, 500ms, 1000ms)
3. Poisson NLL vs MSE loss
4. With/without causal decoder self-attention

Usage:
    conda run -n poyo python scripts/run_ablations.py --ablation id_encoder
    conda run -n poyo python scripts/run_ablations.py --ablation horizon
    conda run -n poyo python scripts/run_ablations.py --ablation loss
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
TRAIN_SCRIPT = BASE_DIR / "examples" / "neurohorizon" / "train.py"


def run_training(name, overrides, epochs=50, config_name="train_v2_ibl"):
    """Run a training with specific overrides."""
    log_dir = f"./logs/ablation_{name}"
    cmd = [
        "conda", "run", "-n", "poyo", "python",
        str(TRAIN_SCRIPT),
        "--config-name", config_name,
        f"log_dir={log_dir}",
        f"epochs={epochs}",
    ] + [f"{k}={v}" for k, v in overrides.items()]

    logger.info(f"Running ablation: {name}")
    logger.info(f"  Config: {config_name}")
    logger.info(f"  Command: {' '.join(cmd)}")
    logger.info(f"  Log dir: {log_dir}")

    result = subprocess.run(cmd, cwd=str(BASE_DIR))
    if result.returncode != 0:
        logger.error(f"Training failed for {name}")
    return result.returncode


ABLATIONS = {
    "id_encoder": {
        "description": "IDEncoder vs random/mean embeddings",
        "config": "train_v2_ibl",
        "variants": {
            "idencoder": {},  # Default: IDEncoder MLP
            "random_emb": {"model.embedding_mode": "random"},
            "mean_emb": {"model.embedding_mode": "mean"},
        },
    },
    "horizon": {
        "description": "Different prediction horizons",
        "config": "train_v2_ibl",
        "variants": {
            "100ms": {"model.pred_length": 0.1},
            "200ms": {"model.pred_length": 0.2},
            "500ms": {"model.pred_length": 0.5},
            "1000ms": {"model.pred_length": 1.0},
        },
    },
    "bin_size": {
        "description": "Different time bin sizes",
        "config": "train_v2_ibl",
        "variants": {
            "10ms": {"model.bin_size": 0.01},
            "20ms": {"model.bin_size": 0.02},
            "50ms": {"model.bin_size": 0.05},
        },
    },
    "multimodal": {
        "description": "Multimodal contribution analysis (neural-only vs +behavior vs +image vs +both)",
        "variants": {
            "neural_only": {
                "__config__": "train_v2_ibl",
            },
            "plus_behavior": {
                "__config__": "train_v2",
            },
            "plus_image": {
                "__config__": "train_v2_mm",
                "model.behavior_dim": 0,
            },
            "plus_both": {
                "__config__": "train_v2_mm",
            },
        },
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ablation",
        type=str,
        choices=list(ABLATIONS.keys()) + ["all"],
        required=True,
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.ablation == "all":
        ablation_names = list(ABLATIONS.keys())
    else:
        ablation_names = [args.ablation]

    for abl_name in ablation_names:
        abl = ABLATIONS[abl_name]
        logger.info(f"\n{'='*60}")
        logger.info(f"Ablation: {abl['description']}")
        logger.info(f"{'='*60}")

        for variant_name, overrides in abl["variants"].items():
            full_name = f"{abl_name}_{variant_name}"
            # Determine config: per-variant override > ablation-level > default
            overrides = dict(overrides)  # copy
            config_name = overrides.pop("__config__", abl.get("config", "train_v2_ibl"))
            if args.dry_run:
                logger.info(f"  [DRY RUN] Would train: {full_name}")
                logger.info(f"    Config: {config_name}")
                logger.info(f"    Overrides: {overrides}")
            else:
                run_training(full_name, overrides, epochs=args.epochs,
                             config_name=config_name)


if __name__ == "__main__":
    main()
