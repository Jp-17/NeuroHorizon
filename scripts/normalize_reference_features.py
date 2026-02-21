#!/usr/bin/env python3
"""Normalize reference features across all sessions.

Computes z-score normalization statistics (mean, std per feature)
across all units from all sessions, then updates the HDF5 files
with normalized features and saves the normalization stats.

Usage:
    conda run -n poyo python scripts/normalize_reference_features.py
"""

import argparse
import json
import logging
from pathlib import Path

import h5py
import numpy as np
from temporaldata import Data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def compute_normalization_stats(data_dir):
    """Compute per-feature mean and std across all units in all sessions."""
    data_dir = Path(data_dir)
    all_features = []

    for fpath in sorted(data_dir.glob("*.h5")):
        with h5py.File(fpath, "r") as f:
            data = Data.from_hdf5(f, lazy=False)
        rf = np.array(data.units.reference_features, dtype=np.float64)
        all_features.append(rf)
        logger.info(f"  {fpath.stem[:8]}: {rf.shape[0]} units")

    all_rf = np.concatenate(all_features, axis=0)
    logger.info(f"Total: {all_rf.shape[0]} units, {all_rf.shape[1]} features")

    feat_mean = all_rf.mean(axis=0).astype(np.float32)
    feat_std = all_rf.std(axis=0).astype(np.float32)
    # Prevent division by zero
    feat_std[feat_std < 1e-6] = 1.0

    return feat_mean, feat_std


def normalize_hdf5_files(data_dir, feat_mean, feat_std):
    """Update HDF5 files with normalized reference features."""
    data_dir = Path(data_dir)

    for fpath in sorted(data_dir.glob("*.h5")):
        with h5py.File(fpath, "r+") as f:
            rf = np.array(f["units"]["reference_features"])
            rf_normalized = (rf - feat_mean) / feat_std

            # Save normalized features (overwrite or create new dataset)
            if "reference_features_raw" not in f["units"]:
                # Backup original
                f["units"].create_dataset(
                    "reference_features_raw",
                    data=rf,
                    dtype=np.float32,
                )

            # Overwrite with normalized
            f["units"]["reference_features"][...] = rf_normalized.astype(np.float32)

        logger.info(f"  Normalized {fpath.stem[:8]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/root/autodl-tmp/datasets/ibl_processed",
    )
    parser.add_argument("--stats-output", type=str, default=None,
                        help="Save normalization stats to JSON")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only compute stats, don't modify files")
    args = parser.parse_args()

    logger.info("Computing normalization statistics...")
    feat_mean, feat_std = compute_normalization_stats(args.data_dir)

    logger.info("\nPer-feature normalization stats:")
    feature_names = (
        ["firing_rate", "isi_cv"]
        + [f"isi_hist_{i}" for i in range(20)]
        + [f"autocorr_{i}" for i in range(10)]
        + ["fano_factor"]
    )
    for i in range(len(feat_mean)):
        name = feature_names[i] if i < len(feature_names) else f"feat_{i}"
        logger.info(f"  {name:<15s}: mean={feat_mean[i]:10.4f}, std={feat_std[i]:10.4f}")

    # Save stats
    stats_path = args.stats_output or str(Path(args.data_dir) / "ref_feature_stats.json")
    stats = {
        "mean": feat_mean.tolist(),
        "std": feat_std.tolist(),
        "feature_names": feature_names[:len(feat_mean)],
    }
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"\nStats saved to {stats_path}")

    if not args.dry_run:
        logger.info("\nNormalizing HDF5 files...")
        normalize_hdf5_files(args.data_dir, feat_mean, feat_std)
        logger.info("Done! Original features backed up as 'reference_features_raw'.")
    else:
        logger.info("\nDry run - no files modified.")


if __name__ == "__main__":
    main()
