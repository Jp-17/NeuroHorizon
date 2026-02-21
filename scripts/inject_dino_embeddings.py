#!/usr/bin/env python3
"""Inject DINOv2 embeddings into Allen HDF5 files.

After running extract_dino_embeddings.py to get .pt embedding files,
this script adds the embeddings to the Allen HDF5 files as an 'images' group
with fields:
  - images.embeddings: (N_presentations, 768) float32
  - images.timestamps: (N_presentations,) float64

For natural movies, each frame presentation gets the embedding of that frame.
The frame -> embedding mapping uses the 'frame' field from natural_movie_* groups.

Usage:
    python scripts/inject_dino_embeddings.py
    python scripts/inject_dino_embeddings.py --embedding-dir /path/to/embeddings
"""

import argparse
import logging
from pathlib import Path

import h5py
import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def inject_movie_embeddings(h5_path, embedding_path, movie_group_name):
    """Inject movie frame embeddings into HDF5 file.

    Maps frame indices from the movie presentation table to DINOv2 embeddings.
    """
    embeddings = torch.load(embedding_path, map_location="cpu").numpy()
    n_unique_frames = len(embeddings)
    logger.info(f"  Loaded {n_unique_frames} unique frame embeddings from {embedding_path}")

    with h5py.File(h5_path, "r+") as f:
        if movie_group_name not in f:
            logger.warning(f"  {movie_group_name} not found in {h5_path}")
            return False

        movie_grp = f[movie_group_name]
        frame_indices = np.array(movie_grp["frame"])
        starts = np.array(movie_grp["start"])

        # Use frame midpoints as timestamps
        ends = np.array(movie_grp["end"])
        timestamps = (starts + ends) / 2.0

        n_presentations = len(frame_indices)
        logger.info(f"  {movie_group_name}: {n_presentations} presentations, "
                     f"frames 0-{frame_indices.max()}")

        # Map each presentation to its frame embedding
        # Clip frame indices to valid range
        valid_mask = frame_indices < n_unique_frames
        if not valid_mask.all():
            n_invalid = (~valid_mask).sum()
            logger.warning(f"  {n_invalid} presentations have invalid frame indices, clipping")
            frame_indices = np.clip(frame_indices, 0, n_unique_frames - 1)

        presentation_embeddings = embeddings[frame_indices]  # (N_pres, 768)

        # Store in images group (create or append)
        if "images" not in f:
            img_grp = f.create_group("images")
            img_grp.attrs["object"] = "IrregularTimeSeries"
            img_grp.attrs["timekeys"] = np.array([b"timestamps"])
            img_grp.attrs["_unicode_keys"] = np.array([])

            # Create domain subgroup
            dom = img_grp.create_group("domain")
            dom.attrs["object"] = "Interval"
            dom.attrs["timekeys"] = np.array([b"start", b"end"])
            dom.attrs["_unicode_keys"] = np.array([])
            dom.attrs["allow_split_mask_overlap"] = False
            dom.create_dataset("start", data=np.array([timestamps.min()]))
            dom.create_dataset("end", data=np.array([timestamps.max()]))

            img_grp.create_dataset("embeddings", data=presentation_embeddings,
                                   dtype=np.float32)
            img_grp.create_dataset("timestamps", data=timestamps, dtype=np.float64)
            img_grp.create_dataset("source_movie", data=np.full(n_presentations,
                                   movie_group_name, dtype=h5py.string_dtype()))
        else:
            # Append to existing images group
            img_grp = f["images"]
            existing_emb = np.array(img_grp["embeddings"])
            existing_ts = np.array(img_grp["timestamps"])

            new_emb = np.concatenate([existing_emb, presentation_embeddings], axis=0)
            new_ts = np.concatenate([existing_ts, timestamps], axis=0)

            # Sort by timestamp
            sort_idx = np.argsort(new_ts)
            new_emb = new_emb[sort_idx]
            new_ts = new_ts[sort_idx]

            del img_grp["embeddings"]
            del img_grp["timestamps"]
            img_grp.create_dataset("embeddings", data=new_emb, dtype=np.float32)
            img_grp.create_dataset("timestamps", data=new_ts, dtype=np.float64)

            # Update domain
            img_grp["domain"]["start"][...] = np.array([new_ts.min()])
            img_grp["domain"]["end"][...] = np.array([new_ts.max()])

        logger.info(f"  Injected {n_presentations} embeddings for {movie_group_name}")
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--allen-dir", type=str,
                        default="/root/autodl-tmp/datasets/allen_processed")
    parser.add_argument("--embedding-dir", type=str,
                        default="/root/autodl-tmp/datasets/allen_embeddings")
    args = parser.parse_args()

    allen_dir = Path(args.allen_dir)
    emb_dir = Path(args.embedding_dir)

    # Find available embeddings
    movie_embeddings = {}
    for pt_file in emb_dir.glob("natural_movie_*_dinov2.pt"):
        # Parse movie number from filename: natural_movie_1_dinov2.pt -> 1
        parts = pt_file.stem.split("_")
        movie_num = parts[2]  # "1" or "3"
        movie_group = f"natural_movie_{'one' if movie_num == '1' else 'three' if movie_num == '3' else movie_num}"
        movie_embeddings[movie_group] = pt_file

    if not movie_embeddings:
        logger.error(f"No embedding files found in {emb_dir}")
        logger.info("Run scripts/extract_dino_embeddings.py first")
        return

    logger.info(f"Found embeddings for: {list(movie_embeddings.keys())}")

    # Process each Allen HDF5 file
    for h5_file in sorted(allen_dir.glob("allen_*.h5")):
        logger.info(f"\nProcessing {h5_file.name}...")

        # Remove existing images group for clean injection
        with h5py.File(h5_file, "r+") as f:
            if "images" in f:
                del f["images"]
                logger.info("  Removed existing images group")

        for movie_group, emb_path in movie_embeddings.items():
            inject_movie_embeddings(h5_file, emb_path, movie_group)

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
