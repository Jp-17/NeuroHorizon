"""Inject image embeddings into Allen HDF5 files.

After extracting embeddings with extract_image_embeddings.py, this script
adds them to the Allen HDF5 files so the NeuroHorizon tokenizer can
automatically include them during training.

Allen HDF5 stores stimulus presentations as:
  natural_movie_one/start, end, frame  (frame indices, repeated for each presentation)
  natural_movie_three/start, end, frame
  natural_scenes/start, end, frame

This script:
1. Reads the per-frame embeddings (from extract_image_embeddings.py)
2. For each presentation, looks up the embedding for that frame
3. Stores expanded embeddings with onset timestamps

Adds to each Allen HDF5:
- image_embeddings/embeddings: (N_presentations, embed_dim) float32
- image_embeddings/timestamps: (N_presentations,) float64 - onset time
- image_embeddings/domain: start/end arrays

Usage:
    conda run -n poyo python scripts/inject_image_embeddings.py
"""

import argparse
import logging
from pathlib import Path

import h5py
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

ALLEN_DIR = Path(
    "/root/autodl-tmp/datasets/allen_processed"
)
EMBEDDINGS_DIR = Path(
    "/root/autodl-tmp/datasets/allen_embeddings"
)

# Map between our naming and Allen HDF5 group names
STIM_KEY_MAP = {
    "natural_movie_1": "natural_movie_one",
    "natural_movie_3": "natural_movie_three",
    "natural_scenes": "natural_scenes",
}


def get_presentations(h5f, stim_name):
    """Get frame presentations (onset times + frame indices) from Allen HDF5.

    Returns:
        Tuple of (onset_times, frame_indices) numpy arrays, or (None, None)
    """
    h5_key = STIM_KEY_MAP.get(stim_name, stim_name)
    if h5_key not in h5f:
        return None, None

    grp = h5f[h5_key]
    if "start" not in grp or "frame" not in grp:
        return None, None

    onset_times = np.array(grp["start"])
    frame_indices = np.array(grp["frame"])
    return onset_times, frame_indices


def inject_embeddings(h5_path, stim_data):
    """Add image embeddings to an Allen HDF5 file.

    Args:
        h5_path: Path to Allen HDF5 file
        stim_data: Dict mapping stim_name -> (frame_embeddings, onset_times, frame_indices)
    """
    with h5py.File(str(h5_path), "a") as f:
        # Remove existing embeddings if present
        if "image_embeddings" in f:
            del f["image_embeddings"]

        all_embeddings = []
        all_timestamps = []

        for stim_name, (frame_embeddings, onset_times, frame_indices) in sorted(
            stim_data.items()
        ):
            # Filter out invalid frames (natural_scenes has frame=-1 for blanks)
            valid_mask = frame_indices >= 0
            valid_mask &= frame_indices < len(frame_embeddings)
            onset_times = onset_times[valid_mask]
            frame_indices = frame_indices[valid_mask]

            # Look up embedding for each presentation
            presentation_embeddings = frame_embeddings[frame_indices]
            all_embeddings.append(presentation_embeddings)
            all_timestamps.append(onset_times)

            logger.info(
                f"  {stim_name}: {len(onset_times)} valid presentations "
                f"({len(frame_embeddings)} unique frames)"
            )

        if not all_embeddings:
            logger.warning(f"  No embeddings to inject for {h5_path.name}")
            return

        all_embeddings = np.concatenate(all_embeddings, axis=0)
        all_timestamps = np.concatenate(all_timestamps, axis=0)

        # Sort by timestamp
        sort_idx = np.argsort(all_timestamps)
        all_embeddings = all_embeddings[sort_idx]
        all_timestamps = all_timestamps[sort_idx]

        # Write to HDF5
        grp = f.create_group("image_embeddings")
        grp.create_dataset(
            "embeddings", data=all_embeddings.astype(np.float32), chunks=True
        )
        grp.create_dataset("timestamps", data=all_timestamps.astype(np.float64))

        # Add domain
        domain_grp = grp.create_group("domain")
        domain_grp.create_dataset(
            "start", data=np.array([all_timestamps[0]], dtype=np.float64)
        )
        domain_grp.create_dataset(
            "end", data=np.array([all_timestamps[-1]], dtype=np.float64)
        )

        # Add temporaldata-compatible metadata (must match format of other groups)
        grp.attrs["object"] = "IrregularTimeSeries"
        grp.attrs["timekeys"] = np.array([b"timestamps"], dtype="S10")
        grp.attrs["_unicode_keys"] = np.array([], dtype="S1")

        # Domain subgroup also needs temporaldata metadata
        domain_grp.attrs["object"] = "Interval"
        domain_grp.attrs["timekeys"] = np.array([b"start", b"end"], dtype="S5")
        domain_grp.attrs["_unicode_keys"] = np.array([], dtype="S1")
        domain_grp.attrs["allow_split_mask_overlap"] = False

        logger.info(
            f"  Total: {len(all_embeddings)} embeddings "
            f"(dim={all_embeddings.shape[1]}), "
            f"time range [{all_timestamps[0]:.1f}, {all_timestamps[-1]:.1f}]s"
        )
        logger.info(
            f"  Storage: {all_embeddings.nbytes / 1e6:.1f} MB"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--allen-dir", type=str, default=str(ALLEN_DIR))
    parser.add_argument("--embeddings-dir", type=str, default=str(EMBEDDINGS_DIR))
    args = parser.parse_args()

    allen_dir = Path(args.allen_dir)
    embeddings_dir = Path(args.embeddings_dir)

    # Load precomputed per-frame embeddings
    available_embeddings = {}
    for stim_name in ["natural_movie_1", "natural_movie_3", "natural_scenes"]:
        emb_path = embeddings_dir / f"{stim_name}_embeddings.npy"
        if emb_path.exists():
            available_embeddings[stim_name] = np.load(str(emb_path))
            logger.info(
                f"Loaded {stim_name} embeddings: "
                f"{available_embeddings[stim_name].shape}"
            )
        else:
            logger.warning(f"Embeddings not found: {emb_path}")

    if not available_embeddings:
        logger.error("No embeddings found. Run extract_image_embeddings.py first.")
        return

    # Process each Allen session
    for h5_path in sorted(allen_dir.glob("*.h5")):
        logger.info(f"Processing {h5_path.name}...")

        with h5py.File(str(h5_path), "r") as f:
            stim_data = {}
            for stim_name, frame_embeddings in available_embeddings.items():
                onset_times, frame_indices = get_presentations(f, stim_name)
                if onset_times is not None:
                    stim_data[stim_name] = (
                        frame_embeddings,
                        onset_times,
                        frame_indices,
                    )

        if stim_data:
            inject_embeddings(h5_path, stim_data)
        else:
            logger.warning(f"  No matching stimulus data in {h5_path.name}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
