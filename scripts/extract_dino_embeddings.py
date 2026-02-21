#!/usr/bin/env python3
"""Extract DINOv2 embeddings from Allen natural movie and scene stimuli.

Uses DINOv2 ViT-B/14 to extract CLS token embeddings for each frame/image.
Saves embeddings as .pt files for use in multimodal NeuroHorizon training.

Usage:
    conda run -n poyo python scripts/extract_dino_embeddings.py
"""

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def load_dinov2(model_name="dinov2_vitb14", device="cuda"):
    """Load DINOv2 model from torch hub."""
    logger.info(f"Loading DINOv2 model: {model_name}")
    model = torch.hub.load("facebookresearch/dinov2", model_name)
    model = model.to(device)
    model.eval()

    # DINOv2 ViT-B/14 outputs 768-dim CLS token
    logger.info(f"Model loaded. Embedding dim: 768")
    return model


def get_transform():
    """Get the DINOv2 preprocessing transform."""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


@torch.no_grad()
def extract_embeddings(model, frames, transform, device="cuda", batch_size=32):
    """Extract DINOv2 CLS embeddings from a batch of frames.

    Args:
        model: DINOv2 model
        frames: (N, H, W) uint8 grayscale or (N, H, W, 3) uint8 RGB
        transform: torchvision transform
        device: cuda/cpu
        batch_size: batch size for inference

    Returns:
        embeddings: (N, 768) float32 tensor
    """
    n_frames = len(frames)
    all_embeddings = []

    for start in range(0, n_frames, batch_size):
        end = min(start + batch_size, n_frames)
        batch_frames = frames[start:end]

        # Convert grayscale to RGB if needed
        batch_tensors = []
        for frame in batch_frames:
            if frame.ndim == 2:
                # Grayscale -> RGB by repeating channels
                frame_rgb = np.stack([frame, frame, frame], axis=-1)
            elif frame.ndim == 3 and frame.shape[-1] == 1:
                frame_rgb = np.repeat(frame, 3, axis=-1)
            else:
                frame_rgb = frame
            batch_tensors.append(transform(frame_rgb))

        batch_tensor = torch.stack(batch_tensors).to(device)

        # Extract CLS token embedding
        features = model(batch_tensor)  # (B, 768)
        all_embeddings.append(features.cpu())

        if (start // batch_size) % 10 == 0:
            logger.info(f"  Processed {end}/{n_frames} frames")

    return torch.cat(all_embeddings, dim=0)


def extract_movie_embeddings(cache, movie_number, model, transform, device, output_dir):
    """Extract embeddings for a natural movie."""
    logger.info(f"Loading Natural Movie {movie_number}...")
    frames = cache.get_natural_movie_template(movie_number)
    logger.info(f"  Shape: {frames.shape}, dtype: {frames.dtype}")

    embeddings = extract_embeddings(model, frames, transform, device=device)
    logger.info(f"  Embeddings shape: {embeddings.shape}")

    output_path = output_dir / f"natural_movie_{movie_number}_dinov2.pt"
    torch.save(embeddings, output_path)
    logger.info(f"  Saved to {output_path}")
    return embeddings


def extract_scene_embeddings(cache, model, transform, device, output_dir):
    """Extract embeddings for natural scenes."""
    logger.info("Loading Natural Scenes...")
    # Allen has 118 natural scene images (numbered 0-117)
    all_frames = []
    for i in range(118):
        try:
            frame = cache.get_natural_scene_template(i)
            all_frames.append(frame)
        except Exception as e:
            logger.warning(f"  Failed to load scene {i}: {e}")

    if not all_frames:
        logger.error("No scene images loaded!")
        return None

    frames = np.stack(all_frames)
    logger.info(f"  Shape: {frames.shape}, dtype: {frames.dtype}")

    embeddings = extract_embeddings(model, frames, transform, device=device)
    logger.info(f"  Embeddings shape: {embeddings.shape}")

    output_path = output_dir / "natural_scenes_dinov2.pt"
    torch.save(embeddings, output_path)
    logger.info(f"  Saved to {output_path}")
    return embeddings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=str,
                        default="/root/autodl-tmp/allen_cache")
    parser.add_argument("--output-dir", type=str,
                        default="/root/autodl-tmp/datasets/allen_embeddings")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--movies", nargs="*", type=int, default=[1, 3],
                        help="Movie numbers to process (default: 1 3)")
    parser.add_argument("--scenes", action="store_true", default=True,
                        help="Also extract scene embeddings")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device if torch.cuda.is_available() else "cpu"
    model = load_dinov2(device=device)
    transform = get_transform()

    # Load Allen cache (only needed for movie/scene templates)
    # Import in function scope to avoid allensdk dependency when not needed
    try:
        from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
        cache = EcephysProjectCache.from_warehouse(
            manifest=os.path.join(args.cache_dir, "manifest.json")
        )
    except ImportError:
        logger.error("allensdk not available. Install with: pip install allensdk")
        logger.info("Alternatively, provide pre-extracted frame arrays.")
        return

    # Extract movie embeddings
    for movie_num in args.movies:
        try:
            extract_movie_embeddings(cache, movie_num, model, transform, device, output_dir)
        except Exception as e:
            logger.error(f"Failed to extract movie {movie_num}: {e}")

    # Extract scene embeddings
    if args.scenes:
        try:
            extract_scene_embeddings(cache, model, transform, device, output_dir)
        except Exception as e:
            logger.error(f"Failed to extract scenes: {e}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
