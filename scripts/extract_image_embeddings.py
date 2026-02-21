"""Extract visual embeddings from Allen stimulus frames using CLIP ViT-L/14.

Uses the cached CLIP ViT-L/14 model (from timm) to extract embeddings from
Allen Brain Observatory stimulus frames (natural movies and scenes).

DINOv2 requires network access (unavailable), so we use CLIP ViT-L/14 which
is already cached locally. Both produce high-quality visual representations.

Output: per-frame embeddings saved as .npy files for injection into training.

Usage:
    conda run -n poyo python scripts/extract_image_embeddings.py
    conda run -n poyo python scripts/extract_image_embeddings.py --stimulus natural_movie_1
"""

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

FRAMES_DIR = Path("/root/autodl-tmp/datasets/allen_embeddings")

STIMULI = {
    "natural_movie_1": {
        "frames_file": "natural_movie_1_frames.npy",
        "n_frames": 900,
        "fps": 30,
        "duration": 30.0,
    },
    "natural_movie_3": {
        "frames_file": "natural_movie_3_frames.npy",
        "n_frames": 3600,
        "fps": 30,
        "duration": 120.0,
    },
    "natural_scenes": {
        "frames_file": "natural_scenes_frames.npy",
        "n_frames": 118,
    },
}


def load_clip_vit(device="cuda"):
    """Load CLIP ViT-L/14 from local HF cache (no network required)."""
    import timm
    from safetensors.torch import load_file

    # Create model architecture without pretrained weights
    model = timm.create_model("vit_large_patch14_clip_224.openai", pretrained=False)

    # Load weights from cached safetensors
    cache_dir = os.path.expanduser(
        "~/.cache/huggingface/hub/models--timm--vit_large_patch14_clip_224.openai"
    )
    blobs_dir = os.path.join(cache_dir, "blobs")
    blob_file = os.path.join(blobs_dir, os.listdir(blobs_dir)[0])
    state_dict = load_file(blob_file)
    model.load_state_dict(state_dict, strict=False)

    model = model.to(device)
    model.eval()
    logger.info(
        f"CLIP ViT-L/14 loaded: {sum(p.numel() for p in model.parameters()):,} params"
    )
    return model


def preprocess_frames(frames, target_size=224):
    """Preprocess stimulus frames for CLIP ViT.

    Args:
        frames: numpy array of shape (N, H, W) (grayscale) or (N, H, W, 3) (RGB)
        target_size: target image size for ViT

    Returns:
        torch.Tensor of shape (N, 3, target_size, target_size), normalized
    """
    from PIL import Image

    processed = []
    for i in range(len(frames)):
        frame = frames[i]
        # Convert grayscale to RGB
        if frame.ndim == 2:
            frame = np.stack([frame] * 3, axis=-1)

        # Resize using PIL
        img = Image.fromarray(frame)
        img = img.resize((target_size, target_size), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0

        # Normalize with ImageNet stats (CLIP uses these)
        mean = np.array([0.48145466, 0.4578275, 0.40821073])
        std = np.array([0.26862954, 0.26130258, 0.27577711])
        arr = (arr - mean) / std

        # HWC -> CHW
        processed.append(arr.transpose(2, 0, 1))

    return torch.tensor(np.stack(processed), dtype=torch.float32)


def extract_embeddings(model, frames_tensor, batch_size=32, device="cuda"):
    """Extract CLS token embeddings from CLIP ViT.

    Args:
        model: CLIP ViT model
        frames_tensor: (N, 3, 224, 224) preprocessed frames
        batch_size: batch size for inference

    Returns:
        numpy array of shape (N, 1024) - CLS token embeddings
    """
    embeddings = []
    n_frames = len(frames_tensor)

    for i in range(0, n_frames, batch_size):
        batch = frames_tensor[i : i + batch_size].to(device)
        with torch.no_grad(), torch.amp.autocast("cuda"):
            features = model.forward_features(batch)
            # CLS token is the first token
            cls_emb = features[:, 0]  # (B, 1024)
            embeddings.append(cls_emb.cpu().float().numpy())

        if (i // batch_size) % 10 == 0:
            logger.info(f"  Processed {min(i + batch_size, n_frames)}/{n_frames} frames")

    return np.concatenate(embeddings, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stimulus",
        type=str,
        choices=list(STIMULI.keys()) + ["all"],
        default="all",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(FRAMES_DIR),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    stimuli = list(STIMULI.keys()) if args.stimulus == "all" else [args.stimulus]

    # Load model
    model = load_clip_vit(device=args.device)

    for stim_name in stimuli:
        stim_info = STIMULI[stim_name]
        frames_path = FRAMES_DIR / stim_info["frames_file"]

        if not frames_path.exists():
            logger.warning(f"Frames file not found: {frames_path}")
            continue

        logger.info(f"Processing {stim_name}...")
        frames = np.load(str(frames_path))
        logger.info(f"  Loaded {frames.shape[0]} frames, shape {frames.shape}")

        # Preprocess
        frames_tensor = preprocess_frames(frames)
        logger.info(f"  Preprocessed to {frames_tensor.shape}")

        # Extract embeddings
        embeddings = extract_embeddings(
            model, frames_tensor, batch_size=args.batch_size, device=args.device
        )
        logger.info(f"  Embeddings shape: {embeddings.shape}")

        # Save
        out_path = output_dir / f"{stim_name}_embeddings.npy"
        np.save(str(out_path), embeddings)
        logger.info(f"  Saved to {out_path}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
