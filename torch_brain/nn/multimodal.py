"""Multimodal Cross-Attention for NeuroHorizon.

Injects external modality information (images, behavior) into the
encoder's latent representations via cross-attention layers.

Modalities supported:
- Image embeddings (DINOv2 CLS tokens) with temporal alignment
- Behavior signals (running speed, wheel velocity) as continuous time series
"""

from typing import Optional

import torch
import torch.nn as nn

from torch_brain.nn import FeedForward, RotaryTimeEmbedding
from torch_brain.nn.rotary_attention import RotaryCrossAttention


class MultimodalCrossAttention(nn.Module):
    """Cross-attention layer for injecting multimodal information.

    Takes encoder latent representations and attends to projected
    modality embeddings. Supports variable-length modality sequences.

    Args:
        dim: Model dimension (latent and output dimension).
        modality_dim: Input dimension of the modality embeddings.
        heads: Number of attention heads.
        dim_head: Dimension per attention head.
        dropout: Attention dropout.
        ffn_dropout: Feed-forward dropout.
    """

    def __init__(
        self,
        *,
        dim: int,
        modality_dim: int,
        heads: int = 4,
        dim_head: int = 64,
        dropout: float = 0.0,
        ffn_dropout: float = 0.2,
    ):
        super().__init__()

        # Project modality embeddings to model dimension
        self.modality_proj = nn.Sequential(
            nn.Linear(modality_dim, dim),
            nn.LayerNorm(dim),
        )

        # Cross-attention: latents attend to modality
        self.cross_attn = RotaryCrossAttention(
            dim=dim,
            context_dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
            rotate_value=False,
        )

        # Feed-forward after cross-attention
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim=dim, dropout=ffn_dropout),
        )

    def forward(
        self,
        latents: torch.Tensor,
        latent_time_emb: torch.Tensor,
        modality_embeddings: torch.Tensor,
        modality_time_emb: torch.Tensor,
        modality_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            latents: (B, N_latent, dim) encoder latent representations
            latent_time_emb: (B, N_latent, dim_head) rotary time embeddings
            modality_embeddings: (B, N_mod, modality_dim) raw modality embeddings
            modality_time_emb: (B, N_mod, dim_head) rotary time embeddings
            modality_mask: (B, N_mod) bool mask for valid modality tokens

        Returns:
            Updated latents: (B, N_latent, dim)
        """
        # Project modality to model dimension
        mod_proj = self.modality_proj(modality_embeddings)

        # Cross-attention: latents (query) attend to modality (key/value)
        latents = latents + self.cross_attn(
            latents, mod_proj,
            latent_time_emb, modality_time_emb,
            modality_mask,
        )
        latents = latents + self.ffn(latents)

        return latents


class MultimodalEncoder(nn.Module):
    """Wraps multiple modality-specific cross-attention layers.

    Can inject image and behavior modalities into encoder latents.
    Modalities are processed sequentially (image first, then behavior).

    Args:
        dim: Model dimension.
        image_dim: Image embedding dimension (768 for DINOv2 ViT-B/14).
        behavior_dim: Behavior feature dimension.
        heads: Number of attention heads.
        dim_head: Dimension per head.
        dropout: Attention dropout.
        ffn_dropout: FFN dropout.
        use_image: Whether to include image cross-attention.
        use_behavior: Whether to include behavior cross-attention.
    """

    def __init__(
        self,
        *,
        dim: int = 256,
        image_dim: int = 768,
        behavior_dim: int = 1,
        heads: int = 4,
        dim_head: int = 64,
        dropout: float = 0.0,
        ffn_dropout: float = 0.2,
        use_image: bool = True,
        use_behavior: bool = True,
    ):
        super().__init__()
        self.use_image = use_image
        self.use_behavior = use_behavior

        if use_image:
            self.image_attn = MultimodalCrossAttention(
                dim=dim,
                modality_dim=image_dim,
                heads=heads,
                dim_head=dim_head,
                dropout=dropout,
                ffn_dropout=ffn_dropout,
            )

        if use_behavior:
            self.behavior_attn = MultimodalCrossAttention(
                dim=dim,
                modality_dim=behavior_dim,
                heads=heads,
                dim_head=dim_head,
                dropout=dropout,
                ffn_dropout=ffn_dropout,
            )

    def forward(
        self,
        latents: torch.Tensor,
        latent_time_emb: torch.Tensor,
        rotary_emb_fn,
        image_embeddings: Optional[torch.Tensor] = None,
        image_timestamps: Optional[torch.Tensor] = None,
        image_mask: Optional[torch.Tensor] = None,
        behavior_values: Optional[torch.Tensor] = None,
        behavior_timestamps: Optional[torch.Tensor] = None,
        behavior_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Inject multimodal information into latent representations.

        Args:
            latents: (B, N_latent, dim) encoder latent representations
            latent_time_emb: (B, N_latent, dim_head) rotary embeddings
            rotary_emb_fn: RotaryTimeEmbedding instance for computing embeddings
            image_embeddings: (B, N_img, image_dim) image embeddings
            image_timestamps: (B, N_img) timestamps for each image
            image_mask: (B, N_img) bool mask for valid images
            behavior_values: (B, N_beh, behavior_dim) behavior values
            behavior_timestamps: (B, N_beh) timestamps for behavior samples
            behavior_mask: (B, N_beh) bool mask for valid behavior tokens

        Returns:
            Updated latents: (B, N_latent, dim)
        """
        if self.use_image and image_embeddings is not None:
            image_time_emb = rotary_emb_fn(image_timestamps)
            latents = self.image_attn(
                latents, latent_time_emb,
                image_embeddings, image_time_emb,
                image_mask,
            )

        if self.use_behavior and behavior_values is not None:
            behavior_time_emb = rotary_emb_fn(behavior_timestamps)
            latents = self.behavior_attn(
                latents, latent_time_emb,
                behavior_values, behavior_time_emb,
                behavior_mask,
            )

        return latents
