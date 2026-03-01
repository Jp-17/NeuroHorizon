"""Autoregressive decoder and PerNeuronMLPHead for NeuroHorizon.

T-token decoder design: operates only in T dimension (time bins),
(T, N) expansion happens only in the PerNeuronMLPHead.
"""

import torch
import torch.nn as nn
from torch_brain.nn import (
    FeedForward,
    RotaryCrossAttention,
    RotarySelfAttention,
)
from torch_brain.nn.rotary_attention import create_causal_mask


class AutoregressiveDecoder(nn.Module):
    """T-token autoregressive decoder for spike count prediction.

    Architecture per layer:
        1. Cross-Attention: bin_queries attend to encoder_latents (bidirectional)
        2. Causal Self-Attention: bin_queries attend to each other (causal)
        3. FFN (GEGLU)

    Args:
        dim: Model dimension
        depth: Number of decoder layers
        dim_head: Dimension per attention head
        cross_heads: Number of cross-attention heads
        self_heads: Number of self-attention heads
        ffn_dropout: Dropout rate for FFN
        atn_dropout: Dropout rate for attention
    """

    def __init__(
        self,
        *,
        dim: int = 512,
        depth: int = 2,
        dim_head: int = 64,
        cross_heads: int = 2,
        self_heads: int = 8,
        ffn_dropout: float = 0.2,
        atn_dropout: float = 0.0,
        causal: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.causal = causal
        self.depth = depth

        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        # Cross-attention: bin_queries -> encoder_latents
                        RotaryCrossAttention(
                            dim=dim,
                            heads=cross_heads,
                            dropout=atn_dropout,
                            dim_head=dim_head,
                            rotate_value=False,  # Decoder uses rotate_value=False
                        ),
                        # Causal self-attention among bin_queries
                        RotarySelfAttention(
                            dim=dim,
                            heads=self_heads,
                            dropout=atn_dropout,
                            dim_head=dim_head,
                            rotate_value=False,  # Decoder uses rotate_value=False
                        ),
                        # FFN
                        nn.Sequential(
                            nn.LayerNorm(dim),
                            FeedForward(dim=dim, dropout=ffn_dropout),
                        ),
                    ]
                )
            )

    def forward(
        self,
        bin_queries,
        bin_time_emb,
        encoder_latents,
        latent_time_emb,
        causal_mask=None,
    ):
        """Forward pass of the autoregressive decoder.

        Args:
            bin_queries: [B, T_pred, dim] - learnable bin query embeddings
            bin_time_emb: [B, T_pred, dim_head] - rotary time embeddings for bins
            encoder_latents: [B, N_latents, dim] - output of encoder+processor
            latent_time_emb: [B, N_latents, dim_head] - rotary time embeddings for latents
            causal_mask: [B, T_pred, T_pred] or [T_pred, T_pred] bool mask (optional)
                If None, created automatically.

        Returns:
            [B, T_pred, dim] - decoded bin representations
        """
        B, T_pred, _ = bin_queries.shape

        # Create causal mask if not provided (skip for non-AR mode)
        if causal_mask is None and self.causal:
            causal_mask = create_causal_mask(T_pred, device=bin_queries.device)
            causal_mask = causal_mask.unsqueeze(0).expand(B, -1, -1)

        for cross_attn, self_attn, ffn in self.layers:
            # Cross-attention: bin_queries attend to encoder latents (bidirectional)
            bin_queries = bin_queries + cross_attn(
                bin_queries,
                encoder_latents,
                bin_time_emb,
                latent_time_emb,
            )

            # Causal self-attention among bin_queries
            bin_queries = bin_queries + self_attn(
                bin_queries,
                bin_time_emb,
                x_mask=causal_mask,
            )

            # FFN
            bin_queries = bin_queries + ffn(bin_queries)

        return bin_queries


class PerNeuronMLPHead(nn.Module):
    """Per-neuron MLP prediction head.

    Takes bin representations from the decoder and unit embeddings,
    concatenates them, and predicts log spike rates.

    The (T, N) expansion happens here, not in the decoder.

    Args:
        dim: Model dimension (bin_repr and unit_emb will each be projected to dim//2)
    """

    def __init__(self, dim: int):
        super().__init__()
        half_dim = dim // 2

        # Project bin repr and unit emb to half-dim each
        self.bin_proj = nn.Linear(dim, half_dim)
        self.unit_proj = nn.Linear(dim, half_dim)

        # MLP: concat(bin_repr[half_dim], unit_emb[half_dim]) -> 1
        self.mlp = nn.Sequential(
            nn.Linear(dim, half_dim),
            nn.GELU(),
            nn.Linear(half_dim, half_dim // 2),
            nn.GELU(),
            nn.Linear(half_dim // 2, 1),
        )

    def forward(self, bin_repr, unit_embs):
        """Predict log spike rates for all (time_bin, neuron) pairs.

        Args:
            bin_repr: [B, T, dim] - decoder output representations
            unit_embs: [B, N, dim] or [N, dim] - unit embeddings for current session

        Returns:
            log_rate: [B, T, N] - predicted log spike rates, clamped to [-10, 10]
        """
        # Project to half dimension
        bin_h = self.bin_proj(bin_repr)  # [B, T, dim//2]
        unit_h = self.unit_proj(unit_embs)  # [B, N, dim//2] or [N, dim//2]

        B, T, half_dim = bin_h.shape

        if unit_h.ndim == 2:
            # [N, dim//2] -> [1, 1, N, dim//2]
            N = unit_h.shape[0]
            unit_h = unit_h.unsqueeze(0).unsqueeze(0).expand(B, T, N, -1)
        else:
            # [B, N, dim//2] -> [B, 1, N, dim//2]
            N = unit_h.shape[1]
            unit_h = unit_h.unsqueeze(1).expand(B, T, N, -1)

        # Expand bin_h: [B, T, dim//2] -> [B, T, N, dim//2]
        bin_h = bin_h.unsqueeze(2).expand(B, T, N, -1)

        # Concatenate: [B, T, N, dim]
        combined = torch.cat([bin_h, unit_h], dim=-1)

        # MLP prediction: [B, T, N, 1] -> [B, T, N]
        log_rate = self.mlp(combined).squeeze(-1)

        return log_rate.clamp(-10, 10)
