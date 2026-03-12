"""Structured prediction memory encoding for NeuroHorizon.

Encodes a population spike-count vector from the previous prediction bin into a
small set of memory tokens. Each token summarizes the full neuron population and
can later be attended by the decoder as an explicit autoregressive memory.
"""

from typing import Optional

import torch
import torch.nn as nn


class PredictionMemoryEncoder(nn.Module):
    """Encode one population count vector into K structured memory tokens.

    Args:
        dim: Model dimension.
        num_memory_tokens: Number of summary tokens to emit per time bin.
        num_heads: Number of attention heads for pooling.
    """

    def __init__(self, dim: int, num_memory_tokens: int = 4, num_heads: int = 4):
        super().__init__()
        self.dim = dim
        self.num_memory_tokens = num_memory_tokens

        self.count_proj = nn.Sequential(
            nn.Linear(1, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.per_neuron_mlp = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.summary_queries = nn.Parameter(torch.randn(1, num_memory_tokens, dim) * 0.02)
        self.pool = nn.MultiheadAttention(dim, num_heads, batch_first=True)

    def forward(
        self,
        counts: torch.Tensor,
        unit_embs: torch.Tensor,
        unit_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode counts into K structured memory tokens.

        Args:
            counts: [B, N] transformed spike counts (e.g. log1p(count)).
            unit_embs: [B, N, D] unit embeddings.
            unit_mask: [B, N] bool mask, True for valid units.

        Returns:
            [B, K, D] memory tokens.
        """
        if counts.ndim != 2:
            raise ValueError(f"counts must be [B, N], got {tuple(counts.shape)}")
        if unit_embs.ndim != 3:
            raise ValueError(f"unit_embs must be [B, N, D], got {tuple(unit_embs.shape)}")

        count_emb = self.count_proj(counts.unsqueeze(-1))
        per_neuron = self.per_neuron_mlp(torch.cat([unit_embs, count_emb], dim=-1))

        B = counts.shape[0]
        queries = self.summary_queries.expand(B, -1, -1)
        key_padding_mask = ~unit_mask if unit_mask is not None else None

        memory_tokens, _ = self.pool(
            queries,
            per_neuron,
            per_neuron,
            key_padding_mask=key_padding_mask,
        )
        return memory_tokens
