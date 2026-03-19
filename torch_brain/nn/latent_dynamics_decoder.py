"""Latent dynamics decoder for NeuroHorizon.

This decoder keeps the existing POYO+ history encoder and per-neuron readout,
but replaces observation-space autoregressive feedback with a latent-space
rollout driven by a compact pooled state.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .feedforward import FeedForward


class LatentDynamicsDecoder(nn.Module):
    """Pool encoder latents into an initial state, then roll forward with a GRU."""

    def __init__(
        self,
        *,
        dim: int,
        num_pool_tokens: int = 4,
        num_layers: int = 2,
        num_heads: int = 2,
        atn_dropout: float = 0.0,
        ffn_dropout: float = 0.2,
        max_steps: int = 50,
        init_scale: float = 0.02,
    ):
        super().__init__()

        if num_pool_tokens < 1:
            raise ValueError("num_pool_tokens must be >= 1")
        if dim % num_pool_tokens != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by num_pool_tokens ({num_pool_tokens})"
            )

        pooled_token_dim = dim // num_pool_tokens

        self.pool_queries = nn.Parameter(torch.randn(1, num_pool_tokens, dim) * init_scale)
        self.pool_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=atn_dropout,
            batch_first=True,
        )
        self.pool_norm = nn.LayerNorm(dim)
        self.pool_ffn = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim=dim, dropout=ffn_dropout),
        )

        self.token_proj = nn.Linear(dim, pooled_token_dim)
        self.init_norm = nn.LayerNorm(num_pool_tokens * pooled_token_dim)
        self.init_proj = nn.Linear(num_pool_tokens * pooled_token_dim, dim)
        self.step_emb = nn.Parameter(torch.randn(1, max_steps, dim) * init_scale)
        self.gru = nn.GRU(
            input_size=dim,
            hidden_size=dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(dim)

    def forward(self, encoder_latents: torch.Tensor, num_steps: int) -> torch.Tensor:
        if num_steps < 1:
            raise ValueError("num_steps must be >= 1")
        if num_steps > self.step_emb.shape[1]:
            raise ValueError(
                f"num_steps ({num_steps}) exceeds max supported steps ({self.step_emb.shape[1]})"
            )

        batch_size = encoder_latents.shape[0]
        queries = self.pool_queries.expand(batch_size, -1, -1)
        pooled, _ = self.pool_attn(queries, encoder_latents, encoder_latents, need_weights=False)
        pooled = queries + pooled
        pooled = pooled + self.pool_ffn(self.pool_norm(pooled))

        init_tokens = self.token_proj(pooled).reshape(batch_size, -1)
        context = self.init_proj(self.init_norm(init_tokens))

        step_inputs = context.unsqueeze(1).expand(batch_size, num_steps, -1)
        step_inputs = step_inputs + self.step_emb[:, :num_steps, :]
        hidden0 = context.unsqueeze(0).expand(self.gru.num_layers, -1, -1).contiguous()
        rollout, _ = self.gru(step_inputs, hidden0)
        return self.output_norm(rollout)
