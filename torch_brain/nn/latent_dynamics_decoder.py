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
        pool_token_dim: int | None = None,
        state_dim: int | None = None,
        context_conditioning: bool = False,
        context_dim: int | None = None,
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
        if pool_token_dim is None:
            if dim % num_pool_tokens != 0:
                raise ValueError(
                    f"dim ({dim}) must be divisible by num_pool_tokens ({num_pool_tokens}) "
                    "when pool_token_dim is not specified"
                )
            pool_token_dim = dim // num_pool_tokens
        if pool_token_dim < 1:
            raise ValueError("pool_token_dim must be >= 1")
        if state_dim is None:
            state_dim = dim
        if state_dim < 1:
            raise ValueError("state_dim must be >= 1")
        if context_dim is None:
            context_dim = state_dim
        if context_dim < 1:
            raise ValueError("context_dim must be >= 1")

        flattened_dim = num_pool_tokens * pool_token_dim

        self.context_conditioning = context_conditioning
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

        self.token_proj = nn.Linear(dim, pool_token_dim)
        self.init_norm = nn.LayerNorm(flattened_dim)
        self.init_proj = nn.Linear(flattened_dim, state_dim)
        self.context_norm = nn.LayerNorm(state_dim)
        self.step_emb = nn.Parameter(torch.randn(1, max_steps, state_dim) * init_scale)
        if context_conditioning:
            self.context_token_norm = nn.LayerNorm(dim)
            self.context_token_proj = nn.Linear(dim, context_dim)
            self.context_vector_norm = nn.LayerNorm(context_dim)
            self.context_input_proj = nn.Linear(context_dim, state_dim)
            self.context_output_proj = nn.Linear(context_dim, state_dim)
        else:
            self.context_token_norm = None
            self.context_token_proj = None
            self.context_vector_norm = None
            self.context_input_proj = None
            self.context_output_proj = None
        self.gru = nn.GRU(
            input_size=state_dim,
            hidden_size=state_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output_proj = nn.Linear(state_dim, dim)
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
        context = self.context_norm(self.init_proj(self.init_norm(init_tokens)))

        step_inputs = context.unsqueeze(1).expand(batch_size, num_steps, -1)
        step_inputs = step_inputs + self.step_emb[:, :num_steps, :]
        context_vector = None
        if self.context_conditioning:
            assert self.context_token_norm is not None
            assert self.context_token_proj is not None
            assert self.context_vector_norm is not None
            assert self.context_input_proj is not None
            assert self.context_output_proj is not None
            pooled_context = pooled.mean(dim=1)
            context_vector = self.context_vector_norm(
                self.context_token_proj(self.context_token_norm(pooled_context))
            )
            step_inputs = step_inputs + self.context_input_proj(context_vector).unsqueeze(1)

        hidden0 = context.unsqueeze(0).expand(self.gru.num_layers, -1, -1).contiguous()
        rollout, _ = self.gru(step_inputs, hidden0)
        if context_vector is not None:
            rollout = rollout + self.context_output_proj(context_vector).unsqueeze(1)
        return self.output_norm(self.output_proj(rollout))
