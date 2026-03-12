"""Autoregressive decoder and PerNeuronMLPHead for NeuroHorizon.

T-token decoder design: operates only in T dimension (time bins),
while (T, N) expansion happens only in the PerNeuronMLPHead.
"""

import torch
import torch.nn as nn
from torch_brain.nn import FeedForward, RotaryCrossAttention, RotarySelfAttention
from torch_brain.nn.rotary_attention import create_causal_mask


class AutoregressiveDecoder(nn.Module):
    """T-token autoregressive decoder for spike count prediction.

    Per-layer structure:
        1. Cross-attention to history latents
        2. Cross-attention to structured prediction memory (optional)
        3. Causal self-attention over bin queries
        4. FFN
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
                        RotaryCrossAttention(
                            dim=dim,
                            heads=cross_heads,
                            dropout=atn_dropout,
                            dim_head=dim_head,
                            rotate_value=False,
                        ),
                        RotaryCrossAttention(
                            dim=dim,
                            heads=cross_heads,
                            dropout=atn_dropout,
                            dim_head=dim_head,
                            rotate_value=False,
                        ),
                        RotarySelfAttention(
                            dim=dim,
                            heads=self_heads,
                            dropout=atn_dropout,
                            dim_head=dim_head,
                            rotate_value=False,
                        ),
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
        feedback=None,
        prediction_memory=None,
        prediction_memory_time_emb=None,
        prediction_memory_mask=None,
    ):
        """Forward pass of the autoregressive decoder."""
        B, T_pred, _ = bin_queries.shape

        if feedback is not None:
            bin_queries = bin_queries + feedback

        if causal_mask is None and self.causal:
            causal_mask = create_causal_mask(T_pred, device=bin_queries.device)
            causal_mask = causal_mask.unsqueeze(0).expand(B, -1, -1)

        use_prediction_memory = prediction_memory is not None
        if use_prediction_memory and prediction_memory_time_emb is None:
            raise ValueError(
                "prediction_memory_time_emb is required when prediction_memory is set"
            )

        for hist_cross_attn, pred_cross_attn, self_attn, ffn in self.layers:
            bin_queries = bin_queries + hist_cross_attn(
                bin_queries,
                encoder_latents,
                bin_time_emb,
                latent_time_emb,
            )

            if use_prediction_memory:
                bin_queries = bin_queries + pred_cross_attn(
                    bin_queries,
                    prediction_memory,
                    bin_time_emb,
                    prediction_memory_time_emb,
                    prediction_memory_mask,
                )

            bin_queries = bin_queries + self_attn(
                bin_queries,
                bin_time_emb,
                x_mask=causal_mask,
            )
            bin_queries = bin_queries + ffn(bin_queries)

        return bin_queries


class PerNeuronMLPHead(nn.Module):
    """Per-neuron MLP prediction head."""

    def __init__(self, dim: int):
        super().__init__()
        half_dim = dim // 2
        self.bin_proj = nn.Linear(dim, half_dim)
        self.unit_proj = nn.Linear(dim, half_dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, half_dim),
            nn.GELU(),
            nn.Linear(half_dim, half_dim // 2),
            nn.GELU(),
            nn.Linear(half_dim // 2, 1),
        )

    def forward(self, bin_repr, unit_embs):
        bin_h = self.bin_proj(bin_repr)
        unit_h = self.unit_proj(unit_embs)

        B, T, _ = bin_h.shape
        if unit_h.ndim == 2:
            N = unit_h.shape[0]
            unit_h = unit_h.unsqueeze(0).unsqueeze(0).expand(B, T, N, -1)
        else:
            N = unit_h.shape[1]
            unit_h = unit_h.unsqueeze(1).expand(B, T, N, -1)

        bin_h = bin_h.unsqueeze(2).expand(B, T, N, -1)
        combined = torch.cat([bin_h, unit_h], dim=-1)
        log_rate = self.mlp(combined).squeeze(-1)
        return log_rate.clamp(-10, 10)
