"""Prediction feedback encoding for autoregressive decoder.

Encodes per-neuron predicted spike counts/firing rates into a feedback
vector that augments the next time step's bin query.

Implements multiple encoding approaches (selectable via config):
1. PerNeuronMLPPooling: MLP(rate, unit_emb) -> mean pool
2. RateWeightedSum: softmax(rate) * unit_emb
3. CrossAttentionPooling: learned query attends to rate-augmented unit embeddings
4. NoFeedback: baseline, returns zeros
"""

import torch
import torch.nn as nn
from typing import Optional


class PerNeuronMLPPooling(nn.Module):
    """Encoding method 1: Per-neuron MLP + Mean Pooling.

    Each neuron: MLP(concat(rate_emb, unit_emb)) -> D-dim vector.
    Mean pool across valid neurons -> single feedback vector [B, D].
    """

    def __init__(self, dim: int, rate_dim: int = 16):
        super().__init__()
        self.rate_proj = nn.Linear(1, rate_dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim + rate_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(
        self,
        rates: torch.Tensor,
        unit_embs: torch.Tensor,
        unit_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode predicted rates into a feedback vector.

        Args:
            rates: [B, N] predicted spike counts or firing rates
            unit_embs: [B, N, D] unit embeddings
            unit_mask: [B, N] bool mask (True = valid)

        Returns:
            [B, D] feedback vector
        """
        rate_emb = self.rate_proj(rates.unsqueeze(-1))  # [B, N, rate_dim]
        combined = torch.cat([rate_emb, unit_embs], dim=-1)  # [B, N, D+rate_dim]
        per_neuron = self.mlp(combined)  # [B, N, D]

        if unit_mask is not None:
            per_neuron = per_neuron * unit_mask.unsqueeze(-1).float()
            n_valid = unit_mask.sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1]
            feedback = per_neuron.sum(dim=1) / n_valid  # [B, D]
        else:
            feedback = per_neuron.mean(dim=1)

        return feedback


class RateWeightedSum(nn.Module):
    """Encoding method 2: Rate-Weighted Unit Embedding Sum.

    Simplest approach: softmax(rates) as attention weights over unit embeddings.
    Zero extra parameters but loses rate absolute values.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(
        self,
        rates: torch.Tensor,
        unit_embs: torch.Tensor,
        unit_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        scores = rates.clone()
        if unit_mask is not None:
            scores = scores.masked_fill(~unit_mask, float("-inf"))

        weights = torch.softmax(scores, dim=-1)  # [B, N]
        weights = weights.nan_to_num(0.0)

        feedback = torch.einsum("bn,bnd->bd", weights, unit_embs)  # [B, D]
        return feedback


class CrossAttentionPooling(nn.Module):
    """Encoding method 3: Cross-Attention Pooling.

    Learned query attends to (unit_emb as key, rate_emb + unit_emb as value).
    Most flexible: model learns which neurons' predictions matter.
    """

    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.rate_proj = nn.Linear(1, dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

    def forward(
        self,
        rates: torch.Tensor,
        unit_embs: torch.Tensor,
        unit_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, D = unit_embs.shape

        query = self.query.expand(B, -1, -1)  # [B, 1, D]
        key = unit_embs  # [B, N, D]
        value = self.rate_proj(rates.unsqueeze(-1)) + unit_embs  # [B, N, D]

        # Key padding mask: True = ignore in nn.MultiheadAttention
        key_padding_mask = ~unit_mask if unit_mask is not None else None

        out, _ = self.attn(query, key, value, key_padding_mask=key_padding_mask)
        return out.squeeze(1)  # [B, D]


class NoFeedback(nn.Module):
    """Encoding method 5: No feedback (baseline).

    Returns zeros. Equivalent to current architecture.
    Serves as control condition for AR experiments.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(
        self,
        rates: torch.Tensor,
        unit_embs: torch.Tensor,
        unit_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = rates.shape[0]
        return torch.zeros(B, self.dim, device=rates.device, dtype=unit_embs.dtype)


# Factory
FEEDBACK_ENCODERS = {
    "mlp_pool": PerNeuronMLPPooling,
    "rate_weighted": RateWeightedSum,
    "cross_attn": CrossAttentionPooling,
    "none": NoFeedback,
}


def build_feedback_encoder(method: str, dim: int, **kwargs) -> nn.Module:
    """Build a feedback encoder by method name.

    Args:
        method: One of 'mlp_pool', 'rate_weighted', 'cross_attn', 'none'
        dim: Model dimension

    Returns:
        nn.Module implementing the feedback encoder interface
    """
    if method not in FEEDBACK_ENCODERS:
        raise ValueError(
            f"Unknown feedback method: {method}. "
            f"Choose from {list(FEEDBACK_ENCODERS.keys())}"
        )
    return FEEDBACK_ENCODERS[method](dim=dim, **kwargs)
