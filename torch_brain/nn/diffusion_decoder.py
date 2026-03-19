"""Diffusion-style flow matching decoder for NeuroHorizon.

This decoder keeps the prediction target in count space while avoiding a
full ``T x N`` token transformer, which would be prohibitively expensive for
the current unit counts. The noisy count field is summarized into per-bin
tokens, processed by a DiT-style time backbone, and decoded back to per-unit
velocities with a shared head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .autoregressive_decoder import PerNeuronMLPHead
from .feedforward import FeedForward
from .position_embeddings import SinusoidalTimeEmbedding
from .rotary_attention import RotaryCrossAttention, RotarySelfAttention


class DiTTimeBlock(nn.Module):
    """Time-token block with diffusion-time conditioning."""

    def __init__(
        self,
        *,
        dim: int,
        dim_head: int,
        cross_heads: int,
        self_heads: int,
        ffn_dropout: float,
        atn_dropout: float,
    ):
        super().__init__()
        self.cross_attn = RotaryCrossAttention(
            dim=dim,
            heads=cross_heads,
            dropout=atn_dropout,
            dim_head=dim_head,
            rotate_value=False,
        )
        self.self_attn = RotarySelfAttention(
            dim=dim,
            heads=self_heads,
            dropout=atn_dropout,
            dim_head=dim_head,
            rotate_value=False,
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim=dim, dropout=ffn_dropout),
        )
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim * 6),
        )
        self.cross_norm = nn.LayerNorm(dim)
        self.self_norm = nn.LayerNorm(dim)
        self.ffn_norm = nn.LayerNorm(dim)

    @staticmethod
    def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(
        self,
        x: torch.Tensor,
        *,
        x_time_emb: torch.Tensor,
        encoder_latents: torch.Tensor,
        latent_time_emb: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        shift_cross, scale_cross, gate_cross, shift_self, scale_self, gate_ffn = (
            self.modulation(cond).chunk(6, dim=-1)
        )

        cross_in = self._modulate(self.cross_norm(x), shift_cross, scale_cross)
        x = x + gate_cross.unsqueeze(1) * self.cross_attn(
            cross_in,
            encoder_latents,
            x_time_emb,
            latent_time_emb,
        )

        self_in = self._modulate(self.self_norm(x), shift_self, scale_self)
        x = x + self.self_attn(
            self_in,
            x_time_emb,
        )

        ffn_in = self._modulate(self.ffn_norm(x), shift_self, scale_self)
        x = x + gate_ffn.unsqueeze(1) * self.ffn(ffn_in)
        return x


class DiffusionFlowDecoder(nn.Module):
    """Rectified-flow decoder operating directly on transformed counts."""

    def __init__(
        self,
        *,
        dim: int,
        depth: int,
        dim_head: int,
        cross_heads: int,
        self_heads: int,
        ffn_dropout: float,
        atn_dropout: float,
        condition_dropout: float = 0.0,
        eval_steps: int = 20,
        eval_seed: int = 42,
        target_space: str = "log1p_count",
        solver: str = "euler",
    ):
        super().__init__()
        if target_space != "log1p_count":
            raise ValueError(f"Unsupported target_space={target_space!r}")
        if solver != "euler":
            raise ValueError(f"Unsupported flow solver={solver!r}")
        if eval_steps <= 0:
            raise ValueError("eval_steps must be > 0")

        self.dim = dim
        self.condition_dropout = condition_dropout
        self.eval_steps = eval_steps
        self.eval_seed = eval_seed
        self.target_space = target_space
        self.solver = solver

        self.diffusion_time_emb = SinusoidalTimeEmbedding(dim=dim, t_min=1e-4, t_max=1.0)
        self.diffusion_time_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.input_proj = nn.Sequential(
            nn.Linear(dim + 4, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.layers = nn.ModuleList(
            [
                DiTTimeBlock(
                    dim=dim,
                    dim_head=dim_head,
                    cross_heads=cross_heads,
                    self_heads=self_heads,
                    ffn_dropout=ffn_dropout,
                    atn_dropout=atn_dropout,
                )
                for _ in range(depth)
            ]
        )
        self.final_norm = nn.LayerNorm(dim)
        self.velocity_head = PerNeuronMLPHead(dim)

    def _target_transform(self, counts: torch.Tensor) -> torch.Tensor:
        return torch.log1p(counts.clamp_min(0.0))

    def _target_inverse(self, transformed: torch.Tensor) -> torch.Tensor:
        transformed = transformed.clamp_min(0.0).clamp_max(6.0)
        return torch.expm1(transformed).clamp_min(1e-6)

    def _condition_encoder_latents(
        self,
        encoder_latents: torch.Tensor,
    ) -> torch.Tensor:
        if self.training and self.condition_dropout > 0.0:
            keep = (
                torch.rand(
                    encoder_latents.shape[0],
                    1,
                    1,
                    device=encoder_latents.device,
                )
                >= self.condition_dropout
            ).to(encoder_latents.dtype)
            return encoder_latents * keep
        return encoder_latents

    def _summarize_noisy_counts(
        self,
        noisy_counts: torch.Tensor,
        unit_embs: torch.Tensor,
        unit_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if unit_mask is None:
            unit_mask = torch.ones(
                noisy_counts.shape[0],
                noisy_counts.shape[-1],
                device=noisy_counts.device,
                dtype=torch.bool,
            )

        mask = unit_mask.unsqueeze(1)
        masked_counts = noisy_counts.masked_fill(~mask, 0.0)
        denom = mask.sum(dim=-1, keepdim=True).clamp_min(1).to(noisy_counts.dtype)

        mean = masked_counts.sum(dim=-1, keepdim=True) / denom
        centered = (masked_counts - mean) * mask.to(noisy_counts.dtype)
        std = torch.sqrt(centered.square().sum(dim=-1, keepdim=True) / denom + 1e-6)
        max_value = masked_counts.masked_fill(~mask, float("-inf")).amax(dim=-1, keepdim=True)
        max_value = torch.where(torch.isfinite(max_value), max_value, torch.zeros_like(max_value))
        l1 = masked_counts.abs().sum(dim=-1, keepdim=True) / denom

        pooled = torch.einsum("btn,bnd->btd", masked_counts, unit_embs)
        pooled = pooled / denom

        return torch.cat([pooled, mean, std, max_value, l1], dim=-1)

    def predict_velocity(
        self,
        *,
        noisy_counts: torch.Tensor,
        flow_t: torch.Tensor,
        bin_time_emb: torch.Tensor,
        encoder_latents: torch.Tensor,
        latent_time_emb: torch.Tensor,
        unit_embs: torch.Tensor,
        unit_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        cond_latents = self._condition_encoder_latents(encoder_latents)
        noisy_summary = self._summarize_noisy_counts(noisy_counts, unit_embs, unit_mask)
        x = self.input_proj(noisy_summary)

        flow_cond = self.diffusion_time_mlp(self.diffusion_time_emb(flow_t))
        x = x + flow_cond.unsqueeze(1)

        for layer in self.layers:
            x = layer(
                x,
                x_time_emb=bin_time_emb,
                encoder_latents=cond_latents,
                latent_time_emb=latent_time_emb,
                cond=flow_cond,
            )

        x = self.final_norm(x)
        return self.velocity_head(x, unit_embs)

    def compute_flow_matching_loss(
        self,
        *,
        target_counts: torch.Tensor,
        bin_time_emb: torch.Tensor,
        encoder_latents: torch.Tensor,
        latent_time_emb: torch.Tensor,
        unit_embs: torch.Tensor,
        unit_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        target = self._target_transform(target_counts)
        noise = torch.randn_like(target)
        batch_size = target.shape[0]
        flow_t = torch.rand(batch_size, device=target.device, dtype=target.dtype)
        flow_t_view = flow_t.view(batch_size, 1, 1)
        noisy_counts = (1.0 - flow_t_view) * noise + flow_t_view * target
        velocity_target = target - noise

        velocity_pred = self.predict_velocity(
            noisy_counts=noisy_counts,
            flow_t=flow_t,
            bin_time_emb=bin_time_emb,
            encoder_latents=encoder_latents,
            latent_time_emb=latent_time_emb,
            unit_embs=unit_embs,
            unit_mask=unit_mask,
        )
        if unit_mask is None:
            loss = F.mse_loss(velocity_pred, velocity_target)
        else:
            mask = unit_mask.unsqueeze(1).expand_as(target_counts)
            loss = F.mse_loss(velocity_pred[mask], velocity_target[mask])

        aux = {
            "flow_t_mean": flow_t.mean().detach(),
            "target_log_count_mean": target.mean().detach(),
        }
        return loss, aux

    @torch.no_grad()
    def sample_log_rate(
        self,
        *,
        bin_time_emb: torch.Tensor,
        encoder_latents: torch.Tensor,
        latent_time_emb: torch.Tensor,
        unit_embs: torch.Tensor,
        unit_mask: torch.Tensor | None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        batch_size, num_bins = bin_time_emb.shape[:2]
        num_units = unit_embs.shape[1]
        generator = torch.Generator(device=device)
        generator.manual_seed(self.eval_seed)
        sample = torch.randn(
            batch_size,
            num_bins,
            num_units,
            device=device,
            dtype=dtype,
            generator=generator,
        )

        dt = 1.0 / float(self.eval_steps)
        for step in range(self.eval_steps):
            flow_t = torch.full(
                (batch_size,),
                (step + 0.5) * dt,
                device=device,
                dtype=dtype,
            )
            velocity = self.predict_velocity(
                noisy_counts=sample,
                flow_t=flow_t,
                bin_time_emb=bin_time_emb,
                encoder_latents=encoder_latents,
                latent_time_emb=latent_time_emb,
                unit_embs=unit_embs,
                unit_mask=unit_mask,
            )
            if unit_mask is not None:
                velocity = velocity * unit_mask.unsqueeze(1).to(dtype)
            sample = sample + dt * velocity

        expected_counts = self._target_inverse(sample)
        if unit_mask is not None:
            expected_counts = expected_counts * unit_mask.unsqueeze(1).to(dtype)
        return expected_counts.clamp_min(1e-6).log().clamp(-10, 10)
