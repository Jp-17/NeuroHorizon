"""Diffusion-style flow matching decoder for NeuroHorizon.

This revision keeps explicit ``(time bin, unit)`` tokens and strengthens
conditioning by letting every token directly cross-attend to history latents.
The decoder still uses a factorized stack:

1. dense token-wise cross-attention to history latents
2. per-unit time self-attention across prediction bins
3. per-time unit attention across neurons

This preserves unit-level detail while removing the pooled conditioning
bottleneck of the previous factorized baseline.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .feedforward import FeedForward
from .position_embeddings import SinusoidalTimeEmbedding
from .rotary_attention import RotaryCrossAttention, RotarySelfAttention


class DenseHistoryCrossBlock(nn.Module):
    """Factorized token block with dense token-wise cross-conditioning."""

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
        self.time_attn = RotarySelfAttention(
            dim=dim,
            heads=self_heads,
            dropout=atn_dropout,
            dim_head=dim_head,
            rotate_value=False,
        )
        self.unit_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=self_heads,
            dropout=atn_dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim=dim, dropout=ffn_dropout),
        )
        self.cross_norm = nn.LayerNorm(dim)
        self.time_norm = nn.LayerNorm(dim)
        self.unit_norm = nn.LayerNorm(dim)
        self.ffn_norm = nn.LayerNorm(dim)
        self.flow_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim * 4),
        )

    @staticmethod
    def _apply_unit_mask(
        x: torch.Tensor,
        unit_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if unit_mask is None:
            return x
        return x * unit_mask[:, None, :, None].to(x.dtype)

    def _token_cross_mix(
        self,
        x: torch.Tensor,
        *,
        bin_time_emb: torch.Tensor,
        encoder_latents: torch.Tensor,
        latent_time_emb: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_bins, num_units, dim = x.shape
        token_queries = x.reshape(batch_size, num_bins * num_units, dim)
        token_query_pos = (
            bin_time_emb[:, :, None, :]
            .expand(batch_size, num_bins, num_units, -1)
            .reshape(batch_size, num_bins * num_units, -1)
        )
        mixed = self.cross_attn(
            token_queries,
            encoder_latents,
            token_query_pos,
            latent_time_emb,
        )
        return mixed.reshape(batch_size, num_bins, num_units, dim)

    def _time_mix(
        self,
        x: torch.Tensor,
        bin_time_emb: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_bins, num_units, dim = x.shape
        time_tokens = x.permute(0, 2, 1, 3).reshape(batch_size * num_units, num_bins, dim)
        time_pos = (
            bin_time_emb[:, None, :, :]
            .expand(batch_size, num_units, num_bins, -1)
            .reshape(batch_size * num_units, num_bins, -1)
        )
        time_tokens = self.time_attn(time_tokens, time_pos)
        return time_tokens.reshape(batch_size, num_units, num_bins, dim).permute(0, 2, 1, 3)

    def _unit_mix(
        self,
        x: torch.Tensor,
        unit_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size, num_bins, num_units, dim = x.shape
        unit_tokens = x.reshape(batch_size * num_bins, num_units, dim)
        key_padding_mask = None
        if unit_mask is not None:
            key_padding_mask = (
                (~unit_mask)[:, None, :]
                .expand(batch_size, num_bins, num_units)
                .reshape(batch_size * num_bins, num_units)
            )
        mixed, _ = self.unit_attn(
            unit_tokens,
            unit_tokens,
            unit_tokens,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return mixed.reshape(batch_size, num_bins, num_units, dim)

    def forward(
        self,
        x: torch.Tensor,
        *,
        bin_time_emb: torch.Tensor,
        encoder_latents: torch.Tensor,
        latent_time_emb: torch.Tensor,
        flow_cond: torch.Tensor,
        unit_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        cross_bias, time_bias, unit_bias, ffn_bias = self.flow_mod(flow_cond).chunk(4, dim=-1)

        cross_input = self._apply_unit_mask(self.cross_norm(x) + cross_bias[:, None, None, :], unit_mask)
        cross_context = self._token_cross_mix(
            cross_input,
            bin_time_emb=bin_time_emb,
            encoder_latents=encoder_latents,
            latent_time_emb=latent_time_emb,
        )
        x = self._apply_unit_mask(x + cross_context, unit_mask)

        time_input = self.time_norm(x) + time_bias[:, None, None, :]
        x = self._apply_unit_mask(x + self._time_mix(time_input, bin_time_emb), unit_mask)

        unit_input = self.unit_norm(x) + unit_bias[:, None, None, :]
        x = self._apply_unit_mask(x + self._unit_mix(unit_input, unit_mask), unit_mask)

        ffn_input = self.ffn_norm(x) + ffn_bias[:, None, None, :]
        x = self._apply_unit_mask(x + self.ffn(ffn_input), unit_mask)
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
        self.bin_token_emb = SinusoidalTimeEmbedding(dim=dim, t_min=1e-4, t_max=1.0)
        self.bin_token_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.count_proj = nn.Sequential(
            nn.Linear(1, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.input_norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList(
            [
                DenseHistoryCrossBlock(
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
        self.velocity_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )

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

    def _resolve_unit_mask(
        self,
        noisy_counts: torch.Tensor,
        unit_mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if unit_mask is not None:
            return unit_mask
        return torch.ones(
            noisy_counts.shape[0],
            noisy_counts.shape[-1],
            device=noisy_counts.device,
            dtype=torch.bool,
        )

    def _bin_token_features(
        self,
        batch_size: int,
        num_bins: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        positions = torch.linspace(0.0, 1.0, num_bins, device=device, dtype=dtype)
        features = self.bin_token_mlp(self.bin_token_emb(positions))
        return features.view(1, num_bins, 1, self.dim).expand(batch_size, -1, -1, -1)

    def _build_tokens(
        self,
        *,
        noisy_counts: torch.Tensor,
        unit_embs: torch.Tensor,
        flow_cond: torch.Tensor,
        unit_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size, num_bins, _ = noisy_counts.shape
        count_tokens = self.count_proj(noisy_counts.unsqueeze(-1))
        bin_tokens = self._bin_token_features(
            batch_size,
            num_bins,
            device=noisy_counts.device,
            dtype=noisy_counts.dtype,
        )
        unit_tokens = unit_embs[:, None, :, :]
        x = count_tokens + bin_tokens + unit_tokens + flow_cond[:, None, None, :]
        x = self.input_norm(x)
        if unit_mask is not None:
            x = x * unit_mask[:, None, :, None].to(x.dtype)
        return x

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
        unit_mask = self._resolve_unit_mask(noisy_counts, unit_mask)
        cond_latents = self._condition_encoder_latents(encoder_latents)
        flow_cond = self.diffusion_time_mlp(self.diffusion_time_emb(flow_t))
        x = self._build_tokens(
            noisy_counts=noisy_counts,
            unit_embs=unit_embs,
            flow_cond=flow_cond,
            unit_mask=unit_mask,
        )

        for layer in self.layers:
            x = layer(
                x,
                bin_time_emb=bin_time_emb,
                encoder_latents=cond_latents,
                latent_time_emb=latent_time_emb,
                flow_cond=flow_cond,
                unit_mask=unit_mask,
            )

        x = self.final_norm(x)
        velocity = self.velocity_head(x).squeeze(-1)
        if unit_mask is not None:
            velocity = velocity * unit_mask[:, None, :].to(velocity.dtype)
        return velocity

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
            mask = unit_mask[:, None, :].expand_as(target_counts)
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
                velocity = velocity * unit_mask[:, None, :].to(dtype)
            sample = sample + dt * velocity

        expected_counts = self._target_inverse(sample)
        if unit_mask is not None:
            expected_counts = expected_counts * unit_mask[:, None, :].to(dtype)
        return expected_counts.clamp_min(1e-6).log().clamp(-10, 10)
