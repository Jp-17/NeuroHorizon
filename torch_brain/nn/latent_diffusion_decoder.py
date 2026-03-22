"""Latent diffusion decoder for NeuroHorizon.

This module implements Option 2A:

1. Encode future ``log1p(count)`` targets into a compact factorized latent grid.
2. Run rectified-flow matching in latent space, conditioned on history latents.
3. Decode sampled latent trajectories back into per-bin, per-unit log-rates.

The first implementation keeps a ``time x latent-unit`` factorization so the
diffusion model does not have to denoise the full future count field directly.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .feedforward import FeedForward
from .position_embeddings import SinusoidalTimeEmbedding
from .rotary_attention import RotaryCrossAttention, RotarySelfAttention


def _masked_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    if mask is None:
        return F.mse_loss(prediction, target)
    expanded = mask[:, None, :].expand_as(prediction)
    return F.mse_loss(prediction[expanded], target[expanded])


class FactorizedTokenMixerBlock(nn.Module):
    """Factorized time/unit token mixing without history cross-conditioning."""

    def __init__(
        self,
        *,
        dim: int,
        dim_head: int,
        self_heads: int,
        ffn_dropout: float,
        atn_dropout: float,
    ):
        super().__init__()
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
        self.time_norm = nn.LayerNorm(dim)
        self.unit_norm = nn.LayerNorm(dim)
        self.ffn_norm = nn.LayerNorm(dim)

    @staticmethod
    def _apply_unit_mask(
        x: torch.Tensor,
        unit_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if unit_mask is None:
            return x
        return x * unit_mask[:, None, :, None].to(x.dtype)

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
        mixed = self.time_attn(time_tokens, time_pos)
        return mixed.reshape(batch_size, num_units, num_bins, dim).permute(0, 2, 1, 3)

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
        unit_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        time_input = self.time_norm(x)
        x = self._apply_unit_mask(x + self._time_mix(time_input, bin_time_emb), unit_mask)

        unit_input = self.unit_norm(x)
        x = self._apply_unit_mask(x + self._unit_mix(unit_input, unit_mask), unit_mask)

        ffn_input = self.ffn_norm(x)
        x = self._apply_unit_mask(x + self.ffn(ffn_input), unit_mask)
        return x


class FactorizedCountAutoencoder(nn.Module):
    """Deterministic count autoencoder with factorized time/unit latents."""

    def __init__(
        self,
        *,
        dim: int,
        autoencoder_depth: int,
        num_latent_units: int,
        dim_head: int,
        cross_heads: int,
        self_heads: int,
        ffn_dropout: float,
        atn_dropout: float,
        target_space: str = "log1p_count",
    ):
        super().__init__()
        if target_space != "log1p_count":
            raise ValueError(f"Unsupported target_space={target_space!r}")
        if num_latent_units <= 0:
            raise ValueError("num_latent_units must be > 0")

        self.dim = dim
        self.num_latent_units = num_latent_units
        self.target_space = target_space

        self.count_proj = nn.Sequential(
            nn.Linear(1, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.bin_token_emb = SinusoidalTimeEmbedding(dim=dim, t_min=1e-4, t_max=1.0)
        self.bin_token_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.encoder_input_norm = nn.LayerNorm(dim)
        self.encoder_layers = nn.ModuleList(
            [
                FactorizedTokenMixerBlock(
                    dim=dim,
                    dim_head=dim_head,
                    self_heads=self_heads,
                    ffn_dropout=ffn_dropout,
                    atn_dropout=atn_dropout,
                )
                for _ in range(autoencoder_depth)
            ]
        )

        self.latent_queries = nn.Parameter(torch.randn(num_latent_units, dim) * 0.02)
        self.encoder_query_norm = nn.LayerNorm(dim)
        self.encoder_cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=cross_heads,
            dropout=atn_dropout,
            batch_first=True,
        )
        self.encoder_query_ffn = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim=dim, dropout=ffn_dropout),
        )

        self.latent_refine_layers = nn.ModuleList(
            [
                FactorizedTokenMixerBlock(
                    dim=dim,
                    dim_head=dim_head,
                    self_heads=self_heads,
                    ffn_dropout=ffn_dropout,
                    atn_dropout=atn_dropout,
                )
                for _ in range(autoencoder_depth)
            ]
        )
        self.decoder_query_norm = nn.LayerNorm(dim)
        self.decoder_cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=cross_heads,
            dropout=atn_dropout,
            batch_first=True,
        )
        self.decoder_layers = nn.ModuleList(
            [
                FactorizedTokenMixerBlock(
                    dim=dim,
                    dim_head=dim_head,
                    self_heads=self_heads,
                    ffn_dropout=ffn_dropout,
                    atn_dropout=atn_dropout,
                )
                for _ in range(autoencoder_depth)
            ]
        )
        self.output_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )

    def _target_transform(self, counts: torch.Tensor) -> torch.Tensor:
        return torch.log1p(counts.clamp_min(0.0))

    def _target_inverse(self, transformed: torch.Tensor) -> torch.Tensor:
        transformed = transformed.clamp_min(0.0).clamp_max(6.0)
        return torch.expm1(transformed).clamp_min(1e-6)

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

    @staticmethod
    def _apply_unit_mask(
        x: torch.Tensor,
        unit_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if unit_mask is None:
            return x
        return x * unit_mask[:, None, :, None].to(x.dtype)

    def _repeat_padding_mask(
        self,
        unit_mask: torch.Tensor | None,
        *,
        batch_size: int,
        num_bins: int,
        num_units: int,
    ) -> torch.Tensor | None:
        if unit_mask is None:
            return None
        return (
            (~unit_mask)[:, None, :]
            .expand(batch_size, num_bins, num_units)
            .reshape(batch_size * num_bins, num_units)
        )

    def encode(
        self,
        *,
        target_counts: torch.Tensor,
        unit_embs: torch.Tensor,
        bin_time_emb: torch.Tensor,
        unit_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        transformed = self._target_transform(target_counts)
        batch_size, num_bins, num_units = transformed.shape
        bin_tokens = self._bin_token_features(
            batch_size,
            num_bins,
            device=transformed.device,
            dtype=transformed.dtype,
        )
        x = self.count_proj(transformed.unsqueeze(-1)) + bin_tokens + unit_embs[:, None, :, :]
        x = self.encoder_input_norm(x)
        x = self._apply_unit_mask(x, unit_mask)

        for layer in self.encoder_layers:
            x = layer(x, bin_time_emb=bin_time_emb, unit_mask=unit_mask)

        queries = (
            self.latent_queries[None, None, :, :]
            .expand(batch_size, num_bins, -1, -1)
            .reshape(batch_size * num_bins, self.num_latent_units, self.dim)
        )
        keys = x.reshape(batch_size * num_bins, num_units, self.dim)
        key_padding_mask = self._repeat_padding_mask(
            unit_mask,
            batch_size=batch_size,
            num_bins=num_bins,
            num_units=num_units,
        )
        latent_tokens, _ = self.encoder_cross_attn(
            self.encoder_query_norm(queries),
            keys,
            keys,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        latent_tokens = latent_tokens + self.encoder_query_ffn(latent_tokens)
        latent_tokens = latent_tokens.reshape(batch_size, num_bins, self.num_latent_units, self.dim)
        return latent_tokens, transformed

    def decode(
        self,
        *,
        latent_tokens: torch.Tensor,
        unit_embs: torch.Tensor,
        bin_time_emb: torch.Tensor,
        unit_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size, num_bins, _, _ = latent_tokens.shape
        num_units = unit_embs.shape[1]

        x = latent_tokens
        for layer in self.latent_refine_layers:
            x = layer(x, bin_time_emb=bin_time_emb, unit_mask=None)

        bin_tokens = self._bin_token_features(
            batch_size,
            num_bins,
            device=latent_tokens.device,
            dtype=latent_tokens.dtype,
        )
        unit_queries = (unit_embs[:, None, :, :] + bin_tokens).reshape(
            batch_size * num_bins,
            num_units,
            self.dim,
        )
        values = x.reshape(batch_size * num_bins, self.num_latent_units, self.dim)
        decoded, _ = self.decoder_cross_attn(
            self.decoder_query_norm(unit_queries),
            values,
            values,
            need_weights=False,
        )
        decoded = decoded.reshape(batch_size, num_bins, num_units, self.dim)
        decoded = self._apply_unit_mask(decoded + unit_embs[:, None, :, :] + bin_tokens, unit_mask)

        for layer in self.decoder_layers:
            decoded = layer(decoded, bin_time_emb=bin_time_emb, unit_mask=unit_mask)

        reconstructed = self.output_head(decoded).squeeze(-1)
        if unit_mask is not None:
            reconstructed = reconstructed * unit_mask[:, None, :].to(reconstructed.dtype)
        return reconstructed

    def decode_to_log_rate(
        self,
        *,
        latent_tokens: torch.Tensor,
        unit_embs: torch.Tensor,
        bin_time_emb: torch.Tensor,
        unit_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        transformed = self.decode(
            latent_tokens=latent_tokens,
            unit_embs=unit_embs,
            bin_time_emb=bin_time_emb,
            unit_mask=unit_mask,
        )
        expected_counts = self._target_inverse(transformed)
        if unit_mask is not None:
            expected_counts = expected_counts * unit_mask[:, None, :].to(expected_counts.dtype)
        return expected_counts.clamp_min(1e-6).log().clamp(-10, 10)


class LatentConditionedBlock(nn.Module):
    """Latent diffusion block with pooled time cross-conditioning."""

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
        self.latent_unit_attn = nn.MultiheadAttention(
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
        mixed = self.time_attn(time_tokens, time_pos)
        return mixed.reshape(batch_size, num_units, num_bins, dim).permute(0, 2, 1, 3)

    def _unit_mix(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_bins, num_units, dim = x.shape
        unit_tokens = x.reshape(batch_size * num_bins, num_units, dim)
        mixed, _ = self.latent_unit_attn(
            unit_tokens,
            unit_tokens,
            unit_tokens,
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
    ) -> torch.Tensor:
        cross_bias, time_bias, unit_bias, ffn_bias = self.flow_mod(flow_cond).chunk(4, dim=-1)

        pooled = x.mean(dim=2)
        cross_input = self.cross_norm(pooled) + cross_bias[:, None, :]
        cross_context = self.cross_attn(
            cross_input,
            encoder_latents,
            bin_time_emb,
            latent_time_emb,
        )
        x = x + cross_context[:, :, None, :]

        time_input = self.time_norm(x) + time_bias[:, None, None, :]
        x = x + self._time_mix(time_input, bin_time_emb)

        unit_input = self.unit_norm(x) + unit_bias[:, None, None, :]
        x = x + self._unit_mix(unit_input)

        ffn_input = self.ffn_norm(x) + ffn_bias[:, None, None, :]
        x = x + self.ffn(ffn_input)
        return x


class LatentDiffusionDecoder(nn.Module):
    """Rectified-flow diffusion decoder operating on factorized latents."""

    def __init__(
        self,
        *,
        dim: int,
        depth: int,
        autoencoder_depth: int,
        num_latent_units: int,
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
        latent_loss_weight: float = 1.0,
        recon_loss_weight: float = 1.0,
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
        self.latent_loss_weight = latent_loss_weight
        self.recon_loss_weight = recon_loss_weight

        self.autoencoder = FactorizedCountAutoencoder(
            dim=dim,
            autoencoder_depth=autoencoder_depth,
            num_latent_units=num_latent_units,
            dim_head=dim_head,
            cross_heads=cross_heads,
            self_heads=self_heads,
            ffn_dropout=ffn_dropout,
            atn_dropout=atn_dropout,
            target_space=target_space,
        )
        self.diffusion_time_emb = SinusoidalTimeEmbedding(dim=dim, t_min=1e-4, t_max=1.0)
        self.diffusion_time_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.input_norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList(
            [
                LatentConditionedBlock(
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
            nn.Linear(dim, dim),
        )

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

    def predict_velocity(
        self,
        *,
        noisy_latents: torch.Tensor,
        flow_t: torch.Tensor,
        bin_time_emb: torch.Tensor,
        encoder_latents: torch.Tensor,
        latent_time_emb: torch.Tensor,
    ) -> torch.Tensor:
        cond_latents = self._condition_encoder_latents(encoder_latents)
        flow_cond = self.diffusion_time_mlp(self.diffusion_time_emb(flow_t))
        x = self.input_norm(noisy_latents + flow_cond[:, None, None, :])

        for layer in self.layers:
            x = layer(
                x,
                bin_time_emb=bin_time_emb,
                encoder_latents=cond_latents,
                latent_time_emb=latent_time_emb,
                flow_cond=flow_cond,
            )

        x = self.final_norm(x)
        return self.velocity_head(x)

    def compute_loss(
        self,
        *,
        target_counts: torch.Tensor,
        bin_time_emb: torch.Tensor,
        encoder_latents: torch.Tensor,
        latent_time_emb: torch.Tensor,
        unit_embs: torch.Tensor,
        unit_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        latent_target, transformed_target = self.autoencoder.encode(
            target_counts=target_counts,
            unit_embs=unit_embs,
            bin_time_emb=bin_time_emb,
            unit_mask=unit_mask,
        )
        reconstructed = self.autoencoder.decode(
            latent_tokens=latent_target,
            unit_embs=unit_embs,
            bin_time_emb=bin_time_emb,
            unit_mask=unit_mask,
        )
        recon_loss = _masked_mse(reconstructed, transformed_target, unit_mask)

        noise = torch.randn_like(latent_target)
        batch_size = latent_target.shape[0]
        flow_t = torch.rand(batch_size, device=latent_target.device, dtype=latent_target.dtype)
        flow_t_view = flow_t.view(batch_size, 1, 1, 1)
        noisy_latents = (1.0 - flow_t_view) * noise + flow_t_view * latent_target
        velocity_target = latent_target - noise

        velocity_pred = self.predict_velocity(
            noisy_latents=noisy_latents,
            flow_t=flow_t,
            bin_time_emb=bin_time_emb,
            encoder_latents=encoder_latents,
            latent_time_emb=latent_time_emb,
        )
        latent_loss = F.mse_loss(velocity_pred, velocity_target)
        total_loss = self.latent_loss_weight * latent_loss + self.recon_loss_weight * recon_loss

        aux = {
            "ae_recon_loss": recon_loss.detach(),
            "diffusion_latent_loss": latent_loss.detach(),
            "flow_t_mean": flow_t.mean().detach(),
            "latent_target_mean": latent_target.mean().detach(),
        }
        return total_loss, aux

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
        generator = torch.Generator(device=device)
        generator.manual_seed(self.eval_seed)
        sample = torch.randn(
            batch_size,
            num_bins,
            self.autoencoder.num_latent_units,
            self.dim,
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
                noisy_latents=sample,
                flow_t=flow_t,
                bin_time_emb=bin_time_emb,
                encoder_latents=encoder_latents,
                latent_time_emb=latent_time_emb,
            )
            sample = sample + dt * velocity

        return self.autoencoder.decode_to_log_rate(
            latent_tokens=sample,
            unit_embs=unit_embs,
            bin_time_emb=bin_time_emb,
            unit_mask=unit_mask,
        )
