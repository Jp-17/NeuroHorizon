"""Latent dynamics decoder for NeuroHorizon.

This decoder keeps the existing POYO+ history encoder and per-neuron readout,
but replaces observation-space autoregressive feedback with a latent-space
rollout driven by a compact pooled state.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .feedforward import FeedForward


def _resolve_mamba_backend() -> tuple[str, Any]:
    try:
        from mamba_ssm import Mamba  # type: ignore

        return "mamba_ssm", Mamba
    except Exception:
        try:
            from mamba_ssm.modules.mamba_simple import Mamba  # type: ignore

            return "mamba_ssm", Mamba
        except Exception:
            try:
                import mambapy  # noqa: F401
                from transformers.models.mamba.configuration_mamba import MambaConfig
                from transformers.models.mamba.modeling_mamba import MambaBlock

                return "transformers_mambapy", (MambaBlock, MambaConfig)
            except Exception as exc:  # pragma: no cover - exercised only when dependency missing
                raise ImportError(
                    "latent_dynamics_backbone='mamba' requires either "
                    "'mamba-ssm>=2.2,<2.3' or the lighter fallback "
                    "'transformers' + 'mambapy>=1.2.0'."
                ) from exc


class LatentDynamicsMambaBlock(nn.Module):
    """Residual Mamba block used inside the latent rollout stack."""

    def __init__(
        self,
        *,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
        ffn_dropout: float,
    ) -> None:
        super().__init__()

        backend, payload = _resolve_mamba_backend()
        self.backend = backend
        if backend == "mamba_ssm":
            mamba_cls = payload
            self.mixer_norm = nn.LayerNorm(d_model)
            self.mixer = mamba_cls(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
        else:
            mamba_block_cls, mamba_config_cls = payload
            self.mixer_norm = None
            config = mamba_config_cls(
                hidden_size=d_model,
                state_size=d_state,
                num_hidden_layers=1,
                expand=expand,
                conv_kernel=d_conv,
                use_mambapy=True,
            )
            self.mixer = mamba_block_cls(config, layer_idx=0)
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            FeedForward(dim=d_model, dropout=ffn_dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.backend == "mamba_ssm":
            assert self.mixer_norm is not None
            x = x + self.mixer(self.mixer_norm(x))
        else:
            x = self.mixer(x)
        x = x + self.ffn(x)
        return x


class LatentDynamicsDecoder(nn.Module):
    """Pool encoder latents into an initial state, then roll forward."""

    def __init__(
        self,
        *,
        dim: int,
        num_pool_tokens: int = 4,
        pool_token_dim: int | None = None,
        state_dim: int | None = None,
        context_conditioning: bool = False,
        context_dim: int | None = None,
        backbone: str = "gru",
        backbone_cfg: dict[str, Any] | None = None,
        input_mode: str = "prev_latent",
        num_layers: int = 2,
        num_heads: int = 2,
        atn_dropout: float = 0.0,
        ffn_dropout: float = 0.2,
        max_steps: int = 50,
        init_scale: float = 0.02,
    ) -> None:
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
        if backbone not in {"gru", "mamba"}:
            raise ValueError(f"Unknown latent dynamics backbone: {backbone!r}")
        if input_mode != "prev_latent":
            raise ValueError(
                f"Unsupported latent dynamics input_mode={input_mode!r}; only "
                "'prev_latent' is supported"
            )

        backbone_cfg = {} if backbone_cfg is None else dict(backbone_cfg)
        flattened_dim = num_pool_tokens * pool_token_dim

        self.context_conditioning = context_conditioning
        self.backbone = backbone
        self.input_mode = input_mode
        self.state_dim = state_dim
        self.output_residual = bool(backbone_cfg.get("output_residual", True))

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
        self.state_norm = nn.LayerNorm(state_dim)
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

        if backbone == "gru":
            self.gru = nn.GRU(
                input_size=state_dim,
                hidden_size=state_dim,
                num_layers=num_layers,
                batch_first=True,
            )
            self.mamba_blocks = None
        else:
            d_model = int(backbone_cfg.get("d_model", state_dim))
            if d_model != state_dim:
                raise ValueError(
                    f"Mamba d_model ({d_model}) must match state_dim ({state_dim}) "
                    "in the current implementation"
                )
            d_state = int(backbone_cfg.get("d_state", 64))
            d_conv = int(backbone_cfg.get("d_conv", 4))
            expand = int(backbone_cfg.get("expand", 2))
            self.gru = None
            self.mamba_blocks = nn.ModuleList(
                [
                    LatentDynamicsMambaBlock(
                        d_model=d_model,
                        d_state=d_state,
                        d_conv=d_conv,
                        expand=expand,
                        ffn_dropout=ffn_dropout,
                    )
                    for _ in range(num_layers)
                ]
            )

        self.output_proj = nn.Linear(state_dim, dim)
        self.output_norm = nn.LayerNorm(dim)

    def _pool_encoder_latents(self, encoder_latents: torch.Tensor) -> torch.Tensor:
        batch_size = encoder_latents.shape[0]
        queries = self.pool_queries.expand(batch_size, -1, -1)
        pooled, _ = self.pool_attn(queries, encoder_latents, encoder_latents, need_weights=False)
        pooled = queries + pooled
        pooled = pooled + self.pool_ffn(self.pool_norm(pooled))
        return pooled

    def _initial_state(self, pooled: torch.Tensor) -> torch.Tensor:
        batch_size = pooled.shape[0]
        init_tokens = self.token_proj(pooled).reshape(batch_size, -1)
        return self.state_norm(self.init_proj(self.init_norm(init_tokens)))

    def _context_vector(self, pooled: torch.Tensor) -> torch.Tensor | None:
        if not self.context_conditioning:
            return None

        assert self.context_token_norm is not None
        assert self.context_token_proj is not None
        assert self.context_vector_norm is not None
        pooled_context = pooled.mean(dim=1)
        return self.context_vector_norm(
            self.context_token_proj(self.context_token_norm(pooled_context))
        )

    def _gru_rollout(
        self,
        *,
        init_state: torch.Tensor,
        num_steps: int,
        context_vector: torch.Tensor | None,
    ) -> torch.Tensor:
        assert self.gru is not None
        step_inputs = init_state.unsqueeze(1).expand(init_state.shape[0], num_steps, -1)
        step_inputs = step_inputs + self.step_emb[:, :num_steps, :]
        if context_vector is not None:
            assert self.context_input_proj is not None
            step_inputs = step_inputs + self.context_input_proj(context_vector).unsqueeze(1)

        hidden0 = init_state.unsqueeze(0).expand(self.gru.num_layers, -1, -1).contiguous()
        rollout, _ = self.gru(step_inputs, hidden0)
        if context_vector is not None:
            assert self.context_output_proj is not None
            rollout = rollout + self.context_output_proj(context_vector).unsqueeze(1)
        return rollout

    def _mamba_rollout(
        self,
        *,
        init_state: torch.Tensor,
        num_steps: int,
        context_vector: torch.Tensor | None,
    ) -> torch.Tensor:
        assert self.mamba_blocks is not None

        prev_latent = init_state
        rollout = []
        prefix_inputs = []
        context_input = None
        context_output = None
        if context_vector is not None:
            assert self.context_input_proj is not None
            assert self.context_output_proj is not None
            context_input = self.context_input_proj(context_vector)
            context_output = self.context_output_proj(context_vector)

        for step_idx in range(num_steps):
            step_input = prev_latent + self.step_emb[:, step_idx, :]
            if context_input is not None:
                step_input = step_input + context_input

            # Re-running the causal Mamba stack over the growing latent prefix keeps the
            # rollout logic simple while still feeding each predicted latent back into the
            # next step's input. With <= 50 bins per horizon, the O(T^2) cost is acceptable.
            prefix_inputs.append(step_input)
            hidden = torch.stack(prefix_inputs, dim=1)
            for block in self.mamba_blocks:
                hidden = block(hidden)

            current = hidden[:, -1, :]
            if self.output_residual:
                current = current + step_input
            if context_output is not None:
                current = current + context_output

            rollout.append(current)
            prev_latent = current

        return torch.stack(rollout, dim=1)

    def forward(self, encoder_latents: torch.Tensor, num_steps: int) -> torch.Tensor:
        if num_steps < 1:
            raise ValueError("num_steps must be >= 1")
        if num_steps > self.step_emb.shape[1]:
            raise ValueError(
                f"num_steps ({num_steps}) exceeds max supported steps ({self.step_emb.shape[1]})"
            )

        pooled = self._pool_encoder_latents(encoder_latents)
        init_state = self._initial_state(pooled)
        context_vector = self._context_vector(pooled)

        if self.backbone == "gru":
            rollout = self._gru_rollout(
                init_state=init_state,
                num_steps=num_steps,
                context_vector=context_vector,
            )
        else:
            rollout = self._mamba_rollout(
                init_state=init_state,
                num_steps=num_steps,
                context_vector=context_vector,
            )

        return self.output_norm(self.output_proj(rollout))
