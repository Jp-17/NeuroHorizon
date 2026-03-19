"""NeuroHorizon model with baseline and diffusion-flow decoder variants."""

from __future__ import annotations

from typing import Dict, Optional
import logging

import numpy as np
import torch
import torch.nn as nn
from torchtyping import TensorType
from temporaldata import Data

from torch_brain.data import pad, pad2d, pad8, track_mask, track_mask8
from torch_brain.nn import (
    AutoregressiveDecoder,
    DiffusionFlowDecoder,
    Embedding,
    FeedForward,
    InfiniteVocabEmbedding,
    PerNeuronMLPHead,
    RotaryCrossAttention,
    RotarySelfAttention,
    RotaryTimeEmbedding,
)
from torch_brain.utils import create_linspace_latent_tokens, create_start_end_unit_tokens

logger = logging.getLogger(__name__)


class NeuroHorizon(nn.Module):
    """NeuroHorizon spike prediction model."""

    def __init__(
        self,
        *,
        sequence_length: float,
        pred_window: float = 0.250,
        bin_size: float = 0.020,
        latent_step: float = 0.05,
        num_latents_per_step: int = 64,
        dim: int = 512,
        enc_depth: int = 2,
        dec_depth: int = 2,
        dim_head: int = 64,
        cross_heads: int = 2,
        self_heads: int = 8,
        ffn_dropout: float = 0.2,
        lin_dropout: float = 0.4,
        atn_dropout: float = 0.0,
        emb_init_scale: float = 0.02,
        t_min: float = 1e-4,
        t_max: float = 2.0627,
        max_pred_bins: int = 50,
        causal_decoder: bool = True,
        feedback_method: str = "none",
        decoder_variant: str = "query_aug",
        flow_target_space: str = "log1p_count",
        flow_match_mode: str = "rectified",
        flow_steps_eval: int = 20,
        flow_solver: str = "euler",
        conditioning_dropout: float = 0.0,
        flow_eval_seed: int = 42,
    ):
        super().__init__()

        if sequence_length <= pred_window:
            raise ValueError(
                f"sequence_length ({sequence_length}) must be > pred_window ({pred_window})"
            )
        if decoder_variant not in {"query_aug", "diffusion_flow"}:
            raise ValueError(
                f"Unsupported decoder_variant={decoder_variant!r}; use 'query_aug' or 'diffusion_flow'"
            )
        if decoder_variant == "query_aug" and feedback_method != "none":
            raise ValueError(
                "The dev/diffusion branch keeps only the no-feedback baseline AR path. "
                "Set feedback_method='none'."
            )
        if decoder_variant == "diffusion_flow" and flow_match_mode != "rectified":
            raise ValueError("Only rectified flow matching is implemented in this branch")

        self.sequence_length = sequence_length
        self.pred_window = pred_window
        self.hist_window = sequence_length - pred_window
        self.bin_size = bin_size
        self.T_pred_bins = int(round(pred_window / bin_size))
        self.latent_step = latent_step
        self.num_latents_per_step = num_latents_per_step
        self.dim = dim
        self.decoder_variant = decoder_variant
        self.feedback_method = feedback_method
        self.flow_target_space = flow_target_space
        self.requires_target_counts = decoder_variant == "diffusion_flow"

        self.unit_emb = InfiniteVocabEmbedding(dim, init_scale=emb_init_scale)
        self.session_emb = InfiniteVocabEmbedding(dim, init_scale=emb_init_scale)
        self.token_type_emb = Embedding(4, dim, init_scale=emb_init_scale)
        self.latent_emb = Embedding(num_latents_per_step, dim, init_scale=emb_init_scale)
        self.rotary_emb = RotaryTimeEmbedding(
            head_dim=dim_head,
            rotate_dim=dim_head // 2,
            t_min=t_min,
            t_max=t_max,
        )
        self.dropout = nn.Dropout(p=lin_dropout)

        self.enc_atn = RotaryCrossAttention(
            dim=dim,
            heads=cross_heads,
            dropout=atn_dropout,
            dim_head=dim_head,
            rotate_value=True,
        )
        self.enc_ffn = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim=dim, dropout=ffn_dropout),
        )
        self.proc_layers = nn.ModuleList()
        for _ in range(enc_depth):
            self.proc_layers.append(
                nn.Sequential(
                    RotarySelfAttention(
                        dim=dim,
                        heads=self_heads,
                        dropout=atn_dropout,
                        dim_head=dim_head,
                        rotate_value=True,
                    ),
                    nn.Sequential(
                        nn.LayerNorm(dim),
                        FeedForward(dim=dim, dropout=ffn_dropout),
                    ),
                )
            )

        if decoder_variant == "query_aug":
            self.bin_emb = nn.Parameter(torch.randn(1, max_pred_bins, dim) * emb_init_scale)
            self.ar_decoder = AutoregressiveDecoder(
                dim=dim,
                depth=dec_depth,
                dim_head=dim_head,
                cross_heads=cross_heads,
                self_heads=self_heads,
                ffn_dropout=ffn_dropout,
                atn_dropout=atn_dropout,
                causal=causal_decoder,
            )
            self.head = PerNeuronMLPHead(dim)
            self.diffusion_decoder = None
        else:
            self.bin_emb = None
            self.ar_decoder = None
            self.head = None
            self.diffusion_decoder = DiffusionFlowDecoder(
                dim=dim,
                depth=dec_depth,
                dim_head=dim_head,
                cross_heads=cross_heads,
                self_heads=self_heads,
                ffn_dropout=ffn_dropout,
                atn_dropout=atn_dropout,
                condition_dropout=conditioning_dropout,
                eval_steps=flow_steps_eval,
                eval_seed=flow_eval_seed,
                target_space=flow_target_space,
                solver=flow_solver,
            )

    def _encode_history(
        self,
        input_unit_index,
        input_timestamps,
        input_token_type,
        input_mask,
        latent_index,
        latent_timestamps,
    ):
        inputs = self.unit_emb(input_unit_index) + self.token_type_emb(input_token_type)
        input_time_emb = self.rotary_emb(input_timestamps)

        latents = self.latent_emb(latent_index)
        latent_time_emb = self.rotary_emb(latent_timestamps)

        latents = latents + self.enc_atn(
            latents,
            inputs,
            latent_time_emb,
            input_time_emb,
            input_mask,
        )
        latents = latents + self.enc_ffn(latents)

        for self_attn, self_ff in self.proc_layers:
            latents = latents + self.dropout(self_attn(latents, latent_time_emb))
            latents = latents + self.dropout(self_ff(latents))

        return latents, latent_time_emb

    def compute_training_loss(
        self,
        *,
        input_unit_index: TensorType["batch", "n_in", int],
        input_timestamps: TensorType["batch", "n_in", float],
        input_token_type: TensorType["batch", "n_in", int],
        input_mask: Optional[TensorType["batch", "n_in", bool]] = None,
        latent_index: TensorType["batch", "n_latent", int],
        latent_timestamps: TensorType["batch", "n_latent", float],
        bin_timestamps: TensorType["batch", "n_bins", float],
        target_unit_index: TensorType["batch", "n_units", int],
        target_unit_mask: Optional[TensorType["batch", "n_units", bool]] = None,
        target_counts: Optional[TensorType["batch", "n_bins", "n_units"]] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.decoder_variant != "diffusion_flow":
            raise RuntimeError("compute_training_loss is only used by diffusion_flow")
        if target_counts is None:
            raise ValueError("target_counts is required for diffusion_flow training")

        latents, latent_time_emb = self._encode_history(
            input_unit_index,
            input_timestamps,
            input_token_type,
            input_mask,
            latent_index,
            latent_timestamps,
        )
        bin_time_emb = self.rotary_emb(bin_timestamps)
        unit_embs = self.unit_emb(target_unit_index)
        return self.diffusion_decoder.compute_flow_matching_loss(
            target_counts=target_counts,
            bin_time_emb=bin_time_emb,
            encoder_latents=latents,
            latent_time_emb=latent_time_emb,
            unit_embs=unit_embs,
            unit_mask=target_unit_mask,
        )

    def forward(
        self,
        *,
        input_unit_index: TensorType["batch", "n_in", int],
        input_timestamps: TensorType["batch", "n_in", float],
        input_token_type: TensorType["batch", "n_in", int],
        input_mask: Optional[TensorType["batch", "n_in", bool]] = None,
        latent_index: TensorType["batch", "n_latent", int],
        latent_timestamps: TensorType["batch", "n_latent", float],
        bin_timestamps: TensorType["batch", "n_bins", float],
        target_unit_index: TensorType["batch", "n_units", int],
        target_unit_mask: Optional[TensorType["batch", "n_units", bool]] = None,
        target_counts=None,
    ) -> TensorType["batch", "n_bins", "n_units"]:
        if self.unit_emb.is_lazy():
            raise ValueError(
                "Unit vocabulary not initialized. Call model.unit_emb.initialize_vocab(unit_ids)"
            )
        if self.decoder_variant == "diffusion_flow":
            return self.generate(
                input_unit_index=input_unit_index,
                input_timestamps=input_timestamps,
                input_token_type=input_token_type,
                input_mask=input_mask,
                latent_index=latent_index,
                latent_timestamps=latent_timestamps,
                bin_timestamps=bin_timestamps,
                target_unit_index=target_unit_index,
                target_unit_mask=target_unit_mask,
            )

        latents, latent_time_emb = self._encode_history(
            input_unit_index,
            input_timestamps,
            input_token_type,
            input_mask,
            latent_index,
            latent_timestamps,
        )

        batch_size = input_unit_index.shape[0]
        num_bins = bin_timestamps.shape[1]
        bin_queries = self.bin_emb[:, :num_bins, :].expand(batch_size, -1, -1).clone()
        bin_time_emb = self.rotary_emb(bin_timestamps)
        unit_embs = self.unit_emb(target_unit_index)

        bin_repr = self.ar_decoder(
            bin_queries,
            bin_time_emb,
            latents,
            latent_time_emb,
        )
        return self.head(bin_repr, unit_embs)

    @torch.no_grad()
    def generate(
        self,
        *,
        input_unit_index,
        input_timestamps,
        input_token_type,
        input_mask=None,
        latent_index,
        latent_timestamps,
        bin_timestamps,
        target_unit_index,
        target_unit_mask=None,
    ):
        if self.unit_emb.is_lazy():
            raise ValueError("Unit vocabulary not initialized.")

        latents, latent_time_emb = self._encode_history(
            input_unit_index,
            input_timestamps,
            input_token_type,
            input_mask,
            latent_index,
            latent_timestamps,
        )

        batch_size = input_unit_index.shape[0]
        num_bins = bin_timestamps.shape[1]
        unit_embs = self.unit_emb(target_unit_index)

        if self.decoder_variant == "diffusion_flow":
            bin_time_emb = self.rotary_emb(bin_timestamps)
            return self.diffusion_decoder.sample_log_rate(
                bin_time_emb=bin_time_emb,
                encoder_latents=latents,
                latent_time_emb=latent_time_emb,
                unit_embs=unit_embs,
                unit_mask=target_unit_mask,
                device=latents.device,
                dtype=latents.dtype,
            )

        all_log_rates = []
        for t in range(num_bins):
            cur_queries = self.bin_emb[:, : t + 1, :].expand(batch_size, -1, -1).clone()
            cur_time_emb = self.rotary_emb(bin_timestamps[:, : t + 1])
            cur_repr = self.ar_decoder(
                cur_queries,
                cur_time_emb,
                latents,
                latent_time_emb,
            )
            latest_repr = cur_repr[:, -1:, :]
            log_rate_t = self.head(latest_repr, unit_embs)
            all_log_rates.append(log_rate_t)

        return torch.cat(all_log_rates, dim=1)

    def tokenize(self, data: Data) -> Dict:
        hist_end = self.hist_window
        pred_start = hist_end
        pred_end = self.sequence_length

        unit_ids = data.units.id
        num_units = len(unit_ids)
        spike_unit_index = data.spikes.unit_index
        spike_timestamps = data.spikes.timestamps

        hist_mask = spike_timestamps < hist_end
        hist_spike_idx = spike_unit_index[hist_mask]
        hist_spike_ts = spike_timestamps[hist_mask]

        se_token_type, se_unit_idx, se_ts = create_start_end_unit_tokens(unit_ids, 0, hist_end)

        combined_token_type = np.concatenate([se_token_type, np.zeros_like(hist_spike_idx)])
        combined_unit_idx = np.concatenate([se_unit_idx, hist_spike_idx])
        combined_ts = np.concatenate([se_ts, hist_spike_ts])

        local_to_global = np.array(self.unit_emb.tokenizer(unit_ids))
        combined_unit_idx_global = local_to_global[combined_unit_idx]

        latent_index, latent_timestamps = create_linspace_latent_tokens(
            0,
            hist_end,
            step=self.latent_step,
            num_latents_per_step=self.num_latents_per_step,
        )

        pred_mask = (spike_timestamps >= pred_start) & (spike_timestamps < pred_end)
        pred_spike_idx = spike_unit_index[pred_mask]
        pred_spike_ts = spike_timestamps[pred_mask]

        spike_counts = np.zeros((self.T_pred_bins, num_units), dtype=np.float32)
        if len(pred_spike_ts) > 0:
            bin_indices = np.floor((pred_spike_ts - pred_start) / self.bin_size).astype(np.int64)
            bin_indices = np.clip(bin_indices, 0, self.T_pred_bins - 1)
            for bin_idx, unit_idx in zip(bin_indices, pred_spike_idx):
                spike_counts[bin_idx, unit_idx] += 1.0

        bin_timestamps = np.linspace(
            pred_start + self.bin_size / 2,
            pred_end - self.bin_size / 2,
            self.T_pred_bins,
        ).astype(np.float32)
        global_unit_indices = local_to_global.copy()

        return {
            "model_inputs": {
                "input_unit_index": pad8(combined_unit_idx_global),
                "input_timestamps": pad8(combined_ts),
                "input_token_type": pad8(combined_token_type),
                "input_mask": track_mask8(combined_unit_idx_global),
                "latent_index": latent_index,
                "latent_timestamps": latent_timestamps,
                "bin_timestamps": bin_timestamps,
                "target_unit_index": pad(global_unit_indices),
                "target_unit_mask": track_mask(global_unit_indices),
            },
            "target_spike_counts": pad2d(torch.tensor(spike_counts)),
            "session_id": data.session.id,
        }
