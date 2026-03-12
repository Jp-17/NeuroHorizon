"""NeuroHorizon model: autoregressive spike prediction with Perceiver encoder.

Based on POYOPlus architecture. Reuses encoder + processor, adds an
autoregressive decoder with either query augmentation (v2 baseline) or
structured prediction memory (1.9 iteration).
"""

from typing import Dict, List, Optional
import logging

import numpy as np
import torch
import torch.nn as nn
from torchtyping import TensorType
from temporaldata import Data

from torch_brain.data import pad, pad8, pad2d, track_mask, track_mask8
from torch_brain.nn import (
    AutoregressiveDecoder,
    Embedding,
    FeedForward,
    InfiniteVocabEmbedding,
    PerNeuronMLPHead,
    PredictionMemoryEncoder,
    RotaryCrossAttention,
    RotarySelfAttention,
    RotaryTimeEmbedding,
    create_causal_mask,
)
from torch_brain.nn.prediction_feedback import build_feedback_encoder
from torch_brain.utils import create_linspace_latent_tokens, create_start_end_unit_tokens

logger = logging.getLogger(__name__)


class NeuroHorizon(nn.Module):
    """NeuroHorizon: autoregressive spike prediction model."""

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
        prediction_memory_k: int = 4,
        prediction_memory_heads: int = 4,
    ):
        super().__init__()

        if sequence_length <= pred_window:
            raise ValueError(
                f"sequence_length ({sequence_length}) must be > pred_window ({pred_window})"
            )
        if decoder_variant not in {"query_aug", "prediction_memory"}:
            raise ValueError(
                "Unknown decoder_variant="
                f"{decoder_variant!r}; choose 'query_aug' or 'prediction_memory'"
            )

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
        self.prediction_memory_k = prediction_memory_k
        self.requires_target_counts = decoder_variant == "prediction_memory" or feedback_method != "none"

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

        if decoder_variant == "prediction_memory":
            self.feedback_encoder = None
            self.prediction_memory_encoder = PredictionMemoryEncoder(
                dim=dim,
                num_memory_tokens=prediction_memory_k,
                num_heads=prediction_memory_heads,
            )
        else:
            self.prediction_memory_encoder = None
            if feedback_method != "none":
                self.feedback_encoder = build_feedback_encoder(feedback_method, dim)
            else:
                self.feedback_encoder = None

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

    def _build_query_aug_feedback(self, target_counts, target_unit_index, target_unit_mask):
        if self.feedback_encoder is None or target_counts is None:
            return None

        unit_embs = self.unit_emb(target_unit_index)
        B, T_pred, N = target_counts.shape
        shifted_counts = torch.cat(
            [
                torch.zeros(B, 1, N, device=target_counts.device, dtype=target_counts.dtype),
                target_counts[:, :-1, :],
            ],
            dim=1,
        )

        BT = B * T_pred
        counts_flat = shifted_counts.reshape(BT, N)
        unit_flat = unit_embs.unsqueeze(1).expand(B, T_pred, -1, -1).reshape(BT, N, self.dim)
        mask_flat = None
        if target_unit_mask is not None:
            mask_flat = target_unit_mask.unsqueeze(1).expand(B, T_pred, -1).reshape(BT, N)

        feedback_flat = self.feedback_encoder(counts_flat, unit_flat, mask_flat)
        return feedback_flat.reshape(B, T_pred, self.dim)

    def _encode_counts_to_memory_tokens(self, counts, unit_embs, unit_mask):
        if self.prediction_memory_encoder is None:
            raise RuntimeError("prediction_memory_encoder is not initialized")
        counts = torch.log1p(counts.clamp_min(0.0))
        return self.prediction_memory_encoder(counts, unit_embs, unit_mask)

    def _build_prediction_memory_mask(self, num_steps, device):
        mask = create_causal_mask(num_steps, device=device)
        return mask.repeat_interleave(self.prediction_memory_k, dim=-1)

    def _build_prediction_memory_time_emb(self, bin_time_emb):
        B, T_pred, _ = bin_time_emb.shape
        return (
            bin_time_emb.unsqueeze(2)
            .expand(B, T_pred, self.prediction_memory_k, -1)
            .reshape(B, T_pred * self.prediction_memory_k, -1)
        )

    def _build_prediction_memory_train(
        self,
        target_counts,
        unit_embs,
        unit_mask,
        bin_time_emb,
    ):
        B, T_pred, N = target_counts.shape
        shifted_counts = torch.cat(
            [
                torch.zeros(B, 1, N, device=target_counts.device, dtype=target_counts.dtype),
                target_counts[:, :-1, :],
            ],
            dim=1,
        )

        BT = B * T_pred
        unit_flat = unit_embs.unsqueeze(1).expand(B, T_pred, -1, -1).reshape(BT, N, self.dim)
        mask_flat = None
        if unit_mask is not None:
            mask_flat = unit_mask.unsqueeze(1).expand(B, T_pred, -1).reshape(BT, N)

        memory_flat = self._encode_counts_to_memory_tokens(
            shifted_counts.reshape(BT, N),
            unit_flat,
            mask_flat,
        )
        memory = memory_flat.reshape(B, T_pred, self.prediction_memory_k, self.dim)
        memory[:, 0] = 0.0

        prediction_memory = memory.reshape(B, T_pred * self.prediction_memory_k, self.dim)
        prediction_memory_time_emb = self._build_prediction_memory_time_emb(bin_time_emb)
        prediction_memory_mask = self._build_prediction_memory_mask(
            T_pred,
            device=target_counts.device,
        )
        prediction_memory_mask = prediction_memory_mask.unsqueeze(0).expand(B, -1, -1)
        return prediction_memory, prediction_memory_time_emb, prediction_memory_mask

    def _build_prediction_memory_rollout(
        self,
        prev_predicted_counts,
        unit_embs,
        unit_mask,
        cur_time_emb,
    ):
        B, T_cur, _ = cur_time_emb.shape
        zero_memory = torch.zeros(
            B,
            self.prediction_memory_k,
            self.dim,
            device=unit_embs.device,
            dtype=unit_embs.dtype,
        )
        memory_slots = [zero_memory]
        for counts in prev_predicted_counts:
            memory_slots.append(self._encode_counts_to_memory_tokens(counts, unit_embs, unit_mask))

        memory = torch.stack(memory_slots, dim=1)
        prediction_memory = memory.reshape(B, T_cur * self.prediction_memory_k, self.dim)
        prediction_memory_time_emb = self._build_prediction_memory_time_emb(cur_time_emb)
        prediction_memory_mask = self._build_prediction_memory_mask(
            T_cur,
            device=unit_embs.device,
        )
        prediction_memory_mask = prediction_memory_mask.unsqueeze(0).expand(B, -1, -1)
        return prediction_memory, prediction_memory_time_emb, prediction_memory_mask

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
        if self.requires_target_counts and target_counts is None:
            raise ValueError(
                f"target_counts is required for decoder_variant={self.decoder_variant!r}"
            )

        latents, latent_time_emb = self._encode_history(
            input_unit_index,
            input_timestamps,
            input_token_type,
            input_mask,
            latent_index,
            latent_timestamps,
        )

        B = input_unit_index.shape[0]
        T_pred = bin_timestamps.shape[1]
        bin_queries = self.bin_emb[:, :T_pred, :].expand(B, -1, -1).clone()
        bin_time_emb = self.rotary_emb(bin_timestamps)
        unit_embs = self.unit_emb(target_unit_index)

        feedback = None
        prediction_memory = None
        prediction_memory_time_emb = None
        prediction_memory_mask = None

        if self.decoder_variant == "prediction_memory":
            prediction_memory, prediction_memory_time_emb, prediction_memory_mask = (
                self._build_prediction_memory_train(
                    target_counts,
                    unit_embs,
                    target_unit_mask,
                    bin_time_emb,
                )
            )
        elif self.feedback_encoder is not None:
            feedback = self._build_query_aug_feedback(
                target_counts,
                target_unit_index,
                target_unit_mask,
            )

        bin_repr = self.ar_decoder(
            bin_queries,
            bin_time_emb,
            latents,
            latent_time_emb,
            feedback=feedback,
            prediction_memory=prediction_memory,
            prediction_memory_time_emb=prediction_memory_time_emb,
            prediction_memory_mask=prediction_memory_mask,
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

        B = input_unit_index.shape[0]
        T_pred = bin_timestamps.shape[1]
        unit_embs = self.unit_emb(target_unit_index)

        all_log_rates = []
        prev_predicted_counts: List[torch.Tensor] = []

        for t in range(T_pred):
            cur_queries = self.bin_emb[:, : t + 1, :].expand(B, -1, -1).clone()
            cur_time_emb = self.rotary_emb(bin_timestamps[:, : t + 1])

            feedback = None
            prediction_memory = None
            prediction_memory_time_emb = None
            prediction_memory_mask = None

            if self.decoder_variant == "prediction_memory":
                prediction_memory, prediction_memory_time_emb, prediction_memory_mask = (
                    self._build_prediction_memory_rollout(
                        prev_predicted_counts,
                        unit_embs,
                        target_unit_mask,
                        cur_time_emb,
                    )
                )
            elif self.feedback_encoder is not None:
                fb_list = []
                for s in range(t + 1):
                    if s == 0 or len(prev_predicted_counts) == 0:
                        fb_list.append(
                            torch.zeros(B, self.dim, device=latents.device, dtype=latents.dtype)
                        )
                    else:
                        fb_list.append(
                            self.feedback_encoder(
                                prev_predicted_counts[s - 1],
                                unit_embs,
                                target_unit_mask,
                            )
                        )
                feedback = torch.stack(fb_list, dim=1)

            cur_repr = self.ar_decoder(
                cur_queries,
                cur_time_emb,
                latents,
                latent_time_emb,
                feedback=feedback,
                prediction_memory=prediction_memory,
                prediction_memory_time_emb=prediction_memory_time_emb,
                prediction_memory_mask=prediction_memory_mask,
            )
            latest_repr = cur_repr[:, -1:, :]
            log_rate_t = self.head(latest_repr, unit_embs)
            all_log_rates.append(log_rate_t)
            prev_predicted_counts.append(torch.exp(log_rate_t.squeeze(1).clamp(-10, 10)))

        return torch.cat(all_log_rates, dim=1)

    def tokenize(self, data: Data) -> Dict:
        hist_end = self.hist_window
        pred_start = hist_end
        pred_end = self.sequence_length

        unit_ids = data.units.id
        N_units = len(unit_ids)
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

        T_bins = self.T_pred_bins
        spike_counts = np.zeros((T_bins, N_units), dtype=np.float32)
        if len(pred_spike_ts) > 0:
            bin_indices = np.floor((pred_spike_ts - pred_start) / self.bin_size).astype(np.int64)
            bin_indices = np.clip(bin_indices, 0, T_bins - 1)
            for b, u in zip(bin_indices, pred_spike_idx):
                spike_counts[b, u] += 1.0

        bin_timestamps = np.linspace(
            pred_start + self.bin_size / 2,
            pred_end - self.bin_size / 2,
            T_bins,
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
