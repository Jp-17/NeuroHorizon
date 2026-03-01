"""NeuroHorizon model: autoregressive spike prediction with Perceiver encoder.

Based on POYOPlus architecture. Reuses encoder + processor, adds autoregressive
decoder with causal self-attention and per-neuron MLP head.

Architecture:
    Spike events (history) → Perceiver Encoder → Processor → Latents
    Latents + Bin Queries → AR Decoder (causal) → PerNeuronMLPHead → log spike rates
"""

from typing import Dict, List, Optional, Tuple
import logging

import numpy as np
import torch
import torch.nn as nn
from torchtyping import TensorType
from temporaldata import Data

from torch_brain.data import pad, pad8, pad2d, track_mask, track_mask8
from torch_brain.nn import (
    Embedding,
    FeedForward,
    InfiniteVocabEmbedding,
    RotaryCrossAttention,
    RotarySelfAttention,
    RotaryTimeEmbedding,
)
from torch_brain.nn.autoregressive_decoder import AutoregressiveDecoder, PerNeuronMLPHead
from torch_brain.utils import create_linspace_latent_tokens, create_start_end_unit_tokens

logger = logging.getLogger(__name__)


class NeuroHorizon(nn.Module):
    """NeuroHorizon: autoregressive spike prediction model.

    Extends POYOPlus with:
    - Dual-window design: history window (encoder input) + prediction window (target)
    - Autoregressive decoder with causal self-attention
    - Per-neuron MLP head for spike count prediction
    - Poisson NLL loss for spike count targets

    Args:
        sequence_length: Total window length in seconds (history + prediction)
        pred_window: Prediction window length in seconds
        bin_size: Bin size in seconds for spike count discretization
        latent_step: Step size for latent grid in seconds
        num_latents_per_step: Number of latent tokens per step
        dim: Model dimension
        enc_depth: Number of processor self-attention layers
        dec_depth: Number of decoder layers
        dim_head: Dimension per attention head
        cross_heads: Number of cross-attention heads
        self_heads: Number of self-attention heads
        ffn_dropout: FFN dropout rate
        lin_dropout: Linear/residual dropout rate
        atn_dropout: Attention dropout rate
        emb_init_scale: Embedding initialization scale
        t_min: Minimum timescale for rotary embeddings
        t_max: Maximum timescale for rotary embeddings
        max_pred_bins: Maximum number of prediction bins (for bin_emb sizing)
    """

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
    ):
        super().__init__()

        assert sequence_length > pred_window, (
            f"sequence_length ({sequence_length}) must be > pred_window ({pred_window})"
        )

        self.sequence_length = sequence_length
        self.pred_window = pred_window
        self.hist_window = sequence_length - pred_window
        self.bin_size = bin_size
        self.T_pred_bins = int(round(pred_window / bin_size))
        self.latent_step = latent_step
        self.num_latents_per_step = num_latents_per_step
        self.dim = dim

        # ── Embeddings ──
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

        # ── Perceiver Encoder (1 cross-attn + FFN) ──
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

        # ── Processor (enc_depth layers of self-attn + FFN) ──
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

        # ── Autoregressive Decoder ──
        self.bin_emb = nn.Parameter(
            torch.randn(1, max_pred_bins, dim) * emb_init_scale
        )
        self.ar_decoder = AutoregressiveDecoder(
            dim=dim,
            depth=dec_depth,
            dim_head=dim_head,
            cross_heads=cross_heads,
            self_heads=self_heads,
            ffn_dropout=ffn_dropout,
            atn_dropout=atn_dropout,
        )

        # ── Per-Neuron MLP Head ──
        self.head = PerNeuronMLPHead(dim)

    def forward(
        self,
        *,
        # Encoder inputs (history window spike events)
        input_unit_index: TensorType["batch", "n_in", int],
        input_timestamps: TensorType["batch", "n_in", float],
        input_token_type: TensorType["batch", "n_in", int],
        input_mask: Optional[TensorType["batch", "n_in", bool]] = None,
        # Latent grid
        latent_index: TensorType["batch", "n_latent", int],
        latent_timestamps: TensorType["batch", "n_latent", float],
        # Decoder inputs (prediction window)
        bin_timestamps: TensorType["batch", "n_bins", float],
        # Unit info for PerNeuronMLPHead
        target_unit_index: TensorType["batch", "n_units", int],
        target_unit_mask: Optional[TensorType["batch", "n_units", bool]] = None,
    ) -> TensorType["batch", "n_bins", "n_units"]:
        """Forward pass (teacher forcing mode).

        Returns log spike rates [B, T_pred, N_units_padded].
        Use target_unit_mask to mask padding before computing loss.
        """
        if self.unit_emb.is_lazy():
            raise ValueError(
                "Unit vocabulary not initialized. Call model.unit_emb.initialize_vocab(unit_ids)"
            )

        # ── 1. Encoder input embeddings ──
        inputs = self.unit_emb(input_unit_index) + self.token_type_emb(input_token_type)
        input_time_emb = self.rotary_emb(input_timestamps)

        # ── 2. Latent tokens ──
        latents = self.latent_emb(latent_index)
        latent_time_emb = self.rotary_emb(latent_timestamps)

        # ── 3. Perceiver encoder ──
        latents = latents + self.enc_atn(
            latents, inputs, latent_time_emb, input_time_emb, input_mask,
        )
        latents = latents + self.enc_ffn(latents)

        # ── 4. Processor ──
        for self_attn, self_ff in self.proc_layers:
            latents = latents + self.dropout(self_attn(latents, latent_time_emb))
            latents = latents + self.dropout(self_ff(latents))

        # ── 5. Bin queries ──
        B = input_unit_index.shape[0]
        T_pred = bin_timestamps.shape[1]
        bin_queries = self.bin_emb[:, :T_pred, :].expand(B, -1, -1).clone()
        bin_time_emb = self.rotary_emb(bin_timestamps)

        # ── 6. Autoregressive decoder ──
        bin_repr = self.ar_decoder(
            bin_queries, bin_time_emb, latents, latent_time_emb,
        )

        # ── 7. Per-neuron prediction ──
        unit_embs = self.unit_emb(target_unit_index)  # [B, N_padded, dim]
        log_rate = self.head(bin_repr, unit_embs)  # [B, T_pred, N_padded]

        return log_rate

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
        """Autoregressive generation (step-by-step inference).

        Same inputs as forward(), but generates predictions one bin at a time.
        Returns log spike rates [B, T_pred, N_units_padded].
        """
        if self.unit_emb.is_lazy():
            raise ValueError("Unit vocabulary not initialized.")

        # Encode (same as forward)
        inputs = self.unit_emb(input_unit_index) + self.token_type_emb(input_token_type)
        input_time_emb = self.rotary_emb(input_timestamps)

        latents = self.latent_emb(latent_index)
        latent_time_emb = self.rotary_emb(latent_timestamps)

        latents = latents + self.enc_atn(
            latents, inputs, latent_time_emb, input_time_emb, input_mask,
        )
        latents = latents + self.enc_ffn(latents)

        for self_attn, self_ff in self.proc_layers:
            latents = latents + self_attn(latents, latent_time_emb)
            latents = latents + self_ff(latents)

        # Step-by-step decoding
        B = input_unit_index.shape[0]
        T_pred = bin_timestamps.shape[1]
        unit_embs = self.unit_emb(target_unit_index)  # [B, N_padded, dim]

        all_log_rates = []
        for t in range(T_pred):
            # Use all bins up to and including current step
            cur_queries = self.bin_emb[:, :t + 1, :].expand(B, -1, -1).clone()
            cur_time_emb = self.rotary_emb(bin_timestamps[:, :t + 1])

            # Decode with causal mask (auto-created inside decoder)
            cur_repr = self.ar_decoder(
                cur_queries, cur_time_emb, latents, latent_time_emb,
            )

            # Predict only for the latest bin
            latest_repr = cur_repr[:, -1:, :]  # [B, 1, dim]
            log_rate_t = self.head(latest_repr, unit_embs)  # [B, 1, N_padded]
            all_log_rates.append(log_rate_t)

        return torch.cat(all_log_rates, dim=1)  # [B, T_pred, N_padded]

    def tokenize(self, data: Data) -> Dict:
        """Tokenize data for NeuroHorizon model.

        Splits the sampling window into:
        - History window [0, hist_window]: spike events for encoder input
        - Prediction window [hist_window, sequence_length]: binned spike counts as target

        This tokenizer should be applied as the last transform.
        """
        hist_end = self.hist_window
        pred_start = hist_end
        pred_end = self.sequence_length

        unit_ids = data.units.id
        N_units = len(unit_ids)
        spike_unit_index = data.spikes.unit_index
        spike_timestamps = data.spikes.timestamps

        # ── 1. History window: spike events for encoder ──
        hist_mask = spike_timestamps < hist_end
        hist_spike_idx = spike_unit_index[hist_mask]
        hist_spike_ts = spike_timestamps[hist_mask]

        # Create start/end tokens for each unit (spanning history window)
        se_token_type, se_unit_idx, se_ts = create_start_end_unit_tokens(
            unit_ids, 0, hist_end
        )

        # Combine: [start/end tokens] + [spike tokens]
        combined_token_type = np.concatenate(
            [se_token_type, np.zeros_like(hist_spike_idx)]
        )
        combined_unit_idx = np.concatenate([se_unit_idx, hist_spike_idx])
        combined_ts = np.concatenate([se_ts, hist_spike_ts])

        # Map local unit indices to global indices
        local_to_global = np.array(self.unit_emb.tokenizer(unit_ids))
        combined_unit_idx_global = local_to_global[combined_unit_idx]

        # ── 2. Latent tokens (history window only) ──
        latent_index, latent_timestamps = create_linspace_latent_tokens(
            0, hist_end,
            step=self.latent_step,
            num_latents_per_step=self.num_latents_per_step,
        )

        # ── 3. Prediction window: bin spike events into counts ──
        pred_mask = (spike_timestamps >= pred_start) & (spike_timestamps < pred_end)
        pred_spike_idx = spike_unit_index[pred_mask]
        pred_spike_ts = spike_timestamps[pred_mask]

        T_bins = self.T_pred_bins
        spike_counts = np.zeros((T_bins, N_units), dtype=np.float32)

        if len(pred_spike_ts) > 0:
            bin_indices = np.floor(
                (pred_spike_ts - pred_start) / self.bin_size
            ).astype(np.int64)
            # Clip to valid range (edge case: spike exactly at pred_end)
            bin_indices = np.clip(bin_indices, 0, T_bins - 1)
            for b, u in zip(bin_indices, pred_spike_idx):
                spike_counts[b, u] += 1.0

        # Bin center timestamps
        bin_timestamps = np.linspace(
            pred_start + self.bin_size / 2,
            pred_end - self.bin_size / 2,
            T_bins,
        ).astype(np.float32)

        # Global unit indices for PerNeuronMLPHead
        global_unit_indices = local_to_global.copy()

        data_dict = {
            "model_inputs": {
                # Encoder inputs (history window)
                "input_unit_index": pad8(combined_unit_idx_global),
                "input_timestamps": pad8(combined_ts),
                "input_token_type": pad8(combined_token_type),
                "input_mask": track_mask8(combined_unit_idx_global),
                # Latent grid
                "latent_index": latent_index,
                "latent_timestamps": latent_timestamps,
                # Decoder inputs
                "bin_timestamps": bin_timestamps,
                # Unit info
                "target_unit_index": pad(global_unit_indices),
                "target_unit_mask": track_mask(global_unit_indices),
            },
            # Targets
            "target_spike_counts": pad2d(torch.tensor(spike_counts)),
            # Session info for evaluation
            "session_id": data.session.id,
        }

        return data_dict
