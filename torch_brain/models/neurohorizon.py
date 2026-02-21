"""NeuroHorizon: Unified neural encoding model.

Extends POYO+ architecture for neural encoding (spikes -> future spikes):
- IDEncoder replaces InfiniteVocabEmbedding for gradient-free cross-session generalization
- Autoregressive cross-attention decoder for spike count prediction
- Per-neuron shared MLP head for variable n_units output
- Poisson NLL loss for spike count targets
"""

from typing import Dict, Optional, Tuple
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType
from temporaldata import Data

from torch_brain.data import pad8, track_mask8
from torch_brain.nn import (
    Embedding,
    FeedForward,
    IDEncoder,
    MultimodalEncoder,
    RotaryCrossAttention,
    RotarySelfAttention,
    RotaryTimeEmbedding,
)
from torch_brain.utils import create_linspace_latent_tokens, create_start_end_unit_tokens


class CausalRotarySelfAttention(nn.Module):
    """Self-attention with causal masking and rotary embeddings.

    For the autoregressive decoder: each bin token can only attend to
    itself and previous bin tokens.
    """

    def __init__(self, *, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.dropout = dropout
        self.dim_head = dim_head

        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x, rotary_time_emb, x_mask=None):
        """
        Args:
            x: (B, N, D)
            rotary_time_emb: (B, N, D_h)
            x_mask: Optional additional mask (B, N)
        Returns:
            (B, N, D)
        """
        from einops import rearrange

        b, n, _ = x.shape
        x = self.norm(x)
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        q = rearrange(q, "b n (h d) -> b h n d", h=self.heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.heads)

        # Apply rotary embeddings
        q = RotaryTimeEmbedding.rotate(x=q, rotary_emb=rotary_time_emb, unsqueeze_dim=1)
        k = RotaryTimeEmbedding.rotate(x=k, rotary_emb=rotary_time_emb, unsqueeze_dim=1)

        # Build causal mask: (1, 1, N, N), True where attention is allowed
        causal_mask = torch.ones(n, n, dtype=torch.bool, device=x.device).tril()
        attn_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, N, N)

        # Combine with padding mask if provided
        if x_mask is not None:
            # x_mask: (B, N) -> (B, 1, 1, N) for KV masking
            kv_mask = x_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, N)
            attn_mask = attn_mask & kv_mask

        # Convert bool mask to float mask for scaled_dot_product_attention
        attn_mask_float = torch.zeros_like(attn_mask, dtype=q.dtype)
        attn_mask_float.masked_fill_(~attn_mask, float("-inf"))

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask_float,
            dropout_p=self.dropout if self.training else 0,
        )

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class PerNeuronHead(nn.Module):
    """Shared per-neuron MLP head for spike count prediction.

    Takes concatenated (bin_repr, unit_emb) and produces a scalar log-rate.
    This naturally handles variable n_units across sessions.

    Args:
        dim: Dimension of bin representations.
        emb_dim: Dimension of unit embeddings.
        hidden_dim: Hidden layer dimension.
    """

    def __init__(self, dim, emb_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + emb_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, bin_repr, unit_emb):
        """
        Args:
            bin_repr: (B, N_bins, dim) - decoder output for each time bin
            unit_emb: (B, N_units, emb_dim) - unit embeddings

        Returns:
            log_rates: (B, N_bins, N_units) - predicted log firing rates
        """
        B, N_bins, D = bin_repr.shape
        _, N_units, E = unit_emb.shape

        # Expand and concatenate: each bin paired with each unit
        bin_expanded = bin_repr.unsqueeze(2).expand(B, N_bins, N_units, D)
        unit_expanded = unit_emb.unsqueeze(1).expand(B, N_bins, N_units, E)
        combined = torch.cat([bin_expanded, unit_expanded], dim=-1)

        # Shared MLP: (B, N_bins, N_units, dim+emb_dim) -> (B, N_bins, N_units, 1)
        log_rates = self.net(combined).squeeze(-1)  # (B, N_bins, N_units)
        return log_rates


class NeuroHorizon(nn.Module):
    """NeuroHorizon: Unified neural encoding model.

    Architecture:
    1. Encoder (from POYO+): spike tokens -> Perceiver cross-attn -> latents -> self-attn
    2. Decoder (new): autoregressive cross-attention decoder predicting spike counts
    3. Output: per-neuron shared MLP head producing log-rates for Poisson NLL loss

    Args:
        sequence_length: Duration of the input spike window (seconds).
        pred_length: Duration of the prediction window (seconds).
        bin_size: Time bin size for spike count output (seconds).
        latent_step: Timestep of the latent grid (seconds).
        num_latents_per_step: Number of latent tokens per step.
        dim: Model dimension.
        depth: Number of encoder self-attention layers.
        dec_depth: Number of decoder layers.
        dim_head: Dimension per attention head.
        cross_heads: Number of cross-attention heads.
        self_heads: Number of self-attention heads.
        ref_dim: IDEncoder input feature dimension.
        ffn_dropout: Dropout for feed-forward layers.
        lin_dropout: Dropout for linear projections.
        atn_dropout: Dropout for attention.
        emb_init_scale: Embedding initialization scale.
        t_min: Minimum rotary embedding frequency.
        t_max: Maximum rotary embedding frequency.
        use_multimodal: Whether to enable multimodal cross-attention.
        multimodal_every: Insert multimodal layer every N encoder layers.
        image_dim: Dimension of image embeddings (768 for DINOv2 ViT-B/14).
        behavior_dim: Dimension of behavior features.
        use_image: Whether to use image modality.
        use_behavior: Whether to use behavior modality.
        embedding_mode: Unit embedding strategy.
            "idencoder" (default): IDEncoder MLP from reference features.
            "random": Fixed random embeddings (ablation baseline).
            "mean": Project mean-pooled features (tests individual feature importance).
    """

    def __init__(
        self,
        *,
        sequence_length: float = 1.0,
        pred_length: float = 0.5,
        bin_size: float = 0.02,
        latent_step: float = 0.1,
        num_latents_per_step: int = 64,
        dim: int = 256,
        depth: int = 4,
        dec_depth: int = 2,
        dim_head: int = 64,
        cross_heads: int = 4,
        self_heads: int = 4,
        ref_dim: int = 33,
        ffn_dropout: float = 0.2,
        lin_dropout: float = 0.2,
        atn_dropout: float = 0.0,
        emb_init_scale: float = 0.02,
        t_min: float = 1e-4,
        t_max: float = 2.0627,
        use_multimodal: bool = False,
        multimodal_every: int = 2,
        image_dim: int = 768,
        behavior_dim: int = 1,
        use_image: bool = True,
        use_behavior: bool = True,
        embedding_mode: str = "idencoder",
    ):
        super().__init__()

        self.sequence_length = sequence_length
        self.pred_length = pred_length
        self.bin_size = bin_size
        self.latent_step = latent_step
        self.num_latents_per_step = num_latents_per_step
        self.dim = dim
        self.ref_dim = ref_dim
        self.use_multimodal = use_multimodal
        self.embedding_mode = embedding_mode

        num_pred_bins = round(pred_length / bin_size)
        self.num_pred_bins = num_pred_bins

        # ---- Embeddings ----
        self.id_encoder = IDEncoder(
            ref_dim=ref_dim,
            embedding_dim=dim,
            num_layers=3,
            dropout=0.1,
        )
        if embedding_mode == "random":
            # Fixed random projection (not learned) for ablation
            self.random_proj = nn.Linear(ref_dim, dim, bias=False)
            self.random_proj.weight.requires_grad_(False)
        elif embedding_mode == "mean":
            # Simple linear projection from ref_dim to dim
            self.mean_proj = nn.Linear(ref_dim, dim)
        self.token_type_emb = Embedding(4, dim, init_scale=emb_init_scale)
        self.latent_emb = Embedding(
            num_latents_per_step, dim, init_scale=emb_init_scale
        )
        self.bin_type_emb = Embedding(1, dim, init_scale=emb_init_scale)
        self.rotary_emb = RotaryTimeEmbedding(
            head_dim=dim_head,
            rotate_dim=dim_head // 2,
            t_min=t_min,
            t_max=t_max,
        )
        self.dropout = nn.Dropout(p=lin_dropout)

        # ---- Encoder (Perceiver-style, from POYO+) ----
        self.enc_atn = RotaryCrossAttention(
            dim=dim,
            heads=cross_heads,
            dropout=atn_dropout,
            dim_head=dim_head,
            rotate_value=True,
        )
        self.enc_ffn = nn.Sequential(
            nn.LayerNorm(dim), FeedForward(dim=dim, dropout=ffn_dropout)
        )

        # Encoder self-attention processing layers
        self.proc_layers = nn.ModuleList()
        for _ in range(depth):
            self.proc_layers.append(
                nn.ModuleList([
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
                ])
            )

        # Multimodal cross-attention layers (inserted every N encoder layers)
        if use_multimodal:
            self.multimodal_layers = nn.ModuleDict()
            for i in range(depth):
                if (i + 1) % multimodal_every == 0:
                    self.multimodal_layers[str(i)] = MultimodalEncoder(
                        dim=dim,
                        image_dim=image_dim,
                        behavior_dim=behavior_dim,
                        heads=cross_heads,
                        dim_head=dim_head,
                        dropout=atn_dropout,
                        ffn_dropout=ffn_dropout,
                        use_image=use_image,
                        use_behavior=use_behavior,
                    )

        # ---- Decoder (autoregressive) ----
        self.dec_layers = nn.ModuleList()
        for _ in range(dec_depth):
            self.dec_layers.append(
                nn.ModuleList([
                    # Cross-attention: bin tokens attend to encoder latents
                    RotaryCrossAttention(
                        dim=dim,
                        heads=cross_heads,
                        dropout=atn_dropout,
                        dim_head=dim_head,
                        rotate_value=False,
                    ),
                    # Causal self-attention among bin tokens
                    CausalRotarySelfAttention(
                        dim=dim,
                        heads=self_heads,
                        dropout=atn_dropout,
                        dim_head=dim_head,
                    ),
                    # Feed-forward
                    nn.Sequential(
                        nn.LayerNorm(dim),
                        FeedForward(dim=dim, dropout=ffn_dropout),
                    ),
                ])
            )

        # ---- Output Head ----
        self.output_head = PerNeuronHead(
            dim=dim, emb_dim=dim, hidden_dim=dim // 2
        )

    def forward(
        self,
        *,
        # Input sequence
        input_unit_index: TensorType["batch", "n_in", int],
        input_timestamps: TensorType["batch", "n_in", float],
        input_token_type: TensorType["batch", "n_in", int],
        input_mask: Optional[TensorType["batch", "n_in", bool]] = None,
        # Latent sequence
        latent_index: TensorType["batch", "n_latent", int],
        latent_timestamps: TensorType["batch", "n_latent", float],
        # Reference features for IDEncoder
        reference_features: TensorType["batch", "n_units", "ref_dim", float],
        unit_mask: Optional[TensorType["batch", "n_units", bool]] = None,
        # Prediction bin timestamps
        bin_timestamps: TensorType["batch", "n_bins", float],
        # Multimodal inputs (optional)
        image_embeddings: Optional[TensorType["batch", "n_img", "img_dim"]] = None,
        image_timestamps: Optional[TensorType["batch", "n_img"]] = None,
        image_mask: Optional[TensorType["batch", "n_img"]] = None,
        behavior_values: Optional[TensorType["batch", "n_beh", "beh_dim"]] = None,
        behavior_timestamps: Optional[TensorType["batch", "n_beh"]] = None,
        behavior_mask: Optional[TensorType["batch", "n_beh"]] = None,
    ) -> TensorType["batch", "n_bins", "n_units"]:
        """Forward pass.

        Returns:
            log_rates: (B, N_bins, N_units) predicted log firing rates
        """
        B = input_unit_index.shape[0]

        # ---- Compute unit embeddings ----
        # reference_features: (B, N_units, ref_dim) -> (B, N_units, dim)
        if self.embedding_mode == "idencoder":
            unit_embeddings = self.id_encoder.mlp(reference_features)
        elif self.embedding_mode == "random":
            with torch.no_grad():
                unit_embeddings = self.random_proj(reference_features)
        elif self.embedding_mode == "mean":
            unit_embeddings = self.mean_proj(reference_features)
        else:
            unit_embeddings = self.id_encoder.mlp(reference_features)

        # ---- Encode input spikes ----
        # Look up unit embeddings for each spike
        # input_unit_index: (B, N_in) with values in [0, N_units)
        input_emb = torch.gather(
            unit_embeddings,
            1,
            input_unit_index.unsqueeze(-1).expand(-1, -1, self.dim).clamp(min=0),
        )
        inputs = input_emb + self.token_type_emb(input_token_type)
        input_timestamp_emb = self.rotary_emb(input_timestamps)

        # ---- Latents ----
        latents = self.latent_emb(latent_index)
        latent_timestamp_emb = self.rotary_emb(latent_timestamps)

        # ---- Perceiver cross-attention: inputs -> latents ----
        latents = latents + self.enc_atn(
            latents, inputs,
            latent_timestamp_emb, input_timestamp_emb,
            input_mask,
        )
        latents = latents + self.enc_ffn(latents)

        # ---- Encoder self-attention processing ----
        for i, (self_attn, self_ff) in enumerate(self.proc_layers):
            latents = latents + self.dropout(
                self_attn(latents, latent_timestamp_emb)
            )
            latents = latents + self.dropout(self_ff(latents))

            # Inject multimodal information after designated layers
            if self.use_multimodal and str(i) in self.multimodal_layers:
                latents = self.multimodal_layers[str(i)](
                    latents, latent_timestamp_emb, self.rotary_emb,
                    image_embeddings=image_embeddings,
                    image_timestamps=image_timestamps,
                    image_mask=image_mask,
                    behavior_values=behavior_values,
                    behavior_timestamps=behavior_timestamps,
                    behavior_mask=behavior_mask,
                )

        # ---- Decoder: autoregressive bin prediction ----
        # Initialize bin queries
        bin_type_idx = torch.zeros(
            B, bin_timestamps.shape[1], dtype=torch.long, device=bin_timestamps.device
        )
        bin_queries = self.bin_type_emb(bin_type_idx)
        bin_timestamp_emb = self.rotary_emb(bin_timestamps)

        for cross_attn, causal_self_attn, ffn in self.dec_layers:
            # Cross-attend to encoder latents
            bin_queries = bin_queries + cross_attn(
                bin_queries, latents,
                bin_timestamp_emb, latent_timestamp_emb,
            )
            # Causal self-attention among bin tokens
            bin_queries = bin_queries + causal_self_attn(
                bin_queries, bin_timestamp_emb,
            )
            # Feed-forward
            bin_queries = bin_queries + ffn(bin_queries)

        # ---- Output: per-neuron log-rate prediction ----
        log_rates = self.output_head(bin_queries, unit_embeddings)

        # Apply unit mask if provided (mask out padded units)
        if unit_mask is not None:
            log_rates = log_rates.masked_fill(
                ~unit_mask.unsqueeze(1), 0.0
            )

        return log_rates

    def tokenize(self, data: Data) -> Dict:
        """Tokenize a temporaldata.Data object for NeuroHorizon.

        Prepares:
        - Input spike tokens from [0, sequence_length]
        - Target spike counts from [sequence_length, sequence_length + pred_length]
        - Reference features for IDEncoder
        """
        start = 0.0
        end = self.sequence_length
        pred_start = end
        pred_end = end + self.pred_length

        # ---- Input spike tokens ----
        unit_ids = data.units.id
        n_units = len(unit_ids)

        spike_unit_index = data.spikes.unit_index
        spike_timestamps = data.spikes.timestamps

        # Filter spikes to input window
        in_mask = (spike_timestamps >= start) & (spike_timestamps < end)
        input_spike_ts = spike_timestamps[in_mask]
        input_spike_idx = spike_unit_index[in_mask]

        # Create start/end tokens
        se_token_type, se_unit_idx, se_ts = create_start_end_unit_tokens(
            unit_ids, start, end
        )

        # Concatenate
        spike_token_type = np.concatenate(
            [se_token_type, np.zeros_like(input_spike_idx)]
        )
        all_spike_idx = np.concatenate([se_unit_idx, input_spike_idx])
        all_spike_ts = np.concatenate([se_ts, input_spike_ts])

        # ---- Latent tokens ----
        latent_index, latent_timestamps = create_linspace_latent_tokens(
            start, end, step=self.latent_step,
            num_latents_per_step=self.num_latents_per_step,
        )

        # ---- Target: binned spike counts in prediction window ----
        pred_mask = (spike_timestamps >= pred_start) & (spike_timestamps < pred_end)
        pred_spike_ts = spike_timestamps[pred_mask]
        pred_spike_idx = spike_unit_index[pred_mask]

        # Bin spikes
        bin_edges = np.arange(pred_start, pred_end + self.bin_size / 2, self.bin_size)
        n_bins = len(bin_edges) - 1
        spike_counts = np.zeros((n_bins, n_units), dtype=np.float32)

        if len(pred_spike_ts) > 0:
            bin_assignment = np.digitize(pred_spike_ts, bin_edges) - 1
            valid = (bin_assignment >= 0) & (bin_assignment < n_bins)
            np.add.at(
                spike_counts,
                (bin_assignment[valid], pred_spike_idx[valid]),
                1,
            )

        # Bin center timestamps
        bin_timestamps = (bin_edges[:-1] + bin_edges[1:]) / 2.0

        # ---- Reference features ----
        if hasattr(data.units, "reference_features"):
            reference_features = np.array(
                data.units.reference_features, dtype=np.float32
            )
        else:
            # Fallback: compute simple features on the fly
            reference_features = np.zeros((n_units, self.ref_dim), dtype=np.float32)
            for u in range(n_units):
                u_mask = input_spike_idx == u
                u_spikes = input_spike_ts[u_mask]
                if len(u_spikes) > 0:
                    reference_features[u, 0] = len(u_spikes) / (end - start)

        return {
            "model_inputs": {
                "input_unit_index": pad8(all_spike_idx.astype(np.int64)),
                "input_timestamps": pad8(all_spike_ts.astype(np.float64)),
                "input_token_type": pad8(spike_token_type.astype(np.int64)),
                "input_mask": track_mask8(all_spike_idx),
                "latent_index": latent_index.astype(np.int64),
                "latent_timestamps": latent_timestamps.astype(np.float64),
                "reference_features": reference_features,
                "bin_timestamps": bin_timestamps.astype(np.float64),
                **self._tokenize_multimodal(data, start, end),
            },
            "target_counts": spike_counts,
            "n_units": n_units,
            "session_id": data.session.id,
            "absolute_start": data.absolute_start,
        }

    def _tokenize_multimodal(self, data: Data, start: float, end: float) -> Dict:
        """Extract multimodal tokens from data if available.

        Returns dict of additional model_inputs keys for multimodal data.
        Empty dict if no multimodal data or use_multimodal is False.
        """
        if not self.use_multimodal:
            return {}

        result = {}

        # ---- Image embeddings (e.g., DINOv2 for Allen visual stimuli) ----
        if hasattr(data, "images") and hasattr(data.images, "embeddings"):
            img_ts = np.array(data.images.timestamps)
            img_emb = np.array(data.images.embeddings)

            # Filter to input window
            img_mask = (img_ts >= start) & (img_ts < end)
            if img_mask.any():
                result["image_embeddings"] = pad8(
                    img_emb[img_mask].astype(np.float32)
                )
                result["image_timestamps"] = pad8(
                    img_ts[img_mask].astype(np.float64)
                )
                result["image_mask"] = track_mask8(img_ts[img_mask])

        # ---- Behavior data (running speed, wheel velocity) ----
        if hasattr(data, "running") and hasattr(data.running, "timestamps"):
            beh_ts = np.array(data.running.timestamps)
            beh_vals = np.array(data.running.running_speed)

            # Filter to input window
            beh_mask = (beh_ts >= start) & (beh_ts < end)
            if beh_mask.any():
                beh_filtered = beh_vals[beh_mask]
                if beh_filtered.ndim == 1:
                    beh_filtered = beh_filtered[:, np.newaxis]
                result["behavior_values"] = pad8(
                    beh_filtered.astype(np.float32)
                )
                result["behavior_timestamps"] = pad8(
                    beh_ts[beh_mask].astype(np.float64)
                )
                result["behavior_mask"] = track_mask8(beh_ts[beh_mask])
        elif hasattr(data, "behavior") and hasattr(data.behavior, "timestamps"):
            beh_ts = np.array(data.behavior.timestamps)
            beh_vals = np.array(data.behavior.wheel_velocity)

            beh_mask = (beh_ts >= start) & (beh_ts < end)
            if beh_mask.any():
                beh_filtered = beh_vals[beh_mask]
                if beh_filtered.ndim == 1:
                    beh_filtered = beh_filtered[:, np.newaxis]
                result["behavior_values"] = pad8(
                    beh_filtered.astype(np.float32)
                )
                result["behavior_timestamps"] = pad8(
                    beh_ts[beh_mask].astype(np.float64)
                )
                result["behavior_mask"] = track_mask8(beh_ts[beh_mask])

        return result

    def compute_loss(
        self,
        log_rates: TensorType["batch", "n_bins", "n_units"],
        target_counts: TensorType["batch", "n_bins", "n_units"],
        unit_mask: Optional[TensorType["batch", "n_units", bool]] = None,
    ) -> torch.Tensor:
        """Compute Poisson NLL loss for spike count prediction.

        Args:
            log_rates: Predicted log firing rates (B, N_bins, N_units).
            target_counts: True spike counts (B, N_bins, N_units).
            unit_mask: Mask for valid units (B, N_units). True = valid.

        Returns:
            Scalar loss value.
        """
        # Poisson NLL: exp(log_rate) - target * log_rate
        loss = F.poisson_nll_loss(log_rates, target_counts, log_input=True, reduction="none")

        if unit_mask is not None:
            # Mask out padded units: (B, 1, N_units)
            mask = unit_mask.unsqueeze(1).expand_as(loss)
            loss = loss * mask.float()
            # Mean over valid entries
            return loss.sum() / mask.float().sum().clamp(min=1.0)
        else:
            return loss.mean()
