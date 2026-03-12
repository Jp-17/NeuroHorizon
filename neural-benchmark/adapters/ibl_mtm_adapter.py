"""
IBL-MtM (NDT1) adapter for benchmark.
Wraps IBL-MtM's NDT1 architecture with causal masking for forward prediction.
"""
import sys
import os
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, '/root/autodl-tmp/NeuroHorizon')
sys.path.insert(0, '/root/autodl-tmp/NeuroHorizon/neural_benchmark/benchmark_models/ibl-mtm/src')

from neural_benchmark.adapters.base_adapter import BenchmarkConfig


class IBLMtMWrapper(nn.Module):
    """IBL-MtM style NDT1 model for benchmark.
    
    Implements the core NDT1 architecture:
    - Linear embedding + positional encoding
    - Transformer encoder with causal attention
    - Linear readout
    
    Uses temporal causal masking + zero-masked future inputs:
    - Prediction window inputs are zeroed out to prevent data leakage
    - Causal mask prevents attending to future positions
    - Trained with prediction + auxiliary observation loss
    """
    
    def __init__(self, n_units: int, config: BenchmarkConfig,
                 hidden_size: int = 512, n_heads: int = 8, n_layers: int = 5,
                 dropout: float = 0.4, embed_mult: int = 2):
        super().__init__()
        self.n_units = n_units
        self.config = config
        self.hidden_size = hidden_size
        
        # NDT1-style embedding: Linear(N, N*mult) -> activation -> Linear(N*mult, H)
        self.embed_spikes = nn.Linear(n_units, n_units * embed_mult)
        self.embed_proj = nn.Linear(n_units * embed_mult, hidden_size)
        self.embed_act = nn.Softsign()
        self.embed_scale = hidden_size ** 0.5
        
        # Positional embedding
        max_len = config.total_bins + 10
        self.pos_embed = nn.Embedding(max_len, hidden_size)
        
        self.embed_dropout = nn.Dropout(dropout)
        
        # Transformer encoder with causal masking
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_heads,
            dim_feedforward=hidden_size * 2,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output projection
        self.out_norm = nn.LayerNorm(hidden_size)
        self.output_head = nn.Linear(hidden_size, n_units)
        
        # Causal mask (lower triangular)
        causal = torch.triu(torch.ones(max_len, max_len), diagonal=1).bool()
        self.register_buffer('causal_mask', causal)
    
    def _encode(self, spike_counts, zero_future=True):
        """Shared encoding logic.
        
        Args:
            spike_counts: [B, T, N] binned spike counts
            zero_future: if True, zero out prediction window inputs
        
        Returns:
            log_rate: [B, T, N] predicted log rates for all positions
        """
        B, T, N = spike_counts.shape
        obs_bins = self.config.obs_bins
        
        # Zero out prediction window to prevent data leakage
        x = spike_counts.clone()
        if zero_future:
            x[:, obs_bins:, :] = 0.0
        
        # Embedding
        x = self.embed_spikes(x)
        x = self.embed_act(x) * self.embed_scale
        x = self.embed_proj(x)
        
        # Add positional embedding
        positions = torch.arange(T, device=x.device)
        x = x + self.pos_embed(positions).unsqueeze(0)
        x = self.embed_dropout(x)
        
        # Causal mask
        mask = self.causal_mask[:T, :T]
        x = self.encoder(x, mask=mask)
        
        # Output projection
        x = self.out_norm(x)
        log_rate = self.output_head(x)
        return log_rate
    
    def forward(self, spike_counts, obs_mask=None, pred_mask=None, unit_mask=None):
        """Forward pass: predict future from observation data only.
        
        Returns:
            log_rate: [B, T_pred, N] predicted log firing rates for prediction window
        """
        obs_bins = self.config.obs_bins
        pred_bins = self.config.pred_bins
        
        log_rate = self._encode(spike_counts, zero_future=True)
        log_rate_pred = log_rate[:, obs_bins:obs_bins + pred_bins, :]
        return log_rate_pred
    
    def compute_loss(self, batch, loss_fn):
        """Compute training loss.
        
        Prediction window loss: zero-masked future, loss on pred window outputs.
        Auxiliary observation loss: shifted prediction on obs window
        (output[t] predicts target[t+1], using actual obs inputs with causal mask).
        """
        spike_counts = batch['spike_counts']  # [B, T, N]
        unit_mask = batch['unit_mask']        # [B, N]
        B, T, N = spike_counts.shape
        obs_bins = self.config.obs_bins
        pred_bins = self.config.pred_bins
        
        # Main loss: prediction window (future inputs zeroed)
        log_rate_all = self._encode(spike_counts, zero_future=True)
        log_rate_pred = log_rate_all[:, obs_bins:obs_bins + pred_bins, :]
        target_pred = spike_counts[:, obs_bins:obs_bins + pred_bins, :]
        mask_3d = unit_mask.unsqueeze(1).expand_as(log_rate_pred)
        loss_pred = loss_fn(log_rate_pred[mask_3d], target_pred[mask_3d])
        
        # Auxiliary loss: shifted prediction on observation window
        # Use actual obs data (no zeroing), with causal mask
        if obs_bins > 1:
            log_rate_obs = log_rate_all[:, :obs_bins - 1, :]   # output at 0..obs-2
            target_obs = spike_counts[:, 1:obs_bins, :]         # target at 1..obs-1
            mask_obs = unit_mask.unsqueeze(1).expand(-1, obs_bins - 1, -1)
            loss_obs = loss_fn(log_rate_obs[mask_obs], target_obs[mask_obs])
            loss = 0.7 * loss_pred + 0.3 * loss_obs
        else:
            loss = loss_pred
        
        return loss


def create_ibl_mtm_model(n_units: int, config: BenchmarkConfig) -> IBLMtMWrapper:
    """Create IBL-MtM benchmark model."""
    return IBLMtMWrapper(
        n_units=n_units,
        config=config,
        hidden_size=512,
        n_heads=8,
        n_layers=5,
        dropout=0.4,
        embed_mult=2,
    )
