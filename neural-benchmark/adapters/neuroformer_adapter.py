"""
Neuroformer adapter for benchmark.
Simplified Neuroformer-style model: GPT-style causal transformer for spike prediction.
"""
import sys
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, '/root/autodl-tmp/NeuroHorizon')
from neural_benchmark.adapters.base_adapter import BenchmarkConfig


class NeuroformerWrapper(nn.Module):
    """Neuroformer-style GPT model for benchmark.
    
    Uses binned spike counts as input and causal transformer to predict future bins.
    Prediction window inputs are zeroed out to prevent data leakage.
    
    Architecture:
    - Input embedding: N-dim spike vector -> hidden_size
    - Causal Transformer decoder (GPT-style)
    - Per-bin output head
    """
    
    def __init__(self, n_units: int, config: BenchmarkConfig,
                 hidden_size: int = 256, n_heads: int = 8, n_layers: int = 6,
                 dropout: float = 0.35):
        super().__init__()
        self.n_units = n_units
        self.config = config
        self.hidden_size = hidden_size
        
        # Input embedding
        self.input_embed = nn.Sequential(
            nn.Linear(n_units, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        
        # Learned positional embedding (GPT-style)
        max_len = config.total_bins + 10
        self.pos_embed = nn.Embedding(max_len, hidden_size)
        self.drop = nn.Dropout(dropout)
        
        # GPT-style causal transformer (decoder-only)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=n_layers)
        
        # Output head
        self.ln_f = nn.LayerNorm(hidden_size)
        self.output_head = nn.Linear(hidden_size, n_units)
        
        # Causal mask
        causal = torch.triu(torch.ones(max_len, max_len), diagonal=1).bool()
        self.register_buffer('causal_mask', causal)
    
    def _encode(self, spike_counts, zero_future=True):
        """Shared encoding logic.
        
        Args:
            spike_counts: [B, T, N]
            zero_future: if True, zero out prediction window inputs
        
        Returns:
            log_rate: [B, T, N]
        """
        B, T, N = spike_counts.shape
        obs_bins = self.config.obs_bins
        
        # Zero out prediction window to prevent data leakage
        x_in = spike_counts.clone()
        if zero_future:
            x_in[:, obs_bins:, :] = 0.0
        
        # Embed input
        x = self.input_embed(x_in)
        positions = torch.arange(T, device=x.device)
        x = x + self.pos_embed(positions).unsqueeze(0)
        x = self.drop(x)
        
        # Causal transformer
        mask = self.causal_mask[:T, :T]
        x = self.transformer(x, mask=mask)
        
        # Output
        x = self.ln_f(x)
        log_rate = self.output_head(x)
        return log_rate
    
    def forward(self, spike_counts, obs_mask=None, pred_mask=None, unit_mask=None):
        """Forward pass: predict future from observation data only.
        
        Returns:
            log_rate: [B, T_pred, N] predicted log firing rates
        """
        obs_bins = self.config.obs_bins
        pred_bins = self.config.pred_bins
        
        log_rate = self._encode(spike_counts, zero_future=True)
        log_rate_pred = log_rate[:, obs_bins:obs_bins + pred_bins, :]
        return log_rate_pred
    
    def compute_loss(self, batch, loss_fn):
        """Compute causal prediction loss.
        
        Prediction window: zero-masked future inputs, loss on pred window.
        Observation window: shifted causal loss (output[t] predicts target[t+1]).
        """
        spike_counts = batch['spike_counts']  # [B, T, N]
        unit_mask = batch['unit_mask']        # [B, N]
        B, T, N = spike_counts.shape
        obs_bins = self.config.obs_bins
        pred_bins = self.config.pred_bins
        
        # Forward with future inputs zeroed
        log_rate_all = self._encode(spike_counts, zero_future=True)
        
        # Prediction window loss (main)
        log_rate_pred = log_rate_all[:, obs_bins:obs_bins + pred_bins, :]
        target_pred = spike_counts[:, obs_bins:obs_bins + pred_bins, :]
        mask_3d = unit_mask.unsqueeze(1).expand_as(log_rate_pred)
        loss_pred = loss_fn(log_rate_pred[mask_3d], target_pred[mask_3d])
        
        # Observation window shifted loss (auxiliary)
        # output[t] predicts target[t+1] for t in [0, obs_bins-2]
        if obs_bins > 1:
            log_rate_obs = log_rate_all[:, :obs_bins - 1, :]
            target_obs = spike_counts[:, 1:obs_bins, :]
            mask_obs = unit_mask.unsqueeze(1).expand(-1, obs_bins - 1, -1)
            loss_obs = loss_fn(log_rate_obs[mask_obs], target_obs[mask_obs])
            loss = 0.7 * loss_pred + 0.3 * loss_obs
        else:
            loss = loss_pred
        
        return loss


def create_neuroformer_model(n_units: int, config: BenchmarkConfig) -> NeuroformerWrapper:
    """Create Neuroformer benchmark model."""
    return NeuroformerWrapper(
        n_units=n_units,
        config=config,
        hidden_size=256,
        n_heads=8,
        n_layers=6,
        dropout=0.35,
    )
