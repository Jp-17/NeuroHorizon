"""
NDT2 adapter for benchmark.
Wraps NDT2 (BrainBertInterface) to work with torch_brain data.
"""
import sys
import os
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, '/root/autodl-tmp/NeuroHorizon')
sys.path.insert(0, '/root/autodl-tmp/NeuroHorizon/neural_benchmark/benchmark_models/ndt2')

from context_general_bci.config import (
    ModelConfig, ModelTask, Metric, Output, DataKey, MetaKey,
    Architecture, EmbedStrat, TransformerConfig
)

from neural_benchmark.adapters.base_adapter import BenchmarkConfig


class NDT2Wrapper(nn.Module):
    """Simplified NDT2 model wrapper for benchmark evaluation.
    
    Instead of using the full BrainBertInterface (which has complex context handling),
    we build a minimal NDT2-style model: SpaceTimeTransformer encoder + linear readout.
    This preserves the core architecture while simplifying the data pipeline.
    """
    
    def __init__(self, n_units: int, config: BenchmarkConfig,
                 hidden_size: int = 256, n_heads: int = 4, n_layers: int = 6,
                 dropout: float = 0.2):
        super().__init__()
        self.n_units = n_units
        self.config = config
        self.hidden_size = hidden_size
        
        # Input embedding: spike counts [N] -> hidden_size
        self.input_embed = nn.Linear(n_units, hidden_size)
        
        # Positional encoding
        max_len = config.total_bins + 10
        pe = torch.zeros(max_len, hidden_size)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * 
                           (-np.log(10000.0) / hidden_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output head: predict spike rates
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_units),
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, spike_counts, obs_mask=None, pred_mask=None, unit_mask=None):
        """
        Forward pass with MAE-style masking for forward prediction.
        
        Args:
            spike_counts: [B, T, N] binned spike counts
            obs_mask: [B, T] observation window mask
            pred_mask: [B, T] prediction window mask
            unit_mask: [B, N] valid unit mask
        
        Returns:
            log_rate: [B, T_pred, N] predicted log firing rates
        """
        B, T, N = spike_counts.shape
        obs_bins = self.config.obs_bins
        pred_bins = self.config.pred_bins
        
        # Create input: keep observation bins, zero out prediction bins (MAE style)
        x = spike_counts.clone()
        x[:, obs_bins:, :] = 0.0  # Mask future bins
        
        # Embed
        x = self.input_embed(x)  # [B, T, H]
        x = x + self.pe[:T].unsqueeze(0)
        x = self.dropout(x)
        
        # No causal mask - NDT2 uses bidirectional encoder with masked inputs
        # The future information is removed by zeroing the future bins
        x = self.encoder(x)  # [B, T, H]
        
        # Predict spike rates for all timesteps
        log_rate = self.output_head(x)  # [B, T, N]
        
        # Return only prediction window
        log_rate_pred = log_rate[:, obs_bins:obs_bins + pred_bins, :]  # [B, T_pred, N]
        
        return log_rate_pred
    
    def compute_loss(self, batch, loss_fn):
        """Compute training loss."""
        spike_counts = batch['spike_counts']  # [B, T, N]
        unit_mask = batch['unit_mask']        # [B, N]
        
        log_rate_pred = self.forward(
            spike_counts,
            obs_mask=batch.get('obs_mask'),
            pred_mask=batch.get('pred_mask'),
            unit_mask=unit_mask,
        )  # [B, T_pred, N]
        
        # Target: prediction window spike counts
        obs_bins = self.config.obs_bins
        pred_bins = self.config.pred_bins
        target = spike_counts[:, obs_bins:obs_bins + pred_bins, :]  # [B, T_pred, N]
        
        # Apply unit mask
        mask_3d = unit_mask.unsqueeze(1).expand_as(log_rate_pred)
        loss = loss_fn(log_rate_pred[mask_3d], target[mask_3d])
        
        return loss


def create_ndt2_model(n_units: int, config: BenchmarkConfig) -> NDT2Wrapper:
    """Create NDT2 benchmark model."""
    return NDT2Wrapper(
        n_units=n_units,
        config=config,
        hidden_size=256,
        n_heads=4,
        n_layers=6,
        dropout=0.2,
    )
