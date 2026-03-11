"""
Base adapter for benchmark models.
Provides common data loading, spike binning, and evaluation interface.
"""
import sys
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset as TorchDataset, DataLoader
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# Add torch_brain to path
sys.path.insert(0, '/root/autodl-tmp/NeuroHorizon')

from torch_brain.data import Dataset as TBDataset
from torch_brain.utils.neurohorizon_metrics import (
    fp_bps, fp_bps_per_bin, r2_score, psth_r2, 
    compute_null_rates, build_null_rate_lookup
)
from torch_brain.nn.loss import PoissonNLLLoss


# Data paths
DATA_ROOT = '/root/autodl-tmp/NeuroHorizon/data/processed/'
DATASET_CONFIG = '/root/autodl-tmp/NeuroHorizon/examples/neurohorizon/configs/dataset/perich_miller_10sessions.yaml'


@dataclass
class BenchmarkConfig:
    """Shared configuration for benchmark experiments."""
    bin_size_s: float = 0.020         # 20ms bins
    obs_window_s: float = 0.500      # 500ms observation
    pred_window_s: float = 0.250     # 250ms prediction
    batch_size: int = 64
    epochs: int = 300
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42
    eval_epochs: int = 10
    device: str = 'cuda'
    
    @property
    def sequence_length_s(self):
        return self.obs_window_s + self.pred_window_s
    
    @property
    def obs_bins(self):
        return int(self.obs_window_s / self.bin_size_s)
    
    @property
    def pred_bins(self):
        return int(self.pred_window_s / self.bin_size_s)
    
    @property
    def total_bins(self):
        return self.obs_bins + self.pred_bins


def bin_spike_events(timestamps: np.ndarray, unit_index: np.ndarray,
                     n_units: int, start: float, end: float,
                     bin_size: float) -> np.ndarray:
    """Convert spike events to binned spike counts.
    
    Args:
        timestamps: spike times in seconds
        unit_index: neuron index for each spike
        n_units: total number of neurons
        start, end: time window in seconds
        bin_size: bin width in seconds
    
    Returns:
        spike_counts: [n_bins, n_units] float32 array
    """
    n_bins = int(round((end - start) / bin_size))
    spike_counts = np.zeros((n_bins, n_units), dtype=np.float32)
    
    if len(timestamps) == 0:
        return spike_counts
    
    # Vectorized binning
    bin_indices = np.floor((timestamps - start) / bin_size).astype(np.int64)
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    # Use np.add.at for efficient scatter-add
    np.add.at(spike_counts, (bin_indices, unit_index), 1)
    
    return spike_counts


def create_tb_dataset(split: str) -> TBDataset:
    """Create a torch_brain Dataset for the given split."""
    ds = TBDataset(
        root=DATA_ROOT,
        config=DATASET_CONFIG,
        split=split,
    )
    ds.disable_data_leakage_check()
    return ds


class BenchmarkDataset(TorchDataset):
    """Wraps torch_brain Dataset to provide binned spike data for benchmark models.
    
    Each sample returns:
        - spike_counts: [T, N] float tensor (T = obs + pred bins)
        - obs_mask: [T] bool tensor (True for observation bins)
        - pred_mask: [T] bool tensor (True for prediction bins)
        - unit_mask: [N] bool tensor (True for valid units)
        - unit_ids: [N] global unit indices (for null rate lookup)
        - n_units: actual number of units (before padding)
        - session_id: recording session identifier
    """
    
    def __init__(self, tb_dataset: TBDataset, split: str, config: BenchmarkConfig,
                 max_units: int = 300):
        self.tb_dataset = tb_dataset
        self.config = config
        self.max_units = max_units
        self.split = split
        
        # Get sampling intervals: {recording_id: Interval(start, end)}
        intervals_dict = tb_dataset.get_sampling_intervals()
        self.samples = []
        
        for rec_id, interval in intervals_dict.items():
            starts = np.array(interval.start)
            ends = np.array(interval.end)
            
            for s, e in zip(starts, ends):
                # Slide window across each interval
                window = config.sequence_length_s
                t = float(s)
                while t + window <= float(e):
                    self.samples.append((rec_id, t, t + window))
                    t += window * 0.5  # 50% overlap for more samples
        
        # Build unit ID mapping: (rec_id, local_idx) -> global_idx
        self.global_unit_ids = {}
        self._rec_n_units = {}
        for rec_id in intervals_dict.keys():
            data = tb_dataset.get_recording_data(rec_id)
            n_units = len(data.units.id)
            self._rec_n_units[rec_id] = n_units
            for i, uid in enumerate(data.units.id):
                key = (rec_id, i)
                if key not in self.global_unit_ids:
                    self.global_unit_ids[key] = len(self.global_unit_ids)
        
        print(f"  [{split}] {len(self.samples)} samples, "
              f"{len(self.global_unit_ids)} global units, "
              f"{len(intervals_dict)} recordings")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        rec_id, start, end = self.samples[idx]
        data = self.tb_dataset.get(rec_id, start, end)
        
        n_units = len(data.units.id)
        n_units_padded = min(n_units, self.max_units)
        
        # Get spike timestamps and unit indices as numpy arrays
        ts = np.array(data.spikes.timestamps, dtype=np.float64)
        ui = np.array(data.spikes.unit_index, dtype=np.int64)
        
        # Bin spikes for entire window
        # NOTE: torch_brain's slice() returns timestamps relative to the slice start (0 to window_length)
        # so we use data.start and data.end (which are 0 and window_length) for binning
        spike_counts = bin_spike_events(
            ts, ui, n_units,
            float(data.start), float(data.end),
            self.config.bin_size_s
        )
        
        # Truncate/pad units
        if n_units > self.max_units:
            spike_counts = spike_counts[:, :self.max_units]
            n_units_padded = self.max_units
        
        T = self.config.total_bins
        N = self.max_units
        
        # Pad to fixed size
        padded = np.zeros((T, N), dtype=np.float32)
        actual_T = min(spike_counts.shape[0], T)
        actual_N = min(spike_counts.shape[1], N)
        padded[:actual_T, :actual_N] = spike_counts[:actual_T, :actual_N]
        
        # Masks
        obs_bins = self.config.obs_bins
        obs_mask = np.zeros(T, dtype=bool)
        obs_mask[:obs_bins] = True
        pred_mask = np.zeros(T, dtype=bool)
        pred_mask[obs_bins:] = True
        
        # Unit mask (valid units)
        unit_mask = np.zeros(N, dtype=bool)
        unit_mask[:n_units_padded] = True
        
        # Global unit IDs
        unit_ids = np.zeros(N, dtype=np.int64)
        for i in range(n_units_padded):
            key = (rec_id, i)
            unit_ids[i] = self.global_unit_ids.get(key, 0)
        
        return {
            'spike_counts': torch.from_numpy(padded),           # [T, N]
            'obs_mask': torch.from_numpy(obs_mask),             # [T]
            'pred_mask': torch.from_numpy(pred_mask),           # [T]
            'unit_mask': torch.from_numpy(unit_mask),           # [N]
            'unit_ids': torch.from_numpy(unit_ids),             # [N]
            'n_units': n_units_padded,
            'session_id': rec_id,
        }


def compute_benchmark_null_rates(dataset: BenchmarkDataset) -> torch.Tensor:
    """Compute per-unit null rates from training data.
    
    Returns: [max_global_id + 1] tensor of log(mean_rate_per_bin).
    """
    max_id = max(dataset.global_unit_ids.values()) + 1
    total_counts = np.zeros(max_id, dtype=np.float64)
    total_bins = np.zeros(max_id, dtype=np.float64)
    
    for i in range(len(dataset)):
        sample = dataset[i]
        counts = sample['spike_counts'].numpy()  # [T, N]
        unit_ids = sample['unit_ids'].numpy()      # [N]
        T = counts.shape[0]
        
        for j in range(sample['n_units']):
            uid = unit_ids[j]
            total_counts[uid] += counts[:, j].sum()
            total_bins[uid] += T
    
    # Compute log(mean_rate)
    mean_rates = np.where(total_bins > 0, total_counts / total_bins, 1e-6)
    null_log_rates = np.log(np.maximum(mean_rates, 1e-6))
    
    return torch.from_numpy(null_log_rates.astype(np.float32))


def evaluate_model(model_fn, dataloader, null_rate_lookup, config, device='cuda'):
    """Unified evaluation for benchmark models.
    
    Args:
        model_fn: callable that takes batch dict and returns log_rate [B, T_pred, N]
        dataloader: validation DataLoader
        null_rate_lookup: [max_id+1] tensor of null log rates
        config: BenchmarkConfig
    
    Returns:
        dict with fp_bps, r2, per_bin_fp_bps, poisson_nll
    """
    all_log_rates = []
    all_targets = []
    all_null_rates = []
    all_masks = []
    
    null_rate_lookup = null_rate_lookup.to(device)
    
    for batch in dataloader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                 for k, v in batch.items()}
        
        with torch.no_grad():
            log_rate = model_fn(batch)  # [B, T_pred, N]
        
        # Extract prediction window targets
        T_pred = config.pred_bins
        target = batch['spike_counts']  # [B, T, N]
        target_pred = target[:, config.obs_bins:config.obs_bins + T_pred, :]
        
        # Get null rates for these units
        unit_ids = batch['unit_ids']  # [B, N]
        null_rates = null_rate_lookup[unit_ids]  # [B, N]
        
        unit_mask = batch['unit_mask']  # [B, N]
        
        all_log_rates.append(log_rate.cpu())
        all_targets.append(target_pred.cpu())
        all_null_rates.append(null_rates.cpu())
        all_masks.append(unit_mask.cpu())
    
    # Concatenate
    all_log_rates = torch.cat(all_log_rates, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_null_rates = torch.cat(all_null_rates, dim=0)
    all_masks = torch.cat(all_masks, dim=0)
    
    # Compute metrics
    bps = fp_bps(all_log_rates, all_targets, all_null_rates, all_masks)
    per_bin_bps = fp_bps_per_bin(all_log_rates, all_targets, all_null_rates, all_masks)
    pred_rates = torch.exp(all_log_rates)
    r2 = r2_score(pred_rates, all_targets, all_masks)
    
    # Poisson NLL
    loss_fn = PoissonNLLLoss()
    mask_3d = all_masks.unsqueeze(1).expand_as(all_log_rates)
    nll = loss_fn(all_log_rates[mask_3d], all_targets[mask_3d]).item()
    
    return {
        'fp_bps': bps.item() if isinstance(bps, torch.Tensor) else bps,
        'r2': r2.item() if isinstance(r2, torch.Tensor) else r2,
        'per_bin_fp_bps': per_bin_bps.tolist() if isinstance(per_bin_bps, torch.Tensor) else per_bin_bps,
        'poisson_nll': nll,
    }
