"""Estimate Poisson noise floor for R² on spike count data.

If spike counts follow Poisson(lambda), then:
- E[spike_count] = lambda
- Var(spike_count) = lambda (Poisson property)
- Total variance = Var(rate) + E[rate]  (signal + noise)
- Best achievable R² = Var(rate) / (Var(rate) + E[rate])

We estimate Var(rate) from smoothed spike counts (trial-averaged PSTH).
"""

import sys
sys.path.insert(0, "/root/autodl-tmp/NeuroHorizon")

import torch
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from torch_brain.data import Dataset, collate
from torch_brain.data.sampler import RandomFixedWindowSampler
from torch_brain.models import NeuroHorizon

# Create a minimal model just to use tokenize
model = NeuroHorizon(
    sequence_length=0.75, pred_window=0.250, bin_size=0.020,
    latent_step=0.05, num_latents_per_step=32,
    dim=128, enc_depth=6, dec_depth=2,
    dim_head=64, cross_heads=2, self_heads=8,
    max_pred_bins=50,
)

yaml_path = "/root/autodl-tmp/NeuroHorizon/examples/neurohorizon/configs/dataset/perich_miller_10sessions.yaml"

dataset = Dataset(
    root="/root/autodl-tmp/NeuroHorizon/data/processed/",
    config=yaml_path,
    split="valid",
    transform=model.tokenize,
)
dataset.disable_data_leakage_check()

model.unit_emb.initialize_vocab(dataset.get_unit_ids())
model.session_emb.initialize_vocab(dataset.get_session_ids())

sampler = RandomFixedWindowSampler(
    sampling_intervals=dataset.get_sampling_intervals(),
    window_length=model.sequence_length,
    generator=torch.Generator().manual_seed(42),
)

from torch.utils.data import DataLoader
loader = DataLoader(
    dataset, sampler=sampler, collate_fn=collate,
    batch_size=64, num_workers=0, drop_last=False,
)

# Collect spike count statistics
all_counts = []
for i, batch in enumerate(loader):
    counts = batch["target_spike_counts"]  # [B, T, N]
    mask = batch["model_inputs"]["target_unit_mask"]  # [B, N]
    T = counts.shape[1]
    mask_expanded = mask.unsqueeze(1).expand(-1, T, -1)
    valid = counts[mask_expanded].numpy()
    all_counts.append(valid)
    if i >= 15:  # ~1000 samples
        break

all_counts = np.concatenate(all_counts)
mean_rate = all_counts.mean()
var_total = all_counts.var()
var_noise = mean_rate  # Poisson: Var = E[rate]
var_signal = max(var_total - var_noise, 0)

r2_ceiling = var_signal / (var_signal + var_noise) if var_total > 0 else 0

logger.info(f"=== Poisson Noise Floor Analysis ===")
logger.info(f"Total samples: {len(all_counts)}")
logger.info(f"Mean spike count (per 20ms bin): {mean_rate:.4f}")
logger.info(f"Variance (total):  {var_total:.4f}")
logger.info(f"Variance (noise, Poisson):  {var_noise:.4f}")
logger.info(f"Variance (signal): {var_signal:.4f}")
logger.info(f"Signal-to-noise ratio: {var_signal/var_noise:.4f}" if var_noise > 0 else "SNR: inf")
logger.info(f"Theoretical R² ceiling: {r2_ceiling:.4f}")
logger.info(f"Achieved R²: ~0.26")
logger.info(f"Fraction of ceiling: {0.26/r2_ceiling:.2%}" if r2_ceiling > 0 else "N/A")
