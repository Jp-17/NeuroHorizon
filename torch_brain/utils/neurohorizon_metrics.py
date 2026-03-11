"""NeuroHorizon evaluation metrics.

Provides:
- fp_bps: Forward Prediction Bits Per Spike
- fp_bps_per_bin: Per-bin fp-bps for decay analysis
- psth_r2: PSTH R-squared (peri-stimulus time histogram)
- r2_score: R-squared for spike count predictions
- firing_rate_correlation: Pearson correlation
- compute_null_rates: Training set mean firing rate computation
- build_null_rate_lookup: Tensor lookup for null rates
"""

import math
import logging
from collections import defaultdict
from typing import Dict, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def poisson_nll_elementwise(
    log_rate: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Per-element Poisson NLL: exp(log_rate) - target * log_rate."""
    log_rate = log_rate.clamp(-10, 10)
    return torch.exp(log_rate) - target * log_rate


def fp_bps(
    log_rate: torch.Tensor,
    target: torch.Tensor,
    null_log_rates: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Forward Prediction Bits Per Spike.

    fp-bps = (NLL_null - NLL_model) / (N_spikes * ln2)

    Higher is better. 0 = same as null model. Negative = worse than null.

    Args:
        log_rate: Model log rates [B, T, N]
        target: Spike counts [B, T, N]
        null_log_rates: Null model log rates [B, N] (per-neuron training mean)
        mask: Unit mask [B, N] (True for valid units)

    Returns:
        Scalar fp-bps
    """
    B, T, N = log_rate.shape

    # Expand null rates: [B, N] -> [B, 1, N] -> [B, T, N]
    null_expanded = null_log_rates.unsqueeze(1).expand(B, T, N)

    nll_model = poisson_nll_elementwise(log_rate, target)
    nll_null = poisson_nll_elementwise(null_expanded, target)

    if mask is not None:
        # mask [B, N] -> [B, T, N]
        mask_3d = mask.unsqueeze(1).expand(B, T, N)
        nll_model = nll_model[mask_3d]
        nll_null = nll_null[mask_3d]
        total_spikes = target[mask_3d].sum()
    else:
        nll_model = nll_model.reshape(-1)
        nll_null = nll_null.reshape(-1)
        total_spikes = target.sum()

    if total_spikes < 1:
        return torch.tensor(0.0, device=log_rate.device)

    return (nll_null.sum() - nll_model.sum()) / (total_spikes * math.log(2))


def fp_bps_per_bin(
    log_rate: torch.Tensor,
    target: torch.Tensor,
    null_log_rates: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Per-bin fp-bps for decay analysis.

    Args:
        log_rate: Model log rates [B, T, N]
        target: Spike counts [B, T, N]
        null_log_rates: Null model log rates [B, N]
        mask: Unit mask [B, N]

    Returns:
        [T] tensor of per-bin fp-bps values
    """
    B, T, N = log_rate.shape
    null_expanded = null_log_rates.unsqueeze(1).expand(B, T, N)

    nll_model = poisson_nll_elementwise(log_rate, target)
    nll_null = poisson_nll_elementwise(null_expanded, target)

    results = []
    for t in range(T):
        if mask is not None:
            nll_m_t = nll_model[:, t, :][mask]
            nll_n_t = nll_null[:, t, :][mask]
            spikes_t = target[:, t, :][mask].sum()
        else:
            nll_m_t = nll_model[:, t, :].reshape(-1)
            nll_n_t = nll_null[:, t, :].reshape(-1)
            spikes_t = target[:, t, :].sum()

        if spikes_t < 1:
            results.append(torch.tensor(0.0, device=log_rate.device))
        else:
            results.append(
                (nll_n_t.sum() - nll_m_t.sum()) / (spikes_t * math.log(2))
            )

    return torch.stack(results)


def r2_score(
    pred_rate: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """R-squared for spike count predictions.

    Args:
        pred_rate: Predicted rates [B, T, N]
        target: Spike counts [B, T, N]
        mask: Unit mask [B, N]

    Returns:
        Scalar R-squared
    """
    if mask is not None:
        mask_3d = mask.unsqueeze(1).expand_as(pred_rate)
        pred_flat = pred_rate[mask_3d]
        tgt_flat = target[mask_3d]
    else:
        pred_flat = pred_rate.reshape(-1)
        tgt_flat = target.reshape(-1)

    ss_res = ((pred_flat - tgt_flat) ** 2).sum()
    ss_tot = ((tgt_flat - tgt_flat.mean()) ** 2).sum()
    return 1 - ss_res / (ss_tot + 1e-8)


def firing_rate_correlation(
    pred_rate: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Pearson correlation between predicted and actual firing rates.

    Args:
        pred_rate: Predicted rates [B, T, N]
        target: Spike counts [B, T, N]
        mask: Unit mask [B, N]

    Returns:
        Scalar Pearson correlation
    """
    if mask is not None:
        mask_3d = mask.unsqueeze(1).expand_as(pred_rate)
        pred_flat = pred_rate[mask_3d]
        tgt_flat = target[mask_3d]
    else:
        pred_flat = pred_rate.reshape(-1)
        tgt_flat = target.reshape(-1)

    pred_centered = pred_flat - pred_flat.mean()
    tgt_centered = tgt_flat - tgt_flat.mean()
    cov = (pred_centered * tgt_centered).sum()
    std_pred = pred_centered.pow(2).sum().sqrt()
    std_tgt = tgt_centered.pow(2).sum().sqrt()
    return cov / (std_pred * std_tgt + 1e-8)


def compute_null_rates(dataset, model, bin_size: float) -> Dict[int, float]:
    """Compute per-neuron mean spike count per bin from training data.

    Iterates over the training dataset to compute mean spike counts per bin
    for each neuron, serving as the null model for fp-bps.

    Args:
        dataset: Training Dataset instance
        model: NeuroHorizon model (for unit_emb.tokenizer)
        bin_size: Bin size in seconds

    Returns:
        Dict mapping global unit index to log(mean_count_per_bin)
    """
    total_counts = defaultdict(float)
    total_bins = defaultdict(float)

    for recording_id in dataset.recording_dict.keys():
        data = dataset._get_data_object(recording_id)

        # Get prefixed unit IDs and their global indices
        unit_ids = dataset._get_unit_ids_with_prefix(data)
        global_indices = list(model.unit_emb.tokenizer(list(unit_ids)))

        # Get training intervals
        intervals = dataset.get_sampling_intervals().get(recording_id)
        if intervals is None:
            continue

        for start, end in zip(intervals.start, intervals.end):
            duration = float(end - start)
            n_bins = duration / bin_size
            if n_bins <= 0:
                continue

            # Load spike data for this interval
            sample = data.slice(float(start), float(end))

            if (
                hasattr(sample, "spikes")
                and hasattr(sample.spikes, "unit_index")
                and len(sample.spikes.timestamps) > 0
            ):
                spike_uid = np.array(sample.spikes.unit_index)
                for local_idx, global_idx in enumerate(global_indices):
                    count = int((spike_uid == local_idx).sum())
                    total_counts[global_idx] += count
                    total_bins[global_idx] += n_bins
            else:
                for global_idx in global_indices:
                    total_bins[global_idx] += n_bins

    null_rates = {}
    for gid in set(list(total_counts.keys()) + list(total_bins.keys())):
        if total_bins.get(gid, 0) > 0:
            mean_count = total_counts.get(gid, 0) / total_bins[gid]
            null_rates[gid] = math.log(max(mean_count, 1e-6))
        else:
            null_rates[gid] = -10.0

    logger.info(
        f"Null model computed: {len(null_rates)} neurons, "
        f"mean log-rate = {sum(null_rates.values()) / max(len(null_rates), 1):.3f}"
    )
    return null_rates


def build_null_rate_lookup(
    null_rates: Dict[int, float],
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Build a tensor lookup from null rates dict.

    Args:
        null_rates: Dict mapping global unit index to log(mean_rate)
        device: Target device

    Returns:
        Tensor of shape [max_id + 1] for direct indexing
    """
    if not null_rates:
        return torch.zeros(1, device=device)
    max_id = max(null_rates.keys())
    lookup = torch.full((max_id + 1,), -10.0, device=device)
    for uid, lr in null_rates.items():
        lookup[uid] = lr
    return lookup


def psth_r2(
    predicted_rates: Dict[int, torch.Tensor],
    true_rates: Dict[int, torch.Tensor],
    sigma_bins: int = 1,
) -> torch.Tensor:
    """PSTH R-squared across conditions.

    Args:
        predicted_rates: {target_id: [n_trials, T, N]} predicted firing rates
        true_rates: {target_id: [n_trials, T, N]} actual spike counts
        sigma_bins: Gaussian smoothing sigma (in bins). 0 = no smoothing.

    Returns:
        Scalar PSTH R-squared
    """
    all_psth_pred = []
    all_psth_true = []

    for tid in sorted(predicted_rates.keys()):
        pred = predicted_rates[tid]
        true = true_rates[tid]
        if pred.shape[0] < 1:
            continue

        # Average across trials -> PSTH: [T, N]
        psth_pred = pred.mean(dim=0)
        psth_true = true.mean(dim=0)

        # Optional Gaussian smoothing along time axis
        if sigma_bins > 0:
            psth_pred = _gaussian_smooth_1d(psth_pred, sigma_bins)
            psth_true = _gaussian_smooth_1d(psth_true, sigma_bins)

        all_psth_pred.append(psth_pred)
        all_psth_true.append(psth_true)

    if not all_psth_pred:
        return torch.tensor(0.0)

    # Stack all conditions and flatten
    pred_flat = torch.cat([p.reshape(-1) for p in all_psth_pred])
    true_flat = torch.cat([t.reshape(-1) for t in all_psth_true])

    ss_res = ((pred_flat - true_flat) ** 2).sum()
    ss_tot = ((true_flat - true_flat.mean()) ** 2).sum()
    return 1 - ss_res / (ss_tot + 1e-8)


def _gaussian_smooth_1d(x: torch.Tensor, sigma: int) -> torch.Tensor:
    """Apply Gaussian smoothing along dim=0 (time axis).

    Args:
        x: [T, N] tensor
        sigma: Sigma in bins

    Returns:
        Smoothed [T, N] tensor
    """
    if sigma <= 0:
        return x

    kernel_size = sigma * 6 + 1
    t = torch.arange(kernel_size, dtype=x.dtype, device=x.device) - kernel_size // 2
    kernel = torch.exp(-0.5 * (t / sigma) ** 2)
    kernel = kernel / kernel.sum()

    # x: [T, N] -> [N, 1, T] for conv1d
    x_t = x.T.unsqueeze(1)
    kernel = kernel.view(1, 1, -1)
    padding = kernel_size // 2
    smoothed = torch.nn.functional.conv1d(x_t, kernel, padding=padding)
    return smoothed.squeeze(1).T
