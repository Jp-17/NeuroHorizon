"""NeuroHorizon Evaluation Metrics

Metrics for evaluating neural encoding (spike count prediction):
- Poisson log-likelihood
- Bits per spike
- Pearson correlation of firing rates
- R² of binned spike counts
"""

import numpy as np
import torch
import torch.nn.functional as F


def poisson_log_likelihood(log_rates, target_counts, unit_mask=None):
    """Compute average Poisson log-likelihood.

    Args:
        log_rates: (B, N_bins, N_units) predicted log firing rates
        target_counts: (B, N_bins, N_units) true spike counts
        unit_mask: (B, N_units) mask for valid units

    Returns:
        Scalar average log-likelihood
    """
    # Poisson LL: target * log_rate - exp(log_rate) - log(target!)
    # F.poisson_nll_loss computes exp(input) - target * input (the NLL part)
    nll = F.poisson_nll_loss(log_rates, target_counts, log_input=True, reduction="none")

    if unit_mask is not None:
        mask = unit_mask.unsqueeze(1).expand_as(nll)
        nll = nll * mask.float()
        return -(nll.sum() / mask.float().sum().clamp(min=1.0))
    return -nll.mean()


def bits_per_spike(log_rates, target_counts, unit_mask=None):
    """Compute bits per spike relative to a null (mean-rate) model.

    bits/spike = (LL_model - LL_null) / (n_spikes * log(2))

    Args:
        log_rates: (B, N_bins, N_units) predicted log firing rates
        target_counts: (B, N_bins, N_units) true spike counts
        unit_mask: (B, N_units) mask for valid units

    Returns:
        Scalar bits per spike value
    """
    mask_expanded = None
    if unit_mask is not None:
        mask_expanded = unit_mask.unsqueeze(1).expand_as(target_counts).float()

    # Model NLL
    model_nll = F.poisson_nll_loss(
        log_rates, target_counts, log_input=True, reduction="none"
    )

    # Null model: constant rate = mean count per bin per unit
    mean_count = target_counts.mean(dim=1, keepdim=True)  # (B, 1, N_units)
    null_log_rate = mean_count.clamp(min=1e-8).log()
    null_nll = F.poisson_nll_loss(
        null_log_rate.expand_as(target_counts),
        target_counts,
        log_input=True,
        reduction="none",
    )

    # Compute difference
    nll_diff = null_nll - model_nll  # positive means model is better

    if mask_expanded is not None:
        nll_diff = nll_diff * mask_expanded
        total_spikes = (target_counts * mask_expanded).sum()
    else:
        total_spikes = target_counts.sum()

    if total_spikes == 0:
        return torch.tensor(0.0, device=log_rates.device)

    return nll_diff.sum() / (total_spikes * np.log(2))


def firing_rate_correlation(log_rates, target_counts, unit_mask=None):
    """Compute Pearson correlation of predicted vs true firing rates.

    Averages predicted rates and true counts across time bins,
    then computes correlation across units.

    Args:
        log_rates: (B, N_bins, N_units) predicted log firing rates
        target_counts: (B, N_bins, N_units) true spike counts
        unit_mask: (B, N_units) mask for valid units

    Returns:
        Mean Pearson correlation across batch
    """
    pred_rates = torch.exp(log_rates)  # (B, N_bins, N_units)

    # Average across time bins to get per-unit rates
    pred_mean = pred_rates.mean(dim=1)  # (B, N_units)
    target_mean = target_counts.mean(dim=1)  # (B, N_units)

    correlations = []
    B = log_rates.shape[0]
    for b in range(B):
        if unit_mask is not None:
            mask = unit_mask[b]
            p = pred_mean[b][mask]
            t = target_mean[b][mask]
        else:
            p = pred_mean[b]
            t = target_mean[b]

        if len(p) < 2:
            continue

        # Pearson correlation
        p_centered = p - p.mean()
        t_centered = t - t.mean()
        num = (p_centered * t_centered).sum()
        den = (p_centered.norm() * t_centered.norm()).clamp(min=1e-8)
        correlations.append((num / den).item())

    if not correlations:
        return 0.0
    return np.mean(correlations)


def r2_binned_counts(log_rates, target_counts, unit_mask=None):
    """Compute R² of binned spike counts.

    Args:
        log_rates: (B, N_bins, N_units) predicted log firing rates
        target_counts: (B, N_bins, N_units) true spike counts
        unit_mask: (B, N_units) mask for valid units

    Returns:
        R² score (can be negative if model is worse than mean predictor)
    """
    pred_counts = torch.exp(log_rates)  # predicted rate ≈ expected count

    if unit_mask is not None:
        mask = unit_mask.unsqueeze(1).expand_as(target_counts).float()
        pred_flat = pred_counts[mask.bool()]
        target_flat = target_counts[mask.bool()]
    else:
        pred_flat = pred_counts.reshape(-1)
        target_flat = target_counts.reshape(-1)

    if len(target_flat) == 0:
        return 0.0

    ss_res = ((target_flat - pred_flat) ** 2).sum()
    ss_tot = ((target_flat - target_flat.mean()) ** 2).sum()

    if ss_tot == 0:
        return 0.0

    return (1.0 - ss_res / ss_tot).item()
