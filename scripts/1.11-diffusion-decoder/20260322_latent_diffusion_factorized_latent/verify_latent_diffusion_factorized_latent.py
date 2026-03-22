#!/usr/bin/env python3
"""Minimal functional checks for the first Option 2A latent diffusion decoder."""

from __future__ import annotations

import numpy as np
import torch

from torch_brain.models import NeuroHorizon
from torch_brain.utils import create_linspace_latent_tokens


def main() -> None:
    torch.manual_seed(0)

    model = NeuroHorizon(
        sequence_length=0.75,
        pred_window=0.25,
        bin_size=0.02,
        latent_step=0.05,
        num_latents_per_step=4,
        dim=64,
        enc_depth=1,
        dec_depth=2,
        dim_head=32,
        cross_heads=2,
        self_heads=4,
        ffn_dropout=0.0,
        lin_dropout=0.0,
        atn_dropout=0.0,
        feedback_method="none",
        decoder_variant="latent_diffusion",
        flow_steps_eval=4,
        latent_ae_depth=1,
        latent_factor_units=6,
    )

    unit_ids = np.array([101, 102, 103], dtype=np.int64)
    model.unit_emb.initialize_vocab(unit_ids)
    model.session_emb.initialize_vocab(np.array([0], dtype=np.int64))

    target_unit_index = torch.tensor(
        [model.unit_emb.tokenizer(unit_ids)],
        dtype=torch.long,
    )
    target_unit_mask = torch.tensor([[True, True, True]], dtype=torch.bool)

    hist_end = model.hist_window
    latent_index, latent_timestamps = create_linspace_latent_tokens(
        0.0,
        hist_end,
        step=model.latent_step,
        num_latents_per_step=model.num_latents_per_step,
    )
    latent_index = torch.tensor(latent_index, dtype=torch.long).unsqueeze(0)
    latent_timestamps = torch.tensor(latent_timestamps, dtype=torch.float32).unsqueeze(0)

    input_unit_index = torch.tensor(
        [[target_unit_index[0, 0], target_unit_index[0, 1], target_unit_index[0, 2], target_unit_index[0, 0]]],
        dtype=torch.long,
    )
    input_timestamps = torch.tensor([[0.05, 0.12, 0.26, 0.41]], dtype=torch.float32)
    input_token_type = torch.zeros_like(input_unit_index)
    input_mask = torch.tensor([[True, True, True, True]], dtype=torch.bool)
    bin_timestamps = torch.linspace(
        hist_end + model.bin_size / 2,
        model.sequence_length - model.bin_size / 2,
        model.T_pred_bins,
        dtype=torch.float32,
    ).unsqueeze(0)
    target_counts = torch.rand(1, model.T_pred_bins, target_unit_index.shape[1]) * 2.0

    model.train()
    loss, aux = model.compute_training_loss(
        input_unit_index=input_unit_index,
        input_timestamps=input_timestamps,
        input_token_type=input_token_type,
        input_mask=input_mask,
        latent_index=latent_index,
        latent_timestamps=latent_timestamps,
        bin_timestamps=bin_timestamps,
        target_unit_index=target_unit_index,
        target_unit_mask=target_unit_mask,
        target_counts=target_counts,
    )
    assert torch.isfinite(loss), "training loss must be finite"
    assert {"ae_recon_loss", "diffusion_latent_loss"} <= set(aux), "missing aux metrics"

    model.eval()
    with torch.no_grad():
        generated = model.generate(
            input_unit_index=input_unit_index,
            input_timestamps=input_timestamps,
            input_token_type=input_token_type,
            input_mask=input_mask,
            latent_index=latent_index,
            latent_timestamps=latent_timestamps,
            bin_timestamps=bin_timestamps,
            target_unit_index=target_unit_index,
            target_unit_mask=target_unit_mask,
        )
    assert generated.shape == target_counts.shape, "generate() must return [B, T, N]"
    assert torch.isfinite(generated).all(), "generated log-rates must be finite"

    print("latent diffusion verify: ok")
    print(f"loss={loss.item():.6f}")
    print(f"ae_recon_loss={aux['ae_recon_loss'].item():.6f}")
    print(f"diffusion_latent_loss={aux['diffusion_latent_loss'].item():.6f}")


if __name__ == "__main__":
    main()
