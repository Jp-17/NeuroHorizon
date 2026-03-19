#!/usr/bin/env python3
"""Basic functional checks for the 1.10 GRU latent-dynamics decoder."""

from __future__ import annotations

import torch

from torch_brain.models import NeuroHorizon


def main() -> None:
    torch.manual_seed(0)

    model = NeuroHorizon(
        sequence_length=0.75,
        pred_window=0.25,
        bin_size=0.02,
        latent_step=0.05,
        num_latents_per_step=32,
        dim=128,
        enc_depth=2,
        dec_depth=2,
        max_pred_bins=50,
        decoder_variant="latent_dynamics",
        feedback_method="none",
        latent_dynamics_num_queries=4,
    )
    model.eval()

    unit_ids = [f"unit_{i}" for i in range(6)]
    model.unit_emb.initialize_vocab(unit_ids)
    model.session_emb.initialize_vocab(["session_0"])
    global_unit_indices = torch.tensor(model.unit_emb.tokenizer(unit_ids), dtype=torch.long)

    batch_size = 2
    n_inputs = 24
    n_latent_steps = 10
    n_latents = n_latent_steps * model.num_latents_per_step
    n_bins = model.T_pred_bins

    latent_index = (
        torch.arange(model.num_latents_per_step)
        .repeat(n_latent_steps)
        .unsqueeze(0)
        .repeat(batch_size, 1)
    )
    latent_timestamps = (
        torch.arange(n_latent_steps, dtype=torch.float32)
        .repeat_interleave(model.num_latents_per_step)
        .mul(model.latent_step)
        .unsqueeze(0)
        .repeat(batch_size, 1)
    )

    inputs = {
        "input_unit_index": torch.randint(0, len(unit_ids), (batch_size, n_inputs)),
        "input_timestamps": torch.linspace(0.01, model.hist_window - 0.01, n_inputs).unsqueeze(0).repeat(batch_size, 1),
        "input_token_type": torch.zeros(batch_size, n_inputs, dtype=torch.long),
        "input_mask": torch.ones(batch_size, n_inputs, dtype=torch.bool),
        "latent_index": latent_index,
        "latent_timestamps": latent_timestamps,
        "bin_timestamps": torch.linspace(
            model.hist_window + model.bin_size / 2,
            model.sequence_length - model.bin_size / 2,
            n_bins,
        ).unsqueeze(0).repeat(batch_size, 1),
        "target_unit_index": global_unit_indices.unsqueeze(0).repeat(batch_size, 1),
        "target_unit_mask": torch.ones(batch_size, len(unit_ids), dtype=torch.bool),
    }

    assert model.requires_target_counts is False
    forward_out = model(**inputs)
    generate_out = model.generate(**inputs)

    assert forward_out.shape == (batch_size, n_bins, len(unit_ids))
    assert generate_out.shape == forward_out.shape

    max_diff = (forward_out - generate_out).abs().max().item()
    if max_diff > 1e-6:
        raise AssertionError(f"forward/generate mismatch too large: {max_diff}")

    print("latent_dynamics verification passed")
    print(f"output_shape={tuple(forward_out.shape)}")
    print(f"tf_vs_rollout_max_delta={max_diff:.6f}")


if __name__ == "__main__":
    main()
