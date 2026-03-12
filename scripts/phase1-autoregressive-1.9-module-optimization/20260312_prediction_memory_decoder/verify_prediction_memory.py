#!/usr/bin/env python3
"""Minimal functional verification for the prediction-memory decoder."""

import torch

from torch_brain.models import NeuroHorizon


def assert_true(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)


def build_dummy_batch(model: NeuroHorizon):
    batch_size = 2
    num_units = 6
    num_input_tokens = 20
    num_latents = 16
    pred_bins = model.T_pred_bins

    unit_ids = [f"unit_{i}" for i in range(num_units)]
    session_ids = ["session_a", "session_b"]
    model.unit_emb.initialize_vocab(unit_ids)
    model.session_emb.initialize_vocab(session_ids)
    token_ids = torch.tensor(model.unit_emb.tokenizer(unit_ids), dtype=torch.long)

    unit_index = token_ids.unsqueeze(0).expand(batch_size, -1)
    unit_mask = torch.tensor(
        [
            [True, True, True, True, False, False],
            [True, True, True, True, True, True],
        ],
        dtype=torch.bool,
    )

    sampled = torch.randint(0, num_units, (batch_size, num_input_tokens))
    input_unit_index = token_ids[sampled]
    input_timestamps = torch.linspace(0.01, model.hist_window - 0.01, num_input_tokens).unsqueeze(0).repeat(batch_size, 1)
    input_token_type = torch.zeros(batch_size, num_input_tokens, dtype=torch.long)
    input_mask = torch.ones(batch_size, num_input_tokens, dtype=torch.bool)

    latent_steps = torch.arange(num_latents, dtype=torch.long) % model.num_latents_per_step
    latent_index = latent_steps.unsqueeze(0).repeat(batch_size, 1)
    latent_timestamps = torch.linspace(0.01, model.hist_window - 0.01, num_latents).unsqueeze(0).repeat(batch_size, 1)

    bin_timestamps = torch.linspace(
        model.hist_window + model.bin_size / 2,
        model.sequence_length - model.bin_size / 2,
        pred_bins,
    ).unsqueeze(0).repeat(batch_size, 1)

    target_counts = torch.tensor(
        [
            [
                [1.0, 0.0, 2.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 2.0, 0.0, 0.0],
            ],
            [
                [0.0, 2.0, 1.0, 0.0, 3.0, 1.0],
                [1.0, 0.0, 0.0, 2.0, 1.0, 0.0],
                [0.0, 1.0, 3.0, 1.0, 0.0, 2.0],
                [2.0, 0.0, 1.0, 0.0, 2.0, 1.0],
            ],
        ],
        dtype=torch.float32,
    )
    assert_true(pred_bins == target_counts.shape[1], "Dummy target_counts must match T_pred_bins")

    return {
        "input_unit_index": input_unit_index,
        "input_timestamps": input_timestamps,
        "input_token_type": input_token_type,
        "input_mask": input_mask,
        "latent_index": latent_index,
        "latent_timestamps": latent_timestamps,
        "bin_timestamps": bin_timestamps,
        "target_unit_index": unit_index,
        "target_unit_mask": unit_mask,
        "target_counts": target_counts,
    }


def main():
    torch.manual_seed(0)

    model = NeuroHorizon(
        sequence_length=0.28,
        pred_window=0.08,
        bin_size=0.02,
        latent_step=0.05,
        num_latents_per_step=4,
        dim=64,
        enc_depth=1,
        dec_depth=1,
        dim_head=32,
        cross_heads=2,
        self_heads=4,
        max_pred_bins=8,
        decoder_variant="prediction_memory",
        prediction_memory_k=4,
        prediction_memory_heads=4,
    ).eval()

    batch = build_dummy_batch(model)
    unit_embs = model.unit_emb(batch["target_unit_index"])

    memory = model.prediction_memory_encoder(
        torch.log1p(batch["target_counts"][:, 0, :]),
        unit_embs,
        batch["target_unit_mask"],
    )
    assert_true(memory.shape == (2, 4, model.dim), f"Unexpected memory shape: {tuple(memory.shape)}")

    masked_changed = batch["target_counts"][:, 0, :].clone()
    masked_changed[0, 4:] = 99.0
    memory_masked = model.prediction_memory_encoder(
        torch.log1p(masked_changed),
        unit_embs,
        batch["target_unit_mask"],
    )
    assert_true(
        torch.allclose(memory[0], memory_masked[0], atol=1e-5),
        "Masked units should not affect memory pooling",
    )

    tf_output = model(**batch)
    rollout_output = model.generate(
        input_unit_index=batch["input_unit_index"],
        input_timestamps=batch["input_timestamps"],
        input_token_type=batch["input_token_type"],
        input_mask=batch["input_mask"],
        latent_index=batch["latent_index"],
        latent_timestamps=batch["latent_timestamps"],
        bin_timestamps=batch["bin_timestamps"],
        target_unit_index=batch["target_unit_index"],
        target_unit_mask=batch["target_unit_mask"],
    )
    assert_true(tf_output.shape == rollout_output.shape, "TF and rollout outputs must have the same shape")
    assert_true(
        (tf_output - rollout_output).abs().max().item() > 1e-5,
        "Prediction-memory decoder should make TF and rollout differ",
    )

    shifted_batch = dict(batch)
    shifted_batch["target_counts"] = batch["target_counts"].clone()
    shifted_batch["target_counts"][:, 0, :4] += 5.0
    shifted_output = model(**shifted_batch)

    first_bin_delta = (shifted_output[:, 0, :] - tf_output[:, 0, :]).abs().max().item()
    future_delta = (shifted_output[:, 1:, :] - tf_output[:, 1:, :]).abs().max().item()
    assert_true(first_bin_delta < 1e-6, f"shift-right failed: bin0 changed by {first_bin_delta}")
    assert_true(future_delta > 1e-5, "shift-right failed: future bins were unchanged")

    _, _, memory_mask = model._build_prediction_memory_train(
        batch["target_counts"],
        unit_embs,
        batch["target_unit_mask"],
        model.rotary_emb(batch["bin_timestamps"]),
    )
    expected_mask_shape = (
        batch["target_counts"].shape[0],
        model.T_pred_bins,
        model.T_pred_bins * model.prediction_memory_k,
    )
    assert_true(memory_mask.shape == expected_mask_shape, f"Unexpected mask shape: {tuple(memory_mask.shape)}")
    assert_true(memory_mask[0, 0, : model.prediction_memory_k].all().item(), "Bin 0 should see the zero-memory slot")
    assert_true(
        not memory_mask[0, 0, model.prediction_memory_k :].any().item(),
        "Bin 0 must not see future prediction-memory slots",
    )

    print("prediction_memory verification passed")
    print(
        f"  tf_vs_rollout_max_delta="
        f"{float((tf_output - rollout_output).abs().max().detach()):.6f}"
    )
    print(f"  first_bin_delta_after_shift_change={first_bin_delta:.6f}")
    print(f"  future_delta_after_shift_change={future_delta:.6f}")


if __name__ == "__main__":
    main()
