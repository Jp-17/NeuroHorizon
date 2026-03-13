#!/usr/bin/env python3
"""Minimal functional verification for prediction-memory alignment tuning."""

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
    input_timestamps = torch.linspace(
        0.01, model.hist_window - 0.01, num_input_tokens
    ).unsqueeze(0).repeat(batch_size, 1)
    input_token_type = torch.zeros(batch_size, num_input_tokens, dtype=torch.long)
    input_mask = torch.ones(batch_size, num_input_tokens, dtype=torch.bool)

    latent_steps = torch.arange(num_latents, dtype=torch.long) % model.num_latents_per_step
    latent_index = latent_steps.unsqueeze(0).repeat(batch_size, 1)
    latent_timestamps = torch.linspace(
        0.01, model.hist_window - 0.01, num_latents
    ).unsqueeze(0).repeat(batch_size, 1)

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

    tuned_model = NeuroHorizon(
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
        ffn_dropout=0.0,
        lin_dropout=0.0,
        atn_dropout=0.0,
        max_pred_bins=8,
        decoder_variant="local_prediction_memory",
        prediction_memory_k=4,
        prediction_memory_heads=4,
        prediction_memory_train_mix_prob=0.35,
        prediction_memory_input_dropout=0.05,
        prediction_memory_input_noise_std=0.03,
    )
    assert_true(
        abs(tuned_model.prediction_memory_train_mix_prob - 0.35) < 1e-8,
        "Tuned mix_prob should be 0.35",
    )
    assert_true(
        abs(tuned_model.prediction_memory_input_dropout - 0.05) < 1e-8,
        "Tuned input_dropout should be 0.05",
    )
    assert_true(
        abs(tuned_model.prediction_memory_input_noise_std - 0.03) < 1e-8,
        "Tuned input_noise_std should be 0.03",
    )

    mix_model = NeuroHorizon(
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
        ffn_dropout=0.0,
        lin_dropout=0.0,
        atn_dropout=0.0,
        max_pred_bins=8,
        decoder_variant="local_prediction_memory",
        prediction_memory_k=4,
        prediction_memory_heads=4,
        prediction_memory_train_mix_prob=1.0,
        prediction_memory_input_dropout=0.0,
        prediction_memory_input_noise_std=0.0,
    )
    mix_model.train()

    batch = build_dummy_batch(mix_model)
    tf_output = mix_model(**batch)
    rollout_output = mix_model.generate(
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

    shifted_batch = dict(batch)
    shifted_batch["target_counts"] = batch["target_counts"].clone()
    shifted_batch["target_counts"][:, :, :4] += 7.0
    shifted_output = mix_model(**shifted_batch)
    target_independence_delta = (shifted_output - tf_output).abs().max().item()
    assert_true(
        target_independence_delta < 1e-6,
        f"Mixed-memory training should stop reading GT counts when mix_prob=1.0, got delta={target_independence_delta}",
    )

    reg_model = NeuroHorizon(
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
        ffn_dropout=0.0,
        lin_dropout=0.0,
        atn_dropout=0.0,
        max_pred_bins=8,
        decoder_variant="local_prediction_memory",
        prediction_memory_k=4,
        prediction_memory_heads=4,
        prediction_memory_train_mix_prob=0.0,
        prediction_memory_input_dropout=0.5,
        prediction_memory_input_noise_std=0.1,
    )
    reg_model.unit_emb.initialize_vocab([f"unit_{i}" for i in range(6)])
    reg_model.session_emb.initialize_vocab(["session_a", "session_b"])
    unit_embs = reg_model.unit_emb(batch["target_unit_index"])
    counts = batch["target_counts"][:, 0, :]

    torch.manual_seed(123)
    reg_model.train()
    train_memory = reg_model._encode_counts_to_memory_tokens(
        counts,
        unit_embs,
        batch["target_unit_mask"],
    )

    torch.manual_seed(123)
    reg_model.eval()
    eval_memory = reg_model._encode_counts_to_memory_tokens(
        counts,
        unit_embs,
        batch["target_unit_mask"],
    )
    regularization_delta = (train_memory - eval_memory).abs().max().item()
    assert_true(
        regularization_delta > 1e-5,
        "Memory input noise/dropout should perturb train-time memory tokens",
    )

    print("prediction_memory_alignment_tuning verification passed")
    print("  tuned_mix_prob=0.35")
    print("  tuned_input_dropout=0.05")
    print("  tuned_input_noise_std=0.03")
    print(f"  target_independence_delta={target_independence_delta:.6f}")
    print(f"  train_eval_memory_delta={regularization_delta:.6f}")
    print(
        f"  tf_vs_rollout_max_delta="
        f"{float((tf_output - rollout_output).abs().max().detach()):.6f}"
    )


if __name__ == "__main__":
    main()
