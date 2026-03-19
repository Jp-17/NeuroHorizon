#!/usr/bin/env python3
"""Functional verification for decoder scheduled sampling."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from torch_brain.models import NeuroHorizon  # noqa: E402


def assert_true(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)


def build_model(**overrides) -> NeuroHorizon:
    kwargs = dict(
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
        feedback_method="none",
        prediction_memory_k=4,
        prediction_memory_heads=4,
        prediction_memory_train_mix_prob=0.0,
        prediction_memory_input_dropout=0.0,
        prediction_memory_input_noise_std=0.0,
    )
    kwargs.update(overrides)
    model = NeuroHorizon(**kwargs)
    unit_ids = [f"unit_{i}" for i in range(6)]
    session_ids = ["session_a", "session_b"]
    model.unit_emb.initialize_vocab(unit_ids)
    model.session_emb.initialize_vocab(session_ids)
    return model


def build_dummy_batch(model: NeuroHorizon):
    batch_size = 2
    num_units = 6
    num_input_tokens = 20
    num_latents = 16
    pred_bins = model.T_pred_bins

    token_ids = torch.tensor(model.unit_emb.tokenizer([f"unit_{i}" for i in range(num_units)]))
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

    try:
        build_model(
            decoder_variant="query_aug",
            feedback_method="none",
            decoder_train_mode="scheduled_sampling",
        )
    except ValueError as exc:
        unsupported_message = str(exc)
    else:
        raise AssertionError("baseline_v2-like route should reject decoder scheduled sampling")

    parallel_model = build_model(decoder_train_mode="parallel_teacher_forced")
    scheduled_model = build_model(
        decoder_train_mode="scheduled_sampling",
        decoder_rollout_prob_mode="fixed",
        decoder_rollout_prob=0.0,
    )
    scheduled_model.load_state_dict(parallel_model.state_dict())

    batch = build_dummy_batch(parallel_model)
    parallel_model.train()
    scheduled_model.train()
    scheduled_model.set_decoder_rollout_prob(0.0)

    parallel_output = parallel_model(**batch)
    scheduled_output = scheduled_model(**batch)
    zero_prob_delta = float((parallel_output - scheduled_output).abs().max().detach())
    assert_true(
        zero_prob_delta < 1e-5,
        f"rollout_prob=0 should match parallel teacher forcing, got delta={zero_prob_delta}",
    )

    rollout_model = build_model(
        decoder_train_mode="scheduled_sampling",
        decoder_rollout_prob_mode="fixed",
        decoder_rollout_prob=1.0,
    )
    rollout_model.train()
    rollout_model.set_decoder_rollout_prob(1.0)
    rollout_output = rollout_model(**batch)

    shifted_batch = dict(batch)
    shifted_batch["target_counts"] = batch["target_counts"].clone()
    shifted_batch["target_counts"][:, :, :4] += 9.0
    shifted_output = rollout_model(**shifted_batch)
    target_independence_delta = float(
        (rollout_output - shifted_output).abs().max().detach()
    )
    assert_true(
        target_independence_delta < 1e-6,
        "rollout_prob=1.0 should stop reading GT counts in the train path",
    )

    generated = rollout_model.generate(
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
    generate_shape = tuple(generated.shape)
    assert_true(generate_shape == tuple(rollout_output.shape), "generate output shape mismatch")

    print("decoder_scheduled_sampling verification passed")
    print(f"  unsupported_route_error={unsupported_message}")
    print(f"  zero_prob_delta={zero_prob_delta:.6e}")
    print(f"  target_independence_delta={target_independence_delta:.6e}")
    print(f"  generate_shape={generate_shape}")


if __name__ == "__main__":
    main()
