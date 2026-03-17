#!/usr/bin/env python3
"""Faithful Neuroformer bridge on top of the canonical 1.8.3 protocol.

This bridge keeps the upstream Neuroformer model core intact and only adds the
minimum compatibility layer needed to:

1. materialize canonical Perich-Miller windows from the shared protocol
2. convert raw spike events into the original ID/dt token streams
3. train with upstream Neuroformer id/dt cross-entropy losses
4. evaluate via autoregressive generation followed by 20 ms re-binning
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

REPO_ROOT = "/root/autodl-tmp/NeuroHorizon"
NF_ROOT = f"{REPO_ROOT}/neural-benchmark/benchmark_models/neuroformer"
NF_PKG = f"{NF_ROOT}/neuroformer"

sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, f"{REPO_ROOT}/neural_benchmark")
sys.path.insert(0, NF_ROOT)
sys.path.insert(0, NF_PKG)

if "skimage" not in sys.modules:
    skimage_mod = types.ModuleType("skimage")
    skimage_io = types.ModuleType("skimage.io")

    def _missing_imread(*args, **kwargs):
        raise RuntimeError("skimage is not available; frame loading is disabled in faithful benchmark bridge")

    skimage_io.imread = _missing_imread
    skimage_mod.io = skimage_io
    sys.modules["skimage"] = skimage_mod
    sys.modules["skimage.io"] = skimage_io

from neural_benchmark.repro_protocol import (
    BenchmarkProtocolSpec,
    build_continuous_windows,
    build_global_unit_index,
    build_trial_windows,
    compute_max_units,
    compute_raw_null_rates,
    evaluate_prediction_tensors,
    evaluate_trial_aligned_loader,
    load_binned_window,
    load_split_datasets,
    make_result_payload,
)
from torch_brain.utils.neurohorizon_metrics import build_null_rate_lookup
from utils import load_config, set_seed as nf_set_seed
from data_utils import Tokenizer
from model_neuroformer import Neuroformer


@dataclass(frozen=True)
class FaithfulNeuroformerConfig:
    hidden_size: int = 256
    n_heads: int = 8
    dropout: float = 0.2
    learning_rate: float = 1e-4
    weight_decay: float = 0.1
    grad_norm_clip: float = 1.0
    betas_0: float = 0.9
    betas_1: float = 0.95
    dt_resolution: float = 0.01
    prev_id_block_size: int = 512
    id_block_size: int = 256
    max_generate_steps: int = 256
    lr_decay: bool = True
    warmup_tokens: int = 50000


class FaithfulNeuroformerDataset(TorchDataset):
    """Canonical window dataset converted into Neuroformer token batches."""

    def __init__(
        self,
        tb_dataset,
        records,
        spec: BenchmarkProtocolSpec,
        global_unit_index: Mapping[Tuple[str, int], int],
        tokenizer: Tokenizer,
        prev_id_block_size: int,
        id_block_size: int,
        channel_capacity: int,
    ) -> None:
        self.tb_dataset = tb_dataset
        self.records = list(records)
        self.spec = spec
        self.global_unit_index = global_unit_index
        self.tokenizer = tokenizer
        self.prev_id_block_size = int(prev_id_block_size)
        self.id_block_size = int(id_block_size)
        self.channel_capacity = int(channel_capacity)

    def __len__(self) -> int:
        return len(self.records)

    def _build_full_sequence(
        self,
        event_ids: Sequence[int],
        event_dts: Sequence[float],
        block_size: int,
    ) -> Tuple[List[int], List[int], int]:
        max_events = max(block_size - 2, 0)
        event_ids = list(event_ids)[-max_events:]
        event_dts = list(event_dts)[-max_events:]

        encoded_ids = []
        if event_ids:
            encoded_ids = list(self.tokenizer.encode(event_ids, "ID"))
        encoded_dts = []
        if event_dts:
            encoded_dts = list(self.tokenizer.encode(event_dts, "dt"))

        id_full = [self.tokenizer.stoi["ID"]["SOS"]] + encoded_ids + [self.tokenizer.stoi["ID"]["EOS"]]
        pad_n = block_size - (len(id_full) - 1)
        id_full = id_full + [self.tokenizer.stoi["ID"]["PAD"]] * pad_n

        eos_dt = self.tokenizer.stoi["dt"].get("EOS", max(self.tokenizer.stoi["dt"].values()))
        pad_dt = self.tokenizer.stoi["dt"]["PAD"]
        dt_full = [0] + encoded_dts + [eos_dt]
        if len(dt_full) > block_size:
            dt_full = dt_full[-block_size:]
        dt_full = dt_full + [pad_dt] * pad_n
        return id_full, dt_full, pad_n

    def __getitem__(self, idx: int) -> Dict[str, object]:
        record = self.records[idx]
        sample = self.tb_dataset.get(record.recording_id, record.start_s, record.end_s)
        timestamps = np.asarray(sample.spikes.timestamps, dtype=np.float64)
        unit_index = np.asarray(sample.spikes.unit_index, dtype=np.int64)
        n_units = len(sample.units.id)
        local_to_global = np.asarray(
            [self.global_unit_index[(record.recording_id, local_idx)] for local_idx in range(n_units)],
            dtype=np.int64,
        )
        global_spike_ids = local_to_global[unit_index] if len(unit_index) else np.asarray([], dtype=np.int64)
        rel_timestamps = timestamps - float(sample.start)
        prev_mask = rel_timestamps < self.spec.obs_window_s
        curr_mask = rel_timestamps >= self.spec.obs_window_s

        prev_ids = global_spike_ids[prev_mask].tolist()
        prev_dts = rel_timestamps[prev_mask].tolist()
        curr_ids = global_spike_ids[curr_mask].tolist()
        curr_dts = (rel_timestamps[curr_mask] - self.spec.obs_window_s).tolist()

        prev_id_full, prev_dt_full, _ = self._build_full_sequence(
            prev_ids,
            prev_dts,
            self.prev_id_block_size,
        )
        curr_id_full, curr_dt_full, pad_n = self._build_full_sequence(
            curr_ids,
            curr_dts,
            self.id_block_size,
        )

        binned = load_binned_window(self.tb_dataset, record, self.spec, self.global_unit_index)
        spike_counts = np.zeros((self.spec.total_bins, self.channel_capacity), dtype=np.float32)
        binned_window = binned["spike_counts"][: self.spec.total_bins]
        spike_counts[: binned_window.shape[0], :n_units] = binned_window[:, :n_units]
        target_counts = spike_counts[self.spec.obs_bins : self.spec.obs_bins + self.spec.pred_bins].copy()
        unit_ids = np.zeros((self.channel_capacity,), dtype=np.int64)
        unit_mask = np.zeros((self.channel_capacity,), dtype=bool)
        unit_ids[:n_units] = binned["unit_ids"]
        unit_mask[:n_units] = True

        return {
            "x": {
                "id_prev": torch.tensor(prev_id_full[:-1], dtype=torch.long),
                "dt_prev": torch.tensor(prev_dt_full[:-1], dtype=torch.float32),
                "id": torch.tensor(curr_id_full[:-1], dtype=torch.long),
                "dt": torch.tensor(curr_dt_full[:-1], dtype=torch.float32),
                "pad": torch.tensor(pad_n, dtype=torch.long),
            },
            "y": {
                "id": torch.tensor(curr_id_full[1:], dtype=torch.long),
                "dt": torch.tensor(curr_dt_full[1:], dtype=torch.long),
            },
            "spike_counts": torch.from_numpy(spike_counts),
            "target_counts": torch.from_numpy(target_counts),
            "unit_ids": torch.from_numpy(unit_ids),
            "unit_mask": torch.from_numpy(unit_mask),
            "session_id": record.recording_id,
            "split": record.split,
            "target_id": torch.tensor(
                -1 if getattr(record, "target_id", None) is None else int(record.target_id),
                dtype=torch.long,
            ),
            "go_cue_time_s": torch.tensor(
                float("nan")
                if getattr(record, "go_cue_time_s", None) is None
                else float(record.go_cue_time_s),
                dtype=torch.float32,
            ),
        }


def collate_neuroformer_batch(batch: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    x = {key: torch.stack([item["x"][key] for item in batch]) for key in batch[0]["x"].keys()}
    y = {key: torch.stack([item["y"][key] for item in batch]) for key in batch[0]["y"].keys()}
    return {
        "x": x,
        "y": y,
        "spike_counts": torch.stack([item["spike_counts"] for item in batch]),
        "target_counts": torch.stack([item["target_counts"] for item in batch]),
        "unit_ids": torch.stack([item["unit_ids"] for item in batch]),
        "unit_mask": torch.stack([item["unit_mask"] for item in batch]),
        "session_id": [item["session_id"] for item in batch],
        "split": [item["split"] for item in batch],
        "target_id": torch.stack([item["target_id"] for item in batch]),
        "go_cue_time_s": torch.stack([item["go_cue_time_s"] for item in batch]),
    }


def move_batch_to_device(batch: Mapping[str, object], device: torch.device) -> Dict[str, object]:
    moved: Dict[str, object] = {}
    for key, value in batch.items():
        if isinstance(value, dict):
            moved[key] = {
                sub_key: sub_value.to(device) if isinstance(sub_value, torch.Tensor) else sub_value
                for sub_key, sub_value in value.items()
            }
        elif isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def maybe_limit_records(records, limit: Optional[int]):
    if limit is None or limit <= 0:
        return list(records)
    return list(records[:limit])


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    nf_set_seed(seed)


def build_window_loader(
    *,
    tb_dataset,
    records,
    spec: BenchmarkProtocolSpec,
    global_unit_index: Mapping[Tuple[str, int], int],
    tokenizer: Tokenizer,
    prev_id_block_size: int,
    id_block_size: int,
    channel_capacity: int,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    dataset = FaithfulNeuroformerDataset(
        tb_dataset=tb_dataset,
        records=records,
        spec=spec,
        global_unit_index=global_unit_index,
        tokenizer=tokenizer,
        prev_id_block_size=prev_id_block_size,
        id_block_size=id_block_size,
        channel_capacity=channel_capacity,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        collate_fn=collate_neuroformer_batch,
    )


def build_tokenizer(
    *,
    global_unit_index: Mapping[Tuple[str, int], int],
    spec: BenchmarkProtocolSpec,
    dt_resolution: float,
) -> Tokenizer:
    unit_tokens = sorted(set(global_unit_index.values()))
    max_window = max(spec.obs_window_s, spec.pred_window_s)
    dt_tokens = np.arange(0.0, max_window + dt_resolution, dt_resolution).tolist()
    return Tokenizer(
        {
            "ID": {"tokens": unit_tokens},
            "dt": {"tokens": dt_tokens, "resolution": dt_resolution},
        }
    )


def create_faithful_neuroformer_model(
    *,
    tokenizer: Tokenizer,
    spec: BenchmarkProtocolSpec,
    bridge_cfg: FaithfulNeuroformerConfig,
    config_path: str,
):
    config = load_config(config_path)
    config.layers.stimulus = 0
    config.window.frame = 0.0
    config.block_size.frame = 1
    config.modalities = None
    config.predict_behavior = False
    config.contrastive.contrastive = False
    config.window.curr = spec.pred_window_s
    config.window.prev = spec.obs_window_s
    config.resolution.dt = bridge_cfg.dt_resolution
    config.block_size.prev_id = bridge_cfg.prev_id_block_size
    config.block_size.id = bridge_cfg.id_block_size
    config.n_embd = bridge_cfg.hidden_size
    config.n_head = bridge_cfg.n_heads
    for key in ("attn", "embd", "pos", "resid", "temp", "b", "id", "im"):
        setattr(config.dropout, key, bridge_cfg.dropout)
    config.id_vocab_size = tokenizer.ID_vocab_size
    model = Neuroformer(config, tokenizer)

    def _feature_encoder_forward(self, neural, visual):
        for mod in self.neural_state_blocks:
            neural = mod(neural, neural, neural, self.mask)
        if self.frame_state_blocks is not None and visual is not None and hasattr(visual, "nelement") and visual.nelement() > 0:
            for mod in self.frame_state_blocks:
                visual = mod(visual, visual, visual)
        return {"id": neural, "frames": visual}

    model.feature_encoder.forward = MethodType(_feature_encoder_forward, model.feature_encoder)
    return model


def build_train_config(
    *,
    bridge_cfg: FaithfulNeuroformerConfig,
    train_dataset_len: int,
    epochs: int,
) -> SimpleNamespace:
    final_tokens = max(train_dataset_len * bridge_cfg.id_block_size * epochs, 1)
    warmup_tokens = min(int(bridge_cfg.warmup_tokens), final_tokens)
    return SimpleNamespace(
        learning_rate=bridge_cfg.learning_rate,
        betas=(bridge_cfg.betas_0, bridge_cfg.betas_1),
        weight_decay=bridge_cfg.weight_decay,
        decay_weights=True,
        grad_norm_clip=bridge_cfg.grad_norm_clip,
        lr_decay=bridge_cfg.lr_decay,
        warmup_tokens=warmup_tokens,
        final_tokens=final_tokens,
    )


def compute_total_loss(loss_dict: Mapping[str, torch.Tensor]) -> torch.Tensor:
    total = None
    for value in loss_dict.values():
        mean_value = value.mean()
        total = mean_value if total is None else total + mean_value
    if total is None:
        raise RuntimeError("Neuroformer loss dict was empty.")
    return total


def update_neuroformer_lr(
    optimizer: torch.optim.Optimizer,
    *,
    train_cfg: SimpleNamespace,
    tokens_seen: int,
) -> float:
    lr_mult = 1.0
    if tokens_seen < train_cfg.warmup_tokens:
        lr_mult = float(tokens_seen) / float(max(1, train_cfg.warmup_tokens))
    elif train_cfg.lr_decay:
        progress = float(tokens_seen - train_cfg.warmup_tokens) / float(
            max(1, train_cfg.final_tokens - train_cfg.warmup_tokens)
        )
        lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
    lr = train_cfg.learning_rate * lr_mult
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def decode_token(tokenizer: Tokenizer, token_type: str, token_id: int):
    return tokenizer.itos[token_type][int(token_id)]


def build_generation_inputs(
    sample_x: Mapping[str, torch.Tensor],
    current_ids: Sequence[int],
    current_dts: Sequence[int],
    *,
    pad_id_token: int,
    pad_dt_token: int,
    block_size: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    trimmed_ids = list(current_ids)[-block_size:]
    trimmed_dts = list(current_dts)[-block_size:]
    pad_n = max(block_size - len(trimmed_ids), 0)
    return {
        "id_prev": sample_x["id_prev"].unsqueeze(0),
        "dt_prev": sample_x["dt_prev"].unsqueeze(0),
        "id": torch.tensor(trimmed_ids + [pad_id_token] * pad_n, device=device, dtype=torch.long).unsqueeze(0),
        "dt": torch.tensor(trimmed_dts + [pad_dt_token] * pad_n, device=device, dtype=torch.float32).unsqueeze(0),
        "pad": torch.tensor([pad_n], device=device, dtype=torch.long),
    }


def select_next_tokens(
    model: Neuroformer,
    x: Mapping[str, torch.Tensor],
    valid_id_tokens: Sequence[int],
    *,
    eos_id_token: int,
    device: torch.device,
) -> Tuple[int, int]:
    preds, _, _ = model(x, None)
    step_index = int(x["id"].shape[1] - int(x["pad"][0].item()) - 1)
    id_logits = preds["id"][0, step_index].clone()
    dt_logits = preds["dt"][0, step_index].clone()
    invalid = torch.ones_like(id_logits, dtype=torch.bool, device=device)
    for token in valid_id_tokens:
        token = int(token)
        if 0 <= token < invalid.numel():
            invalid[token] = False
    invalid[eos_id_token] = False
    id_logits[invalid] = float("-inf")
    next_id = int(torch.argmax(id_logits).item())
    next_dt = int(torch.argmax(dt_logits).item())
    return next_id, next_dt


def collect_predicted_counts(
    predicted_events: Sequence[Tuple[int, int]],
    *,
    tokenizer: Tokenizer,
    valid_unit_ids: Sequence[int],
    spec: BenchmarkProtocolSpec,
    channel_capacity: int,
) -> torch.Tensor:
    local_unit_lookup = {int(unit_id): idx for idx, unit_id in enumerate(valid_unit_ids)}
    pred_counts = np.zeros((spec.pred_bins, channel_capacity), dtype=np.float32)
    for id_token, dt_token in predicted_events:
        decoded_id = decode_token(tokenizer, "ID", id_token)
        decoded_dt = decode_token(tokenizer, "dt", dt_token)
        if decoded_id in {"SOS", "EOS", "PAD"}:
            continue
        if decoded_dt in {"SOS", "EOS", "PAD"}:
            continue
        unit_id = int(decoded_id)
        time_s = float(decoded_dt)
        local_idx = local_unit_lookup.get(unit_id)
        if local_idx is None:
            continue
        if time_s < 0.0 or time_s >= spec.pred_window_s:
            continue
        bin_idx = min(int(math.floor(time_s / spec.bin_size_s)), spec.pred_bins - 1)
        pred_counts[bin_idx, local_idx] += 1.0
    pred_counts = np.clip(pred_counts, 1e-6, None)
    return torch.log(torch.from_numpy(pred_counts))


def decode_teacher_forced_logrates(
    preds: Mapping[str, torch.Tensor],
    tokenizer: Tokenizer,
    batch: Mapping[str, object],
    *,
    spec: BenchmarkProtocolSpec,
    channel_capacity: int,
) -> torch.Tensor:
    eos_id_token = tokenizer.stoi["ID"]["EOS"]
    pad_id_token = tokenizer.stoi["ID"]["PAD"]
    eos_dt_token = tokenizer.stoi["dt"].get("EOS", max(tokenizer.stoi["dt"].values()))
    pad_dt_token = tokenizer.stoi["dt"]["PAD"]
    outputs = []

    for i in range(batch["unit_ids"].shape[0]):
        valid_unit_ids = [int(x) for x in batch["unit_ids"][i][batch["unit_mask"][i]].detach().cpu().tolist() if int(x) >= 0]
        valid_id_tokens = [int(x) for x in tokenizer.encode(valid_unit_ids, "ID")] if valid_unit_ids else []
        predicted_events: List[Tuple[int, int]] = []
        for step in range(batch["y"]["id"].shape[1]):
            true_id = int(batch["y"]["id"][i, step].item())
            true_dt = int(batch["y"]["dt"][i, step].item())
            if true_id in {eos_id_token, pad_id_token}:
                break
            if true_dt in {eos_dt_token, pad_dt_token}:
                break
            id_logits = preds["id"][i, step].clone()
            dt_logits = preds["dt"][i, step].clone()
            invalid = torch.ones_like(id_logits, dtype=torch.bool)
            for token in valid_id_tokens:
                if 0 <= token < invalid.numel():
                    invalid[token] = False
            invalid[eos_id_token] = False
            id_logits[invalid] = float("-inf")
            next_id = int(torch.argmax(id_logits).item())
            next_dt = int(torch.argmax(dt_logits).item())
            if next_id != eos_id_token and next_dt != eos_dt_token:
                predicted_events.append((next_id, next_dt))
        outputs.append(
            collect_predicted_counts(
                predicted_events,
                tokenizer=tokenizer,
                valid_unit_ids=valid_unit_ids,
                spec=spec,
                channel_capacity=channel_capacity,
            )
        )
    return torch.stack(outputs, dim=0)


def generate_true_past_logrates(
    model: Neuroformer,
    tokenizer: Tokenizer,
    batch: Mapping[str, object],
    *,
    spec: BenchmarkProtocolSpec,
    channel_capacity: int,
) -> torch.Tensor:
    model_device = next(iter(model.parameters())).device
    batch = move_batch_to_device(batch, model_device)
    preds, _, _ = model(batch["x"], batch["y"])
    return decode_teacher_forced_logrates(
        preds,
        tokenizer,
        batch,
        spec=spec,
        channel_capacity=channel_capacity,
    ).to(model_device)


def generate_sample_counts(
    model: Neuroformer,
    tokenizer: Tokenizer,
    sample_x: Mapping[str, torch.Tensor],
    sample_y: Mapping[str, torch.Tensor],
    valid_unit_ids: torch.Tensor,
    *,
    spec: BenchmarkProtocolSpec,
    channel_capacity: int,
    max_generate_steps: int,
    device: torch.device,
    true_past: bool,
) -> torch.Tensor:
    model.eval()
    valid_unit_ids = [int(x) for x in valid_unit_ids.detach().cpu().tolist() if int(x) >= 0]
    valid_id_tokens = [int(x) for x in tokenizer.encode(valid_unit_ids, "ID")] if valid_unit_ids else []
    eos_id_token = tokenizer.stoi["ID"]["EOS"]
    sos_id_token = tokenizer.stoi["ID"]["SOS"]
    eos_dt_token = tokenizer.stoi["dt"].get("EOS", max(tokenizer.stoi["dt"].values()))
    pad_id_token = tokenizer.stoi["ID"]["PAD"]
    pad_dt_token = tokenizer.stoi["dt"]["PAD"]
    block_size = int(model.config.block_size.id)
    predicted_events: List[Tuple[int, int]] = []

    with torch.no_grad():
        if true_past:
            current_ids = [sos_id_token]
            current_dts = [0]
            true_ids = sample_y["id"].detach().cpu().tolist()
            true_dts = sample_y["dt"].detach().cpu().tolist()
            for step, (true_id, true_dt) in enumerate(zip(true_ids, true_dts), start=1):
                if step > max_generate_steps:
                    break
                if int(true_id) in {eos_id_token, pad_id_token}:
                    break
                if int(true_dt) in {eos_dt_token, pad_dt_token}:
                    break
                x = build_generation_inputs(
                    sample_x,
                    current_ids,
                    current_dts,
                    pad_id_token=pad_id_token,
                    pad_dt_token=pad_dt_token,
                    block_size=block_size,
                    device=device,
                )
                next_id, next_dt = select_next_tokens(
                    model,
                    x,
                    valid_id_tokens,
                    eos_id_token=eos_id_token,
                    device=device,
                )
                if next_id != eos_id_token and next_dt != eos_dt_token:
                    predicted_events.append((next_id, next_dt))
                current_ids.append(int(true_id))
                current_dts.append(int(true_dt))
        else:
            current_ids = [sos_id_token]
            current_dts = [0]
            for step in range(max_generate_steps):
                x = build_generation_inputs(
                    sample_x,
                    current_ids,
                    current_dts,
                    pad_id_token=pad_id_token,
                    pad_dt_token=pad_dt_token,
                    block_size=block_size,
                    device=device,
                )
                next_id, next_dt = select_next_tokens(
                    model,
                    x,
                    valid_id_tokens,
                    eos_id_token=eos_id_token,
                    device=device,
                )
                if next_id == eos_id_token or next_dt == eos_dt_token:
                    break
                predicted_events.append((next_id, next_dt))
                current_ids.append(next_id)
                current_dts.append(next_dt)
                if step + 1 >= max_generate_steps:
                    break

    return collect_predicted_counts(
        predicted_events,
        tokenizer=tokenizer,
        valid_unit_ids=valid_unit_ids,
        spec=spec,
        channel_capacity=channel_capacity,
    )


def generate_neuroformer_logrates(
    model: Neuroformer,
    tokenizer: Tokenizer,
    batch: Mapping[str, object],
    *,
    spec: BenchmarkProtocolSpec,
    channel_capacity: int,
    max_generate_steps: int,
    device: torch.device,
    true_past: bool,
) -> torch.Tensor:
    if true_past:
        return generate_true_past_logrates(
            model,
            tokenizer,
            batch,
            spec=spec,
            channel_capacity=channel_capacity,
        )
    outputs = []
    unit_ids = batch["unit_ids"]
    unit_mask = batch["unit_mask"]
    for i in range(unit_ids.shape[0]):
        valid_unit_ids = unit_ids[i][unit_mask[i]]
        sample_x = {key: value[i].to(device) for key, value in batch["x"].items()}
        sample_y = {key: value[i].to(device) for key, value in batch["y"].items()}
        outputs.append(
            generate_sample_counts(
                model,
                tokenizer,
                sample_x,
                sample_y,
                valid_unit_ids,
                spec=spec,
                channel_capacity=channel_capacity,
                max_generate_steps=max_generate_steps,
                device=device,
                true_past=true_past,
            )
        )
    return torch.stack(outputs, dim=0).to(device)


def evaluate_faithful_neuroformer_loader(
    model: Neuroformer,
    tokenizer: Tokenizer,
    dataloader: DataLoader,
    *,
    spec: BenchmarkProtocolSpec,
    channel_capacity: int,
    null_lookup: torch.Tensor,
    device: torch.device,
    max_generate_steps: int,
    true_past: bool,
) -> Dict[str, object]:
    all_logrates = []
    all_targets = []
    all_unit_ids = []
    all_unit_masks = []
    teacher_forced_loss = 0.0
    teacher_forced_batches = 0

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            batch = move_batch_to_device(batch, device)
            preds, _, loss_dict = model(batch["x"], batch["y"])
            teacher_forced_loss += float(compute_total_loss(loss_dict).detach().cpu().item())
            teacher_forced_batches += 1
            if true_past:
                logrates = decode_teacher_forced_logrates(
                    preds,
                    tokenizer,
                    batch,
                    spec=spec,
                    channel_capacity=channel_capacity,
                ).to(device)
            else:
                logrates = generate_neuroformer_logrates(
                    model,
                    tokenizer,
                    batch,
                    spec=spec,
                    channel_capacity=channel_capacity,
                    max_generate_steps=max_generate_steps,
                    device=device,
                    true_past=False,
                )
            all_logrates.append(logrates.cpu())
            all_targets.append(batch["target_counts"].cpu())
            all_unit_ids.append(batch["unit_ids"].cpu())
            all_unit_masks.append(batch["unit_mask"].cpu())

    metrics = evaluate_prediction_tensors(
        log_rates=torch.cat(all_logrates, dim=0),
        targets=torch.cat(all_targets, dim=0),
        unit_ids=torch.cat(all_unit_ids, dim=0),
        unit_mask=torch.cat(all_unit_masks, dim=0),
        null_lookup=null_lookup.cpu(),
    )
    metrics["n_samples"] = int(sum(x.shape[0] for x in all_logrates))
    metrics["teacher_forced_loss"] = float(teacher_forced_loss / max(teacher_forced_batches, 1))
    metrics["inference_mode"] = "true_past" if true_past else "rollout"
    return metrics


def build_trial_model_fn(
    model: Neuroformer,
    tokenizer: Tokenizer,
    *,
    spec: BenchmarkProtocolSpec,
    channel_capacity: int,
    max_generate_steps: int,
    device: torch.device,
    true_past: bool,
):
    def predict(batch: Mapping[str, object]) -> torch.Tensor:
        return generate_neuroformer_logrates(
            model,
            tokenizer,
            batch,
            spec=spec,
            channel_capacity=channel_capacity,
            max_generate_steps=max_generate_steps,
            device=device,
            true_past=true_past,
        )

    return predict


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def save_checkpoint(
    path: Path,
    *,
    epoch: int,
    model: Neuroformer,
    optimizer: torch.optim.Optimizer,
    valid_metrics: Mapping[str, object],
    train_loss: float,
    tokens_seen: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "valid_metrics": dict(valid_metrics),
            "train_loss": float(train_loss),
            "tokens_seen": int(tokens_seen),
        },
        path,
    )


def run_smoke(args: argparse.Namespace) -> Dict[str, object]:
    set_global_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = BenchmarkProtocolSpec(
        obs_window_s=args.obs_window,
        pred_window_s=args.pred_window,
        bin_size_s=args.bin_size,
    )
    bridge_cfg = FaithfulNeuroformerConfig(
        dt_resolution=args.dt_resolution,
        prev_id_block_size=args.prev_id_block_size,
        id_block_size=args.id_block_size,
        max_generate_steps=args.max_generate_steps,
    )
    datasets = load_split_datasets(dataset_config=args.dataset_config)
    global_unit_index = build_global_unit_index(datasets)
    channel_capacity = compute_max_units(datasets)
    tokenizer = build_tokenizer(
        global_unit_index=global_unit_index,
        spec=spec,
        dt_resolution=bridge_cfg.dt_resolution,
    )

    train_records = maybe_limit_records(build_continuous_windows(datasets["train"], "train", spec), 8)
    valid_records = maybe_limit_records(build_continuous_windows(datasets["valid"], "valid", spec), 4)
    trial_records = maybe_limit_records(build_trial_windows(datasets["valid"], "valid", spec), 4)

    train_loader = build_window_loader(
        tb_dataset=datasets["train"],
        records=train_records,
        spec=spec,
        global_unit_index=global_unit_index,
        tokenizer=tokenizer,
        prev_id_block_size=bridge_cfg.prev_id_block_size,
        id_block_size=bridge_cfg.id_block_size,
        channel_capacity=channel_capacity,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )
    valid_loader = build_window_loader(
        tb_dataset=datasets["valid"],
        records=valid_records,
        spec=spec,
        global_unit_index=global_unit_index,
        tokenizer=tokenizer,
        prev_id_block_size=bridge_cfg.prev_id_block_size,
        id_block_size=bridge_cfg.id_block_size,
        channel_capacity=channel_capacity,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )
    trial_loader = build_window_loader(
        tb_dataset=datasets["valid"],
        records=trial_records,
        spec=spec,
        global_unit_index=global_unit_index,
        tokenizer=tokenizer,
        prev_id_block_size=bridge_cfg.prev_id_block_size,
        id_block_size=bridge_cfg.id_block_size,
        channel_capacity=channel_capacity,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

    model = create_faithful_neuroformer_model(
        tokenizer=tokenizer,
        spec=spec,
        bridge_cfg=bridge_cfg,
        config_path=args.neuroformer_config,
    ).to(device)
    train_cfg = build_train_config(bridge_cfg=bridge_cfg, train_dataset_len=len(train_loader.dataset), epochs=1)
    optimizer = model.configure_optimizers(train_cfg)
    null_lookup = build_null_rate_lookup(
        compute_raw_null_rates(datasets["train"], global_unit_index, spec.bin_size_s),
        device=device,
    )

    batch = move_batch_to_device(next(iter(train_loader)), device)
    preds, _, loss_dict = model(batch["x"], batch["y"])
    total_loss = compute_total_loss(loss_dict)
    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), bridge_cfg.grad_norm_clip)
    optimizer.step()

    valid_metrics_rollout = evaluate_faithful_neuroformer_loader(
        model,
        tokenizer,
        valid_loader,
        spec=spec,
        channel_capacity=channel_capacity,
        null_lookup=null_lookup,
        device=device,
        max_generate_steps=bridge_cfg.max_generate_steps,
        true_past=False,
    )
    valid_metrics_true_past = evaluate_faithful_neuroformer_loader(
        model,
        tokenizer,
        valid_loader,
        spec=spec,
        channel_capacity=channel_capacity,
        null_lookup=null_lookup,
        device=device,
        max_generate_steps=bridge_cfg.max_generate_steps,
        true_past=True,
    )
    trial_metrics_rollout = evaluate_trial_aligned_loader(
        build_trial_model_fn(
            model,
            tokenizer,
            spec=spec,
            channel_capacity=channel_capacity,
            max_generate_steps=bridge_cfg.max_generate_steps,
            device=device,
            true_past=False,
        ),
        trial_loader,
        spec,
        null_lookup,
        device,
    )
    trial_metrics_true_past = evaluate_trial_aligned_loader(
        build_trial_model_fn(
            model,
            tokenizer,
            spec=spec,
            channel_capacity=channel_capacity,
            max_generate_steps=bridge_cfg.max_generate_steps,
            device=device,
            true_past=True,
        ),
        trial_loader,
        spec,
        null_lookup,
        device,
    )
    payload = {
        "mode": "smoke",
        "model": "neuroformer_faithful",
        "spec": asdict(spec),
        "bridge_config": asdict(bridge_cfg),
        "train_id_shape": list(batch["x"]["id"].shape),
        "train_prev_id_shape": list(batch["x"]["id_prev"].shape),
        "train_loss": float(total_loss.detach().cpu().item()),
        "valid_metrics_rollout": valid_metrics_rollout,
        "valid_metrics_true_past": valid_metrics_true_past,
        "trial_metrics_rollout": trial_metrics_rollout,
        "trial_metrics_true_past": trial_metrics_true_past,
    }
    output_dir = Path(args.output_dir or f"{REPO_ROOT}/results/logs/phase1_benchmark_faithful_neuroformer_smoke")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "smoke.json", payload)
    print(json.dumps(payload, indent=2))
    return payload


def run_train(args: argparse.Namespace) -> Dict[str, object]:
    set_global_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = BenchmarkProtocolSpec(
        obs_window_s=args.obs_window,
        pred_window_s=args.pred_window,
        bin_size_s=args.bin_size,
    )
    bridge_cfg = FaithfulNeuroformerConfig(
        hidden_size=args.hidden_size,
        n_heads=args.n_heads,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_norm_clip=args.grad_norm_clip,
        betas_0=args.beta1,
        betas_1=args.beta2,
        dt_resolution=args.dt_resolution,
        prev_id_block_size=args.prev_id_block_size,
        id_block_size=args.id_block_size,
        max_generate_steps=args.max_generate_steps,
        lr_decay=not args.no_lr_decay,
        warmup_tokens=args.warmup_tokens,
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(
            f"{REPO_ROOT}/results/logs/phase1_benchmark_repro_faithful_neuroformer_{int(round(args.pred_window * 1000))}ms"
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = load_split_datasets(dataset_config=args.dataset_config)
    global_unit_index = build_global_unit_index(datasets)
    channel_capacity = compute_max_units(datasets)
    tokenizer = build_tokenizer(
        global_unit_index=global_unit_index,
        spec=spec,
        dt_resolution=bridge_cfg.dt_resolution,
    )

    train_records = maybe_limit_records(
        build_continuous_windows(datasets["train"], "train", spec),
        args.max_train_windows,
    )
    valid_records = maybe_limit_records(
        build_continuous_windows(datasets["valid"], "valid", spec),
        args.max_valid_windows,
    )
    test_records = maybe_limit_records(
        build_continuous_windows(datasets["test"], "test", spec),
        args.max_test_windows,
    )
    test_trial_records = maybe_limit_records(
        build_trial_windows(datasets["test"], "test", spec),
        args.max_trial_windows,
    )
    if not train_records:
        raise ValueError("No canonical training windows were generated for faithful Neuroformer.")
    if not valid_records:
        raise ValueError("No canonical validation windows were generated for faithful Neuroformer.")
    if not test_records:
        raise ValueError("No canonical test windows were generated for faithful Neuroformer.")

    train_loader = build_window_loader(
        tb_dataset=datasets["train"],
        records=train_records,
        spec=spec,
        global_unit_index=global_unit_index,
        tokenizer=tokenizer,
        prev_id_block_size=bridge_cfg.prev_id_block_size,
        id_block_size=bridge_cfg.id_block_size,
        channel_capacity=channel_capacity,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    valid_loader = build_window_loader(
        tb_dataset=datasets["valid"],
        records=valid_records,
        spec=spec,
        global_unit_index=global_unit_index,
        tokenizer=tokenizer,
        prev_id_block_size=bridge_cfg.prev_id_block_size,
        id_block_size=bridge_cfg.id_block_size,
        channel_capacity=channel_capacity,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = build_window_loader(
        tb_dataset=datasets["test"],
        records=test_records,
        spec=spec,
        global_unit_index=global_unit_index,
        tokenizer=tokenizer,
        prev_id_block_size=bridge_cfg.prev_id_block_size,
        id_block_size=bridge_cfg.id_block_size,
        channel_capacity=channel_capacity,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_trial_loader = build_window_loader(
        tb_dataset=datasets["test"],
        records=test_trial_records,
        spec=spec,
        global_unit_index=global_unit_index,
        tokenizer=tokenizer,
        prev_id_block_size=bridge_cfg.prev_id_block_size,
        id_block_size=bridge_cfg.id_block_size,
        channel_capacity=channel_capacity,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = create_faithful_neuroformer_model(
        tokenizer=tokenizer,
        spec=spec,
        bridge_cfg=bridge_cfg,
        config_path=args.neuroformer_config,
    ).to(device)
    train_cfg = build_train_config(
        bridge_cfg=bridge_cfg,
        train_dataset_len=len(train_loader.dataset),
        epochs=args.epochs,
    )
    optimizer = model.configure_optimizers(train_cfg)
    null_lookup = build_null_rate_lookup(
        compute_raw_null_rates(datasets["train"], global_unit_index, spec.bin_size_s),
        device=device,
    )

    best_epoch = 0
    best_valid_metrics: Optional[Dict[str, object]] = None
    best_valid_fp_bps = float("-inf")
    final_epoch_metrics: Optional[Dict[str, object]] = None
    history: List[Dict[str, object]] = []
    best_checkpoint_path = output_dir / "best_model.pt"
    last_checkpoint_path = output_dir / "last_model.pt"
    tokens_seen = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_total = 0.0
        loss_steps = 0
        for batch in train_loader:
            batch = move_batch_to_device(batch, device)
            preds, _, loss_dict = model(batch["x"], batch["y"])
            total_loss = compute_total_loss(loss_dict)
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), bridge_cfg.grad_norm_clip)
            optimizer.step()
            tokens_seen += int((batch["y"]["id"] >= 0).sum().item())
            current_lr = update_neuroformer_lr(optimizer, train_cfg=train_cfg, tokens_seen=tokens_seen)
            loss_total += float(total_loss.detach().cpu().item())
            loss_steps += 1

        valid_metrics = evaluate_faithful_neuroformer_loader(
            model,
            tokenizer,
            valid_loader,
            spec=spec,
            channel_capacity=channel_capacity,
            null_lookup=null_lookup,
            device=device,
            max_generate_steps=bridge_cfg.max_generate_steps,
            true_past=False,
        )
        mean_train_loss = float(loss_total / max(loss_steps, 1))
        final_epoch_metrics = dict(valid_metrics)
        history.append(
            {
                "epoch": int(epoch),
                "train_loss": mean_train_loss,
                "valid_fp_bps": float(valid_metrics["fp_bps"]),
                "valid_r2": float(valid_metrics["r2"]),
                "lr": current_lr,
            }
        )
        save_checkpoint(
            last_checkpoint_path,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            valid_metrics=valid_metrics,
            train_loss=mean_train_loss,
            tokens_seen=tokens_seen,
        )
        if float(valid_metrics["fp_bps"]) > best_valid_fp_bps:
            best_valid_fp_bps = float(valid_metrics["fp_bps"])
            best_epoch = int(epoch)
            best_valid_metrics = dict(valid_metrics)
            save_checkpoint(
                best_checkpoint_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                valid_metrics=valid_metrics,
                train_loss=mean_train_loss,
                tokens_seen=tokens_seen,
            )
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_loss": mean_train_loss,
                    "valid_fp_bps": float(valid_metrics["fp_bps"]),
                    "valid_r2": float(valid_metrics["r2"]),
                }
            )
        )

    if best_valid_metrics is None:
        raise RuntimeError("Faithful Neuroformer training finished without any validation metrics.")

    best_state = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(best_state["model_state_dict"])
    test_metrics_rollout = evaluate_faithful_neuroformer_loader(
        model,
        tokenizer,
        test_loader,
        spec=spec,
        channel_capacity=channel_capacity,
        null_lookup=null_lookup,
        device=device,
        max_generate_steps=bridge_cfg.max_generate_steps,
        true_past=False,
    )
    test_metrics_true_past = evaluate_faithful_neuroformer_loader(
        model,
        tokenizer,
        test_loader,
        spec=spec,
        channel_capacity=channel_capacity,
        null_lookup=null_lookup,
        device=device,
        max_generate_steps=bridge_cfg.max_generate_steps,
        true_past=True,
    )
    trial_metrics_rollout = evaluate_trial_aligned_loader(
        build_trial_model_fn(
            model,
            tokenizer,
            spec=spec,
            channel_capacity=channel_capacity,
            max_generate_steps=bridge_cfg.max_generate_steps,
            device=device,
            true_past=False,
        ),
        test_trial_loader,
        spec,
        null_lookup,
        device,
    )
    trial_metrics_true_past = evaluate_trial_aligned_loader(
        build_trial_model_fn(
            model,
            tokenizer,
            spec=spec,
            channel_capacity=channel_capacity,
            max_generate_steps=bridge_cfg.max_generate_steps,
            device=device,
            true_past=True,
        ),
        test_trial_loader,
        spec,
        null_lookup,
        device,
    )

    payload = make_result_payload(
        model_name="faithful_neuroformer",
        protocol_name="canonical_protocol_v1",
        spec=spec,
        best_epoch=best_epoch,
        best_valid_metrics=best_valid_metrics,
        final_epoch_metrics=final_epoch_metrics or best_valid_metrics,
        test_metrics={
            "rollout": test_metrics_rollout,
            "true_past": test_metrics_true_past,
        },
        notes=[
            "Faithful bridge keeps upstream Neuroformer tokenization and id/dt loss path.",
            "Evaluation uses autoregressive generation and 20 ms count re-binning.",
            "Held-out test reports both rollout (true_past=False) and oracle-history (true_past=True) modes.",
            "Visual and behavior branches are disabled because Perich-Miller has no matching inputs.",
        ],
    )
    payload["selection_metric_mode"] = "rollout"
    payload["trial_aligned_test_metrics"] = {
        "rollout": dict(trial_metrics_rollout),
        "true_past": dict(trial_metrics_true_past),
    }
    payload["history"] = history
    payload["bridge_config"] = asdict(bridge_cfg)
    payload["model_fidelity_notes"] = [
        "Upstream Tokenizer and Neuroformer core retained.",
        "Compatibility layer maps Perich-Miller raw spike events into ID/dt token streams.",
        "Generation is session-constrained at decode time so predictions stay within the current recording vocabulary.",
        "Both rollout and true_past inference modes are evaluated on the same held-out windows.",
    ]
    write_json(output_dir / "results.json", payload)
    print(json.dumps(payload, indent=2))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Faithful Neuroformer bridge for benchmark 1.8.3")
    parser.add_argument("--mode", type=str, default="smoke", choices=["smoke", "train"])
    parser.add_argument("--obs-window", type=float, default=0.5)
    parser.add_argument("--pred-window", type=float, default=0.25)
    parser.add_argument("--bin-size", type=float, default=0.02)
    parser.add_argument("--dataset-config", type=str, default=None)
    parser.add_argument(
        "--neuroformer-config",
        type=str,
        default=f"{NF_ROOT}/configs/V1AL/mconf.yaml",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=25)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-windows", type=int, default=None)
    parser.add_argument("--max-valid-windows", type=int, default=None)
    parser.add_argument("--max-test-windows", type=int, default=None)
    parser.add_argument("--max-trial-windows", type=int, default=None)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-norm-clip", type=float, default=1.0)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--dt-resolution", type=float, default=0.01)
    parser.add_argument("--prev-id-block-size", type=int, default=512)
    parser.add_argument("--id-block-size", type=int, default=256)
    parser.add_argument("--max-generate-steps", type=int, default=256)
    parser.add_argument("--warmup-tokens", type=int, default=50000)
    parser.add_argument("--no-lr-decay", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "smoke":
        run_smoke(args)
    else:
        run_train(args)


if __name__ == "__main__":
    main()
