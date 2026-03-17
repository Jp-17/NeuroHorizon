#!/usr/bin/env python3
"""Faithful NDT2 bridge on top of the canonical 1.8.3 protocol.

This module is intentionally scoped to the first reproducible milestone:

1. build canonical Perich-Miller windows with the shared protocol
2. convert binned counts into the original NDT2 flat token format
3. instantiate the upstream BrainBertInterface instead of a local substitute
4. verify that train loss and predict-time logrates work end-to-end

It does not yet claim full benchmark completion. The CLI currently exposes a
smoke-mode entrypoint so the bridge can be validated before full-scale training.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

sys.path.insert(0, "/root/autodl-tmp/NeuroHorizon")
sys.path.insert(0, "/root/autodl-tmp/NeuroHorizon/neural_benchmark")
sys.path.insert(
    0,
    "/root/autodl-tmp/NeuroHorizon/neural-benchmark/benchmark_models/ndt2",
)

from context_general_bci.config import (
    Architecture,
    DataKey,
    EmbedStrat,
    MetaKey,
    ModelConfig,
    ModelTask,
    Output,
    TaskConfig,
    TransformerConfig,
)
from context_general_bci.dataset import CHANNEL_KEY, LENGTH_KEY, ContextAttrs, DataAttrs
from context_general_bci.model import BrainBertInterface
from neural_benchmark.repro_protocol import (
    BenchmarkProtocolSpec,
    build_continuous_windows,
    build_global_unit_index,
    compute_raw_null_rates,
    evaluate_prediction_tensors,
    load_binned_window,
    load_split_datasets,
)
from torch_brain.utils.neurohorizon_metrics import build_null_rate_lookup


def build_max_units(datasets: Mapping[str, object]) -> int:
    """Local alias kept here to avoid cross-file import churn."""
    max_units = 0
    seen_recordings = set()
    for dataset in datasets.values():
        for recording_id in dataset.recording_dict.keys():
            if recording_id in seen_recordings:
                continue
            data = dataset.get_recording_data(recording_id)
            max_units = max(max_units, len(data.units.id))
            seen_recordings.add(recording_id)
    return max_units


@dataclass(frozen=True)
class FaithfulNDT2Config:
    hidden_size: int = 256
    n_layers: int = 6
    n_heads: int = 4
    dropout: float = 0.2
    neurons_per_token: int = 8
    mask_ratio: float = 0.25
    lr: float = 5e-4
    weight_decay: float = 1e-2
    pad_token: int = 20


def round_up_channels(n_units: int, neurons_per_token: int) -> int:
    return int(math.ceil(n_units / neurons_per_token) * neurons_per_token)


def tokenize_spike_counts(
    spike_counts: np.ndarray,
    *,
    channel_capacity: int,
    neurons_per_token: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Match NDT2's flat tokenization layout for spike counts."""
    n_time, n_units = spike_counts.shape
    padded = np.zeros((n_time, channel_capacity), dtype=np.float32)
    padded[:, :n_units] = spike_counts

    n_tokens = channel_capacity // neurons_per_token
    tokenized = padded.reshape(n_time, n_tokens, neurons_per_token)
    flat_spikes = torch.from_numpy(tokenized.reshape(n_time * n_tokens, neurons_per_token, 1))

    time_index = torch.arange(n_time, dtype=torch.long).unsqueeze(1).expand(n_time, n_tokens)
    position_index = torch.arange(n_tokens, dtype=torch.long).unsqueeze(0).expand(n_time, n_tokens)
    channel_counts = torch.full((n_time, n_tokens), neurons_per_token, dtype=torch.long)
    if n_units % neurons_per_token:
        channel_counts[:, -1] = n_units % neurons_per_token

    return (
        flat_spikes,
        time_index.reshape(-1),
        position_index.reshape(-1),
        channel_counts.reshape(-1),
    )


class FaithfulNDT2WindowDataset(TorchDataset):
    """Canonical window dataset converted into NDT2 flat-token batches."""

    def __init__(
        self,
        tb_dataset,
        records,
        spec: BenchmarkProtocolSpec,
        global_unit_index: Mapping[Tuple[str, int], int],
        session_to_idx: Mapping[str, int],
        channel_capacity: int,
        neurons_per_token: int,
    ) -> None:
        self.tb_dataset = tb_dataset
        self.records = list(records)
        self.spec = spec
        self.global_unit_index = global_unit_index
        self.session_to_idx = session_to_idx
        self.channel_capacity = channel_capacity
        self.neurons_per_token = neurons_per_token

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[object, object]:
        record = self.records[idx]
        window = load_binned_window(
            self.tb_dataset,
            record,
            self.spec,
            self.global_unit_index,
        )
        spike_counts = window["spike_counts"][: self.spec.total_bins]
        unit_ids = window["unit_ids"]
        n_units = int(spike_counts.shape[1])

        flat_spikes, flat_time, flat_position, flat_channel_counts = tokenize_spike_counts(
            spike_counts,
            channel_capacity=self.channel_capacity,
            neurons_per_token=self.neurons_per_token,
        )

        target_counts = np.zeros((self.spec.total_bins, self.channel_capacity), dtype=np.float32)
        target_counts[:, :n_units] = spike_counts
        unit_ids_padded = np.zeros((self.channel_capacity,), dtype=np.int64)
        unit_ids_padded[:n_units] = unit_ids
        unit_mask = np.zeros((self.channel_capacity,), dtype=bool)
        unit_mask[:n_units] = True

        return {
            DataKey.spikes: flat_spikes.to(torch.int64),
            DataKey.time: flat_time,
            DataKey.position: flat_position,
            MetaKey.session: torch.tensor(self.session_to_idx[window["recording_id"]], dtype=torch.long),
            LENGTH_KEY: torch.tensor(flat_spikes.shape[0], dtype=torch.long),
            CHANNEL_KEY: flat_channel_counts,
            "target_counts": torch.from_numpy(target_counts),
            "unit_ids": torch.from_numpy(unit_ids_padded),
            "unit_mask": torch.from_numpy(unit_mask),
            "n_units": torch.tensor(n_units, dtype=torch.long),
            "session_name": window["recording_id"],
        }


def collate_faithful_ndt2_batch(batch: Sequence[Mapping[object, object]]) -> Dict[object, object]:
    spikes = pad_sequence([item[DataKey.spikes] for item in batch], batch_first=True, padding_value=0)
    times = pad_sequence([item[DataKey.time] for item in batch], batch_first=True, padding_value=0)
    positions = pad_sequence([item[DataKey.position] for item in batch], batch_first=True, padding_value=0)
    channel_counts = pad_sequence([item[CHANNEL_KEY] for item in batch], batch_first=True, padding_value=0)
    lengths = torch.stack([item[LENGTH_KEY] for item in batch])
    sessions = torch.stack([item[MetaKey.session] for item in batch])

    return {
        DataKey.spikes: spikes,
        DataKey.time: times,
        DataKey.position: positions,
        MetaKey.session: sessions,
        LENGTH_KEY: lengths,
        CHANNEL_KEY: channel_counts,
        "target_counts": torch.stack([item["target_counts"] for item in batch]),
        "unit_ids": torch.stack([item["unit_ids"] for item in batch]),
        "unit_mask": torch.stack([item["unit_mask"] for item in batch]),
        "n_units": torch.stack([item["n_units"] for item in batch]),
        "session_name": [item["session_name"] for item in batch],
    }


def create_faithful_ndt2_model(
    *,
    session_ids: Sequence[str],
    spec: BenchmarkProtocolSpec,
    channel_capacity: int,
    config: FaithfulNDT2Config,
) -> BrainBertInterface:
    task_cfg = TaskConfig(
        tasks=[ModelTask.shuffle_infill],
        task_weights=[1.0],
        mask_ratio=config.mask_ratio,
        outputs=[Output.logrates, Output.spikes],
        metrics=[],
    )
    transformer_cfg = TransformerConfig(
        n_state=config.hidden_size,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        dropout=config.dropout,
        max_trial_length=spec.total_bins,
        transform_space=True,
        flat_encoder=True,
        embed_space=True,
        max_spatial_tokens=channel_capacity // config.neurons_per_token,
    )
    model_cfg = ModelConfig(
        hidden_size=config.hidden_size,
        arch=Architecture.ndt,
        transformer=transformer_cfg,
        task=task_cfg,
        encode_decode=True,
        decoder_layers=2,
        lr_init=config.lr,
        weight_decay=config.weight_decay,
        dropout=config.dropout,
        session_embed_strategy=EmbedStrat.token,
        subject_embed_strategy=EmbedStrat.none,
        task_embed_strategy=EmbedStrat.none,
        array_embed_strategy=EmbedStrat.none,
        readout_strategy=EmbedStrat.none,
        transform_space=True,
        spike_embed_style=EmbedStrat.project,
        neurons_per_token=config.neurons_per_token,
        causal=False,
    )
    data_attrs = DataAttrs(
        bin_size_ms=int(round(spec.bin_size_s * 1000)),
        spike_dim=1,
        max_channel_count=channel_capacity,
        context=ContextAttrs(session=list(session_ids)),
        max_arrays=1,
        pad_token=config.pad_token,
        serve_tokens=True,
        serve_tokens_flat=True,
        neurons_per_token=config.neurons_per_token,
    )
    return BrainBertInterface(model_cfg, data_attrs)


def clone_ndt2_core_batch(batch: Mapping[object, object]) -> Dict[object, object]:
    result: Dict[object, object] = {}
    for key in (DataKey.spikes, DataKey.time, DataKey.position, MetaKey.session, LENGTH_KEY, CHANNEL_KEY):
        value = batch[key]
        result[key] = value.clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value)
    return result


def move_batch_to_device(batch: Mapping[object, object], device: torch.device) -> Dict[object, object]:
    moved: Dict[object, object] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def assemble_valid_flat_predictions(
    flat_logrates: torch.Tensor,
    times: torch.Tensor,
    positions: torch.Tensor,
    *,
    length: int,
    total_bins: int,
    channel_capacity: int,
    neurons_per_token: int,
) -> torch.Tensor:
    n_positions = channel_capacity // neurons_per_token
    assembled = torch.zeros(
        (total_bins, n_positions, neurons_per_token),
        device=flat_logrates.device,
        dtype=flat_logrates.dtype,
    )
    valid_logrates = flat_logrates[:length]
    valid_times = times[:length]
    valid_positions = positions[:length]
    assembled[valid_times, valid_positions] = valid_logrates
    return assembled.reshape(total_bins, channel_capacity)


def faithful_ndt2_predict_logrates(
    model: BrainBertInterface,
    batch: Mapping[object, object],
    *,
    spec: BenchmarkProtocolSpec,
    channel_capacity: int,
    neurons_per_token: int,
) -> torch.Tensor:
    ndt2_batch = clone_ndt2_core_batch(batch)
    for task in model.cfg.task.tasks:
        model.task_pipelines[task.value].update_batch(ndt2_batch, eval_mode=True)
    features = model(ndt2_batch)
    task_output = model.task_pipelines[model.cfg.task.tasks[0].value](
        ndt2_batch,
        features,
        compute_metrics=False,
        eval_mode=True,
    )
    flat_logrates = task_output[Output.logrates]

    assembled = []
    lengths = batch[LENGTH_KEY].detach().cpu().tolist()
    for i, length in enumerate(lengths):
        assembled.append(
            assemble_valid_flat_predictions(
                flat_logrates=flat_logrates[i],
                times=batch[DataKey.time][i],
                positions=batch[DataKey.position][i],
                length=int(length),
                total_bins=spec.total_bins,
                channel_capacity=channel_capacity,
                neurons_per_token=neurons_per_token,
            )
        )
    full_window = torch.stack(assembled, dim=0)
    return full_window[:, spec.obs_bins : spec.obs_bins + spec.pred_bins, :]


def compute_train_loss(model: BrainBertInterface, batch: Mapping[object, object]) -> torch.Tensor:
    train_batch = clone_ndt2_core_batch(batch)
    output = model._step(train_batch, phase="train", eval_mode=False)
    return output["loss"]


def evaluate_faithful_ndt2_loader(
    model: BrainBertInterface,
    dataloader: DataLoader,
    *,
    spec: BenchmarkProtocolSpec,
    channel_capacity: int,
    neurons_per_token: int,
    null_lookup: torch.Tensor,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Dict[str, object]:
    all_logrates = []
    all_targets = []
    all_unit_ids = []
    all_unit_masks = []

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            batch = move_batch_to_device(batch, device)
            logrates = faithful_ndt2_predict_logrates(
                model,
                batch,
                spec=spec,
                channel_capacity=channel_capacity,
                neurons_per_token=neurons_per_token,
            )
            targets = batch["target_counts"][:, spec.obs_bins : spec.obs_bins + spec.pred_bins, :]
            all_logrates.append(logrates.cpu())
            all_targets.append(targets.cpu())
            all_unit_ids.append(batch["unit_ids"].cpu())
            all_unit_masks.append(batch["unit_mask"].cpu())

    metrics = evaluate_prediction_tensors(
        log_rates=torch.cat(all_logrates, dim=0),
        targets=torch.cat(all_targets, dim=0),
        unit_ids=torch.cat(all_unit_ids, dim=0),
        unit_mask=torch.cat(all_unit_masks, dim=0),
        null_lookup=null_lookup.cpu(),
    )
    metrics["n_batches"] = len(all_logrates)
    metrics["n_samples"] = int(sum(x.shape[0] for x in all_logrates))
    return metrics


def run_smoke(args: argparse.Namespace) -> Dict[str, object]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = BenchmarkProtocolSpec(
        obs_window_s=args.obs_window,
        pred_window_s=args.pred_window,
        bin_size_s=args.bin_size,
    )
    bridge_cfg = FaithfulNDT2Config(
        hidden_size=args.hidden_size,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
        neurons_per_token=args.neurons_per_token,
        mask_ratio=args.mask_ratio,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    datasets = load_split_datasets(dataset_config=args.dataset_config)
    global_unit_index = build_global_unit_index(datasets)
    raw_max_units = build_max_units(datasets)
    channel_capacity = round_up_channels(raw_max_units, bridge_cfg.neurons_per_token)
    session_ids = sorted(datasets["train"].recording_dict.keys())
    session_to_idx = {session_id: idx for idx, session_id in enumerate(session_ids)}

    train_records = build_continuous_windows(datasets["train"], "train", spec)[: args.train_windows]
    valid_records = build_continuous_windows(datasets["valid"], "valid", spec)[: args.valid_windows]

    train_loader = DataLoader(
        FaithfulNDT2WindowDataset(
            datasets["train"],
            train_records,
            spec,
            global_unit_index,
            session_to_idx,
            channel_capacity,
            bridge_cfg.neurons_per_token,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_faithful_ndt2_batch,
    )
    valid_loader = DataLoader(
        FaithfulNDT2WindowDataset(
            datasets["valid"],
            valid_records,
            spec,
            global_unit_index,
            session_to_idx,
            channel_capacity,
            bridge_cfg.neurons_per_token,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_faithful_ndt2_batch,
    )

    model = create_faithful_ndt2_model(
        session_ids=session_ids,
        spec=spec,
        channel_capacity=channel_capacity,
        config=bridge_cfg,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=bridge_cfg.lr,
        weight_decay=bridge_cfg.weight_decay,
    )

    train_batch = move_batch_to_device(next(iter(train_loader)), device)
    train_loss = compute_train_loss(model, train_batch)
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    null_rates = compute_raw_null_rates(
        datasets["train"],
        global_unit_index=global_unit_index,
        bin_size_s=spec.bin_size_s,
    )
    null_lookup = build_null_rate_lookup(null_rates, device=device)

    valid_batch = move_batch_to_device(next(iter(valid_loader)), device)
    pred_logrates = faithful_ndt2_predict_logrates(
        model,
        valid_batch,
        spec=spec,
        channel_capacity=channel_capacity,
        neurons_per_token=bridge_cfg.neurons_per_token,
    )
    smoke_metrics = evaluate_prediction_tensors(
        log_rates=pred_logrates.cpu(),
        targets=valid_batch["target_counts"][:, spec.obs_bins : spec.obs_bins + spec.pred_bins, :].cpu(),
        unit_ids=valid_batch["unit_ids"].cpu(),
        unit_mask=valid_batch["unit_mask"].cpu(),
        null_lookup=null_lookup.cpu(),
    )
    aggregate_metrics = evaluate_faithful_ndt2_loader(
        model,
        valid_loader,
        spec=spec,
        channel_capacity=channel_capacity,
        neurons_per_token=bridge_cfg.neurons_per_token,
        null_lookup=null_lookup,
        device=device,
        max_batches=args.eval_batches,
    )

    summary = {
        "mode": "smoke",
        "device": str(device),
        "spec": asdict(spec),
        "bridge_config": asdict(bridge_cfg),
        "n_sessions": len(session_ids),
        "raw_max_units": raw_max_units,
        "channel_capacity": channel_capacity,
        "train_windows": len(train_records),
        "valid_windows": len(valid_records),
        "train_batch_shape": list(train_batch[DataKey.spikes].shape),
        "valid_pred_shape": list(pred_logrates.shape),
        "train_loss_after_one_step": float(train_loss.detach().cpu().item()),
        "smoke_batch_metrics": smoke_metrics,
        "valid_loader_metrics": aggregate_metrics,
    }

    output_dir = Path(
        "/root/autodl-tmp/NeuroHorizon/results/logs/phase1_benchmark_faithful_ndt2_smoke"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "smoke.json"
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Faithful NDT2 bridge utilities")
    parser.add_argument("--mode", choices=["smoke"], default="smoke")
    parser.add_argument("--pred-window", type=float, default=0.25)
    parser.add_argument("--obs-window", type=float, default=0.5)
    parser.add_argument("--bin-size", type=float, default=0.02)
    parser.add_argument("--dataset-config", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--train-windows", type=int, default=8)
    parser.add_argument("--valid-windows", type=int, default=8)
    parser.add_argument("--eval-batches", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--neurons-per-token", type=int, default=8)
    parser.add_argument("--mask-ratio", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    args = parser.parse_args()

    if args.mode == "smoke":
        run_smoke(args)


if __name__ == "__main__":
    main()
