#!/usr/bin/env python3
"""Faithful NDT2 bridge on top of the canonical 1.8.3 protocol.

This bridge keeps the upstream NDT2 model core intact and only adds the minimum
compatibility layer needed to:

1. materialize canonical Perich-Miller windows from the shared protocol
2. convert binned counts into the original NDT2 flat token format
3. train with upstream ``BrainBertInterface + ShuffleInfill``
4. evaluate with the unified continuous and trial-aligned protocol
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
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
    build_trial_windows,
    compute_raw_null_rates,
    evaluate_prediction_tensors,
    evaluate_trial_aligned_loader,
    load_binned_window,
    load_split_datasets,
    make_result_payload,
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
    mask_ratio: float = 0.5
    lr: float = 5e-4
    weight_decay: float = 1e-2
    lr_schedule: str = "cosine_warmup"
    lr_ramp_init_factor: float = 0.1
    lr_ramp_steps: int = 50
    lr_decay_steps: int = 1000
    lr_min: float = 1e-6
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
            "spike_counts": torch.from_numpy(target_counts),
            "target_counts": torch.from_numpy(target_counts),
            "unit_ids": torch.from_numpy(unit_ids_padded),
            "unit_mask": torch.from_numpy(unit_mask),
            "n_units": torch.tensor(n_units, dtype=torch.long),
            "session_name": window["recording_id"],
            "session_id": window["recording_id"],
            "split": window["split"],
            "target_id": torch.tensor(
                -1 if window["target_id"] is None else int(window["target_id"]),
                dtype=torch.long,
            ),
            "go_cue_time_s": torch.tensor(
                float("nan") if window["go_cue_time_s"] is None else float(window["go_cue_time_s"]),
                dtype=torch.float32,
            ),
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
        "spike_counts": torch.stack([item["spike_counts"] for item in batch]),
        "target_counts": torch.stack([item["target_counts"] for item in batch]),
        "unit_ids": torch.stack([item["unit_ids"] for item in batch]),
        "unit_mask": torch.stack([item["unit_mask"] for item in batch]),
        "n_units": torch.stack([item["n_units"] for item in batch]),
        "session_name": [item["session_name"] for item in batch],
        "session_id": [item["session_id"] for item in batch],
        "split": [item["split"] for item in batch],
        "target_id": torch.stack([item["target_id"] for item in batch]),
        "go_cue_time_s": torch.stack([item["go_cue_time_s"] for item in batch]),
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
        lr_schedule=config.lr_schedule,
        lr_ramp_init_factor=config.lr_ramp_init_factor,
        lr_ramp_steps=config.lr_ramp_steps,
        lr_decay_steps=config.lr_decay_steps,
        lr_min=config.lr_min,
        dropout=config.dropout,
        session_embed_strategy=EmbedStrat.token,
        subject_embed_strategy=EmbedStrat.none,
        task_embed_strategy=EmbedStrat.none,
        array_embed_strategy=EmbedStrat.none,
        readout_strategy=EmbedStrat.none,
        transform_space=True,
        spike_embed_style=EmbedStrat.project,
        neurons_per_token=config.neurons_per_token,
        # The upstream f8 architecture preset enables causal decoding for
        # next-bin prediction; keep that behavior in the faithful bridge.
        causal=True,
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
    model.train()
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


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def maybe_limit_records(records, limit: int):
    if limit is None or limit <= 0:
        return list(records)
    return list(records[:limit])


def collect_session_ids(datasets: Mapping[str, object]) -> Sequence[str]:
    return sorted(
        {
            recording_id
            for dataset in datasets.values()
            for recording_id in dataset.recording_dict.keys()
        }
    )


def build_window_loader(
    *,
    tb_dataset,
    records,
    spec: BenchmarkProtocolSpec,
    global_unit_index: Mapping[Tuple[str, int], int],
    session_to_idx: Mapping[str, int],
    channel_capacity: int,
    neurons_per_token: int,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        FaithfulNDT2WindowDataset(
            tb_dataset,
            records,
            spec,
            global_unit_index,
            session_to_idx,
            channel_capacity,
            neurons_per_token,
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        collate_fn=collate_faithful_ndt2_batch,
    )


def build_prediction_fn(
    model: BrainBertInterface,
    *,
    spec: BenchmarkProtocolSpec,
    channel_capacity: int,
    neurons_per_token: int,
):
    def predict(batch: Mapping[object, object]) -> torch.Tensor:
        return faithful_ndt2_predict_logrates(
            model,
            batch,
            spec=spec,
            channel_capacity=channel_capacity,
            neurons_per_token=neurons_per_token,
        )

    return predict


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def save_checkpoint(
    path: Path,
    *,
    epoch: int,
    model: BrainBertInterface,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[object],
    valid_metrics: Mapping[str, object],
    train_loss: float,
    n_optimizer_steps: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "valid_metrics": dict(valid_metrics),
        "train_loss": float(train_loss),
        "n_optimizer_steps": int(n_optimizer_steps),
    }
    if scheduler is not None and hasattr(scheduler, "state_dict"):
        payload["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(
        payload,
        path,
    )


def resolve_optimizer_bundle(
    model: BrainBertInterface,
) -> Tuple[torch.optim.Optimizer, Optional[object]]:
    bundle = model.configure_optimizers()
    if isinstance(bundle, dict):
        return bundle["optimizer"], bundle.get("lr_scheduler")
    if isinstance(bundle, (list, tuple)):
        if not bundle:
            raise ValueError("configure_optimizers() returned an empty sequence.")
        optimizer = bundle[0]
        scheduler = bundle[1] if len(bundle) > 1 else None
        return optimizer, scheduler
    return bundle, None


def get_current_lr(optimizer: torch.optim.Optimizer) -> float:
    if not optimizer.param_groups:
        return float("nan")
    return float(optimizer.param_groups[0]["lr"])


def run_train(args: argparse.Namespace) -> Dict[str, object]:
    set_global_seed(args.seed)
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
        lr_schedule=args.lr_schedule,
        lr_ramp_init_factor=args.lr_ramp_init_factor,
        lr_ramp_steps=args.lr_ramp_steps,
        lr_decay_steps=args.lr_decay_steps,
        lr_min=args.lr_min,
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(
            "/root/autodl-tmp/NeuroHorizon/results/logs/"
            f"phase1_benchmark_repro_faithful_ndt2_{int(round(args.pred_window * 1000))}ms"
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = load_split_datasets(dataset_config=args.dataset_config)
    global_unit_index = build_global_unit_index(datasets)
    raw_max_units = build_max_units(datasets)
    channel_capacity = round_up_channels(raw_max_units, bridge_cfg.neurons_per_token)
    session_ids = collect_session_ids(datasets)
    session_to_idx = {session_id: idx for idx, session_id in enumerate(session_ids)}

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
        raise ValueError("No canonical training windows were generated for faithful NDT2.")
    if not valid_records:
        raise ValueError("No canonical validation windows were generated for faithful NDT2.")
    if not test_records:
        raise ValueError("No canonical test windows were generated for faithful NDT2.")

    train_loader = build_window_loader(
        tb_dataset=datasets["train"],
        records=train_records,
        spec=spec,
        global_unit_index=global_unit_index,
        session_to_idx=session_to_idx,
        channel_capacity=channel_capacity,
        neurons_per_token=bridge_cfg.neurons_per_token,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    valid_loader = build_window_loader(
        tb_dataset=datasets["valid"],
        records=valid_records,
        spec=spec,
        global_unit_index=global_unit_index,
        session_to_idx=session_to_idx,
        channel_capacity=channel_capacity,
        neurons_per_token=bridge_cfg.neurons_per_token,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = build_window_loader(
        tb_dataset=datasets["test"],
        records=test_records,
        spec=spec,
        global_unit_index=global_unit_index,
        session_to_idx=session_to_idx,
        channel_capacity=channel_capacity,
        neurons_per_token=bridge_cfg.neurons_per_token,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_trial_loader = build_window_loader(
        tb_dataset=datasets["test"],
        records=test_trial_records,
        spec=spec,
        global_unit_index=global_unit_index,
        session_to_idx=session_to_idx,
        channel_capacity=channel_capacity,
        neurons_per_token=bridge_cfg.neurons_per_token,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = create_faithful_ndt2_model(
        session_ids=session_ids,
        spec=spec,
        channel_capacity=channel_capacity,
        config=bridge_cfg,
    ).to(device)
    optimizer, scheduler = resolve_optimizer_bundle(model)
    null_rates = compute_raw_null_rates(
        datasets["train"],
        global_unit_index=global_unit_index,
        bin_size_s=spec.bin_size_s,
    )
    null_lookup = build_null_rate_lookup(null_rates, device=device)

    best_epoch = 0
    best_valid_metrics: Optional[Dict[str, object]] = None
    best_valid_fp_bps = float("-inf")
    final_valid_metrics: Optional[Dict[str, object]] = None
    history = []
    best_checkpoint_path = output_dir / "best_model.pt"
    last_checkpoint_path = output_dir / "last_model.pt"
    accumulate_batches = max(int(args.accumulate_batches), 1)
    total_optimizer_steps = 0

    for epoch in range(1, args.epochs + 1):
        epoch_losses = []
        optimizer.zero_grad(set_to_none=True)
        for batch_idx, batch in enumerate(train_loader, start=1):
            batch = move_batch_to_device(batch, device)
            train_loss = compute_train_loss(model, batch)
            (train_loss / accumulate_batches).backward()
            should_step = (
                batch_idx % accumulate_batches == 0
                or batch_idx == len(train_loader)
            )
            if should_step:
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                total_optimizer_steps += 1
            epoch_losses.append(float(train_loss.detach().cpu().item()))

        if scheduler is not None:
            scheduler.step()

        mean_train_loss = float(sum(epoch_losses) / max(len(epoch_losses), 1))
        epoch_record: Dict[str, object] = {
            "epoch": int(epoch),
            "mean_train_loss": mean_train_loss,
            "n_train_batches": len(epoch_losses),
            "n_optimizer_steps": int(total_optimizer_steps),
            "lr": get_current_lr(optimizer),
        }

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            valid_metrics = evaluate_faithful_ndt2_loader(
                model,
                valid_loader,
                spec=spec,
                channel_capacity=channel_capacity,
                neurons_per_token=bridge_cfg.neurons_per_token,
                null_lookup=null_lookup,
                device=device,
            )
            epoch_record["valid_metrics"] = valid_metrics
            final_valid_metrics = valid_metrics
            if valid_metrics["fp_bps"] > best_valid_fp_bps:
                best_valid_fp_bps = float(valid_metrics["fp_bps"])
                best_valid_metrics = dict(valid_metrics)
                best_epoch = epoch
                save_checkpoint(
                    best_checkpoint_path,
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    valid_metrics=valid_metrics,
                    train_loss=mean_train_loss,
                    n_optimizer_steps=total_optimizer_steps,
                )

        save_checkpoint(
            last_checkpoint_path,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            valid_metrics=epoch_record.get("valid_metrics", final_valid_metrics or {}),
            train_loss=mean_train_loss,
            n_optimizer_steps=total_optimizer_steps,
        )
        history.append(epoch_record)
        print(json.dumps(epoch_record, ensure_ascii=False))

    if best_valid_metrics is None:
        best_valid_metrics = evaluate_faithful_ndt2_loader(
            model,
            valid_loader,
            spec=spec,
            channel_capacity=channel_capacity,
            neurons_per_token=bridge_cfg.neurons_per_token,
            null_lookup=null_lookup,
            device=device,
        )
        best_epoch = args.epochs
        best_valid_fp_bps = float(best_valid_metrics["fp_bps"])
        save_checkpoint(
            best_checkpoint_path,
            epoch=best_epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            valid_metrics=best_valid_metrics,
            train_loss=float(history[-1]["mean_train_loss"]),
            n_optimizer_steps=total_optimizer_steps,
        )
    if final_valid_metrics is None:
        final_valid_metrics = dict(best_valid_metrics)

    best_checkpoint = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    predict_fn = build_prediction_fn(
        model,
        spec=spec,
        channel_capacity=channel_capacity,
        neurons_per_token=bridge_cfg.neurons_per_token,
    )
    test_continuous = evaluate_faithful_ndt2_loader(
        model,
        test_loader,
        spec=spec,
        channel_capacity=channel_capacity,
        neurons_per_token=bridge_cfg.neurons_per_token,
        null_lookup=null_lookup,
        device=device,
    )
    test_trial = evaluate_trial_aligned_loader(
        predict_fn,
        test_trial_loader,
        spec,
        null_lookup,
        device,
    ) if test_trial_records else {"trial_fp_bps": float("nan"), "n_trials": 0}

    payload = make_result_payload(
        model_name="faithful_ndt2",
        protocol_name="phase1_benchmark_repro",
        spec=spec,
        best_epoch=best_epoch,
        best_valid_metrics=best_valid_metrics,
        final_epoch_metrics={
            "epoch": args.epochs,
            "mean_train_loss": float(history[-1]["mean_train_loss"]),
            "valid_continuous": final_valid_metrics,
        },
        test_metrics={
            "continuous": test_continuous,
            "trial_aligned": test_trial,
        },
        notes=[
            "Uses upstream BrainBertInterface with ShuffleInfill instead of the legacy simplified baseline.",
            "Canonical windows, null model, and test evaluation follow neural_benchmark.repro_protocol.",
            "This bridge preserves NDT2 flat-token input formatting and session token embeddings.",
        ],
    )
    payload["mode"] = "train"
    payload["device"] = str(device)
    payload["bridge_config"] = asdict(bridge_cfg)
    payload["n_sessions"] = len(session_ids)
    payload["raw_max_units"] = raw_max_units
    payload["channel_capacity"] = channel_capacity
    payload["window_counts"] = {
        "train_continuous": len(train_records),
        "valid_continuous": len(valid_records),
        "test_continuous": len(test_records),
        "test_trial_aligned": len(test_trial_records),
    }
    payload["train_protocol"] = {
        "batch_size": int(args.batch_size),
        "accumulate_batches": int(accumulate_batches),
        "effective_batch_size": int(args.batch_size * accumulate_batches),
        "optimizer_name": optimizer.__class__.__name__,
        "scheduler_name": None if scheduler is None else scheduler.__class__.__name__,
        "scheduler_from_model_config": True,
    }
    payload["history"] = history
    payload["checkpoint_paths"] = {
        "best_model": str(best_checkpoint_path),
        "last_model": str(last_checkpoint_path),
    }
    payload["model_fidelity_notes"] = list(payload["notes"])
    payload["best_valid_fp_bps"] = best_valid_fp_bps
    payload["total_optimizer_steps"] = int(total_optimizer_steps)
    write_json(output_dir / "results.json", payload)
    print(
        json.dumps(
            {
                "mode": "train",
                "output_dir": str(output_dir),
                "best_epoch": best_epoch,
                "best_valid_fp_bps": best_valid_fp_bps,
                "test_fp_bps": test_continuous["fp_bps"],
                "test_psth_r2": test_trial.get("per_neuron_psth_r2"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return payload


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
        lr_schedule=args.lr_schedule,
        lr_ramp_init_factor=args.lr_ramp_init_factor,
        lr_ramp_steps=args.lr_ramp_steps,
        lr_decay_steps=args.lr_decay_steps,
        lr_min=args.lr_min,
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
    parser.add_argument("--mode", choices=["smoke", "train"], default="smoke")
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
    parser.add_argument("--mask-ratio", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--lr-schedule", type=str, default="cosine_warmup")
    parser.add_argument("--lr-ramp-init-factor", type=float, default=0.1)
    parser.add_argument("--lr-ramp-steps", type=int, default=50)
    parser.add_argument("--lr-decay-steps", type=int, default=1000)
    parser.add_argument("--lr-min", type=float, default=1e-6)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--accumulate-batches", type=int, default=16)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--max-train-windows", type=int, default=0)
    parser.add_argument("--max-valid-windows", type=int, default=0)
    parser.add_argument("--max-test-windows", type=int, default=0)
    parser.add_argument("--max-trial-windows", type=int, default=0)
    args = parser.parse_args()

    if args.mode == "smoke":
        run_smoke(args)
    elif args.mode == "train":
        run_train(args)


if __name__ == "__main__":
    main()
