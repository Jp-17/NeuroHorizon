#!/usr/bin/env python3
"""Faithful IBL-MtM bridge on top of the canonical 1.8.3 protocol.

This bridge keeps the upstream IBL-MtM NDT1 core intact and only adds the
minimum compatibility layer needed to:

1. materialize canonical Perich-Miller windows from the shared protocol
2. feed binned counts into the original NDT1 + stitching + session prompting path
3. train with the upstream SSL multi-mask Poisson NLL objective
4. evaluate with the unified held-out one-step forward-pred protocol
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import Sampler

REPO_ROOT = "/root/autodl-tmp/NeuroHorizon"
IBL_ROOT = f"{REPO_ROOT}/neural-benchmark/benchmark_models/ibl-mtm"
IBL_SRC = f"{IBL_ROOT}/src"

sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, f"{REPO_ROOT}/neural_benchmark")

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


def import_ibl_modules():
    cwd = os.getcwd()
    os.chdir(IBL_ROOT)
    try:
        sys.path.insert(0, IBL_SRC)
        ndt1_module = importlib.import_module("models.ndt1")
        config_utils = importlib.import_module("utils.config_utils")
        return ndt1_module, ndt1_module.NDT1, config_utils.update_config
    finally:
        os.chdir(cwd)


NDT1_MODULE, NDT1, update_ibl_config = import_ibl_modules()


@dataclass(frozen=True)
class FaithfulIBLMtMConfig:
    lr: float = 1e-4
    weight_decay: float = 1e-2
    eps: float = 1e-8
    warmup_pct: float = 0.15
    div_factor: float = 10.0
    grad_clip: float = 1.0
    dropout: float = 0.4
    hidden_size: int = 512
    n_layers: int = 5
    n_heads: int = 8
    embed_mult: int = 2
    stitch_channels: Optional[int] = None
    mask_ratio: float = 0.3
    train_mask_mode: str = "combined"
    grad_accum_steps: int = 1


class FaithfulIBLMtMWindowDataset(TorchDataset):
    """Canonical window dataset converted into the original IBL-MtM tensor contract."""

    def __init__(
        self,
        tb_dataset,
        records,
        spec: BenchmarkProtocolSpec,
        global_unit_index: Mapping[Tuple[str, int], int],
        recording_n_units: Mapping[str, int],
    ) -> None:
        self.tb_dataset = tb_dataset
        self.records = list(records)
        self.spec = spec
        self.global_unit_index = global_unit_index
        self.recording_n_units = dict(recording_n_units)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        record = self.records[idx]
        window = load_binned_window(
            self.tb_dataset,
            record,
            self.spec,
            self.global_unit_index,
        )
        spike_counts = window["spike_counts"][: self.spec.total_bins].astype(np.float32)
        n_units = int(spike_counts.shape[1])
        return {
            "spikes_data": torch.from_numpy(spike_counts),
            "target": torch.from_numpy(spike_counts.copy()),
            "spike_counts": torch.from_numpy(spike_counts.copy()),
            "time_attn_mask": torch.ones(self.spec.total_bins, dtype=torch.long),
            "space_attn_mask": torch.ones(n_units, dtype=torch.long),
            "spikes_timestamps": torch.arange(self.spec.total_bins, dtype=torch.long),
            "spikes_spacestamps": torch.arange(n_units, dtype=torch.long),
            "unit_ids": torch.from_numpy(window["unit_ids"].astype(np.int64)),
            "unit_mask": torch.ones(n_units, dtype=torch.bool),
            "n_units": torch.tensor(n_units, dtype=torch.long),
            "session_id": window["recording_id"],
            "eid": window["recording_id"],
            "split": window["split"],
            "target_id": torch.tensor(
                -1 if window["target_id"] is None else int(window["target_id"]),
                dtype=torch.long,
            ),
            "go_cue_time_s": torch.tensor(
                float("nan") if window["go_cue_time_s"] is None else float(window["go_cue_time_s"]),
                dtype=torch.float32,
            ),
            "neuron_regions": ["unknown"] * n_units,
        }


class SessionBatchSampler(Sampler[List[int]]):
    """Batch windows by recording so upstream session prompting semantics remain valid."""

    def __init__(
        self,
        records,
        batch_size: int,
        shuffle: bool,
    ) -> None:
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.groups: Dict[str, List[int]] = {}
        for idx, record in enumerate(records):
            self.groups.setdefault(record.recording_id, []).append(idx)
        self.session_ids = list(self.groups.keys())

    def __iter__(self):
        session_ids = list(self.session_ids)
        if self.shuffle:
            random.shuffle(session_ids)
        for session_id in session_ids:
            indices = list(self.groups[session_id])
            if self.shuffle:
                random.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                yield indices[start : start + self.batch_size]

    def __len__(self) -> int:
        return sum(math.ceil(len(indices) / self.batch_size) for indices in self.groups.values())


def collate_ibl_batch(batch: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    result: Dict[str, object] = {}
    tensor_keys = {
        "spikes_data",
        "target",
        "spike_counts",
        "time_attn_mask",
        "space_attn_mask",
        "spikes_timestamps",
        "spikes_spacestamps",
        "unit_ids",
        "unit_mask",
        "n_units",
        "target_id",
        "go_cue_time_s",
    }
    list_keys = {"session_id", "eid", "split", "neuron_regions"}
    for key in batch[0].keys():
        if key in tensor_keys:
            result[key] = torch.stack([item[key] for item in batch])
        elif key in list_keys:
            result[key] = [item[key] for item in batch]
        else:
            raise KeyError(f"Unexpected batch key: {key}")
    return result


def move_batch_to_device(batch: Mapping[str, object], device: torch.device) -> Dict[str, object]:
    moved: Dict[str, object] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def heldout_forward_pred(spikes: torch.Tensor, obs_bins: int) -> Dict[str, torch.Tensor]:
    mask = torch.ones_like(spikes, dtype=torch.int64)
    mask[:, obs_bins:, :] = 0
    return {
        "spikes": spikes * mask.to(spikes.dtype),
        "eval_mask": 1 - mask,
    }


def batch_has_region_annotations(batch: Mapping[str, object]) -> bool:
    regions = batch.get("neuron_regions", [])
    if not regions:
        return False
    unique = {
        str(region)
        for sample_regions in regions
        for region in sample_regions
        if str(region).strip() and str(region).lower() != "unknown"
    }
    return len(unique) > 1


def choose_training_masking_mode(
    model,
    batch: Mapping[str, object],
    *,
    train_mask_mode: str,
    base_mask_ratio: float,
) -> str:
    if train_mask_mode in {"combined", "all"}:
        masking_schemes = ["neuron", "causal"]
        if train_mask_mode == "all" and batch_has_region_annotations(batch):
            masking_schemes += ["intra-region", "inter-region"]
        masking_mode = random.choice(masking_schemes)
    else:
        masking_mode = train_mask_mode

    if masking_mode == "causal":
        model.encoder.masker.ratio = 0.6
    else:
        model.encoder.masker.ratio = float(base_mask_ratio)
    return masking_mode


def resolve_forward_pred_masking_name(model) -> str:
    return "causal" if getattr(model, "use_prompt", False) else "forward-pred"


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def maybe_limit_records(records, limit: Optional[int]):
    if limit is None or limit <= 0:
        return list(records)
    return list(records[:limit])


def build_recording_n_units(datasets: Mapping[str, object]) -> Dict[str, int]:
    result: Dict[str, int] = {}
    for dataset in datasets.values():
        for recording_id in dataset.recording_dict.keys():
            if recording_id in result:
                continue
            result[recording_id] = len(dataset.get_recording_data(recording_id).units.id)
    return result


def build_window_loader(
    *,
    tb_dataset,
    records,
    spec: BenchmarkProtocolSpec,
    global_unit_index: Mapping[Tuple[str, int], int],
    recording_n_units: Mapping[str, int],
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    dataset = FaithfulIBLMtMWindowDataset(
        tb_dataset=tb_dataset,
        records=records,
        spec=spec,
        global_unit_index=global_unit_index,
        recording_n_units=recording_n_units,
    )
    batch_sampler = SessionBatchSampler(records, batch_size=batch_size, shuffle=shuffle)
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        collate_fn=collate_ibl_batch,
    )


def create_faithful_ibl_mtm_model(
    *,
    session_ids: Sequence[str],
    unique_unit_counts: Sequence[int],
    spec: BenchmarkProtocolSpec,
    channel_capacity: int,
    bridge_cfg: FaithfulIBLMtMConfig,
    config_path: str,
):
    NDT1_MODULE.include_eids = list(session_ids)
    config = update_ibl_config(config_path, None)
    config.encoder.stitching = True
    config.encoder.masker.force_active = True
    config.encoder.masker.mode = bridge_cfg.train_mask_mode
    config.encoder.masker.ratio = float(bridge_cfg.mask_ratio)
    config.encoder.masker.timesteps = None
    config.encoder.context.forward = 0
    config.encoder.context.backward = -1
    config.encoder.embedder.use_prompt = True
    config.encoder.embedder.use_session = True
    config.encoder.embedder.n_channels = (
        int(bridge_cfg.stitch_channels)
        if bridge_cfg.stitch_channels is not None
        else int(channel_capacity)
    )
    config.encoder.embedder.n_blocks = len(session_ids)
    config.encoder.embedder.n_dates = len(session_ids)
    config.encoder.embedder.max_F = spec.total_bins
    config.encoder.embedder.mult = int(bridge_cfg.embed_mult)
    config.encoder.embedder.dropout = float(bridge_cfg.dropout)
    config.encoder.transformer.hidden_size = int(bridge_cfg.hidden_size)
    config.encoder.transformer.n_layers = int(bridge_cfg.n_layers)
    config.encoder.transformer.n_heads = int(bridge_cfg.n_heads)
    config.encoder.transformer.dropout = float(bridge_cfg.dropout)

    cwd = os.getcwd()
    os.chdir(IBL_ROOT)
    try:
        model = NDT1(
            config,
            method_name="ssl",
            use_lograte=True,
            loss="poisson_nll",
            output_size=0,
            clf=False,
            reg=False,
            num_neurons=[int(x) for x in unique_unit_counts],
        )
    finally:
        os.chdir(cwd)
    return model


def run_ibl_train_forward(
    model,
    batch: Mapping[str, object],
    *,
    spec: BenchmarkProtocolSpec,
    train_mask_mode: str,
    base_mask_ratio: float,
):
    if train_mask_mode == "forward_pred":
        mask_result = heldout_forward_pred(batch["spikes_data"], spec.obs_bins)
        prev_mask = model.encoder.mask
        prev_force_active = model.encoder.masker.force_active
        model.encoder.mask = False
        model.encoder.masker.force_active = False
        try:
            outputs = model(
                mask_result["spikes"],
                time_attn_mask=batch["time_attn_mask"],
                space_attn_mask=batch["space_attn_mask"],
                spikes_timestamps=batch["spikes_timestamps"],
                spikes_spacestamps=batch["spikes_spacestamps"],
                targets=batch["target"],
                neuron_regions=batch["neuron_regions"],
                masking_mode=resolve_forward_pred_masking_name(model),
                eval_mask=mask_result["eval_mask"],
                num_neuron=int(batch["n_units"][0].item()),
                eid=batch["eid"][0],
            )
        finally:
            model.encoder.mask = prev_mask
            model.encoder.masker.force_active = prev_force_active
        return outputs, "forward_pred"

    model.encoder.mask = True
    model.encoder.masker.force_active = True
    masking_mode = choose_training_masking_mode(
        model,
        batch,
        train_mask_mode=train_mask_mode,
        base_mask_ratio=base_mask_ratio,
    )
    outputs = model(
        batch["spikes_data"],
        time_attn_mask=batch["time_attn_mask"],
        space_attn_mask=batch["space_attn_mask"],
        spikes_timestamps=batch["spikes_timestamps"],
        spikes_spacestamps=batch["spikes_spacestamps"],
        targets=batch["target"],
        neuron_regions=batch["neuron_regions"],
        masking_mode=masking_mode,
        num_neuron=int(batch["n_units"][0].item()),
        eid=batch["eid"][0],
    )
    return outputs, masking_mode


def run_ibl_eval_forward(
    model,
    batch: Mapping[str, object],
    *,
    spec: BenchmarkProtocolSpec,
):
    mask_result = heldout_forward_pred(batch["spikes_data"], spec.obs_bins)
    prev_mask = model.encoder.mask
    prev_force_active = model.encoder.masker.force_active
    model.encoder.mask = False
    model.encoder.masker.force_active = False
    try:
        outputs = model(
            mask_result["spikes"],
            time_attn_mask=batch["time_attn_mask"],
            space_attn_mask=batch["space_attn_mask"],
            spikes_timestamps=batch["spikes_timestamps"],
            spikes_spacestamps=batch["spikes_spacestamps"],
            targets=batch["target"],
            neuron_regions=batch["neuron_regions"],
            masking_mode=resolve_forward_pred_masking_name(model),
            eval_mask=mask_result["eval_mask"],
            num_neuron=int(batch["n_units"][0].item()),
            eid=batch["eid"][0],
        )
    finally:
        model.encoder.mask = prev_mask
        model.encoder.masker.force_active = prev_force_active
    return outputs


def predict_ibl_logrates(
    model,
    batch: Mapping[str, object],
    *,
    spec: BenchmarkProtocolSpec,
) -> torch.Tensor:
    outputs = run_ibl_eval_forward(model, batch, spec=spec)
    return outputs.preds[:, spec.obs_bins : spec.obs_bins + spec.pred_bins, :]


def evaluate_faithful_ibl_loader(
    model,
    dataloader: DataLoader,
    *,
    spec: BenchmarkProtocolSpec,
    channel_capacity: int,
    null_lookup: torch.Tensor,
    device: torch.device,
) -> Dict[str, object]:
    all_logrates = []
    all_targets = []
    all_unit_ids = []
    all_unit_masks = []
    total_loss = 0.0
    total_examples = 0.0

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            batch = move_batch_to_device(batch, device)
            outputs = run_ibl_eval_forward(model, batch, spec=spec)
            pred = outputs.preds[:, spec.obs_bins : spec.obs_bins + spec.pred_bins, :]
            target = batch["target"][:, spec.obs_bins : spec.obs_bins + spec.pred_bins, :]
            batch_size, pred_bins, n_units = pred.shape

            padded_pred = torch.zeros(
                batch_size,
                pred_bins,
                channel_capacity,
                device=pred.device,
                dtype=pred.dtype,
            )
            padded_target = torch.zeros(
                batch_size,
                pred_bins,
                channel_capacity,
                device=target.device,
                dtype=target.dtype,
            )
            padded_unit_ids = torch.zeros(
                batch_size,
                channel_capacity,
                device=batch["unit_ids"].device,
                dtype=batch["unit_ids"].dtype,
            )
            padded_unit_mask = torch.zeros(
                batch_size,
                channel_capacity,
                device=batch["unit_mask"].device,
                dtype=batch["unit_mask"].dtype,
            )
            padded_pred[:, :, :n_units] = pred
            padded_target[:, :, :n_units] = target
            padded_unit_ids[:, :n_units] = batch["unit_ids"]
            padded_unit_mask[:, :n_units] = batch["unit_mask"]

            all_logrates.append(padded_pred.cpu())
            all_targets.append(padded_target.cpu())
            all_unit_ids.append(padded_unit_ids.cpu())
            all_unit_masks.append(padded_unit_mask.cpu())
            total_loss += float(outputs.loss.detach().cpu().item())
            total_examples += float(outputs.n_examples.detach().cpu().item())

    metrics = evaluate_prediction_tensors(
        log_rates=torch.cat(all_logrates, dim=0),
        targets=torch.cat(all_targets, dim=0),
        unit_ids=torch.cat(all_unit_ids, dim=0),
        unit_mask=torch.cat(all_unit_masks, dim=0),
        null_lookup=null_lookup.cpu(),
    )
    metrics["n_samples"] = int(sum(x.shape[0] for x in all_logrates))
    metrics["loss_per_example"] = float(total_loss / max(total_examples, 1.0))
    return metrics


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def save_checkpoint(
    path: Path,
    *,
    epoch: int,
    model,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[object],
    valid_metrics: Mapping[str, object],
    train_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "valid_metrics": dict(valid_metrics),
        "train_loss": float(train_loss),
    }
    if scheduler is not None and hasattr(scheduler, "state_dict"):
        payload["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(payload, path)


def get_current_lr(optimizer: torch.optim.Optimizer) -> float:
    if not optimizer.param_groups:
        return float("nan")
    return float(optimizer.param_groups[0]["lr"])


def compute_warmup_progress(*, optimizer_steps: int, total_optimizer_steps: int, warmup_pct: float) -> float:
    warmup_steps = max(int(math.ceil(total_optimizer_steps * max(warmup_pct, 0.0))), 1)
    return float(min(optimizer_steps / warmup_steps, 1.0))


def build_trial_model_fn(model, spec: BenchmarkProtocolSpec):
    def predict(batch: Mapping[str, object]) -> torch.Tensor:
        return predict_ibl_logrates(model, batch, spec=spec)

    return predict


def build_train_protocol_summary(train_mask_mode: str) -> Dict[str, str]:
    if train_mask_mode == "forward_pred":
        return {
            "train_mask_mode": train_mask_mode,
            "mask_geometry": "explicit_future_window_forward_pred",
            "eval_geometry_match": "exact",
            "objective_family": "canonical_forward_prediction_control",
        }
    if train_mask_mode in {"combined", "all"}:
        return {
            "train_mask_mode": train_mask_mode,
            "mask_geometry": "upstream_ssl_multi_mask",
            "eval_geometry_match": "partial",
            "objective_family": "upstream_ssl_multitask",
        }
    return {
        "train_mask_mode": train_mask_mode,
        "mask_geometry": f"upstream_single_mask::{train_mask_mode}",
        "eval_geometry_match": "partial",
        "objective_family": "upstream_ssl_single_task",
    }


def run_smoke(args: argparse.Namespace) -> Dict[str, object]:
    set_global_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = BenchmarkProtocolSpec(
        obs_window_s=args.obs_window,
        pred_window_s=args.pred_window,
        bin_size_s=args.bin_size,
    )
    datasets = load_split_datasets(dataset_config=args.dataset_config)
    global_unit_index = build_global_unit_index(datasets)
    recording_n_units = build_recording_n_units(datasets)
    session_ids = sorted(recording_n_units.keys())
    unique_unit_counts = sorted(set(recording_n_units.values()))
    channel_capacity = compute_max_units(datasets)

    train_records = maybe_limit_records(build_continuous_windows(datasets["train"], "train", spec), 8)
    valid_records = maybe_limit_records(build_continuous_windows(datasets["valid"], "valid", spec), 4)
    trial_records = maybe_limit_records(build_trial_windows(datasets["valid"], "valid", spec), 4)

    train_loader = build_window_loader(
        tb_dataset=datasets["train"],
        records=train_records,
        spec=spec,
        global_unit_index=global_unit_index,
        recording_n_units=recording_n_units,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )
    valid_loader = build_window_loader(
        tb_dataset=datasets["valid"],
        records=valid_records,
        spec=spec,
        global_unit_index=global_unit_index,
        recording_n_units=recording_n_units,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )
    trial_loader = build_window_loader(
        tb_dataset=datasets["valid"],
        records=trial_records,
        spec=spec,
        global_unit_index=global_unit_index,
        recording_n_units=recording_n_units,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

    bridge_cfg = FaithfulIBLMtMConfig(
        lr=args.lr,
        weight_decay=args.weight_decay,
        eps=args.eps,
        warmup_pct=args.warmup_pct,
        div_factor=args.div_factor,
        grad_clip=args.grad_clip,
        dropout=args.dropout,
        hidden_size=args.hidden_size,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        embed_mult=args.embed_mult,
        stitch_channels=channel_capacity,
        mask_ratio=args.mask_ratio,
        train_mask_mode=args.train_mask_mode,
    )
    model = create_faithful_ibl_mtm_model(
        session_ids=session_ids,
        unique_unit_counts=unique_unit_counts,
        spec=spec,
        channel_capacity=channel_capacity,
        bridge_cfg=bridge_cfg,
        config_path=args.ibl_config,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=bridge_cfg.lr,
        weight_decay=bridge_cfg.weight_decay,
        eps=bridge_cfg.eps,
    )
    null_lookup = build_null_rate_lookup(
        compute_raw_null_rates(datasets["train"], global_unit_index, spec.bin_size_s),
        device=device,
    )

    batch = move_batch_to_device(next(iter(train_loader)), device)
    outputs, masking_mode = run_ibl_train_forward(
        model,
        batch,
        spec=spec,
        train_mask_mode=bridge_cfg.train_mask_mode,
        base_mask_ratio=bridge_cfg.mask_ratio,
    )
    optimizer.zero_grad(set_to_none=True)
    outputs.loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), bridge_cfg.grad_clip)
    optimizer.step()

    valid_metrics = evaluate_faithful_ibl_loader(
        model,
        valid_loader,
        spec=spec,
        channel_capacity=channel_capacity,
        null_lookup=null_lookup,
        device=device,
    )
    trial_metrics = evaluate_trial_aligned_loader(
        build_trial_model_fn(model, spec),
        trial_loader,
        spec,
        null_lookup,
        device,
    )

    payload = {
        "mode": "smoke",
        "model": "ibl_mtm_faithful",
        "spec": asdict(spec),
        "bridge_config": asdict(bridge_cfg),
        "train_batch_shape": list(batch["spikes_data"].shape),
        "train_loss": float(outputs.loss.detach().cpu().item()),
        "train_masking_mode": masking_mode,
        "valid_metrics": valid_metrics,
        "trial_metrics": trial_metrics,
    }
    output_dir = Path(args.output_dir or f"{REPO_ROOT}/results/logs/phase1_benchmark_faithful_ibl_mtm_smoke")
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
    bridge_cfg = FaithfulIBLMtMConfig(
        lr=args.lr,
        weight_decay=args.weight_decay,
        eps=args.eps,
        warmup_pct=args.warmup_pct,
        div_factor=args.div_factor,
        grad_clip=args.grad_clip,
        dropout=args.dropout,
        hidden_size=args.hidden_size,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        embed_mult=args.embed_mult,
        stitch_channels=args.stitch_channels,
        mask_ratio=args.mask_ratio,
        train_mask_mode=args.train_mask_mode,
        grad_accum_steps=args.grad_accum_steps,
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(
            f"{REPO_ROOT}/results/logs/phase1_benchmark_repro_faithful_ibl_mtm_{int(round(args.pred_window * 1000))}ms"
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = load_split_datasets(dataset_config=args.dataset_config)
    global_unit_index = build_global_unit_index(datasets)
    recording_n_units = build_recording_n_units(datasets)
    session_ids = sorted(recording_n_units.keys())
    unique_unit_counts = sorted(set(recording_n_units.values()))
    channel_capacity = compute_max_units(datasets)

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
        raise ValueError("No canonical training windows were generated for faithful IBL-MtM.")
    if not valid_records:
        raise ValueError("No canonical validation windows were generated for faithful IBL-MtM.")
    if not test_records:
        raise ValueError("No canonical test windows were generated for faithful IBL-MtM.")

    train_loader = build_window_loader(
        tb_dataset=datasets["train"],
        records=train_records,
        spec=spec,
        global_unit_index=global_unit_index,
        recording_n_units=recording_n_units,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    valid_loader = build_window_loader(
        tb_dataset=datasets["valid"],
        records=valid_records,
        spec=spec,
        global_unit_index=global_unit_index,
        recording_n_units=recording_n_units,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = build_window_loader(
        tb_dataset=datasets["test"],
        records=test_records,
        spec=spec,
        global_unit_index=global_unit_index,
        recording_n_units=recording_n_units,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_trial_loader = build_window_loader(
        tb_dataset=datasets["test"],
        records=test_trial_records,
        spec=spec,
        global_unit_index=global_unit_index,
        recording_n_units=recording_n_units,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = create_faithful_ibl_mtm_model(
        session_ids=session_ids,
        unique_unit_counts=unique_unit_counts,
        spec=spec,
        channel_capacity=channel_capacity,
        bridge_cfg=bridge_cfg,
        config_path=args.ibl_config,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=bridge_cfg.lr,
        weight_decay=bridge_cfg.weight_decay,
        eps=bridge_cfg.eps,
    )
    grad_accum_steps = max(int(bridge_cfg.grad_accum_steps), 1)
    effective_batch_size = int(args.batch_size * grad_accum_steps)
    steps_per_epoch = max(math.ceil(max(len(train_loader), 1) / grad_accum_steps), 1)
    total_optimizer_steps = max(args.epochs * steps_per_epoch, 1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        total_steps=total_optimizer_steps,
        max_lr=bridge_cfg.lr,
        pct_start=bridge_cfg.warmup_pct,
        div_factor=bridge_cfg.div_factor,
    )
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
    optimizer_steps = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_examples = 0.0
        train_mask_counts: Dict[str, int] = {}
        optimizer.zero_grad(set_to_none=True)
        for batch_idx, batch in enumerate(train_loader, start=1):
            batch = move_batch_to_device(batch, device)
            outputs, masking_mode = run_ibl_train_forward(
                model,
                batch,
                spec=spec,
                train_mask_mode=bridge_cfg.train_mask_mode,
                base_mask_ratio=bridge_cfg.mask_ratio,
            )
            (outputs.loss / grad_accum_steps).backward()
            should_step = (batch_idx % grad_accum_steps == 0) or (batch_idx == len(train_loader))
            if should_step:
                if bridge_cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), bridge_cfg.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
            train_loss_sum += float(outputs.loss.detach().cpu().item())
            train_examples += float(outputs.n_examples.detach().cpu().item())
            train_mask_counts[masking_mode] = train_mask_counts.get(masking_mode, 0) + 1

        valid_metrics = evaluate_faithful_ibl_loader(
            model,
            valid_loader,
            spec=spec,
            channel_capacity=channel_capacity,
            null_lookup=null_lookup,
            device=device,
        )
        mean_train_loss = float(train_loss_sum / max(train_examples, 1.0))
        final_epoch_metrics = dict(valid_metrics)
        history.append(
            {
                "epoch": int(epoch),
                "train_loss": mean_train_loss,
                "valid_fp_bps": float(valid_metrics["fp_bps"]),
                "valid_r2": float(valid_metrics["r2"]),
                "lr": get_current_lr(optimizer),
                "weight_decay": float(bridge_cfg.weight_decay),
                "grad_accum_steps": int(grad_accum_steps),
                "effective_batch_size": int(effective_batch_size),
                "optimizer_steps": int(optimizer_steps),
                "warmup_progress": compute_warmup_progress(
                    optimizer_steps=optimizer_steps,
                    total_optimizer_steps=total_optimizer_steps,
                    warmup_pct=bridge_cfg.warmup_pct,
                ),
                "train_mask_counts": dict(train_mask_counts),
            }
        )
        save_checkpoint(
            last_checkpoint_path,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            valid_metrics=valid_metrics,
            train_loss=mean_train_loss,
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
                scheduler=scheduler,
                valid_metrics=valid_metrics,
                train_loss=mean_train_loss,
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
        raise RuntimeError("Faithful IBL-MtM training finished without any validation metrics.")

    best_state = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(best_state["model_state_dict"])
    test_metrics = evaluate_faithful_ibl_loader(
        model,
        test_loader,
        spec=spec,
        channel_capacity=channel_capacity,
        null_lookup=null_lookup,
        device=device,
    )
    trial_metrics = evaluate_trial_aligned_loader(
        build_trial_model_fn(model, spec),
        test_trial_loader,
        spec,
        null_lookup,
        device,
    )

    payload = make_result_payload(
        model_name="faithful_ibl_mtm",
        protocol_name="canonical_protocol_v1",
        spec=spec,
        best_epoch=best_epoch,
        best_valid_metrics=best_valid_metrics,
        final_epoch_metrics=final_epoch_metrics or best_valid_metrics,
        test_metrics=test_metrics,
        notes=[
            "Faithful bridge keeps upstream IBL-MtM NDT1 + stitching + session prompting core.",
            "Training keeps the upstream SSL masking path unless train_mask_mode=forward_pred is explicitly requested.",
            "Held-out evaluation uses explicit one-step forward-pred masking on canonical windows.",
            "Batches are session-pure so upstream eid/session-token assumptions remain valid.",
        ],
    )
    payload["trial_aligned_test_metrics"] = dict(trial_metrics)
    payload["history"] = history
    payload["bridge_config"] = asdict(bridge_cfg)
    payload["train_protocol"] = build_train_protocol_summary(bridge_cfg.train_mask_mode)
    payload["optimizer_protocol"] = {
        "microbatch_size": int(args.batch_size),
        "grad_accum_steps": int(grad_accum_steps),
        "effective_batch_size": int(effective_batch_size),
        "weight_decay": float(bridge_cfg.weight_decay),
        "warmup_pct": float(bridge_cfg.warmup_pct),
        "total_optimizer_steps": int(total_optimizer_steps),
    }
    payload["model_fidelity_notes"] = [
        "Upstream NDT1 core retained.",
        "Perich-Miller adaptation replaces IBL EID list with canonical recording IDs at runtime.",
        "Stitching target channels are adapted to the Perich-Miller max unit count.",
        "Region-based masks are only activated when real region annotations are available; otherwise training falls back to the upstream combined neuron/causal scheme.",
        "The forward_pred training control matches the canonical held-out future-window geometry exactly when enabled.",
    ]
    write_json(output_dir / "results.json", payload)
    print(json.dumps(payload, indent=2))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Faithful IBL-MtM bridge for benchmark 1.8.3")
    parser.add_argument("--mode", type=str, default="smoke", choices=["smoke", "train"])
    parser.add_argument("--obs-window", type=float, default=0.5)
    parser.add_argument("--pred-window", type=float, default=0.25)
    parser.add_argument("--bin-size", type=float, default=0.02)
    parser.add_argument("--dataset-config", type=str, default=None)
    parser.add_argument(
        "--ibl-config",
        type=str,
        default=f"{IBL_ROOT}/src/configs/ndt1_stitching_prompting.yaml",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-windows", type=int, default=None)
    parser.add_argument("--max-valid-windows", type=int, default=None)
    parser.add_argument("--max-test-windows", type=int, default=None)
    parser.add_argument("--max-trial-windows", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--warmup-pct", type=float, default=0.15)
    parser.add_argument("--div-factor", type=float, default=10.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--n-layers", type=int, default=5)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--embed-mult", type=int, default=2)
    parser.add_argument("--stitch-channels", type=int, default=None)
    parser.add_argument("--mask-ratio", type=float, default=0.3)
    parser.add_argument(
        "--train-mask-mode",
        type=str,
        default="combined",
        choices=["combined", "all", "neuron", "causal", "intra-region", "inter-region", "forward_pred"],
    )
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "smoke":
        run_smoke(args)
    else:
        run_train(args)


if __name__ == "__main__":
    main()
