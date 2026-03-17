#!/usr/bin/env python3
"""Canonical benchmark protocol helpers for faithful 1.8.3 reproduction.

This module does not train any model by itself. It provides the shared data and
evaluation protocol that every faithful reproduction must use:

1. deterministic, no-overlap continuous windows derived from torch_brain splits
2. shared global unit indexing
3. raw-event null model computation
4. unified fp-bps / R2 / Poisson NLL aggregation
5. a stable result schema for best-valid / final / held-out test metrics

The original `phase1_benchmark_*` outputs remain legacy artifacts and should
not be mixed with this protocol.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset

sys.path.insert(0, "/root/autodl-tmp/NeuroHorizon")

from torch_brain.data import Dataset as TBDataset
from torch_brain.utils.neurohorizon_metrics import (
    build_null_rate_lookup,
    fp_bps,
    fp_bps_per_bin,
    fp_bps_stats,
    finalize_fp_bps_from_stats,
    psth_r2,
    r2_score,
)
from torch_brain.nn.loss import PoissonNLLLoss


DATA_ROOT = "/root/autodl-tmp/NeuroHorizon/data/processed/"
DATASET_CONFIG = (
    "/root/autodl-tmp/NeuroHorizon/examples/neurohorizon/configs/dataset/"
    "perich_miller_10sessions.yaml"
)


@dataclass(frozen=True)
class BenchmarkProtocolSpec:
    obs_window_s: float = 0.500
    pred_window_s: float = 0.250
    bin_size_s: float = 0.020
    step_s: Optional[float] = None

    @property
    def total_window_s(self) -> float:
        return self.obs_window_s + self.pred_window_s

    @property
    def obs_bins(self) -> int:
        return int(round(self.obs_window_s / self.bin_size_s))

    @property
    def pred_bins(self) -> int:
        return int(round(self.pred_window_s / self.bin_size_s))

    @property
    def total_bins(self) -> int:
        return self.obs_bins + self.pred_bins

    @property
    def resolved_step_s(self) -> float:
        return self.total_window_s if self.step_s is None else self.step_s


@dataclass(frozen=True)
class ContinuousWindowRecord:
    split: str
    recording_id: str
    start_s: float
    end_s: float


@dataclass(frozen=True)
class TrialWindowRecord:
    split: str
    recording_id: str
    start_s: float
    end_s: float
    go_cue_time_s: float
    target_id: int


def create_tb_dataset(split: str, dataset_config: Optional[str] = None) -> TBDataset:
    dataset = TBDataset(
        root=DATA_ROOT,
        config=dataset_config or DATASET_CONFIG,
        split=split,
    )
    dataset.disable_data_leakage_check()
    return dataset


def load_split_datasets(
    dataset_config: Optional[str] = None,
    splits: Sequence[str] = ("train", "valid", "test"),
) -> Dict[str, TBDataset]:
    return {split: create_tb_dataset(split, dataset_config=dataset_config) for split in splits}


def build_global_unit_index(datasets: Mapping[str, TBDataset]) -> Dict[Tuple[str, int], int]:
    global_index: Dict[Tuple[str, int], int] = {}
    next_idx = 0
    seen_recordings = set()
    for dataset in datasets.values():
        for recording_id in dataset.recording_dict.keys():
            if recording_id in seen_recordings:
                continue
            data = dataset.get_recording_data(recording_id)
            for local_idx, _ in enumerate(data.units.id):
                global_index[(recording_id, local_idx)] = next_idx
                next_idx += 1
            seen_recordings.add(recording_id)
    return global_index


def compute_max_units(datasets: Mapping[str, TBDataset]) -> int:
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


def iter_interval_windows(
    starts: Iterable[float],
    ends: Iterable[float],
    window_s: float,
    step_s: float,
) -> Iterable[Tuple[float, float]]:
    for start, end in zip(starts, ends):
        start = float(start)
        end = float(end)
        if end - start < window_s:
            continue
        last_start: Optional[float] = None
        last_end: Optional[float] = None
        t = start
        while t + window_s <= end + 1e-9:
            t_out = float(t)
            end_out = t_out + window_s
            yield t_out, end_out
            last_start = t_out
            last_end = end_out
            t += step_s
        if last_end is not None and last_end < end - 1e-9:
            tail_start = float(end - window_s)
            if last_start is None or not math.isclose(
                tail_start,
                last_start,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                yield tail_start, end


def build_continuous_windows(
    dataset: TBDataset,
    split: str,
    spec: BenchmarkProtocolSpec,
) -> List[ContinuousWindowRecord]:
    windows: List[ContinuousWindowRecord] = []
    intervals_dict = dataset.get_sampling_intervals()
    for recording_id, intervals in intervals_dict.items():
        for start_s, end_s in iter_interval_windows(
            intervals.start,
            intervals.end,
            spec.total_window_s,
            spec.resolved_step_s,
        ):
            windows.append(
                ContinuousWindowRecord(
                    split=split,
                    recording_id=recording_id,
                    start_s=start_s,
                    end_s=end_s,
                )
            )
    return windows


def build_trial_windows(
    dataset: TBDataset,
    split: str,
    spec: BenchmarkProtocolSpec,
) -> List[TrialWindowRecord]:
    windows: List[TrialWindowRecord] = []
    trial_info = dataset.get_trial_intervals(split=split)
    intervals_dict = dataset.get_sampling_intervals()
    for recording_id, info in trial_info.items():
        allowed = intervals_dict.get(recording_id)
        if allowed is None:
            continue
        allowed_intervals = list(zip(np.asarray(allowed.start), np.asarray(allowed.end)))
        for go_cue_time, target_id in zip(info["go_cue_time"], info["target_id"]):
            start_s = float(go_cue_time) - spec.obs_window_s
            end_s = float(go_cue_time) + spec.pred_window_s
            valid = any(start_s >= float(a) and end_s <= float(b) for a, b in allowed_intervals)
            if not valid:
                continue
            windows.append(
                TrialWindowRecord(
                    split=split,
                    recording_id=recording_id,
                    start_s=start_s,
                    end_s=end_s,
                    go_cue_time_s=float(go_cue_time),
                    target_id=int(target_id),
                )
            )
    return windows


def bin_spike_events(
    timestamps: np.ndarray,
    unit_index: np.ndarray,
    n_units: int,
    start_s: float,
    end_s: float,
    bin_size_s: float,
) -> np.ndarray:
    n_bins = int(round((end_s - start_s) / bin_size_s))
    counts = np.zeros((n_bins, n_units), dtype=np.float32)
    if len(timestamps) == 0:
        return counts
    bin_indices = np.floor((timestamps - start_s) / bin_size_s).astype(np.int64)
    valid = (bin_indices >= 0) & (bin_indices < n_bins)
    if not np.any(valid):
        return counts
    np.add.at(counts, (bin_indices[valid], unit_index[valid]), 1)
    return counts


def load_binned_window(
    dataset: TBDataset,
    record: ContinuousWindowRecord | TrialWindowRecord,
    spec: BenchmarkProtocolSpec,
    global_unit_index: Mapping[Tuple[str, int], int],
) -> Dict[str, object]:
    sample = dataset.get(record.recording_id, record.start_s, record.end_s)
    timestamps = np.asarray(sample.spikes.timestamps, dtype=np.float64)
    unit_index = np.asarray(sample.spikes.unit_index, dtype=np.int64)
    n_units = len(sample.units.id)
    counts = bin_spike_events(
        timestamps=timestamps,
        unit_index=unit_index,
        n_units=n_units,
        start_s=float(sample.start),
        end_s=float(sample.end),
        bin_size_s=spec.bin_size_s,
    )
    unit_ids = np.asarray(
        [global_unit_index[(record.recording_id, local_idx)] for local_idx in range(n_units)],
        dtype=np.int64,
    )
    return {
        "spike_counts": counts,
        "unit_ids": unit_ids,
        "unit_mask": np.ones(n_units, dtype=bool),
        "recording_id": record.recording_id,
        "split": record.split,
        "target_id": getattr(record, "target_id", None),
        "go_cue_time_s": getattr(record, "go_cue_time_s", None),
    }


class WindowedBinnedDataset(TorchDataset):
    """Materialize canonical continuous or trial-aligned windows for legacy baselines."""

    def __init__(
        self,
        tb_dataset: TBDataset,
        records: Sequence[ContinuousWindowRecord | TrialWindowRecord],
        spec: BenchmarkProtocolSpec,
        global_unit_index: Mapping[Tuple[str, int], int],
        max_units: int,
    ) -> None:
        self.tb_dataset = tb_dataset
        self.records = list(records)
        self.spec = spec
        self.global_unit_index = global_unit_index
        self.max_units = max_units

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        record = self.records[idx]
        sample = load_binned_window(
            self.tb_dataset,
            record,
            self.spec,
            self.global_unit_index,
        )
        spike_counts = sample["spike_counts"]
        unit_ids = sample["unit_ids"]
        n_units = min(int(spike_counts.shape[1]), self.max_units)
        padded_counts = np.zeros((self.spec.total_bins, self.max_units), dtype=np.float32)
        effective_t = min(int(spike_counts.shape[0]), self.spec.total_bins)
        padded_counts[:effective_t, :n_units] = spike_counts[:effective_t, :n_units]
        obs_mask = np.zeros(self.spec.total_bins, dtype=bool)
        obs_mask[: self.spec.obs_bins] = True
        pred_mask = np.zeros(self.spec.total_bins, dtype=bool)
        pred_mask[self.spec.obs_bins :] = True
        unit_mask = np.zeros(self.max_units, dtype=bool)
        unit_mask[:n_units] = True
        padded_unit_ids = np.zeros(self.max_units, dtype=np.int64)
        padded_unit_ids[:n_units] = unit_ids[:n_units]
        return {
            "spike_counts": torch.from_numpy(padded_counts),
            "obs_mask": torch.from_numpy(obs_mask),
            "pred_mask": torch.from_numpy(pred_mask),
            "unit_mask": torch.from_numpy(unit_mask),
            "unit_ids": torch.from_numpy(padded_unit_ids),
            "n_units": n_units,
            "session_id": sample["recording_id"],
            "split": sample["split"],
            "target_id": -1 if sample["target_id"] is None else int(sample["target_id"]),
            "go_cue_time_s": (
                float("nan") if sample["go_cue_time_s"] is None else float(sample["go_cue_time_s"])
            ),
        }


def collate_benchmark_batch(batch: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    result: Dict[str, object] = {}
    for key in batch[0].keys():
        if key in {"session_id", "split"}:
            result[key] = [item[key] for item in batch]
        elif isinstance(batch[0][key], torch.Tensor):
            result[key] = torch.stack([item[key] for item in batch])
        else:
            result[key] = torch.tensor([item[key] for item in batch])
    return result


def compute_raw_null_rates(
    dataset: TBDataset,
    global_unit_index: Mapping[Tuple[str, int], int],
    bin_size_s: float,
) -> Dict[int, float]:
    total_counts = defaultdict(float)
    total_bins = defaultdict(float)
    for recording_id in dataset.recording_dict.keys():
        data = dataset._get_data_object(recording_id)
        intervals = dataset.get_sampling_intervals().get(recording_id)
        if intervals is None:
            continue
        n_units = len(data.units.id)
        for local_idx in range(n_units):
            total_bins[global_unit_index[(recording_id, local_idx)]] += 0.0
        for start_s, end_s in zip(intervals.start, intervals.end):
            start_s = float(start_s)
            end_s = float(end_s)
            duration_bins = (end_s - start_s) / bin_size_s
            if duration_bins <= 0:
                continue
            sliced = data.slice(start_s, end_s)
            spike_uid = np.asarray(sliced.spikes.unit_index)
            for local_idx in range(n_units):
                gid = global_unit_index[(recording_id, local_idx)]
                total_counts[gid] += float((spike_uid == local_idx).sum())
                total_bins[gid] += duration_bins
    null_rates: Dict[int, float] = {}
    for gid, denom in total_bins.items():
        mean_count = total_counts[gid] / denom if denom > 0 else 1e-6
        null_rates[gid] = math.log(max(mean_count, 1e-6))
    return null_rates


def evaluate_prediction_tensors(
    log_rates: torch.Tensor,
    targets: torch.Tensor,
    unit_ids: torch.Tensor,
    unit_mask: torch.Tensor,
    null_lookup: torch.Tensor,
) -> Dict[str, object]:
    null_log_rates = null_lookup[unit_ids]
    pred_rates = torch.exp(log_rates.clamp(-10, 10))
    poisson_loss = PoissonNLLLoss()
    mask_3d = unit_mask.unsqueeze(1).expand_as(log_rates)
    return {
        "fp_bps": float(fp_bps(log_rates, targets, null_log_rates, unit_mask).item()),
        "r2": float(r2_score(pred_rates, targets, unit_mask).item()),
        "poisson_nll": float(poisson_loss(log_rates[mask_3d], targets[mask_3d]).item()),
        "per_bin_fp_bps": [
            float(x) for x in fp_bps_per_bin(log_rates, targets, null_log_rates, unit_mask).tolist()
        ],
    }


def evaluate_continuous_loader(
    model_fn,
    dataloader,
    spec: BenchmarkProtocolSpec,
    null_lookup: torch.Tensor,
    device: torch.device,
) -> Dict[str, object]:
    all_log_rates = []
    all_targets = []
    all_unit_ids = []
    all_unit_masks = []
    with torch.no_grad():
        for batch in dataloader:
            batch = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            log_rate = model_fn(batch)
            target = batch["spike_counts"][:, spec.obs_bins : spec.obs_bins + spec.pred_bins, :]
            all_log_rates.append(log_rate.cpu())
            all_targets.append(target.cpu())
            all_unit_ids.append(batch["unit_ids"].cpu())
            all_unit_masks.append(batch["unit_mask"].cpu())
    log_rates = torch.cat(all_log_rates, dim=0)
    targets = torch.cat(all_targets, dim=0)
    unit_ids = torch.cat(all_unit_ids, dim=0)
    unit_mask = torch.cat(all_unit_masks, dim=0)
    metrics = evaluate_prediction_tensors(
        log_rates=log_rates,
        targets=targets,
        unit_ids=unit_ids,
        unit_mask=unit_mask,
        null_lookup=null_lookup.cpu(),
    )
    metrics["n_samples"] = int(log_rates.shape[0])
    return metrics


def evaluate_trial_aligned_loader(
    model_fn,
    dataloader,
    spec: BenchmarkProtocolSpec,
    null_lookup: torch.Tensor,
    device: torch.device,
    sigma_bins: int = 1,
) -> Dict[str, object]:
    pred_by_group: Dict[Tuple[str, int], List[torch.Tensor]] = defaultdict(list)
    true_by_group: Dict[Tuple[str, int], List[torch.Tensor]] = defaultdict(list)
    trials_per_target: Dict[str, int] = defaultdict(int)
    total_nll_model = torch.zeros((), device=device, dtype=torch.float64)
    total_nll_null = torch.zeros((), device=device, dtype=torch.float64)
    total_spikes = torch.zeros((), device=device, dtype=torch.float64)

    with torch.no_grad():
        for batch in dataloader:
            batch = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            log_rate = model_fn(batch)
            target = batch["spike_counts"][:, spec.obs_bins : spec.obs_bins + spec.pred_bins, :]
            unit_ids = batch["unit_ids"]
            unit_mask = batch["unit_mask"]
            null_log_rates = null_lookup[unit_ids.clamp(0, null_lookup.shape[0] - 1)]
            nll_model_sum, nll_null_sum, spikes_sum = fp_bps_stats(
                log_rate,
                target,
                null_log_rates,
                unit_mask,
            )
            total_nll_model += nll_model_sum
            total_nll_null += nll_null_sum
            total_spikes += spikes_sum

            pred_rate = torch.exp(log_rate.clamp(-10, 10)).cpu()
            target_cpu = target.cpu()
            unit_mask_cpu = unit_mask.cpu()
            target_ids = batch["target_id"].detach().cpu().tolist()
            session_ids = batch["session_id"]

            for i, target_id in enumerate(target_ids):
                if int(target_id) < 0:
                    continue
                mask_i = unit_mask_cpu[i]
                if mask_i.sum().item() == 0:
                    continue
                group_key = (session_ids[i], int(target_id))
                pred_by_group[group_key].append(pred_rate[i][:, mask_i])
                true_by_group[group_key].append(target_cpu[i][:, mask_i])
                trials_per_target[str(int(target_id))] += 1

    results: Dict[str, object] = {
        "trial_fp_bps": float(
            finalize_fp_bps_from_stats(total_nll_model, total_nll_null, total_spikes)
        ) if total_spikes > 0 else 0.0,
        "n_trials": int(sum(len(v) for v in pred_by_group.values())),
    }
    if pred_by_group:
        pred_stacked = {key: torch.stack(value) for key, value in pred_by_group.items()}
        true_stacked = {key: torch.stack(value) for key, value in true_by_group.items()}
        results["per_neuron_psth_r2"] = float(
            psth_r2(pred_stacked, true_stacked, sigma_bins=sigma_bins)
        )
        per_target_r2 = {}
        target_ids = sorted({target_id for _, target_id in pred_stacked.keys()})
        for target_id in target_ids:
            pred_sub = {
                key: value for key, value in pred_stacked.items() if key[1] == target_id
            }
            true_sub = {
                key: value for key, value in true_stacked.items() if key[1] == target_id
            }
            per_target_r2[str(target_id)] = float(
                psth_r2(pred_sub, true_sub, sigma_bins=sigma_bins)
            )
        results["per_target_per_neuron_psth_r2"] = per_target_r2
        results["trials_per_target"] = dict(trials_per_target)
    else:
        results["per_neuron_psth_r2"] = float("nan")
        results["per_target_per_neuron_psth_r2"] = {}
        results["trials_per_target"] = {}
    return results


def make_result_payload(
    *,
    model_name: str,
    protocol_name: str,
    spec: BenchmarkProtocolSpec,
    best_epoch: int,
    best_valid_metrics: Mapping[str, object],
    final_epoch_metrics: Mapping[str, object],
    test_metrics: Mapping[str, object],
    notes: Sequence[str],
) -> Dict[str, object]:
    return {
        "model": model_name,
        "protocol": protocol_name,
        "spec": asdict(spec),
        "best_epoch": int(best_epoch),
        "best_valid_metrics": dict(best_valid_metrics),
        "final_epoch_metrics": dict(final_epoch_metrics),
        "test_metrics": dict(test_metrics),
        "sampler_spec": {
            "continuous": "deterministic_non_overlapping_windows",
            "trial_aligned": "go_cue_aligned_trials",
        },
        "null_model_spec": "raw_event_train_split_mean_count_per_bin",
        "notes": list(notes),
    }


def inspect_protocol(
    dataset_config: Optional[str],
    spec: BenchmarkProtocolSpec,
) -> Dict[str, object]:
    datasets = load_split_datasets(dataset_config=dataset_config)
    global_unit_index = build_global_unit_index(datasets)
    summary: Dict[str, object] = {
        "spec": asdict(spec),
        "n_global_units": len(global_unit_index),
        "splits": {},
    }
    for split, dataset in datasets.items():
        continuous = build_continuous_windows(dataset, split, spec)
        trial = build_trial_windows(dataset, split, spec)
        summary["splits"][split] = {
            "continuous_windows": len(continuous),
            "trial_windows": len(trial),
            "recordings": len(dataset.recording_dict),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect canonical faithful benchmark protocol")
    parser.add_argument("--obs-window", type=float, default=0.5)
    parser.add_argument("--pred-window", type=float, default=0.25)
    parser.add_argument("--bin-size", type=float, default=0.02)
    parser.add_argument("--step", type=float, default=None)
    parser.add_argument("--dataset-config", type=str, default=None)
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    spec = BenchmarkProtocolSpec(
        obs_window_s=args.obs_window,
        pred_window_s=args.pred_window,
        bin_size_s=args.bin_size,
        step_s=args.step,
    )
    summary = inspect_protocol(args.dataset_config, spec)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
