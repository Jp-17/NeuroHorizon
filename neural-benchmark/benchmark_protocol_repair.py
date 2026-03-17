#!/usr/bin/env python3
"""Protocol-fixed reevaluation of the legacy 1.8.3 simplified baselines.

This script does not attempt faithful benchmark reproduction. It reevaluates
the existing project-local simplified checkpoints under a stricter shared
protocol: deterministic continuous windows, held-out test split, raw-event null
rates, and explicit trial-aligned PSTH metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "/root/autodl-tmp/NeuroHorizon")
sys.path.insert(0, "/root/autodl-tmp/NeuroHorizon/neural_benchmark")

from neural_benchmark.adapters.base_adapter import BenchmarkConfig
from neural_benchmark.adapters.ibl_mtm_adapter import create_ibl_mtm_model
from neural_benchmark.adapters.ndt2_adapter import create_ndt2_model
from neural_benchmark.adapters.neuroformer_adapter import create_neuroformer_model
from neural_benchmark.repro_protocol import (
    BenchmarkProtocolSpec,
    WindowedBinnedDataset,
    build_continuous_windows,
    build_global_unit_index,
    build_trial_windows,
    collate_benchmark_batch,
    compute_max_units,
    compute_raw_null_rates,
    evaluate_continuous_loader,
    evaluate_trial_aligned_loader,
    load_split_datasets,
    make_result_payload,
)
from torch_brain.utils.neurohorizon_metrics import build_null_rate_lookup


MODEL_CREATORS = {
    "ndt2": create_ndt2_model,
    "ibl_mtm": create_ibl_mtm_model,
    "neuroformer": create_neuroformer_model,
}


def model_forward(model, batch):
    return model(
        batch["spike_counts"],
        obs_mask=batch.get("obs_mask"),
        pred_mask=batch.get("pred_mask"),
        unit_mask=batch.get("unit_mask"),
    )


def default_legacy_dir(model: str, pred_window: float) -> Path:
    return Path(
        "/root/autodl-tmp/NeuroHorizon/results/logs/"
        f"phase1_benchmark_{model}_{int(round(pred_window * 1000))}ms"
    )


def default_output_dir(model: str, pred_window: float) -> Path:
    return Path(
        "/root/autodl-tmp/NeuroHorizon/results/logs/"
        f"phase1_benchmark_protocolfix_{model}_{int(round(pred_window * 1000))}ms"
    )


def load_legacy_reference(legacy_dir: Path) -> Dict[str, object]:
    results_path = legacy_dir / "results.json"
    checkpoint_path = legacy_dir / "best_model.pt"
    with open(results_path, "r", encoding="utf-8") as handle:
        legacy_results = json.load(handle)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    history = legacy_results.get("history", {})
    return {
        "legacy_results_path": str(results_path),
        "legacy_checkpoint_path": str(checkpoint_path),
        "best_epoch": int(checkpoint.get("epoch", -1)),
        "legacy_best_val_fp_bps": legacy_results.get("best_val_fp_bps"),
        "legacy_best_checkpoint_metrics": checkpoint.get("metrics", {}),
        "legacy_last_logged_val_loss": history.get("val_loss", [None])[-1] if history.get("val_loss") else None,
        "legacy_last_logged_val_r2": history.get("val_r2", [None])[-1] if history.get("val_r2") else None,
        "legacy_last_logged_val_fp_bps": history.get("val_fp_bps", [None])[-1] if history.get("val_fp_bps") else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Protocol-fix reevaluation for legacy 1.8.3 checkpoints")
    parser.add_argument("--model", required=True, choices=sorted(MODEL_CREATORS.keys()))
    parser.add_argument("--pred-window", type=float, required=True)
    parser.add_argument("--obs-window", type=float, default=0.5)
    parser.add_argument("--bin-size", type=float, default=0.02)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--trial-batch-size", type=int, default=16)
    parser.add_argument("--sigma-bins", type=int, default=1)
    parser.add_argument("--dataset-config", type=str, default=None)
    parser.add_argument("--legacy-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--skip-trial", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    legacy_dir = Path(args.legacy_dir) if args.legacy_dir else default_legacy_dir(args.model, args.pred_window)
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(args.model, args.pred_window)
    output_dir.mkdir(parents=True, exist_ok=True)

    spec = BenchmarkProtocolSpec(
        obs_window_s=args.obs_window,
        pred_window_s=args.pred_window,
        bin_size_s=args.bin_size,
        step_s=None,
    )
    bench_cfg = BenchmarkConfig(
        pred_window_s=args.pred_window,
        obs_window_s=args.obs_window,
        batch_size=args.batch_size,
    )

    datasets = load_split_datasets(dataset_config=args.dataset_config)
    global_unit_index = build_global_unit_index(datasets)
    max_units = compute_max_units(datasets)

    legacy_reference = load_legacy_reference(legacy_dir)
    creator = MODEL_CREATORS[args.model]
    model = creator(max_units, bench_cfg).to(device)
    checkpoint = torch.load(legacy_reference["legacy_checkpoint_path"], map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    null_rates = compute_raw_null_rates(
        datasets["train"],
        global_unit_index=global_unit_index,
        bin_size_s=spec.bin_size_s,
    )
    null_lookup = build_null_rate_lookup(null_rates, device=device)

    loaders = {}
    window_counts = {}
    for split in ("valid", "test"):
        continuous_records = build_continuous_windows(datasets[split], split, spec)
        trial_records = build_trial_windows(datasets[split], split, spec)
        window_counts[split] = {
            "continuous_windows": len(continuous_records),
            "trial_windows": len(trial_records),
        }
        loaders[f"{split}_continuous"] = DataLoader(
            WindowedBinnedDataset(
                datasets[split],
                continuous_records,
                spec,
                global_unit_index,
                max_units=max_units,
            ),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_benchmark_batch,
        )
        if not args.skip_trial:
            loaders[f"{split}_trial"] = DataLoader(
                WindowedBinnedDataset(
                    datasets[split],
                    trial_records,
                    spec,
                    global_unit_index,
                    max_units=max_units,
                ),
                batch_size=args.trial_batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=collate_benchmark_batch,
            )

    best_valid_metrics = {
        "continuous": evaluate_continuous_loader(
            lambda batch: model_forward(model, batch),
            loaders["valid_continuous"],
            spec,
            null_lookup,
            device,
        ),
    }
    test_metrics = {
        "continuous": evaluate_continuous_loader(
            lambda batch: model_forward(model, batch),
            loaders["test_continuous"],
            spec,
            null_lookup,
            device,
        ),
    }

    if not args.skip_trial:
        best_valid_metrics["trial_aligned"] = evaluate_trial_aligned_loader(
            lambda batch: model_forward(model, batch),
            loaders["valid_trial"],
            spec,
            null_lookup,
            device,
            sigma_bins=args.sigma_bins,
        )
        test_metrics["trial_aligned"] = evaluate_trial_aligned_loader(
            lambda batch: model_forward(model, batch),
            loaders["test_trial"],
            spec,
            null_lookup,
            device,
            sigma_bins=args.sigma_bins,
        )

    payload = make_result_payload(
        model_name=args.model,
        protocol_name="legacy_protocolfix_v1",
        spec=spec,
        best_epoch=int(legacy_reference["best_epoch"]),
        best_valid_metrics=best_valid_metrics,
        final_epoch_metrics={
            "available": False,
            "reason": "legacy pipeline did not persist a final checkpoint for reevaluation",
        },
        test_metrics=test_metrics,
        notes=[
            "This output reevaluates the legacy simplified baseline checkpoint only.",
            "It is not a faithful reproduction of the original benchmark model.",
            "Held-out test metrics are comparable to the protocol-fixed manifest only, not to the old validation-only pipeline.",
        ],
    )
    payload["checkpoint"] = legacy_reference["legacy_checkpoint_path"]
    payload["legacy_reference"] = legacy_reference
    payload["window_counts"] = window_counts
    payload["n_global_units"] = len(global_unit_index)
    payload["max_units"] = max_units
    payload["model_fidelity_notes"] = (
        "project-local simplified baseline wrapper; upstream original model integration still pending"
    )
    payload["null_lookup_size"] = int(null_lookup.shape[0])

    results_path = output_dir / "results.json"
    with open(results_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print("=" * 72)
    print(f"Protocol-fix reevaluation complete: model={args.model}, pred_window={args.pred_window:.3f}s")
    print(f"  valid continuous fp-bps: {best_valid_metrics['continuous']['fp_bps']:.4f}")
    print(f"  test  continuous fp-bps: {test_metrics['continuous']['fp_bps']:.4f}")
    if not args.skip_trial and "trial_aligned" in test_metrics:
        print(f"  test  trial PSTH-R2:    {test_metrics['trial_aligned']['per_neuron_psth_r2']:.4f}")
    print(f"  saved: {results_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
