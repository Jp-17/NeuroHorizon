#!/usr/bin/env python3
"""Collect 1.10 latent-dynamics results into summary json and TSV."""

from __future__ import annotations

import csv
import json
from pathlib import Path


WINDOWS = ["250ms", "500ms", "1000ms"]
MODULE = "20260320_latent_dynamics_decoder"
BASELINE = {"250ms": 0.2115, "500ms": 0.1744, "1000ms": 0.1317}


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def latest_metrics(window_dir: Path) -> Path | None:
    matches = sorted(window_dir.glob("lightning_logs/version_*/metrics.csv"))
    return matches[-1] if matches else None


def best_val_fp_bps(metrics_path: Path) -> float | None:
    import pandas as pd

    df = pd.read_csv(metrics_path)
    if "val/fp_bps" not in df.columns:
        return None
    values = pd.to_numeric(df["val/fp_bps"], errors="coerce").dropna()
    return float(values.max()) if not values.empty else None


def update_results_tsv(tsv_path: Path, row_update: dict[str, str]) -> None:
    with tsv_path.open() as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = list(reader)
        fieldnames = reader.fieldnames

    assert fieldnames is not None
    updated = False
    for row in rows:
        if row["name"] == MODULE:
            row.update(row_update)
            updated = True
            break
    if not updated:
        rows.append(row_update)

    with tsv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    root = project_root()
    log_root = root / "results" / "logs" / "1.10-latent_dynamics_decoder" / MODULE
    fig_root = root / "results" / "figures" / "1.10-latent_dynamics_decoder" / MODULE
    fig_root.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {"module": MODULE, "windows": {}}
    tsv_update = {
        "name": MODULE,
        "commit": "",
        "date": "2026-03-20",
        "description": "GRU latent dynamics decoder",
        "fp_bps_250ms": "",
        "fp_bps_500ms": "",
        "fp_bps_1000ms": "",
        "best_val_fp_bps_250ms": "",
        "best_val_fp_bps_500ms": "",
        "best_val_fp_bps_1000ms": "",
        "best_test_fp_bps_250ms": "",
        "best_test_fp_bps_500ms": "",
        "best_test_fp_bps_1000ms": "",
        "best_ckpt_250ms": "",
        "best_ckpt_500ms": "",
        "best_ckpt_1000ms": "",
        "gpu_memory_gb": "",
        "train_time_hours": "",
        "notes": "obs=500ms; cont sampling; 10 sessions; 300 epochs; latent dynamics teacher-forced == rollout",
    }

    for window in WINDOWS:
        window_dir = log_root / window
        valid_path = window_dir / "eval_v2_valid_results.json"
        test_path = window_dir / "eval_v2_test_results.json"
        if not valid_path.exists() or not test_path.exists():
            continue

        valid = load_json(valid_path)
        test = load_json(test_path)
        metrics_path = latest_metrics(window_dir)
        best_val = best_val_fp_bps(metrics_path) if metrics_path else None
        continuous_valid = valid.get("continuous", {})
        continuous_test = test.get("continuous", {})

        summary["windows"][window] = {
            "baseline_fp_bps": BASELINE[window],
            "teacher_forced": {
                "checkpoint": valid.get("checkpoint"),
                "continuous": continuous_valid,
            },
            "rollout": {
                "checkpoint": valid.get("checkpoint"),
                "continuous": continuous_valid,
            },
        }

        tsv_update[f"fp_bps_{window}"] = str(continuous_valid.get("fp_bps", ""))
        tsv_update[f"best_val_fp_bps_{window}"] = "" if best_val is None else f"{best_val:.4f}"
        tsv_update[f"best_test_fp_bps_{window}"] = str(continuous_test.get("fp_bps", ""))
        checkpoint_name = Path(valid.get("checkpoint", "")).name if valid.get("checkpoint") else ""
        tsv_update[f"best_ckpt_{window}"] = checkpoint_name

    summary_path = fig_root / "latent_dynamics_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    results_tsv = root / "cc_todo" / "1.10-latent_dynamics_decoder" / "results.tsv"
    update_results_tsv(results_tsv, tsv_update)


if __name__ == "__main__":
    main()
