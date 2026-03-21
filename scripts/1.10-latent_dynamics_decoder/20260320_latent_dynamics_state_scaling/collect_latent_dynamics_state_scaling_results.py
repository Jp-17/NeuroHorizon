#!/usr/bin/env python3
"""Collect 1.10 state-scaling gate results into summary json and TSV."""

from __future__ import annotations

import csv
import json
from pathlib import Path


MODULE = "20260320_latent_dynamics_state_scaling"
WINDOW = "500ms"
BASELINE_FP_BPS = 0.1744


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
    window_dir = log_root / WINDOW
    fig_root = root / "results" / "figures" / "1.10-latent_dynamics_decoder" / MODULE
    fig_root.mkdir(parents=True, exist_ok=True)

    valid_path = window_dir / "eval_v2_valid_results.json"
    test_path = window_dir / "eval_v2_test_results.json"
    if not valid_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Missing eval json under {window_dir}")

    valid = load_json(valid_path)
    test = load_json(test_path)
    metrics_path = latest_metrics(window_dir)
    best_val = best_val_fp_bps(metrics_path) if metrics_path else None
    continuous_valid = valid.get("continuous", {})
    continuous_test = test.get("continuous", {})

    summary = {
        "module": MODULE,
        "windows": {
            WINDOW: {
                "baseline_fp_bps": BASELINE_FP_BPS,
                "teacher_forced": {
                    "checkpoint": valid.get("checkpoint"),
                    "continuous": continuous_valid,
                },
                "rollout": {
                    "checkpoint": valid.get("checkpoint"),
                    "continuous": continuous_valid,
                },
            }
        },
    }
    summary_path = fig_root / "latent_dynamics_state_scaling_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    tsv_update = {
        "name": MODULE,
        "commit": "ef0f6fd",
        "date": "2026-03-20",
        "description": "Wider latent state 500ms gate",
        "fp_bps_250ms": "",
        "fp_bps_500ms": str(continuous_valid.get("fp_bps", "")),
        "fp_bps_1000ms": "",
        "best_val_fp_bps_250ms": "",
        "best_val_fp_bps_500ms": "" if best_val is None else f"{best_val:.4f}",
        "best_val_fp_bps_1000ms": "",
        "best_test_fp_bps_250ms": "",
        "best_test_fp_bps_500ms": str(continuous_test.get("fp_bps", "")),
        "best_test_fp_bps_1000ms": "",
        "best_ckpt_250ms": "",
        "best_ckpt_500ms": Path(valid.get("checkpoint", "")).name if valid.get("checkpoint") else "",
        "best_ckpt_1000ms": "",
        "gpu_memory_gb": "",
        "train_time_hours": "",
        "notes": (
            "500ms gate only; obs=500ms; cont sampling; 10 sessions; 300 epochs; "
            "larger latent state regressed sharply vs 20260320_latent_dynamics_decoder"
        ),
    }

    results_tsv = root / "cc_todo" / "1.10-latent_dynamics_decoder" / "results.tsv"
    update_results_tsv(results_tsv, tsv_update)


if __name__ == "__main__":
    main()
