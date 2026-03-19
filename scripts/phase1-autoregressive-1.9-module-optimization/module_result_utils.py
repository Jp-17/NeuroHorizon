#!/usr/bin/env python3
"""Shared utilities for Phase 1.9 module-optimization result collection."""

from __future__ import annotations

import csv
import json
import subprocess
from datetime import date
from pathlib import Path

import pandas as pd
import torch


ROOT = Path("/root/autodl-tmp/NeuroHorizon")
WINDOWS = ["250ms", "500ms", "1000ms"]
TSV_PATH = ROOT / "cc_todo" / "phase1-autoregressive" / "1.9-module-optimization" / "results.tsv"
PLOT_SCRIPT = ROOT / "cc_todo" / "phase1-autoregressive" / "1.9-module-optimization" / "plot_optimization_progress.py"
CURVE_PLOT_SCRIPT = ROOT / "scripts" / "phase1-autoregressive-1.9-module-optimization" / "plot_optimization_training_curves.py"
LEGACY_FP_COLUMNS = ["fp_bps_250ms", "fp_bps_500ms", "fp_bps_1000ms"]
BEST_VAL_COLUMNS = [
    "best_val_fp_bps_250ms",
    "best_val_fp_bps_500ms",
    "best_val_fp_bps_1000ms",
]
BEST_TEST_COLUMNS = [
    "best_test_fp_bps_250ms",
    "best_test_fp_bps_500ms",
    "best_test_fp_bps_1000ms",
]
BEST_CKPT_COLUMNS = [
    "best_ckpt_250ms",
    "best_ckpt_500ms",
    "best_ckpt_1000ms",
]
FIELDNAMES = [
    "name",
    "commit",
    "date",
    "description",
    *LEGACY_FP_COLUMNS,
    *BEST_VAL_COLUMNS,
    *BEST_TEST_COLUMNS,
    *BEST_CKPT_COLUMNS,
    "gpu_memory_gb",
    "train_time_hours",
    "notes",
]


def load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def current_commit() -> str:
    result = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def format_value(value):
    if value is None:
        return "-"
    return f"{float(value):.4f}"


def format_checkpoint_id(path: str | None) -> str:
    if not path:
        return "-"
    return str(Path(path).name)


def load_rows():
    with TSV_PATH.open() as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    normalized = []
    for row in rows:
        normalized_row = {field: row.get(field, "-") for field in FIELDNAMES}
        normalized.append(normalized_row)
    return normalized


def save_rows(rows):
    with TSV_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _metrics_frame(metrics_path: Path) -> pd.DataFrame:
    df = pd.read_csv(metrics_path)
    for column in ["epoch", "step", "val/fp_bps", "val_loss"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    val = df.dropna(subset=["epoch"]).copy()
    if "val/fp_bps" not in val.columns or "val_loss" not in val.columns:
        return pd.DataFrame(columns=["epoch", "val/fp_bps", "val_loss"])
    val = val[val[["val/fp_bps", "val_loss"]].notna().any(axis=1)]
    if val.empty:
        return pd.DataFrame(columns=["epoch", "val/fp_bps", "val_loss"])
    return (
        val.groupby("epoch", as_index=False)[["val/fp_bps", "val_loss"]]
        .last()
        .sort_values("epoch")
    )


def extract_curve_metrics(metrics_path: Path) -> dict:
    val = _metrics_frame(metrics_path)
    if val.empty:
        return {}

    best = val.sort_values(["val/fp_bps", "val_loss"], ascending=[False, True]).iloc[0]
    last = val.iloc[-1]
    return {
        "metrics_path": str(metrics_path),
        "curve_best_epoch": int(best["epoch"]),
        "curve_best_val_fp_bps": float(best["val/fp_bps"]),
        "curve_best_val_loss": float(best["val_loss"]),
        "curve_last_epoch": int(last["epoch"]),
        "curve_last_val_fp_bps": float(last["val/fp_bps"]),
        "curve_last_val_loss": float(last["val_loss"]),
    }


def checkpoint_epoch(path: str | None) -> int | None:
    if not path:
        return None
    state = torch.load(path, map_location="cpu", weights_only=False)
    return int(state.get("epoch", -1))


def checkpoint_summary(checkpoint_path: str | None) -> dict | None:
    if not checkpoint_path:
        return None
    summary_path = Path(checkpoint_path).parent / "checkpoint_summary.json"
    if not summary_path.exists():
        return None
    return json.loads(summary_path.read_text())


def build_protocol_metadata() -> dict:
    return {
        "dataset_config": "examples/neurohorizon/configs/dataset/perich_miller_10sessions.yaml",
        "train_sampler": "continuous / RandomFixedWindowSampler",
        "valid_sampler": "continuous / SequentialFixedWindowSampler",
        "test_sampler": "continuous / SequentialFixedWindowSampler",
        "obs_window_ms": 500,
        "pred_windows_ms": [250, 500, 1000],
        "training_entrypoint": "examples/neurohorizon/train.py",
        "evaluation_entrypoint": "scripts/analysis/neurohorizon/eval_phase1_v2.py",
        "best_checkpoint_rule": "max(val/fp_bps), tie-break=min(val_loss)",
    }


def _continuous_fp(payload: dict | None) -> float | None:
    if not payload:
        return None
    return payload.get("continuous", {}).get("fp_bps")


def _checkpoint_from_payload(*payloads: dict | None) -> str | None:
    for payload in payloads:
        if payload and payload.get("checkpoint"):
            return payload["checkpoint"]
    return None


def collect_module_results(
    *,
    module: str,
    description: str,
    summary_filename: str,
    notes: str,
) -> Path:
    log_root = ROOT / "results" / "logs" / "phase1-autoregressive-1.9-module-optimization" / module
    fig_root = ROOT / "results" / "figures" / "phase1-autoregressive-1.9-module-optimization" / module
    fig_root.mkdir(parents=True, exist_ok=True)

    rows = load_rows()
    baseline = next(row for row in rows if row["name"] == "baseline_v2")
    protocol = build_protocol_metadata()

    window_summaries = {}
    rollout_values = {}

    for window in WINDOWS:
        window_dir = log_root / window
        tf_valid = load_json(window_dir / "eval_teacher_forced_best_valid.json")
        tf_test = load_json(window_dir / "eval_teacher_forced_best_test.json")
        rollout_valid = load_json(window_dir / "eval_rollout_best_valid.json")
        rollout_test = load_json(window_dir / "eval_rollout_best_test.json")

        if tf_valid is None or tf_test is None:
            raise FileNotFoundError(
                f"Missing best-ckpt teacher-forced eval jsons for {module} {window}: {window_dir}"
            )

        best_ckpt = _checkpoint_from_payload(tf_valid, tf_test, rollout_valid, rollout_test)
        if best_ckpt is None:
            raise FileNotFoundError(f"Missing checkpoint path in eval jsons for {module} {window}")

        summary = checkpoint_summary(best_ckpt)
        last_ckpt = summary.get("last_alias_path") if summary else str(Path(best_ckpt).parent / "last.ckpt")
        metrics_path = Path(best_ckpt).parent.parent / "metrics.csv"
        curve_metrics = extract_curve_metrics(metrics_path) if metrics_path.exists() else {}

        baseline_key = f"fp_bps_{window}"
        baseline_fp = float(baseline[baseline_key]) if baseline[baseline_key] != "-" else None
        rollout_valid_fp = _continuous_fp(rollout_valid)
        rollout_values[window] = rollout_valid_fp

        window_summaries[window] = {
            "baseline_fp_bps": baseline_fp,
            "protocol": protocol,
            "curve_metrics": curve_metrics,
            "checkpoint_selection": {
                "rule": protocol["best_checkpoint_rule"],
                "best_checkpoint": best_ckpt,
                "best_checkpoint_epoch": tf_valid.get("checkpoint_epoch"),
                "best_checkpoint_kind": tf_valid.get("checkpoint_kind"),
                "last_checkpoint": last_ckpt,
                "last_checkpoint_epoch": checkpoint_epoch(last_ckpt),
                "historical_note": (
                    "Historical 1.9 runs only preserved one monitored checkpoint file plus last.ckpt; "
                    "best-ckpt backfill therefore uses the available monitored checkpoint."
                ),
            },
            "teacher_forced": {
                "valid": tf_valid,
                "test": tf_test,
            },
            "rollout": {
                "valid": rollout_valid,
                "test": rollout_test,
            },
            "rollout_delta_vs_baseline": None if baseline_fp is None or rollout_valid_fp is None else rollout_valid_fp - baseline_fp,
        }

    summary = {
        "module": module,
        "commit": current_commit(),
        "protocol": protocol,
        "baseline_v2": baseline,
        "windows": window_summaries,
    }
    summary_path = fig_root / summary_filename
    summary_path.write_text(json.dumps(summary, indent=2))

    new_row = {
        "name": module,
        "commit": summary["commit"],
        "date": str(date.today()),
        "description": description,
        "gpu_memory_gb": "-",
        "train_time_hours": "-",
        "notes": notes,
    }
    for field in LEGACY_FP_COLUMNS + BEST_VAL_COLUMNS + BEST_TEST_COLUMNS + BEST_CKPT_COLUMNS:
        new_row[field] = "-"

    for window in WINDOWS:
        tf_valid = window_summaries[window]["teacher_forced"]["valid"]
        tf_test = window_summaries[window]["teacher_forced"]["test"]
        best_ckpt = window_summaries[window]["checkpoint_selection"]["best_checkpoint"]
        rollout_valid = window_summaries[window]["rollout"]["valid"]
        suffix = window
        new_row[f"fp_bps_{suffix}"] = format_value(_continuous_fp(rollout_valid))
        new_row[f"best_val_fp_bps_{suffix}"] = format_value(_continuous_fp(tf_valid))
        new_row[f"best_test_fp_bps_{suffix}"] = format_value(_continuous_fp(tf_test))
        new_row[f"best_ckpt_{suffix}"] = format_checkpoint_id(best_ckpt)

    updated = False
    for idx, row in enumerate(rows):
        if row["name"] == module:
            rows[idx] = new_row
            updated = True
            break
    if not updated:
        rows.append(new_row)
    save_rows(rows)

    subprocess.run(
        ["/root/miniconda3/envs/poyo/bin/python", str(PLOT_SCRIPT)],
        check=True,
    )
    subprocess.run(
        ["/root/miniconda3/envs/poyo/bin/python", str(CURVE_PLOT_SCRIPT), "--module", module],
        check=True,
    )

    return summary_path
