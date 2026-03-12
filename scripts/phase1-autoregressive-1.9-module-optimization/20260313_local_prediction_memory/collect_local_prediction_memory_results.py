#!/usr/bin/env python3
"""Collect, summarize, and register local-prediction-memory experiment results."""

from __future__ import annotations

import csv
import json
import subprocess
from datetime import date
from pathlib import Path


ROOT = Path("/root/autodl-tmp/NeuroHorizon")
MODULE = "20260313_local_prediction_memory"
LOG_ROOT = ROOT / "results" / "logs" / "phase1-autoregressive-1.9-module-optimization" / MODULE
FIG_ROOT = ROOT / "results" / "figures" / "phase1-autoregressive-1.9-module-optimization" / MODULE
TSV_PATH = ROOT / "cc_todo" / "phase1-autoregressive" / "1.9-module-optimization" / "results.tsv"
PLOT_SCRIPT = ROOT / "cc_todo" / "phase1-autoregressive" / "1.9-module-optimization" / "plot_optimization_progress.py"
FIELDNAMES = [
    "name",
    "commit",
    "date",
    "description",
    "fp_bps_250ms",
    "fp_bps_500ms",
    "fp_bps_1000ms",
    "gpu_memory_gb",
    "train_time_hours",
    "notes",
]


def load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def load_rows():
    with TSV_PATH.open() as f:
        return list(csv.DictReader(f, delimiter="\t"))


def save_rows(rows):
    with TSV_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def current_commit():
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
    return f"{value:.4f}"


def main():
    FIG_ROOT.mkdir(parents=True, exist_ok=True)

    rows = load_rows()
    baseline = next(row for row in rows if row["name"] == "baseline_v2")

    window_summaries = {}
    rollout_values = {}

    for window in ["250ms", "500ms", "1000ms"]:
        window_dir = LOG_ROOT / window
        tf_result = load_json(window_dir / "eval_teacher_forced.json")
        rollout_result = load_json(window_dir / "eval_rollout.json")
        if rollout_result is None or "continuous" not in rollout_result:
            raise FileNotFoundError(f"Missing rollout result for {window}: {window_dir / 'eval_rollout.json'}")

        baseline_key = f"fp_bps_{window}"
        baseline_fp = float(baseline[baseline_key]) if baseline[baseline_key] != "-" else None
        rollout_fp = rollout_result["continuous"]["fp_bps"]
        rollout_values[window] = rollout_fp
        window_summaries[window] = {
            "baseline_fp_bps": baseline_fp,
            "teacher_forced": tf_result,
            "rollout": rollout_result,
            "rollout_delta_vs_baseline": None if baseline_fp is None else rollout_fp - baseline_fp,
        }

    summary = {
        "module": MODULE,
        "commit": current_commit(),
        "baseline_v2": baseline,
        "windows": window_summaries,
    }
    summary_path = FIG_ROOT / "local_prediction_memory_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    new_row = {
        "name": MODULE,
        "commit": summary["commit"],
        "date": str(date.today()),
        "description": "local prediction memory decoder",
        "fp_bps_250ms": format_value(rollout_values["250ms"]),
        "fp_bps_500ms": format_value(rollout_values["500ms"]),
        "fp_bps_1000ms": format_value(rollout_values["1000ms"]),
        "gpu_memory_gb": "-",
        "train_time_hours": "-",
        "notes": "obs=500ms; cont sampling; 10 sessions; 300 epochs; rollout eval",
    }

    updated = False
    for i, row in enumerate(rows):
        if row["name"] == MODULE:
            rows[i] = new_row
            updated = True
            break
    if not updated:
        rows.append(new_row)
    save_rows(rows)

    subprocess.run(
        ["/root/miniconda3/envs/poyo/bin/python", str(PLOT_SCRIPT)],
        check=True,
    )

    print(f"Saved summary to {summary_path}")
    print(f"Updated TSV row for {MODULE}")
    for window, result in window_summaries.items():
        fp_bps = result["rollout"]["continuous"]["fp_bps"]
        delta = result["rollout_delta_vs_baseline"]
        print(f"{window}: rollout fp-bps={fp_bps:.4f}, delta_vs_baseline={delta:+.4f}")


if __name__ == "__main__":
    main()
