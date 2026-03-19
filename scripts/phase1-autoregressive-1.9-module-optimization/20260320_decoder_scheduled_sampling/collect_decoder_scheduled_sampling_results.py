#!/usr/bin/env python3
"""Collect decoder scheduled sampling results into summary json and results.tsv."""

from __future__ import annotations

import json
import importlib.util
import subprocess
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
UTILS_PATH = ROOT / "scripts" / "phase1-autoregressive-1.9-module-optimization" / "module_result_utils.py"
UTILS_SPEC = importlib.util.spec_from_file_location("module_result_utils", UTILS_PATH)
if UTILS_SPEC is None or UTILS_SPEC.loader is None:
    raise RuntimeError(f"Failed to load module_result_utils from {UTILS_PATH}")
module_result_utils = importlib.util.module_from_spec(UTILS_SPEC)
UTILS_SPEC.loader.exec_module(module_result_utils)

FIELDNAMES = module_result_utils.FIELDNAMES
PLOT_SCRIPT = module_result_utils.PLOT_SCRIPT
WINDOWS = module_result_utils.WINDOWS
_continuous_fp = module_result_utils._continuous_fp
build_protocol_metadata = module_result_utils.build_protocol_metadata
current_commit = module_result_utils.current_commit
format_checkpoint_id = module_result_utils.format_checkpoint_id
format_value = module_result_utils.format_value
load_json = module_result_utils.load_json
load_rows = module_result_utils.load_rows
save_rows = module_result_utils.save_rows


MODULE = "20260320_decoder_scheduled_sampling"
LOG_ROOT = ROOT / "results" / "logs" / "phase1-autoregressive-1.9-module-optimization" / MODULE
FIG_ROOT = ROOT / "results" / "figures" / "phase1-autoregressive-1.9-module-optimization" / MODULE
SUMMARY_PATH = FIG_ROOT / "decoder_scheduled_sampling_matrix_summary.json"
SETTING_DESCRIPTIONS = {
    "memory_only_mix035": "memory-only control with mix_prob=0.35",
    "decoder_ss_fixed_025": "decoder scheduled sampling fixed rollout_prob=0.25",
    "decoder_ss_fixed_050": "decoder scheduled sampling fixed rollout_prob=0.50",
    "decoder_ss_fixed_075": "decoder scheduled sampling fixed rollout_prob=0.75",
    "decoder_ss_linear_0_to_050": "decoder scheduled sampling linear rollout_prob 0.0->0.50",
    "decoder_ss_linear_0_to_075": "decoder scheduled sampling linear rollout_prob 0.0->0.75",
    "hybrid_mix035_plus_linear_050": "hybrid mix_prob=0.35 plus decoder scheduled sampling linear 0.0->0.50",
}


def load_setting_window_payload(setting: str, window: str) -> dict | None:
    window_dir = LOG_ROOT / setting / window
    tf_valid = load_json(window_dir / "eval_teacher_forced_best_valid.json")
    tf_test = load_json(window_dir / "eval_teacher_forced_best_test.json")
    rollout_valid = load_json(window_dir / "eval_rollout_best_valid.json")
    rollout_test = load_json(window_dir / "eval_rollout_best_test.json")
    if not all([tf_valid, tf_test, rollout_valid, rollout_test]):
        return None

    best_ckpt = (
        tf_valid.get("checkpoint")
        or tf_test.get("checkpoint")
        or rollout_valid.get("checkpoint")
        or rollout_test.get("checkpoint")
    )
    return {
        "teacher_forced": {
            "valid": tf_valid,
            "test": tf_test,
        },
        "rollout": {
            "valid": rollout_valid,
            "test": rollout_test,
        },
        "best_checkpoint": best_ckpt,
    }


def build_row(setting: str, payload: dict) -> dict:
    row = {field: "-" for field in FIELDNAMES}
    row["name"] = f"{MODULE}__{setting}"
    row["commit"] = current_commit()
    row["date"] = str(date.today())
    row["description"] = SETTING_DESCRIPTIONS[setting]
    row["gpu_memory_gb"] = "-"
    row["train_time_hours"] = "-"
    row["notes"] = "obs=500ms; cont sampling; 10 sessions; three-window decoder scheduled sampling matrix"

    for window in WINDOWS:
        window_payload = payload["windows"].get(window)
        if window_payload is None:
            continue
        tf_valid = window_payload["teacher_forced"]["valid"]
        tf_test = window_payload["teacher_forced"]["test"]
        rollout_valid = window_payload["rollout"]["valid"]
        row[f"fp_bps_{window}"] = format_value(_continuous_fp(rollout_valid))
        row[f"best_val_fp_bps_{window}"] = format_value(_continuous_fp(tf_valid))
        row[f"best_test_fp_bps_{window}"] = format_value(_continuous_fp(tf_test))
        row[f"best_ckpt_{window}"] = format_checkpoint_id(window_payload["best_checkpoint"])
    return row


def main():
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    protocol = build_protocol_metadata()
    rows = load_rows()
    summary = {
        "module": MODULE,
        "commit": current_commit(),
        "protocol": protocol,
        "settings": {},
    }

    updated_names = set()
    for setting in sorted(SETTING_DESCRIPTIONS):
        setting_summary = {
            "description": SETTING_DESCRIPTIONS[setting],
            "windows": {},
            "status": "complete",
        }
        complete = True
        for window in WINDOWS:
            window_payload = load_setting_window_payload(setting, window)
            if window_payload is None:
                complete = False
                continue
            setting_summary["windows"][window] = window_payload

        if not complete:
            setting_summary["status"] = "incomplete"
            summary["settings"][setting] = setting_summary
            continue

        summary["settings"][setting] = setting_summary
        new_row = build_row(setting, setting_summary)
        updated = False
        for idx, row in enumerate(rows):
            if row["name"] == new_row["name"]:
                rows[idx] = new_row
                updated = True
                break
        if not updated:
            rows.append(new_row)
        updated_names.add(new_row["name"])

    save_rows(rows)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    subprocess.run(["/root/miniconda3/envs/poyo/bin/python", str(PLOT_SCRIPT)], check=True)
    print(f"saved summary: {SUMMARY_PATH}")
    if updated_names:
        print("updated rows:")
        for name in sorted(updated_names):
            print(f"  - {name}")
    else:
        print("no complete settings found; summary written without TSV row updates")


if __name__ == "__main__":
    main()
