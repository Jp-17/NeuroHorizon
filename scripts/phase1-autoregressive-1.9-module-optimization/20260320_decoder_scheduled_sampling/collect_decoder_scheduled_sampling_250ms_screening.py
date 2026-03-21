#!/usr/bin/env python3
"""Collect and rank 250ms decoder scheduled sampling screening results."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MODULE = "20260320_decoder_scheduled_sampling"
WINDOW = "250ms"
LOG_ROOT = ROOT / "results" / "logs" / "phase1-autoregressive-1.9-module-optimization" / MODULE
FIG_ROOT = ROOT / "results" / "figures" / "phase1-autoregressive-1.9-module-optimization" / MODULE
SUMMARY_JSON = FIG_ROOT / "decoder_scheduled_sampling_250ms_screening_summary.json"
SUMMARY_MD = FIG_ROOT / "decoder_scheduled_sampling_250ms_screening_summary.md"
SUMMARY_TSV = FIG_ROOT / "decoder_scheduled_sampling_250ms_screening_summary.tsv"

SETTINGS = {
    "memory_only_mix035": "memory-only control with mix_prob=0.35",
    "decoder_ss_fixed_025": "decoder scheduled sampling fixed rollout_prob=0.25",
    "decoder_ss_fixed_050": "decoder scheduled sampling fixed rollout_prob=0.50",
    "decoder_ss_fixed_075": "decoder scheduled sampling fixed rollout_prob=0.75",
    "decoder_ss_linear_0_to_050": "decoder scheduled sampling linear rollout_prob 0.0->0.50",
    "decoder_ss_linear_0_to_075": "decoder scheduled sampling linear rollout_prob 0.0->0.75",
    "hybrid_mix035_plus_linear_050": "hybrid mix_prob=0.35 plus decoder scheduled sampling linear 0.0->0.50",
}


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open() as handle:
        return json.load(handle)


def continuous_fp(payload: dict) -> float:
    return float(payload["continuous"]["fp_bps"])


def extract_setting(setting: str) -> dict | None:
    window_dir = LOG_ROOT / setting / WINDOW
    tf_valid = load_json(window_dir / "eval_teacher_forced_best_valid.json")
    tf_test = load_json(window_dir / "eval_teacher_forced_best_test.json")
    rollout_valid = load_json(window_dir / "eval_rollout_best_valid.json")
    rollout_test = load_json(window_dir / "eval_rollout_best_test.json")
    if not all([tf_valid, tf_test, rollout_valid, rollout_test]):
        return None

    tf_valid_fp = continuous_fp(tf_valid)
    tf_test_fp = continuous_fp(tf_test)
    rollout_valid_fp = continuous_fp(rollout_valid)
    rollout_test_fp = continuous_fp(rollout_test)

    return {
        "setting": setting,
        "description": SETTINGS[setting],
        "checkpoint": tf_test.get("checkpoint") or rollout_test.get("checkpoint"),
        "checkpoint_epoch": tf_test.get("checkpoint_epoch") or rollout_test.get("checkpoint_epoch"),
        "teacher_forced_valid_fp_bps": tf_valid_fp,
        "teacher_forced_test_fp_bps": tf_test_fp,
        "rollout_valid_fp_bps": rollout_valid_fp,
        "rollout_test_fp_bps": rollout_test_fp,
        "valid_gap": tf_valid_fp - rollout_valid_fp,
        "test_gap": tf_test_fp - rollout_test_fp,
    }


def write_markdown(complete_rows: list[dict], incomplete_settings: list[str]) -> None:
    lines = [
        "# Decoder Scheduled Sampling 250ms Screening Summary",
        "",
        f"Generated: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"Complete settings: {len(complete_rows)}/{len(SETTINGS)}",
        "",
        "## Ranking by rollout test fp_bps",
        "",
        "| rank | setting | rollout test | rollout valid | tf test | test gap | tf valid | valid gap | checkpoint epoch |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in complete_rows:
        lines.append(
            "| {rank} | `{setting}` | {rollout_test_fp_bps:.4f} | {rollout_valid_fp_bps:.4f} | "
            "{teacher_forced_test_fp_bps:.4f} | {test_gap:.4f} | {teacher_forced_valid_fp_bps:.4f} | "
            "{valid_gap:.4f} | {checkpoint_epoch} |".format(**row)
        )

    if complete_rows:
        best_rollout = complete_rows[0]
        best_gap = min(complete_rows, key=lambda row: row["test_gap"])
        lines.extend(
            [
                "",
                f"Best rollout test setting: `{best_rollout['setting']}` ({best_rollout['rollout_test_fp_bps']:.4f})",
                f"Smallest test gap setting: `{best_gap['setting']}` ({best_gap['test_gap']:.4f})",
            ]
        )

    if incomplete_settings:
        lines.extend(
            [
                "",
                "## Incomplete settings",
                "",
                *[f"- `{setting}`" for setting in incomplete_settings],
            ]
        )

    SUMMARY_MD.write_text("\n".join(lines) + "\n")


def write_tsv(rows: list[dict]) -> None:
    fieldnames = [
        "rank",
        "setting",
        "description",
        "rollout_test_fp_bps",
        "rollout_valid_fp_bps",
        "teacher_forced_test_fp_bps",
        "teacher_forced_valid_fp_bps",
        "test_gap",
        "valid_gap",
        "checkpoint_epoch",
        "checkpoint",
    ]
    with SUMMARY_TSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    complete_rows = []
    incomplete_settings = []

    for setting in SETTINGS:
        row = extract_setting(setting)
        if row is None:
            incomplete_settings.append(setting)
            continue
        complete_rows.append(row)

    complete_rows.sort(
        key=lambda row: (row["rollout_test_fp_bps"], row["rollout_valid_fp_bps"]),
        reverse=True,
    )
    for rank, row in enumerate(complete_rows, start=1):
        row["rank"] = rank

    summary = {
        "module": MODULE,
        "window": WINDOW,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "complete_settings": complete_rows,
        "incomplete_settings": incomplete_settings,
    }

    SUMMARY_JSON.write_text(json.dumps(summary, indent=2))
    write_markdown(complete_rows, incomplete_settings)
    write_tsv(complete_rows)
    print(f"saved summary: {SUMMARY_JSON}")
    print(f"saved markdown: {SUMMARY_MD}")
    print(f"saved tsv: {SUMMARY_TSV}")
    if complete_rows:
        best = complete_rows[0]
        print(f"best rollout test setting: {best['setting']} ({best['rollout_test_fp_bps']:.4f})")
    else:
        print("no complete 250ms settings found")


if __name__ == "__main__":
    main()
