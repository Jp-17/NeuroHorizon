#!/usr/bin/env python3
"""Summarize fp-bps and IBL-MtM-style bps for causal evalfix baselines."""

import json
from pathlib import Path


ROOT = Path("/root/autodl-tmp/NeuroHorizon/results/logs")
CONDITIONS = [
    ("250ms-cont", "phase1_v2_evalfix_250ms_cont"),
    ("500ms-cont", "phase1_v2_evalfix_500ms_cont"),
    ("1000ms-cont", "phase1_v2_evalfix_1000ms_cont"),
]


def load_json(path: Path):
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def load_eval_result(log_dir: str, split: str):
    return load_json(ROOT / log_dir / "lightning_logs" / "version_0" / f"eval_v2_{split}_results.json")


def extract_metrics(payload):
    if payload is None:
        return None
    continuous = payload.get("continuous", {})
    return {
        "fp_bps": continuous.get("fp_bps"),
        "ibl_mtm_bps": continuous.get("ibl_mtm_bps"),
        "r2": continuous.get("r2"),
    }


def main():
    out_dir = ROOT / "phase1_v2_metric_extension_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for label, log_dir in CONDITIONS:
        rows.append(
            {
                "label": label,
                "valid": extract_metrics(load_eval_result(log_dir, "valid")),
                "test": extract_metrics(load_eval_result(log_dir, "test")),
            }
        )

    with (out_dir / "comparison.json").open("w") as f:
        json.dump(rows, f, indent=2)

    lines = [
        "# Phase 1.3.5 metric extension comparison",
        "",
        "| Condition | Valid fp-bps | Valid ibl_mtm_bps | Valid R2 | Test fp-bps | Test ibl_mtm_bps | Test R2 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        valid = row["valid"] or {}
        test = row["test"] or {}
        lines.append(
            f"| {row['label']} | "
            f"{valid.get('fp_bps', 'NA')} | "
            f"{valid.get('ibl_mtm_bps', 'NA')} | "
            f"{valid.get('r2', 'NA')} | "
            f"{test.get('fp_bps', 'NA')} | "
            f"{test.get('ibl_mtm_bps', 'NA')} | "
            f"{test.get('r2', 'NA')} |"
        )

    with (out_dir / "comparison.md").open("w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Wrote {out_dir / 'comparison.json'}")
    print(f"Wrote {out_dir / 'comparison.md'}")


if __name__ == "__main__":
    main()
