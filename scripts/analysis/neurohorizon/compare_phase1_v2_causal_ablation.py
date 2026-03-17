#!/usr/bin/env python3
"""Compare causal evalfix baselines against non-causal ablations."""

import json
from pathlib import Path


ROOT = Path("/root/autodl-tmp/NeuroHorizon/results/logs")
CONDITIONS = [
    ("250ms", "phase1_v2_evalfix_250ms_cont", "phase1_v2_nocausal_250ms_cont"),
    ("500ms", "phase1_v2_evalfix_500ms_cont", "phase1_v2_nocausal_500ms_cont"),
    ("1000ms", "phase1_v2_evalfix_1000ms_cont", "phase1_v2_nocausal_1000ms_cont"),
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
    trial = payload.get("trial_aligned", {})
    return {
        "fp_bps": continuous.get("fp_bps"),
        "ibl_mtm_bps": continuous.get("ibl_mtm_bps"),
        "r2": continuous.get("r2"),
        "per_neuron_psth_r2": trial.get("per_neuron_psth_r2"),
        "trial_fp_bps": trial.get("trial_fp_bps"),
    }


def delta(new_val, old_val):
    if new_val is None or old_val is None:
        return None
    return new_val - old_val


def main():
    out_dir = ROOT / "phase1_v2_nocausal_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for label, causal_dir, nocausal_dir in CONDITIONS:
        causal_valid = extract_metrics(load_eval_result(causal_dir, "valid"))
        causal_test = extract_metrics(load_eval_result(causal_dir, "test"))
        nocausal_valid = extract_metrics(load_eval_result(nocausal_dir, "valid"))
        nocausal_test = extract_metrics(load_eval_result(nocausal_dir, "test"))
        rows.append(
            {
                "label": label,
                "causal_valid": causal_valid,
                "causal_test": causal_test,
                "nocausal_valid": nocausal_valid,
                "nocausal_test": nocausal_test,
                "valid_delta_fp_bps": delta(
                    None if nocausal_valid is None else nocausal_valid.get("fp_bps"),
                    None if causal_valid is None else causal_valid.get("fp_bps"),
                ),
                "valid_delta_ibl_mtm_bps": delta(
                    None if nocausal_valid is None else nocausal_valid.get("ibl_mtm_bps"),
                    None if causal_valid is None else causal_valid.get("ibl_mtm_bps"),
                ),
                "test_delta_fp_bps": delta(
                    None if nocausal_test is None else nocausal_test.get("fp_bps"),
                    None if causal_test is None else causal_test.get("fp_bps"),
                ),
                "test_delta_ibl_mtm_bps": delta(
                    None if nocausal_test is None else nocausal_test.get("ibl_mtm_bps"),
                    None if causal_test is None else causal_test.get("ibl_mtm_bps"),
                ),
            }
        )

    with (out_dir / "comparison.json").open("w") as f:
        json.dump(rows, f, indent=2)

    lines = [
        "# Phase 1.3.6 causal vs non-causal comparison",
        "",
        "| Window | Causal valid fp-bps | Non-causal valid fp-bps | Delta | Causal valid ibl_mtm_bps | Non-causal valid ibl_mtm_bps | Delta | Causal test fp-bps | Non-causal test fp-bps | Delta | Causal test ibl_mtm_bps | Non-causal test ibl_mtm_bps | Delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        cv = row["causal_valid"] or {}
        nv = row["nocausal_valid"] or {}
        ct = row["causal_test"] or {}
        nt = row["nocausal_test"] or {}
        lines.append(
            f"| {row['label']} | "
            f"{cv.get('fp_bps', 'NA')} | {nv.get('fp_bps', 'NA')} | {row.get('valid_delta_fp_bps', 'NA')} | "
            f"{cv.get('ibl_mtm_bps', 'NA')} | {nv.get('ibl_mtm_bps', 'NA')} | {row.get('valid_delta_ibl_mtm_bps', 'NA')} | "
            f"{ct.get('fp_bps', 'NA')} | {nt.get('fp_bps', 'NA')} | {row.get('test_delta_fp_bps', 'NA')} | "
            f"{ct.get('ibl_mtm_bps', 'NA')} | {nt.get('ibl_mtm_bps', 'NA')} | {row.get('test_delta_ibl_mtm_bps', 'NA')} |"
        )

    with (out_dir / "comparison.md").open("w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Wrote {out_dir / 'comparison.json'}")
    print(f"Wrote {out_dir / 'comparison.md'}")


if __name__ == "__main__":
    main()
