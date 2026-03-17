#!/usr/bin/env python3
"""Compare legacy Phase 1 v2 results against evalfix reruns."""

import json
from pathlib import Path


ROOT = Path("/root/autodl-tmp/NeuroHorizon/results/logs")
CONDITIONS = [
    ("250ms-cont", "phase1_v2_250ms_cont", "phase1_v2_evalfix_250ms_cont"),
    ("250ms-trial", "phase1_v2_250ms_trial", "phase1_v2_evalfix_250ms_trial"),
    ("500ms-cont", "phase1_v2_500ms_cont", "phase1_v2_evalfix_500ms_cont"),
    ("500ms-trial", "phase1_v2_500ms_trial", "phase1_v2_evalfix_500ms_trial"),
    ("1000ms-cont", "phase1_v2_1000ms_cont", "phase1_v2_evalfix_1000ms_cont"),
    ("1000ms-trial", "phase1_v2_1000ms_trial", "phase1_v2_evalfix_1000ms_trial"),
]


def load_json(path: Path):
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def trial_metric(trial_section):
    if not trial_section:
        return None
    return trial_section.get(
        "per_neuron_psth_r2",
        trial_section.get("psth_r2"),
    )


def build_entry(label, legacy_dir, evalfix_dir):
    legacy = load_json(ROOT / legacy_dir / "lightning_logs" / "eval_v2_results.json")
    valid = load_json(ROOT / evalfix_dir / "lightning_logs" / "eval_v2_valid_results.json")
    test = load_json(ROOT / evalfix_dir / "lightning_logs" / "eval_v2_test_results.json")

    if legacy is None and valid is None and test is None:
        return None

    def extract(section):
        if section is None:
            return None
        return {
            "fp_bps": section.get("continuous", {}).get("fp_bps"),
            "r2": section.get("continuous", {}).get("r2"),
            "trial_fp_bps": section.get("trial_aligned", {}).get("trial_fp_bps"),
            "per_neuron_psth_r2": trial_metric(section.get("trial_aligned")),
        }

    return {
        "label": label,
        "legacy_valid": extract(legacy),
        "evalfix_valid": extract(valid),
        "evalfix_test": extract(test),
    }


def main():
    out_dir = ROOT / "phase1_v2_evalfix_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [entry for entry in (build_entry(*item) for item in CONDITIONS) if entry is not None]

    with (out_dir / "comparison.json").open("w") as f:
        json.dump(rows, f, indent=2)

    lines = [
        "# Phase 1 v2 evalfix comparison",
        "",
        "| Condition | Legacy valid fp-bps | Evalfix valid fp-bps | Evalfix test fp-bps | Legacy valid per-neuron PSTH-R2 | Evalfix valid per-neuron PSTH-R2 | Evalfix test per-neuron PSTH-R2 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        legacy = row["legacy_valid"] or {}
        valid = row["evalfix_valid"] or {}
        test = row["evalfix_test"] or {}
        lines.append(
            f"| {row['label']} | "
            f"{legacy.get('fp_bps', 'NA')} | "
            f"{valid.get('fp_bps', 'NA')} | "
            f"{test.get('fp_bps', 'NA')} | "
            f"{legacy.get('per_neuron_psth_r2', 'NA')} | "
            f"{valid.get('per_neuron_psth_r2', 'NA')} | "
            f"{test.get('per_neuron_psth_r2', 'NA')} |"
        )

    with (out_dir / "comparison.md").open("w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Wrote {out_dir / 'comparison.json'}")
    print(f"Wrote {out_dir / 'comparison.md'}")


if __name__ == "__main__":
    main()
