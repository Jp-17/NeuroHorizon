#!/usr/bin/env python3
"""Backfill best-ckpt valid/test eval JSONs for historical Phase 1.9 runs."""

from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path("/root/autodl-tmp/NeuroHorizon")
PYTHON = "/root/miniconda3/envs/poyo/bin/python"
EVAL_SCRIPT = ROOT / "scripts" / "analysis" / "neurohorizon" / "eval_phase1_v2.py"
MODULES = {
    "20260312_prediction_memory_decoder": ROOT
    / "scripts"
    / "phase1-autoregressive-1.9-module-optimization"
    / "20260312_prediction_memory_decoder"
    / "collect_prediction_memory_results.py",
    "20260313_local_prediction_memory": ROOT
    / "scripts"
    / "phase1-autoregressive-1.9-module-optimization"
    / "20260313_local_prediction_memory"
    / "collect_local_prediction_memory_results.py",
    "20260313_prediction_memory_alignment": ROOT
    / "scripts"
    / "phase1-autoregressive-1.9-module-optimization"
    / "20260313_prediction_memory_alignment"
    / "collect_prediction_memory_alignment_results.py",
    "20260313_prediction_memory_alignment_tuning": ROOT
    / "scripts"
    / "phase1-autoregressive-1.9-module-optimization"
    / "20260313_prediction_memory_alignment_tuning"
    / "collect_prediction_memory_alignment_tuning_results.py",
}
WINDOWS = ["250ms", "500ms", "1000ms"]


def run_eval(window_dir: Path, *, split: str, rollout: bool) -> None:
    output_name = (
        f"eval_rollout_best_{split}.json"
        if rollout
        else f"eval_teacher_forced_best_{split}.json"
    )
    cmd = [
        PYTHON,
        str(EVAL_SCRIPT),
        "--log-dir",
        str(window_dir),
        "--checkpoint-kind",
        "best",
        "--split",
        split,
        "--skip-trial",
        "--output",
        str(window_dir / output_name),
    ]
    if rollout:
        cmd.append("--rollout")
    subprocess.run(cmd, check=True)


def main():
    log_root = ROOT / "results" / "logs" / "phase1-autoregressive-1.9-module-optimization"
    for module, collect_script in MODULES.items():
        for window in WINDOWS:
            window_dir = log_root / module / window
            for split in ["valid", "test"]:
                run_eval(window_dir, split=split, rollout=False)
                run_eval(window_dir, split=split, rollout=True)

        subprocess.run([PYTHON, str(collect_script)], check=True)


if __name__ == "__main__":
    main()
