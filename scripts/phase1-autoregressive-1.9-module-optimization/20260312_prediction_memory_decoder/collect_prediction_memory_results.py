#!/usr/bin/env python3
"""Collect, summarize, and register structured prediction-memory results."""

from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from module_result_utils import collect_module_results


def main():
    summary_path = collect_module_results(
        module="20260312_prediction_memory_decoder",
        description="structured prediction memory decoder",
        summary_filename="prediction_memory_summary.json",
        notes=(
            "obs=500ms; cont sampling; 10 sessions; 300 epochs; "
            "results.tsv stores rollout-valid fp-bps plus best-ckpt teacher-forced valid/test fp-bps"
        ),
    )
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
