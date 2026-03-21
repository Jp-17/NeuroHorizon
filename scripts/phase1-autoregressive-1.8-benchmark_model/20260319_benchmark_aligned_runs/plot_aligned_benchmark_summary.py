#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot aligned benchmark summary figures")
    parser.add_argument("--ibl-compare-json", required=True)
    parser.add_argument("--neuro-compare-json", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    ibl = load_json(Path(args.ibl_compare_json))
    neuro = load_json(Path(args.neuro_compare_json))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "ibl_mtm": {
            "e10_best_valid_fp_bps": ibl["baseline"].get("best_valid_fp_bps"),
            "e10_test_fp_bps": ibl["baseline"].get("test_fp_bps"),
            "e50_best_valid_fp_bps": ibl["control"].get("best_valid_fp_bps"),
            "e50_test_fp_bps": ibl["control"].get("test_fp_bps"),
            "delta_test_fp_bps": ibl.get("delta_test_fp_bps"),
        },
        "neuroformer": neuro,
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    width = 0.35
    x = [0, 1]

    ax = axes[0, 0]
    best_vals = [summary["ibl_mtm"]["e10_best_valid_fp_bps"], summary["ibl_mtm"]["e50_best_valid_fp_bps"]]
    test_vals = [summary["ibl_mtm"]["e10_test_fp_bps"], summary["ibl_mtm"]["e50_test_fp_bps"]]
    ax.bar([i - width / 2 for i in x], best_vals, width=width, label="best valid fp-bps", color="#6baed6")
    ax.bar([i + width / 2 for i in x], test_vals, width=width, label="test fp-bps", color="#2171b5")
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(["IBL e10", "IBL e50"])
    ax.set_title("IBL-MtM 250ms Combined")
    ax.set_ylabel("fp-bps")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    rollout = [neuro["canonical"]["modes"]["rollout"]["fp_bps"], neuro["reference"]["modes"]["rollout"]["fp_bps"]]
    true_past = [neuro["canonical"]["modes"]["true_past"]["fp_bps"], neuro["reference"]["modes"]["true_past"]["fp_bps"]]
    ax.bar([i - width / 2 for i in x], rollout, width=width, label="rollout", color="#fd8d3c")
    ax.bar([i + width / 2 for i in x], true_past, width=width, label="true_past", color="#e6550d")
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(["500/250", "150/50"])
    ax.set_title("Neuroformer Test fp-bps")
    ax.set_ylabel("fp-bps")
    ax.legend(frameon=False)

    ax = axes[1, 0]
    rollout_t = [neuro["canonical"]["modes"]["rollout"]["elapsed_s"], neuro["reference"]["modes"]["rollout"]["elapsed_s"]]
    true_past_t = [neuro["canonical"]["modes"]["true_past"]["elapsed_s"], neuro["reference"]["modes"]["true_past"]["elapsed_s"]]
    ax.bar([i - width / 2 for i in x], rollout_t, width=width, label="rollout", color="#74c476")
    ax.bar([i + width / 2 for i in x], true_past_t, width=width, label="true_past", color="#238b45")
    ax.set_xticks(x)
    ax.set_xticklabels(["500/250", "150/50"])
    ax.set_title("Neuroformer Test Eval Runtime")
    ax.set_ylabel("elapsed_s")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    prev_mean = [neuro["canonical"]["token_stats"]["prev_tokens_mean"], neuro["reference"]["token_stats"]["prev_tokens_mean"]]
    curr_mean = [neuro["canonical"]["token_stats"]["curr_tokens_mean"], neuro["reference"]["token_stats"]["curr_tokens_mean"]]
    ax.bar([i - width / 2 for i in x], prev_mean, width=width, label="prev mean tokens", color="#9ecae1")
    ax.bar([i + width / 2 for i in x], curr_mean, width=width, label="curr mean tokens", color="#3182bd")
    ax.set_xticks(x)
    ax.set_xticklabels(["500/250", "150/50"])
    ax.set_title("Neuroformer Test Token Stats")
    ax.set_ylabel("mean tokens per sample")
    ax.legend(frameon=False)

    fig.suptitle("Benchmark Aligned Summary (20260319)")
    fig.tight_layout()
    fig.savefig(out_dir / "aligned_benchmark_summary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    md = """# Benchmark Aligned Summary

## IBL-MtM

- e10 best valid fp-bps: {e10_best}
- e10 test fp-bps: {e10_test}
- e50 best valid fp-bps: {e50_best}
- e50 test fp-bps: {e50_test}
- delta test fp-bps: {delta}

## Neuroformer test comparison

| setting | mode | fp-bps | r2 | elapsed_s |
| --- | --- | ---: | ---: | ---: |
| canonical 500/250 | rollout | {c_rollout_fp} | {c_rollout_r2} | {c_rollout_t} |
| canonical 500/250 | true_past | {c_true_fp} | {c_true_r2} | {c_true_t} |
| reference 150/50 | rollout | {r_rollout_fp} | {r_rollout_r2} | {r_rollout_t} |
| reference 150/50 | true_past | {r_true_fp} | {r_true_r2} | {r_true_t} |

## Figure

- aligned_benchmark_summary.png
""".format(
        e10_best=summary["ibl_mtm"]["e10_best_valid_fp_bps"],
        e10_test=summary["ibl_mtm"]["e10_test_fp_bps"],
        e50_best=summary["ibl_mtm"]["e50_best_valid_fp_bps"],
        e50_test=summary["ibl_mtm"]["e50_test_fp_bps"],
        delta=summary["ibl_mtm"]["delta_test_fp_bps"],
        c_rollout_fp=neuro["canonical"]["modes"]["rollout"]["fp_bps"],
        c_rollout_r2=neuro["canonical"]["modes"]["rollout"]["r2"],
        c_rollout_t=neuro["canonical"]["modes"]["rollout"]["elapsed_s"],
        c_true_fp=neuro["canonical"]["modes"]["true_past"]["fp_bps"],
        c_true_r2=neuro["canonical"]["modes"]["true_past"]["r2"],
        c_true_t=neuro["canonical"]["modes"]["true_past"]["elapsed_s"],
        r_rollout_fp=neuro["reference"]["modes"]["rollout"]["fp_bps"],
        r_rollout_r2=neuro["reference"]["modes"]["rollout"]["r2"],
        r_rollout_t=neuro["reference"]["modes"]["rollout"]["elapsed_s"],
        r_true_fp=neuro["reference"]["modes"]["true_past"]["fp_bps"],
        r_true_r2=neuro["reference"]["modes"]["true_past"]["r2"],
        r_true_t=neuro["reference"]["modes"]["true_past"]["elapsed_s"],
    )

    (out_dir / "aligned_benchmark_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out_dir / "aligned_benchmark_summary.md").write_text(md)


if __name__ == "__main__":
    main()
