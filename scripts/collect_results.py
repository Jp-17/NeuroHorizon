"""Collect and summarize all experiment results.

Scans training logs and evaluation results to generate a comprehensive
summary table for the paper. Handles:
- Multiple model variants (v1, v2, multimodal, ablations)
- Cross-session generalization results
- Horizon evaluation results
- Per-session breakdowns

Usage:
    conda run -n poyo python scripts/collect_results.py --output results/final_summary
"""

import argparse
import csv
import json
import logging
import math
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def find_best_metrics(metrics_csv):
    """Extract best metrics from a Lightning CSV log."""
    if not Path(metrics_csv).exists():
        return None

    train_losses = {}
    val_metrics = {}

    with open(metrics_csv) as f:
        for row in csv.DictReader(f):
            epoch = int(row["epoch"]) if row.get("epoch") and row["epoch"] else None
            if epoch is None:
                continue

            if row.get("train_loss") and row["train_loss"]:
                v = float(row["train_loss"])
                if not math.isnan(v):
                    train_losses.setdefault(epoch, []).append(v)

            for k in ["val_loss", "val_bits_per_spike", "val_r2",
                       "val_fr_corr", "val_co_bps"]:
                if row.get(k) and row[k]:
                    v = float(row[k])
                    if not math.isnan(v):
                        val_metrics.setdefault(k, {})[epoch] = v

    if not val_metrics:
        return None

    result = {
        "total_epochs": max(train_losses.keys()) + 1 if train_losses else 0,
        "final_train_loss": np.mean(train_losses.get(
            max(train_losses.keys()), [float("nan")]
        )),
    }

    if "val_loss" in val_metrics:
        best_epoch = min(val_metrics["val_loss"], key=val_metrics["val_loss"].get)
        result["best_val_loss"] = val_metrics["val_loss"][best_epoch]
        result["best_val_loss_epoch"] = best_epoch
        last_epoch = max(val_metrics["val_loss"].keys())
        result["final_val_loss"] = val_metrics["val_loss"][last_epoch]

    if "val_bits_per_spike" in val_metrics:
        best_epoch = max(val_metrics["val_bits_per_spike"],
                         key=val_metrics["val_bits_per_spike"].get)
        result["best_bps"] = val_metrics["val_bits_per_spike"][best_epoch]
        result["best_bps_epoch"] = best_epoch
        last_epoch = max(val_metrics["val_bits_per_spike"].keys())
        result["final_bps"] = val_metrics["val_bits_per_spike"][last_epoch]

    if "val_r2" in val_metrics:
        best_epoch = max(val_metrics["val_r2"], key=val_metrics["val_r2"].get)
        result["best_r2"] = val_metrics["val_r2"][best_epoch]
        result["best_r2_epoch"] = best_epoch

    if "val_fr_corr" in val_metrics:
        best_epoch = max(val_metrics["val_fr_corr"],
                         key=val_metrics["val_fr_corr"].get)
        result["best_fr_corr"] = val_metrics["val_fr_corr"][best_epoch]

    return result


def scan_training_logs(base_dir):
    """Find and load all training log directories."""
    base_dir = Path(base_dir)
    results = {}

    log_dirs = [
        ("NH v1 (unnormalized)", "logs/neurohorizon"),
        ("NH v2 (IBL, normalized)", "logs/neurohorizon_v2_ibl"),
        ("NH v2 (IBL+Allen, behavior)", "logs/neurohorizon_v2_beh"),
        ("NH v2 (multimodal)", "logs/neurohorizon_v2_mm"),
        ("POYO baseline", "logs/poyo_baseline"),
    ]

    # Add ablation logs
    ablation_dir = base_dir / "logs"
    if ablation_dir.exists():
        for d in sorted(ablation_dir.iterdir()):
            if d.is_dir() and d.name.startswith("ablation_"):
                name = d.name.replace("ablation_", "Ablation: ")
                log_dirs.append((name, str(d.relative_to(base_dir))))

    for name, rel_path in log_dirs:
        log_path = base_dir / rel_path
        if not log_path.exists():
            continue

        # Find the latest version
        lightning_dir = log_path / "lightning_logs"
        if not lightning_dir.exists():
            continue

        versions = sorted(lightning_dir.glob("version_*"),
                          key=lambda p: int(p.name.split("_")[1]))
        if not versions:
            continue

        latest = versions[-1]
        metrics_csv = latest / "metrics.csv"

        metrics = find_best_metrics(metrics_csv)
        if metrics:
            metrics["log_dir"] = str(latest)
            metrics["version"] = latest.name
            results[name] = metrics
            logger.info(f"Found {name}: {latest}")

    return results


def load_eval_results(base_dir):
    """Load post-training evaluation results."""
    base_dir = Path(base_dir)
    eval_results = {}

    # NH evaluation
    nh_eval = base_dir / "results" / "neurohorizon_eval" / "metrics.json"
    if nh_eval.exists():
        with open(nh_eval) as f:
            eval_results["NH eval"] = json.load(f)

    # POYO evaluation
    poyo_eval = base_dir / "results" / "poyo_eval" / "metrics.json"
    if poyo_eval.exists():
        with open(poyo_eval) as f:
            eval_results["POYO eval"] = json.load(f)

    # Cross-session results
    for p in sorted((base_dir / "results").glob("cross_session*.json")):
        with open(p) as f:
            eval_results[f"Cross-session: {p.stem}"] = json.load(f)

    # Horizon results
    horizon_eval = base_dir / "results" / "horizon_eval" / "horizon_results.json"
    if horizon_eval.exists():
        with open(horizon_eval) as f:
            eval_results["Horizon eval"] = json.load(f)

    return eval_results


def format_table(results):
    """Format results as a markdown table."""
    if not results:
        return "No results found."

    lines = []
    lines.append("## Training Results Summary\n")

    # Determine columns based on available metrics
    all_keys = set()
    for r in results.values():
        all_keys.update(r.keys())

    # Core columns
    cols = [
        ("Model", None),
        ("Epochs", "total_epochs"),
        ("Best val_loss", "best_val_loss"),
        ("Best BPS", "best_bps"),
        ("Best R²", "best_r2"),
        ("Final BPS", "final_bps"),
        ("Best epoch", "best_bps_epoch"),
    ]

    header = " | ".join(c[0] for c in cols)
    separator = " | ".join("-" * len(c[0]) for c in cols)
    lines.append(f"| {header} |")
    lines.append(f"| {separator} |")

    for name, metrics in results.items():
        values = [name]
        for col_name, key in cols[1:]:
            if key and key in metrics:
                v = metrics[key]
                if isinstance(v, float):
                    values.append(f"{v:.4f}")
                else:
                    values.append(str(v))
            else:
                values.append("-")
        lines.append(f"| {' | '.join(values)} |")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=str, default=".")
    parser.add_argument("--output", type=str, default="results/final_summary")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect training results
    logger.info("Scanning training logs...")
    training_results = scan_training_logs(base_dir)

    if training_results:
        logger.info(f"Found {len(training_results)} training runs")
        table = format_table(training_results)
        logger.info("\n" + table)
    else:
        logger.warning("No training results found")

    # Collect evaluation results
    logger.info("\nScanning evaluation results...")
    eval_results = load_eval_results(base_dir)
    if eval_results:
        logger.info(f"Found {len(eval_results)} evaluation results")

    # Save comprehensive summary
    summary = {
        "training": {name: {k: float(v) if isinstance(v, (np.floating, float)) else v
                            for k, v in m.items()}
                     for name, m in training_results.items()},
        "evaluation": eval_results,
    }

    with open(output_dir / "all_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Save markdown summary
    with open(output_dir / "summary.md", "w") as f:
        f.write("# NeuroHorizon Experiment Results\n\n")
        f.write(format_table(training_results))
        f.write("\n\n")

        if "Horizon eval" in eval_results:
            f.write("## Prediction Horizon Analysis\n\n")
            f.write("| Horizon (ms) | BPS Mean | BPS Std | FR Corr | R² |\n")
            f.write("| --- | --- | --- | --- | --- |\n")
            for r in eval_results["Horizon eval"].get("results", []):
                f.write(f"| {r['pred_length']*1000:.0f} | "
                        f"{r['bits_per_spike_mean']:.4f} | "
                        f"{r['bits_per_spike_std']:.4f} | "
                        f"{r.get('fr_corr_mean', '-') or '-'} | "
                        f"{r.get('r2_mean', '-') or '-'} |\n")
            f.write("\n")

        if any("Cross-session" in k for k in eval_results):
            f.write("## Cross-Session Generalization\n\n")
            for k, v in eval_results.items():
                if "Cross-session" not in k:
                    continue
                summary_data = v.get("summary", {})
                f.write(f"### {k}\n")
                f.write(f"- Train sessions: {summary_data.get('n_train', '-')}\n")
                f.write(f"- Test sessions: {summary_data.get('n_test', '-')}\n")
                f.write(f"- Train BPS: {summary_data.get('train_bps_mean', '-')}\n")
                f.write(f"- Test BPS: {summary_data.get('test_bps_mean', '-')}\n\n")

    logger.info(f"\nResults saved to {output_dir}")
    logger.info(f"  JSON: {output_dir / 'all_results.json'}")
    logger.info(f"  Markdown: {output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
