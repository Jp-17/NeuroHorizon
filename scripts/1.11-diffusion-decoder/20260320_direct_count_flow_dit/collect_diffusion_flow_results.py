#!/usr/bin/env python3
"""Collect formal Phase 1.11 diffusion-flow results and generate figures."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


MODULE_NAME = "20260320_direct_count_flow_dit"
WINDOW_ORDER = ["250ms", "500ms", "1000ms"]
WINDOW_TO_MS = {"250ms": 250, "500ms": 500, "1000ms": 1000}
TRAINING_CONFIGS = {
    "250ms": {
        "epochs": 300,
        "batch_size": 64,
        "eval_batch_size": 16,
        "eval_epochs": 10,
        "base_lr": 3.125e-5,
        "weight_decay": 1e-4,
        "flow_steps_eval": 20,
    },
    "500ms": {
        "epochs": 300,
        "batch_size": 64,
        "eval_batch_size": 12,
        "eval_epochs": 10,
        "base_lr": 3.125e-5,
        "weight_decay": 1e-4,
        "flow_steps_eval": 20,
    },
    "1000ms": {
        "epochs": 300,
        "batch_size": 32,
        "eval_batch_size": 8,
        "eval_epochs": 10,
        "base_lr": 3.125e-5,
        "weight_decay": 1e-4,
        "flow_steps_eval": 20,
    },
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def diffusion_log_root(root: Path) -> Path:
    return root / "results" / "logs" / "1.11-diffusion-decoder" / MODULE_NAME


def diffusion_figure_root(root: Path) -> Path:
    return root / "results" / "figures" / "1.11-diffusion-decoder" / MODULE_NAME


def results_tsv_path(root: Path) -> Path:
    return root / "cc_todo" / "1.11-diffusion-decoder" / "results.tsv"


def baseline_eval_dir(root: Path, window: str) -> Path:
    return root / "results" / "logs" / f"phase1_v2_evalfix_{window}_cont" / "lightning_logs" / "version_0"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def float_or_none(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def epoch_int(value: str | None) -> int | None:
    numeric = float_or_none(value)
    if numeric is None:
        return None
    return int(numeric)


def load_metrics(metrics_path: Path) -> dict:
    with metrics_path.open() as handle:
        rows = list(csv.DictReader(handle))

    fieldnames = rows[0].keys() if rows else []
    per_bin_columns = sorted(
        [name for name in fieldnames if name.startswith("val/fp_bps_bin")],
        key=lambda name: int(name.split("bin", 1)[1]),
    )

    train_losses: dict[int, list[float]] = {}
    val_rows: dict[int, dict[str, float]] = {}
    best_val_fp: tuple[float, int, int] | None = None
    best_val_loss: tuple[float, int, int] | None = None
    last_epoch = -1

    for row in rows:
        epoch = epoch_int(row.get("epoch"))
        step = epoch_int(row.get("step"))
        if epoch is None:
            continue

        last_epoch = max(last_epoch, epoch)

        train_loss = float_or_none(row.get("train_loss"))
        if train_loss is not None:
            train_losses.setdefault(epoch, []).append(train_loss)

        val_fp = float_or_none(row.get("val/fp_bps"))
        val_loss = float_or_none(row.get("val_loss"))
        val_r2 = float_or_none(row.get("val/r2"))
        if val_fp is not None or val_loss is not None or val_r2 is not None:
            payload = {
                "epoch": epoch,
                "step": step,
                "val_loss": val_loss,
                "val_fp_bps": val_fp,
                "val_r2": val_r2,
                "per_bin_fp_bps": {
                    column.split("bin", 1)[1]: float_or_none(row.get(column)) for column in per_bin_columns
                },
            }
            val_rows[epoch] = payload

        if val_fp is not None and step is not None:
            current = (val_fp, epoch, step)
            if best_val_fp is None or current[0] > best_val_fp[0]:
                best_val_fp = current

        if val_loss is not None and step is not None:
            current = (val_loss, epoch, step)
            if best_val_loss is None or current[0] < best_val_loss[0]:
                best_val_loss = current

    train_curve = [
        {"epoch": epoch, "train_loss": float(np.mean(values))}
        for epoch, values in sorted(train_losses.items())
    ]
    val_curve = [val_rows[epoch] for epoch in sorted(val_rows)]

    return {
        "train_curve": train_curve,
        "val_curve": val_curve,
        "best_val_fp": best_val_fp,
        "best_val_loss": best_val_loss,
        "actual_last_epoch": last_epoch,
    }


def summarize_window(root: Path, window: str) -> dict:
    window_log_dir = diffusion_log_root(root) / window
    metrics_path = window_log_dir / "lightning_logs" / "version_0" / "metrics.csv"
    valid_eval_path = window_log_dir / "eval_v2_valid_results.json"
    test_eval_path = window_log_dir / "eval_v2_test_results.json"
    baseline_valid_path = baseline_eval_dir(root, window) / "eval_v2_valid_results.json"
    baseline_test_path = baseline_eval_dir(root, window) / "eval_v2_test_results.json"

    metrics = load_metrics(metrics_path)
    valid_eval = load_json(valid_eval_path)
    test_eval = load_json(test_eval_path)
    baseline_valid = load_json(baseline_valid_path)
    baseline_test = load_json(baseline_test_path)
    train_cfg = TRAINING_CONFIGS[window]

    checkpoint_path = test_eval["checkpoint"]
    test_fp = test_eval["continuous"]["fp_bps"]
    baseline_test_fp = baseline_test["continuous"]["fp_bps"]

    return {
        "window": window,
        "pred_window_ms": WINDOW_TO_MS[window],
        "log_dir": str(window_log_dir),
        "metrics_path": str(metrics_path),
        "checkpoint_path": checkpoint_path,
        "checkpoint_epoch": test_eval["checkpoint_epoch"],
        "checkpoint_global_step": test_eval["checkpoint_global_step"],
        "configured_epochs": train_cfg["epochs"],
        "actual_last_epoch": metrics["actual_last_epoch"],
        "batch_size": train_cfg["batch_size"],
        "eval_batch_size": train_cfg["eval_batch_size"],
        "eval_epochs": train_cfg["eval_epochs"],
        "base_lr": train_cfg["base_lr"],
        "max_lr": train_cfg["base_lr"] * train_cfg["batch_size"],
        "weight_decay": train_cfg["weight_decay"],
        "flow_steps_eval": train_cfg["flow_steps_eval"],
        "best_val_fp_bps": metrics["best_val_fp"][0],
        "best_val_fp_epoch": metrics["best_val_fp"][1],
        "best_val_fp_step": metrics["best_val_fp"][2],
        "best_val_loss": metrics["best_val_loss"][0],
        "best_val_loss_epoch": metrics["best_val_loss"][1],
        "best_val_loss_step": metrics["best_val_loss"][2],
        "train_curve": metrics["train_curve"],
        "val_curve": metrics["val_curve"],
        "formal_valid": valid_eval,
        "formal_test": test_eval,
        "baseline_valid": baseline_valid,
        "baseline_test": baseline_test,
        "delta_vs_baseline_test_fp_bps": test_fp - baseline_test_fp,
    }


def collect_summary(root: Path) -> dict:
    windows = {window: summarize_window(root, window) for window in WINDOW_ORDER}
    return {
        "module": MODULE_NAME,
        "figures_dir": str(diffusion_figure_root(root)),
        "log_dir": str(diffusion_log_root(root)),
        "notes": (
            "Direct count-space flow matching with time-only DiT summary tokens. "
            "Formal 250/500/1000ms results are all deeply below baseline_v2."
        ),
        "windows": windows,
    }


def figure_ylim(values: list[float], pad_ratio: float = 0.05) -> tuple[float, float]:
    finite = [value for value in values if np.isfinite(value)]
    if not finite:
        return 0.0, 1.0
    low = min(finite)
    high = max(finite)
    if np.isclose(low, high):
        return low - 0.05, high + 0.05
    margin = (high - low) * pad_ratio
    return low - margin, high + margin


def plot_training_curves(summary: dict, figure_dir: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), sharex="col")

    loss_values: list[float] = []
    fp_values: list[float] = []
    for window in WINDOW_ORDER:
        payload = summary["windows"][window]
        loss_values.extend([point["train_loss"] for point in payload["train_curve"]])
        loss_values.extend(
            [point["val_loss"] for point in payload["val_curve"] if point["val_loss"] is not None]
        )
        fp_values.extend(
            [point["val_fp_bps"] for point in payload["val_curve"] if point["val_fp_bps"] is not None]
        )
        fp_values.append(payload["formal_test"]["continuous"]["fp_bps"])
        fp_values.append(payload["baseline_test"]["continuous"]["fp_bps"])

    loss_ylim = figure_ylim(loss_values)
    fp_ylim = figure_ylim(fp_values)

    for index, window in enumerate(WINDOW_ORDER):
        ax_loss = axes[0, index]
        ax_fp = axes[1, index]
        payload = summary["windows"][window]
        train_curve = payload["train_curve"]
        val_curve = payload["val_curve"]

        if train_curve:
            ax_loss.plot(
                [point["epoch"] for point in train_curve],
                [point["train_loss"] for point in train_curve],
                color="#d95f02",
                linewidth=2.0,
                label="train_loss",
            )
        if val_curve:
            ax_loss.plot(
                [point["epoch"] for point in val_curve],
                [point["val_loss"] for point in val_curve],
                color="#1b9e77",
                linewidth=2.0,
                label="val_loss",
            )
        ax_loss.set_title(
            f"{window}: best fp@e{payload['best_val_fp_epoch']}, best loss@e{payload['best_val_loss_epoch']}",
            fontsize=11,
            fontweight="bold",
        )
        ax_loss.set_ylim(*loss_ylim)
        ax_loss.grid(alpha=0.25)
        if index == 0:
            ax_loss.set_ylabel("Loss")

        if val_curve:
            x = [point["epoch"] for point in val_curve]
            y = [point["val_fp_bps"] for point in val_curve]
            ax_fp.plot(x, y, color="#1f78b4", linewidth=2.0, label="val/fp_bps")
            best_x = payload["best_val_fp_epoch"]
            best_y = payload["best_val_fp_bps"]
            ax_fp.scatter(best_x, best_y, color="#1f78b4", s=35, zorder=5)
            ax_fp.annotate(
                f"{best_y:.3f}@e{best_x}",
                (best_x, best_y),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=8,
                color="#1f78b4",
            )
        ax_fp.axhline(
            payload["formal_test"]["continuous"]["fp_bps"],
            color="#e31a1c",
            linestyle="-",
            linewidth=1.5,
            label="formal test fp-bps",
        )
        ax_fp.axhline(
            payload["baseline_test"]["continuous"]["fp_bps"],
            color="#333333",
            linestyle="--",
            linewidth=1.5,
            label="baseline_v2 test fp-bps",
        )
        ax_fp.set_ylim(*fp_ylim)
        ax_fp.grid(alpha=0.25)
        ax_fp.set_xlabel("Epoch")
        if index == 0:
            ax_fp.set_ylabel("fp-bps")
        if train_curve:
            ax_loss.set_xlim(0, train_curve[-1]["epoch"])
            ax_fp.set_xlim(0, train_curve[-1]["epoch"])

    fig.suptitle("Phase 1.11 diffusion-flow training curves", fontsize=15, fontweight="bold")
    fig.text(
        0.5,
        0.02,
        "Top row: train_loss and val_loss. Bottom row: training-time val/fp_bps with formal test and baseline_v2 references.",
        ha="center",
        fontsize=10,
    )
    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.965), frameon=False)
    fig.tight_layout(rect=(0.02, 0.04, 0.98, 0.92))
    fig.savefig(figure_dir / "training_curves.png", dpi=180, bbox_inches="tight")
    fig.savefig(figure_dir / "training_curves.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_fp_bps_vs_window(summary: dict, figure_dir: Path) -> None:
    windows_ms = [WINDOW_TO_MS[window] for window in WINDOW_ORDER]
    diffusion_valid = [summary["windows"][window]["formal_valid"]["continuous"]["fp_bps"] for window in WINDOW_ORDER]
    diffusion_test = [summary["windows"][window]["formal_test"]["continuous"]["fp_bps"] for window in WINDOW_ORDER]
    baseline_valid = [summary["windows"][window]["baseline_valid"]["continuous"]["fp_bps"] for window in WINDOW_ORDER]
    baseline_test = [summary["windows"][window]["baseline_test"]["continuous"]["fp_bps"] for window in WINDOW_ORDER]
    deltas = [summary["windows"][window]["delta_vs_baseline_test_fp_bps"] for window in WINDOW_ORDER]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    axes[0].plot(windows_ms, diffusion_valid, marker="o", color="#1f78b4", linewidth=2.0, label="diffusion valid")
    axes[0].plot(windows_ms, diffusion_test, marker="o", color="#e31a1c", linewidth=2.0, label="diffusion test")
    axes[0].plot(windows_ms, baseline_valid, marker="s", color="#6a3d9a", linestyle="--", linewidth=1.8, label="baseline valid")
    axes[0].plot(windows_ms, baseline_test, marker="s", color="#333333", linestyle="--", linewidth=1.8, label="baseline test")
    axes[0].set_title("Continuous fp-bps vs prediction window", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Prediction window (ms)")
    axes[0].set_ylabel("fp-bps")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=9)

    axes[1].bar(
        [str(value) for value in windows_ms],
        deltas,
        color=["#b2182b", "#d6604d", "#f4a582"],
    )
    axes[1].axhline(0.0, color="#333333", linewidth=1.2)
    axes[1].set_title("Diffusion test delta vs baseline_v2", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Prediction window (ms)")
    axes[1].set_ylabel("diffusion test fp-bps - baseline test fp-bps")
    axes[1].grid(alpha=0.25, axis="y")
    for idx, delta in enumerate(deltas):
        axes[1].text(idx, delta, f"{delta:.3f}", ha="center", va="top", fontsize=9)

    fig.tight_layout()
    fig.savefig(figure_dir / "fp_bps_vs_window.png", dpi=180, bbox_inches="tight")
    fig.savefig(figure_dir / "fp_bps_vs_window.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_per_bin_fp_bps(summary: dict, figure_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), sharey=True)

    all_values: list[float] = []
    for window in WINDOW_ORDER:
        diffusion = summary["windows"][window]["formal_test"]["continuous"]["per_bin_fp_bps"]
        baseline = summary["windows"][window]["baseline_test"]["continuous"]["per_bin_fp_bps"]
        all_values.extend(diffusion.values())
        all_values.extend(baseline.values())
    y_limits = figure_ylim(all_values)

    for index, window in enumerate(WINDOW_ORDER):
        ax = axes[index]
        payload = summary["windows"][window]
        diffusion = payload["formal_test"]["continuous"]["per_bin_fp_bps"]
        baseline = payload["baseline_test"]["continuous"]["per_bin_fp_bps"]
        x = list(range(len(diffusion)))
        y_diffusion = [diffusion[str(bin_index)] for bin_index in x]
        y_baseline = [baseline[str(bin_index)] for bin_index in x]

        ax.plot(x, y_diffusion, color="#e31a1c", linewidth=2.0, label="diffusion test")
        ax.plot(x, y_baseline, color="#333333", linestyle="--", linewidth=1.8, label="baseline_v2 test")
        ax.set_title(f"{window} ({len(x)} bins)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Prediction bin index")
        ax.set_ylim(*y_limits)
        ax.grid(alpha=0.25)
        if index == 0:
            ax.set_ylabel("per-bin fp-bps")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02), frameon=False)
    fig.suptitle("Phase 1.11 per-bin fp-bps vs baseline_v2", fontsize=15, fontweight="bold", y=1.08)
    fig.tight_layout()
    fig.savefig(figure_dir / "per_bin_fp_bps.png", dpi=180, bbox_inches="tight")
    fig.savefig(figure_dir / "per_bin_fp_bps.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_summary_table(summary: dict, figure_dir: Path) -> None:
    rows = [
        [
            window,
            str(summary["windows"][window]["configured_epochs"]),
            str(summary["windows"][window]["actual_last_epoch"]),
            f"{summary['windows'][window]['best_val_fp_bps']:.4f}",
            f"e{summary['windows'][window]['best_val_fp_epoch']}",
            f"{summary['windows'][window]['formal_test']['continuous']['fp_bps']:.4f}",
            f"{summary['windows'][window]['delta_vs_baseline_test_fp_bps']:.4f}",
            f"{summary['windows'][window]['formal_test']['continuous']['r2']:.4f}",
            f"{summary['windows'][window]['formal_test']['trial_aligned']['per_neuron_psth_r2']:.4f}",
        ]
        for window in WINDOW_ORDER
    ]

    columns = [
        "window",
        "cfg epochs",
        "last epoch",
        "best val fp",
        "best epoch",
        "test fp",
        "vs base test",
        "test R2",
        "test PSTH-R2",
    ]

    fig, ax = plt.subplots(figsize=(15, 2.8))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.55)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#e6eef7")
        elif col in {3, 5, 6, 7, 8}:
            cell.set_facecolor("#fbeaea")
    ax.set_title("Phase 1.11 diffusion-flow formal summary", fontsize=14, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(figure_dir / "summary_table.png", dpi=180, bbox_inches="tight")
    fig.savefig(figure_dir / "summary_table.pdf", bbox_inches="tight")
    plt.close(fig)


def write_summary_json(summary: dict, figure_dir: Path) -> None:
    output_path = figure_dir / "diffusion_flow_summary.json"
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")


def update_results_tsv(root: Path, summary: dict) -> None:
    path = results_tsv_path(root)
    with path.open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))

    retained = [row for row in rows if row["module_name"] != "direct_count_flow_dit"]
    for window in WINDOW_ORDER:
        payload = summary["windows"][window]
        relative_checkpoint = str(Path(payload["checkpoint_path"]).relative_to(root))
        retained.append(
            {
                "date": "2026-03-20",
                "module_name": "direct_count_flow_dit",
                "pred_window_ms": str(payload["pred_window_ms"]),
                "best_val_fp_bps": f"{payload['best_val_fp_bps']:.4f}",
                "test_fp_bps": f"{payload['formal_test']['continuous']['fp_bps']:.4f}",
                "test_checkpoint": relative_checkpoint,
                "notes": (
                    f"formal 300 epochs; best_ckpt_epoch={payload['checkpoint_epoch']}; "
                    f"delta_vs_baseline_test={payload['delta_vs_baseline_test_fp_bps']:.4f}"
                ),
            }
        )

    fieldnames = [
        "date",
        "module_name",
        "pred_window_ms",
        "best_val_fp_bps",
        "test_fp_bps",
        "test_checkpoint",
        "notes",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(retained)


def main() -> None:
    root = project_root()
    figure_dir = diffusion_figure_root(root)
    figure_dir.mkdir(parents=True, exist_ok=True)

    summary = collect_summary(root)
    write_summary_json(summary, figure_dir)
    plot_training_curves(summary, figure_dir)
    plot_fp_bps_vs_window(summary, figure_dir)
    plot_per_bin_fp_bps(summary, figure_dir)
    plot_summary_table(summary, figure_dir)
    update_results_tsv(root, summary)

    print(f"Saved summary json and figures to {figure_dir}")
    print(f"Updated {results_tsv_path(root)}")


if __name__ == "__main__":
    main()
