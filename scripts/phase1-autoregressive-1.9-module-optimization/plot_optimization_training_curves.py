#!/usr/bin/env python3
"""Generate training-curve panels for Phase 1.9 module-optimization runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


WINDOW_ORDER = ["250ms", "500ms", "1000ms"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot epoch-level train/val curves for Phase 1.9 module-optimization "
            "experiments from Lightning metrics.csv logs."
        )
    )
    parser.add_argument(
        "--module",
        action="append",
        dest="modules",
        help="Optional module directory name under results/figures/phase1-autoregressive-1.9-module-optimization.",
    )
    return parser.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def figures_root(root: Path) -> Path:
    return root / "results" / "figures" / "phase1-autoregressive-1.9-module-optimization"


def find_summary_files(root: Path, selected_modules: list[str] | None) -> list[Path]:
    base = figures_root(root)
    summary_files: list[Path] = []

    if selected_modules:
        for module in selected_modules:
            module_dir = base / module
            matches = sorted(module_dir.glob("*_summary.json"))
            if not matches:
                raise FileNotFoundError(f"No summary json found for module: {module_dir}")
            summary_files.append(matches[0])
        return summary_files

    for module_dir in sorted(base.iterdir()):
        if not module_dir.is_dir():
            continue
        matches = sorted(module_dir.glob("*_summary.json"))
        if matches:
            summary_files.append(matches[0])
    return summary_files


def load_summary(summary_path: Path) -> dict:
    return json.loads(summary_path.read_text())


def load_epoch_curves(metrics_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(metrics_path)

    numeric_columns = [
        "epoch",
        "step",
        "train_loss",
        "val_loss",
        "val/fp_bps",
        "val/r2",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    train_rows = df.dropna(subset=["epoch", "train_loss"]).copy()
    train_epoch = (
        train_rows.groupby("epoch", as_index=False)["train_loss"].mean().sort_values("epoch")
    )

    val_columns = [column for column in ["val_loss", "val/fp_bps", "val/r2"] if column in df.columns]
    val_rows = df.dropna(subset=["epoch"]).copy()
    val_rows = val_rows[val_rows[val_columns].notna().any(axis=1)]
    val_epoch = (
        val_rows.groupby("epoch", as_index=False)[val_columns].last().sort_values("epoch")
        if not val_rows.empty
        else pd.DataFrame(columns=["epoch", *val_columns])
    )
    return train_epoch, val_epoch


def metrics_path_from_summary(window_summary: dict) -> Path:
    checkpoint_path = (
        window_summary.get("teacher_forced", {}).get("checkpoint")
        or window_summary.get("rollout", {}).get("checkpoint")
    )
    if not checkpoint_path:
        raise FileNotFoundError("Summary json is missing checkpoint path.")
    checkpoint = Path(checkpoint_path)
    return checkpoint.parent.parent / "metrics.csv"


def compute_limits(series_values: list[np.ndarray], pad_ratio: float = 0.05) -> tuple[float, float]:
    flattened = np.concatenate([values for values in series_values if values.size > 0])
    finite_values = flattened[np.isfinite(flattened)]
    if finite_values.size == 0:
        return 0.0, 1.0
    low = float(finite_values.min())
    high = float(finite_values.max())
    if np.isclose(low, high):
        low -= 0.05
        high += 0.05
        return low, high
    margin = (high - low) * pad_ratio
    return low - margin, high + margin


def humanize_module_name(module_name: str) -> str:
    parts = module_name.split("_", 1)
    label = parts[1] if len(parts) == 2 else module_name
    return label.replace("_", " ").title()


def plot_module(summary_path: Path) -> tuple[Path, Path]:
    summary = load_summary(summary_path)
    module_name = summary["module"]
    module_dir = summary_path.parent

    curve_payload = {}
    loss_values: list[np.ndarray] = []
    fp_values: list[np.ndarray] = []

    for window in WINDOW_ORDER:
        if window not in summary["windows"]:
            continue
        window_summary = summary["windows"][window]
        metrics_path = metrics_path_from_summary(window_summary)
        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing metrics.csv for {module_name} {window}: {metrics_path}")

        train_epoch, val_epoch = load_epoch_curves(metrics_path)
        curve_payload[window] = {
            "metrics_path": metrics_path,
            "version": metrics_path.parent.name,
            "train": train_epoch,
            "val": val_epoch,
            "teacher_forced_fp": window_summary.get("teacher_forced", {})
            .get("continuous", {})
            .get("fp_bps"),
            "rollout_fp": window_summary.get("rollout", {}).get("continuous", {}).get("fp_bps"),
            "baseline_fp": window_summary.get("baseline_fp_bps"),
        }
        if not train_epoch.empty:
            loss_values.append(train_epoch["train_loss"].to_numpy(dtype=float))
        if not val_epoch.empty:
            if "val_loss" in val_epoch:
                loss_values.append(val_epoch["val_loss"].to_numpy(dtype=float))
            if "val/fp_bps" in val_epoch:
                fp_values.append(val_epoch["val/fp_bps"].to_numpy(dtype=float))
        for key in ["teacher_forced_fp", "rollout_fp", "baseline_fp"]:
            value = curve_payload[window][key]
            if value is not None:
                fp_values.append(np.array([float(value)], dtype=float))

    if not curve_payload:
        raise RuntimeError(f"No window payload found for summary: {summary_path}")

    loss_ylim = compute_limits(loss_values)
    fp_ylim = compute_limits(fp_values)

    fig, axes = plt.subplots(2, 3, figsize=(17, 9), sharex="col")
    title = humanize_module_name(module_name)

    legend_handles = {}
    for col, window in enumerate(WINDOW_ORDER):
        ax_loss = axes[0, col]
        ax_fp = axes[1, col]

        if window not in curve_payload:
            ax_loss.axis("off")
            ax_fp.axis("off")
            continue

        payload = curve_payload[window]
        train_epoch = payload["train"]
        val_epoch = payload["val"]

        if not train_epoch.empty:
            handle = ax_loss.plot(
                train_epoch["epoch"],
                train_epoch["train_loss"],
                color="#d95f02",
                linewidth=2.0,
                label="train_loss (epoch mean)",
            )[0]
            legend_handles[handle.get_label()] = handle
        if not val_epoch.empty and "val_loss" in val_epoch:
            handle = ax_loss.plot(
                val_epoch["epoch"],
                val_epoch["val_loss"],
                color="#1b9e77",
                linewidth=2.0,
                label="val_loss",
            )[0]
            legend_handles[handle.get_label()] = handle
        ax_loss.set_ylim(*loss_ylim)
        ax_loss.grid(alpha=0.25)
        ax_loss.set_title(f"{window} ({payload['version']})", fontsize=11, fontweight="bold")
        if col == 0:
            ax_loss.set_ylabel("Loss")

        if not val_epoch.empty and "val/fp_bps" in val_epoch:
            handle = ax_fp.plot(
                val_epoch["epoch"],
                val_epoch["val/fp_bps"],
                color="#1f78b4",
                linewidth=2.0,
                label="val/fp_bps",
            )[0]
            legend_handles[handle.get_label()] = handle
        if payload["teacher_forced_fp"] is not None:
            handle = ax_fp.axhline(
                float(payload["teacher_forced_fp"]),
                color="#6a3d9a",
                linestyle="--",
                linewidth=1.4,
                label="post-train teacher-forced fp-bps",
            )
            legend_handles[handle.get_label()] = handle
        if payload["rollout_fp"] is not None:
            handle = ax_fp.axhline(
                float(payload["rollout_fp"]),
                color="#e31a1c",
                linestyle="-.",
                linewidth=1.4,
                label="post-train rollout fp-bps",
            )
            legend_handles[handle.get_label()] = handle
        if payload["baseline_fp"] is not None:
            handle = ax_fp.axhline(
                float(payload["baseline_fp"]),
                color="#333333",
                linestyle=":",
                linewidth=1.4,
                label="baseline_v2 rollout fp-bps",
            )
            legend_handles[handle.get_label()] = handle

        if not val_epoch.empty and "val/fp_bps" in val_epoch:
            best_idx = val_epoch["val/fp_bps"].idxmax()
            best_epoch = int(val_epoch.loc[best_idx, "epoch"])
            best_fp = float(val_epoch.loc[best_idx, "val/fp_bps"])
            ax_fp.scatter(best_epoch, best_fp, color="#1f78b4", s=30, zorder=5)
            ax_fp.annotate(
                f"best={best_fp:.3f}@e{best_epoch}",
                (best_epoch, best_fp),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=8,
                color="#1f78b4",
            )

        ax_fp.set_ylim(*fp_ylim)
        ax_fp.grid(alpha=0.25)
        ax_fp.set_xlabel("Epoch")
        if col == 0:
            ax_fp.set_ylabel("fp-bps")

        if not train_epoch.empty:
            max_epoch = int(train_epoch["epoch"].max())
            ax_loss.set_xlim(0, max_epoch)
            ax_fp.set_xlim(0, max_epoch)

        tf_fp = payload["teacher_forced_fp"]
        rollout_fp = payload["rollout_fp"]
        baseline_fp = payload["baseline_fp"]
        summary_text = []
        if tf_fp is not None:
            summary_text.append(f"TF {float(tf_fp):.3f}")
        if rollout_fp is not None:
            summary_text.append(f"RO {float(rollout_fp):.3f}")
        if baseline_fp is not None:
            summary_text.append(f"Base {float(baseline_fp):.3f}")
        ax_fp.text(
            0.98,
            0.02,
            " | ".join(summary_text),
            transform=ax_fp.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="#444444",
        )

    fig.suptitle(f"Phase 1.9 Training Curves: {title}", fontsize=15, fontweight="bold")
    fig.text(
        0.5,
        0.01,
        "Top row: epoch-level train_loss / val_loss from Lightning metrics.csv. "
        "Bottom row: training-time val/fp_bps with post-train teacher-forced, rollout, and baseline references.",
        ha="center",
        fontsize=10,
    )
    if legend_handles:
        fig.legend(
            legend_handles.values(),
            legend_handles.keys(),
            loc="upper center",
            ncol=3,
            bbox_to_anchor=(0.5, 0.965),
            fontsize=9,
            frameon=False,
        )
    fig.tight_layout(rect=(0.02, 0.04, 0.98, 0.92))

    png_path = module_dir / "training_curves.png"
    pdf_path = module_dir / "training_curves.pdf"
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def main() -> None:
    args = parse_args()
    root = project_root()
    summary_files = find_summary_files(root, args.modules)
    if not summary_files:
        raise FileNotFoundError("No module summary json files found under the Phase 1.9 figures directory.")

    for summary_path in summary_files:
        png_path, pdf_path = plot_module(summary_path)
        print(f"Saved {png_path}")
        print(f"Saved {pdf_path}")


if __name__ == "__main__":
    main()
