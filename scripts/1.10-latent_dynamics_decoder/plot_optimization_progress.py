#!/usr/bin/env python3
"""Plot 1.10 latent-dynamics optimization progress from results.tsv."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


WINDOWS = ["250ms", "500ms", "1000ms"]
BASELINE = {"250ms": 0.2115, "500ms": 0.1744, "1000ms": 0.1317}


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> None:
    root = project_root()
    tsv_path = root / "cc_todo" / "1.10-latent_dynamics_decoder" / "results.tsv"
    df = pd.read_csv(tsv_path, sep="\t")
    df = df[df["name"].str.match(r"^20", na=False)].copy()

    if df.empty:
        raise RuntimeError("No 1.10 experiment rows found in results.tsv")

    labels = df["name"].tolist()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
    for ax, window in zip(axes, WINDOWS):
        values = pd.to_numeric(df[f"fp_bps_{window}"], errors="coerce")
        ax.plot(labels, values, marker="o", linewidth=2.0)
        ax.axhline(BASELINE[window], linestyle="--", color="black", linewidth=1.2)
        ax.set_title(window)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(alpha=0.25)
        ax.set_ylabel("valid fp-bps")
    fig.suptitle("1.10 Latent Dynamics Progress", fontsize=14, fontweight="bold")
    fig.tight_layout()

    out_dir = root / "results" / "figures" / "1.10-latent_dynamics_decoder"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "optimization_progress.png", dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / "optimization_progress.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
