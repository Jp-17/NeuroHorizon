#!/usr/bin/env python3
"""Monitor progress for the 1.10 latent-dynamics experiments."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd


WINDOWS = ["250ms", "500ms", "1000ms"]
MODULE = "20260320_latent_dynamics_decoder"


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def latest_metrics(window_dir: Path) -> Path | None:
    candidates = sorted(window_dir.glob("lightning_logs/version_*/metrics.csv"))
    return candidates[-1] if candidates else None


def summarize_window(window_dir: Path) -> str:
    metrics_path = latest_metrics(window_dir)
    if metrics_path is None:
        return f"- {window_dir.name}: no metrics yet"

    df = pd.read_csv(metrics_path)
    df["epoch"] = pd.to_numeric(df.get("epoch"), errors="coerce")
    df["train_loss"] = pd.to_numeric(df.get("train_loss"), errors="coerce")
    df["val_loss"] = pd.to_numeric(df.get("val_loss"), errors="coerce")
    df["val/fp_bps"] = pd.to_numeric(df.get("val/fp_bps"), errors="coerce")

    latest_epoch = int(df["epoch"].dropna().max()) if df["epoch"].notna().any() else -1
    latest_val = df.dropna(subset=["val/fp_bps"])
    latest_train = df.dropna(subset=["train_loss"])

    train_loss = latest_train.iloc[-1]["train_loss"] if not latest_train.empty else float("nan")
    val_loss = latest_val.iloc[-1]["val_loss"] if not latest_val.empty else float("nan")
    val_fp = latest_val.iloc[-1]["val/fp_bps"] if not latest_val.empty else float("nan")

    return (
        f"- {window_dir.name}: epoch={latest_epoch}, "
        f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_fp_bps={val_fp:.4f}"
    )


def write_status() -> None:
    root = project_root()
    log_root = root / "results" / "logs" / "1.10-latent_dynamics_decoder" / MODULE
    lines = ["# 1.10 Latent Dynamics Progress", ""]
    for window in WINDOWS:
        lines.append(summarize_window(log_root / window))
    status_path = log_root / "progress_status.md"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval-sec", type=int, default=600)
    args = parser.parse_args()

    while True:
        write_status()
        time.sleep(args.interval_sec)


if __name__ == "__main__":
    main()
