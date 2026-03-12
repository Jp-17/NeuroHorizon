#!/usr/bin/env python3
"""Periodic progress monitor for prediction-memory experiments."""

from __future__ import annotations

import argparse
import csv
import subprocess
import time
from datetime import datetime
from pathlib import Path


ROOT = Path("/root/autodl-tmp/NeuroHorizon")
MODULE = "20260312_prediction_memory_decoder"
LOG_ROOT = ROOT / "results" / "logs" / "phase1-autoregressive-1.9-module-optimization" / MODULE
STATUS_MD = LOG_ROOT / "progress_status.md"
STATUS_LOG = LOG_ROOT / "progress_monitor.log"
WINDOWS = ("250ms", "500ms", "1000ms")
TARGET_EPOCHS = 300


def latest_version_dir(window_dir: Path) -> Path | None:
    versions = sorted((window_dir / "lightning_logs").glob("version_*"))
    return versions[-1] if versions else None


def read_latest_metrics(window_dir: Path) -> dict | None:
    version_dir = latest_version_dir(window_dir)
    if version_dir is None:
        return None
    metrics_path = version_dir / "metrics.csv"
    if not metrics_path.exists():
        return None

    latest_epoch = None
    latest_epoch_time = None
    latest_step = None

    with metrics_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            epoch = row.get("epoch", "")
            if epoch != "":
                try:
                    latest_epoch = int(float(epoch))
                except ValueError:
                    pass
            epoch_time = row.get("epoch_time", "")
            if epoch_time not in ("", None):
                try:
                    latest_epoch_time = float(epoch_time)
                except ValueError:
                    pass
            step = row.get("step", "")
            if step != "":
                try:
                    latest_step = int(float(step))
                except ValueError:
                    pass

    return {
        "version_dir": str(version_dir),
        "epoch": latest_epoch,
        "epoch_time": latest_epoch_time,
        "step": latest_step,
    }


def pid_alive(pid: int | None) -> bool:
    if pid is None:
        return False
    result = subprocess.run(["ps", "-p", str(pid)], capture_output=True, text=True)
    return result.returncode == 0 and len(result.stdout.strip().splitlines()) > 1


def read_pid(window_dir: Path) -> int | None:
    pid_path = window_dir / "job.pid"
    if not pid_path.exists():
        return None
    try:
        return int(pid_path.read_text().strip())
    except ValueError:
        return None


def fmt_eta(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    minutes = int(seconds // 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def collect_snapshot() -> tuple[str, bool]:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [f"# Prediction Memory Progress", f"", f"更新时间: {timestamp}", ""]
    any_active = False
    all_finished = True

    for window in WINDOWS:
        window_dir = LOG_ROOT / window
        pid = read_pid(window_dir)
        alive = pid_alive(pid)
        metrics = read_latest_metrics(window_dir)
        tf_done = (window_dir / "eval_teacher_forced.json").exists()
        rollout_done = (window_dir / "eval_rollout.json").exists()

        status = "not_started"
        eta = "unknown"
        progress = "0/300"

        if rollout_done:
            status = "finished"
        elif alive:
            status = "running"
            any_active = True
            if metrics and metrics["epoch"] is not None:
                current_epoch = metrics["epoch"] + 1
                progress = f"{current_epoch}/{TARGET_EPOCHS}"
                if metrics["epoch_time"] is not None:
                    remaining = max(TARGET_EPOCHS - current_epoch, 0) * metrics["epoch_time"]
                    eta = fmt_eta(remaining)
                else:
                    eta = "warming_up"
            else:
                progress = "starting"
                eta = "warming_up"
        elif tf_done:
            status = "eval_rollout_pending"
        elif tf_done or rollout_done:
            status = "evaluating"
        elif metrics and metrics["epoch"] is not None:
            progress = f"{metrics['epoch'] + 1}/{TARGET_EPOCHS}"
            status = "interrupted_or_waiting"

        all_finished = all_finished and rollout_done

        lines.append(f"## {window}")
        lines.append(f"- status: {status}")
        lines.append(f"- progress: {progress}")
        lines.append(f"- eta: {eta}")
        lines.append(f"- pid: {pid if pid is not None else 'none'}")
        if metrics:
            lines.append(f"- latest_version: `{metrics['version_dir']}`")
            lines.append(f"- latest_step: {metrics['step']}")
            lines.append(f"- latest_epoch_time_sec: {metrics['epoch_time']}")
        lines.append("")

    lines.append(f"all_finished: {all_finished}")
    lines.append(f"any_active: {any_active}")
    return "\n".join(lines) + "\n", (any_active or not all_finished)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval-sec", type=int, default=600)
    args = parser.parse_args()

    LOG_ROOT.mkdir(parents=True, exist_ok=True)

    while True:
        snapshot, keep_running = collect_snapshot()
        STATUS_MD.write_text(snapshot)
        with STATUS_LOG.open("a") as f:
            f.write(snapshot + "\n")

        if not keep_running:
            break
        time.sleep(args.interval_sec)


if __name__ == "__main__":
    main()
