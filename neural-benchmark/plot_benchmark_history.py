#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding='utf-8'))


def series(history: List[Dict[str, object]], key: str) -> List[float]:
    out = []
    for row in history:
        value = row.get(key)
        out.append(float(value) if value is not None else float('nan'))
    return out


def save_lineplot(epochs, values, *, ylabel: str, title: str, path: Path) -> None:
    finite = [v for v in values if not math.isnan(v)]
    if not finite:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(epochs, values, marker='o', linewidth=2)
    ax.set_xlabel('epoch')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_config_timeline(history: List[Dict[str, object]], output_dir: Path) -> None:
    keys = ['lr', 'weight_decay', 'effective_batch_size', 'grad_accum_steps', 'warmup_progress', 'tokens_seen']
    available = [key for key in keys if any(key in row for row in history)]
    epochs = series(history, 'epoch')
    n = len(available)
    cols = 2
    rows = max((n + cols - 1) // cols, 1)
    fig, axes = plt.subplots(rows, cols, figsize=(7.5, 3.2 * rows))
    try:
        flat_axes = list(axes.ravel())
    except AttributeError:
        flat_axes = [axes]
    for ax, key in zip(flat_axes, available):
        ax.plot(epochs, series(history, key), marker='o', linewidth=2)
        ax.set_title(key)
        ax.set_xlabel('epoch')
        ax.grid(True, alpha=0.3)
    for ax in flat_axes[len(available):]:
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(output_dir / 'training_config_timeline.png', dpi=180)
    plt.close(fig)


def save_mask_counts(history: List[Dict[str, object]], output_dir: Path) -> None:
    mask_keys = sorted({k for row in history for k in row.get('train_mask_counts', {}).keys()})
    if not mask_keys:
        return
    epochs = series(history, 'epoch')
    bottoms = [0.0] * len(history)
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for key in mask_keys:
        values = [float(row.get('train_mask_counts', {}).get(key, 0)) for row in history]
        ax.bar(epochs, values, bottom=bottoms, label=key)
        bottoms = [b + v for b, v in zip(bottoms, values)]
    ax.set_xlabel('epoch')
    ax.set_ylabel('batch count')
    ax.set_title('train_mask_counts by epoch')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / 'train_mask_counts_curve.png', dpi=180)
    plt.close(fig)


def write_summary(payload: Dict[str, object], history: List[Dict[str, object]], output_dir: Path) -> None:
    best_valid = payload.get('best_valid_metrics', {})
    summary = {
        'epochs': len(history),
        'best_epoch': payload.get('best_epoch'),
        'best_valid_fp_bps': best_valid.get('fp_bps'),
        'history_keys': sorted(history[0].keys()) if history else [],
    }
    (output_dir / 'history_summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    md = [
        '# History Summary',
        '',
        f"- epochs: {summary['epochs']}",
        f"- best_epoch: {summary['best_epoch']}",
        f"- best_valid_fp_bps: {summary['best_valid_fp_bps']}",
        f"- history_keys: {', '.join(summary['history_keys'])}",
    ]
    (output_dir / 'history_summary.md').write_text('\n'.join(md) + '\n', encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description='Plot training history for faithful benchmark runs')
    parser.add_argument('--results-json', required=True)
    parser.add_argument('--output-dir', default=None)
    args = parser.parse_args()

    payload = load_json(Path(args.results_json))
    history = payload.get('history', [])
    if not history:
        raise ValueError('results.json does not contain history')
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.results_json).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = series(history, 'epoch')
    save_lineplot(epochs, series(history, 'train_loss'), ylabel='train_loss', title='Train loss by epoch', path=output_dir / 'train_loss_curve.png')
    save_lineplot(epochs, series(history, 'valid_fp_bps'), ylabel='valid_fp_bps', title='Valid fp-bps by epoch', path=output_dir / 'valid_fp_bps_curve.png')
    if any('valid_rollout_fp_bps' in row for row in history):
        save_lineplot(
            epochs,
            series(history, 'valid_rollout_fp_bps'),
            ylabel='valid_rollout_fp_bps',
            title='Valid rollout fp-bps by epoch',
            path=output_dir / 'valid_rollout_fp_bps_curve.png',
        )
    if any('valid_true_past_fp_bps' in row for row in history):
        save_lineplot(
            epochs,
            series(history, 'valid_true_past_fp_bps'),
            ylabel='valid_true_past_fp_bps',
            title='Valid true_past fp-bps by epoch',
            path=output_dir / 'valid_true_past_fp_bps_curve.png',
        )
    if any('valid_rollout_true_past_gap_fp_bps' in row for row in history):
        save_lineplot(
            epochs,
            series(history, 'valid_rollout_true_past_gap_fp_bps'),
            ylabel='rollout - true_past',
            title='Valid rollout vs true_past gap by epoch',
            path=output_dir / 'valid_rollout_vs_true_past_gap_curve.png',
        )
    if any('valid_teacher_forced_loss' in row for row in history):
        save_lineplot(
            epochs,
            series(history, 'valid_teacher_forced_loss'),
            ylabel='valid_teacher_forced_loss',
            title='Valid teacher-forced loss by epoch',
            path=output_dir / 'valid_teacher_forced_loss_curve.png',
        )
    if any('valid_poisson_nll' in row for row in history):
        save_lineplot(
            epochs,
            series(history, 'valid_poisson_nll'),
            ylabel='valid_poisson_nll',
            title='Valid poisson NLL by epoch',
            path=output_dir / 'valid_poisson_nll_curve.png',
        )
    if any('predicted_to_true_event_ratio_mean' in row for row in history):
        save_lineplot(
            epochs,
            series(history, 'predicted_to_true_event_ratio_mean'),
            ylabel='pred/true event ratio',
            title='Predicted-to-true event ratio by epoch',
            path=output_dir / 'predicted_to_true_event_ratio_curve.png',
        )
    if any('valid_r2' in row for row in history):
        save_lineplot(epochs, series(history, 'valid_r2'), ylabel='valid_r2', title='Valid r2 by epoch', path=output_dir / 'valid_r2_curve.png')
    if any('lr' in row for row in history):
        save_lineplot(epochs, series(history, 'lr'), ylabel='lr', title='Learning rate by epoch', path=output_dir / 'lr_curve.png')
    save_config_timeline(history, output_dir)
    save_mask_counts(history, output_dir)
    write_summary(payload, history, output_dir)
    print(json.dumps({'output_dir': str(output_dir), 'n_epochs': len(history)}, indent=2))


if __name__ == '__main__':
    main()
