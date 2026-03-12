#!/usr/bin/env python3
"""
Plot NeuroHorizon module optimization progress from results.tsv.

Reads results.tsv and generates a line chart showing fp-bps trends
across optimization iterations for 250ms/500ms/1000ms prediction windows.

Usage:
    python plot_optimization_progress.py

Output:
    results/figures/phase1-autoregressive-1.9-module-optimization/optimization_progress.{png,pdf}
"""
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent
TSV_PATH = SCRIPT_DIR / 'results.tsv'
PROJECT_ROOT = Path('/root/autodl-tmp/NeuroHorizon')
OUTPUT_DIR = PROJECT_ROOT / 'results' / 'figures' / 'phase1-autoregressive-1.9-module-optimization'


def load_results(tsv_path):
    """Load results from TSV file."""
    rows = []
    with open(tsv_path, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            # Parse fp-bps values (handle '-' as NaN)
            for key in ['fp_bps_250ms', 'fp_bps_500ms', 'fp_bps_1000ms']:
                val = row[key].strip()
                row[key] = float(val) if val != '-' else np.nan
            rows.append(row)
    return rows


def plot_optimization_progress(rows, output_dir):
    """Generate optimization progress line chart."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Separate baseline/benchmark from optimization experiments
    baseline = None
    benchmarks = []
    experiments = []

    for row in rows:
        name = row['name']
        if name == 'baseline_v2':
            baseline = row
        elif name.startswith('benchmark_'):
            benchmarks.append(row)
        else:
            experiments.append(row)

    # Collect all points to plot (baseline + experiments)
    plot_points = []
    if baseline:
        plot_points.append(baseline)
    plot_points.extend(experiments)

    if not plot_points:
        print("No data to plot.")
        return

    # Setup figure
    fig, ax = plt.subplots(figsize=(12, 6))

    windows = [
        ('fp_bps_250ms', '250ms', '#2196F3', 'o'),
        ('fp_bps_500ms', '500ms', '#FF9800', 's'),
        ('fp_bps_1000ms', '1000ms', '#4CAF50', '^'),
    ]

    x_labels = [p['name'] for p in plot_points]
    x_pos = np.arange(len(plot_points))

    # Plot lines for each prediction window
    for key, label, color, marker in windows:
        values = [p[key] for p in plot_points]

        # Plot baseline with star marker
        if baseline:
            ax.plot(0, baseline[key], marker='*', markersize=15, color=color,
                    zorder=5, markeredgecolor='black', markeredgewidth=0.5)

        # Plot experiment points
        if len(values) > 0:
            ax.plot(x_pos, values, marker=marker, markersize=8, color=color,
                    linewidth=2, label=f'pred={label}', zorder=4)

        # Add value annotations
        for i, v in enumerate(values):
            if not np.isnan(v):
                ax.annotate(f'{v:.4f}', (x_pos[i], v),
                            textcoords="offset points", xytext=(0, 10),
                            ha='center', fontsize=7, color=color)

    # Plot benchmark reference lines (best among benchmarks for each window)
    benchmark_colors = {'250ms': '#2196F3', '500ms': '#FF9800', '1000ms': '#4CAF50'}
    for key, label, color, _ in windows:
        if benchmarks:
            best_benchmark = max(b[key] for b in benchmarks if not np.isnan(b[key]))
            best_name = [b['name'] for b in benchmarks
                         if b[key] == best_benchmark][0]
            ax.axhline(y=best_benchmark, color=color, linestyle='--',
                       alpha=0.4, linewidth=1)
            ax.text(len(plot_points) - 0.5, best_benchmark,
                    f'best benchmark ({label}): {best_benchmark:.4f}',
                    fontsize=7, color=color, alpha=0.6,
                    va='bottom', ha='right')

    # Formatting
    ax.set_xlabel('Experiment', fontsize=12)
    ax.set_ylabel('fp-bps (bits per spike)', fontsize=12)
    ax.set_title('NeuroHorizon Module Optimization Progress', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=9)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))

    # Add subtle background shading for baseline
    if baseline:
        ax.axvspan(-0.5, 0.5, alpha=0.05, color='gold', label='_nolegend_')

    plt.tight_layout()

    # Save
    png_path = output_dir / 'optimization_progress.png'
    pdf_path = output_dir / 'optimization_progress.pdf'
    fig.savefig(png_path, dpi=150, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == '__main__':
    rows = load_results(TSV_PATH)
    print(f"Loaded {len(rows)} entries from {TSV_PATH}")

    # Print summary
    for row in rows:
        print(f"  {row['name']}: 250ms={row['fp_bps_250ms']}, "
              f"500ms={row['fp_bps_500ms']}, 1000ms={row['fp_bps_1000ms']}")

    plot_optimization_progress(rows, OUTPUT_DIR)
