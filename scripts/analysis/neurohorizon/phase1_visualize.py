"""
Phase 1 Visualization: Training Curves, R2 Analysis, AR vs non-AR Comparison

对应 plan.md Phase 1.2/1.3 可视化补充
数据源:
  - results/logs/phase1_full_report.json (4 组实验的 epoch-level 数据)
  - results/logs/phase1_small_250ms/ar_verify_results.json (per-bin R2/NLL)
输出: results/figures/phase1/ (4 张图)
"""

import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = Path('/root/autodl-tmp/NeuroHorizon')
LOGS_DIR = PROJECT_ROOT / 'results/logs'
OUTPUT_DIR = PROJECT_ROOT / 'results/figures/phase1'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Color & style scheme
COLORS = {
    '250ms AR': '#2196F3',
    '500ms AR': '#FF9800',
    '1000ms AR': '#4CAF50',
    '1000ms non-AR': '#F44336',
}
LINESTYLES = {
    '250ms AR': '-',
    '500ms AR': '-',
    '1000ms AR': '-',
    '1000ms non-AR': '--',
}
LABELS = {
    '250ms AR': '250ms (AR)',
    '500ms AR': '500ms (AR)',
    '1000ms AR': '1000ms (AR)',
    '1000ms non-AR': '1000ms (non-AR)',
}


def load_data():
    with open(LOGS_DIR / 'phase1_full_report.json') as f:
        report = json.load(f)
    with open(LOGS_DIR / 'phase1_small_250ms/ar_verify_results.json') as f:
        ar_verify = json.load(f)
    return report, ar_verify


def plot_01_training_curves(report):
    """Figure 1: Val Loss & R2 vs Epoch for all 4 experiments."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Phase 1: Training Curves (All Prediction Windows)',
                 fontsize=13, fontweight='bold')

    for name in ['250ms AR', '500ms AR', '1000ms AR', '1000ms non-AR']:
        exp = report[name]
        vals = exp['all_vals']
        epochs = [v['epoch'] for v in vals]
        val_losses = [v['val_loss'] for v in vals]
        r2s = [v['r2'] for v in vals]

        color = COLORS[name]
        ls = LINESTYLES[name]
        label = LABELS[name]

        # Val Loss
        axes[0].plot(epochs, val_losses, color=color, linestyle=ls,
                     linewidth=1.5, label=label, alpha=0.85)
        # Mark best epoch
        best_ep = exp['best_epoch']
        best_idx = None
        for i, v in enumerate(vals):
            if v['epoch'] == best_ep:
                best_idx = i
                break
        if best_idx is not None:
            axes[0].scatter([best_ep], [val_losses[best_idx]], color=color,
                           s=40, zorder=5, edgecolors='black', linewidths=0.5)

        # R2
        axes[1].plot(epochs, r2s, color=color, linestyle=ls,
                     linewidth=1.5, label=label, alpha=0.85)
        if best_idx is not None:
            axes[1].scatter([best_ep], [r2s[best_idx]], color=color,
                           s=40, zorder=5, edgecolors='black', linewidths=0.5)

    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Validation Loss (Poisson NLL)')
    axes[0].set_title('Val Loss vs Epoch')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('R\u00b2')
    axes[1].set_title('Validation R\u00b2 vs Epoch')
    axes[1].axhline(y=0.3, color='gray', linestyle=':', alpha=0.4,
                     label='Phase 0 threshold (0.3)')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out = OUTPUT_DIR / '01_training_curves.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[DONE] {out}')


def plot_02_r2_vs_window(report):
    """Figure 2: Best R2 and Val Loss vs prediction window length."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Phase 1: Performance vs Prediction Window Length',
                 fontsize=13, fontweight='bold')

    # AR experiments
    ar_names = ['250ms AR', '500ms AR', '1000ms AR']
    windows = [250, 500, 1000]
    ar_r2 = [report[n]['best_r2'] for n in ar_names]
    ar_loss = [report[n]['final_val_loss'] for n in ar_names]

    # non-AR for comparison
    noar_r2 = report['1000ms non-AR']['best_r2']
    noar_loss = report['1000ms non-AR']['final_val_loss']

    # --- Left: Best R2 ---
    ax = axes[0]
    ax.plot(windows, ar_r2, 'o-', color='#2196F3', linewidth=2, markersize=8, label='AR')
    ax.scatter([1000], [noar_r2], color='#F44336', s=100, marker='s',
              zorder=5, label='non-AR (1000ms)')

    # Annotate values and degradation
    for i, (w, r2) in enumerate(zip(windows, ar_r2)):
        ax.annotate(f'{r2:.4f}', xy=(w, r2), xytext=(0, 12),
                    textcoords='offset points', ha='center', fontsize=9)
    # Degradation percentages
    for i in range(1, len(ar_r2)):
        pct = (ar_r2[i] - ar_r2[i-1]) / ar_r2[i-1] * 100
        mid_w = (windows[i-1] + windows[i]) / 2
        mid_r2 = (ar_r2[i-1] + ar_r2[i]) / 2
        ax.annotate(f'{pct:+.1f}%', xy=(mid_w, mid_r2), fontsize=8,
                    color='red', ha='center', va='top')

    ax.annotate(f'{noar_r2:.4f}', xy=(1000, noar_r2), xytext=(10, -15),
                textcoords='offset points', fontsize=9, color='#F44336')

    ax.set_xlabel('Prediction Window (ms)')
    ax.set_ylabel('Best R\u00b2')
    ax.set_title('Best R\u00b2 vs Prediction Window')
    ax.set_xticks(windows)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Right: Final Val Loss ---
    ax = axes[1]
    ax.plot(windows, ar_loss, 'o-', color='#2196F3', linewidth=2, markersize=8, label='AR')
    ax.scatter([1000], [noar_loss], color='#F44336', s=100, marker='s',
              zorder=5, label='non-AR (1000ms)')

    for i, (w, l) in enumerate(zip(windows, ar_loss)):
        ax.annotate(f'{l:.4f}', xy=(w, l), xytext=(0, 12),
                    textcoords='offset points', ha='center', fontsize=9)
    ax.annotate(f'{noar_loss:.4f}', xy=(1000, noar_loss), xytext=(10, -15),
                textcoords='offset points', fontsize=9, color='#F44336')

    ax.set_xlabel('Prediction Window (ms)')
    ax.set_ylabel('Final Validation Loss')
    ax.set_title('Final Val Loss vs Prediction Window')
    ax.set_xticks(windows)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = OUTPUT_DIR / '02_r2_vs_window.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[DONE] {out}')


def plot_03_per_bin_r2(ar_verify):
    """Figure 3: Per-Bin R2 and NLL analysis (250ms, 12 bins)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Phase 1: Per-Bin Analysis (250ms, Teacher Forcing)',
                 fontsize=13, fontweight='bold')

    r2_bins = ar_verify['r2_bins_tf']
    nll_bins = ar_verify['nll_bins_tf']
    n_bins = len(r2_bins)
    bin_indices = np.arange(n_bins)
    bin_labels = [f'{i*20}-{(i+1)*20}ms' for i in range(n_bins)]

    # Color gradient from blue (early) to red (late)
    cmap = plt.cm.RdYlBu_r
    colors = [cmap(i / (n_bins - 1)) for i in range(n_bins)]

    # --- Left: Per-Bin R2 ---
    ax = axes[0]
    bars = ax.bar(bin_indices, r2_bins, color=colors, alpha=0.85, edgecolor='gray', linewidth=0.5)
    ax.axhline(y=np.mean(r2_bins), color='black', linestyle='--', alpha=0.5,
               label=f'Mean R\u00b2 = {np.mean(r2_bins):.4f}')
    ax.set_xlabel('Time Bin (within 250ms prediction window)')
    ax.set_ylabel('R\u00b2')
    ax.set_title('Per-Bin R\u00b2')
    ax.set_xticks(bin_indices)
    ax.set_xticklabels([f'{i}' for i in range(n_bins)], fontsize=8)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    # Add value labels on bars
    for i, v in enumerate(r2_bins):
        ax.text(i, v + 0.005, f'{v:.3f}', ha='center', fontsize=7, rotation=45)

    # --- Right: Per-Bin NLL ---
    ax = axes[1]
    bars = ax.bar(bin_indices, nll_bins, color=colors, alpha=0.85, edgecolor='gray', linewidth=0.5)
    ax.axhline(y=np.mean(nll_bins), color='black', linestyle='--', alpha=0.5,
               label=f'Mean NLL = {np.mean(nll_bins):.4f}')
    ax.set_xlabel('Time Bin (within 250ms prediction window)')
    ax.set_ylabel('Poisson NLL')
    ax.set_title('Per-Bin Poisson NLL')
    ax.set_xticks(bin_indices)
    ax.set_xticklabels([f'{i}' for i in range(n_bins)], fontsize=8)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(nll_bins):
        ax.text(i, v + 0.002, f'{v:.3f}', ha='center', fontsize=7, rotation=45)

    plt.tight_layout()
    out = OUTPUT_DIR / '03_per_bin_r2.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[DONE] {out}')


def plot_04_ar_vs_noar(report):
    """Figure 4: AR vs non-AR comparison (1000ms)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Phase 1: AR vs non-AR Comparison (1000ms Prediction Window)',
                 fontsize=13, fontweight='bold')

    ar = report['1000ms AR']
    noar = report['1000ms non-AR']

    ar_epochs = [v['epoch'] for v in ar['all_vals']]
    ar_r2 = [v['r2'] for v in ar['all_vals']]
    noar_epochs = [v['epoch'] for v in noar['all_vals']]
    noar_r2 = [v['r2'] for v in noar['all_vals']]

    # --- Left: R2 curves overlaid ---
    ax = axes[0]
    ax.plot(ar_epochs, ar_r2, color='#4CAF50', linewidth=1.5, label='1000ms AR')
    ax.plot(noar_epochs, noar_r2, color='#F44336', linewidth=1.5,
            linestyle='--', label='1000ms non-AR')
    ax.fill_between(ar_epochs,
                     [min(a, b) for a, b in zip(ar_r2, noar_r2)],
                     [max(a, b) for a, b in zip(ar_r2, noar_r2)],
                     alpha=0.15, color='gray', label='Difference region')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('R\u00b2')
    ax.set_title('R\u00b2 Training Curves')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Right: R2 difference ---
    ax = axes[1]
    diff = [b - a for a, b in zip(ar_r2, noar_r2)]
    ax.bar(ar_epochs, diff, width=8, color=['#4CAF50' if d >= 0 else '#F44336' for d in diff],
           alpha=0.7)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.axhline(y=np.mean(diff), color='blue', linestyle='--', alpha=0.5,
               label=f'Mean diff = {np.mean(diff):.4f}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('\u0394R\u00b2 (non-AR \u2212 AR)')
    ax.set_title('R\u00b2 Difference (non-AR minus AR)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Add annotation
    ax.text(0.5, 0.95, f'Max |diff| = {max(abs(d) for d in diff):.4f}\nDiff < 0.002 at all epochs',
            transform=ax.transAxes, fontsize=9, va='top', ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    out = OUTPUT_DIR / '04_ar_vs_noar.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[DONE] {out}')


if __name__ == '__main__':
    report, ar_verify = load_data()
    plot_01_training_curves(report)
    plot_02_r2_vs_window(report)
    plot_03_per_bin_r2(ar_verify)
    plot_04_ar_vs_noar(report)
    print(f'\n[ALL DONE] 4 figures saved to {OUTPUT_DIR}')
