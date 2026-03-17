#!/usr/bin/env python3
"""
Legacy simplified-baseline visualization for the original 1.8.3 experiment.

The source results in `phase1_benchmark_*` are project-local simplified
Transformer baselines, not faithful reproductions of NDT2 / Neuroformer /
IBL-MtM. All titles and labels in this script must keep that caveat explicit.
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path('/root/autodl-tmp/NeuroHorizon/results/logs')
FIGURE_DIR = Path('/root/autodl-tmp/NeuroHorizon/results/figures/phase1_benchmark')
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ['ndt2', 'ibl_mtm', 'neuroformer']
MODEL_LABELS = {
    'ndt2': 'Legacy NDT2-like',
    'ibl_mtm': 'Legacy IBL-MtM-like',
    'neuroformer': 'Legacy Neuroformer-like',
}
WINDOWS = [250, 500, 1000]
COLORS = {'ndt2': '#2196F3', 'ibl_mtm': '#FF9800', 'neuroformer': '#4CAF50'}

# Optional: NeuroHorizon 1.3.4 results (fill in when available)
NEUROHORIZON_RESULTS = None  # Will be populated if results file exists


def load_results():
    """Load all legacy simplified-baseline results."""
    results = {}
    for model in MODELS:
        results[model] = {}
        for window in WINDOWS:
            path = RESULTS_DIR / f'phase1_benchmark_{model}_{window}ms' / 'results.json'
            if path.exists():
                with open(path) as f:
                    results[model][window] = json.load(f)
            else:
                print(f"  WARNING: Missing results for {model} @ {window}ms")
    return results


def load_neurohorizon_results():
    """Try to load NeuroHorizon 1.3.4 results for legacy comparison."""
    # Look for v2 results
    nh_results = {}
    for window in WINDOWS:
        for pattern in [f'phase1_v2_{window}ms*', f'phase1_{window}ms*']:
            candidates = list(RESULTS_DIR.glob(pattern))
            for c in candidates:
                rpath = c / 'results.json'
                if rpath.exists():
                    with open(rpath) as f:
                        data = json.load(f)
                    if 'best_val_fp_bps' in data or 'fp_bps' in data:
                        nh_results[window] = data
                        break
            if window in nh_results:
                break
    return nh_results if nh_results else None


def plot_fpbps_comparison(results, nh_results=None):
    """Plot 1: Multi-model legacy baseline fp-bps comparison bar chart."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(WINDOWS))
    width = 0.2
    n_models = len(MODELS) + (1 if nh_results else 0)
    offsets = np.arange(n_models) - (n_models - 1) / 2
    
    for i, model in enumerate(MODELS):
        vals = []
        for w in WINDOWS:
            if w in results[model]:
                vals.append(results[model][w].get('best_val_fp_bps', 0))
            else:
                vals.append(0)
        bars = ax.bar(x + offsets[i] * width, vals, width * 0.9,
                      label=MODEL_LABELS[model], color=COLORS[model], alpha=0.85)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    if nh_results:
        vals = [nh_results.get(w, {}).get('best_val_fp_bps', 
                nh_results.get(w, {}).get('fp_bps', 0)) for w in WINDOWS]
        idx = len(MODELS)
        bars = ax.bar(x + offsets[idx] * width, vals, width * 0.9,
                      label='NeuroHorizon', color='#E91E63', alpha=0.85)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Prediction Window')
    ax.set_ylabel('fp-bps (bits per spike)')
    ax.set_title('Legacy Simplified Baselines: fp-bps Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{w}ms' for w in WINDOWS])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / 'fpbps_comparison.png', dpi=150)
    plt.close()
    print("  Saved: fpbps_comparison.png (legacy simplified baselines)")


def plot_per_bin_decay(results):
    """Plot 2: Per-bin fp-bps decay curves for legacy simplified baselines."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    
    for j, window in enumerate(WINDOWS):
        ax = axes[j]
        for model in MODELS:
            if window in results[model]:
                per_bin = results[model][window].get('history', {}).get('val_fp_bps', [])
                # Try to get per-bin from best model checkpoint
                ckpt_path = RESULTS_DIR / f'phase1_benchmark_{model}_{window}ms' / 'best_model.pt'
                if ckpt_path.exists():
                    import torch
                    ckpt = torch.load(ckpt_path, map_location='cpu')
                    if 'metrics' in ckpt and 'per_bin_fp_bps' in ckpt['metrics']:
                        per_bin = ckpt['metrics']['per_bin_fp_bps']
                
                if per_bin and isinstance(per_bin, list) and len(per_bin) > 0:
                    bins = np.arange(len(per_bin)) * 20  # 20ms bins
                    ax.plot(bins, per_bin, '-o', label=MODEL_LABELS[model],
                            color=COLORS[model], markersize=3, alpha=0.8)
        
        ax.set_xlabel('Prediction Time (ms)')
        ax.set_title(f'{window}ms Window')
        ax.grid(alpha=0.3)
        if j == 0:
            ax.set_ylabel('fp-bps per bin')
        ax.legend(fontsize=8)
    
    plt.suptitle('Per-bin fp-bps Decay Across Legacy Simplified Baselines', y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / 'per_bin_fpbps_decay.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: per_bin_fpbps_decay.png (legacy simplified baselines)")


def plot_r2_comparison(results, nh_results=None):
    """Plot 3: Multi-model legacy baseline R² comparison bar chart."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(WINDOWS))
    width = 0.2
    n_models = len(MODELS) + (1 if nh_results else 0)
    offsets = np.arange(n_models) - (n_models - 1) / 2
    
    for i, model in enumerate(MODELS):
        vals = []
        for w in WINDOWS:
            if w in results[model]:
                r2_hist = results[model][w].get('history', {}).get('val_r2', [])
                val = r2_hist[-1] if r2_hist else 0
                vals.append(val)
            else:
                vals.append(0)
        bars = ax.bar(x + offsets[i] * width, vals, width * 0.9,
                      label=MODEL_LABELS[model], color=COLORS[model], alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, max(bar.get_height(), 0) + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    if nh_results:
        vals = [nh_results.get(w, {}).get('r2', 0) for w in WINDOWS]
        idx = len(MODELS)
        bars = ax.bar(x + offsets[idx] * width, vals, width * 0.9,
                      label='NeuroHorizon', color='#E91E63', alpha=0.85)
    
    ax.set_xlabel('Prediction Window')
    ax.set_ylabel('R²')
    ax.set_title('Legacy Simplified Baselines: R² Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{w}ms' for w in WINDOWS])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / 'r2_comparison.png', dpi=150)
    plt.close()
    print("  Saved: r2_comparison.png (legacy simplified baselines)")


def plot_summary_table(results, nh_results=None):
    """Plot 4: Legacy summary table + radar chart."""
    fig, (ax_table, ax_radar) = plt.subplots(1, 2, figsize=(16, 7),
                                               gridspec_kw={'width_ratios': [1.2, 1]})
    
    # Table
    ax_table.axis('off')
    
    headers = ['Model', 'Params'] + [f'{w}ms fp-bps' for w in WINDOWS] + [f'{w}ms R²' for w in WINDOWS]
    table_data = []
    
    for model in MODELS:
        row = [MODEL_LABELS[model]]
        if WINDOWS[0] in results[model]:
            row.append(f"{results[model][WINDOWS[0]].get('n_params', 0):,}")
        else:
            row.append('N/A')
        
        for w in WINDOWS:
            if w in results[model]:
                row.append(f"{results[model][w].get('best_val_fp_bps', 0):.4f}")
            else:
                row.append('N/A')
        
        for w in WINDOWS:
            if w in results[model]:
                r2_hist = results[model][w].get('history', {}).get('val_r2', [])
                val = r2_hist[-1] if r2_hist else 0
                row.append(f'{val:.4f}')
            else:
                row.append('N/A')
        
        table_data.append(row)
    
    if nh_results:
        row = ['NeuroHorizon', 'N/A']
        for w in WINDOWS:
            val = nh_results.get(w, {}).get('best_val_fp_bps', nh_results.get(w, {}).get('fp_bps', 0))
            row.append(f'{val:.4f}' if val else 'N/A')
        for w in WINDOWS:
            row.append(f"{nh_results.get(w, {}).get('r2', 0):.4f}" if w in nh_results else 'N/A')
        table_data.append(row)
    
    table = ax_table.table(cellText=table_data, colLabels=headers, loc='center',
                           cellLoc='center', colColours=['#E8E8E8'] * len(headers))
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    ax_table.set_title('Legacy Simplified Baseline Summary', fontsize=14, pad=20)
    
    # Radar chart (for 250ms results as representative)
    categories = ['fp-bps', 'R²', '1/NLL']
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    ax_radar = plt.subplot(122, polar=True)
    for model in MODELS:
        if 250 in results[model]:
            r = results[model][250]
            fp = r.get('best_val_fp_bps', 0)
            r2_hist = r.get('history', {}).get('val_r2', [])
            r2_val = r2_hist[-1] if r2_hist else 0
            nll_hist = r.get('history', {}).get('val_loss', [])
            nll_val = 1.0 / max(nll_hist[-1], 0.01) if nll_hist else 0
            
            # Normalize to [0, 1] range for radar
            values = [fp, max(r2_val, 0), min(nll_val, 10) / 10]
            values += values[:1]
            ax_radar.plot(angles, values, '-o', label=MODEL_LABELS[model],
                          color=COLORS[model], markersize=4)
            ax_radar.fill(angles, values, color=COLORS[model], alpha=0.1)
    
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(categories)
    ax_radar.set_title('250ms Legacy Metrics (Normalized)', pad=20)
    ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=8)
    
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / 'summary_table_radar.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: summary_table_radar.png (legacy simplified baselines)")


def main():
    print("Loading benchmark results...")
    results = load_results()
    nh_results = load_neurohorizon_results()
    
    if nh_results:
        print(f"  Found NeuroHorizon results for windows: {list(nh_results.keys())}")
    else:
        print("  No NeuroHorizon baseline results found (will compare benchmarks only)")
    
    print("\nGenerating legacy-baseline visualizations...")
    plot_fpbps_comparison(results, nh_results)
    plot_per_bin_decay(results)
    plot_r2_comparison(results, nh_results)
    plot_summary_table(results, nh_results)
    
    print(f"\nAll figures saved to: {FIGURE_DIR}")
    
    # Print summary
    print("\n=== Legacy Simplified Baseline Results Summary ===")
    for model in MODELS:
        print(f"\n{MODEL_LABELS[model]}:")
        for w in WINDOWS:
            if w in results[model]:
                r = results[model][w]
                fp = r.get('best_val_fp_bps', 0)
                r2_hist = r.get('history', {}).get('val_r2', [])
                r2_val = r2_hist[-1] if r2_hist else 0
                params = r.get('n_params', 0)
                t = r.get('total_time_s', 0)
                print(f"  {w}ms: fp-bps={fp:.4f}, R²={r2_val:.4f}, "
                      f"params={params:,}, time={t:.0f}s")


if __name__ == '__main__':
    main()
