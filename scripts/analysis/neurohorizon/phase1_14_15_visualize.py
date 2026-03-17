#!/usr/bin/env python3
"""Visualize 1.4 and 1.5 results against legacy simplified baselines.

The reused `phase1_benchmark_*` values come from the original 1.8.3 simplified
Transformer baselines and should not be interpreted as faithful benchmark
reproductions of NDT2 / Neuroformer / IBL-MtM.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_BASE = Path('/root/autodl-tmp/NeuroHorizon/results/logs')
FIG_BASE = Path('/root/autodl-tmp/NeuroHorizon/results/figures')

# =====================================================
# 1.4: obs_window experiment
# =====================================================
obs_windows = [250, 500, 750, 1000]

# NeuroHorizon results (from eval_v2_results.json)
nh_obs_results = {}
nh_obs_dirs = {
    250: 'phase1_v2_obs250ms',
    500: 'phase1_v2_250ms_cont',  # baseline
    750: 'phase1_v2_obs750ms',
    1000: 'phase1_v2_obs1000ms',
}
for obs, dname in nh_obs_dirs.items():
    eval_path = RESULTS_BASE / dname / 'lightning_logs' / 'eval_v2_results.json'
    if eval_path.exists():
        with open(eval_path) as f:
            r = json.load(f)
        nh_obs_results[obs] = r['continuous']['fp_bps']
    else:
        print(f'WARNING: {eval_path} not found')
        nh_obs_results[obs] = None

# Benchmark results for obs_window
bench_obs_models = ['ndt2', 'neuroformer', 'ibl_mtm']
bench_obs_labels = ['Legacy NDT2-like', 'Legacy Neuroformer-like', 'Legacy IBL-MtM-like']
bench_obs_results = {m: {} for m in bench_obs_models}

for obs in obs_windows:
    for model in bench_obs_models:
        if obs == 500:
            # use 1.8.3 legacy simplified-baseline result (pred=250ms, obs=500ms)
            rpath = RESULTS_BASE / f'phase1_benchmark_{model}_250ms' / 'results.json'
        else:
            rpath = RESULTS_BASE / f'phase1_benchmark_{model}_obs{obs}ms' / 'results.json'
        if rpath.exists():
            with open(rpath) as f:
                r = json.load(f)
            bench_obs_results[model][obs] = r['best_val_fp_bps']
        else:
            print(f'WARNING: {rpath} not found')
            bench_obs_results[model][obs] = None

# Plot 1.4
fig_dir = FIG_BASE / 'phase1_obs_window'
fig_dir.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
x = np.arange(len(obs_windows))
width = 0.2

# NeuroHorizon bars
nh_vals = [nh_obs_results[o] for o in obs_windows]
bars1 = ax.bar(x - 1.5*width, nh_vals, width, label='NeuroHorizon', color='#2196F3', edgecolor='black', linewidth=0.5)

# Benchmark bars
colors = ['#FF9800', '#4CAF50', '#9C27B0']
for i, (model, label) in enumerate(zip(bench_obs_models, bench_obs_labels)):
    vals = [bench_obs_results[model][o] for o in obs_windows]
    ax.bar(x + (i-0.5)*width, vals, width, label=label, color=colors[i], edgecolor='black', linewidth=0.5)

ax.set_xlabel('Observation Window (ms)', fontsize=12)
ax.set_ylabel('fp-bps', fontsize=12)
ax.set_title('1.4: fp-bps vs Observation Window with Legacy Baselines', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels([f'{o}ms' for o in obs_windows])
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(fig_dir / '01_obs_window_comparison.png', dpi=150, bbox_inches='tight')
print(f'Saved: {fig_dir}/01_obs_window_comparison.png')
plt.close()

# Line plot version
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.plot(obs_windows, nh_vals, 'o-', label='NeuroHorizon', color='#2196F3', linewidth=2, markersize=8)
for i, (model, label) in enumerate(zip(bench_obs_models, bench_obs_labels)):
    vals = [bench_obs_results[model][o] for o in obs_windows]
    ax.plot(obs_windows, vals, 's--', label=label, color=colors[i], linewidth=1.5, markersize=6)

ax.set_xlabel('Observation Window (ms)', fontsize=12)
ax.set_ylabel('fp-bps', fontsize=12)
ax.set_title('1.4: fp-bps vs Observation Window with Legacy Baselines', fontsize=14)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(fig_dir / '02_obs_window_line.png', dpi=150, bbox_inches='tight')
print(f'Saved: {fig_dir}/02_obs_window_line.png')
plt.close()


# =====================================================
# 1.5: session count experiment
# =====================================================
sessions = [1, 4, 7, 10]

# NeuroHorizon results
nh_sess_results = {}
nh_sess_dirs = {
    1: 'phase1_v2_1session',
    4: 'phase1_v2_4sessions',
    7: 'phase1_v2_7sessions',
    10: 'phase1_v2_250ms_cont',  # baseline
}
for s, dname in nh_sess_dirs.items():
    eval_path = RESULTS_BASE / dname / 'lightning_logs' / 'eval_v2_results.json'
    if eval_path.exists():
        with open(eval_path) as f:
            r = json.load(f)
        nh_sess_results[s] = r['continuous']['fp_bps']
    else:
        print(f'WARNING: {eval_path} not found')
        nh_sess_results[s] = None

# Benchmark results for session count
bench_sess_results = {m: {} for m in bench_obs_models}

for s in sessions:
    for model in bench_obs_models:
        if s == 10:
            # use 1.8.3 legacy simplified-baseline result (10 sessions, pred=250ms)
            rpath = RESULTS_BASE / f'phase1_benchmark_{model}_250ms' / 'results.json'
        else:
            suffix = '1session' if s == 1 else f'{s}sessions'
            rpath = RESULTS_BASE / f'phase1_benchmark_{model}_{suffix}' / 'results.json'
        if rpath.exists():
            with open(rpath) as f:
                r = json.load(f)
            bench_sess_results[model][s] = r['best_val_fp_bps']
        else:
            print(f'WARNING: {rpath} not found')
            bench_sess_results[model][s] = None

# Plot 1.5 - bar chart
fig_dir2 = FIG_BASE / 'phase1_sessions'
fig_dir2.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
x = np.arange(len(sessions))
width = 0.2

nh_vals2 = [nh_sess_results[s] for s in sessions]
bars1 = ax.bar(x - 1.5*width, nh_vals2, width, label='NeuroHorizon', color='#2196F3', edgecolor='black', linewidth=0.5)

for i, (model, label) in enumerate(zip(bench_obs_models, bench_obs_labels)):
    vals = [bench_sess_results[model][s] for s in sessions]
    ax.bar(x + (i-0.5)*width, vals, width, label=label, color=colors[i], edgecolor='black', linewidth=0.5)

ax.set_xlabel('Number of Training Sessions', fontsize=12)
ax.set_ylabel('fp-bps', fontsize=12)
ax.set_title('1.5: fp-bps vs Training Sessions with Legacy Baselines', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels([str(s) for s in sessions])
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

for bars in [bars1]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(fig_dir2 / '01_sessions_comparison.png', dpi=150, bbox_inches='tight')
print(f'Saved: {fig_dir2}/01_sessions_comparison.png')
plt.close()

# Line plot version
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.plot(sessions, nh_vals2, 'o-', label='NeuroHorizon', color='#2196F3', linewidth=2, markersize=8)
for i, (model, label) in enumerate(zip(bench_obs_models, bench_obs_labels)):
    vals = [bench_sess_results[model][s] for s in sessions]
    ax.plot(sessions, vals, 's--', label=label, color=colors[i], linewidth=1.5, markersize=6)

ax.set_xlabel('Number of Training Sessions', fontsize=12)
ax.set_ylabel('fp-bps', fontsize=12)
ax.set_title('1.5: fp-bps vs Training Sessions with Legacy Baselines', fontsize=14)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(fig_dir2 / '02_sessions_line.png', dpi=150, bbox_inches='tight')
print(f'Saved: {fig_dir2}/02_sessions_line.png')
plt.close()

# =====================================================
# Print summary tables
# =====================================================
print('\n' + '='*70)
print('1.4 obs_window Results (pred=250ms, 10 sessions)')
print('='*70)
print(f'{"Model":<15} {"obs250":<10} {"obs500":<10} {"obs750":<10} {"obs1000":<10}')
print('-'*55)
print(f'{"NeuroHorizon":<15} {nh_obs_results[250]:.4f}    {nh_obs_results[500]:.4f}    {nh_obs_results[750]:.4f}    {nh_obs_results[1000]:.4f}')
for model, label in zip(bench_obs_models, bench_obs_labels):
    vals = [bench_obs_results[model][o] for o in obs_windows]
    print(f'{label:<15} {vals[0]:.4f}    {vals[1]:.4f}    {vals[2]:.4f}    {vals[3]:.4f}')

print('\n' + '='*70)
print('1.5 Session Count Results (pred=250ms, obs=500ms)')
print('='*70)
print(f'{"Model":<15} {"1 sess":<10} {"4 sess":<10} {"7 sess":<10} {"10 sess":<10}')
print('-'*55)
print(f'{"NeuroHorizon":<15} {nh_sess_results[1]:.4f}    {nh_sess_results[4]:.4f}    {nh_sess_results[7]:.4f}    {nh_sess_results[10]:.4f}')
for model, label in zip(bench_obs_models, bench_obs_labels):
    vals = [bench_sess_results[model][s] for s in sessions]
    print(f'{label:<15} {vals[0]:.4f}    {vals[1]:.4f}    {vals[2]:.4f}    {vals[3]:.4f}')

print('\nDone!')
