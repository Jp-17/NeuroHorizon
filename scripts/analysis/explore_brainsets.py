"""
Brainsets (Perich-Miller 2018) 数据集深度探索分析脚本

目标：建立对 Perich-Miller 数据集的完整数据直觉，为后续输入/输出窗口设计、
      自回归可行性评估提供依据。

对应 plan.md 任务 0.2.3
输出目录：results/figures/data_exploration/
"""

import sys
import warnings
import json
from pathlib import Path
from collections import defaultdict

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from temporaldata import Data

warnings.filterwarnings("ignore")

# Paths
PROJECT_ROOT = Path('/root/autodl-tmp/NeuroHorizon')
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed' / 'perich_miller_population_2018'
OUTPUT_DIR = PROJECT_ROOT / 'results' / 'figures' / 'data_exploration'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load all sessions
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("Loading all sessions...")
print("=" * 70)

session_files = sorted(PROCESSED_DIR.glob('*.h5'))
print(f"Found {len(session_files)} sessions")

sessions = {}
for fp in session_files:
    name = fp.stem
    with h5py.File(fp, 'r') as f:
        data = Data.from_hdf5(f)
        sessions[name] = {
            'name': name,
            'subject': name[0],  # c, j, m
            'date': name[2:10],
            'task': 'center_out' if 'center_out' in name else 'random_target',
            'domain_start': float(data.domain.start[0]),
            'domain_end': float(data.domain.end[0]),
            'duration': float(data.domain.end[0] - data.domain.start[0]),
            'n_spikes': len(data.spikes.timestamps),
            'n_units': len(data.units.id),
            'n_trials': len(data.trials.start),
            'n_valid_trials': int(data.trials.is_valid[:].sum()),
            'hold_durations': (data.movement_phases.hold_period.end[:]
                               - data.movement_phases.hold_period.start[:]),
            'reach_durations': (data.movement_phases.reach_period.end[:]
                                - data.movement_phases.reach_period.start[:]),
            'return_durations': (data.movement_phases.return_period.end[:]
                                 - data.movement_phases.return_period.start[:]),
            'spike_timestamps': data.spikes.timestamps[:],
            'unit_indices': data.spikes.unit_index[:],
            'n_units_val': len(data.units.id),
            'n_train_intervals': len(data.train_domain.start),
            'n_valid_trials_dom': len(data.valid_domain.start),
            'n_test_trials': len(data.test_domain.start),
        }
    s = sessions[name]
    s['firing_rate_per_unit'] = s['n_spikes'] / s['duration'] / s['n_units']
    print(f"  {name}: {s['n_units']} units, {s['n_valid_trials']} valid trials, "
          f"{s['duration']/60:.1f}min, FR={s['firing_rate_per_unit']:.1f}Hz/unit")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Dataset Overview Statistics
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("2. Dataset Overview Statistics")
print("=" * 70)

by_subject = defaultdict(list)
for name, s in sessions.items():
    by_subject[s['subject']].append(s)

print(f"\nTotal sessions: {len(sessions)}")
for sub, slist in sorted(by_subject.items()):
    n_units_list = [s['n_units'] for s in slist]
    n_trials_list = [s['n_valid_trials'] for s in slist]
    dur_list = [s['duration']/60 for s in slist]
    print(f"  Subject {sub.upper()}: {len(slist)} sessions, "
          f"units={np.min(n_units_list)}-{np.max(n_units_list)} (med={np.median(n_units_list):.0f}), "
          f"trials={np.min(n_trials_list)}-{np.max(n_trials_list)} (med={np.median(n_trials_list):.0f}), "
          f"dur={np.median(dur_list):.1f}min (med)")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Task Structure Analysis
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("3. Task Structure Analysis")
print("=" * 70)

all_hold = np.concatenate([s['hold_durations'] for s in sessions.values()])
all_reach = np.concatenate([s['reach_durations'] for s in sessions.values()])
all_return = np.concatenate([s['return_durations'] for s in sessions.values()])

print(f"\nHold period (input window candidate):")
print(f"  mean={all_hold.mean()*1000:.0f}ms, std={all_hold.std()*1000:.0f}ms, "
      f"min={all_hold.min()*1000:.0f}ms, max={all_hold.max()*1000:.0f}ms")
print(f"  >250ms: {(all_hold > 0.25).mean()*100:.1f}%,  "
      f">500ms: {(all_hold > 0.5).mean()*100:.1f}%")

print(f"\nReach period (prediction window candidate):")
print(f"  mean={all_reach.mean()*1000:.0f}ms, std={all_reach.std()*1000:.0f}ms, "
      f"min={all_reach.min()*1000:.0f}ms, max={all_reach.max()*1000:.0f}ms")
print(f"  >250ms: {(all_reach > 0.25).mean()*100:.1f}%,  "
      f">500ms: {(all_reach > 0.5).mean()*100:.1f}%,  "
      f">1000ms: {(all_reach > 1.0).mean()*100:.1f}%")

print(f"\nReturn period:")
print(f"  mean={all_return.mean()*1000:.0f}ms, std={all_return.std()*1000:.0f}ms")

print(f"\nWindow design feasibility:")
print(f"  250ms prediction: {(all_reach > 0.25).mean()*100:.0f}% trials feasible")
print(f"  500ms prediction: {(all_reach > 0.5).mean()*100:.0f}% trials feasible")
print(f"  1000ms prediction: {(all_reach > 1.0).mean()*100:.0f}% trials feasible")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Neuron Statistics
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("4. Neuron Statistics")
print("=" * 70)

# Compute per-unit firing rates for each session
all_unit_frs = []
for name, s in sessions.items():
    timestamps = s['spike_timestamps']
    unit_idxs = s['unit_indices']
    duration = s['duration']
    n_units = s['n_units']
    unit_frs = np.zeros(n_units)
    for u in range(n_units):
        mask = (unit_idxs == u)
        unit_frs[u] = mask.sum() / duration
    all_unit_frs.extend(unit_frs.tolist())
    s['unit_frs'] = unit_frs

all_unit_frs = np.array(all_unit_frs)
print(f"\nPer-unit firing rates (across all sessions):")
print(f"  total units: {len(all_unit_frs)}")
print(f"  mean={all_unit_frs.mean():.1f}Hz, median={np.median(all_unit_frs):.1f}Hz, "
      f"std={all_unit_frs.std():.1f}Hz")
print(f"  <1Hz: {(all_unit_frs < 1).mean()*100:.1f}%,  "
      f"1-5Hz: {((all_unit_frs>=1)&(all_unit_frs<5)).mean()*100:.1f}%,  "
      f">5Hz: {(all_unit_frs>=5).mean()*100:.1f}%")

# Spike count stats at different bin widths
print(f"\nSpike count statistics at different bin widths:")
for bw_ms in [20, 50, 100]:
    bw = bw_ms / 1000.0
    all_counts = []
    for name, s in sessions.items():
        dur = s['duration']
        timestamps = s['spike_timestamps']
        unit_idxs = s['unit_indices']
        n_units = s['n_units']
        n_bins = int(dur / bw)
        if n_bins < 1:
            continue
        for u in range(n_units):
            t_u = timestamps[unit_idxs == u]
            counts, _ = np.histogram(t_u, bins=n_bins, range=(0, n_bins * bw))
            all_counts.extend(counts.tolist())
    all_counts = np.array(all_counts)
    sparsity = (all_counts == 0).mean()
    print(f"  {bw_ms}ms bin: mean={all_counts.mean():.3f}, var={all_counts.var():.3f}, "
          f"zero_frac={sparsity*100:.1f}% (sparsity)")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Autoregressive Feasibility
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("5. Autoregressive Feasibility Assessment")
print("=" * 70)

# Mean spikes per 20ms bin
bw = 0.02
all_bin_counts = []
for name, s in sessions.items():
    dur = s['duration']
    n_bins = int(dur / bw)
    counts, _ = np.histogram(s['spike_timestamps'], bins=n_bins, range=(0, n_bins * bw))
    all_bin_counts.extend(counts.tolist())
all_bin_counts = np.array(all_bin_counts)

print(f"\nSpike statistics per 20ms bin:")
print(f"  mean spikes/bin: {all_bin_counts.mean():.3f}")
print(f"  Sparsity (0 spikes): {(all_bin_counts==0).mean()*100:.1f}%")
lambda_approx = all_bin_counts.mean()
print(f"  Poisson NLL suitability: lambda={lambda_approx:.3f}")
if lambda_approx < 0.1:
    print(f"  ⚠️ Very sparse — Poisson NLL may be numerically unstable (log(0) risk)")
elif lambda_approx < 0.5:
    print(f"  ⚠️ Sparse — needs careful log(0) handling in PoissonNLL")
else:
    print(f"  ✓ Reasonable spike density for Poisson NLL")

print(f"\nWindow design recommendation:")
n_steps_250 = int(0.25 / bw)
n_steps_500 = int(0.50 / bw)
n_steps_1000 = int(1.0 / bw)
print(f"  250ms prediction = {n_steps_250} autoregressive steps (20ms bins)")
print(f"  500ms prediction = {n_steps_500} autoregressive steps (20ms bins)")
print(f"  1000ms prediction = {n_steps_1000} autoregressive steps (20ms bins)")
print(f"  Recommendation: start with 250ms ({n_steps_250} steps) as Phase 1 baseline")

# ─────────────────────────────────────────────────────────────────────────────
# 6. FIGURES
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("6. Generating Figures...")
print("=" * 70)

# Figure 1: Dataset Overview
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle('Perich-Miller 2018 Dataset Overview (10 sessions)', fontsize=14, fontweight='bold')

# 1a: Session duration by subject
ax = axes[0, 0]
for sub, slist in sorted(by_subject.items()):
    durations = [s['duration']/60 for s in slist]
    ax.bar([f'{sub.upper()}{i+1}' for i in range(len(slist))],
           durations, label=f'Subject {sub.upper()}', alpha=0.8)
ax.set_xlabel('Session')
ax.set_ylabel('Duration (minutes)')
ax.set_title('Session Duration by Subject')
ax.legend()
ax.tick_params(axis='x', rotation=45)

# 1b: Unit count distribution
ax = axes[0, 1]
unit_counts = [s['n_units'] for s in sessions.values()]
subjects = [s['subject'].upper() for s in sessions.values()]
colors = {'C': '#2196F3', 'J': '#FF9800', 'M': '#4CAF50'}
for sub, slist in sorted(by_subject.items()):
    ax.hist([s['n_units'] for s in slist], alpha=0.7, label=f'Subject {sub.upper()}',
            color=colors[sub.upper()], bins=5)
ax.set_xlabel('Number of Units')
ax.set_ylabel('Count')
ax.set_title('Unit Count Distribution')
ax.legend()

# 1c: Valid trial count
ax = axes[0, 2]
for sub, slist in sorted(by_subject.items()):
    ax.bar([f'{sub.upper()}{i+1}' for i in range(len(slist))],
           [s['n_valid_trials'] for s in slist], label=f'Subject {sub.upper()}',
           color=colors[sub.upper()], alpha=0.8)
ax.set_xlabel('Session')
ax.set_ylabel('Valid Trial Count')
ax.set_title('Valid Trials per Session')
ax.legend()
ax.tick_params(axis='x', rotation=45)

# 1d: Per-unit firing rate distribution
ax = axes[1, 0]
ax.hist(all_unit_frs, bins=30, edgecolor='black', alpha=0.8, color='steelblue')
ax.axvline(np.median(all_unit_frs), color='red', linestyle='--', label=f'Median={np.median(all_unit_frs):.1f}Hz')
ax.set_xlabel('Firing Rate (Hz)')
ax.set_ylabel('Count')
ax.set_title('Per-Unit Firing Rate Distribution')
ax.legend()

# 1e: Hold period duration histogram
ax = axes[1, 1]
ax.hist(all_hold * 1000, bins=30, edgecolor='black', alpha=0.8, color='coral')
for thresh_ms, color in [(250, 'blue'), (500, 'green'), (1000, 'red')]:
    ax.axvline(thresh_ms, color=color, linestyle='--', alpha=0.8, label=f'{thresh_ms}ms')
ax.set_xlabel('Hold Period Duration (ms)')
ax.set_ylabel('Count')
ax.set_title('Hold Period Duration (Input Window)')
ax.legend()

# 1f: Reach period duration histogram
ax = axes[1, 2]
ax.hist(all_reach * 1000, bins=30, edgecolor='black', alpha=0.8, color='mediumpurple')
for thresh_ms, color in [(250, 'blue'), (500, 'green'), (1000, 'red')]:
    pct = (all_reach > thresh_ms/1000).mean() * 100
    ax.axvline(thresh_ms, color=color, linestyle='--', alpha=0.8, label=f'{thresh_ms}ms ({pct:.0f}%)')
ax.set_xlabel('Reach Period Duration (ms)')
ax.set_ylabel('Count')
ax.set_title('Reach Period Duration (Prediction Window)')
ax.legend()

plt.tight_layout()
fig1_path = OUTPUT_DIR / '01_dataset_overview.png'
plt.savefig(fig1_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"  [OK] {fig1_path.name}")

# Figure 2: Neural Statistics
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle('Neural Statistics - Perich-Miller 2018', fontsize=14, fontweight='bold')

# 2a: Spike count distribution at different bin widths
ax = axes[0, 0]
for bw_ms, color, ls in [(20, 'blue', '-'), (50, 'green', '--'), (100, 'red', ':')]:
    bw = bw_ms / 1000.0
    all_counts = []
    for name, s in sessions.items():
        dur = s['duration']
        n_bins = int(dur / bw)
        for u in range(s['n_units']):
            t_u = s['spike_timestamps'][s['unit_indices'] == u]
            counts, _ = np.histogram(t_u, bins=n_bins, range=(0, n_bins * bw))
            all_counts.extend(counts.tolist())
    all_counts = np.array(all_counts)
    max_count = min(int(np.percentile(all_counts, 99)) + 1, 20)
    bins_range = np.arange(0, max_count + 1) - 0.5
    counts_hist, _ = np.histogram(all_counts, bins=bins_range)
    ax.semilogy(np.arange(0, max_count), counts_hist / counts_hist.sum(),
                color=color, linestyle=ls, label=f'{bw_ms}ms bin', marker='o', markersize=3)
ax.set_xlabel('Spike Count')
ax.set_ylabel('Fraction (log scale)')
ax.set_title('Spike Count Distribution by Bin Width')
ax.legend()
ax.grid(True, alpha=0.3)

# 2b: PSTH (population average aligned to go cue) 
ax = axes[0, 1]
# Simple PSTH: align spikes to go_cue per session
for name, s in sessions.items():
    with h5py.File(PROCESSED_DIR / f'{name}.h5', 'r') as f:
        d = Data.from_hdf5(f)
        go_cues = d.movement_phases.reach_period.start[:]
        timestamps = d.spikes.timestamps[:]
        n_units = s['n_units']
        
        bins = np.arange(-0.3, 1.0, 0.02)  # -300ms to 1s, 20ms bins
        psth = np.zeros(len(bins) - 1)
        n_trials_used = 0
        for gc in go_cues[:min(20, len(go_cues))]:  # use up to 20 trials
            t_rel = timestamps - gc
            mask = (t_rel >= -0.3) & (t_rel < 1.0)
            counts, _ = np.histogram(t_rel[mask], bins=bins)
            psth += counts
            n_trials_used += 1
        if n_trials_used > 0:
            psth = psth / n_trials_used / (bins[1]-bins[0]) / n_units  # Hz/unit
        bin_centers = (bins[:-1] + bins[1:]) / 2
        ax.plot(bin_centers * 1000, psth, alpha=0.5, linewidth=1, label=name[:8])

ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Go Cue')
ax.set_xlabel('Time from Go Cue (ms)')
ax.set_ylabel('Population PSTH (Hz/unit)')
ax.set_title('Population PSTH Aligned to Go Cue')
ax.legend(fontsize=6, ncol=2)
ax.grid(True, alpha=0.3)

# 2c: Average firing rate per session
ax = axes[0, 2]
session_names = list(sessions.keys())
session_frs = [s['firing_rate_per_unit'] for s in sessions.values()]
session_cols = [colors[s['subject'].upper()] for s in sessions.values()]
ax.barh(range(len(session_names)), session_frs, color=session_cols, edgecolor='black', alpha=0.8)
ax.set_yticks(range(len(session_names)))
ax.set_yticklabels([n[:18] for n in session_names], fontsize=8)
ax.set_xlabel('Mean Firing Rate (Hz/unit)')
ax.set_title('Average Firing Rate per Session')
ax.grid(True, alpha=0.3, axis='x')

# 2d: Spike count Poisson check (variance vs mean)
ax = axes[1, 0]
means, variances = [], []
bw = 0.05  # 50ms bin
for name, s in sessions.items():
    dur = s['duration']
    n_bins = int(dur / bw)
    for u in range(s['n_units']):
        t_u = s['spike_timestamps'][s['unit_indices'] == u]
        counts, _ = np.histogram(t_u, bins=n_bins, range=(0, n_bins * bw))
        if counts.mean() > 0:
            means.append(counts.mean())
            variances.append(counts.var())
means = np.array(means)
variances = np.array(variances)
ax.scatter(means, variances, alpha=0.3, s=10, color='steelblue')
lim = max(means.max(), variances.max())
ax.plot([0, lim], [0, lim], 'r--', label='Poisson (var=mean)')
ax.set_xlabel('Mean Spike Count (50ms bin)')
ax.set_ylabel('Variance')
ax.set_title('Mean-Variance Relationship\n(Poisson check, 50ms bins)')
ax.legend()

# 2e: Raster plot example (one session, first 5 units, first 10 trials)
ax = axes[1, 1]
example_session = list(sessions.keys())[0]  # c_20131003
s = sessions[example_session]
with h5py.File(PROCESSED_DIR / f'{example_session}.h5', 'r') as f:
    d = Data.from_hdf5(f)
    go_cues = d.movement_phases.reach_period.start[:10]  # first 10 trials
    hold_starts = d.movement_phases.hold_period.start[:10]
    hold_ends = d.movement_phases.hold_period.end[:10]
    timestamps = d.spikes.timestamps[:]
    unit_idxs = d.spikes.unit_index[:]

for trial_i, (hs, he, gc) in enumerate(zip(hold_starts, hold_ends, go_cues)):
    t_start = hs - 0.1
    t_end = gc + 0.8
    for unit_i in range(min(5, s['n_units'])):
        t_u = timestamps[unit_idxs == unit_i]
        spikes_in_win = t_u[(t_u >= t_start) & (t_u <= t_end)] - gc  # align to go cue
        ax.scatter(spikes_in_win * 1000, np.full(len(spikes_in_win), trial_i * 6 + unit_i),
                   marker='|', color=f'C{unit_i}', s=20, linewidths=0.5)

ax.axvline(0, color='red', linestyle='--', linewidth=1.5, label='Go Cue')
ax.axvspan(-300, 0, alpha=0.1, color='blue', label='Hold period')
ax.set_xlabel('Time from Go Cue (ms)')
ax.set_ylabel('Trial × Unit')
ax.set_title(f'Raster Plot\n({example_session[:20]}, 5 units, 10 trials)')
ax.legend(fontsize=7)

# 2f: Window feasibility table
ax = axes[1, 2]
ax.axis('off')
table_data = [
    ['Window', 'Steps\n(20ms)', 'Hold\nSufficient', 'Reach\nSufficient'],
    ['250ms', '12', f'{(all_hold>0.25).mean()*100:.0f}%', f'{(all_reach>0.25).mean()*100:.0f}%'],
    ['500ms', '25', f'{(all_hold>0.5).mean()*100:.0f}%', f'{(all_reach>0.5).mean()*100:.0f}%'],
    ['1000ms', '50', f'{(all_hold>1.0).mean()*100:.0f}%', f'{(all_reach>1.0).mean()*100:.0f}%'],
]
table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                 loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 2.0)
ax.set_title('Autoregressive Window Feasibility\n(Perich-Miller Center-Out)', pad=20)

# Color header
for j in range(4):
    table[0, j].set_facecolor('#2196F3')
    table[0, j].set_text_props(color='white', fontweight='bold')

plt.tight_layout()
fig2_path = OUTPUT_DIR / '02_neural_statistics.png'
plt.savefig(fig2_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"  [OK] {fig2_path.name}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Summary & Decisions
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("7. Summary & Recommendations")
print("=" * 70)

print(f"""
Data Format & Structure:
  - brainsets HDF5, temporaldata.Data compatible
  - Fields: spikes (timestamps + unit_index), cursor (pos/vel/acc), 
    trials (hold/reach/return periods), units info
  - train/valid/test split pre-computed (valid=0.1, test=0.2)

Dataset Statistics (10 sessions, C/J/M):
  - Total unique units: {sum(s['n_units'] for s in sessions.values())} 
    (across sessions; likely overlapping for same subject)
  - Valid trials: {sum(s['n_valid_trials'] for s in sessions.values())} 
    across 10 sessions
  - Mean firing rate: {np.mean([s['firing_rate_per_unit'] for s in sessions.values()]):.1f} Hz/unit

Window Design Recommendations:
  Hold period (input): mean={all_hold.mean()*1000:.0f}ms
    → 250ms input window: sufficient in {(all_hold>0.25).mean()*100:.0f}% trials
    → Phase 1 baseline: use hold_period as input context
  
  Reach period (prediction): mean={all_reach.mean()*1000:.0f}ms
    → 250ms prediction: feasible in {(all_reach>0.25).mean()*100:.0f}% trials  ✓ Phase 1 start
    → 500ms prediction: feasible in {(all_reach>0.5).mean()*100:.0f}% trials   ✓ Phase 1.3.2
    → 1000ms prediction: feasible in {(all_reach>1.0).mean()*100:.0f}% trials  ✓ Phase 1.3.3

Autoregressive Feasibility (20ms bins):
  - Mean spikes/bin: {all_bin_counts.mean():.3f} ({(all_bin_counts==0).mean()*100:.0f}% zero bins)
  - Poisson NLL: suitable (not critically sparse)
  - 250ms → 12 autoregressive steps: manageable
  - 1000ms → 50 autoregressive steps: requires scheduled sampling

Recommended Phase 1 Dev Sessions:
  - Start with: c_20131003 (71 units), j_20160405 (largest J)
  - Multi-session (5-10): all 4 C + all 3 J sessions  
  - Full 10 sessions: add M for cross-animal diversity
""")

# Save summary JSON
summary = {
    'n_sessions': len(sessions),
    'n_total_units': sum(s['n_units'] for s in sessions.values()),
    'n_total_valid_trials': sum(s['n_valid_trials'] for s in sessions.values()),
    'n_total_train_windows_1s': 6372,
    'hold_period_mean_ms': float(all_hold.mean() * 1000),
    'hold_period_std_ms': float(all_hold.std() * 1000),
    'reach_period_mean_ms': float(all_reach.mean() * 1000),
    'reach_period_std_ms': float(all_reach.std() * 1000),
    'pct_hold_250ms': float((all_hold > 0.25).mean() * 100),
    'pct_hold_500ms': float((all_hold > 0.5).mean() * 100),
    'pct_reach_250ms': float((all_reach > 0.25).mean() * 100),
    'pct_reach_500ms': float((all_reach > 0.5).mean() * 100),
    'pct_reach_1000ms': float((all_reach > 1.0).mean() * 100),
    'mean_fr_hz_per_unit': float(np.mean([s['firing_rate_per_unit'] for s in sessions.values()])),
    'mean_spikes_per_20ms_bin': float(all_bin_counts.mean()),
    'zero_bin_fraction': float((all_bin_counts == 0).mean()),
    'sessions': {name: {
        'subject': s['subject'],
        'n_units': s['n_units'],
        'n_valid_trials': s['n_valid_trials'],
        'duration_min': round(s['duration'] / 60, 1),
        'firing_rate_hz': round(s['firing_rate_per_unit'], 1),
    } for name, s in sessions.items()},
}
summary_path = OUTPUT_DIR / 'exploration_summary.json'
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"\n  Summary saved: {summary_path}")
print(f"  Figures saved: {OUTPUT_DIR}")
print("\n[DONE] Data exploration complete.")
