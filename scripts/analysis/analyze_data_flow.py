"""
Perich-Miller 数据在 POYO+ 中的数据流分析脚本

对应 plan.md 任务 0.3.4
- Part A: 时间尺度属性关系可视化
- Part B: 训练/评估中的采样、损失权重、eval_mask 可视化

输出：
  results/figures/data_exploration/03_timescale_relationships.png
  results/figures/data_exploration/04_sampling_windows_overlay.png
  results/figures/data_exploration/05_eval_pipeline_flow.png
  results/figures/data_exploration/data_flow_summary.json
"""

import sys
import json
import math
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

PROJECT_ROOT = Path('/root/autodl-tmp/NeuroHorizon')
sys.path.insert(0, str(PROJECT_ROOT))

from temporaldata import Data, Interval

PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed' / 'perich_miller_population_2018'
OUTPUT_DIR = PROJECT_ROOT / 'results' / 'figures' / 'data_exploration'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SESSION_ID = 'c_20131003_center_out_reaching'


def load_session(session_id):
    """Load a single session HDF5 file."""
    fpath = PROCESSED_DIR / f'{session_id}.h5'
    f = h5py.File(fpath, 'r')
    data = Data.from_hdf5(f, lazy=False)
    return data, f


def interval_stats(interval, name):
    """Compute statistics for an Interval object."""
    if len(interval) == 0:
        return {'name': name, 'n_intervals': 0, 'total_seconds': 0.0,
                'mean_duration': 0.0, 'min_duration': 0.0, 'max_duration': 0.0}
    durations = interval.end - interval.start
    return {
        'name': name,
        'n_intervals': int(len(interval)),
        'total_seconds': float(np.sum(durations)),
        'mean_duration': float(np.mean(durations)),
        'min_duration': float(np.min(durations)),
        'max_duration': float(np.max(durations)),
    }


def draw_intervals(ax, interval, y_center, height, color, alpha=0.7, label=None, hatch=None):
    """Draw interval blocks on a matplotlib axis using broken_barh."""
    if len(interval) == 0:
        return
    xranges = [(s, e - s) for s, e in zip(interval.start, interval.end)]
    ax.broken_barh(xranges, (y_center - height/2, height),
                   facecolors=color, alpha=alpha, edgecolors='none',
                   label=label, hatch=hatch)


def figure1_timescale_relationships(data, summary):
    """Figure 1: Multi-track timeline showing all time-scale attributes."""
    fig, axes = plt.subplots(2, 1, figsize=(20, 16), gridspec_kw={'height_ratios': [1, 1]})

    track_configs = [
        ('domain',              data.domain,                          'domain',            '#888888', None),
        ('trials (valid)',      data.trials.select_by_mask(data.trials.is_valid),  'trials_valid',    '#4CAF50', None),
        ('trials (invalid)',    data.trials.select_by_mask(~data.trials.is_valid), 'trials_invalid',  '#F44336', None),
        ('hold_period',         data.movement_phases.hold_period,     'hold',              '#2196F3', None),
        ('reach_period',        data.movement_phases.reach_period,    'reach',             '#FF9800', None),
        ('return_period',       data.movement_phases.return_period,   'return',            '#4CAF50', None),
        ('random_period',       data.movement_phases.random_period,   'random',            '#9E9E9E', None),
        ('outlier_segments',    data.cursor_outlier_segments,         'outlier',           '#F44336', '///'),
        ('train_domain',        data.train_domain,                    'train',             '#009688', None),
        ('valid_domain',        data.valid_domain,                    'valid',             '#9C27B0', None),
        ('test_domain',         data.test_domain,                     'test',              '#E91E63', None),
    ]

    n_tracks = len(track_configs)
    track_labels = [tc[0] for tc in track_configs]

    # Determine zoom range: find a region with several trials
    valid_trials = data.trials.select_by_mask(data.trials.is_valid)
    # Pick trials 10-20 for zoom
    zoom_trial_start = max(10, 0)
    zoom_trial_end = min(zoom_trial_start + 12, len(valid_trials))
    zoom_start = valid_trials.start[zoom_trial_start] - 1.0
    zoom_end = valid_trials.end[zoom_trial_end - 1] + 1.0

    for panel_idx, ax in enumerate(axes):
        for i, (label, interval, key, color, hatch) in enumerate(track_configs):
            y_pos = n_tracks - i
            draw_intervals(ax, interval, y_pos, 0.6, color, alpha=0.75, hatch=hatch)

            # Add statistics annotation on the right
            stats = summary.get(key, {})
            n_int = stats.get('n_intervals', 0)
            total = stats.get('total_seconds', 0.0)
            if panel_idx == 0:
                ax.text(data.domain.end[0] + 2, y_pos, f'n={n_int}, {total:.1f}s',
                        va='center', fontsize=8, color='gray')

        ax.set_yticks(range(1, n_tracks + 1))
        ax.set_yticklabels(track_labels[::-1], fontsize=9)
        ax.set_ylim(0.3, n_tracks + 0.7)

        if panel_idx == 0:
            ax.set_xlim(data.domain.start[0], data.domain.end[0] + 50)
            ax.set_title(f'Panel A: Full Timeline ({SESSION_ID})', fontsize=12, fontweight='bold')
        else:
            ax.set_xlim(zoom_start, zoom_end)
            ax.set_title(f'Panel B: Zoomed View (t = {zoom_start:.1f}s to {zoom_end:.1f}s)',
                         fontsize=12, fontweight='bold')
            # Add vertical lines at trial boundaries in zoom view
            for s, e in zip(valid_trials.start, valid_trials.end):
                if zoom_start <= s <= zoom_end:
                    ax.axvline(s, color='gray', alpha=0.3, linewidth=0.5, linestyle='--')

        ax.set_xlabel('Time (seconds)', fontsize=10)
        ax.grid(axis='x', alpha=0.2)

    plt.tight_layout()
    out_path = OUTPUT_DIR / '03_timescale_relationships.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")
    return zoom_start, zoom_end


def figure2_sampling_windows(data, zoom_start, zoom_end, summary):
    """Figure 2: Sampling windows overlaid on movement phases with weights/eval_mask."""
    fig, axes = plt.subplots(3, 1, figsize=(20, 12),
                             gridspec_kw={'height_ratios': [3, 1, 1]})

    ax_main, ax_weight, ax_eval = axes

    # --- Main panel: movement phases background + sampling windows ---
    phases = [
        ('hold_period',   data.movement_phases.hold_period,   '#2196F3', 0.3),
        ('reach_period',  data.movement_phases.reach_period,  '#FF9800', 0.3),
        ('return_period', data.movement_phases.return_period, '#4CAF50', 0.3),
        ('random_period', data.movement_phases.random_period, '#9E9E9E', 0.2),
    ]

    # Draw movement phase backgrounds (full height)
    for name, interval, color, alpha in phases:
        for s, e in zip(interval.start, interval.end):
            if e > zoom_start and s < zoom_end:
                ax_main.axvspan(max(s, zoom_start), min(e, zoom_end),
                                color=color, alpha=alpha)

    # Draw train_domain region
    for s, e in zip(data.train_domain.start, data.train_domain.end):
        if e > zoom_start and s < zoom_end:
            ax_main.axvspan(max(s, zoom_start), min(e, zoom_end),
                            color='#009688', alpha=0.08)

    # Simulate training windows (RandomFixedWindowSampler)
    window_length = 1.0
    train_windows = []
    for s, e in zip(data.train_domain.start, data.train_domain.end):
        if e - s < window_length:
            continue
        offset = 0.3 * window_length  # simulate random offset
        t = s + offset
        while t + window_length <= e:
            if t + window_length > zoom_start and t < zoom_end:
                train_windows.append((t, t + window_length))
            t += window_length

    y_train = 2.5
    for i, (ws, we) in enumerate(train_windows[:8]):
        rect = plt.Rectangle((ws, y_train - 0.3), we - ws, 0.6,
                              linewidth=1.5, edgecolor='#00796B', facecolor='#009688',
                              alpha=0.25)
        ax_main.add_patch(rect)
        ax_main.text(ws + 0.05, y_train, f'TW{i+1}', fontsize=7, color='#00796B',
                     va='center')

    # Simulate evaluation windows (DistributedStitchingFixedWindowSampler)
    step = window_length / 2
    eval_windows = []
    for s, e in zip(data.valid_domain.start, data.valid_domain.end):
        if e - s < window_length:
            continue
        t = s
        while t + window_length <= e + 1e-9:
            if t + window_length > zoom_start and t < zoom_end:
                eval_windows.append((t, t + window_length))
            t += step
        if len(eval_windows) > 0 and eval_windows[-1][1] < e:
            eval_windows.append((e - window_length, e))

    y_eval = 1.5
    for i, (ws, we) in enumerate(eval_windows[:8]):
        rect = plt.Rectangle((ws, y_eval - 0.3 + (i % 2) * 0.05), we - ws, 0.55,
                              linewidth=1.5, edgecolor='#7B1FA2', facecolor='#9C27B0',
                              alpha=0.15, linestyle='--')
        ax_main.add_patch(rect)
        ax_main.text(ws + 0.05, y_eval + (i % 2) * 0.05, f'EW{i+1}',
                     fontsize=7, color='#7B1FA2', va='center')

    # Trial boundary markers
    valid_trials = data.trials.select_by_mask(data.trials.is_valid)
    for s in valid_trials.start:
        if zoom_start <= s <= zoom_end:
            ax_main.axvline(s, color='gray', alpha=0.4, linewidth=0.8, linestyle=':')

    ax_main.set_xlim(zoom_start, zoom_end)
    ax_main.set_ylim(0.8, 3.5)
    ax_main.set_yticks([1.5, 2.5])
    ax_main.set_yticklabels(['Eval Windows', 'Train Windows'], fontsize=9)
    ax_main.set_title('Sampling Windows over Movement Phases (Zoomed View)', fontsize=12, fontweight='bold')

    legend_patches = [
        mpatches.Patch(color='#2196F3', alpha=0.3, label='hold_period'),
        mpatches.Patch(color='#FF9800', alpha=0.3, label='reach_period'),
        mpatches.Patch(color='#4CAF50', alpha=0.3, label='return_period'),
        mpatches.Patch(color='#009688', alpha=0.25, label='Train Window (1.0s)'),
        mpatches.Patch(color='#9C27B0', alpha=0.15, label='Eval Window (1.0s, step=0.5s)'),
    ]
    ax_main.legend(handles=legend_patches, loc='upper right', fontsize=8)

    # --- Weight profile panel ---
    cursor_ts = data.cursor.timestamps
    mask_zoom = (cursor_ts >= zoom_start) & (cursor_ts <= zoom_end)
    ts_zoom = cursor_ts[mask_zoom]

    weights = np.ones_like(cursor_ts, dtype=np.float32)
    # Apply weight rules from config
    weight_configs = [
        (data.movement_phases.reach_period, 5.0),
        (data.movement_phases.hold_period, 0.1),
        (data.movement_phases.return_period, 1.0),
        (data.movement_phases.random_period, 1.0),
        (data.cursor_outlier_segments, 0.0),
    ]
    for interval, w in weight_configs:
        if len(interval) == 0:
            continue
        for s, e in zip(interval.start, interval.end):
            in_mask = (cursor_ts >= s) & (cursor_ts < e)
            weights[in_mask] *= w

    weights_zoom = weights[mask_zoom]
    ax_weight.fill_between(ts_zoom, 0, weights_zoom, color='#FF5722', alpha=0.4, step='mid')
    ax_weight.set_xlim(zoom_start, zoom_end)
    ax_weight.set_ylabel('Loss Weight', fontsize=9)
    ax_weight.set_title('Per-timestamp Loss Weights (reach=5.0, hold=0.1, return=1.0, outlier=0.0)', fontsize=10)
    ax_weight.set_ylim(-0.2, 6.0)
    ax_weight.axhline(1.0, color='gray', alpha=0.3, linestyle='--')
    ax_weight.axhline(5.0, color='gray', alpha=0.3, linestyle='--')
    ax_weight.grid(axis='x', alpha=0.2)

    # --- Eval mask panel ---
    eval_mask = np.zeros_like(cursor_ts, dtype=np.float32)
    reach = data.movement_phases.reach_period
    for s, e in zip(reach.start, reach.end):
        in_mask = (cursor_ts >= s) & (cursor_ts < e)
        eval_mask[in_mask] = 1.0

    eval_mask_zoom = eval_mask[mask_zoom]
    ax_eval.fill_between(ts_zoom, 0, eval_mask_zoom, color='#E91E63', alpha=0.4, step='mid')
    ax_eval.set_xlim(zoom_start, zoom_end)
    ax_eval.set_ylim(-0.1, 1.3)
    ax_eval.set_ylabel('Eval Mask', fontsize=9)
    ax_eval.set_xlabel('Time (seconds)', fontsize=10)
    ax_eval.set_title('Eval Mask (1 = within reach_period, 0 = excluded from R\u00b2)', fontsize=10)
    ax_eval.grid(axis='x', alpha=0.2)

    plt.tight_layout()
    out_path = OUTPUT_DIR / '04_sampling_windows_overlay.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


def figure3_eval_pipeline(data, summary):
    """Figure 3: Evaluation pipeline flow diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(18, 14))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 14)
    ax.axis('off')
    ax.set_title('POYO+ Evaluation Pipeline: Data Flow from HDF5 to R\u00b2', fontsize=14, fontweight='bold', pad=20)

    # --- Flow boxes ---
    boxes = [
        (2, 12.5, 14, 1.0,  'Session HDF5',
         f'e.g. {SESSION_ID}.h5\ndomain: {summary["domain"]["total_seconds"]:.1f}s, '
         f'trials: {summary["trials"]["total"]} ({summary["trials"]["valid"]} valid)',
         '#E3F2FD'),
        (2, 10.8, 14, 1.0,  'Dataset.get_sampling_intervals()',
         f'split="valid" \u2192 valid_domain: {summary["valid_domain"]["n_intervals"]} intervals, '
         f'{summary["valid_domain"]["total_seconds"]:.1f}s total\n'
         f'split="train" \u2192 train_domain: {summary["train_domain"]["n_intervals"]} intervals, '
         f'{summary["train_domain"]["total_seconds"]:.1f}s total',
         '#E8F5E9'),
        (2, 9.0, 6.5, 1.2,  'Training: RandomFixedWindowSampler',
         f'window=1.0s, random offset\n'
         f'N = floor(interval_len/window_len)\n'
         f'approx {summary["sampling"]["train_windows_estimate"]} windows/epoch\n'
         f'windows NOT aligned to trial boundaries',
         '#FFF3E0'),
        (9.5, 9.0, 6.5, 1.2, 'Eval: DistributedStitchingFixedWindowSampler',
         f'window=1.0s, step=0.5s (50% overlap)\n'
         f'approx {summary["sampling"]["eval_windows_estimate"]} windows\n'
         f'overlapping windows on same GPU rank\n'
         f'sequence_index tracks stitching groups',
         '#F3E5F5'),
        (2, 7.2, 6.5, 1.0,  'Loss = MSE(pred, target) * weight',
         f'Weights per timestamp:\n'
         f'reach=5.0, hold=0.1, return=1.0, outlier=0.0\n'
         f'loss = sum(taskwise_loss * n_seq) / batch_size',
         '#FFEBEE'),
        (9.5, 7.2, 6.5, 1.0, 'model.forward() per window',
         'spikes \u2192 encoder \u2192 latent grid (8Hz)\n'
         '\u2192 processor \u2192 decoder \u2192 cursor_velocity_2d',
         '#E0F7FA'),
        (9.5, 5.5, 6.5, 1.0, 'Apply eval_mask',
         'Config: eval_interval = movement_phases.reach_period\n'
         'Only reach_period timestamps kept for metric\n'
         '(see note on eval_mask implementation below)',
         '#FFF9C4'),
        (9.5, 3.8, 6.5, 1.0, 'Stitch overlapping predictions',
         'stitch(timestamps, values):\n'
         '  unique_timestamps \u2192 mean-pool predictions\n'
         '  at duplicate timestamps across windows',
         '#E8EAF6'),
        (9.5, 2.1, 6.5, 1.0, 'Per-session R\u00b2 \u2192 average across sessions',
         f'R\u00b2 = torchmetrics.R2Score\n'
         f'computed per session, then averaged\n'
         f'{summary["valid_domain"]["n_intervals"]} validation trial intervals across sessions',
         '#E0F2F1'),
    ]

    for x, y, w, h, title, desc, color in boxes:
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                              facecolor=color, edgecolor='#455A64', linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x + 0.3, y + h - 0.2, title, fontsize=10, fontweight='bold',
                va='top', ha='left')
        ax.text(x + 0.3, y + h - 0.45, desc, fontsize=7.5, va='top', ha='left',
                color='#37474F', linespacing=1.4)

    # --- Arrows ---
    arrows = [
        (9, 12.5, 9, 11.85),     # HDF5 -> get_sampling_intervals
        (5.25, 10.8, 5.25, 10.25),  # get_sampling_intervals -> train sampler
        (12.75, 10.8, 12.75, 10.25), # get_sampling_intervals -> eval sampler
        (5.25, 9.0, 5.25, 8.25),    # train sampler -> loss
        (12.75, 9.0, 12.75, 8.25),  # eval sampler -> model forward
        (12.75, 7.2, 12.75, 6.55),  # model forward -> eval_mask
        (12.75, 5.5, 12.75, 4.85),  # eval_mask -> stitch
        (12.75, 3.8, 12.75, 3.15),  # stitch -> R2
    ]

    for x1, y1, x2, y2 in arrows:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#455A64', lw=1.5))

    # --- Note box ---
    note_x, note_y = 2, 2.0
    note = ('NOTE on eval_mask implementation:\n'
            'In multitask_readout.py:237, the code reads:\n'
            '  eval_interval_key = data.config.get("eval_interval", None)\n'
            'This reads from top-level config dict. However in the YAML config,\n'
            'eval_interval is nested inside multitask_readout list items.\n'
            'Therefore data.config.get("eval_interval") returns None,\n'
            'and eval_mask defaults to all-True (all timestamps evaluated).\n'
            'The readout_config.get("eval_interval") would be correct.')
    rect_note = FancyBboxPatch((note_x, note_y - 0.3), 6.5, 2.2,
                               boxstyle="round,pad=0.15",
                               facecolor='#FFF8E1', edgecolor='#F57F17',
                               linewidth=1.5, linestyle='--')
    ax.add_patch(rect_note)
    ax.text(note_x + 0.2, note_y + 1.7, note, fontsize=7, va='top', ha='left',
            color='#E65100', family='monospace', linespacing=1.4)

    plt.tight_layout()
    out_path = OUTPUT_DIR / '05_eval_pipeline_flow.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    print(f"Loading session: {SESSION_ID}")
    data, f = load_session(SESSION_ID)
    print(f"Session loaded successfully.\n")

    # --- Compute summary statistics ---
    summary = {}

    # domain
    summary['domain'] = interval_stats(data.domain, 'domain')

    # trials
    valid_mask = data.trials.is_valid
    valid_trials = data.trials.select_by_mask(valid_mask)
    invalid_trials = data.trials.select_by_mask(~valid_mask)
    summary['trials'] = {
        'total': int(len(data.trials)),
        'valid': int(np.sum(valid_mask)),
        'invalid': int(np.sum(~valid_mask)),
    }
    summary['trials_valid'] = interval_stats(valid_trials, 'trials_valid')
    summary['trials_invalid'] = interval_stats(invalid_trials, 'trials_invalid')

    # movement phases
    mp = data.movement_phases
    for phase_name in ['hold_period', 'reach_period', 'return_period', 'random_period']:
        phase = getattr(mp, phase_name, None)
        if phase is not None:
            summary[phase_name] = interval_stats(phase, phase_name)
        else:
            summary[phase_name] = interval_stats(Interval(start=np.array([]), end=np.array([])), phase_name)

    # outlier segments
    summary['outlier'] = interval_stats(data.cursor_outlier_segments, 'cursor_outlier_segments')

    # domains
    summary['train_domain'] = interval_stats(data.train_domain, 'train_domain')
    summary['valid_domain'] = interval_stats(data.valid_domain, 'valid_domain')
    summary['test_domain'] = interval_stats(data.test_domain, 'test_domain')

    # sampling simulation
    window_length = 1.0
    train_windows = 0
    for s, e in zip(data.train_domain.start, data.train_domain.end):
        if e - s >= window_length:
            train_windows += math.floor((e - s) / window_length)

    eval_windows = 0
    step = 0.5
    for s, e in zip(data.valid_domain.start, data.valid_domain.end):
        if e - s >= window_length:
            eval_windows += int((e - s - window_length + 1e-9) / step) + 1

    summary['sampling'] = {
        'window_length': window_length,
        'train_windows_estimate': train_windows,
        'eval_windows_estimate': eval_windows,
        'eval_step': step,
        'trial_boundary_awareness': 'None - pure temporal domain sampling',
    }

    # eval_mask analysis
    summary['eval_mask_analysis'] = {
        'intended_eval_interval': 'movement_phases.reach_period',
        'config_read_method': 'data.config.get("eval_interval") -- reads top-level config',
        'yaml_location': 'nested inside multitask_readout list items',
        'effective_result': 'None (eval_mask defaults to all-True)',
        'impact': 'R^2 computed on ALL timestamps, not just reach_period',
    }

    # --- Print summary ---
    print("=" * 70)
    print(f"DATA FLOW ANALYSIS: {SESSION_ID}")
    print("=" * 70)

    print(f"\n--- Domain ---")
    d = summary['domain']
    print(f"  Full recording: {d['total_seconds']:.1f}s "
          f"({d['n_intervals']} interval)")

    print(f"\n--- Trials ---")
    t = summary['trials']
    print(f"  Total: {t['total']}, Valid: {t['valid']}, Invalid: {t['invalid']}")
    tv = summary['trials_valid']
    print(f"  Valid trials: mean {tv['mean_duration']:.2f}s, "
          f"range [{tv['min_duration']:.2f}, {tv['max_duration']:.2f}]s, "
          f"total {tv['total_seconds']:.1f}s")

    print(f"\n--- Movement Phases ---")
    for phase in ['hold_period', 'reach_period', 'return_period', 'random_period']:
        ps = summary[phase]
        print(f"  {phase}: n={ps['n_intervals']}, "
              f"mean={ps['mean_duration']:.3f}s, total={ps['total_seconds']:.1f}s")

    print(f"\n--- Outlier Segments ---")
    o = summary['outlier']
    print(f"  n={o['n_intervals']}, total={o['total_seconds']:.1f}s")

    print(f"\n--- Train/Valid/Test Domains ---")
    for dom_name in ['train_domain', 'valid_domain', 'test_domain']:
        ds = summary[dom_name]
        print(f"  {dom_name}: {ds['n_intervals']} intervals, "
              f"total={ds['total_seconds']:.1f}s")

    print(f"\n--- Sampling Simulation (window_length={window_length}s) ---")
    ss = summary['sampling']
    print(f"  Train: approx {ss['train_windows_estimate']} windows/epoch")
    print(f"  Eval: approx {ss['eval_windows_estimate']} windows (step={ss['eval_step']}s)")
    print(f"  Trial boundary awareness: {ss['trial_boundary_awareness']}")

    print(f"\n--- Eval Mask Analysis ---")
    em = summary['eval_mask_analysis']
    for k, v in em.items():
        print(f"  {k}: {v}")

    print(f"\n--- Relationship Summary ---")
    print("  1. domain = cursor.domain (entire recording timespan)")
    print("  2. trials = behavioral episodes (target_on -> stop_time -> next trial)")
    print("     - valid trials: success + valid target + duration in [0.5s, 6.0s]")
    print("  3. movement_phases = from valid trials only:")
    print("     - hold_period: target_on_time -> go_cue_time")
    print("     - reach_period: go_cue_time -> stop_time")
    print("     - return_period: stop_time -> trial_end")
    print("     - random_period: domain - (hold + reach + return + invalid trials)")
    print("     - all phases have cursor_outlier_segments removed")
    print("  4. Split creation (pipeline):")
    print("     - test_trials = 20% of valid trials (random)")
    print("     - valid_trials = 10% of remaining valid trials")
    print("     - train_domain = domain - (valid_trials | test_trials).dilate(1.0s)")
    print("     - 1s dilation creates safety gaps preventing data leakage")
    print("  5. Sampling is purely temporal-domain-based:")
    print("     - Windows can cross trial boundaries")
    print("     - Model never explicitly sees trial start/end")

    # --- Save JSON ---
    json_path = OUTPUT_DIR / 'data_flow_summary.json'
    with open(json_path, 'w', encoding='utf-8') as jf:
        json.dump(summary, jf, indent=2, ensure_ascii=False)
    print(f"\nSaved summary: {json_path}")

    # --- Generate figures ---
    print(f"\n{'='*70}")
    print("Generating figures...")
    print("=" * 70)

    zoom_start, zoom_end = figure1_timescale_relationships(data, summary)
    figure2_sampling_windows(data, zoom_start, zoom_end, summary)
    figure3_eval_pipeline(data, summary)

    f.close()
    print(f"\nAll figures saved to: {OUTPUT_DIR}")
    print("Done!")


if __name__ == '__main__':
    main()
