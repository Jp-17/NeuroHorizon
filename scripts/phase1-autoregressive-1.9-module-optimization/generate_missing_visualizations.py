"""
Generate missing 1.9 module optimization visualizations:
1. Per-experiment: fp_bps_by_window.png, per_bin_fp_bps_decay.png, config_timeline.png
2. Global: summary_table.png
"""
import os, json, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import yaml

MAIN_LOG = '/root/autodl-tmp/NeuroHorizon/results/logs/phase1-autoregressive-1.9-module-optimization'
DEV_LOG  = '/root/autodl-tmp/NeuroHorizon_dev_20260320_decoder_scheduled_sampling/results/logs/phase1-autoregressive-1.9-module-optimization'
FIG_BASE = '/root/autodl-tmp/NeuroHorizon_dev_20260320_decoder_scheduled_sampling/results/figures/phase1-autoregressive-1.9-module-optimization'
TSV_PATH = '/root/autodl-tmp/NeuroHorizon_dev_20260320_decoder_scheduled_sampling/cc_todo/phase1-autoregressive/1.9-module-optimization/results.tsv'

# ─── experiment catalog ──────────────────────────────────────────────────────
EXPS = [
    dict(name='baseline_v2',
         label='baseline_v2',
         log_root=None,   # no logs stored in 1.9 dir
         windows=['250ms','500ms','1000ms'],
         rollout_test={'250ms':0.2115,'500ms':0.1744,'1000ms':0.1317},
         per_bin=None),  # no per-bin stored
    dict(name='20260312_prediction_memory_decoder',
         label='PM Decoder',
         log_root=MAIN_LOG,
         winner_subdir=None,
         windows=['250ms','500ms','1000ms']),
    dict(name='20260313_local_prediction_memory',
         label='Local PM',
         log_root=MAIN_LOG,
         winner_subdir=None,
         windows=['250ms','500ms','1000ms']),
    dict(name='20260313_prediction_memory_alignment',
         label='PM Alignment',
         log_root=MAIN_LOG,
         winner_subdir=None,
         windows=['250ms','500ms','1000ms']),
    dict(name='20260313_prediction_memory_alignment_tuning',
         label='PM Align Tune',
         log_root=MAIN_LOG,
         winner_subdir=None,
         windows=['250ms','500ms','1000ms']),
    dict(name='20260320_decoder_scheduled_sampling',
         label='Decoder SS',
         log_root=DEV_LOG,
         winner_subdir='decoder_ss_linear_0_to_050',
         windows=['250ms']),  # 500/1000 pending
]

COLORS = ['#888888','#4C72B0','#DD8452','#55A868','#C44E52','#8172B2']
WINDOW_MS = [250, 500, 1000]

# ─── helpers ─────────────────────────────────────────────────────────────────
def load_eval_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

def get_latest_metrics_csv(exp_log_root, window):
    pattern = os.path.join(exp_log_root, window, 'lightning_logs', 'version_*', 'metrics.csv')
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None

def get_latest_hparams(exp_log_root, window):
    pattern = os.path.join(exp_log_root, window, 'lightning_logs', 'version_*', 'hparams.yaml')
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None

def load_epoch_metrics(csv_path):
    df = pd.read_csv(csv_path)
    val_rows = df[df['val/fp_bps'].notna()].copy()
    val_rows = val_rows.sort_values('epoch').reset_index(drop=True)
    # get train_loss per epoch (last step in epoch)
    train_rows = df[df['train_loss'].notna()].copy()
    # get lr per epoch: take last lr row before each val epoch
    lr_rows = df[df['lr-SparseLamb/pg1'].notna()].copy()
    return val_rows, train_rows, lr_rows

def get_exp_log_root(exp, window=None):
    if exp['log_root'] is None:
        return None
    if exp['winner_subdir']:
        if window:
            return os.path.join(exp['log_root'], exp['name'], exp['winner_subdir'], window)
        return os.path.join(exp['log_root'], exp['name'], exp['winner_subdir'])
    if window:
        return os.path.join(exp['log_root'], exp['name'], window)
    return os.path.join(exp['log_root'], exp['name'])

def get_eval_json(exp, window, kind='rollout', split='test'):
    log_root = get_exp_log_root(exp)
    if log_root is None:
        return None
    fname = f'eval_{kind}_best_{split}.json'
    path = os.path.join(log_root, window, fname) if not exp['winner_subdir'] else os.path.join(log_root, window, fname)
    # for non-winner exps the path is log_root/window/fname
    if exp['winner_subdir']:
        path = os.path.join(exp['log_root'], exp['name'], exp['winner_subdir'], window, fname)
    else:
        path = os.path.join(exp['log_root'], exp['name'], window, fname)
    return load_eval_json(path)

# ─── 1. fp-bps trend by window ───────────────────────────────────────────────
def make_fp_bps_by_window(exp, fig_dir):
    windows_ms = [int(w.replace('ms','')) for w in exp['windows']]
    rollout_vals = []
    tf_valid_vals = []
    tf_test_vals = []
    for w in exp['windows']:
        r = get_eval_json(exp, w, 'rollout', 'test')
        r_v = get_eval_json(exp, w, 'rollout', 'valid') if exp['log_root'] else None
        tf = get_eval_json(exp, w, 'teacher_forced', 'test')
        rollout_vals.append(r['continuous']['fp_bps'] if r else None)
        tf_test_vals.append(tf['continuous']['fp_bps'] if tf else None)
    # baseline_v2 reference
    bv2 = {'250ms':0.2115,'500ms':0.1744,'1000ms':0.1317}

    fig, ax = plt.subplots(figsize=(7,4))
    x = np.array([int(w.replace('ms','')) for w in exp['windows']])
    ax.plot(x, [v for v in rollout_vals], 'o-', color='#4C72B0', label='rollout test', linewidth=2)
    ax.plot(x, [v for v in tf_test_vals], 's--', color='#DD8452', label='TF test', linewidth=1.5)
    bv2_x = [int(w.replace('ms','')) for w in exp['windows']]
    bv2_y = [bv2.get(w) for w in exp['windows']]
    ax.plot(bv2_x, bv2_y, '^:', color='#888888', label='baseline_v2 rollout', linewidth=1.5, alpha=0.7)
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.4)
    ax.set_xlabel('Prediction Window (ms)')
    ax.set_ylabel('fp-bps (test)')
    ax.set_xticks(bv2_x)
    ax.set_title(f"fp-bps by Prediction Window\n{exp['name']}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(fig_dir, 'fp_bps_by_window.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'  saved: {out}')

# ─── 2. per-bin fp-bps decay curves ─────────────────────────────────────────
def make_per_bin_decay(exp, fig_dir):
    windows = exp['windows']
    n = len(windows)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 4), sharey=False)
    if n == 1:
        axes = [axes]
    bv2_per_bin_250 = {0:0.2947,1:0.2882,2:0.2699,3:0.2510,4:0.2428,5:0.2325,6:0.2151,7:0.2059,8:0.2013,9:0.1901,10:0.1860,11:0.2522}

    for i, w in enumerate(windows):
        ax = axes[i]
        r = get_eval_json(exp, w, 'rollout', 'test')
        if r and 'per_bin_fp_bps' in r['continuous']:
            bins = r['continuous']['per_bin_fp_bps']
            idx = sorted(int(k) for k in bins.keys())
            vals = [bins[str(k)] for k in idx]
            bin_ms = [int(w.replace('ms','')) / len(idx) * (k+1) for k in idx]
            ax.plot(bin_ms, vals, 'o-', color='#4C72B0', linewidth=2, label=exp['label'])
            if w == '250ms':
                bv2_x = [250/12*(k+1) for k in range(12)]
                bv2_y = [bv2_per_bin_250.get(k, np.nan) for k in range(12)]
                ax.plot(bv2_x, bv2_y, '^:', color='#888888', linewidth=1.5, label='baseline_v2', alpha=0.7)
        ax.axhline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.4)
        ax.set_xlabel('Time into prediction (ms)')
        ax.set_ylabel('fp-bps')
        ax.set_title(f'Per-bin fp-bps: {w}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'Per-bin fp-bps Decay — {exp["name"]}', fontsize=11)
    plt.tight_layout()
    out = os.path.join(fig_dir, 'per_bin_fp_bps_decay.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'  saved: {out}')

# ─── 3. config timeline ──────────────────────────────────────────────────────
def make_config_timeline(exp, fig_dir):
    # use 250ms window as representative
    w = '250ms'
    if exp['winner_subdir']:
        win_path = os.path.join(exp['log_root'], exp['name'], exp['winner_subdir'], w)
    else:
        win_path = os.path.join(exp['log_root'], exp['name'], w) if exp['log_root'] else None

    if win_path is None:
        return

    # find latest metrics csv and hparams
    metrics_files = sorted(glob.glob(os.path.join(win_path, 'lightning_logs', 'version_*', 'metrics.csv')))
    hparams_files = sorted(glob.glob(os.path.join(win_path, 'lightning_logs', 'version_*', 'hparams.yaml')))
    if not metrics_files or not hparams_files:
        return

    df = pd.read_csv(metrics_files[-1])
    with open(hparams_files[-1]) as f:
        hp = yaml.safe_load(f)

    # extract per-epoch LR
    lr_rows = df[df['lr-SparseLamb/pg1'].notna()].copy()
    # aggregate: take median lr per epoch (from step-level records)
    if 'epoch' in lr_rows.columns:
        lr_epoch = lr_rows.groupby('step')['lr-SparseLamb/pg1'].first().reset_index()
    else:
        lr_epoch = lr_rows

    total_steps = lr_epoch['step'].max()
    total_epochs = hp.get('epochs', 300)
    warmup_steps = int(0.05 * total_steps)  # typically 5% warmup
    lr_decay_start = hp['optim'].get('lr_decay_start', 0.5)
    weight_decay = hp['optim'].get('weight_decay', 0.0001)
    batch_size = hp.get('batch_size', 64)
    base_lr = hp['optim'].get('base_lr', 3.125e-5)

    steps = lr_epoch['step'].values
    lrs = lr_epoch['lr-SparseLamb/pg1'].values

    fig, axes = plt.subplots(4, 1, figsize=(9, 8), sharex=True)

    # (a) LR schedule
    ax = axes[0]
    ax.plot(steps / total_steps * total_epochs, lrs, color='#4C72B0', linewidth=1.5)
    ax.set_ylabel('LR (SparseLamb)', fontsize=9)
    ax.set_title('Config Timeline (250ms representative)', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.annotate(f'base_lr={base_lr:.2e}  decay_start={lr_decay_start}',
                xy=(0.02, 0.85), xycoords='axes fraction', fontsize=8, color='#4C72B0')

    # (b) Weight decay (constant)
    ax = axes[1]
    epoch_arr = np.linspace(0, total_epochs, 200)
    ax.plot(epoch_arr, np.full_like(epoch_arr, weight_decay), color='#DD8452', linewidth=1.5)
    ax.set_ylabel('Weight Decay', fontsize=9)
    ax.set_ylim(0, weight_decay * 2)
    ax.annotate(f'wd={weight_decay}', xy=(0.02, 0.7), xycoords='axes fraction', fontsize=8, color='#DD8452')
    ax.grid(True, alpha=0.3)

    # (c) Effective batch size (constant unless gradient accumulation used)
    ax = axes[2]
    ax.plot(epoch_arr, np.full_like(epoch_arr, float(batch_size)), color='#55A868', linewidth=1.5)
    ax.set_ylabel('Batch Size', fontsize=9)
    ax.set_ylim(0, batch_size * 1.5)
    ax.annotate(f'bs={batch_size}', xy=(0.02, 0.7), xycoords='axes fraction', fontsize=8, color='#55A868')
    ax.grid(True, alpha=0.3)

    # (d) Warmup progress (fraction of warmup completed)
    ax = axes[3]
    warmup_frac = np.clip(steps / max(warmup_steps, 1), 0, 1)
    warmup_epoch = steps / total_steps * total_epochs
    ax.plot(warmup_epoch, warmup_frac, color='#C44E52', linewidth=1.5)
    ax.set_ylabel('Warmup Progress', fontsize=9)
    ax.set_xlabel('Epoch', fontsize=9)
    ax.set_ylim(-0.05, 1.1)
    ax.axhline(1.0, color='gray', linewidth=0.5, linestyle='--')
    ax.annotate(f'warmup≈5% of total steps', xy=(0.02, 0.6), xycoords='axes fraction', fontsize=8, color='#C44E52')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(fig_dir, 'config_timeline.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'  saved: {out}')

# ─── training curves for decoder_ss (winner only) ───────────────────────────
def make_training_curves_decoder_ss(exp, fig_dir):
    w = '250ms'
    if exp['winner_subdir']:
        win_path = os.path.join(exp['log_root'], exp['name'], exp['winner_subdir'], w)
    else:
        return
    metrics_files = sorted(glob.glob(os.path.join(win_path, 'lightning_logs', 'version_*', 'metrics.csv')))
    if not metrics_files:
        return
    df = pd.read_csv(metrics_files[-1])
    val_rows = df[df['val/fp_bps'].notna()].sort_values('epoch')
    train_rows = df[df['train_loss'].notna()]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # train loss
    ax = axes[0]
    if not train_rows.empty:
        # aggregate by epoch: group step-level rows
        tl_by_step = train_rows[['step','train_loss']].dropna()
        # map step to epoch fraction
        total_steps = tl_by_step['step'].max()
        total_epochs = 300
        tl_by_step = tl_by_step.copy()
        tl_by_step['epoch_frac'] = tl_by_step['step'] / total_steps * total_epochs
        ax.plot(tl_by_step['epoch_frac'], tl_by_step['train_loss'], alpha=0.6, linewidth=0.8, color='#4C72B0')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Train Loss')
    ax.set_title('Train Loss (250ms winner)')
    ax.grid(True, alpha=0.3)

    # val fp-bps
    ax = axes[1]
    ax.plot(val_rows['epoch'], val_rows['val/fp_bps'], 'o-', color='#55A868', linewidth=1.5, markersize=3)
    ax.axhline(0.2115, color='#888888', linestyle='--', linewidth=1, label='baseline_v2 (0.2115)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Val fp-bps (rollout)')
    ax.set_title('Val fp-bps (250ms winner)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # val R²
    ax = axes[2]
    if 'val/r2' in val_rows.columns:
        ax.plot(val_rows['epoch'], val_rows['val/r2'], 'o-', color='#C44E52', linewidth=1.5, markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Val R²')
    ax.set_title('Val R² (250ms winner)')
    ax.grid(True, alpha=0.3)

    # annotation: which variant this is
    fig.suptitle(f'Training Curves — decoder_ss_linear_0_to_050 (winner)\n{exp["name"]}', fontsize=10)
    plt.tight_layout()
    out = os.path.join(fig_dir, 'training_curves.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'  saved: {out}')

# ─── global summary table ────────────────────────────────────────────────────
def make_summary_table():
    tsv = pd.read_csv(TSV_PATH, sep='\t')
    # relevant columns
    cols_show = ['name','best_test_fp_bps_250ms','best_test_fp_bps_500ms','best_test_fp_bps_1000ms',
                 'fp_bps_250ms','fp_bps_500ms','fp_bps_1000ms',
                 'best_ckpt_250ms','date','description']
    # filter to 1.9 exps + baseline + benchmarks
    row_labels = [
        'baseline_v2','benchmark_ndt2','benchmark_neuroformer','benchmark_ibl_mtm',
        '20260312_prediction_memory_decoder','20260313_local_prediction_memory',
        '20260313_prediction_memory_alignment','20260313_prediction_memory_alignment_tuning',
        '20260320_decoder_scheduled_sampling'
    ]
    short_labels = {
        'baseline_v2': 'baseline_v2',
        'benchmark_ndt2': 'NDT2',
        'benchmark_neuroformer': 'Neuroformer',
        'benchmark_ibl_mtm': 'IBL-MtM',
        '20260312_prediction_memory_decoder': 'PM Decoder\n(20260312)',
        '20260313_local_prediction_memory': 'Local PM\n(20260313)',
        '20260313_prediction_memory_alignment': 'PM Align\n(20260313)',
        '20260313_prediction_memory_alignment_tuning': 'PM Align+\n(20260313)',
        '20260320_decoder_scheduled_sampling': 'Dec SS\n(20260320)',
    }
    metric_rows = []
    for rn in row_labels:
        row = tsv[tsv['name'] == rn]
        if row.empty:
            continue
        row = row.iloc[0]
        def fv(v):
            if pd.isna(v) or str(v).strip() == '-':
                return '—'
            try:
                return f'{float(v):.4f}'
            except:
                return str(v)
        metric_rows.append([
            short_labels.get(rn, rn),
            fv(row.get('fp_bps_250ms', np.nan)),
            fv(row.get('fp_bps_500ms', np.nan)),
            fv(row.get('fp_bps_1000ms', np.nan)),
            fv(row.get('best_test_fp_bps_250ms', np.nan)),
            fv(row.get('best_test_fp_bps_500ms', np.nan)),
            fv(row.get('best_test_fp_bps_1000ms', np.nan)),
        ])

    col_headers = ['Model', 'Rollout\n250ms', 'Rollout\n500ms', 'Rollout\n1000ms',
                   'BestCkpt\nTest 250ms', 'BestCkpt\nTest 500ms', 'BestCkpt\nTest 1000ms']
    n_rows = len(metric_rows)
    n_cols = len(col_headers)

    fig, ax = plt.subplots(figsize=(14, 1.2 + n_rows*0.55))
    ax.axis('off')

    table = ax.table(
        cellText=metric_rows,
        colLabels=col_headers,
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.6)

    # highlight best in each numeric column (columns 1-6)
    for col_i in range(1, n_cols):
        vals = []
        for row_i, row in enumerate(metric_rows):
            try:
                vals.append((float(row[col_i].replace('—','nan')), row_i))
            except:
                vals.append((float('nan'), row_i))
        valid = [(v, r) for v, r in vals if not np.isnan(v)]
        if valid:
            best_val, best_row = max(valid)
            cell = table[best_row+1, col_i]  # +1 for header
            cell.set_facecolor('#d4f0c0')

    # color header
    for col_i in range(n_cols):
        table[0, col_i].set_facecolor('#2C4F7C')
        table[0, col_i].set_text_props(color='white', fontweight='bold')

    # alternating row colors
    for row_i in range(n_rows):
        bg = '#f0f0f0' if row_i % 2 == 0 else '#ffffff'
        for col_i in range(n_cols):
            cell = table[row_i+1, col_i]
            if cell.get_facecolor() == (1,1,1,1) or cell.get_facecolor()[:3] == (1,1,1):
                fc = cell.get_facecolor()
                if list(fc[:3]) != [0.83, 0.94, 0.75]:  # not already highlighted
                    cell.set_facecolor(bg)

    # separator between benchmarks and 1.9 exps (after row index 3)
    separator_row = 4  # after IBL-MtM
    for col_i in range(n_cols):
        table[separator_row, col_i].set_edgecolor('#2C4F7C')
        table[separator_row, col_i].visible_edges = 'B'

    ax.set_title('1.9 Module Optimization — fp-bps Summary Table\n(rollout=best val ckpt rollout; BestCkpt=teacher-forced best ckpt)\nGreen highlight = best value per column; Blue bar = benchmark section',
                 fontsize=9, pad=15)
    plt.tight_layout()
    out = os.path.join(FIG_BASE, 'summary_table.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  saved: {out}')

# ─── main ────────────────────────────────────────────────────────────────────
for exp in EXPS[1:]:   # skip baseline_v2 (no 1.9 fig dir)
    fig_dir = os.path.join(FIG_BASE, exp['name'])
    os.makedirs(fig_dir, exist_ok=True)
    print(f'\n[{exp["name"]}]')

    if exp['log_root'] is None:
        print('  no log root, skipping')
        continue

    print('  → fp_bps_by_window')
    try:
        make_fp_bps_by_window(exp, fig_dir)
    except Exception as e:
        print(f'  ERROR: {e}')

    print('  → per_bin_fp_bps_decay')
    try:
        make_per_bin_decay(exp, fig_dir)
    except Exception as e:
        print(f'  ERROR: {e}')

    print('  → config_timeline')
    try:
        make_config_timeline(exp, fig_dir)
    except Exception as e:
        print(f'  ERROR: {e}')

    # training curves only for decoder_ss (others already have it)
    if exp['name'] == '20260320_decoder_scheduled_sampling':
        print('  → training_curves (decoder_ss winner)')
        try:
            make_training_curves_decoder_ss(exp, fig_dir)
        except Exception as e:
            print(f'  ERROR: {e}')

print('\n[summary_table]')
try:
    make_summary_table()
except Exception as e:
    print(f'  ERROR: {e}')

print('\nDone.')
