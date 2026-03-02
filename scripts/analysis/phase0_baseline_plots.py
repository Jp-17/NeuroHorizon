"""
Phase 0.3 POYO+ Baseline Training Curves & Per-Session Performance

对应 plan.md 任务 0.3.1/0.3.3 可视化补充
数据源: results/logs/phase0_baseline/lightning_logs/version_0/metrics.csv
输出: results/figures/baseline/03_baseline_training_curves.png
"""

import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = Path('/root/autodl-tmp/NeuroHorizon')
METRICS_PATH = PROJECT_ROOT / 'results/logs/phase0_baseline/lightning_logs/version_0/metrics.csv'
OUTPUT_DIR = PROJECT_ROOT / 'results/figures/baseline'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Session info
SESSIONS = [
    'c_20131003', 'c_20131022', 'c_20131101', 'c_20131204',
    'j_20160405', 'j_20160406', 'j_20160407',
    'm_20150610', 'm_20150612', 'm_20150615',
]

SUBJECT_COLORS = {'c': '#2196F3', 'j': '#FF9800', 'm': '#4CAF50'}
SUBJECT_LABELS = {'c': 'Subject C', 'j': 'Subject J', 'm': 'Subject M'}

def get_session_col(session):
    return f'perich_miller_population_2018/{session}_center_out_reaching/cursor_velocity_2d/R2Score()/val'


def parse_metrics():
    """Parse Lightning metrics.csv, aggregating per-epoch."""
    epochs_train = {}  # epoch -> list of train_loss
    epochs_val = {}    # epoch -> {avg_r2, val_loss, per_session_r2}

    with open(METRICS_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ep = row.get('epoch')
            if not ep:
                continue
            ep = int(ep)

            # Train loss rows
            tl = row.get('train_loss')
            if tl:
                epochs_train.setdefault(ep, []).append(float(tl))

            # Val metric rows
            avg_val = row.get('average_val_metric')
            if avg_val:
                vl = row.get('losses/cursor_velocity_2d', '')
                entry = {
                    'avg_r2': float(avg_val),
                    'val_loss': float(vl) if vl else np.nan,
                }
                for s in SESSIONS:
                    col = get_session_col(s)
                    v = row.get(col)
                    if v:
                        entry[s] = float(v)
                epochs_val[ep] = entry

    # Aggregate train loss per epoch (mean)
    train_data = {}
    for ep, losses in sorted(epochs_train.items()):
        train_data[ep] = np.mean(losses)

    return train_data, epochs_val


def plot():
    train_data, val_data = parse_metrics()

    val_epochs = sorted(val_data.keys())
    train_epochs = sorted(train_data.keys())

    # Extract arrays
    val_ep_arr = np.array(val_epochs)
    avg_r2 = np.array([val_data[e]['avg_r2'] for e in val_epochs])
    val_loss = np.array([val_data[e]['val_loss'] for e in val_epochs])
    train_ep_arr = np.array(train_epochs)
    train_loss_arr = np.array([train_data[e] for e in train_epochs])

    best_idx = np.argmax(avg_r2)
    best_epoch = val_ep_arr[best_idx]
    best_r2 = avg_r2[best_idx]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Phase 0.3: POYO+ Baseline Training (500 epochs, Perich-Miller 10 sessions)',
                 fontsize=14, fontweight='bold', y=0.98)

    # --- (0,0) Train/Val Loss vs Epoch ---
    ax = axes[0, 0]
    # Subsample train loss for cleaner plot (every 10 epochs)
    subsample = range(0, len(train_ep_arr), max(1, len(train_ep_arr) // 50))
    ax.plot(train_ep_arr[list(subsample)], train_loss_arr[list(subsample)],
            alpha=0.6, color='#2196F3', label='Train Loss', linewidth=1)
    ax.plot(val_ep_arr, val_loss, color='#F44336', label='Val Loss', linewidth=1.5)
    ax.axvline(x=best_epoch, color='gray', linestyle='--', alpha=0.5, label=f'Best epoch ({best_epoch})')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training & Validation Loss')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- (0,1) Average R2 vs Epoch ---
    ax = axes[0, 1]
    ax.plot(val_ep_arr, avg_r2, color='#4CAF50', linewidth=1.5, label='Average R\u00b2')
    ax.axhline(y=0.3, color='red', linestyle=':', alpha=0.5, label='Acceptance threshold (R\u00b2=0.3)')
    ax.axvline(x=best_epoch, color='gray', linestyle='--', alpha=0.5)
    ax.scatter([best_epoch], [best_r2], color='#4CAF50', s=80, zorder=5, edgecolors='black')
    ax.annotate(f'Best: {best_r2:.4f}\n(epoch {best_epoch})',
                xy=(best_epoch, best_r2), xytext=(best_epoch - 80, best_r2 - 0.05),
                fontsize=9, arrowprops=dict(arrowstyle='->', color='gray'))
    ax.set_xlabel('Epoch')
    ax.set_ylabel('R\u00b2')
    ax.set_title('Average Validation R\u00b2')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- (1,0) Per-Session R2 vs Epoch ---
    ax = axes[1, 0]
    for s in SESSIONS:
        subject = s[0]
        color = SUBJECT_COLORS[subject]
        r2_vals = [val_data[e].get(s, np.nan) for e in val_epochs]
        ax.plot(val_ep_arr, r2_vals, color=color, alpha=0.6, linewidth=1,
                label=s if s == SESSIONS[[s2[0] for s2 in SESSIONS].index(subject)] else None)
    # Add subject-level legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color=SUBJECT_COLORS[k], linewidth=2, label=SUBJECT_LABELS[k])
                       for k in ['c', 'j', 'm']]
    ax.legend(handles=legend_elements, fontsize=9)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('R\u00b2')
    ax.set_title('Per-Session R\u00b2 (colored by subject)')
    ax.grid(True, alpha=0.3)

    # --- (1,1) Per-Session R2 Bar Chart at Best Epoch ---
    ax = axes[1, 1]
    best_vals = val_data[best_epoch]
    session_r2 = [(s, best_vals.get(s, 0)) for s in SESSIONS]
    y_pos = np.arange(len(session_r2))
    colors = [SUBJECT_COLORS[s[0]] for s, _ in session_r2]
    bars = ax.barh(y_pos, [r2 for _, r2 in session_r2], color=colors, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([s for s, _ in session_r2], fontsize=9)
    ax.axvline(x=best_r2, color='gray', linestyle='--', alpha=0.5, label=f'Mean R\u00b2={best_r2:.3f}')
    for i, (s, r2) in enumerate(session_r2):
        ax.text(r2 + 0.005, i, f'{r2:.3f}', va='center', fontsize=8)
    ax.set_xlabel('R\u00b2')
    ax.set_title(f'Per-Session R\u00b2 at Best Epoch ({best_epoch})')
    ax.legend(fontsize=9)
    ax.invert_yaxis()

    plt.tight_layout()
    out_path = OUTPUT_DIR / '03_baseline_training_curves.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[DONE] Saved: {out_path}')
    print(f'  Best epoch: {best_epoch}, Best R2: {best_r2:.4f}')
    print(f'  Val epochs count: {len(val_epochs)}')
    print(f'  Train epochs count: {len(train_epochs)}')


if __name__ == '__main__':
    plot()
