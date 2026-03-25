#!/usr/bin/env python3
"""
Generate supplementary figures for the 20260321 benchmark task.
Per plan.md 1.8.3 requirements, in addition to training curves, this script produces:
  1. fp-bps trend by prediction window length (3 panels)
  2. per-bin fp-bps decay curves  (3 panels: IBL-MtM 250ms | NF 250ms | NF 50ms)
  3. training config timeline (lr / weight_decay / effective_batch_size / warmup_progress)
  4. summary table PNG of all benchmark runs

All text is English-only (server only has DejaVu fonts, no CJK support).
"""
import json
import os
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False

BASE = '/root/autodl-tmp/NeuroHorizon'
OUT_DIR = os.path.join(
    BASE,
    'results/figures/phase1-autoregressive-1.8-benchmark_model'
    '/20260321_benchmark_ibl_e300_neuroformer_session_conditioning'
)
os.makedirs(OUT_DIR, exist_ok=True)

LOG_BASE = os.path.join(
    BASE,
    'results/logs/phase1-autoregressive-1.8-benchmark_model'
    '/20260321_benchmark_ibl_e300_neuroformer_session_conditioning'
)

# ── helpers ──────────────────────────────────────────────────────────────────

def load(path):
    with open(path) as f:
        return json.load(f)

def get_per_bin(d, split='test', mode='rollout'):
    if 'test_metrics' in d:
        tm = d['test_metrics']
        if 'per_bin_fp_bps' in tm:
            return tm['per_bin_fp_bps']
        if mode in tm and 'per_bin_fp_bps' in tm[mode]:
            return tm[mode]['per_bin_fp_bps']
    if 'continuous_metrics' in d:
        return d['continuous_metrics'][split][mode].get('per_bin_fp_bps', [])
    return []

def get_per_bin_tp(d, split='test'):
    if 'test_metrics' in d:
        tm = d['test_metrics']
        if 'true_past' in tm and 'per_bin_fp_bps' in tm['true_past']:
            return tm['true_past']['per_bin_fp_bps']
    if 'continuous_metrics' in d:
        return d['continuous_metrics'][split]['true_past'].get('per_bin_fp_bps', [])
    return []

# ── load results ─────────────────────────────────────────────────────────────

ibl_e10  = load(f'{BASE}/results/logs/phase1_benchmark_repro_faithful_ibl_mtm_250ms_combined_e10/results.json')
ibl_e50  = load(f'{BASE}/results/logs/phase1_benchmark_repro_faithful_ibl_mtm_250ms_combined_e50_aligned/results.json')
ibl_e300 = load(f'{LOG_BASE}/ibl_mtm_combined_e300_aligned/results.json')

nf_can   = load(f'{BASE}/results/logs/phase1_benchmark_repro_faithful_neuroformer_250ms_canonical_e50_aligned/formal_eval/eval_results.json')
nf_ref50 = load(f'{BASE}/results/logs/phase1_benchmark_repro_faithful_neuroformer_50ms_reference_e50_aligned/formal_eval/eval_results.json')
nf_sc    = load(f'{LOG_BASE}/neuroformer_250ms_session_conditioning_e50/results.json')

ibl_e300_hist = ibl_e300['history']
nf_sc_hist    = nf_sc['history']

# ═══════════════════════════════════════════════════════════
# Figure 1: fp-bps trend by prediction window  (3 panels)
#   Left  : Neuroformer fp-bps vs pred window
#   Middle: IBL-MtM fp-bps vs training epochs (pred=250ms fixed)
#   Right : IBL-MtM e300 valid fp-bps over training (epoch curve)
# ═══════════════════════════════════════════════════════════
print('[1/4] fp-bps trend by prediction window (3 panels) ...')

fig, (ax_nf, ax_ibl_epoch, ax_ibl_curve) = plt.subplots(1, 3, figsize=(19, 5.5))
fig.suptitle(
    'fp-bps Summary  |  Faithful Benchmark, Perich-Miller 10-session',
    fontsize=12, fontweight='bold'
)

# ---- Panel 1: Neuroformer fp-bps vs pred window ----
pred_wins    = [50, 250]
nf_rollout   = [-6.877741813659668, -8.034954071044922]
nf_sc_roll   = [-7.938905239105225]
nf_tp        = [-8.373963356018066, -8.570061683654785]
nf_sc_tp     = [-8.658399]

ax_nf.axhline(0, color='black', lw=0.8, ls='--', alpha=0.5)
ax_nf.plot(pred_wins, nf_rollout, 'o-',  color='#1565C0', lw=2,   ms=7,  label='NF canonical (rollout)')
ax_nf.plot([250],     nf_sc_roll, 's',   color='#9C27B0', ms=10,         label='NF +session_cond (rollout)', zorder=5)
ax_nf.plot(pred_wins, nf_tp,      'o--', color='#E65100', lw=2,   ms=7,  label='NF canonical (true_past)')
ax_nf.plot([250],     nf_sc_tp,   's',   color='#C62828', ms=10,         label='NF +session_cond (true_past)', zorder=5)

for pw, rv, tv in zip(pred_wins, nf_rollout, nf_tp):
    ax_nf.annotate(f'{rv:.3f}', (pw, rv), xytext=(5, 6),   textcoords='offset points', fontsize=8, color='#1565C0')
    ax_nf.annotate(f'{tv:.3f}', (pw, tv), xytext=(5, -13), textcoords='offset points', fontsize=8, color='#E65100')
ax_nf.annotate(f'{nf_sc_roll[0]:.3f}', (250, nf_sc_roll[0]), xytext=(-58, 8),   textcoords='offset points', fontsize=8, color='#9C27B0')
ax_nf.annotate(f'{nf_sc_tp[0]:.3f}',   (250, nf_sc_tp[0]),   xytext=(-58, -13), textcoords='offset points', fontsize=8, color='#C62828')

ax_nf.set_xticks(pred_wins)
ax_nf.set_xticklabels(['50 ms\n(reference)', '250 ms\n(canonical)'])
ax_nf.set_xlabel('Prediction Window', fontsize=11)
ax_nf.set_ylabel('Test fp-bps', fontsize=11)
ax_nf.set_title('Neuroformer: fp-bps vs. pred window', fontsize=11)
ax_nf.legend(fontsize=8, loc='lower right')
ax_nf.grid(True, alpha=0.3)
ax_nf.set_ylim(-10.5, 1.0)

# ---- Panel 2: IBL-MtM test fp-bps at e10 / e50 / e300 ----
epochs_pts = [10, 50, 300]
ibl_test   = [-0.001689, 0.134455, 0.193806]
ibl_val    = [-0.002602, 0.131080, 0.193752]

ax_ibl_epoch.axhline(0, color='black', lw=0.8, ls='--', alpha=0.5)
ax_ibl_epoch.plot(epochs_pts, ibl_test, 'o-',  color='#2E7D32', lw=2.5, ms=9,  label='IBL-MtM test fp-bps')
ax_ibl_epoch.plot(epochs_pts, ibl_val,  's--', color='#81C784', lw=1.5, ms=7,  label='IBL-MtM best valid fp-bps')
for e, tv in zip(epochs_pts, ibl_test):
    ax_ibl_epoch.annotate(f'{tv:.4f}', (e, tv), xytext=(4, 8),
                          textcoords='offset points', fontsize=9,
                          color='#2E7D32', fontweight='bold')

ax_ibl_epoch.set_xticks(epochs_pts)
ax_ibl_epoch.set_xlabel('Training Epochs  (pred=250 ms)', fontsize=11)
ax_ibl_epoch.set_ylabel('fp-bps', fontsize=11)
ax_ibl_epoch.set_title('IBL-MtM: best fp-bps at e10 / e50 / e300', fontsize=11)
ax_ibl_epoch.legend(fontsize=9)
ax_ibl_epoch.grid(True, alpha=0.3)
ax_ibl_epoch.set_xlim(-10, 325)

# ---- Panel 3: IBL-MtM e300 valid fp-bps curve over all epochs ----
train_epochs = [h['epoch']        for h in ibl_e300_hist]
valid_fp_bps = [h['valid_fp_bps'] for h in ibl_e300_hist]
best_epoch   = ibl_e300['best_epoch']
best_val_fp  = ibl_e300['best_valid_metrics']['fp_bps']

ax_ibl_curve.axhline(0, color='black', lw=0.8, ls='--', alpha=0.5, label='fp-bps = 0')
ax_ibl_curve.plot(train_epochs, valid_fp_bps, color='#388E3C', lw=1.6, label='valid fp-bps (per epoch)')
ax_ibl_curve.axvline(best_epoch, color='red', lw=1.2, ls='--', alpha=0.8, label=f'best epoch ({best_epoch})')
ax_ibl_curve.scatter([best_epoch], [best_val_fp], color='red', zorder=5, s=60)
ax_ibl_curve.annotate(
    f'best ep={best_epoch}\n{best_val_fp:.4f}',
    (best_epoch, best_val_fp),
    xytext=(-70, -25), textcoords='offset points',
    fontsize=8, color='red',
    arrowprops=dict(arrowstyle='->', color='red', lw=0.8)
)

ax_ibl_curve.set_xlabel('Epoch', fontsize=11)
ax_ibl_curve.set_ylabel('Valid fp-bps', fontsize=11)
ax_ibl_curve.set_title('IBL-MtM e300: valid fp-bps over training', fontsize=11)
ax_ibl_curve.legend(fontsize=8)
ax_ibl_curve.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'fp_bps_by_pred_window.png'), dpi=150, bbox_inches='tight')
plt.close()
print('  -> fp_bps_by_pred_window.png saved')

# ═══════════════════════════════════════════════════════════
# Figure 2: per-bin fp-bps decay curves  (3 panels)
#   Left  : IBL-MtM  pred=250ms (e10 / e50 / e300)
#   Middle: Neuroformer pred=250ms (canonical + +SC, rollout & true_past)
#   Right : Neuroformer pred=50ms  (reference, rollout & true_past)
# ═══════════════════════════════════════════════════════════
print('[2/4] per-bin fp-bps decay curves (3 panels) ...')

fig, (ax_ibl, ax_nf250, ax_nf50) = plt.subplots(1, 3, figsize=(19, 5.5))
fig.suptitle(
    'Per-bin fp-bps Decay Curves  |  Faithful Benchmark, Perich-Miller 10-session',
    fontsize=12, fontweight='bold'
)

bins_250 = np.arange(1, 13) * 20   # 20, 40, ..., 240 ms

ibl_e10_pb  = get_per_bin(ibl_e10)
ibl_e50_pb  = get_per_bin(ibl_e50)
ibl_e300_pb = ibl_e300['test_metrics']['per_bin_fp_bps']
nf_can_pb   = get_per_bin(nf_can)
nf_can_tp   = get_per_bin_tp(nf_can)
nf_sc_pb    = nf_sc['test_metrics']['rollout']['per_bin_fp_bps']
nf_sc_tp_pb = nf_sc['test_metrics']['true_past']['per_bin_fp_bps']

# ---- Panel 1: IBL-MtM, pred=250ms ----
ax_ibl.axhline(0, color='black', lw=0.8, ls='--', alpha=0.5)
if ibl_e10_pb:
    ax_ibl.plot(bins_250, ibl_e10_pb, 'o--', color='#A5D6A7', lw=1.5, ms=5, label='IBL-MtM e10')
if ibl_e50_pb:
    ax_ibl.plot(bins_250, ibl_e50_pb, 'o--', color='#4CAF50', lw=1.5, ms=5, label='IBL-MtM e50')
ax_ibl.plot(bins_250, ibl_e300_pb, 'o-', color='#1B5E20', lw=2.5, ms=7, label='IBL-MtM e300', zorder=5)

ax_ibl.set_xlabel('Time from pred window start (ms)', fontsize=10)
ax_ibl.set_ylabel('per-bin fp-bps', fontsize=10)
ax_ibl.set_title('IBL-MtM  |  pred=250 ms  (12 bins x 20 ms)', fontsize=10)
ax_ibl.set_xticks(bins_250)
ax_ibl.legend(fontsize=9)
ax_ibl.grid(True, alpha=0.3)

# ---- Panel 2: Neuroformer, pred=250ms ----
ax_nf250.axhline(0, color='black', lw=0.8, ls='--', alpha=0.5)
if nf_can_pb:
    ax_nf250.plot(bins_250, nf_can_pb,  's--', color='#1565C0', lw=1.8, ms=5, label='NF canonical (rollout)')
if nf_can_tp:
    ax_nf250.plot(bins_250, nf_can_tp,  's:',  color='#42A5F5', lw=1.8, ms=5, label='NF canonical (true_past)')
ax_nf250.plot(bins_250, nf_sc_pb,    's-',  color='#7B1FA2', lw=2,   ms=6, label='NF +session_cond (rollout)')
ax_nf250.plot(bins_250, nf_sc_tp_pb, 's:',  color='#CE93D8', lw=2,   ms=6, label='NF +session_cond (true_past)')

ax_nf250.set_xlabel('Time from pred window start (ms)', fontsize=10)
ax_nf250.set_ylabel('per-bin fp-bps', fontsize=10)
ax_nf250.set_title('Neuroformer  |  pred=250 ms  (12 bins x 20 ms)', fontsize=10)
ax_nf250.set_xticks(bins_250)
ax_nf250.legend(fontsize=8.5)
ax_nf250.grid(True, alpha=0.3)

# ---- Panel 3: Neuroformer reference, pred=50ms ----
bins_50   = [25, 50]
nf_ref_pb = get_per_bin(nf_ref50)
nf_ref_tp = get_per_bin_tp(nf_ref50)

ax_nf50.axhline(0, color='black', lw=0.8, ls='--', alpha=0.5)
if nf_ref_pb:
    ax_nf50.plot(bins_50, nf_ref_pb, 'o-',  color='#FF7043', lw=2.5, ms=9, label='NF reference (rollout)')
    for b, v in zip(bins_50, nf_ref_pb):
        ax_nf50.annotate(f'{v:.3f}', (b, v), xytext=(4, 6), textcoords='offset points', fontsize=9, color='#FF7043')
if nf_ref_tp:
    ax_nf50.plot(bins_50, nf_ref_tp, 'o--', color='#BF360C', lw=2,   ms=8, label='NF reference (true_past)')
    for b, v in zip(bins_50, nf_ref_tp):
        ax_nf50.annotate(f'{v:.3f}', (b, v), xytext=(4, -14), textcoords='offset points', fontsize=9, color='#BF360C')

ax_nf50.set_xlabel('Time from pred window start (ms)', fontsize=10)
ax_nf50.set_ylabel('per-bin fp-bps', fontsize=10)
ax_nf50.set_title('Neuroformer reference  |  pred=50 ms\n(2 bins x 25 ms, 150ms obs)', fontsize=10)
ax_nf50.set_xticks(bins_50)
ax_nf50.legend(fontsize=9)
ax_nf50.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'per_bin_fp_bps_decay.png'), dpi=150, bbox_inches='tight')
plt.close()
print('  -> per_bin_fp_bps_decay.png saved')

# ═══════════════════════════════════════════════════════════
# Figure 3: training config timelines (unchanged)
# ═══════════════════════════════════════════════════════════
print('[3/4] training config timelines ...')

def save_config_timeline(history, title, out_name):
    epochs = [h['epoch'] for h in history]
    lr     = [h['lr'] for h in history]
    wd     = [h.get('weight_decay') for h in history]
    eff_bs = [h.get('effective_batch_size') for h in history]
    warmup = [h.get('warmup_progress', 0.0) or 0.0 for h in history]

    fig, axs = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(title, fontsize=11, fontweight='bold')

    axs[0].plot(epochs, lr, color='#1565C0', lw=1.8)
    axs[0].set_ylabel('Learning Rate', fontsize=9)
    axs[0].set_yscale('log')
    axs[0].grid(True, alpha=0.3)
    axs[0].set_title(f'LR: {min(lr):.2e} - {max(lr):.2e}', fontsize=8)

    wd_clean = [v if v is not None else 0 for v in wd]
    axs[1].plot(epochs, wd_clean, color='#6A1B9A', lw=1.8)
    axs[1].set_ylabel('Weight Decay', fontsize=9)
    axs[1].set_title(f'Weight Decay: {min(wd_clean):.4f} - {max(wd_clean):.4f}', fontsize=8)
    axs[1].grid(True, alpha=0.3)

    ebs_clean = [v if v is not None else 0 for v in eff_bs]
    axs[2].plot(epochs, ebs_clean, color='#BF360C', lw=1.8)
    axs[2].set_ylabel('Eff. Batch Size', fontsize=9)
    axs[2].set_title(f'Effective Batch Size: {min(ebs_clean)} - {max(ebs_clean)}', fontsize=8)
    axs[2].grid(True, alpha=0.3)

    axs[3].plot(epochs, warmup, color='#1B5E20', lw=1.8, label='warmup progress')
    axs[3].axhline(1.0, color='red', lw=1, ls='--', alpha=0.7, label='warmup complete (1.0)')
    axs[3].set_ylabel('Warmup Progress', fontsize=9)
    axs[3].set_xlabel('Epoch', fontsize=10)
    axs[3].set_ylim(-0.05, 1.15)
    axs[3].legend(fontsize=8)
    axs[3].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, out_name)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    return out_path

p1 = save_config_timeline(
    ibl_e300_hist,
    'IBL-MtM combined_e300_aligned — Training Config Timeline\n'
    '(lr / weight_decay / effective_batch_size / warmup_progress)',
    'ibl_mtm_e300_config_timeline.png'
)
print(f'  -> {p1}')

p2 = save_config_timeline(
    nf_sc_hist,
    'Neuroformer 250ms +session_conditioning — Training Config Timeline\n'
    '(lr / weight_decay / effective_batch_size / warmup_progress)',
    'neuroformer_sc_config_timeline.png'
)
print(f'  -> {p2}')

# ═══════════════════════════════════════════════════════════
# Figure 4: summary table PNG (unchanged)
# ═══════════════════════════════════════════════════════════
print('[4/4] benchmark summary table PNG ...')

col_headers = ['Model', 'Variant', 'Pred Win',
               'Best Val\nfp-bps', 'Test fp-bps\n(rollout)',
               'Test fp-bps\n(true_past)', 'Checkpoint\n(best epoch)', 'Notes']

rows = [
    ['IBL-MtM', 'combined e10',     '250ms',  '-0.0026', '-0.0017', 'N/A',     'best_model.pt (auto)',  '20260319 aligned'],
    ['IBL-MtM', 'combined e50',     '250ms',   '0.1311',  '0.1345', 'N/A',     'best_model.pt (auto)',  '20260319 aligned'],
    ['IBL-MtM', 'combined e300',    '250ms',   '0.1938',  '0.1938', 'N/A',     'best_model.pt (ep282)', '20260321; pred_to_true~11x'],
    ['Neuroformer', 'reference',    '50ms',   '-6.870', '-6.878', '-8.374',  'best_model.pt (auto)',  '20260319; 150ms obs'],
    ['Neuroformer', 'canonical',    '250ms',  '-7.992', '-8.035', '-8.570',  'best_model.pt (auto)',  '20260319; 500ms obs'],
    ['Neuroformer', '+session_cond','250ms',  '-7.913', '-7.939', '-8.658',  'best_model.pt (ep39)',  '20260321; SC bug (not injected)'],
]

fig, ax = plt.subplots(figsize=(18, 5))
ax.axis('off')
fig.patch.set_facecolor('white')

col_widths = [0.09, 0.12, 0.08, 0.09, 0.10, 0.11, 0.15, 0.26]

tbl = ax.table(
    cellText=rows, colLabels=col_headers,
    cellLoc='center', loc='center', colWidths=col_widths,
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1, 2.1)

for j in range(len(col_headers)):
    tbl[0, j].set_facecolor('#1565C0')
    tbl[0, j].set_text_props(color='white', fontweight='bold', fontsize=9)

c_ibl = ['#E8F5E9', '#C8E6C9', '#A5D6A7']
c_nf  = ['#FFF8E1', '#FFE0B2', '#FFD180']
for i, row in enumerate(rows):
    for j in range(len(col_headers)):
        tbl[i+1, j].set_facecolor(c_ibl[i % 3] if row[0] == 'IBL-MtM' else c_nf[(i-3) % 3])

for i, row in enumerate(rows):
    for col_idx in [3, 4, 5]:
        try:
            val = float(row[col_idx])
            if val > 0.1:
                tbl[i+1, col_idx].set_text_props(color='#1B5E20', fontweight='bold')
            elif val < -5.0:
                tbl[i+1, col_idx].set_text_props(color='#B71C1C')
        except (ValueError, TypeError):
            pass

ax.set_title(
    '1.8 Faithful Benchmark — Summary Table\n'
    'Perich-Miller 10-session | Best val fp-bps / test fp-bps across benchmark runs  |  Generated 2026-03-25',
    fontsize=11, pad=18, fontweight='bold'
)
fig.text(
    0.01, 0.01,
    'Note: Neuroformer +session_cond — session conditioning was NOT injected due to session_idx routing bug; '
    'result treated as canonical rerun for speed validation only.',
    fontsize=7.5, color='#555555', style='italic'
)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'benchmark_summary_table.png'), dpi=150, bbox_inches='tight')
plt.close()
print('  -> benchmark_summary_table.png saved')

# ── copy training curves from logs -> figures ─────────────────────────────────
print('[+] Copying training curves from logs to figures dir ...')
src_ibl = os.path.join(LOG_BASE, 'ibl_mtm_combined_e300_aligned')
src_nf  = os.path.join(LOG_BASE, 'neuroformer_250ms_session_conditioning_e50')
dst_ibl = os.path.join(OUT_DIR,  'ibl_mtm_e300_training_curves')
dst_nf  = os.path.join(OUT_DIR,  'neuroformer_sc_training_curves')
os.makedirs(dst_ibl, exist_ok=True)
os.makedirs(dst_nf,  exist_ok=True)

for src, dst in [(src_ibl, dst_ibl), (src_nf, dst_nf)]:
    for fname in os.listdir(src):
        if fname.endswith('.png'):
            shutil.copy2(os.path.join(src, fname), os.path.join(dst, fname))
            print(f'     {fname}')

print('\nAll figures saved to:', OUT_DIR)
for f in sorted(os.listdir(OUT_DIR)):
    print(f'  {f}')
