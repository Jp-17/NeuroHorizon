#!/usr/bin/env python3
"""
Generate supplementary figures for the 20260321 benchmark task.
Per plan.md 1.8.3 requirements, in addition to training curves, this script produces:
  1. fp-bps trend by prediction window length
  2. per-bin fp-bps decay curves for each prediction window
  3. training config timeline (lr / weight_decay / effective_batch_size / warmup_progress)
  4. summary table PNG of all benchmark runs

All text is English-only to avoid missing-font issues on servers with only DejaVu fonts.
"""
import json
import os
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

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
# Figure 1: fp-bps trend by prediction window
# ═══════════════════════════════════════════════════════════
print('[1/4] fp-bps trend by prediction window ...')

fig, (ax_nf, ax_ibl) = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle(
    'fp-bps vs. Prediction Window  |  Faithful Benchmark, Perich-Miller 10-session',
    fontsize=12, fontweight='bold'
)

# --- Left: Neuroformer rollout vs true_past across pred windows ---
pred_wins = [50, 250]

nf_rollout   = [-6.877741813659668, -8.034954071044922]
nf_sc_roll   = [-7.938905239105225]            # 250ms only
nf_tp        = [-8.373963356018066, -8.570061683654785]
nf_sc_tp     = [-8.658399]                      # 250ms only

ax_nf.axhline(0, color='black', lw=0.8, ls='--', alpha=0.5, label='_zero')
ax_nf.plot(pred_wins, nf_rollout, 'o-',  color='#1565C0', lw=2,   ms=7,  label='NF canonical (rollout)')
ax_nf.plot([250],     nf_sc_roll, 's',   color='#9C27B0', ms=10,         label='NF +session_cond (rollout)', zorder=5)
ax_nf.plot(pred_wins, nf_tp,      'o--', color='#E65100', lw=2,   ms=7,  label='NF canonical (true_past)')
ax_nf.plot([250],     nf_sc_tp,   's',   color='#C62828', ms=10,         label='NF +session_cond (true_past)', zorder=5)

for pw, rv, tv in zip(pred_wins, nf_rollout, nf_tp):
    ax_nf.annotate(f'{rv:.3f}', (pw, rv), xytext=(5, 6),  textcoords='offset points', fontsize=8, color='#1565C0')
    ax_nf.annotate(f'{tv:.3f}', (pw, tv), xytext=(5, -13), textcoords='offset points', fontsize=8, color='#E65100')
ax_nf.annotate(f'{nf_sc_roll[0]:.3f}', (250, nf_sc_roll[0]), xytext=(-55, 8),  textcoords='offset points', fontsize=8, color='#9C27B0')
ax_nf.annotate(f'{nf_sc_tp[0]:.3f}',   (250, nf_sc_tp[0]),   xytext=(-55, -13), textcoords='offset points', fontsize=8, color='#C62828')

ax_nf.set_xticks(pred_wins)
ax_nf.set_xticklabels(['50 ms\n(reference)', '250 ms\n(canonical)'])
ax_nf.set_xlabel('Prediction Window', fontsize=11)
ax_nf.set_ylabel('Test fp-bps', fontsize=11)
ax_nf.set_title('Neuroformer: fp-bps vs. pred window', fontsize=11)
ax_nf.legend(fontsize=8, loc='lower right')
ax_nf.grid(True, alpha=0.3)
ax_nf.set_ylim(-10.5, 1.0)

# --- Right: IBL-MtM epoch progression (pred=250ms fixed) ---
epochs   = [10,  50,   300]
ibl_test = [-0.001689, 0.134455, 0.193806]
ibl_val  = [-0.002602, 0.131080, 0.193752]

ax_ibl.axhline(0, color='black', lw=0.8, ls='--', alpha=0.5)
ax_ibl.plot(epochs, ibl_test, 'o-',  color='#2E7D32', lw=2.5, ms=9,  label='IBL-MtM test fp-bps')
ax_ibl.plot(epochs, ibl_val,  's--', color='#81C784', lw=1.5, ms=7,  label='IBL-MtM best valid fp-bps')
for e, tv in zip(epochs, ibl_test):
    ax_ibl.annotate(f'{tv:.4f}', (e, tv), xytext=(4, 8), textcoords='offset points', fontsize=9,
                    color='#2E7D32', fontweight='bold')

ax_ibl.set_xticks(epochs)
ax_ibl.set_xlabel('Training Epochs  (pred=250 ms)', fontsize=11)
ax_ibl.set_ylabel('fp-bps', fontsize=11)
ax_ibl.set_title('IBL-MtM: fp-bps trend over epochs', fontsize=11)
ax_ibl.legend(fontsize=9)
ax_ibl.grid(True, alpha=0.3)
ax_ibl.set_xlim(-10, 325)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'fp_bps_by_pred_window.png'), dpi=150, bbox_inches='tight')
plt.close()
print('  -> fp_bps_by_pred_window.png saved')

# ═══════════════════════════════════════════════════════════
# Figure 2: per-bin fp-bps decay curves
# ═══════════════════════════════════════════════════════════
print('[2/4] per-bin fp-bps decay curves ...')

fig, (ax250, ax50) = plt.subplots(1, 2, figsize=(15, 5.5))
fig.suptitle(
    'Per-bin fp-bps Decay Curves  |  Faithful Benchmark, Perich-Miller 10-session',
    fontsize=12, fontweight='bold'
)

# ---------- pred=250ms ----------
bins_250 = np.arange(1, 13) * 20   # 20,40,...,240 ms

ibl_e10_pb  = get_per_bin(ibl_e10)
ibl_e50_pb  = get_per_bin(ibl_e50)
ibl_e300_pb = ibl_e300['test_metrics']['per_bin_fp_bps']
nf_can_pb   = get_per_bin(nf_can)
nf_can_tp   = get_per_bin_tp(nf_can)
nf_sc_pb    = nf_sc['test_metrics']['rollout']['per_bin_fp_bps']
nf_sc_tp_pb = nf_sc['test_metrics']['true_past']['per_bin_fp_bps']

ax250.axhline(0, color='black', lw=0.8, ls='--', alpha=0.5)

if ibl_e10_pb:
    ax250.plot(bins_250, ibl_e10_pb,  'o--', color='#A5D6A7', lw=1.5, ms=4, label='IBL-MtM e10')
if ibl_e50_pb:
    ax250.plot(bins_250, ibl_e50_pb,  'o--', color='#66BB6A', lw=1.5, ms=4, label='IBL-MtM e50')
ax250.plot(bins_250,   ibl_e300_pb, 'o-',  color='#1B5E20', lw=2.5, ms=7, label='IBL-MtM e300', zorder=5)

if nf_can_pb:
    ax250.plot(bins_250, nf_can_pb,  's--', color='#90CAF9', lw=1.5, ms=4, label='NF canonical (rollout)')
if nf_can_tp:
    ax250.plot(bins_250, nf_can_tp,  's:',  color='#42A5F5', lw=1.5, ms=4, label='NF canonical (true_past)')

ax250.plot(bins_250, nf_sc_pb,    's-',  color='#CE93D8', lw=2, ms=5, label='NF +session_cond (rollout)')
ax250.plot(bins_250, nf_sc_tp_pb, 's:',  color='#9C27B0', lw=2, ms=5, label='NF +session_cond (true_past)')

ax250.set_xlabel('Time from pred window start (ms)', fontsize=11)
ax250.set_ylabel('per-bin fp-bps', fontsize=11)
ax250.set_title('pred=250 ms  (12 bins x 20 ms)', fontsize=11)
ax250.set_xticks(bins_250)
ax250.legend(fontsize=7.5, loc='lower left')
ax250.grid(True, alpha=0.3)

# ---------- pred=50ms ----------
bins_50    = [25, 50]
nf_ref_pb  = get_per_bin(nf_ref50)
nf_ref_tp  = get_per_bin_tp(nf_ref50)

ax50.axhline(0, color='black', lw=0.8, ls='--', alpha=0.5)
if nf_ref_pb:
    ax50.plot(bins_50, nf_ref_pb, 'o-',  color='#FF7043', lw=2.5, ms=9, label='NF reference 50ms (rollout)')
    for b, v in zip(bins_50, nf_ref_pb):
        ax50.annotate(f'{v:.3f}', (b, v), xytext=(4, 6), textcoords='offset points', fontsize=9, color='#FF7043')
if nf_ref_tp:
    ax50.plot(bins_50, nf_ref_tp, 'o--', color='#BF360C', lw=2,   ms=8, label='NF reference 50ms (true_past)')
    for b, v in zip(bins_50, nf_ref_tp):
        ax50.annotate(f'{v:.3f}', (b, v), xytext=(4, -14), textcoords='offset points', fontsize=9, color='#BF360C')

ax50.set_xlabel('Time from pred window start (ms)', fontsize=11)
ax50.set_ylabel('per-bin fp-bps', fontsize=11)
ax50.set_title('pred=50 ms  (2 bins x 25 ms)\n[Neuroformer reference run only]', fontsize=11)
ax50.set_xticks(bins_50)
ax50.legend(fontsize=9)
ax50.grid(True, alpha=0.3)
ax50.text(0.05, 0.05,
    'Note: IBL-MtM only tested pred=250ms.\nNeuroformer tested 50ms and 250ms.',
    transform=ax50.transAxes, fontsize=8.5, va='bottom',
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'per_bin_fp_bps_decay.png'), dpi=150, bbox_inches='tight')
plt.close()
print('  -> per_bin_fp_bps_decay.png saved')

# ═══════════════════════════════════════════════════════════
# Figure 3: training config timelines
# ═══════════════════════════════════════════════════════════
print('[3/4] training config timelines ...')

def save_config_timeline(history, title, out_name):
    epochs  = [h['epoch'] for h in history]
    lr      = [h['lr'] for h in history]
    wd      = [h.get('weight_decay') for h in history]
    eff_bs  = [h.get('effective_batch_size') for h in history]
    warmup  = [h.get('warmup_progress', 0.0) or 0.0 for h in history]

    fig, axs = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(title, fontsize=11, fontweight='bold')

    axs[0].plot(epochs, lr, color='#1565C0', lw=1.8)
    axs[0].set_ylabel('Learning Rate', fontsize=9)
    axs[0].set_yscale('log')
    axs[0].grid(True, alpha=0.3)
    axs[0].set_title(f'LR: {min(lr):.2e} - {max(lr):.2e}', fontsize=8)

    if any(v is not None for v in wd):
        wd_clean = [v if v is not None else 0 for v in wd]
        axs[1].plot(epochs, wd_clean, color='#6A1B9A', lw=1.8)
        axs[1].set_title(f'Weight Decay: {min(wd_clean):.4f} - {max(wd_clean):.4f}', fontsize=8)
    axs[1].set_ylabel('Weight Decay', fontsize=9)
    axs[1].grid(True, alpha=0.3)

    if any(v is not None for v in eff_bs):
        ebs_clean = [v if v is not None else 0 for v in eff_bs]
        axs[2].plot(epochs, ebs_clean, color='#BF360C', lw=1.8)
        axs[2].set_title(f'Effective Batch Size: {min(ebs_clean)} - {max(ebs_clean)}', fontsize=8)
    axs[2].set_ylabel('Eff. Batch Size', fontsize=9)
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
# Figure 4: summary table PNG
# ═══════════════════════════════════════════════════════════
print('[4/4] benchmark summary table PNG ...')

col_headers = ['Model', 'Variant', 'Pred Win',
               'Best Val\nfp-bps', 'Test fp-bps\n(rollout)',
               'Test fp-bps\n(true_past)', 'Checkpoint\n(best epoch)', 'Notes']

rows = [
    ['IBL-MtM', 'combined e10',     '250ms',  '-0.0026', '-0.0017', 'N/A',     'best_model.pt (auto)',  '20260319 aligned'],
    ['IBL-MtM', 'combined e50',     '250ms',   '0.1311',  '0.1345', 'N/A',     'best_model.pt (auto)',  '20260319 aligned'],
    ['IBL-MtM', 'combined e300',    '250ms',   '0.1938',  '0.1938', 'N/A',     'best_model.pt (ep282)', '20260321; ratio~11x'],
    ['Neuroformer', 'reference',    '50ms',   '-6.870', '-6.878', '-8.374',  'best_model.pt (auto)',  '20260319; 150ms obs'],
    ['Neuroformer', 'canonical',    '250ms',  '-7.992', '-8.035', '-8.570',  'best_model.pt (auto)',  '20260319; 500ms obs'],
    ['Neuroformer', '+session_cond','250ms',  '-7.913', '-7.939', '-8.658',  'best_model.pt (ep39)',  '20260321; SC bug (not injected)'],
]

fig, ax = plt.subplots(figsize=(18, 5))
ax.axis('off')
fig.patch.set_facecolor('white')

col_widths = [0.09, 0.12, 0.08, 0.09, 0.10, 0.11, 0.15, 0.26]

tbl = ax.table(
    cellText=rows,
    colLabels=col_headers,
    cellLoc='center',
    loc='center',
    colWidths=col_widths,
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1, 2.1)

# Header style
for j in range(len(col_headers)):
    tbl[0, j].set_facecolor('#1565C0')
    tbl[0, j].set_text_props(color='white', fontweight='bold', fontsize=9)

# Row colours
c_ibl = ['#E8F5E9', '#C8E6C9', '#A5D6A7']
c_nf  = ['#FFF8E1', '#FFE0B2', '#FFD180']
for i, row in enumerate(rows):
    for j in range(len(col_headers)):
        if row[0] == 'IBL-MtM':
            tbl[i+1, j].set_facecolor(c_ibl[i % 3])
        else:
            tbl[i+1, j].set_facecolor(c_nf[(i-3) % 3])

# Highlight positive / very negative fp-bps
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

print('\nAll figures saved to:')
print(f'  {OUT_DIR}')
for f in sorted(os.listdir(OUT_DIR)):
    print(f'  {f}')
