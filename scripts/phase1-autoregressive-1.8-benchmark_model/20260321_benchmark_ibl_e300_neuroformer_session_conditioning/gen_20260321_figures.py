#!/usr/bin/env python3
"""
生成 20260321 benchmark 任务的补充可视化图表。
按 plan.md 1.8.3 实验记录规范，除 training curves 外补充：
  1. 随预测窗口长度变化的 fp-bps 趋势图
  2. 每个预测窗口的 per-bin fp-bps 衰减曲线
  3. 配置时间轴图（lr / weight_decay / effective_batch_size / warmup_progress）
  4. 表格型 PNG（汇总各 benchmark run 核心结果）
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec

BASE = '/root/autodl-tmp/NeuroHorizon'
OUT_DIR = os.path.join(BASE, 'results/figures/phase1-autoregressive-1.8-benchmark_model/20260321_benchmark_ibl_e300_neuroformer_session_conditioning')
os.makedirs(OUT_DIR, exist_ok=True)

LOG_BASE = os.path.join(BASE, 'results/logs/phase1-autoregressive-1.8-benchmark_model/20260321_benchmark_ibl_e300_neuroformer_session_conditioning')

# ─────────────────────────────────────────────
# 加载所有相关 results.json
# ─────────────────────────────────────────────

def load(path):
    with open(path) as f:
        return json.load(f)

ibl_e10 = load(os.path.join(BASE, 'results/logs/phase1_benchmark_repro_faithful_ibl_mtm_250ms_combined_e10/results.json'))
ibl_e50 = load(os.path.join(BASE, 'results/logs/phase1_benchmark_repro_faithful_ibl_mtm_250ms_combined_e50_aligned/results.json'))
ibl_e300 = load(os.path.join(LOG_BASE, 'ibl_mtm_combined_e300_aligned/results.json'))
nf_canonical = load(os.path.join(BASE, 'results/logs/phase1_benchmark_repro_faithful_neuroformer_250ms_canonical_e50_aligned/formal_eval/eval_results.json'))
nf_ref50 = load(os.path.join(BASE, 'results/logs/phase1_benchmark_repro_faithful_neuroformer_50ms_reference_e50_aligned/formal_eval/eval_results.json'))
nf_sc = load(os.path.join(LOG_BASE, 'neuroformer_250ms_session_conditioning_e50/results.json'))

# history
ibl_e300_hist = ibl_e300['history']
nf_sc_hist = nf_sc['history']

def get_val(d):
    """从不同格式的 results.json 取出 test fp_bps"""
    if 'test_metrics' in d:
        tm = d['test_metrics']
        if 'fp_bps' in tm:
            return tm['fp_bps']
        if 'rollout' in tm:
            return tm['rollout']['fp_bps']
    if 'continuous_metrics' in d:
        return d['continuous_metrics']['test']['rollout']['fp_bps']
    return None

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

def get_val_tp(d):
    if 'test_metrics' in d:
        tm = d['test_metrics']
        if 'true_past' in tm:
            return tm['true_past']['fp_bps']
    if 'continuous_metrics' in d:
        return d['continuous_metrics']['test']['true_past']['fp_bps']
    return None

# ═══════════════════════════════════════════════════════════
# 图 1: 随预测窗口长度变化的 fp-bps 趋势图
# ═══════════════════════════════════════════════════════════
print('[1/4] 生成随预测窗口变化 fp-bps 趋势图...')

# 数据
# x轴: pred_window_ms; 各模型在该窗口下的 test fp-bps
# Neuroformer rollout: 50ms → -6.878, 250ms (canonical) → -8.035, 250ms (+SC) → -7.939
# Neuroformer true_past: 50ms → -8.374, 250ms (canonical) → -8.570, 250ms (+SC) → -8.659
# IBL-MtM: 250ms e10 → -0.0017, e50 → 0.1345, e300 → 0.1938 (only 250ms tested)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('随预测窗口长度变化的 fp-bps 趋势\n(faithful benchmark, Perich-Miller 10-session)', fontsize=13, y=1.01)

# --- Left: Neuroformer rollout vs true_past across pred windows ---
ax = axes[0]
pred_windows = [50, 250]

nf_rollout_fp = [-6.877741813659668, -8.034954071044922]  # ref-50ms, canonical-250ms
nf_sc_rollout_fp = [None, -7.938905239105225]              # only 250ms for +SC
nf_tp_fp = [-8.373963356018066, -8.570061683654785]        # ref-50ms, canonical-250ms
nf_sc_tp_fp = [None, -8.658399]                            # +SC true_past 250ms

ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
ax.plot(pred_windows, nf_rollout_fp, 'o-', color='#2196F3', linewidth=2, markersize=7, label='Neuroformer canonical (rollout)')
ax.plot([250], [-7.938905239105225], 's', color='#9C27B0', markersize=9, label='Neuroformer +session cond (rollout)', zorder=5)
ax.plot(pred_windows, nf_tp_fp, 'o--', color='#FF9800', linewidth=2, markersize=7, label='Neuroformer canonical (true_past)')
ax.plot([250], [-8.6584], 's', color='#E91E63', markersize=9, label='Neuroformer +session cond (true_past)', zorder=5)

ax.set_xlabel('预测窗口长度 (ms)', fontsize=11)
ax.set_ylabel('Test fp-bps', fontsize=11)
ax.set_title('Neuroformer: 不同预测窗口 fp-bps', fontsize=11)
ax.set_xticks(pred_windows)
ax.set_xticklabels(['50ms\n(ref)', '250ms\n(canonical)'])
ax.legend(fontsize=8, loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_ylim(-10.5, 1.0)
for pw, rv, tv in zip(pred_windows, nf_rollout_fp, nf_tp_fp):
    ax.annotate(f'{rv:.3f}', (pw, rv), textcoords='offset points', xytext=(-25, 8), fontsize=7.5, color='#2196F3')
    ax.annotate(f'{tv:.3f}', (pw, tv), textcoords='offset points', xytext=(-25, -14), fontsize=7.5, color='#FF9800')
ax.annotate(f'-7.939', (250, -7.938905239105225), textcoords='offset points', xytext=(5, 8), fontsize=7.5, color='#9C27B0')
ax.annotate(f'-8.658', (250, -8.6584), textcoords='offset points', xytext=(5, -14), fontsize=7.5, color='#E91E63')

# --- Right: IBL-MtM epoch progression at 250ms pred ---
ax2 = axes[1]
epochs = [10, 50, 300]
ibl_fp = [-0.0016890950500965118, 0.13445547223091125, 0.19380563497543335]
ibl_valid_fp = [
    ibl_e10.get('best_valid_metrics', {}).get('fp_bps', ibl_e10.get('best_epoch_metrics', {}).get('fp_bps', None)),
    ibl_e50.get('best_valid_metrics', {}).get('fp_bps', 0.1311),
    0.1937519609928131
]
ax2.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
ax2.plot(epochs, ibl_fp, 'o-', color='#4CAF50', linewidth=2.5, markersize=9, label='IBL-MtM test fp-bps (pred=250ms)')
ax2.plot(epochs, ibl_valid_fp, 's--', color='#8BC34A', linewidth=1.5, markersize=7, label='IBL-MtM best valid fp-bps')
for e, tv, vv in zip(epochs, ibl_fp, ibl_valid_fp):
    ax2.annotate(f'{tv:.4f}', (e, tv), textcoords='offset points', xytext=(4, 6), fontsize=9, color='#4CAF50', fontweight='bold')
ax2.set_xlabel('训练轮数 (epochs)', fontsize=11)
ax2.set_ylabel('fp-bps', fontsize=11)
ax2.set_title('IBL-MtM: epoch 增长趋势 (pred=250ms)', fontsize=11)
ax2.set_xticks(epochs)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(-10, 320)

plt.tight_layout()
out_path = os.path.join(OUT_DIR, 'fp_bps_by_pred_window.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'  -> 已保存: {out_path}')

# ═══════════════════════════════════════════════════════════
# 图 2: 每个预测窗口的 per-bin fp-bps 衰减曲线
# ═══════════════════════════════════════════════════════════
print('[2/4] 生成 per-bin fp-bps 衰减曲线...')

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('每个预测窗口的 per-bin fp-bps 衰减曲线\n(faithful benchmark, Perich-Miller 10-session)', fontsize=13)

# --- subplot 1: pred = 250ms (12 bins × 20ms) ---
ax = axes[0]
ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
bin_size_ms = 20
bins_250 = np.arange(1, 13) * bin_size_ms  # 20,40,...,240 ms

# IBL-MtM 三个 epoch
ibl_e10_pb = get_per_bin(ibl_e10)
ibl_e50_pb = get_per_bin(ibl_e50)
ibl_e300_pb = ibl_e300['test_metrics']['per_bin_fp_bps']
if ibl_e10_pb:
    ax.plot(bins_250, ibl_e10_pb, 'o--', color='#A5D6A7', linewidth=1.5, markersize=5, label='IBL-MtM e10')
if ibl_e50_pb:
    ax.plot(bins_250, ibl_e50_pb, 'o--', color='#66BB6A', linewidth=1.5, markersize=5, label='IBL-MtM e50')
ax.plot(bins_250, ibl_e300_pb, 'o-', color='#2E7D32', linewidth=2.5, markersize=7, label='IBL-MtM e300', zorder=5)

# Neuroformer canonical
nf_can_pb = get_per_bin(nf_canonical)
nf_can_tp_pb = get_per_bin_tp(nf_canonical)
if nf_can_pb:
    ax.plot(bins_250, nf_can_pb, 's--', color='#90CAF9', linewidth=1.5, markersize=5, label='NF canonical (rollout)')
if nf_can_tp_pb:
    ax.plot(bins_250, nf_can_tp_pb, 's:', color='#42A5F5', linewidth=1.5, markersize=5, label='NF canonical (true_past)')

# Neuroformer +SC
nf_sc_pb = nf_sc['test_metrics']['rollout']['per_bin_fp_bps']
nf_sc_tp_pb = nf_sc['test_metrics']['true_past']['per_bin_fp_bps']
ax.plot(bins_250, nf_sc_pb, 's-', color='#CE93D8', linewidth=2, markersize=6, label='NF +session cond (rollout)')
ax.plot(bins_250, nf_sc_tp_pb, 's:', color='#AB47BC', linewidth=2, markersize=6, label='NF +session cond (true_past)')

ax.set_xlabel('预测时间 (ms from pred start)', fontsize=11)
ax.set_ylabel('per-bin fp-bps', fontsize=11)
ax.set_title('pred=250ms (12 bins × 20ms)', fontsize=11)
ax.set_xticks(bins_250)
ax.legend(fontsize=7.5, loc='lower left')
ax.grid(True, alpha=0.3)

# --- subplot 2: pred = 50ms (2 bins × 25ms) ---
ax2 = axes[1]
ax2.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
bins_50 = [25, 50]
nf_ref_pb = get_per_bin(nf_ref50)
nf_ref_tp_pb = get_per_bin_tp(nf_ref50)
if nf_ref_pb:
    ax2.plot(bins_50, nf_ref_pb, 'o-', color='#FF8A65', linewidth=2.5, markersize=9, label='NF reference 50ms (rollout)')
    for b, v in zip(bins_50, nf_ref_pb):
        ax2.annotate(f'{v:.3f}', (b, v), textcoords='offset points', xytext=(4, 6), fontsize=9, color='#FF8A65')
if nf_ref_tp_pb:
    ax2.plot(bins_50, nf_ref_tp_pb, 'o--', color='#FF5722', linewidth=2, markersize=8, label='NF reference 50ms (true_past)')
    for b, v in zip(bins_50, nf_ref_tp_pb):
        ax2.annotate(f'{v:.3f}', (b, v), textcoords='offset points', xytext=(4, -14), fontsize=9, color='#FF5722')
ax2.set_xlabel('预测时间 (ms from pred start)', fontsize=11)
ax2.set_ylabel('per-bin fp-bps', fontsize=11)
ax2.set_title('pred=50ms (2 bins × 25ms)\n[Neuroformer reference run]', fontsize=11)
ax2.set_xticks(bins_50)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
note = 'IBL-MtM 仅测试了 250ms 预测窗口\nNeuroformer 测试了 50ms 和 250ms'
ax2.text(0.05, 0.05, note, transform=ax2.transAxes, fontsize=9, 
         verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

plt.tight_layout()
out_path = os.path.join(OUT_DIR, 'per_bin_fp_bps_decay.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'  -> 已保存: {out_path}')

# ═══════════════════════════════════════════════════════════
# 图 3: 配置时间轴图 (IBL-MtM e300 + Neuroformer +SC)
# ═══════════════════════════════════════════════════════════
print('[3/4] 生成配置时间轴图...')

def make_config_timeline(history, title, out_name, model='ibl_mtm'):
    epochs = [h['epoch'] for h in history]
    lr = [h['lr'] for h in history]
    wd = [h.get('weight_decay', None) for h in history]
    eff_bs = [h.get('effective_batch_size', None) for h in history]
    warmup = [h.get('warmup_progress', None) for h in history]
    
    has_warmup = any(w is not None for w in warmup)
    has_wd = any(w is not None for w in wd)
    has_ebs = any(w is not None for w in eff_bs)
    
    n_rows = 4
    fig, axs = plt.subplots(n_rows, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(title, fontsize=12)
    
    # LR
    axs[0].plot(epochs, lr, color='#1565C0', linewidth=1.8)
    axs[0].set_ylabel('Learning Rate', fontsize=9)
    axs[0].set_yscale('log')
    axs[0].grid(True, alpha=0.3)
    axs[0].set_title(f'Max LR: {max(lr):.2e}  Min LR: {min(lr):.2e}', fontsize=8)
    
    # Weight decay
    if has_wd:
        axs[1].plot(epochs, wd, color='#6A1B9A', linewidth=1.8)
    axs[1].set_ylabel('Weight Decay', fontsize=9)
    axs[1].grid(True, alpha=0.3)
    if has_wd:
        axs[1].set_title(f'WD range: {min(wd):.4f} – {max(wd):.4f}', fontsize=8)
    
    # Effective batch size
    if has_ebs:
        axs[2].plot(epochs, eff_bs, color='#BF360C', linewidth=1.8)
    axs[2].set_ylabel('Effective\nBatch Size', fontsize=9)
    axs[2].grid(True, alpha=0.3)
    if has_ebs:
        axs[2].set_title(f'Eff. BS: {min(eff_bs)} – {max(eff_bs)}', fontsize=8)
    
    # Warmup progress
    if has_warmup:
        wp_clean = [w if w is not None else 0 for w in warmup]
        axs[3].plot(epochs, wp_clean, color='#1B5E20', linewidth=1.8)
        axs[3].axhline(1.0, color='red', linewidth=1, linestyle='--', alpha=0.7, label='warmup complete')
        axs[3].legend(fontsize=8)
    axs[3].set_ylabel('Warmup Progress', fontsize=9)
    axs[3].set_xlabel('Epoch', fontsize=10)
    axs[3].grid(True, alpha=0.3)
    axs[3].set_ylim(-0.05, 1.15)
    
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, out_name)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    return out_path

p1 = make_config_timeline(
    ibl_e300_hist,
    'IBL-MtM combined_e300_aligned — 训练配置时间轴\n(lr / weight_decay / effective_batch_size / warmup_progress)',
    'ibl_mtm_e300_config_timeline.png',
    model='ibl_mtm'
)
print(f'  -> 已保存: {p1}')

p2 = make_config_timeline(
    nf_sc_hist,
    'Neuroformer 250ms +session_conditioning — 训练配置时间轴\n(lr / weight_decay / effective_batch_size / warmup_progress)',
    'neuroformer_sc_config_timeline.png',
    model='neuroformer'
)
print(f'  -> 已保存: {p2}')

# ═══════════════════════════════════════════════════════════
# 图 4: 表格型 PNG（汇总所有 benchmark runs）
# ═══════════════════════════════════════════════════════════
print('[4/4] 生成表格型 PNG 汇总...')

import datetime

rows = [
    # [Model Variant, Task, pred_win, Best Val fp-bps, Test fp-bps (rollout), Test fp-bps (true_past), Best/Test Ckpt, Notes]
    ['IBL-MtM', 'combined e10',     '250ms',  '-0.0026', '-0.0017', 'N/A',     'best_model.pt (e-best)', '20260319 aligned run'],
    ['IBL-MtM', 'combined e50',     '250ms',   '0.1311',  '0.1345', 'N/A',     'best_model.pt (e-best)', '20260319 aligned run'],
    ['IBL-MtM', 'combined e300',    '250ms',   '0.1938',  '0.1938', 'N/A',     'best_model.pt (ep.282)', '20260321, pred_to_true≈11x'],
    ['Neuroformer', 'canonical',    '50ms (ref)', '-6.870', '-6.878', '-8.374', 'best_model.pt', '20260319 reference run, 150ms obs'],
    ['Neuroformer', 'canonical',    '250ms',  '-7.992', '-8.035', '-8.570',  'best_model.pt', '20260319 canonical run'],
    ['Neuroformer', '+session cond','250ms',  '-7.913', '-7.939', '-8.658',  'best_model.pt (ep.39)', '20260321; SC未实际注入 (bug)'],
]

col_headers = ['Model', 'Variant', 'Pred Win', 'Best Val\nfp-bps', 'Test fp-bps\n(rollout)', 'Test fp-bps\n(true_past)', 'Checkpoint', 'Notes']

fig, ax = plt.subplots(figsize=(18, 5))
ax.axis('off')

col_widths = [0.09, 0.12, 0.09, 0.09, 0.10, 0.11, 0.16, 0.24]

table = ax.table(
    cellText=rows,
    colLabels=col_headers,
    cellLoc='center',
    loc='center',
    colWidths=col_widths
)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2.0)

# Style header
for j in range(len(col_headers)):
    table[0, j].set_facecolor('#1565C0')
    table[0, j].set_text_props(color='white', fontweight='bold', fontsize=9)

# Color rows by model
colors_ibl = ['#E8F5E9', '#C8E6C9', '#A5D6A7']
colors_nf  = ['#FFF3E0', '#FFE0B2', '#FFCC80']
for i, row in enumerate(rows):
    model = row[0]
    ci = i
    for j in range(len(col_headers)):
        if model == 'IBL-MtM':
            table[i+1, j].set_facecolor(colors_ibl[ci % 3])
        else:
            table[i+1, j].set_facecolor(colors_nf[(ci-3) % 3])

# Highlight positive fp-bps in green, very negative in red
for i, row in enumerate(rows):
    for col_idx, metric in [(3, row[3]), (4, row[4])]:
        try:
            val = float(metric)
            if val > 0.1:
                table[i+1, col_idx].set_text_props(color='#1B5E20', fontweight='bold')
            elif val < -5:
                table[i+1, col_idx].set_text_props(color='#B71C1C')
        except:
            pass

ax.set_title(
    '1.8 Faithful Benchmark 汇总表\n'
    'Perich-Miller 10-session | 各模型改进在不同预测窗口下的核心结果\n'
    f'生成时间: 2026-03-25',
    fontsize=11, pad=20, fontweight='bold'
)

note = ('注：Neuroformer +session cond 本轮 session conditioning 因 batch["session_idx"] 传递路径 bug 实际未注入模型，\n'
        '结果更接近 canonical rerun（speed test），不能作为 SC 效果的结论依据。')
fig.text(0.01, 0.01, note, fontsize=7.5, color='#555555', style='italic')

plt.tight_layout()
out_path = os.path.join(OUT_DIR, 'benchmark_summary_table.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'  -> 已保存: {out_path}')

# ═══════════════════════════════════════════════════════════
# 复制 training curves (从 logs 到 figures 目录)
# ═══════════════════════════════════════════════════════════
print('[+] 复制 training curves 到 figures 目录...')
import shutil

src_ibl = os.path.join(LOG_BASE, 'ibl_mtm_combined_e300_aligned')
src_nf = os.path.join(LOG_BASE, 'neuroformer_250ms_session_conditioning_e50')
out_ibl = os.path.join(OUT_DIR, 'ibl_mtm_e300_training_curves')
out_nf = os.path.join(OUT_DIR, 'neuroformer_sc_training_curves')
os.makedirs(out_ibl, exist_ok=True)
os.makedirs(out_nf, exist_ok=True)

for src, dst in [(src_ibl, out_ibl), (src_nf, out_nf)]:
    for f in os.listdir(src):
        if f.endswith('.png'):
            shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
            print(f'  -> copied {f}')

print('\n所有图表已生成完毕。')
print(f'输出目录: {OUT_DIR}')
for f in sorted(os.listdir(OUT_DIR)):
    print(f'  {f}')
