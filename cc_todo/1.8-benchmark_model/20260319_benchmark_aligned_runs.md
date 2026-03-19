# 20260319 Benchmark Aligned Runs

> 状态：进行中
> 总入口：`cc_todo/1.8-benchmark_model/benchmark_index.md`
> 相关审计：`cc_todo/1.8-benchmark_model/20260318_benchmark_faithful_audit_detail_codex.md`
> 分支：`dev/benchmark`

## 任务目的

按上游 repo 训练协议尽量对齐当前 faithful benchmark runner，并补充训练曲线、配置时间轴图和正式记录，作为 `1.8.3` benchmark 持续维护的第一批 aligned long-run。

## 当前任务与动机

### 1. IBL-MtM combined_e50_aligned

- 动机：验证 faithful IBL-MtM 在保留 upstream `combined multi-mask` 语义下，随着 epoch 增加能否从 near-zero 继续抬升到稳定正值
- 相比现有方案的改动点：
  - `e10 -> e50`
  - 补充 `history`、`lr`、`weight_decay`、`effective_batch_size`、`warmup_progress`
  - 补充训练曲线和配置时间轴图
- 涉及模块：
  - `neural-benchmark/faithful_ibl_mtm.py`
  - `neural-benchmark/plot_benchmark_history.py`
  - `neural-benchmark/run_faithful_1p8_aligned.sh`

**实验配置**：
- 数据集：Perich-Miller 2018
- sessions：10
- 采样方式：canonical continuous benchmark protocol
- obs/pred：`500ms / 250ms`
- 关键超参数：`epoch=50, batch_size=16, lr=1e-4, weight_decay=0.01`

**当前结果**：
- 训练 loss：最终 `train_loss = 0.3151`
- 最佳 val `fp-bps`：`0.1311`
- test `fp-bps`：`0.1345`
- test trial `fp-bps`：`0.1116`
- test `per_neuron_psth_r2`：`0.5679`
- checkpoint：`results/logs/phase1_benchmark_repro_faithful_ibl_mtm_250ms_combined_e50_aligned/best_model.pt`

**可视化**：
- `results/logs/phase1_benchmark_repro_faithful_ibl_mtm_250ms_combined_e50_aligned/train_loss_curve.png`
- `results/logs/phase1_benchmark_repro_faithful_ibl_mtm_250ms_combined_e50_aligned/valid_fp_bps_curve.png`
- `results/logs/phase1_benchmark_repro_faithful_ibl_mtm_250ms_combined_e50_aligned/valid_r2_curve.png`
- `results/logs/phase1_benchmark_repro_faithful_ibl_mtm_250ms_combined_e50_aligned/lr_curve.png`
- `results/logs/phase1_benchmark_repro_faithful_ibl_mtm_250ms_combined_e50_aligned/training_config_timeline.png`
- `results/logs/phase1_benchmark_repro_faithful_ibl_mtm_250ms_combined_e50_aligned/train_mask_counts_curve.png`
- e10/e50 对比：`results/logs/phase1_benchmark_repro_faithful_ibl_mtm_250ms_compare_e10_e50_aligned/comparison.md`

**与 baseline 对比**：
- 相比 `combined_e10` test `fp-bps = -0.0017`，当前 `combined_e50` 为 `0.1345`，提升 `+0.1361`
- 当前更支持“继续保留 `combined` 路线，而不是回到 exact forward_pred control”

## 2. Neuroformer canonical 500/250 e50 aligned

- 动机：在更接近上游训练超参的条件下，重新判断 faithful Neuroformer 的学习曲线和 formal eval 潜力
- 相比现有方案的改动点：
  - `e3/eval-only baseline -> e50 aligned`
  - `weight_decay=1.0`
  - `warmup_tokens=8e7`
  - `microbatch=8 + grad_accum=20`
  - 补充 `tokens_seen / warmup_progress / effective_batch_size`
- 涉及模块：
  - `neural-benchmark/faithful_neuroformer.py`
  - `neural-benchmark/plot_benchmark_history.py`
  - `neural-benchmark/run_faithful_1p8_aligned.sh`

**实验配置**：
- 数据集：Perich-Miller 2018
- sessions：10
- 采样方式：canonical continuous benchmark protocol
- obs/pred：`500ms / 250ms`
- 关键超参数：`epoch=50, batch_size=8, grad_accum=20, lr=1e-4, weight_decay=1.0`

**当前进度**（2026-03-19 23:48 CST）：
- 当前仍在训练中
- 运行进程：`faithful_neuroformer.py --mode train --obs-window 0.5 --pred-window 0.25 --epochs 50`
- 当前 checkpoint：`results/logs/phase1_benchmark_repro_faithful_neuroformer_250ms_canonical_e50_aligned/last_model.pt`
- 最近一次 checkpoint 更新时间：`2026-03-19 23:44:17 +0800`
- formal eval 尚未开始

**待回填结果**：
- 训练 loss
- 最佳 val `fp-bps`
- test `fp-bps`
- `rollout / true_past`
- token stats
- training curves 与 `training_config_timeline.png`

## 3. Neuroformer 150/50 reference e50 aligned

- 动机：保留更接近官方 repo 常见窗口的 reference sanity experiment，判断更短 horizon 下 `rollout / true_past` 是否明显改善
- 实验配置：
  - 数据集：Perich-Miller 2018
  - sessions：10
  - 采样方式：canonical continuous benchmark protocol
  - obs/pred：`150ms / 50ms`
  - 关键超参数：`epoch=50, batch_size=8, grad_accum=20, lr=1e-4, weight_decay=1.0`
- 当前状态：尚未开始；等待 canonical `500/250` run 完成后执行
- 正式 eval：必须同时记录 `rollout / true_past`

## 当前结论

1. `IBL-MtM combined_e50_aligned` 已经从 near-zero 提升为正值，是当前 1.8 benchmark 里最值得继续推进的一条 faithful benchmark 路线。
2. `Neuroformer canonical e50 aligned` 仍在训练中，因此当前还不能对 aligned 长跑是否有效下结论。
3. `Neuroformer 150/50` 仍作为 reference sanity experiment 保留，但不替代 canonical benchmark。

## 后续安排

1. 等待 `Neuroformer canonical 500/250 e50 aligned` 训练完成，补 formal eval（`rollout / true_past`）和训练曲线。
2. canonical 完成后继续执行 `Neuroformer 150/50 reference e50 aligned`。
3. 全部完成后，将 aligned 长跑结果继续回填到：
   - `cc_todo/1.8-benchmark_model/benchmark_index.md`
   - `cc_todo/1.8-benchmark_model/20260312_benchmark_main_task_log.md`
   - `cc_todo/1.8-benchmark_model/20260318_benchmark_faithful_audit_detail_codex.md`
