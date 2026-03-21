# 20260321 Benchmark IBL e300 + Neuroformer Session Conditioning

> 状态：实施中
> 总入口：`cc_todo/1.8-benchmark_model/benchmark_index.md`
> 分支：`dev/benchmark`

## 想法描述

在 `1.8` faithful benchmark 的下一阶段，继续沿已经出现明确正向趋势的 `IBL-MtM combined` 路线做 `e300` 长跑；同时仅保留 `Neuroformer canonical 500/250`，引入显式 session conditioning，并把训练期监控从“只看 rollout”扩展为“rollout + true_past + teacher-forced loss”的并行诊断。

## 动机与目的

- `IBL-MtM combined_e50_aligned` 已从 near-zero 提升到明确正值，需要确认是否仍有长训练收益，还是已经平台化。
- `Neuroformer` 当前更像 `from-scratch + token/count mismatch + session conditioning不足` 的问题，而不是单纯 exposure bias。
- 当前 `Neuroformer` 训练期缺少足够的诊断信号，无法有效判断失败原因，因此需要补全训练监控与 formal 诊断指标。

## 相比现有方案的改动点

- `IBL-MtM`
  - `e50 -> e300`
  - 新增训练期与 formal eval 诊断指标：`valid_poisson_nll / predicted_to_true_event_ratio / per_session_metrics`
- `Neuroformer`
  - 只保留 canonical `500/250`
  - 引入显式 `session conditioning`
  - 训练期同步监控 `valid rollout fp-bps / valid true_past fp-bps / valid teacher-forced loss / rollout-true_past gap`
  - 不再把 `R²` 作为训练期主监控项

## 模型改进的实现方案

### IBL-MtM

- 保持 upstream `combined` 训练语义
- 保持 current aligned 超参
- 仅增加 epoch 到 `300`
- 继续采用 `best ckpt -> formal valid/test continuous`

### Neuroformer

- 保持 upstream `Tokenizer + ID/dt cross-entropy` 主路径
- 保持 canonical `500ms obs + 250ms pred`
- 保留现有 `session-constrained decoding`
- 在训练和 eval 时新增 learnable `recording/session embedding`
  - 加性注入到 `id_prev` 和 `id` token embeddings
  - 不改变 ID/dt token vocab，不插入 session token
- `best_model.pt` 继续按 `valid rollout fp-bps` 选择
- formal eval 继续报告 `rollout / true_past`

### Neuroformer 当前 session-constrained decoding 的含义

- 它不是显式 session conditioning，而是解码阶段的最小安全约束。
- 当前 faithful bridge 使用全局 neuron ID vocab；如果不加限制，模型可能为某个 recording 生成出该 recording 根本不存在的 neuron ID。
- 现在的实现方式是：
  - 数据侧保留当前样本的 `recording_id / unit_ids / unit_mask`
  - 解码时先构造该样本合法的 `valid_id_tokens`
  - 对不属于当前 recording 的 ID logits 统一置为 `-inf`
- 这可以防止跨 session 的无效 neuron 预测，但它只在输出空间做限制，并没有在训练时向模型显式提供 recording-level 条件。
- 因此本轮新增的 `session embedding` 与现有 `session-constrained decoding` 是互补关系：
  - 前者解决“训练期缺少 session 条件”
  - 后者解决“解码时不能越界生成无效 neuron ID”

## 涉及改动模块

- `neural-benchmark/faithful_ibl_mtm.py`
- `neural-benchmark/faithful_neuroformer.py`
- `neural-benchmark/plot_benchmark_history.py`
- `scripts/phase1-autoregressive-1.8-benchmark_model/20260321_benchmark_ibl_e300_neuroformer_session_conditioning/run_benchmark.sh`

## 想法摘要

- `IBL-MtM` 现在值得继续堆 epoch，但要同时记录足够的诊断指标来决定后续是继续长跑还是转 metadata/pretraining。
- `Neuroformer` 先不扩窗口，优先补 recording-level condition 和更完整的训练期诊断。

## 详细实验配置

### IBL-MtM combined_e300_aligned

- 数据集：Perich-Miller 2018
- sessions：10
- 采样方式：canonical continuous benchmark protocol
- obs/pred：`500ms / 250ms`

### Neuroformer canonical 500/250 + session conditioning

- 数据集：Perich-Miller 2018
- sessions：10
- 采样方式：canonical continuous benchmark protocol
- obs/pred：`500ms / 250ms`

## 关键超参数

### IBL-MtM

- `epoch = 300`
- `batch_size = 16`
- `lr = 1e-4`
- `weight_decay = 0.01`

### Neuroformer

- `epoch = 50`
- `batch_size = 8`
- `grad_accum_steps = 20`
- `lr = 1e-4`
- `weight_decay = 1.0`
- `warmup_tokens = 8e7`
- `session_embedding_scale = 1.0`

## 训练与评估脚本命令

统一脚本入口：

```bash
bash scripts/phase1-autoregressive-1.8-benchmark_model/20260321_benchmark_ibl_e300_neuroformer_session_conditioning/run_benchmark.sh
```

单项命令：

```bash
python neural-benchmark/faithful_ibl_mtm.py \
  --mode train \
  --epochs 300 \
  --batch-size 16 \
  --grad-accum-steps 1 \
  --num-workers 4 \
  --train-mask-mode combined \
  --output-dir results/logs/phase1-autoregressive-1.8-benchmark_model/20260321_benchmark_ibl_e300_neuroformer_session_conditioning/ibl_mtm_combined_e300_aligned

python neural-benchmark/faithful_neuroformer.py \
  --mode train \
  --obs-window 0.5 \
  --pred-window 0.25 \
  --epochs 50 \
  --batch-size 8 \
  --grad-accum-steps 20 \
  --num-workers 0 \
  --weight-decay 1.0 \
  --warmup-tokens 80000000 \
  --session-embedding-scale 1.0 \
  --output-dir results/logs/phase1-autoregressive-1.8-benchmark_model/20260321_benchmark_ibl_e300_neuroformer_session_conditioning/neuroformer_250ms_session_conditioning_e50

python neural-benchmark/faithful_neuroformer.py \
  --mode eval \
  --obs-window 0.5 \
  --pred-window 0.25 \
  --batch-size 8 \
  --grad-accum-steps 20 \
  --num-workers 0 \
  --weight-decay 1.0 \
  --warmup-tokens 80000000 \
  --session-embedding-scale 1.0 \
  --checkpoint-path results/logs/phase1-autoregressive-1.8-benchmark_model/20260321_benchmark_ibl_e300_neuroformer_session_conditioning/neuroformer_250ms_session_conditioning_e50/best_model.pt \
  --output-dir results/logs/phase1-autoregressive-1.8-benchmark_model/20260321_benchmark_ibl_e300_neuroformer_session_conditioning/neuroformer_250ms_session_conditioning_e50/formal_eval \
  --eval-split both \
  --inference-mode both
```

## 最小验证结果

### IBL-MtM smoke

- smoke 训练/评估已跑通
- 新增诊断指标已成功输出：
  - `valid_poisson_nll`
  - `predicted_to_true_event_ratio_mean`
  - `per_session_metrics`
- smoke valid 显示当前明显存在 count-scale 失配：
  - `predicted_to_true_event_ratio_mean = 8.1558`

### Neuroformer smoke

- smoke 训练/评估已跑通
- 显式 session conditioning 已经接入训练/评估路径
- 训练期新增监控指标已成功输出：
  - `valid_rollout_fp_bps`
  - `valid_true_past_fp_bps`
  - `valid_teacher_forced_loss`
  - `valid_rollout_true_past_gap_fp_bps`
- smoke 诊断信号已经可用于分析失败来源：
  - rollout `predicted_to_true_event_ratio_mean = 2.2207`
  - true_past `predicted_to_true_event_ratio_mean = 0.9189`
  - rollout `max_generate_steps_hit_rate = 1.0`
  - rollout `eos_termination_rate = 0.0`

## 训练 loss 结果

- 正式 run 待回填

## train 期间最佳 val fp-bps

- 待回填

## test fp-bps

- 待回填

## test 使用的 checkpoint 标识 / 时间

- 待回填

## 各条件指标结果

- `IBL-MtM`：至少回填 `fp-bps / per-bin fp-bps / poisson_nll / predicted_to_true_event_ratio / per_session_metrics`
- `Neuroformer`：至少回填 `rollout / true_past fp-bps / per-bin fp-bps / teacher_forced_loss / predicted_to_true_event_ratio / per_session_metrics`

## 与 baseline 的对比

- `IBL-MtM` 对比 `combined_e10 / combined_e50_aligned`
- `Neuroformer` 对比 `canonical e50 aligned`

## 可视化索引

- 待生成后回填：
  - `train_loss_curve.png`
  - `valid_fp_bps_curve.png`
  - `valid_poisson_nll_curve.png`（IBL-MtM）
  - `valid_rollout_fp_bps_curve.png`（Neuroformer）
  - `valid_true_past_fp_bps_curve.png`（Neuroformer）
  - `valid_teacher_forced_loss_curve.png`（Neuroformer）
  - `valid_rollout_vs_true_past_gap_curve.png`（Neuroformer）
  - `lr_curve.png`
  - `training_config_timeline.png`
  - `train_mask_counts_curve.png`（IBL-MtM）
  - `predicted_to_true_event_ratio_curve.png`

## 当前结论

- 当前已完成 runner / 协议 / 记录体系更新，并通过了 IBL-MtM / Neuroformer 的 smoke 验证。
- Neuroformer smoke 已经提供了一个明确线索：在当前 very-short smoke 下，rollout 明显 over-generate，而 true_past 的 count ratio 已接近 1，说明“生成路径尺度失配”值得继续关注。

## 后续安排

1. 先跑 `IBL-MtM combined_e300_aligned`
2. 再跑 `Neuroformer canonical 500/250 + session conditioning`
3. 训练结束后用 `best ckpt` 做 formal `valid/test`
4. 按 `1.8.3` 规范补齐图表、对比和结论

## 执行进展（2026-03-21 22:10 CST）

- 已完成 runner / 协议更新的中间提交并推送：2761a8b 补充1.8 benchmark后续runner与诊断指标
- 后台正式 run 已启动：screen 会话 phase1_benchmark_20260321
- 主日志路径：results/logs/phase1-autoregressive-1.8-benchmark_model/20260321_benchmark_ibl_e300_neuroformer_session_conditioning/main.log
- 当前阶段：IBL-MtM combined_e300_aligned
- 已确认第一轮训练正常进入 epoch 1，当前可见指标：train_loss=0.8346, valid_fp_bps=-2.9818, valid_poisson_nll=0.5902
- 当前输出目录：results/logs/phase1-autoregressive-1.8-benchmark_model/20260321_benchmark_ibl_e300_neuroformer_session_conditioning/ibl_mtm_combined_e300_aligned
- best_model.pt 与 last_model.pt 已开始更新，说明训练未卡死
