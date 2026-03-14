# Phase 1.9 模块优化：Prediction Memory Alignment Tuning

**日期**：2026-03-13
**模块名**：`prediction_memory_alignment_tuning`
**状态**：验证中
**分支**：`dev/20260313_prediction_memory_alignment_tuning`

## 改进摘要

本轮迭代完全基于 `20260313_prediction_memory_alignment`，不再调整结构和训练逻辑，只做一轮纯超参级小范围 tuning：

- `prediction_memory_train_mix_prob: 0.25 -> 0.35`
- `prediction_memory_input_dropout: 0.10 -> 0.05`
- `prediction_memory_input_noise_std: 0.05 -> 0.03`

## 设计动机

上一轮已经把显式 prediction feedback 的 rollout 表现推到了接近 `baseline_v2` 的水平，剩余差距稳定在约 `0.02 fp-bps`。当前更合理的下一步不是继续改结构，而是测试 alignment 强度与 regularization 强度之间的平衡是否仍偏保守。

本轮的假设是：

1. `mix_prob=0.25` 仍略低，训练期 memory 和推理期 memory 的分布还可以再靠近一些；
2. 上一轮的 `dropout/noise` 已完成“防止 side-channel 作弊”的主要任务，现在可能开始过度抑制 memory 有效信息；
3. 因此应尝试“更强一点的对齐、更轻一点的正则”。

## 新方案定义

- decoder 结构保持 `local_prediction_memory`
- mixed-memory 训练逻辑保持不变
- 只调整三项超参：
  - `mix_prob=0.35`
  - `input_dropout=0.05`
  - `input_noise_std=0.03`

## 实施清单

- [x] 新增 tuning 版模型配置
- [x] 新增 tuning 版验证脚本
- [x] 新增 tuning 版 smoke / 正式实验脚本
- [x] 完成功能验证
- [x] 完成 250ms smoke run
- [x] 执行 Step 2 checkpoint commit + push
- [x] 完成 `250ms / 500ms / 1000ms` 正式实验
- [ ] 执行 Step 4 results commit + push

## 预期验证点

1. tuning 版三项超参值已正确生效
2. mixed-memory / regularization 基础机制未被破坏
3. 250ms smoke run 可训练、可保存 checkpoint、可 rollout eval
4. 若 smoke 正常，再执行 `250ms / 500ms / 1000ms` 正式实验

## 预期对比目标

- 相比 `20260313_prediction_memory_alignment`：
  - 希望 rollout fp-bps 至少在 2/3 个窗口上继续提升
  - 希望进一步缩小与 `baseline_v2` 的约 `0.02 fp-bps` 差距

## 当前验证结果

### 功能验证

- tuning 超参验证通过：
  - `tuned_mix_prob=0.35`
  - `tuned_input_dropout=0.05`
  - `tuned_input_noise_std=0.03`
- mixed-memory / regularization 机制验证通过：
  - `target_independence_delta=0.000000`
  - `train_eval_memory_delta=0.008230`

执行命令：

```bash
conda activate poyo
cd /root/autodl-tmp/NeuroHorizon
python scripts/phase1-autoregressive-1.9-module-optimization/20260313_prediction_memory_alignment_tuning/verify_prediction_memory_alignment_tuning.py
```

### 250ms smoke run（1 epoch）

- 配置：
  - `train_1p9_prediction_memory_alignment_tuning_250ms.yaml`
  - override: `epochs=1 eval_epochs=1 batch_size=256 eval_batch_size=256 num_workers=0`
- 训练结果：
  - `train_loss=0.418`
  - `val_loss=0.411`
  - `val/r2=0.000`
  - `val/fp_bps=-0.823`
- rollout smoke eval：
  - `fp-bps=-0.8217`
  - `R2=0.0002`
  - `val_loss=0.4132`
- 结论：
  - tuning 版已达到“可训练、可保存 checkpoint、可 rollout eval”的最小可用状态
  - 与上一轮 alignment 的 1-epoch 链路表现一致，没有出现明显回归

## 下一步

1. 执行 Step 4 results commit + push
2. 判断是否继续做下一轮更细粒度 tuning
3. 若继续优化，优先测试窗口依赖的 `mix_prob / regularization` 组合

## 正式实验结果（300 epochs, rollout eval）

- rollout fp-bps：
  - `250ms = 0.2004`
  - `500ms = 0.1526`
  - `1000ms = 0.1218`
- 相比 `baseline_v2`：
  - `250ms = -0.0111`
  - `500ms = -0.0218`
  - `1000ms = -0.0099`
- 相比 `20260313_prediction_memory_alignment`：
  - `250ms = +0.0060`
  - `500ms = +0.0013`
  - `1000ms = +0.0115`
- teacher-forced / rollout gap：
  - `250ms = 0.0711`
  - `500ms = 0.1197`
  - `1000ms = 0.1656`

## 结果分析

1. 这轮 tuning 是有效的，但收益是“小幅继续推进”，不是跳变式提升。
2. `250ms` 和 `1000ms` 继续改善，`500ms` 基本持平，说明“更强对齐 + 更轻正则”总体方向正确，但窗口间最佳平衡可能并不完全一致。
3. 当前最好的结果来自 `1000ms`，已经把与 `baseline_v2` 的差距压到 `0.0099 fp-bps`。
4. 相比上一轮 alignment，三个窗口都没有退化，因此这组超参可以视为当前显式 prediction-memory 路线上的新 best-known setting。
