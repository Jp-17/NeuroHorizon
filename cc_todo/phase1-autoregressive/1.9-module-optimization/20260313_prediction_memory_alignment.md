# Phase 1.9 模块优化：Prediction Memory Alignment Training

**日期**：2026-03-13
**模块名**：`prediction_memory_alignment`
**状态**：验证中
**分支**：`dev/20260313_prediction_memory_alignment`

## 改进摘要

本轮迭代不再继续修改 `local_prediction_memory` 的 decoder 结构，而是直接针对上一轮暴露出的 train / inference mismatch 做训练期对齐。核心做法是：

- 训练时将一部分 prediction memory 输入从 `shift-right GT counts` 替换为模型自己 rollout 得到的 `predicted expected counts`
- 对 memory encoder 输入施加轻量 noise / dropout，削弱其作为局部 teacher-forced 侧信道的确定性

## 设计动机

`20260313_local_prediction_memory` 的结果已经说明：

1. 问题不只是 full-history memory 检索过强；
2. 即使 local-only memory，teacher forcing 和 rollout 之间仍有明显 gap；
3. 核心矛盾是训练时 memory 吃 `GT counts`，推理时 memory 吃 `predicted expected counts`。

因此本轮不再继续改 mask，而是优先做 memory 输入对齐和 side-channel regularization。

## 新方案定义

- decoder 结构保持 `local_prediction_memory`
- 新增训练期参数：
  - `prediction_memory_train_mix_prob`
  - `prediction_memory_input_dropout`
  - `prediction_memory_input_noise_std`
- mixed-memory 规则：
  - 先做一次 no-grad rollout bootstrap，得到 `predicted expected counts`
  - 构造 `shift-right GT counts` 与 `shift-right predicted expected counts`
  - 按时间步做整段 population 级混合
- memory regularization：
  - 在 `log1p(count)` 之后加入 additive noise
  - 再对 transformed count 做无重缩放 dropout

## 实施清单

- [x] 新增 mixed-memory 训练逻辑
- [x] 新增 memory-input noise/dropout
- [x] 新增 alignment 版本模型配置
- [x] 新增 alignment 版本验证脚本
- [x] 新增 alignment 版本 smoke / 正式实验脚本
- [x] 完成功能验证
- [x] 完成 250ms smoke run
- [ ] 执行 Step 2 checkpoint commit + push

## 预期验证点

1. `mix_prob=1.0` 时，训练态 `forward()` 不再依赖 `target_counts`
2. 开启 regularization 后，train-time memory tokens 与 eval-time memory tokens 不同
3. 250ms smoke run 可训练、可保存 checkpoint、可 rollout eval
4. 若 smoke 正常，再启动 `250ms / 500ms / 1000ms` 三窗口正式实验

## 预期对比目标

- 相比 `20260313_local_prediction_memory`：
  - teacher-forced / rollout gap 应缩小
  - `500ms / 1000ms` 的 rollout 中后段不应那么早转负
- 相比 `baseline_v2`：
  - 不要求第一轮就全面超过，但至少希望缩小长窗口上的显著退化

## 当前验证结果

### 功能验证

- mixed-memory 目标独立性验证通过：
  - `target_independence_delta=0.000000`
  - 含义：当 `prediction_memory_train_mix_prob=1.0` 且关闭 memory regularization 时，训练态 forward 不再读取 GT target counts
- memory regularization 验证通过：
  - `train_eval_memory_delta=0.011355`
  - 含义：train-time noise/dropout 确实在扰动 memory tokens

执行命令：

```bash
conda activate poyo
cd /root/autodl-tmp/NeuroHorizon
python scripts/phase1-autoregressive-1.9-module-optimization/20260313_prediction_memory_alignment/verify_prediction_memory_alignment.py
```

### 250ms smoke run（1 epoch）

- 配置：
  - `train_1p9_prediction_memory_alignment_250ms.yaml`
  - override: `epochs=1 eval_epochs=1 batch_size=256 eval_batch_size=256 num_workers=0`
- 训练结果：
  - `train_loss=0.418`
  - `val_loss=0.412`
  - `val/r2=-0.000`
  - `val/fp_bps=-0.824`
- rollout smoke eval：
  - `fp-bps=-0.8228`
  - `R2=-0.0000`
  - `val_loss=0.4133`
- 结论：
  - alignment 版本已达到“可训练、可保存 checkpoint、可 rollout eval”的最小可用状态
  - 当前结果仅用于链路验证，不作为与 `local_prediction_memory` 或 `baseline_v2` 的正式优劣判断

## 下一步

1. 执行 Step 2 checkpoint commit + push
2. 启动 `250ms / 500ms / 1000ms` 三窗口正式实验
3. 实验完成后汇总 `teacher-forced / rollout gap`、per-bin fp-bps 和相对 `baseline_v2` 的变化
