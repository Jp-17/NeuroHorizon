# Phase 1.9 模块优化：Local Prediction Memory Decoder

**日期**：2026-03-13
**模块名**：`local_prediction_memory`
**状态**：已放弃
**分支**：`dev/20260313_local_prediction_memory`

## 改进摘要

本次迭代直接针对 `20260312_prediction_memory_decoder` 的失败结论：上一版 teacher forcing 很强，但 rollout 因为全历史高容量 memory 检索而显著崩塌。新的 `local_prediction_memory` 版本保留 structured prediction memory 的思想，但将其约束为“每个 query bin 只访问紧邻上一步的 local memory block”。

## 设计动机

上一版失败的核心不是 `prediction memory` 这个概念本身，而是：
1. decoder 可以访问整段历史 memory，容量太大；
2. teacher forcing 下 local GT counts 形成了过强侧信道；
3. rollout 时早期误差被编码进 memory 后持续传播。

因此本轮优化优先做“收缩通路”，而不是“增强通路”。

## 新方案定义

- 保留 `PredictionMemoryEncoder`
- 保留 `K=4` summary tokens
- `memory[t] = encode(counts[t-1])`
- query `t` 只能访问第 `t` 个 memory block
- 更早历史信息交给 causal self-attention 负责
- local memory 的时间嵌入与 source bin 对齐，第一个 block 为零时间嵌入

## 实施清单

- [x] 新增 `decoder_variant='local_prediction_memory'`
- [x] 新增 local-only prediction memory mask
- [x] 新增 source-aligned memory time embedding
- [x] 新增 local 版本训练配置
- [x] 新增 local 版本验证脚本
- [x] 完成功能验证
- [x] 完成 250ms smoke run

## 预期验证点

1. local mask 正确：query `t` 只能访问 block `t`
2. `shift-right` 仍正确
3. `forward()` 与 `generate()` 不再等价
4. 250ms smoke run 可训练、可保存 checkpoint、可 rollout eval

## 预期对比目标

- 相比 `20260312_prediction_memory_decoder`：
  - rollout fp-bps 应明显提升
  - teacher-forced / rollout gap 应缩小
- 相比 `baseline_v2`：
  - 不要求一开始就超越，但至少不应在长窗口中快速转负

## 当前验证结果

### 功能验证

- local mask 验证通过：
  - query `0` 只访问 block `0`
  - query `1` 只访问 block `1`
- `shift-right` 验证通过：
  - 修改 `target_counts[:, 0, :]` 后，`bin 0` 输出变化 `0.000000`
  - `bin >= 1` 输出变化 `0.003935`
- `forward()` 与 `generate()` 不再等价：
  - `tf_vs_rollout_max_delta=0.000465`

执行命令：

```bash
conda activate poyo
cd /root/autodl-tmp/NeuroHorizon
python scripts/phase1-autoregressive-1.9-module-optimization/20260313_local_prediction_memory/verify_local_prediction_memory.py
```

### 250ms smoke run（1 epoch）

- 配置：
  - `train_1p9_local_prediction_memory_250ms.yaml`
  - override: `epochs=1 eval_epochs=1 batch_size=256 eval_batch_size=256 num_workers=0`
- 训练结果：
  - `train_loss=0.418`
  - `val_loss=0.412`
  - `val/r2=-0.000`
  - `val/fp_bps=-0.825`
- rollout smoke eval：
  - `fp-bps=-0.8234`
  - `R2=-0.0002`
  - `val_loss=0.4134`
- 结论：
  - local prediction memory 版本已达到“可训练、可保存 checkpoint、可 rollout eval”的最小可用状态
  - 正式 300-epoch 实验尚未执行，暂不与前一版或 baseline 做最终结论对比


## 正式实验准备

- 已补充正式实验脚本：
  - `run_local_prediction_memory_experiments.sh`
  - `monitor_local_prediction_memory_progress.py`
  - `collect_local_prediction_memory_results.py`
- 正式实验协议与 `20260312_prediction_memory_decoder` 保持一致：
  - 10 sessions
  - 连续滑动窗口
  - obs=500ms
  - pred=250ms / 500ms / 1000ms
- 启动命令：

```bash
conda activate poyo
cd /root/autodl-tmp/NeuroHorizon
bash scripts/phase1-autoregressive-1.9-module-optimization/20260313_local_prediction_memory/run_local_prediction_memory_experiments.sh
```


## 正式实验结果（300 epochs）

- rollout fp-bps：
  - `250ms = 0.1621`
  - `500ms = -0.0105`
  - `1000ms = -0.2122`
- 相比 `baseline_v2`：
  - `250ms = -0.0494`
  - `500ms = -0.1849`
  - `1000ms = -0.3439`
- 相比 `20260312_prediction_memory_decoder`：
  - `250ms = +0.0135`
  - `500ms = +0.0048`
  - `1000ms = +0.0468`
- teacher-forced / rollout gap：
  - `250ms = 0.1248`
  - `500ms = 0.2951`
  - `1000ms = 0.4853`
- 首次转负 bin：
  - `250ms`: 无
  - `500ms`: `bin 12`
  - `1000ms`: `bin 11`

## 结论与分析

1. `local_prediction_memory` 确实比上一轮 full-history structured memory 更稳，但提升很有限。
2. 结果说明“只收缩 memory 可见性”不足以解决核心问题；主导误差的仍是训练时 `GT counts` memory 与推理时 `predicted expected counts` memory 之间的分布偏移。
3. 显式 prediction memory 通路即使只保留 local block，仍然会形成过强的局部 teacher-forced 侧信道，导致 teacher forcing 很强、rollout 仍弱。
4. 因此本轮也不进入主线，保留为已验证但放弃的 1.9 迭代记录。
