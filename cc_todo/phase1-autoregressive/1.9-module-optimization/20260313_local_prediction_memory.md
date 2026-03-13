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

## 补充讨论：为什么 baseline_v2 / Neuroformer / NDT2 / IBL-MtM 的行为不同

### 1. 为什么 `baseline_v2` 没有出现同等级的 rollout 退化

关键点是：当前仓库里的 `baseline_v2` 并不是“带显式 prediction feedback 的真 rollout 模型”。

- 在 `torch_brain/models/neurohorizon.py` 中，`baseline_v2` 对应的是 `decoder_variant='query_aug'` 且配置里 `feedback_method: none`。
- 这意味着：
  - `forward()` 时不会构造任何 feedback；
  - `generate()` 时同样不会把前一步预测重新编码后喂回 decoder。
- 因此它的 train / inference 主体几乎是同一个模型：
  - history events -> POYO latents
  - learnable future bin queries
  - history cross-attn + causal self-attn
  - per-neuron readout

它在评估脚本里虽然也支持 `--rollout`，但这个 rollout 主要只是“按时间步逐步解码同一个 causal decoder”，而不是“每一步再把自己上一步的预测编码成一条新的高容量 side channel 再喂回来”。也就是说：

- `baseline_v2` 的 rollout 误差主要来自 decoder hidden state 的时间传播；
- `local_prediction_memory` 的 rollout 误差除了 hidden state 传播，还多了一条显式的 prediction-memory 反馈通路，而且这条通路在训练和推理时输入分布不一样。

所以两者不是同强度的问题。`baseline_v2` 更像“弱 AR / causal parallel predictor”，而当前两版 prediction memory 则是“强显式反馈 AR”。后者自然更容易出现 teacher forcing 很好、free rollout 明显变差的现象。

### 2. 为什么当前 benchmark 里的 `Neuroformer` 没有看到这么明显的退化

当前仓库里的 benchmark `Neuroformer` 适配器并不是原论文那种 event-level token autoregressive generation。`neural_benchmark/adapters/neuroformer_adapter.py` 里的实现本质上是：

- 输入是 binned spike counts `[B, T, N]`
- prediction window 的输入直接置零
- 用一个 causal transformer 一次性输出整段 future window 的 `log_rate`

也就是说，它更像“GPT-style causal masked one-shot future predictor”，而不是“每预测一个未来步就把该步输出再编码成新的输入 token 继续生成”的完整 generative loop。

因此它没有当前 `prediction_memory` 方案里的两个核心难点：

- 没有单独的 `GT memory -> predicted memory` 替换过程；
- 没有额外的 structured feedback side channel。

哪怕回到更接近原始 Neuroformer 的真正 event-level AR 模型，通常也还是比现在这个 prediction-memory 方案更容易保持 train / inference 一致性。原因是原始 AR token generation 至少保持了“训练和推理都在同一 token 空间里滚动”，而当前方案是在 decoder 主干之外额外挂了一条 memory encoder 通路：

- 训练时这条通路吃的是 `log1p(GT counts)`；
- 推理时这条通路吃的是 `log1p(predicted expected counts)`。

这会制造一个更强、更明确的 side-channel mismatch。

### 3. NDT2 和 IBL-MtM 这样的非显式 AR 架构，为什么在 fp-bps 上可能更占优

当前 benchmark 里的 `NDT2` 和 `IBL-MtM` 都不是 rollout-based multi-step generator。

- `NDT2`：future bins 直接置零，使用 bidirectional encoder 一次性预测整个 future window。
- `IBL-MtM`：future bins 置零，再加 causal mask，但依然是一趟 forward 直接输出整段 future window。

它们在评估时的输入条件与任务定义都是：

- 只看 observation window；
- 一次性预测 prediction window；
- 然后直接在 prediction window 上算 `fp-bps / R2 / Poisson NLL`。

从这个指标定义出发，one-shot masked predictor 天然有两个现实优势：

1. 没有 rollout 误差累积。
2. 训练和推理路径基本一致，不会出现“训练喂 GT side channel，推理换成 predicted side channel”的显式分布跳变。

所以并不是“非 AR 在原理上一定更强”，而是：

- 在当前 Phase 1 的任务形式下，`fp-bps` 只关心“给定观察窗后，你对未来窗 rate/count 的预测是否准确”；
- 它并不额外奖励“你是否一步一步自回归生成出来”。

因此只要 one-shot masked predictor 能更稳定地拟合 future window，它在这个指标上完全可能优于一个更难训练、更容易 exposure bias 的 AR 架构。

### 4. 当前结果真正说明了什么

综合代码路径和实验结果，当前结论更准确地应该表述为：

- `baseline_v2` 的强项不在于“它是更好的显式 AR”，而在于“它几乎没有显式 prediction-feedback mismatch”。
- benchmark 的 `Neuroformer / NDT2 / IBL-MtM` 的强项也不在于“它们天然更先进”，而在于“它们的训练-推理一致性更强，且不承担 rollout feedback drift”。
- 当前两版 `prediction_memory` 失败的根因不是“decoder 必须不能 AR”，而是“我们新增的 AR feedback 通路太容易在 teacher forcing 下变成捷径”。

所以后续优化的正确方向不是继续只改 memory visibility，而是优先处理两件事：

1. 训练时让 memory 输入部分接近推理期分布，而不是纯 `GT counts`。
2. 主动削弱 memory encoder 这条 side channel 的可靠性，让 decoder 不能把它当作过于确定的局部教师信号。

这也是下一轮 1.9 迭代优先尝试 `mixed GT/predicted memory` 与 `memory-input noise/dropout` 的直接原因。
