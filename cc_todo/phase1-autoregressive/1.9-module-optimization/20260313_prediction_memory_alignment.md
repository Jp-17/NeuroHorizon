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
- [x] 执行 Step 2 checkpoint commit + push
- [x] 完成 `250ms / 500ms / 1000ms` 正式实验
- [ ] 执行 Step 4 results commit + push

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

1. 执行 Step 4 results commit + push
2. 基于正式结果判断是否继续做下一轮 tuning
3. 若继续优化，优先尝试 `mix_prob` 调度、regularization 强度和 bootstrap 策略

## 正式实验结果（300 epochs, rollout eval）

- rollout fp-bps：
  - `250ms = 0.1943`
  - `500ms = 0.1513`
  - `1000ms = 0.1103`
- 相比 `baseline_v2`：
  - `250ms = -0.0172`
  - `500ms = -0.0231`
  - `1000ms = -0.0214`
- 相比 `20260313_local_prediction_memory`：
  - `250ms = +0.0322`
  - `500ms = +0.1618`
  - `1000ms = +0.3225`
- teacher-forced / rollout gap：
  - `250ms = 0.0814`
  - `500ms = 0.1319`
  - `1000ms = 0.1718`

## 结果分析

1. 这轮是当前第一版真正接近 `baseline_v2` 的显式 prediction-memory 方案，三个窗口都把差距压到了约 `0.02 fp-bps`。
2. 和 `20260313_local_prediction_memory` 相比，提升不只是“略好”，而是方向性改善已经非常明确，尤其长窗口提升最大。
3. 训练期 memory 输入对齐确实击中了核心问题：
   - rollout gap 大幅缩小
   - per-bin fp-bps 在三个窗口上都保持正值，不再出现上一轮那种中后段转负崩塌
4. 当前仍略低于 `baseline_v2`，说明显式 feedback 通路还没有完全把额外复杂度转化成稳定收益；但这已经不再是“失败迭代”，而是一个值得继续细调的强候选方案。

## 2026-03-19 补充：training curves 可视化

- 脚本：`scripts/phase1-autoregressive-1.9-module-optimization/plot_optimization_training_curves.py`
- 图表：`results/figures/phase1-autoregressive-1.9-module-optimization/20260313_prediction_memory_alignment/training_curves.png`
- 数据来源：
  - `results/logs/phase1-autoregressive-1.9-module-optimization/20260313_prediction_memory_alignment/250ms/lightning_logs/version_1/metrics.csv`
  - `results/logs/phase1-autoregressive-1.9-module-optimization/20260313_prediction_memory_alignment/500ms/lightning_logs/version_0/metrics.csv`
  - `results/logs/phase1-autoregressive-1.9-module-optimization/20260313_prediction_memory_alignment/1000ms/lightning_logs/version_0/metrics.csv`
- 图表内容：
  - 上排为 `250/500/1000ms` 三个窗口的 epoch-level `train_loss` 与 `val_loss`
  - 下排为训练期 `val/fp_bps` 曲线，并叠加 post-train teacher-forced、rollout 与 `baseline_v2` 参考线
- 观察结论：
  1. 三个窗口的 `val/fp_bps` 都持续抬升到训练后期，不再像上一轮那样较早停滞，说明 training-time memory alignment 的收益是贯穿训练过程的，而不是偶然来自某一小段 epoch。
  2. `1000ms` 曲线改善最明显，且 rollout 参考线和训练末期曲线的距离明显缩小，和“长窗口提升最大、rollout gap 显著收缩”的正式结论一致。

执行命令：

```bash
conda activate poyo
cd /root/autodl-tmp/NeuroHorizon
python scripts/phase1-autoregressive-1.9-module-optimization/plot_optimization_training_curves.py --module 20260313_prediction_memory_alignment
```

## 2026-03-20 补充：best-ckpt valid/test 回填与 checkpoint 审计

- 协议确认：
  - 训练入口：`examples/neurohorizon/train.py`
  - 离线正式评估入口：`scripts/analysis/neurohorizon/eval_phase1_v2.py`
  - 数据：`examples/neurohorizon/configs/dataset/perich_miller_10sessions.yaml`
  - continuous train sampler：`RandomFixedWindowSampler`
  - continuous valid/test sampler：`SequentialFixedWindowSampler`
- checkpoint 审计结论：
  - 历史 `last.ckpt` 与 monitored checkpoint 实际同 epoch，本次回填仍明确按 best checkpoint 生成 `valid/test` JSON
  - best checkpoint 分别位于 `epoch 229 / 249 / 239`

| 窗口 | best ckpt | TF valid | TF test | rollout valid | rollout test |
|------|-----------|----------|---------|---------------|--------------|
| 250ms | `epoch=229-step=30590.ckpt` | 0.2690 | 0.2747 | 0.1904 | 0.2057 |
| 500ms | `epoch=249-step=24750.ckpt` | 0.2799 | 0.2729 | 0.1623 | 0.1614 |
| 1000ms | `epoch=239-step=15600.ckpt` | 0.2762 | 0.2773 | 0.1120 | 0.1249 |

补查命令：

```bash
conda activate poyo
cd /root/autodl-tmp/NeuroHorizon
python scripts/phase1-autoregressive-1.9-module-optimization/backfill_best_checkpoint_evals.py
```

补充结论：
1. best-ckpt 回填后，本轮在 `250/500/1000ms` 的 rollout-valid 仍然整体逼近 `baseline_v2`，尤其 `500ms` 保持在 `0.1623`，说明 alignment 训练策略确实显著缓解了前两轮的 rollout 崩塌。
2. `250ms` 的 rollout-test 反而高于 rollout-valid，提示本轮 improvement 不是单纯的 validation 偶然波动。
