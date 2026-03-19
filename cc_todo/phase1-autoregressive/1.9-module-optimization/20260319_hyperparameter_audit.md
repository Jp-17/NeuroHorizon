# Phase 1.9 模型优化超参数审查与优化空间分析

> **日期**：2026-03-19
> **范围**：`plan.md` §1.9 的 `baseline_v2` 与四轮 1.9 优化尝试（`20260312_prediction_memory_decoder`、`20260313_local_prediction_memory`、`20260313_prediction_memory_alignment`、`20260313_prediction_memory_alignment_tuning`）
> **目的**：核对 1.9 各轮训练/评估超参是否与 `baseline_v2` 保持一致，识别真正的变量变化、对比口径问题，以及后续值得投入的超参数优化空间

## 1. 核心结论

1. **训练主干基本一致，而且明显继承自项目内适配版 POYO+ 而不是 raw defaults**：四轮 1.9 正式实验都复用了 `examples/neurohorizon/train.py`，因此共享同一损失、优化器和 scheduler 逻辑：`PoissonNLLLoss`、`SparseLamb`、`OneCycleLR(cos)`；这条优化主干和 `examples/poyo_plus` 保持同源，但 batch / epoch / precision / eval cadence 已经在 Phase 0/Phase 1 初期被改写过一轮。
2. **真正破坏严格可比性的点只有两个**：
   - `20260313_prediction_memory_alignment` 和 `20260313_prediction_memory_alignment_tuning` 的 `1000ms` 配置把 `batch_size` 从 `32` 提到了 `64`，从而把 `max_lr` 从 `0.001` 抬到了 `0.002`。
   - 1.9 汇总脚本仍以 `results.tsv` 中的 legacy `baseline_v2=0.2115 / 0.1744 / 0.1317` 为参考，而不是当前 evalfix valid / test 主结果。
3. **历史 1.9 主表本质上仍是 continuous-valid rollout 对比，但最优 tuning 分支已经补出 current evalfix valid/test**：批量脚本原始调用 `eval_phase1_v2.py --skip-trial`，且没有传 `--split test`，因此历史 `eval_rollout.json` 默认读取 `valid` split；不过 2026-03-20 已补 `20260313_prediction_memory_alignment_tuning` 的 rollout `valid/test` evalfix JSON 和 trial-aligned 指标，current held-out test 仍低于 baseline_v2：`-0.0232 / -0.0104 / -0.0075`。
4. **超参数优化空间仍然很大，但现在先要做的是“口径收口”而不是盲目搜索**：在 `1000ms` 的 batch/lr 漂移和 current valid/test 还没有对齐到所有 1.9 分支前，继续讨论“是否超过 baseline_v2”并不稳。

## 2. baseline_v2 当前正式训练口径

### 2.1 共享训练主超参

来源：`examples/neurohorizon/configs/defaults.yaml` + `examples/neurohorizon/train.py`

- 损失函数：`PoissonNLLLoss`
- 优化器：`SparseLamb`
- 学习率调度：`OneCycleLR(anneal_strategy="cos", pct_start=0.5, div_factor=1)`
- 基础学习率：`base_lr=3.125e-5`
- 权重衰减：`weight_decay=1e-4`
- 训练脚本实际使用：`max_lr = base_lr × batch_size`
- 共同设置：`epochs=300`、`eval_epochs=10`、`seed=42`、`precision=bf16-mixed`
- 共同训练 transform：`UnitDropout(max_units=200, min_units=30, mode_units=80, peak=4)`
- 数据配置：`examples/neurohorizon/configs/dataset/perich_miller_10sessions.yaml`
- 训练采样：continuous `RandomFixedWindowSampler`
- 当前正式评估主口径：evalfix `SequentialFixedWindowSampler` + 全局累计 `fp-bps / R²`，并区分 valid/test

### 2.2 baseline_v2 的窗口级 batch / max_lr

| window | batch_size | max_lr | 备注 |
|--------|------------|--------|------|
| 250ms | 64 | 0.0020 | 与 1.9 大多数 250ms 一致 |
| 500ms | 64 | 0.0020 | 与 1.9 大多数 500ms 一致 |
| 1000ms | 32 | 0.0010 | baseline_v2 长窗口的正式口径 |

### 2.3 baseline_v2 当前参考值

| reference | 250ms | 500ms | 1000ms | 说明 |
|-----------|-------|-------|--------|------|
| legacy continuous-valid | 0.2115 | 0.1744 | 0.1317 | `results.tsv` 当前仍在用的旧 1.3.4 引用值 |
| evalfix valid | 0.2164 | 0.1823 | 0.1374 | 当前更合理的 valid 主参考 |
| evalfix test | 0.2223 | 0.1740 | 0.1348 | 当前正式 held-out test 主参考 |

### 2.4 与 `examples/poyo_plus` 原始/适配配置相比

| item | raw `examples/poyo_plus/defaults.yaml` | Phase 0 `train_baseline_10sessions.yaml` | `baseline_v2` / 1.9 formal |
|------|----------------------------------------|------------------------------------------|-----------------------------|
| 优化器 / scheduler | `SparseLamb + OneCycleLR(cos)` | 相同 | 相同 |
| `base_lr / weight_decay / lr_decay_start` | `3.125e-5 / 1e-4 / 0.5` | 相同 | 相同 |
| `epochs` | `1000` | `500` | `300` |
| `batch_size` | `128` | `64` | `250/500ms=64`；`1000ms baseline_v2=32`；`alignment/tuning 1000ms=64` |
| `eval_epochs` | `1` | `10` | `10` |
| `precision` | `32` | `bf16-mixed` | `bf16-mixed` |
| train transform | 无额外 transform | `UnitDropout(200,30,80,4)` | 与 Phase 0 适配版相同 |
| train sampler | `RandomFixedWindowSampler` | 相同 | 相同 |
| valid/test sampler | `DistributedStitchingFixedWindowSampler(step=sequence_length/2)` | 相同 | evalfix 口径改为 `SequentialFixedWindowSampler` |
| 任务 / readout | 泛化多任务框架 | `cursor_velocity_2d` 行为解码 | `spike_counts` forward prediction + `fp-bps/R²/PSTH-R²` |

补充判断：
- `baseline_v2` 和 1.9 的训练规程，**更接近项目内 Phase 0 已适配过的 POYO+ baseline**，而不是 raw `poyo_plus` 的原始训练默认值。
- 真正被 1.9 延续下来的，是 `SparseLamb + OneCycleLR(cos)`、线性 lr scaling、`base_lr=3.125e-5`、`weight_decay=1e-4` 这条优化主干；`epochs / batch / precision / eval cadence / transforms` 这些更显眼的训练超参，其实在 Phase 0/Phase 1 交接时就已经改过。
- 因此当前 1.9 超参数讨论，本质上是在 **`baseline_v2`-relative / adapted-POYO+-relative** 的局部优化，而不是重新回到 `examples/poyo_plus` 原始 regime 做 optimizer 级重调。
- 另外，POYO+ Phase 0 的验证/测试主要面向行为解码，使用 `DistributedStitchingFixedWindowSampler(step=0.5s)`；NeuroHorizon baseline_v2 和 1.9 的 current 主口径已经切换到 spike-count forward prediction + evalfix `SequentialFixedWindowSampler`，所以两边数值不能直接横比，只能比较训练规程 lineage。

## 3. 1.9 各轮正式实验的超参与口径审查

| module | 250ms/500ms 与 baseline_v2 的训练口径 | 1000ms 与 baseline_v2 的训练口径 | 对比口径 | 结论 |
|--------|--------------------------------------|----------------------------------|----------|------|
| `20260312_prediction_memory_decoder` | 一致：`batch=64`, `max_lr=0.002` | 一致：`batch=32`, `max_lr=0.001` | continuous-valid rollout vs legacy baseline row | 结构变量主导，训练超参基本公平 |
| `20260313_local_prediction_memory` | 一致：`batch=64`, `max_lr=0.002` | 一致：`batch=32`, `max_lr=0.001` | continuous-valid rollout vs legacy baseline row | 结构变量主导，训练超参基本公平 |
| `20260313_prediction_memory_alignment` | 一致：`batch=64`, `max_lr=0.002` | **不一致**：`batch=64`, `max_lr=0.002` | continuous-valid rollout vs legacy baseline row | `1000ms` 收益混入了 batch/lr 变化 |
| `20260313_prediction_memory_alignment_tuning` | 一致：`batch=64`, `max_lr=0.002` | **不一致**：`batch=64`, `max_lr=0.002` | continuous-valid rollout vs legacy baseline row | `1000ms` 最接近 baseline 的结论不够干净 |

补充说明：
- 四轮 1.9 的批量脚本都额外覆盖了 `num_workers=2`；baseline_v2 主训练脚本未覆盖该项，沿用配置默认 `num_workers=4`。这更接近运行时吞吐差异，不是核心训练语义差异。
- smoke run 的 `epochs=1 / batch_size=256 / eval_batch_size=256 / num_workers=0` 仅用于链路验证，不能和 formal 300-epoch 结果混用。

## 4. 与 baseline_v2 的差值，当前应该怎么看

### 4.1 当前文档里原始引用的 legacy 差值

| module | 250ms | 500ms | 1000ms |
|--------|-------|-------|--------|
| `20260312_prediction_memory_decoder` | -0.0629 | -0.1897 | -0.3907 |
| `20260313_local_prediction_memory` | -0.0494 | -0.1849 | -0.3439 |
| `20260313_prediction_memory_alignment` | -0.0172 | -0.0231 | -0.0214 |
| `20260313_prediction_memory_alignment_tuning` | -0.0111 | -0.0218 | -0.0099 |

这些差值是**相对 legacy baseline_v2 valid 引用值**计算出来的，仍可用于复原当时的实验上下文，但不能直接当作当前正式结论。

### 4.2 若改按 current evalfix valid baseline 计算

| module | 250ms | 500ms | 1000ms |
|--------|-------|-------|--------|
| `20260312_prediction_memory_decoder` | -0.0678 | -0.1976 | -0.3964 |
| `20260313_local_prediction_memory` | -0.0543 | -0.1928 | -0.3496 |
| `20260313_prediction_memory_alignment` | -0.0221 | -0.0310 | -0.0271 |
| `20260313_prediction_memory_alignment_tuning` | -0.0160 | -0.0297 | -0.0156 |

这是一种更合理的 valid 口径，因为 1.9 当前的 `eval_rollout.json` 本身就是 valid split。

### 4.3 若改按 current evalfix test baseline 计算

| module | 250ms | 500ms | 1000ms |
|--------|-------|-------|--------|
| `20260312_prediction_memory_decoder` | -0.0737 | -0.1893 | -0.3938 |
| `20260313_local_prediction_memory` | -0.0602 | -0.1845 | -0.3470 |
| `20260313_prediction_memory_alignment` | -0.0280 | -0.0227 | -0.0245 |
| `20260313_prediction_memory_alignment_tuning` | -0.0219 | -0.0214 | -0.0130 |

这张表对四轮历史模块仍有解释价值，但不能直接当作严格 apples-to-apples：截至 2026-03-20，只有 `20260313_prediction_memory_alignment_tuning` 已经补 current evalfix `test`；其余分支仍主要保留 valid rollout 总表。它至少说明：**即使按更宽松的 legacy valid 口径看起来“只差不到 0.01”，切换到 current test 口径后差距仍然是负的。**

### 4.4 2026-03-20 最优 1.9 分支的 evalfix valid/test 补评估

对象：`20260313_prediction_memory_alignment_tuning`

| window | rollout evalfix valid | vs baseline valid | rollout evalfix test | vs baseline test | test PSTH-R² | vs baseline test PSTH | test trial fp-bps |
|--------|------------------------|-------------------|----------------------|------------------|--------------|-----------------------|-------------------|
| 250ms | `0.1949` | `-0.0215` | `0.1991` | `-0.0232` | `0.6785` | `+0.0705` | `0.2182` |
| 500ms | `0.1635` | `-0.0188` | `0.1637` | `-0.0104` | `0.6210` | `+0.0054` | `0.1841` |
| 1000ms | `0.1264` | `-0.0109` | `0.1273` | `-0.0075` | `0.5738` | `-0.1110` | `0.0744` |

补充判断：
- continuous held-out test `fp-bps` 仍然三窗口全部低于 baseline_v2，说明这轮 tuning 还不能改写“1.9 尚未超过 baseline_v2”的主结论。
- 差距最小的仍是 `1000ms`，但这个窗口仍混入 `batch_size=64 -> max_lr=0.002` 的 parity 问题，因此不能把 `-0.0075` 解读成“已经基本追平”。
- trial-aligned 指标呈现出与 continuous 指标不同的现象：`250/500ms` 的 test `PSTH-R²` 已经略高于 baseline_v2，但 `trial_fp_bps` 仍落后，说明 tuning 更像是在改善 trial-averaged 平滑结构，而不是全面提高 spike-wise information gain。
- `1000ms` 上 trial 指标仍明显落后（`PSTH-R² -0.1110`，`trial fp-bps` 相对 baseline_test 约 `-0.0890`），进一步说明长窗口的“接近 baseline”结论还不稳。

## 5. 当前仍未系统探索的超参数空间

### 5.1 优化器层面基本没有真正探索

到目前为止，1.9 四轮 formal 实验都没有系统改过这些参数：

- `optimizer`：始终为 `SparseLamb`
- `weight_decay`：始终为 `1e-4`
- `lr_decay_start`：始终为 `0.5`
- `div_factor`：始终为 `1`
- `epochs`：始终为 `300`
- `eval_epochs`：始终为 `10`

这意味着“优化器/学习率/权重衰减是否还有明显空间”这个问题，目前**没有被真正回答**；现阶段的 1.9 主要是在比较结构和 memory-specific 训练策略，而不是在比较 optimizer family。

### 5.2 已经动过、但仍未系统扫的超参数

真正被动过的主要只有 memory-specific 超参：

- `decoder_variant`
- `prediction_memory_k=4`
- `prediction_memory_heads=4`
- `prediction_memory_train_mix_prob`
- `prediction_memory_input_dropout`
- `prediction_memory_input_noise_std`

其中只有最后三项在 `20260313_prediction_memory_alignment -> tuning` 之间做了小步调参，而且没有形成系统 grid / schedule 对照。

### 5.3 当前最明显的优化空间

1. **先做公平性收口**
   - `20260313_prediction_memory_alignment` 和 `..._tuning` 的 `1000ms` 需要重跑成与 baseline_v2 同口径：
     - 方案 A：把 `batch_size` 恢复到 `32`
     - 方案 B：保留 `batch_size=64`，但把 `base_lr` 减半到 `1.5625e-5`，让 `max_lr` 保持 `0.001`
   - 这一步不做，当前 `1000ms` “最接近 baseline_v2”的判断都带混杂因素。

2. **先把 1.9 的评估口径补到 evalfix valid/test**
   - 现在最值得做的不是立刻再训新模型，而是用已有 checkpoint 先补：
     - continuous valid rollout
     - continuous test rollout
     - 如果需要，再补 teacher-forced valid/test
   - 至少让 1.9 最优分支和 baseline_v2 站在同一套 valid/test 口径上。

3. **把 500ms 作为下一轮 memory-specific 超参的主战场**
   - 目前 tuning 的 `500ms` 仍然比 legacy baseline 差 `-0.0218`，按 current valid 差 `-0.0297`，是三窗口里最不稳的那个。
   - 这说明下一轮更值得围绕 `500ms` 做：
     - `mix_prob` 网格
     - `dropout / noise` 组合
     - 或 `mix_prob` schedule
   - 因为 `250ms` 已经相对接近，`1000ms` 又被 batch/lr 口径污染，`500ms` 最能反映策略本身是否真正泛化。

4. **再考虑 optimizer 级 sweep**
   - 在 memory-specific 策略已经收口后，再看是否值得做：
     - `weight_decay: 1e-4 -> {5e-5, 2e-4}`
     - `lr_decay_start: 0.5 -> {0.3, 0.7}`
     - `SparseLamb` vs `AdamW`
   - 现阶段直接做 optimizer 大 sweep，解释会和结构变量缠在一起，性价比不高。

## 6. 建议的后续执行顺序

1. **[x] 补文档口径（2026-03-20 已执行）**：`model.md` 已明确区分 legacy baseline、current evalfix valid 和 current evalfix test，并补入 `poyo_plus -> adapted poyo_plus -> baseline_v2 / 1.9` 的训练规程 lineage。
2. **[ ] 重跑 / 重评估 1000ms parity**：优先处理 `alignment` 和 `tuning` 的 `1000ms batch/lr` 不一致。
3. **[x] 给最优 1.9 checkpoint 补 valid/test evalfix（2026-03-20 已执行）**：已补 `20260313_prediction_memory_alignment_tuning` 的 `250 / 500 / 1000ms rollout valid/test`，结果文件为 `eval_rollout_evalfix_{valid,test}.json`。
4. **[ ] 只在口径收口后再做超参调优**：优先从 `500ms` 的 `mix_prob / dropout / noise` 开始，不建议立刻大范围扫 optimizer。

## 7. 本次审查使用的命令

```bash
cd /root/autodl-tmp/NeuroHorizon
sed -n '835,985p' cc_core_files/plan.md
grep -Hn 'epochs:\|batch_size:\|eval_epochs:\|precision:\|seed:\|num_workers:'   examples/neurohorizon/configs/train_v2_250ms.yaml   examples/neurohorizon/configs/train_v2_500ms.yaml   examples/neurohorizon/configs/train_v2_1000ms.yaml   examples/neurohorizon/configs/train_1p9_prediction_memory_250ms.yaml   examples/neurohorizon/configs/train_1p9_prediction_memory_500ms.yaml   examples/neurohorizon/configs/train_1p9_prediction_memory_1000ms.yaml   examples/neurohorizon/configs/train_1p9_local_prediction_memory_250ms.yaml   examples/neurohorizon/configs/train_1p9_local_prediction_memory_500ms.yaml   examples/neurohorizon/configs/train_1p9_local_prediction_memory_1000ms.yaml   examples/neurohorizon/configs/train_1p9_prediction_memory_alignment_250ms.yaml   examples/neurohorizon/configs/train_1p9_prediction_memory_alignment_500ms.yaml   examples/neurohorizon/configs/train_1p9_prediction_memory_alignment_1000ms.yaml   examples/neurohorizon/configs/train_1p9_prediction_memory_alignment_tuning_250ms.yaml   examples/neurohorizon/configs/train_1p9_prediction_memory_alignment_tuning_500ms.yaml   examples/neurohorizon/configs/train_1p9_prediction_memory_alignment_tuning_1000ms.yaml

grep -Hn 'base_lr:\|weight_decay:\|lr_decay_start:' examples/neurohorizon/configs/defaults.yaml
grep -RIn 'SparseLamb\|OneCycleLR\|optim.base_lr\|weight_decay' examples/neurohorizon/train.py
grep -Hn 'epochs:\|batch_size:\|eval_epochs:\|precision:\|seed:\|num_workers:' examples/poyo_plus/configs/defaults.yaml examples/poyo_plus/configs/train_baseline_10sessions.yaml
grep -Hn 'base_lr:\|weight_decay:\|lr_decay_start:' examples/poyo_plus/configs/defaults.yaml
grep -n 'RandomFixedWindowSampler\|DistributedStitchingFixedWindowSampler\|step=self.sequence_length / 2' examples/poyo_plus/train.py
sed -n '1,80p' cc_todo/phase1-autoregressive/1.9-module-optimization/results.tsv
sed -n '1,220p' scripts/phase1-autoregressive-1.9-module-optimization/20260313_prediction_memory_alignment/collect_prediction_memory_alignment_results.py
sed -n '1,220p' scripts/phase1-autoregressive-1.9-module-optimization/20260313_prediction_memory_alignment_tuning/collect_prediction_memory_alignment_tuning_results.py
sed -n '1,200p' results/logs/phase1-autoregressive-1.9-module-optimization/20260313_prediction_memory_alignment_tuning/250ms/eval_rollout.json
python3 scripts/analysis/neurohorizon/eval_phase1_v2.py --log-dir results/logs/phase1-autoregressive-1.9-module-optimization/20260313_prediction_memory_alignment_tuning/250ms --rollout --split valid --output results/logs/phase1-autoregressive-1.9-module-optimization/20260313_prediction_memory_alignment_tuning/250ms/eval_rollout_evalfix_valid.json
python3 scripts/analysis/neurohorizon/eval_phase1_v2.py --log-dir results/logs/phase1-autoregressive-1.9-module-optimization/20260313_prediction_memory_alignment_tuning/250ms --rollout --split test --output results/logs/phase1-autoregressive-1.9-module-optimization/20260313_prediction_memory_alignment_tuning/250ms/eval_rollout_evalfix_test.json
```

## 8. 本次结论的边界

- 这次审查没有新增训练，但 2026-03-20 确实新增了 `20260313_prediction_memory_alignment_tuning` 的 rollout `evalfix valid/test` 补评估。
- 因此本文现在能回答两类问题：一是“当前实验到底是不是同口径”，二是“当前最优 1.9 分支在 current evalfix valid/test 下到底离 baseline_v2 还差多少”。
- 真正要回答“1.9 是否已经超过 baseline_v2”，仍然必须在 evalfix valid/test 口径和 `1000ms` batch/lr 公平性同时收口后再下结论。
