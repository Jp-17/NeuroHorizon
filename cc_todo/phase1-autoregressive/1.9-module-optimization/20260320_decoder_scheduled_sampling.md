# Phase 1.9 模块优化：Decoder Scheduled Sampling

**日期**：2026-03-20
**模块名**：`decoder_scheduled_sampling`
**状态**：实施中
**分支**：`dev/20260320_decoder_scheduled_sampling`

## 改进摘要

本轮在当前最优 `20260313_prediction_memory_alignment_tuning` 路线上新增一条训练期 `decoder scheduled sampling` 路径，用于直接缓解 `teacher-forced` 与 `rollout` 之间的暴露偏差。

本轮明确区分两类机制：

1. `prediction_memory_train_mix_prob`
   - 用 `generate()` 的 bootstrap rollout counts 替换部分 training-time memory token 来源
   - 作用点是 memory 通道
   - 主训练 forward 仍是并行 teacher-forced
2. `decoder scheduled sampling`
   - 训练时按时间步自回归解码
   - 第 `t+1` 步 conditioning 在 GT counts 与模型预测 counts 之间按概率切换
   - 作用点是 decoder 主训练路径

## 设计动机

- 现有 1.9 路线虽然把 rollout `fp-bps` 推到了接近 `baseline_v2` 的水平，但 `teacher-forced fp-bps` 仍明显更高，说明暴露偏差没有真正消失。
- 当前 `mix_prob` 只解决了 memory side channel 的分布失配，未真正改变 decoder 主训练路径的 teacher-forced 语义。
- 因此本轮要验证：把训练 forward 自身变成更接近 rollout 的条件分布，是否能比 memory-only 路线更有效地缩小 gap。

## 新方案定义

- 新增 `decoder_train_mode`：
  - `parallel_teacher_forced`
  - `scheduled_sampling`
- 新增 `decoder_rollout_prob_mode`：
  - `fixed`
  - `linear_ramp`
- 新增 `decoder_rollout_prob{,_start,_end,_ramp_epochs}` 等配置
- 对 `local_prediction_memory` / `prediction_memory` / 显式 feedback 路线支持 training-time step-by-step autoregressive decode
- 对 `baseline_v2` 类无显式 conditioning 路线显式报错

## 实施清单

- [x] 新增 decoder scheduled sampling 训练路径
- [x] 新增三窗口基础模型配置
- [x] 新增三窗口基础训练配置
- [x] 新增模块级功能验证脚本
- [x] 新增 smoke / 正式实验脚本
- [x] 修复训练入口优先命中当前 worktree 源码
- [x] 修复评估入口优先命中当前 worktree 源码
- [x] 完成 smoke matrix
- [x] 执行 Step 2 checkpoint commit + push
- [ ] 完成正式实验 matrix
- [ ] 更新 `results.tsv` / `results.md` / `progress.md`
- [ ] 执行 Step 4 results commit + push

## 当前实现内容

1. `torch_brain/models/neurohorizon.py`
   - 增加 `decoder_train_mode`、`decoder_rollout_prob_mode`、`decoder_rollout_prob*`
   - 增加 training-time autoregressive decode 路径
   - 支持 `rollout_prob=0` 回退到 teacher-forced 语义
   - 支持 `rollout_prob=1` 完全使用模型预测作为条件输入
   - 保留现有 `prediction_memory_train_mix_prob` 机制，用于 `memory-only` 或 `hybrid`

2. `examples/neurohorizon/train.py`
   - 按 epoch 计算当前 `decoder_rollout_prob`
   - 写入 `train/decoder_rollout_prob`
   - 显式优先导入当前 repo 源码，而不是环境里的旧安装包

3. `scripts/analysis/neurohorizon/eval_phase1_v2.py`
   - 显式优先导入当前 repo 源码，确保 best-ckpt eval 与训练使用同一版模型定义

4. 新增配置与实验脚本
   - `examples/neurohorizon/configs/model/neurohorizon_decoder_scheduled_sampling_{250,500,1000}ms.yaml`
   - `examples/neurohorizon/configs/train_1p9_decoder_scheduled_sampling_{250,500,1000}ms.yaml`
   - `scripts/phase1-autoregressive-1.9-module-optimization/20260320_decoder_scheduled_sampling/`

## 功能验证

执行命令：

```bash
conda activate poyo
cd /root/autodl-tmp/NeuroHorizon_dev_20260320_decoder_scheduled_sampling
python scripts/phase1-autoregressive-1.9-module-optimization/20260320_decoder_scheduled_sampling/verify_decoder_scheduled_sampling.py
```

当前验证结果：

- `baseline_v2` 类无显式 conditioning 路线会显式拒绝 scheduled sampling
- `rollout_prob=0`：
  - `zero_prob_delta=7.450581e-09`
- `rollout_prob=1`：
  - `target_independence_delta=0.000000e+00`
- `generate_shape=(2, 4, 6)`

结论：

- 新训练路径的核心语义已成立
- 当前实现已经达到“可进入 smoke matrix”的阶段

## 计划中的正式实验矩阵

三窗口统一按 `1.9.0` 协议执行：

- 数据：`perich_miller_10sessions`
- 采样：continuous train / continuous valid / continuous test
- `obs_window=500ms`
- `pred_window=250ms / 500ms / 1000ms`

设置矩阵：

1. `memory_only_mix035`
2. `decoder_ss_fixed_025`
3. `decoder_ss_fixed_050`
4. `decoder_ss_fixed_075`
5. `decoder_ss_linear_0_to_050`
6. `decoder_ss_linear_0_to_075`
7. `hybrid_mix035_plus_linear_050`

## 执行命令记录

最小 smoke：

```bash
conda activate poyo
cd /root/autodl-tmp/NeuroHorizon_dev_20260320_decoder_scheduled_sampling
bash scripts/phase1-autoregressive-1.9-module-optimization/20260320_decoder_scheduled_sampling/run_decoder_scheduled_sampling_smoke.sh
```

正式实验：

```bash
conda activate poyo
cd /root/autodl-tmp/NeuroHorizon_dev_20260320_decoder_scheduled_sampling
bash scripts/phase1-autoregressive-1.9-module-optimization/20260320_decoder_scheduled_sampling/run_decoder_scheduled_sampling_experiments.sh
```

结果汇总：

```bash
conda activate poyo
cd /root/autodl-tmp/NeuroHorizon_dev_20260320_decoder_scheduled_sampling
python scripts/phase1-autoregressive-1.9-module-optimization/20260320_decoder_scheduled_sampling/collect_decoder_scheduled_sampling_results.py
```

## 2026-03-20 09:30 CST 补充：smoke 完成并启动正式实验

### smoke 当前结果

- `run_decoder_scheduled_sampling_smoke.sh` 已完整跑完
- 已覆盖：
  - `memory_only_mix035`
  - `decoder_ss_fixed_050`
  - `decoder_ss_linear_0_to_050`
  - `hybrid_mix035_plus_linear_050`
- 每个 setting 都覆盖了：
  - `250ms / 500ms / 1000ms`
  - best-ckpt `teacher-forced valid`
  - best-ckpt `rollout valid`

### 继续执行情况

- Step 2 checkpoint commit 已完成并推送：
  - commit: `0b79c60`
  - branch: `dev/20260320_decoder_scheduled_sampling`
- 正式实验已放入 `screen`：
  - session: `phase1_decoder_ss_formal`
  - 当前从 `memory_only_mix035 / 250ms` 开始顺序执行 `7 settings × 3 windows`

### 遇到的问题与解决

- 原先用于等待 smoke 的 formal `screen` 命令把自身也匹配进 `pgrep -f run_decoder_scheduled_sampling_smoke.sh`
  - 现象：smoke 完成后 formal 没有自动接棒
  - 解决：手动终止旧 waiting screen，并在 smoke 确认结束后直接启动正式实验 screen

## 2026-03-21 20:42 CST 补充：正式矩阵调整为先做 250ms setting 筛选

### 调度调整原因

- 按当前单卡顺序执行 `7 settings × 3 windows × 300 epochs`，剩余时间已拉长到约 11-12.5 天。
- 因此改为先完成 `7 settings` 的 `250ms`，用中期 ranking 先判断哪类 setting 更值得继续扩展到 `500ms / 1000ms`。

### 已执行调整

- 已手动停止原 `screen` 会话 `phase1_decoder_ss_formal`
- 已中断正在运行的 `decoder_ss_fixed_025 / 1000ms` 正式训练
- 其未完成日志已归档到：
  - `results/logs/phase1-autoregressive-1.9-module-optimization/20260320_decoder_scheduled_sampling/decoder_ss_fixed_025/1000ms_interrupted_20260321_prioritize_250ms`
- 已保留的正式结果：
  - `memory_only_mix035`: `250ms / 500ms / 1000ms`
  - `decoder_ss_fixed_025`: `250ms / 500ms`

### 新增执行入口

- 新 `screen`：`phase1_decoder_ss_250ms_screening`
- 新 runner：`run_decoder_scheduled_sampling_250ms_screening.sh`
- 新 collector：`collect_decoder_scheduled_sampling_250ms_screening.py`
- 新 collector 输出：
  - `results/figures/phase1-autoregressive-1.9-module-optimization/20260320_decoder_scheduled_sampling/decoder_scheduled_sampling_250ms_screening_summary.{json,md,tsv}`

### 当前 250ms 中期排名（2/7 complete）

1. `decoder_ss_fixed_025`
   - rollout test fp-bps: `0.2230`
   - test gap: `0.0638`
2. `memory_only_mix035`
   - rollout test fp-bps: `0.2176`
   - test gap: `0.0617`

### 后续执行顺序

- 先补齐剩余 5 个 setting 的 `250ms`
- 刷新 `250ms` ranking summary
- 再基于 `rollout test fp-bps` 和 `teacher-forced - rollout gap` 决定后续扩展到 `500ms / 1000ms` 的 setting

## 2026-03-22 05:26 CST 补充：250ms screening 已完成 4/7

### 当前运行状态

- `screen` 会话：`phase1_decoder_ss_250ms_screening`
- 已完成的 `250ms` setting：
  - `memory_only_mix035`
  - `decoder_ss_fixed_025`
  - `decoder_ss_fixed_050`
  - `decoder_ss_fixed_075`
- 正在运行：`decoder_ss_linear_0_to_050 / 250ms`
  - 最新训练进度：`epoch 112`
  - 最新验证进度：`epoch 109`
- 尚未开始：
  - `decoder_ss_linear_0_to_075 / 250ms`
  - `hybrid_mix035_plus_linear_050 / 250ms`

### 当前 250ms 排名（4/7 complete）

1. `decoder_ss_fixed_025`
   - rollout test fp-bps: `0.2230`
   - test gap: `0.0638`
2. `decoder_ss_fixed_075`
   - rollout test fp-bps: `0.2190`
   - test gap: `0.0377`
3. `memory_only_mix035`
   - rollout test fp-bps: `0.2176`
   - test gap: `0.0617`
4. `decoder_ss_fixed_050`
   - rollout test fp-bps: `0.2173`
   - test gap: `0.0568`

### 阶段性观察

- 目前 `rollout test fp-bps` 最好的 setting 仍是 `decoder_ss_fixed_025`
- 当前暴露偏差 gap 最小的是 `decoder_ss_fixed_075`
- `decoder_ss_fixed_050` 尚未表现出明显优于 control 的优势
- 是否继续扩展到 `500ms / 1000ms`，仍需等待 `linear` 与 `hybrid` 三组 `250ms` 结果补齐后再判断
