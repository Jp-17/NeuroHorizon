# 20260321 Benchmark IBL e300 + Neuroformer Session Conditioning

> 状态：进行中（IBL-MtM e300 已完成，Neuroformer canonical 正在修复后重启）
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

Neuroformer 单独重启入口：

```bash
bash scripts/phase1-autoregressive-1.8-benchmark_model/20260321_benchmark_ibl_e300_neuroformer_session_conditioning/rerun_neuroformer_only.sh
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

- `IBL-MtM combined_e300_aligned`
  - history 长度：`300 epochs`
  - 训练 loss：`0.8346 -> 0.3037`
  - 对应图：`train_loss_curve.png`
- `Neuroformer canonical 500/250 + session conditioning`
  - 首次正式 run 尚未进入有效训练；在 loader 构造阶段因 `session_to_idx` 漏传直接退出
  - 当前已定位为 runner wiring bug，修复后将单独重启 Neuroformer 段

## train 期间最佳 val fp-bps

- `IBL-MtM combined_e300_aligned`
  - best epoch：`282`
  - best valid `fp-bps = 0.1938`
- `Neuroformer canonical 500/250 + session conditioning`
  - 首次正式 run 未产生有效 best checkpoint，待重启后回填

## test fp-bps

- `IBL-MtM combined_e300_aligned`
  - formal test `fp-bps = 0.1938`
- `Neuroformer canonical 500/250 + session conditioning`
  - 首次正式 run 未完成，待重启后回填

## test 使用的 checkpoint 标识 / 时间

- `IBL-MtM combined_e300_aligned`
  - checkpoint：`best_model.pt`（epoch `282`）
  - 文件时间：`2026-03-21 22:59`
- `Neuroformer canonical 500/250 + session conditioning`
  - 首次正式 run 未产生可用 checkpoint，待重启后回填

## 各条件指标结果

- `IBL-MtM combined_e300_aligned`
  - formal valid `fp-bps = 0.1938`
  - formal test `fp-bps = 0.1938`
  - formal test `poisson_nll = 0.3046`
  - formal test `predicted_to_true_event_ratio_mean = 11.1438`
  - 训练末轮 `predicted_to_true_event_ratio_mean = 11.2865`
  - `per-bin fp-bps / per_session_metrics` 已写入 `results.json`
- `Neuroformer`：至少回填 `rollout / true_past fp-bps / per-bin fp-bps / teacher_forced_loss / predicted_to_true_event_ratio / per_session_metrics`

## 与 baseline 的对比

- `IBL-MtM` 对比 `combined_e10 / combined_e50_aligned`
- `Neuroformer` 对比 `canonical e50 aligned`

### 当前已知对比（IBL-MtM）

- `combined_e10` test `fp-bps = -0.0017`
- `combined_e50_aligned` test `fp-bps = 0.1345`
- `combined_e300_aligned` test `fp-bps = 0.1938`
- 当前判断：
  - `IBL-MtM` 继续训练仍然有效，尚未完全平台化
  - 但 `predicted_to_true_event_ratio_mean` 长期维持在 `~11x`，说明后续优化不能只盯 `fp-bps`，还要看输出尺度校准 / objective mismatch

## 可视化索引

- `IBL-MtM combined_e300_aligned`
  - `results/logs/phase1-autoregressive-1.8-benchmark_model/20260321_benchmark_ibl_e300_neuroformer_session_conditioning/ibl_mtm_combined_e300_aligned/train_loss_curve.png`
  - `results/logs/phase1-autoregressive-1.8-benchmark_model/20260321_benchmark_ibl_e300_neuroformer_session_conditioning/ibl_mtm_combined_e300_aligned/valid_fp_bps_curve.png`
  - `results/logs/phase1-autoregressive-1.8-benchmark_model/20260321_benchmark_ibl_e300_neuroformer_session_conditioning/ibl_mtm_combined_e300_aligned/valid_poisson_nll_curve.png`
  - `results/logs/phase1-autoregressive-1.8-benchmark_model/20260321_benchmark_ibl_e300_neuroformer_session_conditioning/ibl_mtm_combined_e300_aligned/lr_curve.png`
  - `results/logs/phase1-autoregressive-1.8-benchmark_model/20260321_benchmark_ibl_e300_neuroformer_session_conditioning/ibl_mtm_combined_e300_aligned/training_config_timeline.png`
  - `results/logs/phase1-autoregressive-1.8-benchmark_model/20260321_benchmark_ibl_e300_neuroformer_session_conditioning/ibl_mtm_combined_e300_aligned/train_mask_counts_curve.png`
  - `results/logs/phase1-autoregressive-1.8-benchmark_model/20260321_benchmark_ibl_e300_neuroformer_session_conditioning/ibl_mtm_combined_e300_aligned/predicted_to_true_event_ratio_curve.png`
- `Neuroformer canonical 500/250 + session conditioning`
  - 首次正式 run 尚未产出有效训练曲线；修复 loader 漏传后重启

## 当前结论

- 当前已完成 runner / 协议 / 记录体系更新，并通过了 IBL-MtM / Neuroformer 的 smoke 验证。
- Neuroformer smoke 已经提供了一个明确线索：在当前 very-short smoke 下，rollout 明显 over-generate，而 true_past 的 count ratio 已接近 1，说明“生成路径尺度失配”值得继续关注。
- `IBL-MtM combined_e300_aligned` 已经把 test `fp-bps` 推到 `0.1938`，说明这条 faithful benchmark 线继续训练仍然有明显收益。
- 但 IBL-MtM 当前新增诊断也显示：`predicted_to_true_event_ratio_mean` 仍然在 `~11x`，后续应重点判断 `fp-bps` 提升是否主要来自相对排序改善，而不是 count-scale 已经校准。
- `Neuroformer canonical 500/250 + session conditioning` 首次正式 run 没有形成有效结果；当前失败原因已定位为 `run_train()` 中 `build_window_loader(... session_to_idx=...)` 的漏传，而不是训练发散或 session conditioning 思路本身无效。

## 后续安排

1. 保留 `IBL-MtM combined_e300_aligned` 当前结果，后续仅补 compare / summary，不重跑
2. 修复 `faithful_neuroformer.py` 中 `session_to_idx` 的漏传
3. 单独重启 `Neuroformer canonical 500/250 + session conditioning`
4. 用 `best ckpt` 做 formal `valid/test × rollout/true_past`
5. 按 `1.8.3` 规范补齐图表、对比和结论

## 执行进展（2026-03-21 22:10 CST）

- 已完成 runner / 协议更新的中间提交并推送：2761a8b 补充1.8 benchmark后续runner与诊断指标
- 后台正式 run 已启动：screen 会话 phase1_benchmark_20260321
- 主日志路径：results/logs/phase1-autoregressive-1.8-benchmark_model/20260321_benchmark_ibl_e300_neuroformer_session_conditioning/main.log
- 当前阶段：IBL-MtM combined_e300_aligned
- 已确认第一轮训练正常进入 epoch 1，当前可见指标：train_loss=0.8346, valid_fp_bps=-2.9818, valid_poisson_nll=0.5902
- 当前输出目录：results/logs/phase1-autoregressive-1.8-benchmark_model/20260321_benchmark_ibl_e300_neuroformer_session_conditioning/ibl_mtm_combined_e300_aligned
- best_model.pt 与 last_model.pt 已开始更新，说明训练未卡死

## 当前状态（2026-03-22 00:15 CST）

- `IBL-MtM combined_e300_aligned` 已完整完成并写出正式结果
  - `best_epoch = 282`
  - formal valid `fp-bps = 0.1938`
  - formal test `fp-bps = 0.1938`
  - formal test `poisson_nll = 0.3046`
  - formal test `predicted_to_true_event_ratio_mean = 11.1438`
  - 结果文件：`results/logs/phase1-autoregressive-1.8-benchmark_model/20260321_benchmark_ibl_e300_neuroformer_session_conditioning/ibl_mtm_combined_e300_aligned/results.json`
- `Neuroformer canonical 500/250 + session conditioning` 首次正式 run 已中止
  - 原因：`run_train()` 中 `train_loader / valid_loader` 调用 `build_window_loader()` 时漏传 `session_to_idx`
  - 错误位置：`neural-benchmark/faithful_neuroformer.py`
  - 错误性质：runner wiring bug，不是模型训练稳定性问题
- 下一步：修复该漏传后，仅重启 Neuroformer 段，不重跑已经完成的 IBL-MtM e300

## Neuroformer 重启状态（2026-03-22 06:18 CST）

- 已修复 `faithful_neuroformer.py` 中 `run_train()` 对 `session_to_idx` 的漏传
- 已新增单独重启脚本：`scripts/phase1-autoregressive-1.8-benchmark_model/20260321_benchmark_ibl_e300_neuroformer_session_conditioning/rerun_neuroformer_only.sh`
- 已用新脚本启动后台重跑：screen 会话 `phase1_neuroformer_20260322`
- 当前运行命令：`faithful_neuroformer.py --mode train --obs-window 0.5 --pred-window 0.25 --epochs 50 ... --session-embedding-scale 1.0`
- 当前状态：训练进程存活，CPU 持续高占用；这次已确认不是入口参数缺失导致的即时退出
- 当前主日志仍沿用旧文件 `neuroformer_rerun.log`，其中最早一行保留了上一次错误输出；后续以当前进程存活、checkpoint 更新时间和正式结果文件为准


## Neuroformer 单验证重跑状态（2026-03-22 17:12 CST）

- 背景：在对比 `session conditioning` 重跑与旧 canonical run 的墙钟速度后，发现当前训练期较旧版明显变慢；为隔离“训练期双验证 + 新增诊断”对速度的影响，先把训练期监控回退到旧口径再重跑。
- 本次代码调整：
  - `run_train()` 每个 epoch 只保留一次 `valid rollout`，不再在训练期额外执行 `valid true_past`
  - 训练期 `history` / stdout 回退到旧字段：`valid_fp_bps + valid_r2`
  - `evaluate_faithful_neuroformer_loader()` 新增 `include_diagnostics` 开关；训练期设为 `False`，跳过 `predicted_to_true_event_ratio / per_session_metrics / max_generate_steps_hit_rate / eos_termination_rate`
  - formal eval 与 `session conditioning` 本身保持不变
- 归档情况：
  - 原慢速双验证 run 目录已备份到：`results/logs/phase1-autoregressive-1.8-benchmark_model/20260321_benchmark_ibl_e300_neuroformer_session_conditioning/neuroformer_250ms_session_conditioning_e50_dualval_backup_20260322_171136`
  - 原日志已备份到：`results/logs/phase1-autoregressive-1.8-benchmark_model/20260321_benchmark_ibl_e300_neuroformer_session_conditioning/neuroformer_rerun_dualval_backup_20260322_171136.log`
- 新重跑入口：
  - screen：`phase1_neuroformer_20260322_singleval`
  - 日志：`results/logs/phase1-autoregressive-1.8-benchmark_model/20260321_benchmark_ibl_e300_neuroformer_session_conditioning/neuroformer_rerun_singleval.log`
  - 输出目录：`results/logs/phase1-autoregressive-1.8-benchmark_model/20260321_benchmark_ibl_e300_neuroformer_session_conditioning/neuroformer_250ms_session_conditioning_e50`
- 启动命令：
  - `bash scripts/phase1-autoregressive-1.8-benchmark_model/20260321_benchmark_ibl_e300_neuroformer_session_conditioning/rerun_neuroformer_only.sh`
- 当前早期观测（2026-03-22 17:23 CST）：
  - 新单验证 run 已运行约 `11 min`
  - 训练进程存活，CPU 持续高占用
  - 当前尚未写出首个 `last_model.pt / best_model.pt`
  - 同卡仍有另一个 decoder screening 训练在跑，但当前 GPU `pmon` 显示 Neuroformer 占用约 `54%` SM，高于先前慢速 run 下午观测到的约 `27%` SM
- 当前判断：
  - 训练期双验证与新增诊断至少会带来额外开销，本次重跑用于验证它们是否是主要 slowdown 来源
  - 但本次重跑仍存在 GPU 共享干扰，因此即便速度改善，也只能说明“训练期配置回退后节奏变轻”，不能单独把全部提速归因到代码改动

## 2026-03-23 14:34 CST - Neuroformer 单验证重跑已完成，但当前 session conditioning 实际未生效

- 完成状态：`phase1_neuroformer_20260322_singleval` 已结束，当前无存活 screen / train 进程；输出目录已写出 `best_model.pt / last_model.pt / results.json / training curves`。
- 关键产物时间：
  - `best_model.pt`：`2026-03-23 03:01 CST`
  - `last_model.pt`：`2026-03-23 05:21 CST`
  - `results.json`：`2026-03-23 05:47 CST`
- 本轮最终结果：
  - best epoch：`39`
  - best valid rollout `fp-bps = -7.9134`
  - formal valid rollout / true_past `fp-bps = -7.9134 / -8.6397`
  - test rollout / true_past `fp-bps = -7.9389 / -8.6584`
  - test rollout / true_past `R^2 = -0.5664 / -2.5680`
  - test rollout / true_past `predicted_to_true_event_ratio_mean = 0.8786 / 0.9719`
- 与旧 canonical run 的最终速度对比：
  - 当前单验证重跑训练墙钟约 `874.4 s/epoch`
  - 旧 canonical 训练墙钟约 `871.8 s/epoch`
  - 比值约 `1.0029x`，训练主循环速度基本回到 canonical 同一量级
  - 从启动到 `results.json` 落盘的整条 pipeline 用时比 canonical 约慢 `0.6%`，已不支持“这次 rerun 仍显著慢很多”的结论
- 新发现的代码问题：当前 `session conditioning` 实际没有注入模型输入，因此这轮结果不能当作有效的 session-conditioning 实验结果。
  - 数据侧把 `session_idx` 放在 batch 顶层：`session_idx` 由 dataset 返回，并在 `collate_neuroformer_batch()` 中保留为顶层字段。
  - session wrapper 检查的是 `if "session_idx" in x`，但训练调用是 `model(batch["x"], batch["y"])`。
  - 因此传入模型的 `x` 中并没有 `session_idx`，`session_emb` 分支不会触发；这轮实跑更接近“去掉双验证/新增监控后的 canonical rerun”，而不是“显式 session-conditioning 已生效”的对照实验。
- 当前结论：
  - 这轮单验证重跑已经完成了它原本最重要的用途：证明训练期双验证和额外诊断才是此前大幅 slowdown 的主要来源，回退后整体速度已基本恢复到 canonical。
  - 但由于 `session_idx` 没有真正喂进模型，这轮结果不能用于回答“session conditioning 是否带来收益”。
  - 若要继续评估 session conditioning，需要修正 `session_idx` 的传递路径后，仅重跑 Neuroformer 段。

