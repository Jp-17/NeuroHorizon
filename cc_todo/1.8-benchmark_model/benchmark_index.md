# 1.8 Benchmark Model Index

> **定位**：`1.8` benchmark 文档的唯一总入口。
>
> - `plan.md` 中 `1.8.3` 的规范、任务入口和当前 benchmark 主线，统一以本文档为准
> - legacy / protocol-fix / faithful 结果都在这里做索引，不再分散引用旧 review 路径

## 当前目标

在 `Perich-Miller` forward prediction benchmark 上持续维护 faithful benchmark model 实现，统一遵守 `1.3.7` 的主数据/指标标准，并把每轮 benchmark 实现优化的任务、实验、可视化和结论固定收口到 `cc_todo/1.8-benchmark_model/`。

## 文档索引

- 主任务记录：`cc_todo/1.8-benchmark_model/20260312_benchmark_main_task_log.md`
- legacy 审计：`cc_todo/1.8-benchmark_model/20260316_benchmark_legacy_audit_codex.md`
- faithful 审计总文档：`cc_todo/1.8-benchmark_model/20260318_benchmark_faithful_audit_detail_codex.md`
- faithful 审计执行记录：`cc_todo/1.8-benchmark_model/20260318_benchmark_faithful_audit_task_log.md`
- 当前 aligned 长跑记录：`cc_todo/1.8-benchmark_model/20260319_benchmark_aligned_runs.md`
- 当前后续优化任务：`cc_todo/1.8-benchmark_model/20260321_benchmark_ibl_e300_neuroformer_session_conditioning.md`

## 记录规范

以后每次新的 benchmark 实现优化任务统一记录到：

- `cc_todo/1.8-benchmark_model/{date}_{content}.md`

每份任务记录必须至少包含：

- 想法描述、动机与目的、改动点、实现方案、涉及模块、想法摘要
- 详细实验配置（数据集、sessions、采样方式、obs/pred 窗口）
- 关键超参数（至少 `epoch / batch_size / lr / weight_decay`）
- 训练 loss、最佳 val `fp-bps`、test `fp-bps`、checkpoint 标识/时间
- 各条件指标结果（至少 `fp-bps / per-bin fp-bps`）
- 与 baseline 的对比
- 如模型支持多 inference 模式，必须同时记录 `rollout` 与 `teacher-forced / true_past`
- 每次训练和评估的脚本命令
- 可视化索引、当前结论、后续安排

## 当前协议约束

- benchmark continuous 主协议继续遵守 `1.3.7`：
  - train split 语义对齐 `RandomFixedWindowSampler`
  - valid/test 语义对齐 `SequentialFixedWindowSampler`
  - faithful runner 若不直接调用上述 sampler，也必须在任务记录中说明语义等价性
- `best` checkpoint 默认按 `valid fp-bps` 选择，并在训练结束后用该 checkpoint 重新计算正式 `valid / test` continuous 指标
- `IBL-MtM` 和 `Neuroformer` 当前 benchmark 主流程不要求 `test trial-aligned`
- `Neuroformer` 当前默认按 `valid rollout fp-bps` 选择 `best_model.pt`，`teacher-forced / true_past` 只作诊断和补充报告
- `Neuroformer` 训练期默认同时监控 `rollout` 与 `true_past`，但 best ckpt 仍按 `valid rollout fp-bps` 选择
- `Neuroformer` 训练期不再把 `R²` 作为主监控项；优先跟踪 `fp-bps / teacher-forced loss / rollout-true_past gap`
- `Neuroformer` 当前保留 `session-constrained decoding` 作为解码安全约束，但这不等于显式 session conditioning；若任务记录写到 session 改造，需明确二者区别

## 当前建议脚本入口

- `neural-benchmark/faithful_ibl_mtm.py`
- `neural-benchmark/faithful_neuroformer.py`
- `neural-benchmark/plot_benchmark_history.py`
- `neural-benchmark/run_faithful_1p8_aligned.sh`
- `scripts/phase1-autoregressive-1.8-benchmark_model/20260321_benchmark_ibl_e300_neuroformer_session_conditioning/run_benchmark.sh`
- `scripts/phase1-autoregressive-1.8-benchmark_model/20260321_benchmark_ibl_e300_neuroformer_session_conditioning/rerun_neuroformer_only.sh`

## 路径规范

- 任务记录：`cc_todo/1.8-benchmark_model/{date}_{content}.md`
- 脚本：`scripts/phase1-autoregressive-1.8-benchmark_model/{date}_{content}/`
- 日志：`results/logs/phase1-autoregressive-1.8-benchmark_model/{date}_{content}/`
- 可视化：`results/figures/phase1-autoregressive-1.8-benchmark_model/{date}_{content}/`

说明：legacy / protocol-fix / faithful 的历史结果目录不重命名，只在新任务记录里索引引用。

## 当前维护中的任务

- `IBL-MtM combined aligned`
- `Neuroformer canonical 500/250`
- `Neuroformer 150/50 reference`
- `IBL-MtM combined_e300_aligned`
- `Neuroformer canonical 500/250 + session conditioning`
- `NDT2` 当前只保留现状记录，不继续扩展

## 当前状态（2026-03-21 04:10 CST）

- 当前这一轮 `aligned` benchmark 长跑已全部完成，相关任务记录见：`cc_todo/1.8-benchmark_model/20260319_benchmark_aligned_runs.md`
- `IBL-MtM combined_e50_aligned` 已完成，test `fp-bps = 0.1345`，相较 `combined_e10` 的 `-0.0017` 提升 `+0.1361`
- `Neuroformer canonical 500/250 e50 aligned` 已完成训练与 formal eval
  - best epoch = `42`
  - formal valid/test rollout `fp-bps = -7.9923 / -8.0350`
  - formal valid/test true_past `fp-bps = -8.5479 / -8.5701`
  - `skip_trial_eval = true`，formal eval 已按 `1.8.3` 新协议不再输出 `test trial-aligned`
  - 训练曲线和配置时间轴图已落盘：`train_loss_curve.png / valid_fp_bps_curve.png / valid_r2_curve.png / lr_curve.png / training_config_timeline.png`
- `Neuroformer 150/50 reference e50 aligned` 已完成训练与 formal eval
  - best epoch = `26`
  - formal valid/test rollout `fp-bps = -6.8698 / -6.8777`
  - formal valid/test true_past `fp-bps = -8.3274 / -8.3740`
  - `skip_trial_eval = true`，formal eval 已按 `1.8.3` 新协议执行
  - 训练曲线和配置时间轴图已落盘：`train_loss_curve.png / valid_fp_bps_curve.png / valid_r2_curve.png / lr_curve.png / training_config_timeline.png`
- `Neuroformer 150/50` 相比 canonical `500/250` 的 test rollout `fp-bps` 有所改善（`-8.0350 -> -6.8777`），但仍显著为负，尚不能支持其作为可竞争 benchmark
- benchmark 级汇总图已补充到：
  - `results/figures/phase1-autoregressive-1.8-benchmark_model/20260319_benchmark_aligned_runs/aligned_benchmark_summary.png`
  - `results/figures/phase1-autoregressive-1.8-benchmark_model/20260319_benchmark_aligned_runs/aligned_benchmark_summary.md`
- 下一阶段默认执行：
  - `IBL-MtM combined_e300_aligned`
  - `Neuroformer canonical 500/250 + 显式 session conditioning`
  - `Neuroformer` 训练期同步监控 `rollout / true_past / teacher-forced loss`

## 当前状态补充（2026-03-22 00:15 CST）

- `IBL-MtM combined_e300_aligned` 已完成并达到当前 1.8 faithful benchmark 的最好结果：
  - best epoch = `282`
  - formal valid `fp-bps = 0.1938`
  - formal test `fp-bps = 0.1938`
  - 结果目录：`results/logs/phase1-autoregressive-1.8-benchmark_model/20260321_benchmark_ibl_e300_neuroformer_session_conditioning/ibl_mtm_combined_e300_aligned/`
- `IBL-MtM` 继续训练仍有效，但新增诊断显示 `predicted_to_true_event_ratio_mean` 仍约 `11x`，后续需重点查看输出尺度/校准问题
- `Neuroformer canonical 500/250 + session conditioning` 首次正式 run 未完成
  - 失败原因已定位为 runner wiring bug：`run_train()` 中 `build_window_loader()` 漏传 `session_to_idx`
  - 该问题与模型收敛性无关，修复后应单独重启 Neuroformer 段
- 当前建议：
  - 保留 `IBL-MtM e300` 结果并继续回填总结
  - 仅重启 `Neuroformer canonical 500/250 + session conditioning`
