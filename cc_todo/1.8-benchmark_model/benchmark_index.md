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
- 可视化索引、当前结论、后续安排

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
- `NDT2` 当前只保留现状记录，不继续扩展

## 当前状态（2026-03-19）

- `IBL-MtM combined_e50_aligned` 已完成，test `fp-bps = 0.1345`，相较 `combined_e10` 已从 near-zero 提升为明确正值
- `Neuroformer canonical 500/250 e50 aligned` 正在执行，当前仍处于训练阶段
- `Neuroformer 150/50 reference` 尚未开始，等待 canonical run 完成后继续
