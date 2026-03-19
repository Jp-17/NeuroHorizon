# 20260318 Benchmark Faithful Audit Task Log

> 总入口：`cc_todo/1.8-benchmark_model/benchmark_index.md`
> 本文档定位：faithful 审计执行流水记录

# 2026-03-18 | Phase 1.8 faithful 审计补充与文档收口

## 对应计划

- cc_core_files/plan.md -> 1.8.3
- 审查文档：cc_todo/1.8-benchmark_model/20260318_benchmark_faithful_audit_detail_codex.md

## 本次任务目标

1. 从代码层严格复核 1.8 当前 faithful 线是否如文档所述推进。
2. 对照 1.3.7 的统一标准，明确当前 benchmark 线哪些已经统一、哪些只是语义对齐、哪些仍不能混写。
3. 把 plan.md、results.md、20260312_benchmark_main_task_log.md 中仍残留的旧 benchmark 强结论收口到 legacy internal reference + faithful 250ms gate 未完成。

## 本次检查的核心代码与文档

- neural_benchmark/adapters/*.py
- neural_benchmark/benchmark_train.py
- neural-benchmark/repro_protocol.py
- neural-benchmark/faithful_ndt2.py
- neural-benchmark/faithful_ibl_mtm.py
- neural-benchmark/faithful_neuroformer.py
- cc_core_files/plan.md
- cc_core_files/results.md
- cc_todo/1.8-benchmark_model/20260312_benchmark_main_task_log.md

## 本次完成内容

1. 新增 detailed review：
   - cc_todo/1.8-benchmark_model/20260318_benchmark_faithful_audit_detail_codex.md

2. 文档口径收口：
   - cc_core_files/plan.md
   - cc_core_files/results.md
   - cc_todo/1.8-benchmark_model/20260312_benchmark_main_task_log.md

3. 本次收口的核心判断：
   - legacy simplified baselines 只保留为内部参考，不再视为正式 benchmark
   - faithful 线已经打通到上游核心，但当前仍处于 250ms gate
   - NDT2 暂停扩展，IBL-MtM 继续 short formal run，Neuroformer 先解 runtime blocker

## 关键结论摘要

- repro_protocol.py 已经把 faithful valid/test continuous 统一到了 deterministic canonical windows，语义上接近 1.3.7 的 continuous eval 标准。
- faithful_ndt2.py、faithful_ibl_mtm.py、faithful_neuroformer.py 均不再是旧 wrapper，它们已经直接接了上游核心模型或 tokenization / training path。
- 但 faithful 线并没有、也不应该被写成与 NeuroHorizon 训练和评估完全同构；当前正确说法只能是评估协议尽量统一，训练语义尽量保留上游。
- IBL-MtM 与 Neuroformer 理论上都可以在统一接口下做 forward prediction 风格评估；当前问题分别是：
  - IBL-MtM：metadata / training objective mismatch 仍强
  - Neuroformer：formal dual-mode held-out generation runtime 过高

## 后续建议

1. IBL-MtM：补 250ms short formal run，看结果是否向零靠拢。
2. Neuroformer：优先解决 250ms formal dual-mode eval runtime blocker。
3. NDT2：保留当前结论，不再优先扩展。
4. 任一模型未通过 250ms gate，不进入 500ms / 1000ms faithful 扩展。
