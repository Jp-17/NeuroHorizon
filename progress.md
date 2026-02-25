# NeuroHorizon 项目进展记录

> 记录每次任务的执行情况，包含时间、完成内容、结果、遇到的问题及解决方法。
> 不在此处记录待开展工作（参阅 cc_core_files/plan.md）。

---

## 2026-02-25-17h

### 任务：核心文档审查分析（code_research / proposal / plan）

**完成时间**：2026-02-25-17h

**完成内容**：
对 cc_core_files/ 下三份核心文档进行了深度审查，逐段对照 POYO 代码库实际实现验证准确性，分析逻辑一致性和技术可行性。产出三份审查报告：

1. `cc_todo/20260225-review/20260225-code-research-review.md`
   - 验证了 code_research.md 中的架构描述、模型参数、数据管线描述
   - 发现 POYO-MP heads 数据错误（应为 cross=2/self=8，非 4/4）
   - 识别了 CaPOYO 模型分析缺失、GEGLU 激活未提及、rotate_value 差异未说明等遗漏

2. `cc_todo/20260225-review/20260225-proposal-review.md`
   - 重点分析了 Perceiver "可选"定位不当、SwiGLU vs GEGLU 不一致、decoder 信息瓶颈设计问题
   - 识别了与 SPINT IDEncoder 差异不清晰、优化器选择矛盾、Jia Lab 数据状态未更新等问题
   - 提出了 decoder 替代方案（per-neuron query、neuron interaction layer 等）

3. `cc_todo/20260225-review/20260225-plan-review.md`
   - 分析了任务依赖关系缺失、Phase 2 复杂度低估、causal mask 修改未列为显式步骤等问题
   - 补充了缺失的风险项（IDEncoder 特征维度不足、Poisson NLL 数值稳定性等）
   - 提出了修正后的 Phase 2 任务清单（7 步拆分）

**执行结果**：
- 三份审查文档已写入 cc_todo/20260225-review/
- 发现高优先级问题共 10 项，中优先级问题共 15 项
- 为后续文档修缮提供了具体的修正清单

**遇到的问题**：无

**对应 plan.md 任务**：属于"项目计划优化阶段"的文档审查工作

---

## 2026-02-25

### 任务：初始化项目工作规范文档

**完成时间**：2026-02-25

**完成内容**：
1. 创建 `CLAUDE.md`：项目工作指南，包含：
   - 项目简介（指向 cc_core_files/ 下各文档）
   - 当前项目状态（项目计划优化阶段）
   - 目录结构规范
   - 任务执行规范（任务前/中/后的操作细则）
   - 重要警告（cc_todo/20260221-cc-1st/ 已废弃，禁止参考）
   - Git 配置说明
2. 创建 `progress.md`（本文件）：项目进展记录文档
3. 创建 `cc_core_files/scripts.md`：脚本管理记录（初始化空模板）
4. 创建 `cc_core_files/data.md`：数据集管理记录（初始化空模板）
5. 创建 `cc_core_files/results.md`：实验结果记录（初始化空模板）

**执行结果**：
- 所有文件成功创建
- 项目工作规范已就绪，后续任务按 CLAUDE.md 规范执行

**遇到的问题**：无

**对应 plan.md 任务**：不直接对应 plan.md ��的代码任务，属于项目管理文档建设

---
