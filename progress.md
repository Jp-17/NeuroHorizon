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

---

## 2026-02-28-12h

### 任务：优化 dataset.md 数据集规划文档

**完成时间**：2026-02-28-12h

**完成内容**：

对 `cc_core_files/dataset.md` 进行了多轮系统性优化，主要改动如下：

1. **修正 POYO/IBL 错误说明**（2.2节）
   - 原文误称"POYO/POYO+ 论文本身在 IBL 上有过验证"，实为错误
   - 已更正：POYO-1（NeurIPS 2023）在猕猴运动皮层数据上验证；POYO+（ICLR 2025）在 Allen Brain Observatory 钙成像数据上验证；**两者均未使用 IBL**
   - NeuroHorizon 在 IBL 上需自行建立 baseline，可与 NDT3、NEDS 等对比

2. **按执行阶段重组选型策略**（第3节，核心改动）
   - 原来按数据集划分（Brainsets→IBL→Allen），改为按项目执行进度划分四个阶段：
     - 阶段一：自回归改造验证 + 长时程生成验证（Brainsets 原生）
     - 阶段二：跨 Session 测试（Brainsets 必做，IBL 可选扩展）
     - 阶段三：Data Scaling + 下游任务泛化（Brainsets 必做，IBL 可选扩展）
     - 阶段四：多模态引入（Allen Neuropixels）
   - IBL 从"阶段二主力"改为"阶段二/三可选扩展"

3. **IDEncoder 实现与验证从阶段一移至阶段二**
   - 阶段一聚焦于自回归改造本身（causal mask、损失收敛、预测窗口梯度测试）
   - 阶段二新增首项任务：IDEncoder 基础实现与验证（实现 id_encoder.py，替换 InfiniteVocabEmbedding）

4. **新增预测窗口梯度扩展策略**（4.4节）
   - 250ms → 500ms → 1s（视数据和结果灵活调整），每步说明数据支撑和决策逻辑

5. **新增 Session 动态扩增策略**（4.5节）
   - Brainsets：5→10→20→40→70+；IBL：10-20（调试）→30→50→100→200→459

6. **优化第4节结构**
   - 4.1 聚焦阶段一（任务转换、窗口设计），去除跨 session 划分内容
   - 4.2 先介绍 Brainsets 跨 session 使用（必做基础），再介绍 IBL 可选扩展，明确启动时机

7. **扩充参考对比模型列表**（第6节）
   - 新增：POYO、POYO+、SPINT、Neuroformer、NDT1/NDT2（均附说明与 NeuroHorizon 的对比关系）
   - 保留：NDT3、NEDS（IBL 上有公开结果，阶段二/三直接对比目标）

**执行结果**：
- dataset.md 共 510 行，所有改动已提交（3 次 git commit，均已 push）
- 文档与当前四阶段执行计划完全对齐

**遇到的问题**：
- SSH heredoc 传递含方括号的字符串时 zsh 做了 glob 展开，导致 Python 字符串匹配失败；解决：将脚本写入本地文件再 scp 上传执行
- heredoc 写文件时"载"字的 UTF-8 编码被截断为 3 个替换字符（U+FFFD）；解决：字节级定位并替换修复

**对应 plan.md 任务**：属于"项目计划优化阶段"的文档修缮工作


---

## 2026-02-28-18h

### 任务：重构项目文档体系，建立可执行计划规范

**完成时间**：2026-02-28-18h

**完成内容**：

对项目核心文档进行了全面重构，建立了清晰的文档分工体系：

**1. 新建并重写 `cc_core_files/proposal_review.md`（735 行）**
- 定位：proposal.md 的技术补充，面向 plan.md 各阶段执行，提供代码级改造指南
- 整合来源：原 plan.md 第1-4节（项目目标/POYO差异/合理性/架构）+ 原 plan.md 附录A/B（文件清单/风险）+ code_research.md + 三份审查报告（code-research-review / plan-review / proposal-review）
- 按十大章节组织：架构速览 → POYO接口参考 → Phase 0~4 各节执行参考（含代码级方案、设计隐患、验收标准）→ 关键文件清单 → 风险汇总 → 合理性评估
- 关键技术内容：POYO-MP heads 勘误（cross=2/self=8）、GEGLU激活说明、causal mask修改方案、解码器信息瓶颈分析（4种方案对比）、IDEncoder vs InfiniteVocabEmbedding替换注意事项、优化器分组策略（SparseLamb vs AdamW）、DINOv2 灰度图处理方案

**2. 重写 `cc_core_files/plan.md`（367 行）**
- 以阶段化可执行计划为主体，Phase 0～5 结构化任务列表
- 每个 Phase 标题增加"执行参考"指针，指向 proposal_review.md 对应节
- 移除附录A/B（已迁移至 proposal_review.md）
- 0.3.1 验收标准从"R² > 0"提升为"R² > 0.3"
- 1.1.5 NeuroHorizon 模型实现拆分为 2.5a/b/c 三步（骨架→Decoder→tokenize）
- Plan 0.2 新增：data目录已有数据核查 + Brainsets 数据深度探索分析任务（0.2.3，含脚本/结果管理规范）

**3. 更新 `CLAUDE.md`（182 行）**
- 新增 proposal_review.md 到项目简介和文档导航表
- 当前状态从"项目计划优化阶段"更新为"Phase 0 执行阶段"
- 任务执行规范"任务开始前"新增第2步：读取 proposal_review.md 对应章节
- Plan 任务执行规范"执行前"新增第一步：读取 proposal_review.md 对应 Phase 执行参考

**执行结果**：
- 所有文件已提交并推送到 GitHub（3 次 commit）
- 文档体系分工明确：proposal.md（What/Why）+ proposal_review.md（How）+ plan.md（When/状态）+ dataset.md（数据）

**遇到的问题**：
- 大文件远程写入：使用"本地写 /tmp → scp → SSH 移动"三步法避免 heredoc 的 shell 元字符问题

**对应 plan.md 任务**：属于"项目计划优化阶段"的文档重构工作，为正式执行 Phase 0 做准备


---

## 2026-02-28-16h

### 任务：Phase 0.1 环境验证与代码理解 + 论文精读 + 代码改造建议

**完成时间**：2026-02-28-16h

**完成内容**：

完成 plan.md 中 Phase 0 的 0.1 全部三个子任务：

1. **0.1.1 POYO conda 环境验证**
   - 确认 `poyo` conda 环境可用（PyTorch 2.10.0+cu128, RTX 4090 D, CUDA BF16 支持）
   - 验证所有核心依赖：wandb/hydra/einops/h5py/scipy/sklearn/brainsets 全部就绪
   - 梳理完整代码模块依赖关系图（spike tokenization → unit embedding → Perceiver encoder → processing layers → readout）
   - 精读关键代码：rotary_attention.py（attention mask 机制）、infinite_vocab_embedding.py（tokenizer/vocab 管理）、poyo_plus.py（完整前向传播与 tokenize 接口）

2. **0.1.2 SPINT + Neuroformer 论文精读**
   - SPINT（IDEncoder）：网络结构为"1层cross-attn + 2×三层FC"，输入统计特征（firing rate + variance + spike counts），Dynamic Channel Dropout 是跨 session 鲁棒性的关键；若 33d MLP 方案效果不佳可借鉴其 cross-attn 结构
   - Neuroformer：三阶段处理（对比对齐→跨模态融合→因果解码）��逐 spike 自回归（vs NeuroHorizon 的 bin-level 预测），需对比学习预训练（vs NeuroHorizon 直接用 DINOv2 更简洁）

3. **0.1.3 后续阶段代码改造建议**（基于 0.1.1 + 0.1.2 产出）
   - Phase 1 关键发现：`rotary_attn_pytorch_func` 和 `rotary_attn_xformers_func` 均只处理 1D kv-padding mask，causal mask（2D 上三角）需扩展 mask reshape 逻辑；两个后端（pytorch + xformers）需同时修改
   - Phase 2 关键发现：`InfiniteVocabEmbedding` 含 tokenizer/detokenizer/vocab/state_dict hooks，不能直接删除，IDEncoder 需以 flag 方式作为替代路径而非直接替换
   - 详细建议文档：`cc_todo/phase0-env-baseline/20260228-phase0-env-code-understanding.md`

**附：本次还完成了**：
- plan.md 修改：0.1.3 内容改为"基于0.1.1和0.1.2的执行结果给予对于后续阶段要修改代码的建议"
- plan.md 乱码修复：3.2.2 行 "记录迁移增益" 中"益"字的 UTF-8 损坏（3个U+FFFD替换字符）已修复

**执行结果**：
- 三个 0.1 任务均已在 plan.md 中标记 [x]
- 任务记录文件写入 `cc_todo/phase0-env-baseline/20260228-phase0-env-code-understanding.md`

**遇到的问题**：
- SSH heredoc 传递含反引号的字符串时 bash 做了命令替换，导致 Python 字符串匹配失败；解决：将脚本写入本地文件再 scp 上传执行（与前次 dataset.md 修改问题相同，已成为标准操作流程）

**对应 plan.md 任务**：Phase 0 → 0.1.1、0.1.2、0.1.3
