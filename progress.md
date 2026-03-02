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
- 按十大章节组织：架构速览 → POYO接口参考 → Phase 0–4 各节执行参考（含代码级方案、设计隐患、验收标准）→ 关键文件清单 → 风险汇总 → 合理性评估
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

---

## 2026-02-28-20h

### 任务：执行 Plan 0.2 数据准备与探索（0.2.1 / 0.2.2 / 0.2.3）

**完成时间**：2026-02-28-20h

**完成内容**：

**0.2.1 检查并下载 Brainsets 数据**
- 确认项目中无 Perich-Miller 数据（原有 IBL/Allen/NLB 数据来自废弃任务）
- 通过 DANDI REST API 确认 DANDI:000688 共 111 sessions，总 12.3GB
- 编写 `scripts/data/perich_miller_pipeline.py`（brainsets pipeline 修改版，过滤至 10 sessions）
- 选取 4 C-CO + 3 J-CO + 3 M-CO（320MB），下载并处理为 brainsets HDF5 格式
- 更新 `cc_core_files/data.md` 和 `cc_core_files/scripts.md`

**0.2.2 数据加载验证**
- 通过 `temporaldata.Data.from_hdf5()` 验证单文件加载：结构完整，字段正确
- 通过 `torch_brain.data.Dataset` + `RandomFixedWindowSampler` 验证 pipeline：
  - 10 sessions 正常加载，6372 个训练窗口（1s each）
  - 单样本读取正常（约350 spikes/window，41 units）
  - PASSED

**0.2.3 数据深度探索分析**
- 编写 `scripts/analysis/explore_brainsets.py`，产出：
  - `results/figures/data_exploration/01_dataset_overview.png`（概览统计）
  - `results/figures/data_exploration/02_neural_statistics.png`（神经元统计 + 自回归可行性）
  - `results/figures/data_exploration/exploration_summary.json`（统计摘要 JSON）
- 关键发现（详见 `cc_todo/phase0-env-baseline/20260228-phase0-data-explore.md`）：
  - Hold period 均值 676ms（87% > 250ms）→ 250ms 输入窗口可行
  - Reach period 均值 1090ms（100% > 500ms，75% > 1000ms）→ 三级预测窗口均可行
  - Per-unit spike 稀疏度（20ms bin）：87.6% zero，均值 0.133；Poisson NLL 适用
  - 250ms → 12 AR steps；500ms → 25 steps；1000ms → 50 steps（需 scheduled sampling）
  - Phase 1 推荐起点：c_20131003 session，250ms 预测窗口

**执行结果**：
- Plan 0.2.1、0.2.2、0.2.3 全部完成，plan.md 对应 checkbox 已打勾
- 已 git commit + push

**遇到的问题**：
1. `.gitignore` 的 `data/` 规则导致 `scripts/data/` 被忽略 → 添加 `!scripts/data/` 例外规则
2. temporaldata lazy loading 需在 h5py 上下文内访问 → 数据访问放在 `with` 块内
3. torch_brain Dataset API 参数名变化（`interval_dict` → `sampling_intervals`）→ inspect 检查后修正
4. dandi 包未安装在 poyo 环境 → `pip install dandi==0.61.2 scikit-learn`

**对应 plan.md 任务**：Phase 0.2（0.2.1、0.2.2、0.2.3）

---

## 2026-02-28-19h

### 任务：Phase 0.3 POYO+ 基线复现（0.3.1 / 0.3.2 / 0.3.3）

**完成时间**：2026-02-28-19h

**完成内容**：

**0.3.1 在 Perich-Miller 数据上运行 POYO+ 行为解码，验证 R² > 0.3**
- 新建训练配置：`examples/poyo_plus/configs/train_baseline_10sessions.yaml`
  - 模型：dim=128, depth=12（约8M params）；BF16；500 epochs；batch_size=64
  - 数据集：已有的 10 sessions（`perich_miller_10sessions.yaml`）
- 训练历时约 85 分钟（RTX 4090 D，GPU 利用率 57%，显存 4GB/24GB）
- **最佳平均 R² = 0.8065**（epoch 429），最终 R² = 0.8046（epoch 499）
- 各 session R²：C animal 0.85-0.91，J animal 0.57-0.76，M animal 0.86-0.92
- **验收标准 R² > 0.3 大幅满足**

**0.3.2 分析 POYO encoder latent representation 质量**
- 编写 `scripts/analysis/analyze_latents.py`，hook `dec_atn` pre-hook 提取处理后 latent
- 在 val set 的 1313 个窗口上运行分析：
  - PCA：**PC1=53.8%**，PC2=7.4%（latent 空间结构高度有序）
  - Linear probe（5-fold CV Ridge）：**R²=0.286 ± 0.032**
- 产出：`results/figures/baseline/latent_pca.png`，`latent_linear_probe.json`

**0.3.3 基线性能报告**
- 所有结果记录至 `cc_core_files/results.md`（包含训练曲线、per-session R²、latent 分析）
- plan.md 对应任务全部标记完成

**执行结果**：
- Phase 0 全部任务完成（0.1.1-0.3.3），Phase 1 可正式开始
- 已 git commit + push

**遇到的问题**：
1. `torch.load` PyTorch 2.6 默认 `weights_only=True`，加载 Lightning checkpoint 报错 → 添加 `weights_only=False`
2. latent 分析初始 hook 目标为 `model.perceiver`（不存在）→ 修正为 `model.dec_atn` pre-hook
3. linear probe 维度不匹配：target_values 是 flat [B*T, 2] 而非 [B, T, 2] → repeat latent 匹配目标时序

**对应 plan.md 任务**：Phase 0 → 0.3.1、0.3.2、0.3.3


---

## 2026-02-28-22h

### 任务：补充代码理解文档（模型对比、SPINT 修正）+ 全面更新 IDEncoder 设计方案

**完成时间**：2026-02-28-22h

**完成内容**：

**1. 补充 0.1.1 三大模型对比分析**

在 `cc_todo/phase0-env-baseline/20260228-phase0-env-code-understanding.md` 中新增 POYO / POYOPlus / CaPOYO 三模型的详细对比，涵盖输入表示、readout 机制、tokenize 流程、encoder-processor-decoder 骨架等维度，并给出 NeuroHorizon 基底选择建议：基于 POYOPlus 改造（复用 encoder+processor，重写 decoder+tokenize）。

**2. 修正 0.1.2 SPINT IDEncoder 描述**

- 作者/年份更正：Wei et al., 2024 → Le et al., NeurIPS 2025
- 输入更正：~~手工统计特征（firing rate, ISI 等）~~ → 原始 binned spike counts（20ms bin，从 calibration trials 插值到固定长度）
- 架构更正：明确为 MLP₁ → mean pool → MLP₂，cross-attention 在 IDEncoder 之后用于解码

**3. 全面更新 NeuroHorizon IDEncoder 设计方案（跨 5 个文件 18 处修改）**

核心设计更新：
- **输入**：原始神经活动（非手工统计特征），两种 tokenization 方案：
  - 方案 A (Binned Timesteps, SPINT 风格)：作为 base 实现
  - 方案 B (Spike Event Tokenization, POYO 风格)：作为 NeuroHorizon 创新点之一
- **输出**：unit embedding (d_model 维)，替换 InfiniteVocabEmbedding（非 SPINT 的加法注入）
- **架构**：先参考 SPINT 的 feedforward 设计（MLP₁ → mean pool → MLP₂）

更新的文件：
- `cc_core_files/proposal_review.md`：第五节完全重写（8 处修改）
- `cc_core_files/plan.md`：Phase 2.1 任务重写 + 新增 2.1.2b 方案 B 任务（3 处修改）
- `cc_core_files/proposal.md`：SPINT 描述、创新点一、Unit Embedding 节等（5 处修改）
- `cc_core_files/dataset.md`：SPINT 参考描述（2 处修改）
- `cc_core_files/code_research.md`：IDEncoder 描述（1 处修改）

**执行结果**：
- 所有文件已提交并推送到 GitHub（3 次 commit）
- 项目文档体系中关于 IDEncoder 的描述已统一为最新设计方案

**遇到的问题**：
1. Python 脚本中三引号嵌套导致语法错误 → 改为将替换内容写入独立文件，脚本读取文件内容后替换
2. 首次脚本运行在 Change 2 失败时已修改了 content 变量但 `sys.exit(1)` 导致未写入文件 → 拆分为独立脚本分步执行
3. 远程文件中存在 UTF-8 替换字符（U+FFFD）导致字符串匹配失败 → 改用 index-based 替换

**对应 plan.md 任务**：不直接对应 plan.md 中的代码任务，属于项目文档优化工作（Phase 0 阶段）


---

## 2026-03-01-afternoon

### 任务：proposal.md 大幅重构 — 拆分 background.md + 重写方案结构

**完成时间**：2026-03-01

**完成内容**：

**1. 新建 `cc_core_files/background.md`（103 行）**

从 proposal.md 拆分研究背景内容，形成独立文档：
- §1 研究背景（1.1 现状、1.2 研究意义、1.3 研究动机）
- §2 构建 Spike Foundation Model 的核心挑战（原 proposal.md §2.2 全部 4 条挑战）
- §3 相关工作（NDT3 更正为"自回归生成"，对比表新增"多模态归因分析"列）

研究意义重写为三个维度：梯度无关跨 session 泛化、自回归预测、多模态可解释性。

**2. 重写 `cc_core_files/proposal.md`（899 行 → 458 行）**

新结构：
- 项目概述（一句话）
- §1 核心挑战：仅保留 3 条（移除"计算效率"，该条非创新点，移入 background.md）
- §2 问题定义：去除所有硬编码"T=1s"；新增第 4 条"多模态可解释性"（含 $\Delta_m$ 公式和条件分解）
- §3 研究创新点：创新点二修正（输出是"预测"非"历史"）；创新点四降级为附录补充
- §4 方法设计与创新模块实现（详细）：架构图、双窗口、causal decoder、per-neuron MLP head、bin query、Poisson NLL、IDEncoder、tokenization、多模态注入
- §5 数据集（简表，引用 dataset.md）
- §6 实验设计（简要，对齐 plan.md）
- §7 可能的风险（原§9 更名）
- §8 参考文献
- 附录A 符号表（更新新符号）
- 补充：Perceiver 序列压缩（原创新点四内容）

关键修正：
- NDT3 从"masked prediction"更正为"autoregressive generation"
- 所有"T=1s"硬编码改为灵活描述
- 移除振荡相关内容
- 创新点二输出从"历史重建"更正为"未来预测"

**3. 更新 CLAUDE.md**

在项目简介链接和文档导航表中添加 background.md 条目。

**执行结果**：
- Git commit: e169de9（3 files changed, +564, -899）
- 已 push 到 GitHub

**遇到的问题**：无

**对应 plan.md 任务**：不直接对应 plan.md 代码任务，属于项目文档体系优化


---

## 2026-03-01-evening

### 任务：修订 code_research.md — 删除 §8§9 + 采纳审查建议修正

**完成时间**：2026-03-01

**完成内容**：

对 `cc_core_files/code_research.md` 进行了系统性修订，包含两部分工作：

**1. 删除 §8 和 §9**
- §8「与 NeuroHorizon 改造的关键接口」：内容已在 proposal_review.md 中更详细覆盖，此处冗余
- §9「已验证的基线」：引用已废弃的 cc_todo/poyo_setup_log.md，基线结果已记录在 results.md

**2. 采纳审查建议修正（20+ 处，逐条验证后采纳）**

对照 `cc_todo/20260225-review/20260225-code-research-review.md` 的审查报告，逐条通过 SSH 在代码库中验证后采纳：

- **数据错误修正**：POYO-MP heads 从 4/4 修正为 cross=2/self=8，补充 atn_dropout=0.2
- **文件名修正**：`feed_forward.py` → `feedforward.py`（验证发现原文文件名有误）
- **架构遗漏补全**：
  - GEGLU 激活函数详细说明（feedforward.py 验证）
  - rotate_value 三层差异：enc=True, proc=True, dec=False（poyo_plus.py 验证）
  - TokenType 枚举实际只有 3 种（tokenizers.py 验证），嵌入表容量 4 中 index=3 预留
- **新增 §3.5 CaPOYO 模型分析**：input_value_map + unit_emb 半维 + 拼接机制（capoyo.py 验证）
- **项目结构补全**：position_embeddings.py、transforms 完整列表（6 文件）、nested.py 命名空间说明
- **采样器补全**：从 2 种扩展为 5 种完整表格（sampler.py 验证）
- **训练细节修正**：
  - readout 参数也标记 sparse=True（train.py 验证）
  - OneCycleLR div_factor=1 纠正"50% warmup"误导描述（train.py 验证）
  - 补充 MultiTaskDecodingStitchEvaluator
- **其他**：模态数 16→19（registry.py 验证）、prepare_for_multitask_readout 说明、varlen forward 方法、collation 规范

**唯一异议**：审查建议修正 forward pass 行号（200→166），选择直接移除精确行号改为引用方法名——行号随代码演进极易过时。

**执行结果**：
- code_research.md 从 约350 行调整为 327 行（删除 §8§9 约 -71 行，新增内容 +77 行）
- 已 git commit + push（commit 17da576）

**遇到的问题**：无

**对应 plan.md 任务**：不直接对应 plan.md 代码任务，属于项目文档修缮工作

---

## 2026-03-01-night

### 任务：proposal_review.md 全面重构

**完成时间**：2026-03-01

**完成内容**：

对 `cc_core_files/proposal_review.md` 进行全面重构（801 行 → 1326 行，79% rewrite），与最新 proposal.md（2026-03-01 重构版）和 code understanding 文档对齐。

**新文档结构**：
- 文档定位与结构说明（更新，明确与 proposal.md/plan.md/code_research.md/dataset.md/background.md 的分工）
- §一 Phase 0 概要（**新增**）：开发环境、POYO 代码要点、数据发现（hold/reach 时长、Poisson 适用性）、R²=0.807 基线锚点
- §二 Phase 1 执行参考（**大幅重写**）：整合 code understanding 文档深度分析，12 个子节覆盖双窗口、tokenize 改造、PoissonNLLLoss、causal mask、T-token decoder 设计（含信息瓶颈分析）、PerNeuronMLPHead、bin query、模型组装、行为解码双路径、训练脚本、实验设计、验收标准
- §三 Phase 2 执行参考（精简重组）：IDEncoder 方案 A/B、SPINT 注入对比、IVE 集成、优化器分组、跨 session 实验
- §四 Phase 3 执行参考（**新增**）：数据格式统一、分层采样、encoder 冻结策略、行为解码微调代码、迁移实验设计（3 种对比方案 + 少样本实验）
- §五 Phase 4 执行参考（**大幅扩展**）：Allen Dataset 类代码、DINOv2 离线提取、MultimodalInjection 模块代码、行为数据注入、多模态实验矩阵、Δ_m 条件分解实现
- §六 风险与应对汇总（重组，新增迁移/多模态风险）
- §七 模型规模配置参考（更新，增加显存估算）
- 附录 A 项目架构速览（从原§一移入，更新表格）
- 附录 B POYO 代码接口参考（从原§二移入，路径全部更新为 torch_brain/，新增 xformers/Dataset API 等勘误）
- 附录 C 关键文件清单（从原§八移入，更新路径，按 Phase 标注）

**关键删除/合并**：
- 原§十"合理性评估"全部删除或合并：
  - "SwiGLU vs GEGLU"：proposal.md 已统一 GEGLU → 删除
  - "信息瓶颈"：合并到 §二 decoder 设计
  - "非自回归基线"：合并到 §二实验设计
  - 数据可获取性/Allen 问题：已在 dataset.md 解决 → 删除
  - 变长输出/causal mask 兼容性：已在正文覆盖 → 删除
  - 模型规模配置：保留为独立 §七

**执行结果**：
- Git commit: cc9e016（1 file changed, +1326 insertions, -800 deletions, 79% rewrite）
- 已 push

**遇到的问题**：无

**对应 plan.md 任务**：不直接对应 plan.md 代码任务，属于项目文档体系优化（为 Phase 1 执行做准备）


---

## 2026-03-01-late

### 任务：在 proposal_review.md 和 plan.md 中新增执行通则

**完成时间**：2026-03-01

**完成内容**：

根据用户要求，在两份核心文档中新增"执行通则"小节，明确两条跨 Phase 的执行规范：

1. **proposal_review.md**：在"文档定位与结构说明"与"一、Phase 0 概要"之间插入"执行通则"节
   - 规则 1：显存不足处理——优先自行排查（batch size、梯度累积、混合精度等），确认需要更多资源时告知用户
   - 规则 2：效果不达标处理——优先在计划范围内调整；若穷尽手段仍不行，可质疑 proposal 方案并提出替代方案，但必须写入文档并提前获得用户同意

2. **plan.md**：在"总览/MVP"与"Phase 0"之间插入同样的"执行通则"节（措辞略简洁）

**执行结果**：
- 两个文件已修改，Git commit: 4aaaed4（2 files changed, +16 insertions）
- 已 push 到 GitHub

**遇到的问题**：
- SSH sed 命令插入多行时换行符未正确生效，导致内容被拼成单行；解决：改用 Python 脚本进行字符串替换

**对应 plan.md 任务**：不直接对应 plan.md 代码任务，属于项目执行规范补充


---

## 2026-03-02 | jp-video-1 | 跨文档交叉核查与对齐修复（6 文件 12+ 处修改）

**任务内容**：
对 NeuroHorizon 的 5 份核心规划文档（background.md, dataset.md, proposal.md, proposal_review.md, plan.md）+ CLAUDE.md 进行最终一致性核查，修复跨文档不对齐。

**执行过程**：

1. **CLAUDE.md**（1 处）：项目状态从"Phase 0 执行阶段"更新为"Phase 1 待执行"，附 Phase 0 关键结论摘要（R²=0.807 基线、数据支持确认、基底选择决策）

2. **plan.md**（9 处）：
   - 7 处章节引用修正：proposal_review.md 重构后章节编号变化（§一=Phase 0, §二=Phase 1, ..., §五=Phase 4），更新 Phase 0-4 全部执行参考指针和内部引用
   - 4.2.2 行为条件注入描述更新为"linear projection + rotary time embedding → 输入端拼接"
   - 4.2.3 图像条件注入描述补充"输入端拼接"方式

3. **dataset.md**（1 处）：SPINT 引文从"Liu et al., 2023"修正为"Le et al., NeurIPS 2025"

4. **background.md**（2 处）：对比表修正 — NeuroHorizon 计算效率"高"→"中"（新增模块增加计算量）；POYO+ 跨Session"有限"→"需微调"（更精确描述其 extend_vocab 机制）

5. **proposal.md**（1 处，最大改动）：§4.5 多模态条件注入完整重写 —
   - 新增两种注入方案对比表（Processing Layer 注入 vs 输入端拼接）
   - 明确采用方案 B（输入端拼接）及三点理由
   - 补充行为数据注入完整处理流程：Linear projection → rotary time embedding 时间对齐 → 输入端拼接
   - 保留视觉刺激注入描述，补充"输入端拼接"方式

6. **proposal_review.md**（3 处）：
   - §5.3 注入位置分析表新增"Processing Layer 注入"行
   - §5.4 行为数据注入展开为详细的 Rotary Time Embedding 时间对齐机制说明（4 点：统一时间坐标系、RoPE 自动编码、无需手动插值、拼接融合）
   - §六 风险表"自回归 50 步误差累积"行追加 coarse-to-fine 策略

**验证通过**：
- `grep 'SPINT' cc_core_files/` — 全部文件 SPINT 引文统一为"Le et al."
- `grep '第[三四五六七]节' plan.md` — 零匹配（旧章节编号已清除）
- `grep 'Phase 1 待执行' CLAUDE.md` — 项目状态已更新
- `grep '计算效率' background.md` — NeuroHorizon 列显示"中"

**遇到的问题**：Python 脚本中 LaTeX 反斜杠（\mathbf 等）在字符串匹配时转义不一致，导致首次 proposal.md 替换失败

**解决方法**：改用行号定位（grep -n 找到 §4.5 起止行）+ 逐行替换方式，避免 LaTeX 特殊字符匹配问题

---

## 2026-03-02 | jp-video-1 | Phase 1 核心模块实现 + 基础验证 + 预测窗口实验

**任务内容**：
执行 plan.md Phase 1（自回归改造验证 + 长时程生成验证），包括核心模块实现、训练验证、自回归推理验证和多窗口实验。

### Phase 1.1 核心模块实现（全部完成）

1. **1.1.1 PoissonNLLLoss** — `torch_brain/nn/loss.py`
   - 实现 `PoissonNLLLoss`：loss = exp(log_rate) - target * log_rate
   - log_rate clamp[-10, 10] 保证数值稳定性

2. **1.1.2 注册 spike_counts 模态** — `torch_brain/registry.py`
   - 新增 spike_counts modality（id=20, dim=1, CONTINUOUS, PoissonNLLLoss）

3. **1.1.3 Causal Mask 支持** — `torch_brain/nn/rotary_attention.py`
   - 新增 `create_causal_mask()` 函数
   - 修改 pytorch/xformers 后端支持 2D/3D/4D mask
   - 单元测试验证因果性

4. **1.1.4 AutoregressiveDecoder + PerNeuronMLPHead** — `torch_brain/nn/autoregressive_decoder.py`
   - T-token decoder: cross-attn → causal self-attn → FFN
   - PerNeuronMLPHead: concat(bin_repr, unit_emb) → MLP → log_rate
   - rotate_value=False for decoder

5. **1.1.5 NeuroHorizon 模型** — `torch_brain/models/neurohorizon.py`
   - 384 行完整模型：encoder + processor (复用 POYO) + AR decoder + per-neuron head
   - forward() (teacher forcing) + generate() (逐步自回归) + tokenize()
   - Bug fix: pad8→pad 修复 target_unit_index 维度不匹配问题

6. **1.1.6 训练脚本** — `examples/neurohorizon/train.py` + configs
   - Lightning training wrapper, PoissonNLL loss, R² + per-bin NLL metrics
   - Small 配置: dim=128, enc_depth=6, dec_depth=2, 4.2M params

### Phase 1.2 基础功能验证

**1.2.1 Teacher Forcing 训练（250ms, 300 epochs）**：
- 训练正常收敛，无 NaN/Inf
- 验证 R² 从 0.207 (ep10) 上升至 约0.26 (ep140)，val_loss 从 0.328→0.314
- Per-bin NLL: bins 0-10 约0.31 均匀，bin 11 约0.39 偏高（远期预测更难）
- mean_pred_rate ≈ mean_target_count（模型学到了合理的发放率）

**1.2.2 自回归推理验证**：
- TF vs AR 输出完全一致（max_diff=3e-6，数值精度级）
- Causal mask 验证通过：修改 bins 8-11 不影响 bins 0-7（diff=0.0）
- AR/TF R² ratio = 1.0000

### Phase 1.3 预测窗口实验（进行中）

- 250ms 训练完成中（~epoch 140/300, R²≈0.26）
- 已创建 500ms、1000ms 配置文件
- 待 250ms 完成后依次启动

**遇到的问题与解决**：
1. pad8 vs pad2d 维度不匹配 → 改用 pad/track_mask
2. PyTorch 2.6 weights_only=True 默认 → 添加 weights_only=False
3. InfiniteVocabEmbedding LazyModule → 先初始化 vocab 再计数参数
4. checkpoint 加载后 vocab 已初始化 → 跳过 initialize_vocab
