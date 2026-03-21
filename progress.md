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

**对应 plan.md 任务**：不直接对应 plan.md 中的代码任务，属于项目管理文档建设

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

---

## 2026-03-12-22h

### 任务：Phase 1.9 Structured Prediction Memory Decoder 实施与验证

**完成时间**：2026-03-12-22h

**完成内容**：
1. 从 `dev/20260312_refactor` 新建分支 `dev/20260312_prediction_memory_decoder`
2. 在 `cc_core_files/model.md` 中新增 1.9 架构设计小节，明确记录：
   - `shift-right` 的因果语义
   - `prediction memory` 的设计动机与结构
   - 旧 `feedback_method` / Query Augmentation 的逻辑和新定位
3. 实现 `PredictionMemoryEncoder`，并将 decoder 层顺序改为：
   - history cross-attn
   - prediction-memory cross-attn
   - causal self-attn
   - FFN
4. 在 `NeuroHorizon` 中新增 `decoder_variant='prediction_memory'`，保留旧 `query_aug` 路径作为 baseline/ablation
5. 修改训练与评估入口：
   - `examples/neurohorizon/train.py` 按 `requires_target_counts` 自动传入 `target_counts`
   - `scripts/analysis/neurohorizon/eval_phase1_v2.py` 支持 `--rollout`
6. 新增 1.9 文档、配置和脚本：
   - `cc_todo/phase1-autoregressive/1.9-module-optimization/20260312_prediction_memory_decoder.md`
   - `train_1p9_prediction_memory_{250ms,500ms,1000ms}.yaml`
   - `verify_prediction_memory.py`
   - `run_prediction_memory_experiments.sh`
   - `collect_prediction_memory_results.py`
7. 完成静态编译和功能验证，并跑通 250ms 的 1-epoch smoke run 与 rollout smoke eval

**执行结果**：
- 功能验证通过：
  - `PredictionMemoryEncoder` shape 正确
  - `shift-right` 生效：修改 bin0 target 不影响 bin0，仅影响未来 bins
  - `forward()` 与 `generate()` 不再等价
- 250ms smoke run 跑通：
  - `train_loss=0.424`
  - `val_loss=0.412`
  - `val/fp_bps=-0.823`
- rollout smoke eval 跑通：
  - `fp-bps=-0.8218`
  - `R2=0.0001`
- 正式 1.9 并行实验已后台启动：
  - 脚本：`scripts/phase1-autoregressive-1.9-module-optimization/20260312_prediction_memory_decoder/run_prediction_memory_experiments.sh`
  - 日志：`results/logs/phase1-autoregressive-1.9-module-optimization/20260312_prediction_memory_decoder/run_prediction_memory_experiments.log`
  - 监控：`scripts/phase1-autoregressive-1.9-module-optimization/20260312_prediction_memory_decoder/monitor_prediction_memory_progress.py`
  - 当前已确认 GPU 上存在对应 python 训练进程
  - 22:30 快照：
    - 250ms / 500ms / 1000ms 三个窗口均已进入正式训练
    - 显存占用约 7.7 GiB
    - 粗略 ETA：500ms 约 1h39m，250ms 约 2h14m，1000ms 约 2h11m

**遇到的问题**：
- `torch_brain/nn/autoregressive_decoder.py` 初版有一个 `ValueError` 字符串漏引号的语法错误；通过重新回传修复后，编译和验证通过
- 验证脚本初版打印 `requires_grad` tensor 时触发 warning；已改为 `detach()` 后再格式化输出

**对应 plan.md 任务**：Phase 1.9.2 模型优化迭代记录（Structured Prediction Memory Decoder）

---

## 2026-03-13-00h

### 任务：完成 1.9 Structured Prediction Memory Decoder 正式实验并归档结论

**完成时间**：2026-03-13-00h

**完成内容**：
1. 完成 `250ms / 500ms / 1000ms` 三组 300-epoch 正式训练与 rollout 评估
2. 自动汇总结果到 `prediction_memory_summary.json`
3. 更新 `cc_todo/phase1-autoregressive/1.9-module-optimization/results.tsv`
4. 刷新 1.9 趋势图 `optimization_progress.png/pdf`
5. 将 `Structured Prediction Memory Decoder` 在 `model.md` / `plan.md` / `cc_todo` / `results.md` 中正式标记为失败迭代并记录原因

**执行结果**：
- rollout fp-bps：
  - 250ms: `0.1486`
  - 500ms: `-0.0153`
  - 1000ms: `-0.2590`
- 对比 `baseline_v2`：
  - 250ms: `-0.0629`
  - 500ms: `-0.1897`
  - 1000ms: `-0.3907`
- teacher forcing 指标虽然很高，但 rollout 明显崩塌，尤其长窗口后段大量出现负 fp-bps

**遇到的问题**：
- `progress_status.md` 初次刷新停留在训练中快照，实验结束后已手动补刷为 finished 状态
- 结果表明当前 structured prediction memory 架构存在明显的 train/inference mismatch 和误差累积问题，不适合作为主线继续推进

**对应 plan.md 任务**：Phase 1.9.2 模型优化迭代记录（Structured Prediction Memory Decoder，已放弃）

---

## 2026-03-13-01h

### 任务：启动下一轮 1.9 修正迭代 Local Prediction Memory Decoder

**完成时间**：2026-03-13-01h

**完成内容**：
1. 从当前状态切新分支 `dev/20260313_local_prediction_memory`
2. 在 `model.md` 和 `plan.md` 中登记新的 1.9 迭代，核心思想是：
   - 保留 structured prediction memory
   - 但将其收缩为 local-only block，query 只访问紧邻上一步 memory
3. 在模型中新增 `decoder_variant='local_prediction_memory'`
4. 实现 local-only prediction-memory mask 和 source-aligned memory time embedding
5. 新增 local 版本配置与 smoke 脚本
6. 完成功能验证和 250ms smoke run

**执行结果**：
- 功能验证通过：
  - local block mask 正确
  - `shift-right` 正确
  - `forward()` 与 `generate()` 不再等价
- 250ms smoke run 跑通：
  - `train_loss=0.418`
  - `val_loss=0.412`
  - `val/fp_bps=-0.825`
- rollout smoke eval：
  - `fp-bps=-0.8234`
  - `R2=-0.0002`

**遇到的问题**：
- 无新的代码级错误；当前只完成最小 smoke 验证，尚未做正式 300-epoch 对比实验

**对应 plan.md 任务**：Phase 1.9.2 模型优化迭代记录（Local Prediction Memory Decoder，验证中）
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
   - Neuroformer：三阶段处理（对比对齐→跨模态融合→因果解码），逐 spike 自回归（vs NeuroHorizon 的 bin-level 预测），需对比学习预训练（vs NeuroHorizon 直接用 DINOv2 更简洁）

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

6. 代码级 smoke 验证
   - 使用  跑  的短程训练验证新 checkpoint 逻辑
   - 已确认目录会同时生成 、、真正的 final  和 
   - 已确认  可分别解析新目录，并在输出 JSON 中记录 、、

6. 代码级 smoke 验证
   - 使用 train_1p9_prediction_memory_250ms 跑 epochs=1 的短程训练验证新 checkpoint 逻辑
   - 已确认目录会同时生成 epoch checkpoint、best.ckpt、真正的 final last.ckpt 和 checkpoint_summary.json
   - 已确认 eval_phase1_v2.py 的 --checkpoint-kind best/last 可分别解析新目录，并在输出 JSON 中记录 checkpoint_kind、checkpoint_epoch、split

**遇到的问题与解决**：
1. pad8 vs pad2d 维度不匹配 → 改用 pad/track_mask
2. PyTorch 2.6 weights_only=True 默认 → 添加 weights_only=False
3. InfiniteVocabEmbedding LazyModule → 先初始化 vocab 再计数参数
4. checkpoint 加载后 vocab 已初始化 → 跳过 initialize_vocab


---

## 2026-03-11-18h09

### 任务：同步 AGENTS.md 工作规范文件

**完成时间**：2026-03-11-18h09

**完成内容**：
1. 检查项目根目录说明文件状态，发现缺少 AGENTS.md，但存在 CLAUDE.md
2. 将 CLAUDE.md 原样复制为 AGENTS.md，用于后续按统一规范执行
3. 校验 AGENTS.md 已成功生成，并准备提交到远程仓库

**执行结果**：
- 项目根目录新增 AGENTS.md
- AGENTS.md 内容与 CLAUDE.md 保持一致
- 说明文件已补齐，后续任务可统一按 AGENTS.md 执行

**遇到的问题**：
- 原项目仅存在大写文件名 CLAUDE.md，未直接存在 AGENTS.md
- 解决：按现有规范将 CLAUDE.md 同步为 AGENTS.md

**对应 plan.md 任务**：不直接对应 plan.md 中的代码任务，属于项目工作规范补齐

---

## 2026-03-11-22h

### 任务：Phase 1 完成标记 + proposal_review 同步 + 1.2.4 指标验证

**完成时间**：2026-03-11-22h

**完成内容**：

#### 步骤 0：CLAUDE.md / AGENTS.md 规范更新
在远程 CLAUDE.md（及 AGENTS.md）的plan.md 对应任务中追加两条规范：
- 对 plan.md 中各任务标注依赖/产出/记录文件位置
- 计划内容需足够详细以便直接执行，引用文档需注明出处（文件路径 + 章节号）

#### 步骤 1：proposal_review.md 同步更新
- 在 §2.1 之后插入 §2.1b：Trial-Aligned 数据加载（已实现）
- 在 §2.5 之后插入 §2.5b：AR 预测反馈机制【方案待决策】
- 更新 §2.10 指标表：从 4 行扩展为 6 行，fp-bps 作为主要评估指标
- 更新 §2.12 验收标准：追加 4 项（fp-bps null/random/trained/trial-aligned sampler）

#### 步骤 2：plan.md 完成标记 + 文件引用
- 1.1.7 [x]：评估指标补充（已完成）
- 1.1.8 [ ]（整体未完成）：5/7 子项已完成（TF≡AR 分析、prediction_feedback.py、decoder/model/train 集成），2 项待决策（方案选择、编码方式选择）
- 1.1.9 [x]：Trial-Aligned 数据加载（已完成）
- 为 1.1.1-1.1.6、1.2.1-1.2.4、1.3.x、1.4、1.5 补充依赖/产出/记录引用

#### 步骤 3-4：创建并执行 1.2.4 验证脚本
- 创建 scripts/tests/test_1_2_4_metrics_verification.py（7 项测试）
- 全部 9 项通过、0 项失败、1 项跳过（Test 5b yaml config 路径，已由 Test 5 HDF5 直接验证替代）
- 注意：本验证使用**合成数据**验证公式数学正确性，null_log_rates 从合成 target 中统计 per-neuron 均值得到（与 NLB 方式一致）；实际训练评估中 null_log_rates 应由 compute_null_rates() 从训练集统计

**测试结果摘要**（合成数据，NLB 量级）：
| 测试 | 结果 |
|------|------|
| fp-bps null model = 0 | PASS (0.00000000) |
| fp-bps random < 0 | PASS (-0.3736) |
| fp-bps oracle > 0 | PASS (0.0677, NLB 量级) |
| NLB 交叉验证 | PASS (diff=2.8e-7) |
| checkpoint loadable | PASS |
| per-bin fp-bps shape = [T] | PASS |
| HDF5 trial structure | PASS (655 trials / 3 files) |
| TrialAlignedSampler alignment | PASS (6/6 aligned) |
| Sampler index fields | PASS |

#### 步骤 5：文档管理
- 创建 cc_todo 记录：20260311-phase1-1.1.7-1.1.9-implementation.md、20260311-phase1-1.2.4-metrics-verification.md
- 更新 scripts.md：新增 neurohorizon_metrics.py、eval_psth.py、test_1_2_4_metrics_verification.py
- plan.md 1.2.4 → [x]

**遇到的问题与解决**：
1. **PyTorch 2.6 weights_only 默认值变更**：torch.load() 默认 weights_only=True 导致 checkpoint 加载失败 → 添加 weights_only=False
2. **HDF5 target_id 为 float64 含 NaN**：部分 trial 的 target_id 为 NaN（无效试次），min/max 返回 NaN → 用 np.isnan() 过滤后检查
3. **proposal_review.md 复杂文本替换**：sed 无法处理含反引号的 markdown 内容 → 改用 Python 脚本通过 SSH 执行替换

**对应 plan.md 任务**：1.1.7 [x], 1.1.8 [ ]（部分）, 1.1.9 [x], 1.2.4 [x]


---

## 2026-03-13-02h

### 任务：整理 1.9 两轮分支提交并准备 Local Prediction Memory 正式实验

**完成时间**：2026-03-13-02h

**完成内容**：
1. 将 `20260312_prediction_memory_decoder` 的实现、300-epoch 结果和失败结论提交并推送到 `origin/dev/20260312_prediction_memory_decoder`
2. 更新 `plan.md 1.9.0`：新增 Step 2 和 Step 4 完成后都要执行 `git commit + git push` 的要求
3. 为 `20260313_local_prediction_memory` 补齐正式实验脚本：
   - `run_local_prediction_memory_experiments.sh`
   - `monitor_local_prediction_memory_progress.py`
   - `collect_local_prediction_memory_results.py`
4. 回填 `20260312_prediction_memory_decoder` 的 commit 为 `ebb59fa`
5. 准备在 `dev/20260313_local_prediction_memory` 上提交当前实现并启动正式 `250/500/1000ms` 并行实验

**执行结果**：
- 上一轮失败迭代已独立固化在旧分支，便于后续追踪 teacher-forced / rollout gap 的演变
- 当前轮次的正式实验入口、监控和结果汇总脚本都已就绪
- `plan.md 1.9.0` 现在明确要求每轮优化至少保留两次提交：实现 checkpoint 和结果 checkpoint

**遇到的问题**：
- 由于 `20260312` 与 `20260313` 共用了部分文档和模型文件，需要先把共享文件按轮次拆开后再分别提交；已通过临时备份和文件级回切解决

**对应 plan.md 任务**：Phase 1.9.0 执行规范更新 + Phase 1.9.2 Local Prediction Memory Decoder（验证中）


---

## 2026-03-13-03h

### 任务：完成 Local Prediction Memory 正式实验并归档结果

**完成时间**：2026-03-13-03h

**完成内容**：
1. 完成 `250ms / 500ms / 1000ms` 三组 300-epoch 正式训练与 rollout 评估
2. 生成 `local_prediction_memory_summary.json`，更新 `results.tsv` 和 1.9 趋势图
3. 将 `Local Prediction Memory Decoder` 在 `model.md` / `plan.md` / `cc_todo` / `results.md` 中正式标记为失败迭代并记录原因
4. 归纳本轮相对 `20260312_prediction_memory_decoder` 的改善幅度和相对 `baseline_v2` 的剩余差距

**执行结果**：
- rollout fp-bps：
  - `250ms: 0.1621`
  - `500ms: -0.0105`
  - `1000ms: -0.2122`
- 相比 `20260312_prediction_memory_decoder`：
  - `250ms: +0.0135`
  - `500ms: +0.0048`
  - `1000ms: +0.0468`
- 相比 `baseline_v2`：
  - `250ms: -0.0494`
  - `500ms: -0.1849`
  - `1000ms: -0.3439`

**遇到的问题**：
- local-only memory 虽然削弱了上一轮的全历史错误传播，但 teacher forcing 与 rollout 的分布偏移仍是主导问题，导致长窗口自由 rollout 依旧明显崩塌

**对应 plan.md 任务**：Phase 1.9.2 Local Prediction Memory Decoder（已放弃）


---

## 2026-03-13-11h

### 任务：补充 Local Prediction Memory 机制复盘并实现 Prediction Memory Alignment 新迭代

**完成时间**：2026-03-13-11h

**完成内容**：
1. 在 `20260313_local_prediction_memory.md` 中补写机制讨论，明确解释：
   - 为什么 `baseline_v2` 没有出现同等级 rollout 退化
   - 当前 benchmark `Neuroformer` 为什么不等价于 event-level autoregressive generation
   - 为什么 `NDT2 / IBL-MtM` 这类 one-shot future predictor 在 `fp-bps` 下可能占优
2. 将上述补充提交并推送到 `origin/dev/20260313_local_prediction_memory`，commit `f189e30`
3. 从该节点新建分支 `dev/20260313_prediction_memory_alignment`
4. 实现新的 alignment 训练策略：
   - `prediction_memory_train_mix_prob`
   - `prediction_memory_input_dropout`
   - `prediction_memory_input_noise_std`
   - 训练态 no-grad rollout bootstrap + mixed GT/predicted memory
5. 补齐 1.9 文档、配置和脚本：
   - `cc_core_files/model.md`
   - `cc_core_files/plan.md`
   - `cc_todo/phase1-autoregressive/1.9-module-optimization/20260313_prediction_memory_alignment.md`
   - `scripts/phase1-autoregressive-1.9-module-optimization/20260313_prediction_memory_alignment/`
6. 完成功能验证和 250ms smoke run

**执行结果**：
- 功能验证通过：
  - `target_independence_delta=0.000000`
  - `train_eval_memory_delta=0.011355`
- 250ms smoke run 通过：
  - `train_loss=0.418`
  - `val_loss=0.412`
  - `val/fp_bps=-0.824`
  - rollout smoke eval：`fp-bps=-0.8228`, `val_loss=0.4133`
- 当前新迭代已进入“验证中”状态，具备执行 Step 2 checkpoint 提交和正式三窗口实验的条件

**遇到的问题**：
- 当前 `baseline_v2` 与 benchmark 模型的“rollout”语义和显式 prediction-memory 方案并不一致；在总结文档中已补充代码级区分，避免继续把它们混作同一种 exposure bias 问题

**对应 plan.md 任务**：Phase 1.9.2 Prediction Memory Alignment Training（验证中）


---

## 2026-03-13-19h

### 任务：完成 Prediction Memory Alignment 正式实验并整理结果

**完成时间**：2026-03-13-19h

**完成内容**：
1. 完成 `250ms / 500ms / 1000ms` 三组 300-epoch 正式训练与 rollout 评估
2. 自动生成 `prediction_memory_alignment_summary.json`，并更新 `results.tsv` 与 1.9 趋势图
3. 汇总本轮相对 `baseline_v2` 和 `20260313_local_prediction_memory` 的变化幅度

**执行结果**：
- rollout fp-bps：
  - `250ms: 0.1943`
  - `500ms: 0.1513`
  - `1000ms: 0.1103`
- 相比 `baseline_v2`：
  - `250ms: -0.0172`
  - `500ms: -0.0231`
  - `1000ms: -0.0214`
- 相比 `20260313_local_prediction_memory`：
  - `250ms: +0.0322`
  - `500ms: +0.1618`
  - `1000ms: +0.3225`
- teacher-forced / rollout gap：
  - `250ms: 0.0814`
  - `500ms: 0.1319`
  - `1000ms: 0.1718`

**遇到的问题**：
- alignment 方案虽然显著改善了显式 prediction feedback 的 rollout 稳定性，但目前三个窗口仍略低于 `baseline_v2`，说明 mixed-memory 和 regularization 已经抓住主矛盾，但超参和训练策略还有继续细调空间

**对应 plan.md 任务**：Phase 1.9.2 Prediction Memory Alignment Training（正式结果已完成，等待用户决定是否继续优化）


---

## 2026-03-13-20h

### 任务：启动 Prediction Memory Alignment Tuning 新迭代并完成最小验证

**完成时间**：2026-03-13-20h

**完成内容**：
1. 基于 `dev/20260313_prediction_memory_alignment` 新建分支 `dev/20260313_prediction_memory_alignment_tuning`
2. 新增一轮纯超参级 tuning：
   - `mix_prob 0.25 -> 0.35`
   - `input_dropout 0.10 -> 0.05`
   - `input_noise_std 0.05 -> 0.03`
3. 补齐 1.9 文档、配置和脚本：
   - `cc_core_files/model.md`
   - `cc_core_files/plan.md`
   - `cc_todo/phase1-autoregressive/1.9-module-optimization/20260313_prediction_memory_alignment_tuning.md`
   - `scripts/phase1-autoregressive-1.9-module-optimization/20260313_prediction_memory_alignment_tuning/`
4. 完成功能验证和 250ms smoke run

**执行结果**：
- tuning 功能验证通过：
  - `tuned_mix_prob=0.35`
  - `tuned_input_dropout=0.05`
  - `tuned_input_noise_std=0.03`
  - `target_independence_delta=0.000000`
  - `train_eval_memory_delta=0.008230`
- 250ms smoke run 通过：
  - `train_loss=0.418`
  - `val_loss=0.411`
  - `val/fp_bps=-0.823`
  - rollout smoke eval：`fp-bps=-0.8217`, `val_loss=0.4132`

**遇到的问题**：
- 本轮不改主代码，因此主要风险转到配置名、脚本路径和日志目录命名冲突；已通过单独的新模块名与 smoke run 验证排除

**对应 plan.md 任务**：Phase 1.9.2 Prediction Memory Alignment Tuning（验证中）


---

## 2026-03-14-03h

### 任务：完成 Prediction Memory Alignment Tuning 正式实验并整理结果

**完成时间**：2026-03-14-03h

**完成内容**：
1. 完成 `250ms / 500ms / 1000ms` 三组 300-epoch tuning 正式训练与 rollout 评估
2. 自动生成 `prediction_memory_alignment_tuning_summary.json`，并更新 `results.tsv` 与 1.9 趋势图
3. 汇总本轮相对 `baseline_v2` 和 `20260313_prediction_memory_alignment` 的变化幅度

**执行结果**：
- rollout fp-bps：
  - `250ms: 0.2004`
  - `500ms: 0.1526`
  - `1000ms: 0.1218`
- 相比 `baseline_v2`：
  - `250ms: -0.0111`
  - `500ms: -0.0218`
  - `1000ms: -0.0099`
- 相比 `20260313_prediction_memory_alignment`：
  - `250ms: +0.0060`
  - `500ms: +0.0013`
  - `1000ms: +0.0115`
- teacher-forced / rollout gap：
  - `250ms: 0.0711`
  - `500ms: 0.1197`
  - `1000ms: 0.1656`

**遇到的问题**：
- 这轮 tuning 的收益已经进入“边际改善”阶段，尤其 `500ms` 基本不动；后续如果继续优化，可能需要按窗口分开调 `mix_prob` 与 regularization，而不是维持一套全窗口统一超参

**对应 plan.md 任务**：Phase 1.9.2 Prediction Memory Alignment Tuning（正式结果已完成，等待用户决定是否继续优化）

---

## 2026-03-16-12h16

### 任务：完成 long-horizon forecasting 路线评审并形成 Codex review 文档

**完成时间**：2026-03-16-12h16

**完成内容**：
1. 基于 `results.tsv`、`model.md`、`plan.md`、`progress.md` 和当前 `neurohorizon.py` / `autoregressive_decoder.py` 实现，完成一次面向 long-horizon forecasting 的系统评审
2. 在 `cc_todo/20260316-review/20260316-neurohorizon-long-horizon-review_codex.md` 中整理三部分内容：
   - 对当前 output-side autoregressive 路线的判断
   - 与 `NDT2 / NDT-MTM / Neuroformer / one-step forecast` 的比较判断
   - 面向 NeurIPS 的论文定位与研究转向建议
3. 输出下一步最小实验矩阵：`latent feedback`、`direct multi-horizon objective`、`behavior-conditioned / auxiliary forecast`

**执行结果**：
- 明确确认当前最强 rollout 结果仍为 `baseline_v2`：
  - `250ms: 0.2115`
  - `500ms: 0.1744`
  - `1000ms: 0.1317`
- 当前显式 prediction-memory 最优版本 `prediction_memory_alignment_tuning` 为：
  - `250ms: 0.2004`
  - `500ms: 0.1526`
  - `1000ms: 0.1218`
- 结论：当前 output-side explicit AR feedback 已证明“可被修复到接近 baseline”，但尚未证明比 baseline 更强；后续应优先验证反馈表征层级、multi-horizon 目标和行为/隐变量辅助建模，而不是继续做大量纯超参搜索

**遇到的问题**：
- 当前项目的主要不确定点已不再是“是否做 AR”，而是“什么层级的状态变量和什么训练目标最适合 long-horizon rollout”；评审结论已将主问题重新收敛到 `long-horizon neural forecasting` 本身，而非继续围绕 count-level prediction memory 做小修小补


---

## 2026-03-16-12h

### 任务：项目研究方向深度讨论与评审

**完成时间**：2026-03-16-12h

**完成内容**：

基于对项目全部实验数据（Phase 1 的 4 轮 prediction feedback 优化、baseline_v2 benchmark 对比、ablation studies）和代码架构的深入分析，对三个核心研究方向问题进行了充分讨论，产出三份深度分析文档：

1. **`ar_effectiveness_claude.md`**（约 26KB）
   - 系统分析了 AR 生成在长时程神经活动预测中的有效性
   - 核心结论：显式 AR 反馈在 4 轮实验中均未超越 baseline_v2（无反馈模型）
   - 分析了 exposure bias、高维随机过程、信息瓶颈等根本性挑战
   - 提出 causal self-attention 的实际作用是 temporal regularization 而非 genuine AR generation
   - 对比了 bin-level AR 与 spike-level AR（Neuroformer）的 trade-off

2. **`neurips_innovation_claude.md`**（约 25KB）
   - 评估了项目当前的实际贡献（encoder 迁移、PerNeuronMLPHead、benchmark、exposure bias 分析）
   - 分析了以 AR 为核心创新叙事的风险（reviewer 可能的 5 个关键质疑）
   - 提出了 4 种 reframing 方案（Foundation model / Systematic study / AR failure analysis / Pivot to new approach）
   - 给出了针对 NeurIPS 2026 / ICLR 2027 / Nature Methods 的差异化发表策略
   - 包含具体的额外实验清单和时间线规划

3. **`long_horizon_prediction_claude.md`**（约 33KB）
   - 论证了长时程预测（500ms-1s+）作为研究问题的意义（BCI 应用、科学理解、文献 gap）
   - 详细分析了 4 个技术方向：Latent Dynamics Model / 继续优化 AR / Diffusion&Flow Matching / Behavioral Covariates
   - 对各方向进行了 8 维对比评估（创新性、难度、性能预期、兼容性、NeurIPS 叙事强度等）
   - 推荐 Latent Dynamics Model（Neural ODE/SDE + POYO+ encoder）为最优路径
   - 给出了具体的实现路线图和 3 阶段推荐路径

**执行结果**：
- 三份文档均成功写入 `cc_todo/20260316-review/` 目录
- 无乱码（已通过 grep 验证）
- 总结：当前 AR 方案的实验证据不支持其作为核心创新，建议考虑 Latent Dynamics 方向或重新 framing 论文叙事

**遇到的问题**：无

**对应 plan.md 任务**：不直接对应 plan.md 中的代码任务，属于项目方向性讨论与评审

---

## 2026-03-16-15h12

### 任务：完成视频生成方法迁移视角下的 long-horizon forecasting 补充评审

**完成时间**：2026-03-16-15h12

**完成内容**：
1. 新增 `cc_todo/20260316-review/20260316-video-generation-transfer-review_codex.md`，系统讨论当前神经活动自回归预测与流式视频生成在 `exposure bias / rollout drift / memory design` 层面的结构对应
2. 结合外部方法资料，逐项评审 `self-forcing`、`rolling forcing`、`global sink frame`、`MemFlow / memory bank` 对当前 `NeuroHorizon` 的可迁移性
3. 将外部方法翻译为当前项目下的 3 组最小实验方向：
   - `latent self-forcing`
   - `chunkwise rolling forecast`
   - `global anchor + causal latent bank`

**执行结果**：
- 结论一：当前 `prediction_memory_alignment` 系列已经属于弱版 video-style forcing，但 forcing 的状态变量仍停留在 `raw count / expected count` 层级，尚未真正解决 long-horizon rollout stability
- 结论二：最值得优先借鉴的是 `self-forcing`，但应翻译为 `latent self-forcing`，而不是继续对 raw count feedback 做更强 mixing
- 结论三：`rolling forcing` 在当前项目中更合理的落地方式是 `chunkwise autoregressive rollout`，以减少 `500ms / 1000ms` 预测时的深链条误差累积
- 结论四：`global sink frame` 不能直接照搬，更合理的神经版本是来自 observation history 的 `global anchor latent`
- 结论五：`MemFlow` 式 memory bank 只能以 `causal latent memory bank` 的形式谨慎尝试，不能把当前 raw count prediction-memory 继续做大

**遇到的问题**：
- 视频生成和神经活动预测虽然在 rollout 失稳问题上同构，但在状态变量、全局锚点和不确定性来源上差异很大；因此本轮评审的关键收获不是“找到了可直接复刻的模块”，而是明确了应优先借鉴“训练和记忆机制”，而不是照搬视频 token / frame 级结构


---

## 2026-03-16-14h

### 任务：从流式视频生成技术到神经活动预测的迁移可行性分析

**完成时间**：2026-03-16-14h

**完成内容**：

基于对自回归流式视频生成领域前沿技术（Self-Forcing, Rolling Forcing, MemFlow, Deep Forcing, Reward Forcing, Causal Forcing 等）的系统调研，以及对 NeuroHorizon 代码架构的逐模块映射分析，撰写了一份技术迁移可行性深度分析文档：

**`video_ar_transfer_claude.md`**（约 393 行）
- 建立了 AR 视频生成与 AR 神经活动预测的 10 维系统性类比映射
- 对 6 大技术逐一分析了适配性、具体改造方案（映射到代码 8 个插入点）、预期收益和风险
- 推荐优先级排序：Self-Forcing(1) > Streaming Long Tuning(2) > Rolling Forcing(3) > EMA-Sink(4) > Global Sink(5) > MemFlow(6)
- 将 4 轮 AR 实验的失败/改进模式映射到视频领域的对应经验
- 关键结论：视频技术可帮助缩小 explicit AR 与 baseline_v2 的差距，但 AR 反馈对 neural data 的增量贡献可能本身有限（因时间冗余度远低于视频）
- 建议的实施路线：Self-Forcing 分布匹配训练 + Streaming Long Tuning + EMA-Sink 组合

**参考文献**：Self-Forcing (arXiv:2506.08009), Rolling Forcing (arXiv:2509.25161), MemFlow (arXiv:2512.14699), Deep Forcing (arXiv:2512.05081), Reward Forcing (arXiv:2512.04678), Causal Forcing (arXiv:2602.02214), EAG (arXiv:2511.17606)

**执行结果**：
- 文档成功写入 `cc_todo/20260316-review/video_ar_transfer_claude.md`
- 无乱码（已验证）

**遇到的问题**：无

**对应 plan.md 任务**：不直接对应 plan.md 中的代码任务，属于技术方向调研


---

## 2026-03-16-19h09

### 任务：审查 1.8.3 benchmark 对比实验的实现与结果有效性

**完成时间**：2026-03-16-19h09

**完成内容**：
1. 系统审查 plan.md 中 1.8.3、主记录文档、neural_benchmark 适配/训练/可视化代码、phase1_benchmark 结果文件、results.md 与 1.3.4 / 1.4 / 1.5 下游复用脚本
2. 新增审计报告：cc_todo/20260316-review/1.8.3-benchmark-audit_codex.md
3. 在审计报告中按严重度给出批判性结论和修正建议，重点覆盖数据使用逻辑、模型适配逻辑、评估处理逻辑与结果传播链

**执行结果**：
- 结论一：当前 1.8.3 不能被视为正式 benchmark 对比；更准确的定位是“受 NDT2 / Neuroformer / IBL-MtM 启发的简化版内部 baseline”
- 结论二：文档声称的“同一 test intervals + 统一四指标评估”与真实代码不符；当前结果实质上是 validation 上的选择结果
- 结论三：benchmark 与 NeuroHorizon 1.3.4 的 continuous eval 并非同一采样协议，不能直接写成“完全同条件对比”
- 结论四：问题已经传播到 results.md、1.3.4 benchmark 对比图、1.4 / 1.5 可视化脚本，相关强结论应降级或回收
- 无乱码，已检查 cc_core_files、cc_todo/phase*、cc_todo/20260316-review

**遇到的问题**：
- 远程命令写入 Markdown 时，本地 shell 会提前解释反引号，首次写入失败；改为先在本地生成临时文件，再通过 scp 推送到远程后解决

**对应 plan.md 任务**：Phase 1 → 1.8.3 审计复核（不新增实验，仅补充严格审查结论）


---

## 2026-03-16-16h

### 任务：Option D 展开 — 隐空间动力学模型与扩散模型方案详细设计

**完成时间**：2026-03-16-16h

**完成内容**：

将 neurips_innovation_claude.md 中的 Option D（Pivot to Latent Dynamics / Diffusion Decoder）展开为两个具体可执行的技术方案，并补充了 6.3 节的额外实验清单。

**产出文档**：`cc_todo/20260316-review/option_d_implementation_claude.md`（556 行）

**方案 1: 隐空间动力学模型（Latent Dynamics Decoder）**
- 基于 POSSM (NeurIPS 2025) 启发，引入 SSM backbone (S4D/GRU/Mamba)
- 架构：POYO+ Encoder -> Attention Pooling -> SSM Dynamics -> PerNeuronMLPHead
- 推荐 S4D 作为起步（M1 旋转动力学是准线性的）
- 预期 1000ms fp-bps: 0.15-0.18（vs baseline_v2 的 0.1317）
- 实现周期：3-4 周

**方案 2: 扩散模型生成器（Latent Diffusion Decoder）**
- 基于 LDNS (NeurIPS 2024 Spotlight) 启发
- 架构：Count Autoencoder + Conditional DDPM conditioned on POYO+ latents
- 当前无论文用 diffusion 做 spike forward prediction（研究空白）
- 预期 1000ms fp-bps: 0.14-0.17
- 实现周期：4-6 周

**推荐路线**：优先做方案 1（更快、更低风险），方案 2 作为补充对比

**6.3 额外实验**：6 个实验的详细设计已补充（non-causal ablation, encoder ablation, additional dataset, bin size ablation, iso-parameter, visualization）

**执行结果**：
- 文档成功写入，无乱码
- 内容涵盖：POSSM/LDNS/LFADS/NEDS/EAG 等文献调研 + 具体架构设计 + 训练策略 + 预期性能 + 风险分析 + 实验清单

**遇到的问题**：无

**对应 plan.md 任务**：不直接对应 plan.md 代码任务，属于新方向的实施方案设计


---

## 2026-03-16-22h22

### 任务：审查 plan.md 中 1.2 基础功能验证与 1.3 预测窗口实验的 v2 代码与评估流程

**完成时间**：2026-03-16-22h22

**完成内容**：
1. 系统审查 `plan.md` 中 `1.2` / `1.3` 对应的真实代码路径，包括 `neurohorizon.py`、`autoregressive_decoder.py`、`neurohorizon_metrics.py`、`train.py`、`trial_sampler.py`、`eval_phase1_v2.py`、`eval_psth.py` 与相关测试脚本
2. 核查 v2 baseline 的数据处理、sampler 行为、target 构造、null model 计算、fp-bps / R² / PSTH-R² 的实现与结果聚合方式
3. 新增严格审查文档：`cc_todo/20260316-review/20260316-plan-md-v2-code-review_codex.md`

**执行结果**：
- 结论一：当前 v2 baseline 的数据流、loss 和 fp-bps 指标实现整体合理，作为 Phase 1 baseline 成立
- 结论二：`1.3.4` 主表结果更准确地说是 teacher-forced 条件下的 causal future-bin prediction，而不是严格的 open-loop rollout benchmark
- 结论三：`eval_phase1_v2.py` 里的连续评估结果使用 batch-level 简单平均，不是全局累计聚合；PSTH-R² 在主脚本中实际先做了 neuron mean，和 `psth_r2()` 的完整 neuron 维实现存在口径漂移
- 结论四：`PerNeuronMLPHead` 是合理 baseline，但不是最优输出头；更值得优先修的是 rollout-first 评估协议，而不是继续把 `1.2/1.3` 过度表述为“已严格验证的长时程 AR 生成”
- 无乱码，已准备按远程规范提交

**遇到的问题**：
- 远程仓库中已存在未提交修改：`cc_todo/20260316-review/neurips_innovation_claude.md`；本次未触碰该文件，提交时将只纳入本次新增 review 文档与 `progress.md`
- 远程追加 `progress.md` 时，本地 shell 会提前解释 Markdown 里的反引号；改为先生成临时文件再推送到远程后解决

**对应 plan.md 任务**：Phase 1 → 1.2 / 1.3 的代码与评估流程审计复核（不新增实验，仅补充严格审查结论）


---

## 2026-03-17-14h45

### 任务：修正 v2 评估协议并启动 1.3.4 evalfix 全量重跑

**当前状态时间**：2026-03-17-14h45

**已完成内容**：
1. 修正 continuous validation / test 的采样协议：`examples/neurohorizon/train.py` 和 `scripts/analysis/neurohorizon/eval_phase1_v2.py` 统一改为 `SequentialFixedWindowSampler`
2. 修正 continuous 指标聚合：`fp-bps / R²` 改为全局累计版，`torch_brain/utils/neurohorizon_metrics.py` 增加 stats/finalize helpers
3. 统一 trial-aligned 主 PSTH 指标为 `per_neuron_psth_r2`，同步修改 `eval_phase1_v2.py`、`eval_psth.py`、`phase1_v2_visualize.py`
4. 新增 evalfix 配置与脚本：
   - `examples/neurohorizon/configs/train_v2_evalfix_{250,500,1000}ms{,_trial}.yaml`
   - `scripts/analysis/neurohorizon/run_phase1_v2_evalfix.sh`
   - `scripts/analysis/neurohorizon/compare_phase1_v2_evalfix.py`
5. 同步更新文档：
   - `cc_core_files/plan.md`
   - `cc_core_files/results.md`
   - `cc_core_files/scripts.md`
   - `cc_todo/20260316-review/20260316-plan-md-v2-code-review_codex.md`
   - `cc_todo/phase1-autoregressive/20260317-phase1-1.3.4-evalfix-rerun.md`
6. 完成 smoke 验证：
   - continuous：`Trainer.fit/validate/test` 通过，`eval_phase1_v2.py --split valid/test` 通过
   - trial-aligned：`Trainer.fit/validate/test` 通过

**正在进行**：
- 已通过 `screen` 后台启动 `bash scripts/analysis/neurohorizon/run_phase1_v2_evalfix.sh`
- 日志文件：`results/logs/phase1_v2_evalfix_rerun.log`
- 当前阶段：第一组 `250ms-cont` 训练中

**遇到的问题**：
- `phase1_v2_*` legacy checkpoint 与当前代码不完全兼容，无法直接用新脚本加载；因此按计划采用全量重跑，而不是直接重评估旧 checkpoint
- 六组 300-epoch 训练预计耗时较长，因此先提交代码/文档修正，再在实验完成后追加结果提交

**对应 plan.md 任务**：Phase 1 → 1.3.4（评估协议修正 + 全量重跑）


---

## 2026-03-17-17h55

### 任务：完成 1.3.4 evalfix 全量重跑、离线评估与新旧结果对比

**完成时间**：2026-03-17-17h55

**完成内容**：
1. 完成 `phase1_v2_evalfix_{250ms,500ms,1000ms}_{cont,trial}` 六组 300-epoch 训练
2. 完成六组条件的 valid/test 离线评估，生成 `eval_v2_valid_results.json` 与 `eval_v2_test_results.json`
3. 修正 `compare_phase1_v2_evalfix.py` 的 `version_0` 路径问题并生成 `comparison.{json,md}`
4. 修正 `phase1_v2_visualize.py` 使其优先读取 `phase1_v2_evalfix_*` 目录，重新生成新版 5 张图
5. 回填 `cc_todo/phase1-autoregressive/20260317-phase1-1.3.4-evalfix-rerun.md`、`cc_core_files/results.md`、`cc_todo/20260316-review/20260316-plan-md-v2-code-review_codex.md`

**执行结果**：
- continuous valid `fp-bps`：
  - 250ms: `0.2164`（legacy `0.2115`，+0.0049）
  - 500ms: `0.1823`（legacy `0.1744`，+0.0079）
  - 1000ms: `0.1374`（legacy `0.1317`，+0.0056）
- trial-aligned valid `fp-bps` 仍为负值：
  - 250ms: `-0.1983`
  - 500ms: `-0.2839`
  - 1000ms: `-0.2152`
- 新 test continuous `fp-bps`：
  - 250ms: `0.2223`
  - 500ms: `0.1740`
  - 1000ms: `0.1348`
- 新 `per_neuron_psth_r2` 已统一落地，旧 population-mean `PSTH-R2` 仅保留为 legacy 对照

**遇到的问题**：
- `compare_phase1_v2_evalfix.py` 和 `phase1_v2_visualize.py` 初版都没有正确下探到 `lightning_logs/version_0/`，导致读取新结果失败；已修复并重新生成对比产物

**对应 plan.md 任务**：Phase 1 → 1.3.4（evalfix 重跑完成 + 结果对比）


---

## 2026-03-17-22h35

### 任务：调研 fp-bps 实现、Neuroformer 原始训练/推理范式，以及 IBL-MtM 原始 paper/code 与 base v2 的差异

**完成时间**：2026-03-17-22h35

**完成内容**：
1. 梳理当前项目 `fp-bps` 实现：
   - 阅读 `torch_brain/utils/neurohorizon_metrics.py`
   - 阅读 `examples/neurohorizon/train.py`
   - 阅读 `scripts/analysis/neurohorizon/eval_phase1_v2.py`
   - 明确 neuron / time / batch / session / train-val 的聚合口径
2. 调研 Neuroformer 原始 repo：
   - 阅读 `neural-benchmark/benchmark_models/neuroformer/README.md`
   - 阅读 `neuroformer_train.py` / `neuroformer_inference.py`
   - 阅读 `neuroformer/data_utils.py` / `model_neuroformer.py` / `simulation.py`
   - 确认其训练为 teacher forcing，推理为 token-level rollout，常见窗口为 50ms current + 50–150ms past
3. 调研 IBL-MtM 原始 repo 与 NeurIPS 2024 paper：
   - 阅读 `ibl-mtm/src/configs/{ndt1,ndt1_prompting,trainer_ndt1}.yaml`
   - 阅读 `ibl-mtm/src/models/{ndt1,masker}.py`
   - 阅读 `ibl-mtm/src/utils/eval_utils.py`
   - 对照论文 single-session forward prediction 表，确认高分主要来自 `all mask + prompt` 设定
4. 新建调研记录：
   - `cc_todo/20260316-review/QA_codex.md`

**执行结果**：
- 当前主 `fp-bps` 已确认是：
  - train-split per-neuron null
  - global spike-weighted aggregation
  - 非 per-neuron mean / 非 per-session mean / 非 batch mean
- Neuroformer 原始实现已确认：
  - 训练是 teacher forcing 的 token CE
  - 推理是逐 token autoregressive rollout，且默认 `true_past=False` 会跨窗口回填预测 past
  - 因此 exposure bias 客观存在
- IBL-MtM 原始实现已确认：
  - paper 里的高 forward prediction bps 不能与当前 `base v2` 的 `0.21` 直接比较
  - 差异同时来自任务定义、history 长度、metric/null baseline、聚合方式、数据集与训练目标
- 额外确认：
  - 当前仓库旧 `IBL-MtM-like` / `Neuroformer-like` benchmark wrapper 仍然只是 legacy simplified baseline，不代表原始模型

**遇到的问题与解决**：
- 远程环境缺少 `rg`，改用 `grep` / `sed` 定点阅读源码，不影响结论
- 外部论文 / 官方仓库信息需要与当前项目实现交叉核对，已通过原始 repo 代码与现有审计文档双重验证，避免只看论文摘要得出过强结论

**对应记录文件**：
- `cc_todo/20260316-review/QA_codex.md`

## 2026-03-17 | jp-video-1 | 1.8.3 faithful NDT2 数据路径修正 + 250ms 二次重跑

继续推进 NeuroHorizon 的 1.8.3 faithful benchmark 主线，重点不是继续调 optimizer，而是回到 Claude 补充审查里提到的数据适配细节，重新核对 NDT2 faithful bridge 的 flat tokenization 与 padding 是否真正贴近上游 `serve_tokenized_flat` 行为。

**完成内容**：
1. 继续调研上游 NDT2 配置与任务路径：
   - 阅读 `context_general_bci/config/exp/nlb/rtt_5.yaml`
   - 阅读 `config/exp/flat/rtt_m25.yaml`
   - 阅读 `config/exp/arch_rtt/single_ndt2_nlb_r300.yaml`
   - 阅读 `task_io.py`, `model.py`, `dataset.py`
   - 确认 `heldout_decoding` 在部分 NLB 配置中存在，但更接近 flat-token RTT 的官方路径允许只走 `ShuffleInfill`
2. 修正 `neural-benchmark/faithful_ndt2.py` 的关键数据路径问题：
   - per-session variable-length flat tokenization，不再把所有 session 强行扩到全局 `72-channel` token 序列
   - `collate_faithful_ndt2_batch()` 改为合法 pad token padding
   - 新增 `pad_token < max_spatial_tokens` 保护，修复 mixed-session batch 上的 `position` padding 越界
   - 默认配置进一步对齐到上游 `flat_enc_dec/f8`：`token spike embedding`, `dropout=0.1`, `lr_ramp_steps=100`, `lr_decay_steps=2500`
3. 完成多轮 250ms 验证与 full-data 重跑：
   - `results/logs/phase1_benchmark_faithful_ndt2_f8align_trainverify_250ms/`
   - `results/logs/phase1_benchmark_faithful_ndt2_f8align_trainverify_acc1_250ms/`
   - `results/logs/phase1_benchmark_repro_faithful_ndt2_250ms_f8align_pad8_e10/`
   - `results/logs/phase1_benchmark_repro_faithful_ndt2_250ms_projectfix_pad8_e10/`
4. 同步更新记录文档：
   - `cc_todo/20260316-review/1.8.3-benchmark-audit_codex.md`
   - `cc_todo/phase1-autoregressive/20260312-phase1-1.8-benchmark.md`
   - `cc_core_files/plan.md`
   - `cc_core_files/results.md`
   - `cc_core_files/scripts.md`

**关键结果**：
- `f8align_trainverify_acc1_250ms`：best valid `fp-bps = -3.4261`，test `fp-bps = -3.5676`
- `phase1_benchmark_repro_faithful_ndt2_250ms_f8align_pad8_e10`：
  - best valid `fp-bps = -0.6806`
  - held-out test `fp-bps = -0.6707`
  - test `PSTH-R² = 0.0575`
- `phase1_benchmark_repro_faithful_ndt2_250ms_projectfix_pad8_e10`：
  - best valid `fp-bps = -0.6557`
  - held-out test `fp-bps = -0.6570`
  - test `PSTH-R² = -0.5924`

**结论更新**：
1. 旧 `causalfix_e20` 的 `-0.0078 / 0.3833` 不能再作为 faithful NDT2 主参考；修正 tokenization/padding 后，full-data 250ms 结果稳定落在 `-0.66 ~ -0.67`
2. 这说明旧 near-zero 结果被错误 data path 污染，1.8.3 的 NDT2 faithful 结论需要整体下调
3. `heldout_decoding` 不是当前最高优先级问题；更需要优先回答的是 canonical forward prediction benchmark 与上游 RTT 自监督目标的 objective-level mismatch

**遇到的问题与解决**：
- full-data `f8align_e10` 首次运行触发 CUDA `indexSelectLargeIndex` / `device-side assert`  
  原因是旧 `pad_token=20` 对 mixed-session batch 的 `position` padding 越界；已改为合法 `pad_token=8` 并增加显式保护
- 初始小样本 trainverify 使用默认 `accumulate=16` 导致实际 optimizer step 过少  
  已补跑 `accumulate=1` 对照，用于区分“训练规程欠拟合”与“代码链路报错”

**对应记录文件**：
- `cc_todo/20260316-review/1.8.3-benchmark-audit_codex.md`
- `cc_todo/phase1-autoregressive/20260312-phase1-1.8-benchmark.md`

## 2026-03-18 | jp-video-1 | 补充 1.8.3 NDT2 faithful 当前阶段总结与失败原因分析

继续整理 NeuroHorizon 的 1.8.3 benchmark 文档，这轮不改代码，只修正文档口径，重点回答三件事：`causalfix_e20` 现在到底是什么、当前 faithful NDT2 已经整合到了什么阶段、以及为什么“实现已经打通但 benchmark 结果仍显著不成立”。

**完成内容**：
1. 重新核对当前 `neural-benchmark/faithful_ndt2.py` 的实现边界：
   - 数据源：同一份 `perich_miller_10sessions.yaml`
   - split：同一份 `train / valid / test`
   - eval / metric：同一 held-out continuous / trial-aligned `fp-bps / R² / Poisson NLL / PSTH-R²`
   - 训练过程：仍使用上游 `BrainBertInterface` 与 upstream `_step()` loss
2. 明确当前与 NeuroHorizon **已经统一** 和 **尚未完全同构** 的部分：
   - 已统一：数据源 / split / eval / metric
   - 尚未同构：train dataloader / sampler 实现
   - 但这不再被视为当前主问题
3. 更新两份核心文档：
   - `cc_todo/phase1-autoregressive/20260312-phase1-1.8-benchmark.md`
   - `cc_todo/20260316-review/1.8.3-benchmark-audit_codex.md`
   - 修正 `causalfix_e20` 的定位：从“阶段性主参考”正式降级为“data-path 修复前的历史中间结果”
   - 新增“当前 NDT2 faithful 阶段总结”
   - 新增“为什么实现打通后结果仍显著不成立”的假设清单

**本轮补充的关键判断**：
1. 当前 faithful NDT2 已经不是“实现没打通”，而是“实现打通后，当前 canonical benchmark 上仍显著失败”
2. 当前更可信的 250ms 结果仍是：
   - `f8align_pad8_e10`: held-out test `fp-bps = -0.6707`, `PSTH-R² = 0.0575`
   - `projectfix_pad8_e10`: held-out test `fp-bps = -0.6570`, `PSTH-R² = -0.5924`
3. 当前最可能的失败原因被文档收口为“待验证假设”：
   - `ShuffleInfill` 与 canonical forward prediction 存在 objective mismatch
   - train-time sampling distribution 与 NeuroHorizon 主线不同
   - NDT2 原始 inductive bias 可能不适合当前 canonical forecasting benchmark
   - mixed-session flat token batching 仍可能是结构性困难
   - compatibility bridge 不是上游原生训练生态
   - 上游 optimizer/scheduler 直迁失败，说明问题不太像局部调参不足

**遇到的问题与解决**：
- 两份文档里残留了互相矛盾的旧口径：前文仍把 `causalfix_e20` 当阶段主参考，后文又已把它降级  
  已统一改成“历史中间结果，不再作为当前主参考”
- 需要把“已证实事实”和“当前猜测原因”分开写，避免文档把尚未验证的失败原因写成定论  
  已在两份文档中统一使用“当前最可能的解释 / 待验证假设”口径

**对应记录文件**：
- `cc_todo/phase1-autoregressive/20260312-phase1-1.8-benchmark.md`
- `cc_todo/20260316-review/1.8.3-benchmark-audit_codex.md`


## 2026-03-18 1.8.3 faithful IBL-MtM / Neuroformer bridge

**完成内容**：
1. 新增 `neural-benchmark/faithful_ibl_mtm.py`：
   - 直接接入上游 `NDT1 + stitching + session prompting + ssl/PoissonNLL`
   - 用 canonical windows 驱动 train / best-valid / held-out test / trial-aligned eval
   - 训练与评估统一使用显式 `forward_pred` heldout mask；batch 改成 session-pure
2. 新增 `neural-benchmark/faithful_neuroformer.py`：
   - 直接接入上游 `Tokenizer + Neuroformer.forward + autoregressive generation`
   - 关闭视觉 / 行为分支，只保留 neural token generation 主体
   - generation 后 re-bin 到 `20ms` counts，再接统一 continuous / trial-aligned eval
3. 完成两条 faithful 线的 smoke：
   - IBL-MtM smoke：valid `fp-bps = -7.9962`，trial `fp-bps = -11.3897`
   - Neuroformer smoke：valid `fp-bps = -13.8177`，trial `fp-bps = -10.8324`
4. 完成两条 faithful 线的 250ms limited-window debug e1：
   - IBL-MtM：`results/logs/phase1_benchmark_repro_faithful_ibl_mtm_250ms_debug_e1/results.json`
     - best-valid/test `fp-bps = -7.0094 / -7.4991`
     - test `PSTH-R² = -40.8050`
   - Neuroformer：`results/logs/phase1_benchmark_repro_faithful_neuroformer_250ms_debug_e1/results.json`
     - best-valid/test `fp-bps = -14.0545 / -14.2368`
     - test `teacher_forced_loss = 3.5635`
     - test `PSTH-R² = -1475.9126`
5. 已同步更新：
   - `cc_todo/phase1-autoregressive/20260312-phase1-1.8-benchmark.md`
   - `cc_todo/20260316-review/1.8.3-benchmark-audit_codex.md`
   - `cc_core_files/results.md`
   - `cc_core_files/scripts.md`
   - `cc_core_files/plan.md`

**关键判断**：
1. IBL-MtM / Neuroformer faithful 都已经从“简化替身模型”推进到“上游核心模型 + compatibility bridge”阶段
2. 但当前两条 faithful 线都只到 `smoke + 250ms debug negative stage`，距离正式 benchmark 结果还差很远
3. 当前最直接的结论不是“NeuroHorizon 优于它们”，而是“faithful pipeline 已打通，但结果仍显著失败，需要继续查 fidelity / objective / protocol mismatch”

**遇到的问题与解决**：
- IBL-MtM 上游 `NDT1.__init__` 依赖相对路径 `src/configs/ndt1.yaml`，直接从项目根执行会失败  
  已在 faithful bridge 中对模型实例化增加临时切换到上游仓库根目录的 compatibility 处理
- Neuroformer 在无视觉输入时，`FeatureEncoder.forward()` 对 `visual=None` 处理不完整  
  已在 bridge 中加入最薄的 no-vision compatibility patch，只修空值分支，不改动 neural 主路径
- Neuroformer autoregressive decode 不能直接用变长前缀；其 `pos_emb` 假定当前 `id` 序列始终等于固定 `block_size`  
  已改为“固定 block_size + pad + 从最后一个非 pad 位置取 logits”的 decode 方式
- trial-aligned 统一评估要求 batch 内保留 full-window `spike_counts` 再由评估函数切 prediction 部分；最初只存了 pred-only target，导致 shape 不匹配  
  已改为同时存 `full-window spike_counts` 与 `pred-only target_counts`

## 2026-03-18 1.8.3 严格适配复核与 250ms 先行验证

**完成内容**：
1. 对 NDT2 / IBL-MtM / Neuroformer 三条 faithful 线做了严格复核，重新确认：
   - 三模型都继续使用与 NeuroHorizon 相同的数据源 / split / held-out eval / metric
   - NDT2 继续保留 upstream `ShuffleInfill`
   - IBL-MtM 已从 fixed `forward-pred` 训练修正回 upstream SSL multi-mask 训练
   - Neuroformer 已补齐 `rollout` 与 `true_past` 两种正式推理模式
2. 完成 IBL-MtM `250ms` full-data multimask e1：
   - `results/logs/phase1_benchmark_repro_faithful_ibl_mtm_250ms_multimask_e1/results.json`
   - held-out test `fp-bps = -2.9547`
   - held-out trial `fp-bps = -1.9327`
   - epoch 1 train mask 计数：`causal = 550`, `neuron = 573`
3. 完成 Neuroformer dual-mode smoke 复核：
   - `results/logs/phase1_benchmark_faithful_neuroformer_smoke_dualmode_v2/smoke.json`
   - valid rollout / true_past `fp-bps = -13.8295 / -10.9855`
4. 对 Neuroformer `250ms` full-data formal dual-mode eval 做了两轮尝试：
   - `phase1_benchmark_repro_faithful_neuroformer_250ms_dualmode_e1`：训练后进入 held-out generation，累计超过 `30 min` 仍未产出 `results.json`
   - 统计 250ms test 窗口真实 event：`mean=64.0`, `p95=113`, `p99=134.9`, `max=167`
   - 按此收紧到 `max_generate_steps = 192` 再跑 `phase1_benchmark_repro_faithful_neuroformer_250ms_dualmode_e1_g192`，`13+ min` 仍未到 checkpoint，确认 runtime 已是 formal 250ms blocker
5. 已同步更新：
   - `cc_todo/phase1-autoregressive/20260312-phase1-1.8-benchmark.md`
   - `cc_todo/20260316-review/1.8.3-benchmark-audit_codex.md`
   - `cc_core_files/results.md`
   - `cc_core_files/scripts.md`
   - `cc_core_files/plan.md`

**关键判断**：
1. NDT2 与 IBL-MtM 当前都已经达到“实现语义对齐，可以继续以 250ms 为 gate 做正式 benchmark”的阶段，但结果仍显著为负
2. Neuroformer 当前已经达到“语义对齐”，但还没有达到“250ms formal full-data dual-mode eval 可稳定完成”的阶段
3. 因此 1.8.3 仍然保持打开，且 `500ms / 1000ms` 暂不启动

**遇到的问题与解决**：
- IBL-MtM faithful 之前把训练固定成 `forward-pred`，不符合“保留 upstream multi-mask 训练”要求  
  已改为 combined multi-mask 训练，held-out eval 才单独做 one-step `forward_pred`
- Neuroformer `true_past` 原本也是逐 token 重跑模型，formal eval 代价过高  
  已改为直接复用 teacher-forced 前向输出解码，但 full-data dual-mode formal 仍然过慢，现阶段应把运行成本视为正式 blocker

## 2026-03-18 02:50

**任务**：补充 `cc_todo/20260316-review/QA_codex.md`，扩写 NeuroHorizon 主 `fp-bps` 的合理性判断、建议新增的 IBL-MtM 对齐 comparison metric，以及项目内部数据组织 / evaluation / AR decoder / `PerNeuronMLPHead` 的实现说明

**完成内容**：
1. 在 Q1 末尾新增主指标判断：
   - 明确当前 `global spike-weighted + train-null fp-bps` 适合作为 NeuroHorizon 主指标
   - 逐维度讨论 neuron / time / batch / session / null baseline 的合理性与边界
2. 在 Q1 中补充建议新增的对比指标定义：
   - `per-neuron mean fp-bps`
   - `eval-split per-neuron null`
   - 明确其定位是 IBL-MtM / NLB 风格 compatibility metric，而不是主指标替代
3. 新增 NeuroHorizon 自身实现补充章节：
   - dataset / sampler / split / trial metadata 的组织方式
   - continuous 与 trial-aligned 的样本分布差异
   - validation/test 与离线 `eval_phase1_v2.py` 的 metric 计算链路
   - `AutoregressiveDecoder`、`generate()` 与 `PerNeuronMLPHead` 的职责分工

**执行结果**：
- `QA_codex.md` 现在不仅回答“当前实现是什么”，也明确回答“是否合理、是否要改、若与 IBL-MtM 对齐应如何并列报告”
- 文档中已把“项目内主结论口径”和“外部 benchmark 对齐口径”拆开，避免后续讨论时再把 metric 差异误当作模型差异

**遇到的问题与解决**：
- 这次补充不是单纯加观点，而是要把观点和当前代码实现逐条绑死；为避免把 decoder / sampler / null 逻辑说偏，先重新核对了 `train.py`、`dataset.py`、`sampler.py`、`neurohorizon.py`、`neurohorizon_metrics.py` 与 `eval_phase1_v2.py` 的真实实现，再回写文档

## 2026-03-18 06:35

**任务**：实现 `plan.md` 的 `1.3.5` 与 `1.3.6`

**完成内容**：
1. 新增 IBL-MtM 风格 comparison metric：
   - 在 `torch_brain/utils/neurohorizon_metrics.py` 中加入 `ibl_mtm_bps` 统计 helper
   - 在 `scripts/analysis/neurohorizon/eval_phase1_v2.py` continuous 结果中新增 `continuous.ibl_mtm_bps`
2. 完成 `1.3.5` causal baseline 补评估：
   - 对 `phase1_v2_evalfix_{250,500,1000}ms_cont` 的 valid/test 补跑新指标
   - 产出 `results/logs/phase1_v2_metric_extension_comparison/comparison.{json,md}`
   - 结果：
     - 250ms: valid/test `ibl_mtm_bps = 0.2234 / 0.2321`
     - 500ms: `0.1609 / 0.1555`
     - 1000ms: `0.0381 / 0.0532`
3. 完成 `1.3.6` non-causal ablation：
   - 新增 `neurohorizon_small_{250,500}ms_noar.yaml` 和 `train_v2_nocausal_{250,500,1000}ms.yaml`
   - 启动并完成 `250/500/1000ms` 三个 continuous non-causal 训练与 valid/test 评估
   - 产出 `results/logs/phase1_v2_nocausal_comparison/comparison.{json,md}`
   - 结果：
     - 250ms: causal 更好（test `fp-bps 0.2223 > 0.2178`）
     - 500ms: 基本持平（test `0.1740 < 0.1778`，差值很小）
     - 1000ms: non-causal 更好（test `0.1348 < 0.1375`）
4. 已同步更新：
   - `cc_core_files/plan.md`
   - `cc_core_files/results.md`
   - `cc_core_files/scripts.md`
   - `cc_todo/20260316-review/20260316-plan-md-v2-code-review_codex.md`
   - `cc_todo/phase1-autoregressive/20260318-phase1-1.3.5-ibl-metric.md`
   - `cc_todo/phase1-autoregressive/20260318-phase1-1.3.6-nocausal-ablation.md`

**关键判断**：
1. `ibl_mtm_bps` 与主 `fp-bps` 对窗口排序保持一致，但在 1000ms 上下降更剧烈，更适合作为 IBL-MtM 对齐 comparison metric，而不是替代主指标
2. causal mask 对 baseline_v2 的收益是窗口相关的：
   - 250ms 有明显正收益
   - 500ms 几乎无差别
   - 1000ms 则 non-causal 更优
3. 因此 baseline_v2 的优势不能简单归因于 causal mask；若后续论文要强调 causal ordering 的作用，需要把结论限定在短窗口，或与更明确的 state / memory 设计一起讨论

**遇到的问题与解决**：
- non-causal 三组重跑总时长较长，因此先做了中间提交 `a622359` / `3a87d7f`，再后台启动训练和对比脚本，完成后回填正式文档与结果

## 2026-03-18 12:44

**任务**：补强 1.8.3 三条 faithful 复现线的审计与执行文档，系统整理当前困难、关键适配设计、工程妥协、负结果原因分层，以及后续 250ms gate 优先的复现安排

**完成内容**：
1. 在 `cc_todo/20260316-review/1.8.3-benchmark-audit_codex.md` 新增：
   - 三模型各自的关键适配设计 / 主要困难 / 当前妥协 / 对指标的可能影响
   - 当前负结果原因的四层整理：已修掉实现 bug、fidelity/compatibility mismatch、benchmark-objective mismatch、工程 runtime blocker
   - 250ms gate-first 的后续复现安排与扩窗规则
2. 在 `cc_todo/phase1-autoregressive/20260312-phase1-1.8-benchmark.md` 新增：
   - 当前困难、关键适配设计与妥协总结
   - 当前负结果原因分层与后续任务重排
3. 同步更新：
   - `cc_core_files/results.md`
   - `cc_core_files/scripts.md`
   - `cc_core_files/plan.md`
   使结果页、脚本页与计划页都收口到同一结论：
   - NDT2：实现已打通，但更像 objective mismatch
   - IBL-MtM：训练语义已修正，250ms full-data `multimask_e1` 仍显著为负
   - Neuroformer：双模式语义已对齐，但 250ms formal full-data dual-mode eval 仍被 runtime blocker 卡住

**关键判断**：
1. 当前三个 faithful runner 的主要问题已不再是“是不是还在用 simplified wrapper”，而是“faithful 后是否仍适合当前 canonical benchmark”
2. NDT2 / IBL-MtM / Neuroformer 的负结果不应再混成一个问题：  
   - NDT2 更像 objective mismatch  
   - IBL-MtM 需要区分 training underfit 与 benchmark mismatch  
   - Neuroformer 先有明确的 runtime blocker
3. `500ms / 1000ms` 现在不应被继续默认排队；必须先以各自 `250ms` gate 为前提逐模型放行

**遇到的问题与解决**：
- 现有文档里已有多轮 debug、protocol-fix、faithful bridge 与 strict review 记录，信息容易重复甚至口径冲突  
  这次改成了“审计页负责批判性归因、执行页负责阶段状态、结果页和计划页只保留 gate 结论”的分工，避免后续继续漂移
- 三模型当前问题性质不同，如果只写“结果不好”会导致后续任务仍旧被排成同一种长窗口重跑  
  已明确拆成 objective mismatch、partial fidelity compromise 和 runtime blocker 三类，并据此重排后续任务

## 2026-03-18 16:43

**任务**：补写 `cc_core_files/plan.md` 的 `1.2.5` 与 `1.3.7`，把 baseline_v2 review 中已经落地的评估协议修正写回计划正文，并将 NeuroHorizon 后续实验的数据 / sampler / 指标口径收口为默认标准

**完成内容**：
1. 在 `plan.md` 新增 `1.2.5`：
   - 补记 baseline_v2 已落地的两项关键协议修正：
     - continuous validation / test 改用 `SequentialFixedWindowSampler`
     - continuous 主 `fp-bps` / `R-squared` 改为全局累计版
   - 进一步明确主 `fp-bps` 的正式口径是 `global spike-weighted + train-split null`
2. 在 `plan.md` 新增 `1.3.7`：
   - 固定 NeuroHorizon 默认数据协议：`train / valid / test` split 与 continuous / trial-aligned 的 sampler 选择
   - 固定主指标、comparison metric 与 trial 指标的默认解释方式
   - 指明正式表格应优先引用 `eval_phase1_v2.py` 产出的离线累计结果
   - 明确 `1.3.5` 与 `1.3.6` 已按最新 continuous 主协议执行
3. 新增任务记录：
   - `cc_todo/phase1-autoregressive/20260318-phase1-1.2.5-1.3.7-plan-standard.md`

**执行结果**：
- `plan.md` 中 baseline_v2 的已完成修正与后续默认标准现在已经分开表述，后续引用不会再把 review 意见、已落地修正和 comparison metric 混成一段
- 当前主指标继续维持为 `global spike-weighted fp-bps + train-split null`
- `ibl_mtm_bps` 继续保留为 IBL-MtM 风格 comparison metric，不替代主 `fp-bps`

**遇到的问题与解决**：
- review 文档中的建议编号与本次补写重点容易混淆  
  直接按已经落地的事实项写回 `plan.md`，不沿用建议编号，避免之后再出现“建议 2 / 建议 3”与正文编号不一致的问题
- 需要让 `1.8.x` benchmark 对比可以引用这套标准，但不能误写成 faithful baseline 已完全统一到 NeuroHorizon 协议  
  在 `1.3.7` 中明确写成“NeuroHorizon 结果默认遵循本标准；external benchmark 若协议不同，必须单列说明”

## 2026-03-18 17:29

**任务**：继续调整 `plan.md` 的 `1.3.7`，将默认协议进一步收紧为“continuous + 主 `fp-bps`”，并补充 `per-bin fp-bps` 在长时程 decay 分析中的定位与实现摘要

**完成内容**：
1. 修改 `1.3.7` 的“默认数据与 dataloader 协议”：
   - continuous 保留为默认实验协议
   - trial-aligned 改成仅在 hold/reach 对齐分析时按需启用，不再写成默认流程
2. 修改 `1.3.7` 的“默认指标与聚合协议”：
   - 默认正式结果只使用主 `global spike-weighted fp-bps + train-split null`
   - `R-squared`、`ibl_mtm_bps`、`per_neuron_psth_r2` 改为按需启用的可选补充指标
3. 在 `1.3.7` 中补充 `per-bin fp-bps`：
   - 明确其主要用途是 long-horizon prediction decay 比较
   - 简要写明其实现依赖 `fp_bps_per_bin_stats()` 与 `finalize_fp_bps_per_bin_from_stats()` 的逐 bin 严格累计
   - 说明正式曲线默认引用 `eval_phase1_v2.py` 的离线累计结果，而不是训练期 batch 级 logger
4. 同步更新任务记录：
   - `cc_todo/phase1-autoregressive/20260318-phase1-1.2.5-1.3.7-plan-standard.md`

**执行结果**：
- `1.3.7` 现在已经更明确地区分默认协议与可选分析协议
- 以后未特别说明的 NeuroHorizon 实验，默认理解为 continuous + 主 `fp-bps`
- `per-bin fp-bps` 被保留为长时程衰减分析工具，而不是默认主榜单指标

**遇到的问题与解决**：
- 需要保留 trial-aligned 与其他指标的价值，但又不能让它们在文字上与默认协议并列  
  已改成“按需启用的可选补充口径”，并把默认主线收紧为 continuous + 主 `fp-bps`


## 2026-03-18 18:35

**任务**：补做 1.8 faithful benchmark 详细审计，并同步清理 plan.md / results.md / 20260312-phase1-1.8-benchmark.md 中仍残留的旧 benchmark 强结论

**完成内容**：
1. 新增详细审计文档：
   - cc_todo/20260318-review/20260318-benchmark-faithful-audit-detail_codex.md
2. 新增本次任务记录：
   - cc_todo/phase1-autoregressive/20260318-phase1-1.8-faithful-audit-detail.md
3. 更新 cc_core_files/plan.md：
   - 收紧 1.8 的实验目的和核心标准
   - 在 1.3.4 中把 benchmark 对比回收到 legacy protocol-fix internal reference，不再写成正式公平 benchmark
4. 更新 cc_core_files/results.md：
   - 将旧 1.8 benchmark 主结果改写为 legacy simplified baselines 的内部参考
   - 将 1.3.4 与 benchmark 的对照改写为 internal reference，而不是正式 benchmark 结论
   - 收紧 1.4 / 1.5 中引用 benchmark 列的解释强度
5. 更新 cc_todo/phase1-autoregressive/20260312-phase1-1.8-benchmark.md：
   - 明确 2026-03-12 的 Part A-C 仅保留为 legacy 历史记录
   - 追加 detailed faithful 审计索引和当前优先级

**执行结果**：
- 当前 1.8 的文档叙事已经统一收口到三条线：legacy simplified baselines、protocol-fix internal reference、faithful 250ms gate
- 文档中不再继续把旧 simplified baseline 写成 “Neuroformer 是最强竞争者” 或 “NeuroHorizon 已正式优于 benchmark” 这类强结论
- 对 IBL-MtM / Neuroformer 的当前判断已经收口为：理论可做、桥接已通，但 formal benchmark 尚未完成

**遇到的问题与解决**：
- plan.md 与 results.md 中旧 benchmark 结论和后面的 faithful gate 状态互相冲突  
  通过统一改写成 legacy internal reference，并把正式结论全部回收到 faithful 250ms gate
- 1.3.7 的统一标准容易被误读成训练和评估都必须与 NeuroHorizon 完全同构  
  在新审计中明确区分：统一的是 split / continuous eval semantics / main metric，不是强制统一上游训练语义

2026-03-19 02:02 CST

针对 cc_todo/20260318-review/20260318-benchmark-faithful-audit-detail_codex.md 进行了定点补充解释，重点收紧并展开了 IBL-MtM 与 Neuroformer 的 faithful 审计表述。

1. 更新 cc_todo/20260318-review/20260318-benchmark-faithful-audit-detail_codex.md：
   - 展开解释 IBL-MtM 中“metadata 生态比 tensor shape 更重要”的原因
   - 明确当前 Perich-Miller 上 combined mask 实际退化为 neuron + causal，以及这与 canonical forward prediction 的关系
   - 修正 Neuroformer true_past 的表述，改为“语义已接通，但 faithful bridge 没有直接复用上游 simulation 脚本”
   - 展开说明 token-level generation 与 count-based fp-bps / PSTH 指标之间为何不是无损等价
   - 新增 full-data dual-mode held-out generation runtime 成本来源分析

**执行结果**：
- review 文档现在已经能更严格地区分训练语义原生与评估语义统一
- IBL-MtM 的当前限制被明确收口为 metadata 生态不完整，而不再笼统写成 region 缺失
- Neuroformer 的当前限制被明确收口为 runtime / benchmark bridge 成本，而不再误写成 true_past 尚未真正接通

**遇到的问题与解决**：
- 原文中对 Neuroformer true_past 的描述与当前 faithful_neuroformer.py 实现不完全一致  
  通过复查 true_past / rollout 两条 held-out 代码路径后，将文档改写为“语义已接通、实现为 benchmark bridge 而非直接调用上游 simulation”
- IBL-MtM 的 metadata 缺口容易被误解为只有 region 一项  
  通过补充 session identity、eid 语义、region-conditioned task family 三层解释，收紧为更准确的 fidelity 判断

2026-03-19 02:54 CST

继续补充 cc_todo/20260318-review/20260318-benchmark-faithful-audit-detail_codex.md 中关于 1.8 benchmark 后续推进的分析，重点增加 NDT2 暂停、IBL-MtM 原因拆解、Neuroformer 原因拆解以及短窗口参考实验建议。

1. 更新 cc_todo/20260318-review/20260318-benchmark-faithful-audit-detail_codex.md：
   - 新增“接下来 benchmark 还要继续做什么”一节
   - 明确 NDT2 当前停止继续跟踪，只保留现状记录
   - 对 IBL-MtM 补充训练 mask geometry 与 eval held-out forward-pred 不完全对齐、from-scratch、session-pure batching、指标口径差异等原因判断
   - 对 Neuroformer 补充 train/eval 窗口本身一致但 token loss 与 count-based fp-bps 错位、from-scratch、缺少显式 session conditioning、runtime 成本来源等原因判断
   - 新增 Neuroformer 官方风格短窗口参考实验建议：`window.prev = 0.15`、`window.curr = 0.05`，先看 rollout / true_past 两种 inference 的 bps 作为方向性证据

**执行结果**：
- review 文档现在已经把 benchmark 后续优先级写成可执行顺序：NDT2 暂停、IBL-MtM 继续 250ms short formal run、Neuroformer 先解 250ms dual-mode runtime，并补短窗口 sanity run
- Neuroformer 的 `150ms observation + 50ms prediction` 已作为 reference sanity experiment 写入文档，但明确不替代当前 canonical benchmark

**遇到的问题与解决**：
- 需要确认 `150ms / 50ms` 是否只是用户回忆，还是上游 repo 里确有代码依据  
  复查后确认 `benchmark_models/neuroformer/configs/V1AL/mconf.yaml` 中确有 `window.prev = 0.15`、`window.curr = 0.05`，因此文档改写为“官方风格短窗口参考实验”而不是无依据猜测
- IBL-MtM / Neuroformer 的“窗口是否不一致”容易被误写成主因  
  复查 faithful runner 后确认 train/eval 使用同一 spec，文档改为“窗口秒数一致，真正 mismatch 在训练目标 / mask geometry / token-count bridge”


2026-03-19 03:40 CST

开始执行 1.8 faithful benchmark 审计文档中 7.4 的实际落地工作，先完成 runner 改造、小规模验证和文档跟踪入口。

1. 更新 faithful benchmark 代码入口：
   - `neural-benchmark/faithful_ibl_mtm.py`
     - 新增 `train_mask_mode=forward_pred`
     - 新增 `train_protocol` 字段，显式标注 train/eval mask geometry 是否 `exact / partial` 对齐
   - `neural-benchmark/faithful_neuroformer.py`
     - 新增 `mode=eval`
     - 新增 `--checkpoint-path`、`--eval-split`、`--inference-mode`、`--skip-trial-eval`、`--progress-every`
     - rollout 评估不再额外做冗余 teacher-forced forward
     - 新增 `token_stats`（prev/curr tokens mean/p95/max + truncation_rate）
   - 新增对比脚本：`compare_faithful_ibl_mtm.py`、`compare_faithful_neuroformer.py`

2. 完成小规模验证：
   - IBL-MtM `forwardpred_smoke_v2`
     - `bridge_config.train_mask_mode = forward_pred`
     - `train_masking_mode = forward_pred`
   - IBL-MtM `250ms_forwardpred_smoke_e1`
     - best valid `fp-bps = -7.3364`
     - held-out test `fp-bps = -8.4518`
     - `history[0].train_mask_counts = {"forward_pred": 4}`
   - Neuroformer `eval_smoke_v1`
     - valid rollout `fp-bps = -11.6581`, `elapsed_s = 1.4330`
     - valid true_past `fp-bps = -11.1578`, `elapsed_s = 0.2153`
     - 当前 2 个 valid windows 下 `prev/curr truncation_rate = 0.0 / 0.0`

3. 更新跟踪文档：
   - `cc_todo/20260318-review/20260318-benchmark-faithful-audit-detail_codex.md`
     - 新增 `8. 7.4 执行计划（2026-03-19）`
     - 新增 `9. 7.4 执行进展`
   - `cc_todo/phase1-autoregressive/20260312-phase1-1.8-benchmark.md`
     - 新增 `4.18 7.4 执行入口与小规模验证`

**当前状态**：
- runner 层改造和小验证已完成，准备先做一轮中间提交
- 下一步固定顺序：
  1. IBL-MtM `250ms combined_e10`
  2. IBL-MtM `250ms forwardpred_e10`
  3. Neuroformer `250ms` formal dual-mode eval
  4. Neuroformer `150ms observation + 50ms prediction` reference run

**遇到的问题与解决**：
- 远程环境没有 `apply_patch`，因此改为使用远程 `python3` 做精确文件改写
- IBL-MtM 的 `smoke` 入口最初未透传 `--train-mask-mode`，导致首次 smoke 仍走默认 `combined`
  - 已修正 `run_smoke()`，当前 smoke 与小训练都已真实进入 `forward_pred` 路径
- Neuroformer 的正式 runtime blocker 当前先通过 `eval-only` 模式拆解，而不是继续把 train 和 dual-mode full-data eval 强耦合在一起


2026-03-19 06:45 CST

完成 1.8 faithful benchmark 7.4 正式批次，并把结果回填到 review / task log / plan / results / scripts 文档。

1. IBL-MtM `250ms short formal run`
   - `combined_e10`: best valid / test `fp-bps = -0.0026 / -0.0017`, trial `fp-bps = 0.0396`, `per_neuron_psth_r2 = 0.4559`
   - `forwardpred_e10`: best valid / test `fp-bps = -2.0014 / -1.9843`, trial `fp-bps = -2.2245`
   - 结论：exact future-window `forward_pred` control 明显差于 upstream `combined`；train/eval mask geometry mismatch 不是当前 IBL-MtM 的主因

2. Neuroformer `250ms` formal dual-mode eval
   - rollout test `fp-bps = -8.8025`, `elapsed_s = 1063.0504`
   - true_past test `fp-bps = -9.3982`, `elapsed_s = 77.2959`
   - token stats 显示 `prev/curr truncation_rate = 0 / 0`
   - 结论：formal eval blocker 已解除，但性能本身仍显著为负，且不能归因于 truncation

3. Neuroformer `150ms observation + 50ms prediction` reference run
   - rollout / true_past test `fp-bps = -8.0744 / -8.9540`
   - 相比 canonical `500/250` 仅小幅改善，仍显著为负
   - 结论：问题不只是 horizon 太长，当前更像 `from-scratch + token/count mismatch + session conditioning不足`

4. 文档同步：
   - `cc_todo/20260318-review/20260318-benchmark-faithful-audit-detail_codex.md`
   - `cc_todo/phase1-autoregressive/20260312-phase1-1.8-benchmark.md`
   - `cc_core_files/plan.md`
   - `cc_core_files/results.md`
   - `cc_core_files/scripts.md`

**当前建议**：
- `NDT2` 继续暂停
- `IBL-MtM` 若继续，只保留 `combined` 路线，考虑 `e20/e30` 或更接近上游预训练/metadata 的方案
- `Neuroformer` 当前不建议进入 500ms / 1000ms 扩展


2026-03-19 15:56 CST

按 `cc_core_files/plan.md` 1.9.0 Step 4 补齐了 Phase 1.9 四轮模块优化迭代的 training curves 可视化，并把脚本、figure、任务记录和汇总文档同步到对应位置。

1. 新增脚本：
   - `scripts/phase1-autoregressive-1.9-module-optimization/plot_optimization_training_curves.py`
   - 功能：从各模块 summary JSON 反查正式 checkpoint 对应的 `metrics.csv`，按 epoch 聚合 `train_loss / val_loss / val/fp_bps`，为每轮 1.9 迭代生成 `training_curves.{png,pdf}`

2. 新增 figure：
   - `results/figures/phase1-autoregressive-1.9-module-optimization/20260312_prediction_memory_decoder/training_curves.{png,pdf}`
   - `results/figures/phase1-autoregressive-1.9-module-optimization/20260313_local_prediction_memory/training_curves.{png,pdf}`
   - `results/figures/phase1-autoregressive-1.9-module-optimization/20260313_prediction_memory_alignment/training_curves.{png,pdf}`
   - `results/figures/phase1-autoregressive-1.9-module-optimization/20260313_prediction_memory_alignment_tuning/training_curves.{png,pdf}`

3. 文档同步：
   - `cc_core_files/plan.md`
     - 在 1.9.0 Step 4 中补充 training curves 必做要求
     - 在 1.9.2 四轮迭代条目中补充实际 `training_curves.png` 路径
   - `cc_core_files/scripts.md`
     - 登记新脚本 `plot_optimization_training_curves.py`
   - `cc_core_files/results.md`
     - 为四轮正式结果补充 `training_curves.{png,pdf}` 存储路径和曲线解读
   - `cc_todo/phase1-autoregressive/1.9-module-optimization/*.md`
     - 分别追加 2026-03-19 的 training curves 可视化记录、数据来源、执行命令和观察结论

4. 可视化结论摘要：
   - `20260312_prediction_memory_decoder` 与 `20260313_local_prediction_memory`
     - loss 曲线稳定收敛，但 `val/fp_bps` 中后期较早平台化，且 post-train rollout 参考线仍显著落后于 teacher-forced 水平
   - `20260313_prediction_memory_alignment`
     - 三窗口 `val/fp_bps` 持续抬升到训练后期，长窗口改善最明显
   - `20260313_prediction_memory_alignment_tuning`
     - `250ms / 1000ms` 末段曲线相对上一轮略有上移，`500ms` 基本持平，和正式结果中的“小幅继续推进”一致

**遇到的问题与解决**：
- 1.9 各模块目录里存在多个 `lightning_logs/version_*`，直接凭目录序号引用容易写错正式 run 的来源
  - 解决：统一从各模块 `*_summary.json` 的 checkpoint 字段反查正式 run 对应的 `metrics.csv`
- 原有 1.9 结果已经有 summary JSON 和总趋势图，但缺少 epoch-level training curves
  - 解决：新增统一绘图脚本，按 Step 4 要求把脚本、figure、结果记录和汇总文档一次性补齐


2026-03-19 15:35 CST

完成 Phase 1.9 模型优化超参数审查，并把结论写回 `model.md` 和新的 1.9 超参分析文档。

1. 审查 `baseline_v2` 与四轮 1.9 formal 实验的训练/评估超参
   - 确认四轮 1.9 都复用 `examples/neurohorizon/train.py` 的主训练逻辑：`PoissonNLLLoss`、`SparseLamb`、`OneCycleLR(cos)`、`base_lr=3.125e-5`、`weight_decay=1e-4`
   - 确认 `20260312_prediction_memory_decoder` 与 `20260313_local_prediction_memory` 在 `250/500/1000ms` 上都与 baseline_v2 保持 batch/lr 口径一致
   - 发现 `20260313_prediction_memory_alignment` 与 `20260313_prediction_memory_alignment_tuning` 的 `1000ms` 配置改成 `batch_size=64`，相对 baseline_v2 `batch_size=32` 把 `max_lr` 从 `0.001` 提高到 `0.002`

2. 审查 1.9 与 baseline_v2 的对比口径
   - 确认 1.9 汇总脚本仍读取 `results.tsv` 中的 legacy baseline_v2 行（`0.2115 / 0.1744 / 0.1317`）
   - 确认 1.9 批量评估脚本调用 `eval_phase1_v2.py --skip-trial`，未显式传 `--split test`，因此当前 1.9 主表本质上是 continuous-valid rollout 对比，不是 current evalfix test 主结论

3. 文档更新
   - `cc_core_files/model.md`：在 v2 baseline 和四轮 1.9 模块下补充训练超参、batch/lr 口径与 baseline_v2 对比说明
   - `cc_todo/phase1-autoregressive/1.9-module-optimization/20260319_hyperparameter_audit.md`：新增系统审查文档，整理一致性结论、current valid/test 差值和后续优化建议

**当前建议**：
- 若要继续推进 1.9，先处理 `alignment / tuning` 的 `1000ms` batch/lr 公平性，再补 current evalfix valid/test 的 rollout 评估
- 超参优化优先级建议放在 `500ms` 的 `mix_prob / dropout / noise`，而不是立即做大范围 optimizer sweep


2026-03-19 16:05 CST

完成分支同步收口：将 dev/benchmark 的当前状态整理后同步到本地 main 并推送远程。

1. 在 dev/benchmark 上补充本次分支同步记录
2. 确认 main 可通过 fast-forward 同步到当前 dev/benchmark
3. 推送更新后的 dev/benchmark 与 main 到 GitHub 远程

2026-03-19 17:20 CST

根据最新主线分支同步状态，补充并收紧了 `cc_core_files/plan.md` 中 `1.9.0 执行规范` 的文档要求，重点把增量模型优化流程中的记录规范、默认分支策略、实验协议确认和结果沉淀要求写得更明确。

1. 更新 `cc_core_files/plan.md`：
   - 在 Step 1 中新增“实现方案”和“涉及改动模块”的记录要求
   - 在 Step 2 中明确：若用户未指定分支，默认创建 `dev/{date}_{module_name}`
   - 在 Step 3 中新增：实验前必须确认是否遵循 1.3.7；若不一致必须书面记录；默认至少记录 `fp-bps` 和 `per-bin fp-bps`
   - 在 Step 4 中新增：任务记录必须包含关键超参数、最佳 `val fp-bps`、test `fp-bps`、test checkpoint 标识/时间；可视化除 training curves 外，还要补充窗口级 `fp-bps` 趋势图、各窗口 `per-bin fp-bps` 衰减曲线，以及表格型 PNG 汇总图
   - 在 Step 4 中进一步明确：`cc_todo/phase1-autoregressive/1.9-module-optimization/results.tsv` 至少要记录各窗口的最佳 `val fp-bps`、test `fp-bps`、test checkpoint 标识/时间和必要备注

**执行结果**：
- `1.9.0` 现在已经把想法记录、默认建分支、协议确认、关键指标、超参数、结果表和图表沉淀要求统一写清
- 后续 1.9.x 模型优化任务在立项、实施、实验和总结阶段的最小记录标准更加明确，减少遗漏关键信息的风险

**遇到的问题与解决**：
- 当前 `main`、`dev/benchmark`、`dev/latent`、`dev/diffusion` 的远端同步关系刚发生变化，容易导致“是 cherry-pick 还是直接快进”判断混乱
  先重新 fetch 并核对四个分支的 commit 关系，再按最新状态执行“先在 main 修改，再把三个开发分支直接 forward 到 main”的策略


## 2026-03-19 23:48 Benchmark 文档体系重构

- 新建 benchmark 文档根目录：`cc_todo/1.8-benchmark_model/`
- 新增总入口：`cc_todo/1.8-benchmark_model/benchmark_index.md`
- 迁移文档：
  - `cc_todo/20260316-review/1.8.3-benchmark-audit_codex.md` -> `cc_todo/1.8-benchmark_model/20260316_benchmark_legacy_audit_codex.md`
  - `cc_todo/20260318-review/20260318-benchmark-faithful-audit-detail_codex.md` -> `cc_todo/1.8-benchmark_model/20260318_benchmark_faithful_audit_detail_codex.md`
  - `cc_todo/phase1-autoregressive/20260312-phase1-1.8-benchmark.md` -> `cc_todo/1.8-benchmark_model/20260312_benchmark_main_task_log.md`
  - `cc_todo/phase1-autoregressive/20260318-phase1-1.8-faithful-audit-detail.md` -> `cc_todo/1.8-benchmark_model/20260318_benchmark_faithful_audit_task_log.md`
- `plan.md` 的 `1.8.3` 已重写为 benchmark 持续维护规范，实验执行/记录默认复用 `1.9.0 Step 3 / Step 4`，但路径统一改为 `1.8-benchmark_model`
- 新增当前 aligned 长跑记录：`cc_todo/1.8-benchmark_model/20260319_benchmark_aligned_runs.md`
- 说明：`progress.md` 历史日志保留旧路径原文；自此以后 benchmark 文档统一迁移到 `cc_todo/1.8-benchmark_model/`

## 2026-03-20 01:40 CST

### 任务：修正 1.9 checkpoint 选择与正式评估链路，并回填历史 best-ckpt valid/test 结果

**完成内容**：
1. 查明历史 1.9 run 中 `last.ckpt` 不是 final epoch 的原因
   - 核对 `examples/neurohorizon/train.py` 的原始 `ModelCheckpoint(save_last=True, monitor="val_loss", save_on_train_epoch_end=True)` 配置
   - 确认 Lightning 只会在同一步实际保存 monitored checkpoint 时刷新 `last.ckpt`
   - 因此历史目录里的 `last.ckpt` 实际等同于某个更早 epoch 的 monitored checkpoint，而不是训练结束时的 final weights

2. 修改训练与评估实现
   - `examples/neurohorizon/train.py`
     - 验证期额外记录 `val_fp_bps`
     - 每个 eval epoch 显式保存 checkpoint
     - train end 按 `max(val/fp_bps)` + `min(val_loss)` 选出 `best.ckpt`
     - train end 同时保存真正的 final `last.ckpt`，并写出 `checkpoint_summary.json`
   - `scripts/analysis/neurohorizon/eval_phase1_v2.py`
     - 新增 `--checkpoint-kind {best,last}`
     - 显式支持解析 `best.ckpt` / `last.ckpt`
     - 输出 checkpoint epoch / global_step 等元信息

3. 修改 1.9 批量实验与汇总逻辑
   - 四个 `run_*_experiments.sh` 统一改为：
     - 训练后对 best ckpt 跑 teacher-forced `valid/test`
     - 对 best ckpt 跑 rollout `valid/test`
   - 新增：
     - `scripts/phase1-autoregressive-1.9-module-optimization/module_result_utils.py`
     - `scripts/phase1-autoregressive-1.9-module-optimization/backfill_best_checkpoint_evals.py`
   - 四个 `collect_*_results.py` 改为共用统一汇总逻辑，自动刷新：
     - `results.tsv`
     - `*_summary.json`
     - `optimization_progress.{png,pdf}`
     - `training_curves.{png,pdf}`

4. 完成历史 1.9 四轮实验的 best-ckpt 回填
   - `20260312_prediction_memory_decoder`
     - rollout-valid：`0.1510 / 0.0200 / -0.2192`
   - `20260313_local_prediction_memory`
     - rollout-valid：`0.1679 / 0.0316 / -0.1749`
   - `20260313_prediction_memory_alignment`
     - rollout-valid：`0.1904 / 0.1623 / 0.1120`
   - `20260313_prediction_memory_alignment_tuning`
     - rollout-valid：`0.1949 / 0.1635 / 0.1264`
   - `results.tsv` 已新增：
     - `best_val_fp_bps_*`
     - `best_test_fp_bps_*`
     - `best_ckpt_*`

5. 文档回写
   - `cc_core_files/plan.md`
     - 在 `1.9.0 Step 3 / 1.必做` 中补充数据配置、train/eval 入口、continuous sampler 约定、checkpoint 审计结论和四轮 best-ckpt valid/test 回填结果
   - `cc_core_files/results.md`
     - 新增 `2026-03-20 协议审计与 best-ckpt 回填` 总表
   - `cc_core_files/scripts.md`
     - 登记 `module_result_utils.py`、`backfill_best_checkpoint_evals.py`
     - 更新 1.9 run/collect 脚本说明为 best-ckpt `valid/test` 口径
   - `cc_todo/phase1-autoregressive/1.9-module-optimization/*.md`
     - 四个模块任务记录均追加 checkpoint 审计结论和 best-ckpt valid/test 表格

**遇到的问题与解决**：
- 历史 1.9 目录里只保留了一个 monitored checkpoint 文件，无法回溯“每个 epoch 的真实 best-ckpt 文件集合”
  - 解决：明确标记为“historical available best-ckpt eval”，统一使用目录中实际保留下来的 monitored checkpoint 回填 `valid/test`
- 旧 `results.md` 和 `plan.md` 中的 1.9 数值是按历史 `post-train` 口径写的，与本次 best-ckpt 回填结果不一致
  - 解决：在 `results.md` 增加权威总表，并在 `plan.md` Step 3 中写明以 2026-03-20 回填表和 `results.tsv` 为准

## 2026-03-20 02:09 CST

### 任务：收紧 1.9.0 Step 3 文档口径

**完成内容**：
1. 调整 `cc_core_files/plan.md`
   - 删除 `1.9.0 Step 3 / 1.必做` 下的 `2026-03-20 协议审计补充`
   - 删除 `2026-03-20 历史 1.9 best-ckpt 回填结果`
   - 删除 `详细落点`
   - 保留实现规则：train end 按 `max(val/fp_bps)` + `min(val_loss)` 选 best ckpt，并对 best ckpt 输出 `valid/test` 指标

**遇到的问题与解决**：
- 无额外实现问题，本次仅调整文档层级和信息密度

## 2026-03-20 02:31 CST

### 任务：补 1.9 超参数审查中的 POYO+ 对照口径，并给最优 1.9 分支补 current evalfix valid/test

**完成内容**：
1. 扩充 `cc_todo/phase1-autoregressive/1.9-module-optimization/20260319_hyperparameter_audit.md`
   - 新增 `examples/poyo_plus/defaults.yaml`、`train_baseline_10sessions.yaml` 与 `baseline_v2 / 1.9` 的训练规程对照
   - 明确记录：1.9 继承的是 adapted-POYO+ / baseline_v2 的优化主干，而不是 raw POYO+ defaults
   - 将第 6 节中的“补文档口径”和“给最优 1.9 checkpoint 补 valid/test evalfix”标记为已执行

2. 更新 `cc_core_files/model.md`
   - 在 v2 baseline 小节补充 `poyo_plus -> adapted poyo_plus -> baseline_v2 / 1.9` 的训练规程 lineage
   - 在 `20260313_prediction_memory_alignment_tuning` 小节补充 current evalfix `valid/test` 和 trial 指标
   - 明确写出：current continuous test `fp-bps` 仍然三窗口全部低于 baseline_v2，`1000ms` 的“接近 baseline”仍受 batch/lr parity 污染

3. 补跑 `20260313_prediction_memory_alignment_tuning` 的 rollout evalfix `valid/test`
   - 新生成：
     - `250ms/eval_rollout_evalfix_{valid,test}.json`
     - `500ms/eval_rollout_evalfix_{valid,test}.json`
     - `1000ms/eval_rollout_evalfix_{valid,test}.json`
   - 关键结果：
     - valid `fp-bps`：`0.1949 / 0.1635 / 0.1264`
     - test `fp-bps`：`0.1991 / 0.1637 / 0.1273`
     - 相对 baseline_v2 test 差值：`-0.0232 / -0.0104 / -0.0075`
     - test `PSTH-R²`：`0.6785 / 0.6210 / 0.5738`

4. 更新 `cc_core_files/results.md`
   - 新增 `2026-03-20 最优 1.9 分支 evalfix valid/test 补评估` 小节
   - 记录 current evalfix `valid/test` continuous 与 trial-aligned 结果，并补上与 baseline_v2 的差值解释

**遇到的问题与解决**：
- `20260313_prediction_memory_alignment_tuning` 新生成的 evalfix JSON 与旧 `eval_rollout.json` schema 不同，字段是 `continuous / trial_aligned` 而不是旧脚本里常见的 `metrics`
  - 解决：先核对 JSON schema，再统一按 `continuous['fp_bps'] / trial_aligned['per_neuron_psth_r2']` 读取，避免把新旧格式混用
- `baseline_v2` 的 current evalfix 结果分散在 `phase1_v2_evalfix_{window}_{cont,trial}` 目录
  - 解决：分别从 continuous 和 trial 目录读取 `eval_v2_{valid,test}_results.json`，再与 1.9 tuning 的 current evalfix 结果逐窗口对齐比较

## 2026-03-20 02:37 CST - 1.8.3 benchmark 协议与 formal eval 规范收口

**完成事项**：
1. 更新 `cc_core_files/plan.md` 的 `1.8.3`
   - 不再写“复用 1.9.0 Step 3/4”，直接写入 benchmark 的实验执行规范和实验记录规范
   - 明确 continuous benchmark 的 `train/valid/test`、sampler 语义、best ckpt 选择、best-ckpt formal `valid/test` 结果要求
   - 明确 `IBL-MtM / Neuroformer` 主流程不再要求 `test trial-aligned`
   - 明确 Neuroformer 默认按 `valid rollout fp-bps` 选择 `best_model.pt`
   - 在 `1.9.0 Step 4` 中补充“记录每次训练和评估的脚本命令”
2. 更新 benchmark 文档入口与当前任务记录
   - `cc_todo/1.8-benchmark_model/benchmark_index.md` 新增协议约束、固定脚本入口和命令记录要求
   - `cc_todo/1.8-benchmark_model/20260319_benchmark_aligned_runs.md` 补充当前 `IBL-MtM / Neuroformer` 的训练命令、正式 eval 命令和最新进度时间戳
3. 更新 faithful benchmark runner 的正式结果口径
   - `neural-benchmark/faithful_ibl_mtm.py`：训练结束后加载 `best_model.pt` 重新计算 continuous `valid/test`，写入 `formal_valid_metrics`，不再把 `test trial-aligned` 作为默认结果输出
   - `neural-benchmark/faithful_neuroformer.py`：默认 `skip_trial_eval=True`；训练结果和 eval-only 统一到 best-ckpt continuous `valid/test`，Neuroformer 继续按 `valid rollout fp-bps` 选 best
   - `neural-benchmark/compare_faithful_ibl_mtm.py` 同步改为对比 `best valid / formal valid / test`，不再以 trial 指标为主

**当前运行状态**：
- `screen` 会话 `phase1_benchmark_aligned` 仍在运行
- `IBL-MtM combined_e50_aligned` 已完成并保持正值结果
- `Neuroformer canonical 500/250 e50 aligned` 仍在训练；最近一次 `last_model.pt` 更新时间为 `2026-03-20 02:35:02 +0800`
- 本次文档与 runner 修改未中断当前 benchmark 长跑

**说明与判断**：
- 当前本地克隆的 Neuroformer 上游 README / trainer 代码显示其原生训练器按 holdout loss 保存 best checkpoint，未见按 `fp-bps` 或 `true_past` 选 ckpt 的固定说明
- 在当前 held-out forward prediction benchmark 目标下，继续采用 `valid rollout fp-bps` 作为 Neuroformer 的 checkpoint selection 更合理；`true_past` 保持为 oracle-history 诊断指标

## 2026-03-20 05:20 CST

### 任务：建立 1.10 latent dynamics decoder 主线并完成实现阶段 smoke 验证

**完成内容**：
1. 更新 `cc_core_files/plan.md`
   - 新增 `1.10 Latent Dynamics Decoder` 小节
   - 固定 `1.10.0` 的执行规范、路径体系、默认分支与入口说明
   - 新增首轮任务索引 `20260320_latent_dynamics_decoder`

2. 新建 `1.10` 专用文档与结果索引
   - `cc_todo/1.10-latent_dynamics_decoder/model.md`
   - `cc_todo/1.10-latent_dynamics_decoder/20260320_latent_dynamics_decoder.md`
   - `cc_todo/1.10-latent_dynamics_decoder/results.tsv`

3. 实现 latent dynamics 主线
   - 新增 `torch_brain/nn/latent_dynamics_decoder.py`
   - 修改 `torch_brain/models/neurohorizon.py`
   - 修改 `torch_brain/nn/__init__.py`
   - 新增 `train_1p10_latent_dynamics_{250,500,1000}ms.yaml`
   - 新增 `scripts/1.10-latent_dynamics_decoder/` 下的 verify / smoke / train / monitor / collect / plotting 脚本

4. 完成最小功能验证
   - `verify_latent_dynamics.py` 通过
   - 输出：
     - `output_shape=(2, 12, 6)`
     - `tf_vs_rollout_max_delta=0.000000`

5. 完成 250ms smoke run
   - 训练日志：`results/logs/1.10-latent_dynamics_decoder/20260320_latent_dynamics_decoder/250ms_smoke/`
   - 关键结果：
     - train loss：`0.417`
     - val loss：`0.406`
     - `val/fp_bps=-0.834`
     - 离线 continuous valid：`fp-bps=-0.8339`, `R2=-0.0021`, `val_loss=0.4079`

6. 更新登记文档
   - `cc_core_files/scripts.md`
   - `cc_core_files/results.md`

**遇到的问题与解决**：
- 远程环境没有 `s4`、`mamba_ssm`、`torchdiffeq`
  - 解决：首轮实现改为无新增依赖的 GRU latent dynamics 主线，同时在 `1.10` 文档中明确将 Mamba 保留为后续 `1.10.x` 扩展位
- 远程机器没有 `apply_patch`
  - 解决：先把目标文件镜像到本地临时工作区，用补丁改完后再同步回远程仓库
- 由于 `1.10` 首轮不再依赖 future-count feedback，`forward()` 与 `generate()` 的语义重新回到同一路径
  - 解决：在功能验证脚本里显式校验 `tf_vs_rollout_max_delta=0`

## 2026-03-20 05:28 CST

### 任务：提交 1.10 实现检查点并启动三窗口正式训练

**完成内容**：
1. 提交并推送 `1.10` 实现检查点
   - 分支：`dev/latent`
   - commit：`32a1f5f`
   - 提交信息：`实现1.10 latent dynamics decoder主线与首轮验证`

2. 启动 `250/500/1000ms` 三窗口正式训练
   - 后台会话：`screen -S latent_dynamics_1p10`
   - 主控脚本：`scripts/1.10-latent_dynamics_decoder/20260320_latent_dynamics_decoder/run_latent_dynamics_experiments.sh`
   - 日志文件：`results/logs/1.10-latent_dynamics_decoder/20260320_latent_dynamics_decoder/screen_run.log`

3. 完成启动后健康检查
   - `screen` 会话存在且处于 detached 状态
   - GPU 上已有三路 `python` 训练进程
   - 日志显示三个窗口均已进入 epoch 1–2，暂未出现 OOM 或配置错误

**当前运行状态**：
- `250ms / 500ms / 1000ms` 正式训练均在运行
- 当前仓库实现代码已推送到远端，后续可在训练完成后直接汇总正式结果

**遇到的问题与解决**：
- 正式训练时长较长，不适合占用当前交互终端
  - 解决：使用 `screen` 后台启动，并把完整终端输出写入 `screen_run.log`，便于后续追踪和恢复

## 2026-03-20 09:42 CST

### 任务：完成 1.10 首轮正式三窗口实验与结果回填

**完成内容**：
1. 完成 `250ms / 500ms / 1000ms` 三窗口正式训练与 best-ckpt `valid/test` 评估
   - `250ms`：valid `fp-bps=0.1882`, test `fp-bps=0.1966`, best epoch `289`
   - `500ms`：valid `fp-bps=0.0904`, test `fp-bps=0.0857`, best epoch `259`
   - `1000ms`：valid `fp-bps=0.0674`, test `fp-bps=0.0667`, best epoch `289`

2. 生成正式图表与汇总文件
   - `results/figures/1.10-latent_dynamics_decoder/20260320_latent_dynamics_decoder/training_curves.{png,pdf}`
   - `results/figures/1.10-latent_dynamics_decoder/optimization_progress.{png,pdf}`
   - `results/figures/1.10-latent_dynamics_decoder/20260320_latent_dynamics_decoder/latent_dynamics_summary.json`
   - 更新 `cc_todo/1.10-latent_dynamics_decoder/results.tsv`

3. 回填核心文档
   - 更新 `cc_core_files/results.md`
   - 更新 `cc_core_files/plan.md`
   - 更新 `cc_todo/1.10-latent_dynamics_decoder/model.md`
   - 更新 `cc_todo/1.10-latent_dynamics_decoder/20260320_latent_dynamics_decoder.md`

**当前结论**：
- 当前 GRU latent dynamics baseline 已证明该方向在仓库里可完整实现，但没有优于 `baseline_v2`
- `250ms` 已接近 baseline，说明 latent rollout 有可行性
- `500ms / 1000ms` 明显落后，且 `best epoch` 已出现在 `259–289`，说明这轮的主要问题不是训练轮数不足，而是当前 latent state 压缩和 dynamics 表达能力不够

**遇到的问题与解决**：
- `collect_latent_dynamics_results.py` 已自动写入 `results.tsv`，但图表和结论文档仍需手动补全
  - 解决：补跑绘图脚本，并把 `best epoch / final epoch / baseline delta` 统一写回 `results.md`、`plan.md` 和 `1.10` 任务文档

## 2026-03-20 14:57 CST

### 任务：启动 1.10.x 的 latent dynamics state scaling 500ms gate

**完成内容**：
1. 实现更大 latent state 的 decoder 接口
   - `LatentDynamicsDecoder` 新增 `pool_token_dim / state_dim`
   - `NeuroHorizon` 新增对应配置入口

2. 新建 `500ms` gate 的配置、脚本与任务文档
   - `cc_todo/1.10-latent_dynamics_decoder/20260320_latent_dynamics_state_scaling.md`
   - `examples/neurohorizon/configs/model/neurohorizon_latent_dynamics_state_scaling_500ms.yaml`
   - `examples/neurohorizon/configs/train_1p10_latent_dynamics_state_scaling_500ms.yaml`
   - `scripts/1.10-latent_dynamics_decoder/20260320_latent_dynamics_state_scaling/`

3. 完成功能验证与 smoke 验证
   - verify 输出：`output_shape=(2, 25, 6)`, `tf_vs_rollout_max_delta=0.000000`
   - `500ms` smoke：train loss `0.412`, val loss `0.393`, continuous valid `fp-bps=-0.8411`

4. 启动正式 `500ms` gate run
   - 后台会话：`screen -S latent_dyn_state_500`
   - 日志：`results/logs/1.10-latent_dynamics_decoder/20260320_latent_dynamics_state_scaling/screen_run.log`

**当前判断**：
- 这一轮先不更换 GRU backbone，而是优先验证“更大 latent state”本身是否有价值
- 当前链路已打通，正式 `500ms` gate 已经开始训练

**遇到的问题与解决**：
- 首轮实现里即使增加 pooled query，最终 dynamics hidden size 仍被压回 `dim=128`
  - 解决：把 `pool_token_dim` 和 `state_dim` 做成显式可调参数，避免出现“query 数增加但总状态容量不变”的伪扩容

## 2026-03-21 16:54 CST

### 任务：完成 latent dynamics state scaling 500ms gate 并归档负结果

**完成内容**：
1. 完成 `20260320_latent_dynamics_state_scaling` 的 `500ms` formal gate
   - best epoch：`69`
   - valid：`fp-bps=0.0048`, `R2=0.1791`, `val_loss=0.3250`
   - test：`fp-bps=0.0049`, `R2=0.1790`, `val_loss=0.3235`

2. 生成结果汇总与图表入口
   - 新增 `collect_latent_dynamics_state_scaling_results.py`
   - 更新 `cc_todo/1.10-latent_dynamics_decoder/results.tsv`
   - 生成 `latent_dynamics_state_scaling_summary.json`
   - 更新训练曲线和 `1.10` 总体趋势图

3. 回填核心文档并标记模块状态
   - 更新 `cc_core_files/results.md`
   - 更新 `cc_core_files/plan.md`
   - 更新 `cc_todo/1.10-latent_dynamics_decoder/model.md`
   - 更新 `cc_todo/1.10-latent_dynamics_decoder/20260320_latent_dynamics_state_scaling.md`
   - 该模块状态标记为“已放弃”

**当前结论**：
- 更大 latent state 并没有改善 `500ms`，反而把上一轮的 `0.0904` 拉低到 `0.0048`
- 这说明当前问题不只是“状态容量不够”，而更可能是 dynamics 结构、context 注入方式或优化稳定性本身的问题
- 后续若继续推进 `1.10.x`，更合适的方向应是更强的 dynamics backbone 或显式 context skip

**遇到的问题与解决**：
- 这一轮只跑了 `500ms gate`，没有自然生成完整三窗口 summary
  - 解决：新增模块级 collect 脚本，单独为 `500ms gate` 生成 summary JSON，并把结果补写进 `results.tsv` 和训练曲线图流程
