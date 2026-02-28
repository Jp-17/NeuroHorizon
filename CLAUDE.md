# NeuroHorizon — Claude 工作指南

## 项目简介

**NeuroHorizon** 是一个基于 **POYO/POYO+** 框架（NeurIPS 2023）进行改造扩展的神经科学研究项目，目标是构建"跨 Session 鲁棒的长时程神经脉冲数据预测编码模型"。

项目详细背景、研究动机、核心创新点和技术方案，请参阅：
- **研究方案**：[cc_core_files/proposal.md](cc_core_files/proposal.md)
- **执行参考**：[cc_core_files/proposal_review.md](cc_core_files/proposal_review.md)（各 Phase 代码改造方案、技术考量与风险评估）
- **执行计划**：[cc_core_files/plan.md](cc_core_files/plan.md)
- **代码库分析**：[cc_core_files/code_research.md](cc_core_files/code_research.md)
- **数据集规划**：[cc_core_files/dataset.md](cc_core_files/dataset.md)

> 注：上述文档可能随项目推进持续更新，以文档最新版本为准。

---

## 当前项目状态

**Phase 0 执行阶段**

核心规划文档已完成，项目正式进入执行阶段。按 `cc_core_files/plan.md` 的 Phase 0 开始执行。

---

## 核心文档导航

| 文档 | 路径 | 说明 |
|------|------|------|
| 研究方案 | cc_core_files/proposal.md | 完整研究提案（背景、创新点、实验设计） |
| 执行参考 | cc_core_files/proposal_review.md | 各 Phase 代码改造方案、技术考量、风险与验收标准（**执行 plan 任务前必读**） |
| 执行计划 | cc_core_files/plan.md | 分阶段任务执行计划，按此计划推进项目 |
| 代码分析 | cc_core_files/code_research.md | POYO 代码架构及改造接口分析 |
| 数据集 | cc_core_files/dataset.md | 数据集选型、下载和使用说明 |
| 脚本记录 | cc_core_files/scripts.md | 所有脚本的功能、使用方式、存储位置 |
| 数据记录 | cc_core_files/data.md | 数据集的处理信息、位置、字段含义 |
| 结果记录 | cc_core_files/results.md | 实验结果说明（产生方式、目的、分析） |
| 工作进展 | progress.md | 每次任务的执行记录和问题沉淀 |

> 项目执行计划参考 `cc_core_files/plan.md`（按照 plan.md 规划执行，完成后在 plan.md 打勾）；执行前必读 `cc_core_files/proposal_review.md` 中对应 Phase 的执行参考。

---

## 目录结构规范

```
NeuroHorizon/
├── torch_brain/              # 主要源代码（基于 POYO 框架改造）
├── examples/                 # 训练示例脚本
│   ├── poyo/                 # 原始 POYO 示例
│   ├── poyo_plus/            # 原始 POYO+ 示例
│   └── neurohorizon/         # NeuroHorizon 训练脚本（待建）
├── scripts/                  # 数据处理 & 项目运行 & 测试脚本
│   ├── data/                 # 数据下载、预处理脚本
│   └── analysis/             # 分析脚本
├── data/                     # 数据集存储
│   ├── raw/                  # 原始下载数据
│   ├── processed/            # 预处理后数据（HDF5 等）
│   └── generated/            # 模型生成/推理结果
├── results/                  # 实验结果存储
│   ├── figures/              # 可视化图表
│   ├── logs/                 # 训练日志
│   └── checkpoints/          # 模型权重
├── cc_core_files/            # 项目核心文档
├── cc_todo/                  # 任务工作记录（历史存档）
│   └── 20260221-cc-1st/      # ⚠️ 已废弃任务，勿参考（见下方警告）
└── CLAUDE.md                 # 本文件
```

---

## 任务执行规范

### 任务开始前

1. **阅读 progress.md**：了解历史任务进展，借鉴已有经验，避免重复踩坑
2. **阅读 proposal_review.md 对应章节**：执行 plan.md 中某 Phase 的任务前，查阅 `cc_core_files/proposal_review.md` 中该 Phase 对应的执行参考节（如"第四节 Phase 1 执行参考"），了解代码改造方案、关键注意事项和验收标准
3. **确认任务范围**：明确当前任务是否对应 plan.md 中的某个阶段/步骤

### 任务执行中

4. **文件命名**：
   - 新建的文件夹和文件名称使用**英文**
   - 产出的 Markdown 文档名称在**最前面包含日期**（格式：`YYYYMMDD-` 或 `YYYY-MM-DD-`）

5. **脚本管理**：
   - 脚本文件放在 `scripts/` 下的合适位置
   - 脚本创建后，**必须**在 `cc_core_files/scripts.md` 中记录：功能用途、创建时间、使用方式、存储位置

6. **数据管理**：
   - 下载的数据集放在 `data/` 下，按类型区分（`raw/` `processed/` `generated/` 等）
   - **必须**在 `cc_core_files/data.md` 中记录：数据集名称、处理信息、存储位置、字段含义

7. **结果管理**：
   - 实验结果（可视化图表、分析输出等）放在 `results/` 下的合适位置
   - **必须**在 `cc_core_files/results.md` 中记录：产生方式、目的、结果分析

8. **plan.md 对应任务**：
   - 如果当前任务对应 `cc_core_files/plan.md` 中的某个任务，在 plan.md 对应位置记录执行情况（完成状态、完成程度、后续工作）

### 任务完成后

9. **更新 progress.md**：记录任务完成时间（日期-小时-分）、完成事项、执行结果、遇到的问题及解决方法

10. **检查 CLAUDE.md**：确认本文件内容是否过时，如有需要及时更新

11. **Git 提交**：每完成一个任务或功能模块，立即执行：
    ```bash
    git add <相关文件>
    git commit -m "中文提交信息"
    git push
    ```
    - Git commit 信息使用**中文**
    - 不要一次性积累大量更改后再提交，保持细粒度提交

---

## 重要警告

### ⚠️ 已废弃任务：cc_todo/20260221-cc-1st/

`cc_todo/20260221-cc-1st/` 目录记录了一次**已放弃**的早期任务，代码已回退到该次任务之前的状态。

**禁止参考该目录下的任何内容**，包括但不限于：
- 日志文件（docs/）
- 实验结果（results/、figures/）
- 训练日志（logs/）

如果在代码或配置中发现与该次任务相关的内容，需检查是否需要回退到更早的状态。

---

## 经验沉淀

> 此处记录在多次任务中反复遇到的问题和解决方案，积累经验教训。

*（随项目推进持续补充）*

---

## Git 配置

- **账户**：jpagkr@163.com
- **用户名**：Jp-17
- **Commit 语言**：中文
- **Push 时机**：每个任务/模块完成后立即 push

## Plan 任务执行规范

当收到"执行 plan.md 中某阶段某任务"的指令时，按以下规范操作：

### 执行前：阅读参考文档并建立记录文件

1. **读取 `cc_core_files/proposal_review.md` 中对应 Phase 的执行参考**（如 Phase 1 任务对应"第四节 Phase 1 执行参考"），作为代码改造方案和技术决策的依据
2. 在 `cc_todo/` 下确认该阶段文件夹存在（按 plan.md 大阶段划分，如 `cc_todo/phase1-autoregressive/`）；若无则新建
3. 在该文件夹内新建任务记录文件，命名格式：`{YYYYMMDD}-{大阶段名}-{小任务名}.md`
   - 示例：`20260228-phase1-poisson-loss.md`
4. 文件开头写入：日期、对应 plan.md 任务编号与名称、任务目标

### 执行中：持续记录

在记录文件中详细记录以下内容：
- **做了什么**：执行的具体步骤
- **怎么做的**：方法、命令、代码修改位置
- **遇到的问题及解决方法**：错误信息、排查过程、解决方案
- **结果如何**：输出、性能指标、验证结果
- **各种文件在哪里**：创建/修改的文件路径
- **还有什么没有做**：未完成项、后续工作

如果本次会话未完成任务，下次执行时在**同一文件**中追加记录（注明新日期和会话）。

### 执行中：数据 / 脚本 / 结果管理

- **脚本**：放在 `scripts/` 下合适位置，并记录到 `cc_core_files/scripts.md`
- **数据**：放在 `data/` 下（`raw/` / `processed/` / `generated/`），并记录到 `cc_core_files/data.md`
- **实验结果**：放在 `results/` 下，并记录到 `cc_core_files/results.md`

### 执行完成后：标记 plan.md

任务**完全完成**后，在 plan.md 对应任务的 checkbox 处标记：
- 将 `- [ ]` 改为 `- [x]`
- 在该行末尾追加：`<!-- 记录：cc_todo/{phase-folder}/{filename}.md -->`
