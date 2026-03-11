# NeuroHorizon — Claude 工作指南

## 项目简介

**NeuroHorizon** 是一个基于 **POYO/POYO+** 框架（NeurIPS 2023）进行改造扩展的神经科学研究项目，目标是构建"跨 Session 鲁棒的长时程神经脉冲数据预测编码模型"。

项目详细背景、研究动机、核心创新点和技术方案，请参阅：
- **研究背景**：[cc_core_files/background.md](cc_core_files/background.md)（研究现状、研究意义、研究动机、相关工作）
- **研究方案**：[cc_core_files/proposal.md](cc_core_files/proposal.md)
- **执行参考**：[cc_core_files/proposal_review.md](cc_core_files/proposal_review.md)（各 Phase 代码改造方案、技术考量与风险评估）
- **执行计划**：[cc_core_files/plan.md](cc_core_files/plan.md)
- **代码库分析**：[cc_core_files/code_research.md](cc_core_files/code_research.md)
- **数据集规划**：[cc_core_files/dataset.md](cc_core_files/dataset.md)
- **任务完成记录**：[cc_todo/](cc_todo/)（各阶段任务的详细执行记录）

> 注：上述文档可能随项目推进持续更新，以文档最新版本为准。

---

## 核心文档导航

| 文档 | 路径 | 说明 |
|------|------|------|
| 研究背景 | cc_core_files/background.md | 研究现状、研究意义、研究动机、相关工作 |
| 研究方案 | cc_core_files/proposal.md | 核心挑战、创新点、方法设计、实验设计 |
| 执行参考 | cc_core_files/proposal_review.md | 各 Phase 代码改造方案、技术考量、风险与验收标准（**执行 plan 任务前必读**） |
| 执行计划 | cc_core_files/plan.md | 分阶段任务执行计划，按此计划推进项目 |
| 代码分析 | cc_core_files/code_research.md | POYO 代码架构及改造接口分析 |
| 数据集 | cc_core_files/dataset.md | 数据集选型、下载和使用说明 |
| 脚本记录 | cc_core_files/scripts.md | 所有脚本的功能、使用方式、存储位置 |
| 数据记录 | cc_core_files/data.md | 数据集的处理信息、位置、字段含义 |
| 结果记录 | cc_core_files/results.md | 实验结果说明（产生方式、目的、分析） |
| 知识库 | cc_core_files/knowledge.md | 核心概念、技术讨论与设计决策的深度分析 |
| 任务完成记录 | cc_todo/{phase-folder}/ | 各阶段任务执行的详情记录（做了什么、结果、问题等） |
| 工作进展 | progress.md | 每次任务的执行记录和问题沉淀 |

> 项目执行计划参考 `cc_core_files/plan.md`（按照 plan.md 规划执行，完成后在 plan.md 打勾并写入 cc_todo 记录）；执行前必读 `cc_core_files/proposal_review.md` 中对应 Phase 的执行参考。

---

## 目录结构规范

```
NeuroHorizon/
├── torch_brain/              # 主要源代码（基于 POYO 框架改造）
├── examples/                 # 训练示例脚本
│   ├── poyo/                 # 原始 POYO 示例
│   ├── poyo_plus/            # 原始 POYO+ 示例
│   └── neurohorizon/         # NeuroHorizon 训练脚本
├── scripts/                  # 数据处理 & 项目运行 & 测试脚本
│   ├── data/                 # 数据下载、预处理脚本
│   ├── analysis/             # 分析脚本
│   │   ├── explore_brainsets.py
│   │   ├── analyze_latents.py
│   │   └── neurohorizon/     # NeuroHorizon 专用分析脚本
│   └── tests/                # 测试脚本
├── data/                     # 数据集存储
│   ├── raw/                  # 原始下载数据
│   ├── processed/            # 预处理后数据（HDF5 等）
│   └── generated/            # 模型生成/推理结果
├── results/                  # 实验结果存储
│   ├── figures/              # 可视化图表
│   ├── logs/                 # 训练日志
│   └── checkpoints/          # 模型权重
├── cc_core_files/            # 项目核心文档
├── cc_todo/                  # 任务工作记录
│   ├── phase0-env-baseline/  # Phase 0 任务记录
│   ├── phase1-autoregressive/# Phase 1 任务记录
│   ├── phase2-cross-session/ # Phase 2 任务记录（待建）
│   ├── phase3-data-scaling/  # Phase 3 任务记录（待建）
│   ├── phase4-multimodal/    # Phase 4 任务记录（待建）
│   ├── phase5-paper/         # Phase 5 任务记录（待建）
│   ├── 20260221-cc-1st/      # ⚠️ 已废弃，勿参考
│   └── 20260225-review/      # ⚠️ 已废弃，勿参考
└── CLAUDE.md                 # 本文件
```

---

## 任务执行规范

### 任务开始前

1. **阅读 progress.md**：了解历史任务进展，借鉴已有经验，避免重复踩坑
2. **阅读 proposal_review.md 对应章节**：执行 plan.md 中某 Phase 的任务前，查阅 `cc_core_files/proposal_review.md` 中该 Phase 对应的执行参考节（如"第四节 Phase 1 执行参考"），了解代码改造方案、关键注意事项和验收标准
3. **确认任务范围**：明确当前任务是否对应 plan.md 中的某个阶段/步骤
4. **确认 cc_todo 文件夹存在**：在 `cc_todo/` 下确认该阶段文件夹存在（按 plan.md 大阶段划分，如 `cc_todo/phase1-autoregressive/`）；若无则新建

### 任务执行中

5. **文件命名**：
   - 新建的文件夹和文件名称使用**英文**
   - 产出的 Markdown 文档名称在**最前面包含日期**（格式：`YYYYMMDD-` 或 `YYYY-MM-DD-`）

6. **cc_todo 任务记录**：
   - 在对应阶段文件夹内新建或更新任务记录文件
   - **命名格式**：`{YYYYMMDD}-{phase}-{task_num}-{task}.md`
     - 示例：`20260302-phase1-1.1-core-modules.md`
   - 文件内容包括：日期、对应 plan.md 任务编号与名称、任务目标
   - 持续记录：做了什么、怎么做的、遇到的问题及解决方法、结果如何、文件位置、未完成项
   - **必须包含具体执行命令**：涉及训练、分析、测试等操作时，记录完整的可复现命令（含工作目录、conda 环境、脚本路径、参数），方便后续复现
   - 如果本次会话未完成任务，下次在**同一文件**中追加记录（注明新日期和会话）

7. **脚本管理**：
   - 脚本文件放在 `scripts/` 下的合适位置
   - 脚本创建后，**必须**在 `cc_core_files/scripts.md` 中记录：功能用途、创建时间、使用方式、存储位置
   - scripts.md / results.md / data.md 中的条目**按 Phase + task_num 顺序排列**（如 Phase 0 → Phase 1 → Phase 2，同一 Phase 内按 task_num 排序），便于追溯

8. **数据管理**：
   - 下载的数据集放在 `data/` 下，按类型区分（`raw/` `processed/` `generated/` 等）
   - **必须**在 `cc_core_files/data.md` 中记录：数据集名称、处理信息、存储位置、字段含义

9. **结果管理**：
   - 实验结果（可视化图表、分析输出等）放在 `results/` 下的合适位置
   - **必须**在 `cc_core_files/results.md` 中记录：产生方式、目的、结果分析
   - **可视化要求**：执行 plan.md 任务时，除了 table/JSON 等中间数据记录外，**必须**补充可视化分析图表：
     - 最基本的 loss 随 epochs 变化曲线（val_loss 和/或 train_loss）
     - R² 或其他核心指标随 epoch 的变化曲线
     - 任务相关的分析可视化（对比图、分布图、热力图等）
     - 图表存放在 `results/figures/` 下按 Phase/任务组织的子目录中
   - **结果解读要求**：在 `cc_core_files/results.md` 中对每张图（甚至每张子图）都**必须**提供详细解读：
     - 分析的什么数据、为了什么目的、实验条件如何
     - 结果如何、说明了什么
     - **交叉引用**：标注对应的 figure 路径、中间数据/JSON/log 路径、生成脚本路径、使用的数据路径

10. **脚本与中间数据保留**：
    - 执行 plan.md 任务时（训练、分析、数据处理等），**必须保留对应的脚本文件和中间数据文件**，不能只提供最终结果
    - 脚本放 `scripts/` 对应目录，中间数据放 `results/` 或 `data/` 对应目录
    - **禁止**仅在 `/tmp/` 中创建脚本而不迁移到项目目录

11. **plan.md 对应任务**：
    - 如果当前任务对应 `cc_core_files/plan.md` 中的某个任务，在 plan.md 对应位置记录执行情况（完成状态、完成程度、后续工作）
    - 任务**完全完成**后，在 plan.md 对应任务的 checkbox 处标记：将 `- [ ]` 改为 `- [x]`，在该行末尾追加：`<!-- 记录：cc_todo/{phase-folder}/{filename}.md -->`

### 任务完成后

12. **文档检查清单**：
    - [ ] cc_todo 任务记录已更新
    - [ ] scripts.md 已登记新脚本（如有）
    - [ ] results.md 已登记新结果（如有）
    - [ ] data.md 已登记新数据（如有）
    - [ ] plan.md 已打勾并添加 📄 引用（如有）

13. **检查乱码**：
    - 运行 `grep -rn '��' cc_core_files/ cc_todo/phase*/` 检查是否有乱码字符（覆盖核心文档和任务记录）
    - 如有，立即修复（参见"经验沉淀"中的乱码问题条目）

14. **更新 progress.md**：记录任务完成时间（日期-小时-分）、完成事项、执行结果、遇到的问题及解决方法

15. **检查 CLAUDE.md**：确认本文件内容是否过时，如有需要及时更新

16. **Git 提交**：每完成一个任务或功能模块，立即执行：
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

### ⚠️ 已废弃任务：cc_todo/20260225-review/

`cc_todo/20260225-review/` 目录包含早期文档审查记录，内容已过时，**勿作为当前任务的参考依据**。

---

## 经验沉淀

> 此处记录在多次任务中反复遇到的问题和解决方案，积累经验教训。

### 乱码问题

- **现象**：文档中出现 `��`（Unicode 替换字符 U+FFFD）或类似乱码
- **原因**：UTF-8 多字节字符被截断（如 SSH 传输中断、编辑器保存不完整、sed 操作切割了多字节序列）
- **预防**：每次编辑文档后运行 `grep -rn '��' cc_core_files/ cc_todo/phase*/` 检查
- **修复方法**：根据上下文推断原始字符，用 `sed -i` 替换修复；修复后再次 grep 验证

### Markdown 渲染问题

#### 代码块嵌套（三引号 ``` 冲突）

- **现象**：文档 preview 时代码块范围显示错误，后续内容被吞入代码块或格式混乱
- **原因**：外层 ` ``` ` 代码块内嵌套了 ` ```bash ` 等带语言标记的代码块，内层的 ` ``` ` 会被解析为外层的关闭标记
- **预防**：需要嵌套代码块时，外层使用**四个反引号** ` ```` ` 包裹，内层保持三个反引号 ` ``` `；外层反引号数量必须严格多于内层
- **示例**：
  ````
  外层用 4 个反引号
  ```bash
  内层用 3 个反引号（不会冲突）
  ```
  外层结尾也用 4 个反引号
  ````

#### 波浪号 `~` 被渲染为删除线

- **现象**：`0.25~0.27` 被渲染为 ~~0.25~0.27~~（删除线），`~100` 也可能被渲染异常
- **原因**：部分 Markdown 渲染器（Typora、Obsidian、部分 GitHub preview）将 `~text~` 或 `~~text~~` 解析为删除线
- **预防规则**：
  - **表示"约"**：用 `约` 替代 `~`（如 `约100` 而非 `~100`）
  - **表示"范围"**：用 en-dash `–` 替代 `~`（如 `0.25–0.27` 而非 `0.25~0.27`）
  - **代码块内的 `~`**（如路径 `~/.ssh/`）不受影响，无需修改
- **修复方法**：全局替换 `数字~数字` → `数字–数字`，`~数字` → `约数字`

---

## Git 配置

- **账户**：jpagkr@163.com
- **用户名**：Jp-17
- **Commit 语言**：中文
- **Push 时机**：每个任务/模块完成后立即 push
