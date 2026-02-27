# NeuroHorizon 数据集规划

> 本文档记录 NeuroHorizon 项目的数据集候选清单、各数据集简介，以及阶段化选型逻辑。
> **操作细节**（下载脚本、格式转换、字段说明等）参见 `cc_core_files/data.md` 和 `scripts/data/` 下的脚本。

---

## 目录

1. [候选数据集总览](#1-候选数据集总览)
2. [各候选数据集简介](#2-各候选数据集简介)
   - [2.1 Brainsets 原生数据集（torch_brain 框架自带）](#21-brainsets-原生数据集)
   - [2.2 IBL Brain-wide Map](#22-ibl-brain-wide-map)
   - [2.3 Allen Visual Coding Neuropixels](#23-allen-visual-coding-neuropixels)
   - [2.4 NLB（Neural Latents Benchmark，完整版）](#24-nlb-neural-latents-benchmark)
   - [2.5 FALCON Benchmark](#25-falcon-benchmark)
3. [数据集选型策略](#3-数据集选型策略)
4. [各阶段适配注意事项](#4-各阶段适配注意事项)
5. [存储空间规划](#5-存储空间规划)
6. [参考资源](#6-参考资源)

---

## 1. 候选数据集总览

| 数据集 | 物种/脑区 | Sessions 数量 | 接入方式 | NeuroHorizon 主要用途 | 引入时机 |
|--------|---------|-------------|--------|-------------------|--------|
| **Brainsets 原生（4+1集）** | 猕猴，运动皮层 | ~100+ sessions（Perich-Miller最多：~70+） | brainsets API，零配置 | 初期自回归改造验证、早期跨session测试 | **阶段一** |
| **IBL Brain-wide Map** | 小鼠，全脑241脑区 | 459 sessions，12 labs | ONE API（AWS 公开） | 大规模跨session泛化、data scaling law | **阶段二** |
| **FALCON Benchmark** | 猕猴/人，运动皮层 | 多sub-task，跨日记录 | 官方 challenge API | 标准化跨session泛化benchmark验证 | **阶段二** |
| **Allen Visual Coding Neuropixels** | 小鼠，视觉皮层8个区 | 58 sessions | AllenSDK（AWS 公开） | 多模态（neural + 视觉图像）融合实验 | **阶段三** |
| **NLB 完整版** | 猕猴，多脑区 | 5个子数据集 | nlb_tools / brainsets | 与社区benchmark对比，标准化评估 | **补充/可选** |

> **已排除**：Allen Visual Coding Ophys 2016（钙成像，非spike数据，与NeuroHorizon输入不兼容）；Kemp Sleep（与项目目标无关）。

---

## 2. 各候选数据集简介

### 2.1 Brainsets 原生数据集

**brainsets** 是 torch_brain/POYO 框架自带的数据管理包，提供标准化的 spike 数据加载接口。
以下是排除 allen_visual_coding_ophys 和 kemp_sleep 后，适用于 NeuroHorizon 的 4 个子数据集：

#### （1）Perich-Miller Population 2018 ⭐ 跨session首选
- **论文**：Perich & Miller et al., 2018
- **brainsets ID**：`perich_miller_population_2018`
- **物种/脑区**：猕猴（3只：C、J、M），初级运动皮层（M1）+ 前运动皮层（PMd）
- **任务**：Center-out reaching（8个方向）和 Random target reaching
- **规模**：约 70+ sessions，每 session 50-200 个神经元
- **特点**：brainsets 内 session 数量最多，3只猴子跨动物，是在 brainsets 范围内做跨 session 泛化实验的最佳选择。POYO-1 和 POYO+ 均使用了该数据集。

#### （2）Churchland-Shenoy Neural 2012
- **论文**：Churchland & Shenoy et al., 2012（Nature Neuroscience）
- **brainsets ID**：`churchland_shenoy_neural_2012`
- **物种/脑区**：猕猴，运动皮层（M1/PMd）
- **任务**：To-target reaching，含 preparatory period（运动准备期）和 execution period（运动执行期）
- **规模**：数个 session，每 session ~100-200 个神经元
- **特点**：包含明确的运动准备→执行时间结构，预测窗口设计较为自然（输入=准备期，预测=执行期）

#### （3）O'Doherty-Sabes Nonhuman 2017
- **论文**：O'Doherty & Sabes et al., 2017（Nature Neuroscience）
- **brainsets ID**：`odoherty_sabes_nonhuman_2017`
- **物种/脑区**：猕猴，M1 + 体感皮层（Area 2）
- **任务**：BCI-controlled reaching（2D）
- **规模**：多个 session，每 session ~100-200 个神经元
- **特点**：包含体感皮层同步记录，是唯一包含非运动皮层的 brainsets 数据集，有助于测试跨脑区泛化能力

#### （4）Flint-Slutzky Accurate 2012
- **论文**：Flint & Slutzky et al., 2012
- **brainsets ID**：`flint_slutzky_accurate_2012`
- **物种/脑区**：猕猴，运动皮层
- **任务**：BCI cursor control
- **规模**：小规模，session 数量有限
- **特点**：神经元质量好，高精度记录，但 session 数量较少，主要价值在于丰富训练数据多样性

#### （5）Pei-Pandarinath NLB 2021（brainsets 内的 NLB 子集）
- **brainsets ID**：`pei_pandarinath_nlb_2021`
- **内容**：NLB MC_Maze 数据集（Jenkins 迷宫任务），来自 `jenkins_maze_train`
- **规模**：1 只猴子（Jenkins），~180 个神经元，单一 session
- **特点**：brainsets 内只包含 NLB 的部分数据，主要用于 POYO 与 NLB benchmark 的对接（完整 NLB 见第 2.4 节）

---

### 2.2 IBL Brain-wide Map

- **全称**：International Brain Lab Brain-wide Map（2025 年版）
- **物种/脑区**：小鼠，全脑 **241 个脑区**（视觉、前额叶、纹状体、丘脑、海马、小脑等）
- **规模**：**459 sessions**，139 只小鼠，12 个国际实验室，75,708 个高质量 units（共 621,733 total units）
- **任务**：标准化视觉决策任务——小鼠根据光栅对比度旋转滚轮做左/右判断
- **记录技术**：Neuropixels 探针，每次插入记录多脑区
- **数据格式**：ALF 格式（.npy），通过 ONE API + ibllib 访问，AWS 公开
- **对 NeuroHorizon 的价值**：
  - 跨 session 泛化实验的**唯一真正满足规模需求**的数据集（459 vs 58 session）
  - Data scaling law 实验（10/50/100/200/459 sessions）的核心数据源
  - 全脑多脑区覆盖，IDEncoder 的泛化性可以在真实神经多样性下验证
  - POYO/POYO+ 论文本身在 IBL 上有过验证，方法可直接对比
- **注意**：IBL 是行为驱动的变长 trial，连续时间段的截取策略需要设计（详见第 4 节）

---

### 2.3 Allen Visual Coding Neuropixels

> ⚠️ **注意区分**：这是 **Neuropixels 胞外电生理**版本，记录**尖峰放电**；不同于 Allen Visual Coding Ophys（钙成像版本，已排除）。

- **全称**：Allen Brain Observatory — Visual Coding (Neuropixels)
- **物种/脑区**：小鼠，同步记录最多 **8 个视觉相关脑区**（V1/VISp、LM、AL、PM、AM、RL、LGN、LP）
- **规模**：**58 sessions**，约 100,000 total units
- **刺激类型**：
  - Natural Scenes：118 张自然图像，每张呈现 250ms + ~500ms 灰屏间隔
  - Natural Movies：约 30 秒连续自然视频（无帧间间隔）
  - Drifting Gratings / Static Gratings / Locally Sparse Noise
- **数据格式**：NWB 格式，通过 AllenSDK 访问，AWS 公开
- **对 NeuroHorizon 的价值**：
  - **多模态融合实验的首选**：Natural Scenes 配合 DINOv2 图像 embedding 注入
  - 多脑区同步记录，支持跨脑区分析
  - Natural Movies 是**长时程连续预测实验**的优质数据源（30s 无间隔刺激驱动神经活动）
- **局限**：
  - 仅 58 sessions，不足以支撑 scaling law 实验
  - Natural Scenes 刺激时长短（250ms），预测窗口（500ms-1s）会跨入灰屏期，需要特别处理
  - AllenSDK 依赖复杂，建议在独立环境下载，主环境加载预处理后的 HDF5

---

### 2.4 NLB（Neural Latents Benchmark，完整版）

- **论文**：Pei et al., 2021（NeurIPS 2021 Datasets and Benchmarks）
- **官网**：neurallatents.github.io
- **包含 5 个子数据集**：

| 子数据集 | 脑区 | 任务 | Sessions |
|---------|------|------|---------|
| MC_Maze | M1/PMd，猕猴 | 迷宫导航 reaching | ~3 sessions |
| MC_RTT | M1，猕猴 | Random target reaching | ~3 sessions |
| Area2_Bump | 体感皮层 Area2，猕猴 | 扰动抵抗 reaching | ~5 sessions |
| DMFC_RSG | 背内侧前额叶，猕猴 | Ready-Set-Go时间任务 | ~5 sessions |
| MC_Cycle | M1，猕猴 | 手腕循环运动 | ~3 sessions |

- **数据获取**：通过 `nlb_tools` 或 brainsets（部分）下载
- **对 NeuroHorizon 的价值**：
  - 提供**标准化 train/val/test 分割**，便于与社区方法对比
  - 包含多个脑区（M1、PMd、体感、前额叶），比其他 brainsets 子集脑区更多样
  - 可作为改造后模型的 sanity check：在 NLB 上同时验证自回归预测和行为解码性能
- **定位**：主要作为 benchmark 验证工具，而非主要训练数据扩展

---

### 2.5 FALCON Benchmark

- **全称**：Functional ALignment for CONtinuous Decoding
- **论文**：Versteeg et al., 2023（NeurIPS 2023 Datasets and Benchmarks）
- **核心目标**：专门测试神经解码模型在**跨 session / 跨日 / 跨受试者**场景下的泛化能力
- **包含子任务**：

| Sub-task | 物种/脑区 | 任务 | 跨 session 难度 |
|---------|---------|------|--------------|
| FALCON-M1 | 猕猴，M1 | 手腕力量控制 | 跨日（同一动物） |
| FALCON-M2 | 猕猴，M1 | 3D reaching | 跨日（同一动物） |
| FALCON-H1 | 人（BrainGate2），运动皮层 | BCI cursor control | 跨日（同一受试者） |
| FALCON-H2 | 人（BrainGate2），语音区 | 语音解码 | 跨日（同一受试者） |

- **数据获取**：官方 FALCON challenge 网站，需注册下载
- **对 NeuroHorizon 的价值**：
  - 设计初衷与 NeuroHorizon 的 IDEncoder 目标高度对齐——零样本/少样本跨 session 泛化
  - 提供了标准化的"新 session 校准数据量"评估框架（对比不同程度的 held-out session 数据）
  - 人类 BCI 数据（H1/H2）是实际应用场景的直接体现
- **局限**：
  - 每个 sub-task 的训练数据量相对有限，主要作为**测试/benchmark**而非大规模训练扩展
  - 侧重**解码**任务，用于 NeuroHorizon **编码**任务需要额外适配

---

## 3. 数据集选型策略

### 3.1 总体思路

**渐进式扩展原则**：先利用 torch_brain/brainsets 框架原生支持的数据集快速迭代核心模型改造，待自回归生成机制和跨 session 泛化稳定后，再逐步扩展到 IBL 和 Falcon 的大规模测试，最后引入 Allen Neuropixels 进行多模态实验。

这一策略的核心考量：
1. **降低初期工程障碍**：brainsets 数据集无需额外数据管线，直接集成进 POYO 训练框架，可以专注于模型改造本身
2. **快速验证核心机制**：causal 自回归改造的正确性不依赖数据集规模，小规模数据集就能快速验证
3. **符合风险递进原则**：数据管线复杂度（IBL ONE API、AllenSDK）作为后期工程任务处理，不阻塞模型研究进展

---

### 3.2 阶段一：初期开发验证 ⬅ 当前阶段

**使用数据集**：Brainsets 原生（Perich-Miller 2018 为主，其余为辅）

**目标**：验证 causal 自回归解码器的基本功能，以及 IDEncoder + 自回归 pipeline 的端到端可行性

**具体安排**：

| 实验 | 使用数据 | 目标 |
|------|---------|------|
| 自回归改造功能验证 | Perich-Miller（选 5-10 个 session） | 确认 causal mask 正确、训练损失下降、预测 spike counts 合理 |
| 初步跨 session 测试 | Perich-Miller（3 只猴子，70+ sessions） | 测试 IDEncoder 在同任务不同动物/时间上的零样本泛化基准性能 |
| NLB sanity check（可选） | NLB MC_Maze（brainsets 内） | 验证改造后的模型不破坏原有 POYO 行为解码功能 |

**Perich-Miller 作为阶段一首选的理由**：
- brainsets 内 session 数量最多（70+），可直接做初步跨 session 测试
- 3 只猴子（C/J/M）涵盖跨动物泛化场景，挑战比单动物多 session 更接近真实需求
- POYO/POYO+ 在该数据集上有公开基线，便于对比

---

### 3.3 阶段二：大规模跨 Session 验证

**使用数据集**：IBL Brain-wide Map（主）+ FALCON（验证/benchmark）

**前提条件**：阶段一的自回归改造已验证功能正确，IDEncoder 在 brainsets 数据上有初步泛化表现

**目标**：验证 NeuroHorizon 核心贡献——跨 session、跨实验室、跨脑区的零样本泛化

**具体安排**：

| 实验 | 使用数据 | 目标 |
|------|---------|------|
| 跨 session 泛化主实验 | IBL（全部 459 sessions，train/val/test 按 session 划分） | IDEncoder 零样本跨 session R² / PSTH 相关性 |
| Data Scaling Law | IBL（10/50/100/200/459 sessions 子集） | 揭示模型性能随训练 session 数的增长曲线 |
| 标准化 benchmark 对比 | FALCON M1/M2 | 在社区公认 benchmark 上量化跨 session 泛化改进 |

**IBL 作为阶段二主力的理由**：
- 459 sessions 是唯一满足 scaling law 实验规模需求的公开数据集
- 12 个实验室数据统一标准化，能测试最具挑战性的"跨实验室"泛化
- 241 个脑区覆盖，IDEncoder 的脑区无关性得以在真实神经多样性下验证

---

### 3.4 阶段三：多模态实验

**使用数据集**：Allen Visual Coding Neuropixels

**前提条件**：阶段二的自回归预测和跨 session 泛化已稳定

**目标**：验证视觉图像（DINOv2 embedding）注入对神经活动预测的提升

**具体安排**：

| 实验 | 使用数据 | 目标 |
|------|---------|------|
| 连续预测基准 | Allen Natural Movies（30s 连续视频） | 验证多模态条件下长时程预测能力 |
| 图像-神经对齐实验 | Allen Natural Scenes（118 张图像）+ DINOv2 | 量化图像 embedding 对刺激响应预测精度的贡献 |

**Allen Neuropixels 的时机安排理由**：
- 需要额外实现 DINOv2 特征提取 pipeline，工程量较大
- AllenSDK 依赖环境与 torch_brain 可能有冲突，建议独立环境处理
- 多模态是"锦上添花"的贡献，核心贡献（自回归预测 + 跨 session 泛化）应先独立验证

---

### 3.5 NLB 完整版的定位

NLB 完整版（包含 MC_Maze、MC_RTT、Area2_Bump 等 5 个子集）不是主要的训练数据扩展，而是：

- **社区对比工具**：NLB 提供了标准化 benchmark，NeuroHorizon 可在其上报告自回归预测性能，便于与社区方法对比
- **多脑区 sanity check**：Area2_Bump（体感皮层）、DMFC_RSG（前额叶）等超出运动皮层的数据，可验证 IDEncoder 对更广脑区的适用性
- **引入时机**：可在阶段一完成后、阶段二开始前作为补充验证

---

## 4. 各阶段适配注意事项

### 4.1 Brainsets 原生数据集（阶段一）

**任务转换**：brainsets 数据集原本用于**行为解码**（spikes → cursor velocity），NeuroHorizon 需要转换为**自回归 spike 预测**（past spikes → future spike counts）。  
- spike 时间戳本身不需要改变，只需将输出从行为变量改为 binned spike counts  
- 可复用 brainsets 的数据加载和 tokenization 基础设施，仅替换 readout head

**输入/预测窗口设计**：运动任务有明确的 trial 结构（hold period → reach period）  
- 建议：输入窗口 = hold period（运动准备），预测窗口 = reach period 开头 500ms-1s  
- 或：采用滑动窗口连续截取（不依赖 trial 结构），最大化数据利用率

**跨 session 划分**：Perich-Miller 2018 按动物/日期划分 train/val/test session，不应按 trial 划分

### 4.2 IBL（阶段二）

**连续时间截取策略**：IBL 是行为驱动的变长 trial，有随机 ITI（试次间间隔）  
- 策略 A（trial 对齐）：以 `stimOn_times` 为锚点，输入窗口 = `[stimOn - 500ms, stimOn]`，预测窗口 = `[stimOn, stimOn + 500ms]`；清晰但混入了决策/奖励期  
- 策略 B（连续截取）：直接在连续记录上滑动 1s 窗口，不对齐 trial；最大化数据量，长时程预测实验优先使用  
- **建议**：长时程预测用策略 B，跨 session 实验两种均可，多模态实验不用 IBL

**质量过滤**：仅使用 `clusters.label == 1`（good quality units），去除低质量 unit

**数据量控制**：先下载 10-20 sessions 调试，再扩展到全部 459 sessions

### 4.3 Allen Neuropixels（阶段三）

**Natural Scenes 预测窗口问题**：每张图片仅呈现 250ms + ~500ms 灰屏间隔，1s 预测窗口会跨入灰屏期  
- 在论文中需明确说明：预测的是图像结束后的神经自发活动延续  
- 建议优先使用 **Natural Movies**（30s 连续无间隔）用于长时程预测验证，Natural Scenes 用于图像-神经对齐分析

**DINOv2 预处理**：Allen 图像为灰度图（918×1174），需转为 RGB 后送入 DINOv2  
- 建议**离线预提取**所有 118 张图像的 DINOv2 embedding，缓存为 `.pt` 文件，训练时直接加载

**AllenSDK 环境**：建议在独立 conda 环境下载数据，转存为 HDF5 后在主环境加载

---

## 5. 存储空间规划

| 数据集 | 下载内容 | 预估空间 |
|--------|---------|---------|
| Brainsets 原生（全部） | spike times + behavior（全部子集） | ~10-30 GB |
| IBL（调试阶段，10-20 sessions） | spike times + behavior | ~5-10 GB |
| IBL（完整，459 sessions，不含 LFP） | spike times + behavior | ~100-200 GB |
| IBL（预处理后，HDF5） | 转换后格式 | ~50-100 GB |
| Allen Neuropixels（58 sessions NWB） | spike times + behavior，不含 LFP | ~146.5 GB |
| Allen DINOv2 embeddings（预提取） | 118 张图像 × ViT-B/L | <1 GB |
| FALCON（所有 sub-tasks） | spike times | ~5-20 GB |
| **合计（完整实验）** | | **~350-450 GB** |

> 当前 `/root/autodl-tmp` 可用空间需提前确认。建议阶段一先只下载 brainsets 数据（~30GB），阶段二前再规划 IBL 存储。

---

## 6. 参考资源

### Brainsets / torch_brain
- torch_brain 文档：https://torch-brain.readthedocs.io/en/latest/
- brainsets GitHub（数据加载接口）：https://github.com/neuro-galaxy/brainsets
- Perich-Miller 2018 原始数据：DANDI Archive 或 via brainsets

### IBL
- IBL ONE API 文档：https://docs.internationalbrainlab.org
- IBL 2025 数据发布说明：https://docs.internationalbrainlab.org/notebooks_external/2025_data_release_brainwidemap.html
- IBL AWS 开放数据：https://registry.opendata.aws/ibl-brain-wide-map/

### Allen Neuropixels
- AllenSDK Visual Coding Neuropixels 文档：https://allensdk.readthedocs.io/en/latest/visual_coding_neuropixels.html
- Allen AWS 开放数据：https://registry.opendata.aws/allen-brain-observatory/

### NLB
- NLB 官网：https://neurallatents.github.io
- nlb_tools：https://github.com/neurallatents/nlb_tools

### FALCON
- FALCON Benchmark 官网：https://snel-repo.github.io/falcon/
- FALCON 论文（NeurIPS 2023）：Versteeg et al., 2023

---

*最后更新：2026-02-27*
