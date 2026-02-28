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

| 数据集 | 物种/脑区 | Sessions 数量 | 接入方式 | NeuroHorizon 主要用途 | 引入阶段 |
|--------|---------|-------------|--------|-------------------|--------|
| **Brainsets 原生（4+1集）** | 猕猴，运动皮层 | 100+ sessions（Perich-Miller 最多：~70+） | brainsets API，零配置 | 阶段一：自回归改造验证 + 长时程生成验证；阶段二：跨 session 初期测试；阶段三：scaling 初期测试 | **阶段一～三** |
| **IBL Brain-wide Map** | 小鼠，全脑 241 脑区 | 459 sessions，12 labs | ONE API（AWS 公开） | 阶段二可选扩展：跨 session 大规模泛化；阶段三可选扩展：大规模 scaling law | **阶段二可选 / 阶段三可选** |
| **Allen Visual Coding Neuropixels** | 小鼠，视觉皮层 8 个区 | 58 sessions | AllenSDK（AWS 公开） | 阶段四：多模态（neural + 视觉图像）融合实验 | **阶段四** |
| **FALCON Benchmark** | 猕猴/人，运动皮层 | 多 sub-task，跨日记录 | ���方 challenge API | 标准化跨 session 泛化 benchmark 验证（可选补充） | **补充/可选** |
| **NLB 完整版** | 猕猴，多脑区 | 5 个子数据集 | nlb_tools / brainsets | 与社区 benchmark 对比，标准化评估（可选） | **补充/可选** |

> **已排除**：Allen Visual Coding Ophys 2016（钙成像，非 spike 数据，与 NeuroHorizon 输入不兼容）；Kemp Sleep（与项目目标无关）。

---

## 2. 各候选数据集简介

### 2.1 Brainsets 原生数据集

**brainsets** 是 torch_brain/POYO 框架自带的数据管理包，提供标准化的 spike 数据加载接口。
以下是排除 allen_visual_coding_ophys 和 kemp_sleep 后，适用于 NeuroHorizon 的 4 个子数据集：

#### （1）Perich-Miller Population 2018 ⭐ 阶段一～三首选
- **论文**：Perich & Miller et al., 2018
- **brainsets ID**：`perich_miller_population_2018`
- **物种/脑区**：猕猴（3 只：C、J、M），初级运动皮层（M1）+ 前运动皮层（PMd）
- **任务**：Center-out reaching（8 个方向）和 Random target reaching
- **规模**：约 70+ sessions，每 session 50-200 个神经元
- **trial 时间结构**：
  - Hold period（静止准备期）：约 300-500ms
  - Movement period（运动执行期）：约 300-700ms（视方向而定）
  - 典型 trial 总时长：600-1200ms
- **预测窗口支持**：
  - 250ms：✅ 完全支持（hold 期 / movement 期均可作预测目标）
  - 500ms：✅ 基本支持（movement 期普遍 ≥500ms）
  - 1s：⚠️ 需跨越 movement 期 + 部分 rest 期，可行但需检查 trial 边界
  - 滑动窗口（不对齐 trial）：✅ 可任意窗口大小
- **Session 扩展支持**：可从 5 → 10 → 20 → 40 → 70+ sessions 动态扩增
- **特点**：brainsets 内 session 数量最多，3 只猴子跨动物，是在 brainsets 范围内做跨 session 泛化实验的最佳选择。POYO-1 使用了该数据集。

#### （2）Churchland-Shenoy Neural 2012
- **论文**：Churchland & Shenoy et al., 2012（Nature Neuroscience）
- **brainsets ID**：`churchland_shenoy_neural_2012`
- **物种/脑区**：猕猴，运动皮层（M1/PMd）
- **任务**：To-target reaching，含 preparatory period（运动准备期）和 execution period（运动执行期）
- **规模**：数个 session，每 session ~100-200 个神经元
- **trial 时间结构**：preparatory period ~500ms，execution period ~300-500ms
- **预测窗口支持**：250ms / 500ms ✅，1s ⚠️（需跨期）
- **特点**：包含明确的运动准备→执行时间结构，预测窗口设计较为自然（输入=准备期，预测=执行期）

#### （3）O'Doherty-Sabes Nonhuman 2017
- **论文**：O'Doherty & Sabes et al., 2017（Nature Neuroscience）
- **brainsets ID**：`odoherty_sabes_nonhuman_2017`
- **物种/脑区**：猕猴，M1 + 体感皮层（Area 2）
- **任务**：BCI-controlled reaching（2D）
- **规模**：多个 session，每 session ~100-200 个神经元
- **预测窗口支持**：250ms / 500ms ✅，1s ⚠️
- **特点**：包含体感皮层同步记录，是唯一包含非运动皮层的 brainsets 数据集，有助于测试跨脑区泛化能力

#### （4）Flint-Slutzky Accurate 2012
- **论文**：Flint & Slutzky et al., 2012
- **brainsets ID**：`flint_slutzky_accurate_2012`
- **物种/脑区**：猕猴，运动皮层
- **任务**：BCI cursor control
- **规模**：小规模，session 数量有限
- **预测窗口支持**：250ms / 500ms ✅
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
- **时间特性**：行为驱动的变长 trial（随机 ITI），以及 trial 间的连续记录；理论上可提取任意时间窗口
- **预测窗口支持**：
  - 250ms / 500ms / 1s：✅ 均完全支持（使用滑动窗口策略，连续记录不受 trial 时长限制）
  - 更长窗口：✅ 可行，但 ITI 阶段神经活动可能较弱
- **Session 扩展支持**：可从 10 → 30 → 50 → 100 → 200 → 459 sessions 动态扩增，是 scaling law 实验的核心数据源
- **对 NeuroHorizon 的价值**：
  - 阶段二可选：跨 session 大规模泛化实验（459 sessions，12 实验室，241 脑区）
  - 阶段三可选：data scaling law 实验（10/30/50/100/200/459 sessions 子集）
  - 全脑多脑区覆盖，IDEncoder 的泛化性可以在真实神经多样性下验证
- **重要说明**：
  - ⚠️ **POYO 和 POYO+ 均未在 IBL Brain-wide Map 上进行过验证**。POYO-1（NeurIPS 2023）在猕猴运动皮层电生理数据上验证；POYO+（ICLR 2025）在 Allen Brain Observatory 钙成像数据上验证。因此 NeuroHorizon 在 IBL 上的结果**需自行建立 baseline**，无法直接与 POYO/POYO+ 对比（但可与其他支持 IBL 的模型如 NDT3、NEDS 对比）
  - IBL 数据管线（ONE API、ibllib）工程量较大，建议在阶段一核心模型验证稳定后再接入

---

### 2.3 Allen Visual Coding Neuropixels

> ⚠️ **注意区分**：这是 **Neuropixels 胞外电生理**版本，记录**尖峰放电**；不同于 Allen Visual Coding Ophys（钙成像版本，已排除）。

- **全称**：Allen Brain Observatory — Visual Coding (Neuropixels)
- **物种/脑区**：小鼠，同步记录最多 **8 个视觉相关脑区**（V1/VISp、LM、AL、PM、AM、RL、LGN、LP）
- **规模**：**58 sessions**，约 100,000 total units
- **刺激类型**：
  - Natural Scenes：118 张自然图像，每张呈现 250ms + ~500ms 灰屏间隔
  - Natural Movies：约 **30 秒连续**自然视频（无帧间间隔）
  - Drifting Gratings / Static Gratings / Locally Sparse Noise
- **数据格式**：NWB 格式，通过 AllenSDK 访问，AWS 公开
- **预测窗口支持**（按刺激类型）：
  - Natural Movies：✅ 任意窗口（250ms / 500ms / 1s 均完全支持，30s 连续无间断）
  - Natural Scenes：⚠️ 每张图片仅 250ms 刺激 + 500ms 灰屏；**预测窗口建议 ≤250ms，或明确说明预测跨越了刺激边界**；1s 窗口会包含多个刺激+灰屏周期
- **对 NeuroHorizon 的价值**：
  - **多模态融合实验的首选**（阶段四）：Natural Scenes 配合 DINOv2 图像 embedding 注入；Natural Movies 用于长时程连续预测
  - 多脑区同步记录，支持跨脑区分析
- **局限**：
  - 仅 58 sessions，不足以支撑 scaling law 实验
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
| DMFC_RSG | 背内侧前额叶，猕猴 | Ready-Set-Go 时间任务 | ~5 sessions |
| MC_Cycle | M1，猕猴 | 手腕循环运动 | ~3 sessions |

- **数据获取**：通过 `nlb_tools` 或 brainsets（部分）下载
- **对 NeuroHorizon 的价值**：
  - 提供**标准化 train/val/test 分割**，便于与社区方法对比
  - 包含多个脑区（M1、PMd、体感、前额叶），比其他 brainsets 子集脑区更多样
  - 可作为改造后模型的 sanity check：在 NLB 上同时验证自回归预测和行为解码性能
- **定位**：主要作为 benchmark 验证��具，而非主要训练数据扩展

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
  - 人类 BCI 数据（H1/H2）是实际应用场景的直接体现
- **定位**：补充验证工具，阶段二跨 session 测试稳定后可引入作为外部 benchmark

---

## 3. 数据集选型策略

### 3.1 总体思路

**渐进式扩展原则**：按**项目执行进度**划分四个阶段，各阶段均**以 brainsets 原生数据集为起点**，待核心功能验证通过后再视需要向外扩展（IBL、Allen）。

核心考量：
1. **降低初期工程障碍**：brainsets 数据集无需额外数据管线，直接集成进 POYO 训练框架，可以专注于模型改造本身
2. **快速验证核心机制**：causal 自回归改造的正确性、跨 session 泛化、scaling 规律等均不依赖超大规模数据，小规模验证通过后再扩展
3. **灵活扩展**：预测窗口长度（250ms → 500ms → 1s）和 scaling session 数量（动态递增）均根据实际数据和实验结果灵活调整，不预设固定目标
4. **IBL 为可选扩展**：IBL 数据管线工程量较大，且 POYO/POYO+ 均未在其上验证，将其定位为阶段二/三的"可选升级"而非必选路径

---

### 3.2 阶段一：自回归改造验证 + 长时程生成验证 ⬅ 当前阶段

**使用数据集**：Brainsets 原生（Perich-Miller 2018 为主，其余为辅）

**目标**：
1. 验证 causal 自回归解码器改造的基本功能（pipeline 通畅、causal mask 正确、训练损失下降）
2. 验证不同预测窗口长度下的生成质量（梯度扩展：250ms → 500ms → 1s）

**具体安排**：

| 实验 | 使用数据 | 目��� |
|------|---------|------|
| 自回归改造功能验证 | Perich-Miller（选 5-10 个 session） | causal mask 正确、训练损失下降、预测 spike counts 合理 |
| 预测窗口梯度测试 | Perich-Miller（10-20 sessions） | 从 250ms 起步，评估 500ms、1s 的预测质量，视结果决定后续重点窗口 |
| NLB sanity check（可选） | NLB MC_Maze（brainsets 内） | 验证改造后的模型不破坏原有 POYO 行为解码功能 |

**数据需求**：
- 下载：通过 brainsets API 获取 `perich_miller_population_2018`（~5-10GB），无需额外安装
- 无需下载 IBL / Allen

---

### 3.3 阶段二：跨 Session 测试

**使用数据集**：Brainsets 原生（Perich-Miller 2018）为基础，IBL 为可选扩展

**前提**：阶段一的自回归改造已验证基本功能正确（causal mask、损失收敛、基本 spike 预测可行）

**目标**：
1. 实现并验证 IDEncoder 基础功能（特征提取、嵌入质量、替换 InfiniteVocabEmbedding）
2. 验证 IDEncoder 的跨 session 泛化能力（不同动物、不同日期的零样本泛化）

**具体安排**：

| 实验 | 使用数据 | 目标 | 扩展路径 |
|------|---------|------|---------|
| IDEncoder 基础实现与验证（必做） | Perich-Miller（单动物多 session，5-10 sessions） | 实现 id_encoder.py，替换 InfiniteVocabEmbedding，验证特征提取正确性、嵌入质量、pipeline 端到端运行 | 单动物验证通过后进入跨 session 测试 |
| Brainsets 跨 session 测试（必做） | Perich-Miller（3 只猴子，70+ sessions，按动物划分 train/val/test） | IDEncoder 零样本跨 session/跨动物基准性能（R² / PSTH 相关性） | 先 5 个 session → 验证通过后扩展到全部 70+ |
| IBL 跨 session 扩展（可选） | IBL（先 10-20 sessions 调试，验证通过后逐步扩展） | 在更大规模（跨实验室、跨脑区）数据上验证 IDEncoder 泛化性 | 10 → 50 → 100 → 459 sessions |
| FALCON benchmark（可选补充） | FALCON M1/M2 | 在社区公认 benchmark 上量化跨 session 泛化改进 | 阶段二结果稳定后 |

**IBL 扩展决策依据**：
- 若 Perich-Miller 上跨 session 结果令人满意（例如 IDEncoder 明显优于 POYO 固定嵌入基线），则扩展到 IBL 以获得更强 paper 贡献
- 若 Perich-Miller 上结果一般，先集中优化模型再考虑 IBL

**IBL 数据需求（如扩展）**：
- 安装 ONE API + ibllib（约 1-2 天工程调试）
- 先下载 10-20 sessions（~5-10GB）调试，验证管线后扩展
- 完整 459 sessions 约 100-200GB，需提前规划存储

---

### 3.4 阶段三：Data Scaling + 下游任务泛化

**使用数据集**：Brainsets 原生（Perich-Miller 2018 + 其余子集）为基础，IBL 为可选扩展

**前提**：阶段二跨 session 泛化已有基本验证结果

**目标**：
1. 揭示模型性能随训练数据量（session 数）增长的 scaling 规律
2. 验证预训练的自回归编码器在下游任务（行为解码）上的泛化能力

**具体安排**：

| 实验 | 使用数据 | 目标 | Session 扩展策略 |
|------|---------|------|----------------|
| Brainsets Scaling 测试（必做） | Perich-Miller（5/10/20/40/70+ sessions） | 小规模 scaling 曲线，验证 data scaling 规律存在 | 动态扩增：5 → 10 → 20 → 40 → 70+（视结果决定是否继续）|
| 下游任务泛化（必做） | Perich-Miller（预训练 → 微调行为解码） | 验证自回归预训练是否改善下游行为解码 R² | 与 POYO 基线对比 |
| IBL 大规模 Scaling（可选） | IBL（30/50/100/200/459 sessions 子集） | 大规模 scaling law（论文核心实验之一） | 30 → 50 → 100 → 200 → 459（视资源和结果动态调整）|
| IBL 跨实验室泛化（可选） | IBL（按实验室划分 train/test） | 12 实验室跨实验室零样本泛化 | 在 IBL 管线就绪后进行 |

**Scaling 策略说明**：
- Session 数量目标不是固定的，根据实际结果是否出现明显 scaling 趋势动态决定是否继续增加
- 若 Brainsets 的 70 sessions 已能体现 scaling 规律，IBL 扩展作为锦上添花
- 若 Brainsets scaling 曲线平坦，再考虑 IBL 来验证更大规模

---

### 3.5 阶段四：多模态引入

**使用数据集**：Allen Visual Coding Neuropixels

**前提**：阶段二/三的自回归预测和跨 session 泛化已稳定，有基本结论

**目标**：验证视觉图像（DINOv2 embedding）注入对神经活动预测的提升，以及不同模态条件下的分析实验

**具体安排**：

| 实验 | 使用数据 | 目标 |
|------|---------|------|
| 长时程连续预测基准 | Allen Natural Movies（30s 连续视频） | 验证多模态条件下的连续长时程预测能力（250ms/500ms/1s 梯度测试） |
| 图像-神经对齐实验 | Allen Natural Scenes（118 张图像）+ DINOv2 | 量化图像 embedding 对刺激响应预测精度的贡献 |
| 图像条件消融 | Allen Natural Scenes（有无图像条件对比） | 分析不同模态对预测的独立贡献 |

**Allen 的时机安排理由**：
- 需要额外实现 DINOv2 特征提取 pipeline，工程量较大
- AllenSDK 依赖环境与 torch_brain 可能有冲突，需独立环境处理
- 多模态是"锦上添花"的贡献，核心贡献（自回归预测 + 跨 session 泛化）应先独立验证

---

### 3.6 NLB 完整版与 FALCON 的定位

**NLB 完整版**（MC_Maze、MC_RTT、Area2_Bump 等 5 个子集）：
- 社区 benchmark 对比工具，可在任何阶段引入
- 多脑区 sanity check（Area2_Bump 体感皮层、DMFC_RSG 前额叶）
- **引入时机**：阶段一完成后、阶段二开始前可选做

**FALCON Benchmark**：
- 跨 session 泛化的标准化 benchmark，与 IDEncoder 目标高度契合
- **引入时机**：阶段二跨 session 测试有基本结论后，作为外部验证

---

## 4. 各阶段适配注意事项

### 4.1 阶段一：Brainsets 原生数据集

**任务转换**：brainsets 数据集原本用于**行为解码**（spikes → cursor velocity），NeuroHorizon 需要转换为**自回归 spike 预测**（past spikes → future spike counts）。
- spike 时间戳本身不需要改变，只需将输出从行为变量改为 binned spike counts
- 可复用 brainsets 的数据加载和 tokenization 基础设施，仅替换 readout head

**输入/预测窗口设计**：
- 方案 A（trial 对齐）：输入窗口 = hold period（运动准备），预测窗口 = reach period
  - 250ms：✅ 取 reach 期前 250ms 作为预测目标，自然且无需跨 trial
  - 500ms：✅ 大部分 reach trial 时长 ≥500ms，可行
  - 1s：⚠️ 需将 reach + 部分 rest 纳入预测，需检查是否存在 trial 结束截断问题
- 方案 B（滑动窗口，不依赖 trial 结构）：✅ 灵活支持任意窗口大小，最大化数据利用率
- **建议**：初期用方案 A 快速验证（trial 结构清晰），稳定后切换方案 B 做窗口梯度测试

**数据下载**：通过 brainsets API 自动处理，无需手动配置

---

### 4.2 阶段二/三：Brainsets 跨 Session 使用，及 IBL 可选扩展

#### Brainsets（必做基础）

**Session 划分**：Perich-Miller 2018 按动物/日期划分 train/val/test session，不应按 trial 划分
- 建议划分：以动物为粒度（例如 C/J 训练，M 测试），或在同一动物内按日期划分，确保 test session 的神经元在训练中未曾出现
- IDEncoder 验证目标：test session 仅用参考窗口原始神经活动数据生成 unit embedding，不参与任何梯度更新

**Session 数量扩增**：从小规模验证开始，视跨 session 效果逐步扩增

| 步骤 | Session 数量 | 说明 |
|------|------------|------|
| IDEncoder 单动物验证 | 5-10 sessions（单动物） | 确认 IDEncoder 替换正确、pipeline 通 |
| 初步跨 session | 20-40 sessions（多动物） | 测试跨动物零样本泛化基准性能 |
| 全量 Brainsets | 70+ sessions（3 只动物全量） | Brainsets 范围内最终跨 session 结论 |

**预测窗口**：沿用阶段一确定的最优窗口（250ms 或 500ms）；跨 session 实验阶段不必同时测试多个窗口，以控制实验量

---

#### IBL（可选扩展）

**启动时机**：Brainsets 跨 session 结果令人满意（IDEncoder 明显优于固定嵌入基线）时，扩展到 IBL 以获得更大规模验证

**连续时间截取策略**：IBL 是行为驱动的变长 trial，有随机 ITI（试次间间隔）
- 策略 A（trial 对齐）：以 `stimOn_times` 为锚点，输入 `[stimOn - T_in, stimOn]`，预测 `[stimOn, stimOn + T_pred]`；清晰但混入了决策/奖励期
- 策略 B（连续截取）：直接在连续记录上滑动窗口，不对齐 trial；最大化数据量，支持任意预测窗口长度
- **建议**：长时程预测用策略 B，跨 session 实验两种均可

**Session 扩展策略（动态调整）**：

| 步骤 | Session 数量 | 说明 |
|------|------------|------|
| 调试阶段 | 10-20 sessions（~5-10GB） | 验证数据管线、检查格式兼容性 |
| 初步实验 | 30 sessions（~15-20GB） | 首次跨 session / scaling 实验基准 |
| 中期扩展 | 50 → 100 sessions（~30-60GB） | 视结果是否出现明显增益决定是否继续 |
| 大规模实验 | 200 → 459 sessions（~100-200GB） | 最终 scaling law 曲线，论文核心数据 |

> 每次扩展前先分析现有结果：若曲线已经趋于平坦，则不必再增加。

**质量过滤**：仅使用 `clusters.label == 1`（good quality units），去除低质量 unit

**存储预规划**：IBL 完整下载前需确认 `/root/autodl-tmp` 剩余空间 > 200GB

---

### 4.3 阶段四：Allen Neuropixels

**刺激类型选择**：
- **Natural Movies**（优先）：30s 连续无间隔，✅ 完全支持 250ms / 500ms / 1s 等任意预测窗口；是长时程预测实验的首选
- **Natural Scenes**：每张图片 250ms + ~500ms 灰屏；**建议预测窗口 ≤250ms**，或在分析时明确说明跨刺激边界；若做 DINOv2 图像-神经对齐实验，使用 Natural Scenes 是必要的

**预测窗口选择（Allen 各刺激类型）**：

| 刺激类型 | 250ms | 500ms | 1s | 说明 |
|---------|-------|-------|-----|------|
| Natural Movies | ✅ | ✅ | ✅ | 30s 连续，无约束 |
| Natural Scenes | ✅ | ⚠️（跨灰屏） | ⚠️（跨多刺激） | 250ms 最清晰 |
| Drifting Gratings | ✅ | ✅（典型 2s 时长） | ✅ | 时长充足 |

**DINOv2 预处理**：Allen 图像为灰度图（918×1174），需转为 RGB 后送入 DINOv2
- 建议**离线预提取**所有 118 张图像的 DINOv2 embedding，缓存为 `.pt` 文件，训练时直接加载

**AllenSDK 环境**：建议在独立 conda 环境下载数据，转存为 HDF5 后在主环境加载

---

### 4.4 预测窗口梯度扩展策略

**整体原则**：不预设固定的"目标预测窗口"，从短到长梯度验证，根据数据支撑和实验结果灵活调整

**推荐扩展路径**：

```
250ms → 500ms → 1s（→ 更长，视数据和结果而定）
```

| 窗口 | 数据支撑 | 预期难度 | 决策逻辑 |
|------|---------|---------|---------|
| **250ms** | Brainsets / IBL / Allen 均支持 | 低（基础验证） | 必做，作为 baseline |
| **500ms** | Brainsets 基本支持，IBL / Allen Movies 完全支持 | 中 | 若 250ms 结果好，扩展到 500ms |
| **1s** | IBL / Allen Movies 完全支持；Brainsets 需滑动窗口策略 | 高（自回归 50 步） | 若 500ms 结果稳定，扩展到 1s |
| **>1s** | IBL / Allen Movies 均可支持 | 探索性 | 视 1s 结果和项目资源决定 |

**实验记录建议**：每个窗口均记录 PSTH 相关性和 R²，在 results.md 中对比折线图

---

### 4.5 Session 数量动态扩增策略

**整体原则**：不预设固定 session 数量目标，视 scaling 曲线是否仍有增益动态决定是否继续扩增

**Brainsets（Perich-Miller）扩增路径**：
```
5 sessions → 10 → 20 → 40 → 70+（全量）
```

**IBL 扩增路径（阶段二/三，如启动）**：
```
10-20（调试） → 30（初步实验） → 50 → 100 → 200 → 459（全量）
```

**扩增决策依据**：
- 若性能在当前规模下已趋于饱和（scaling 曲线斜率接近 0），不必继续扩增
- 若资源（显存 / 存储 / 时间）出现压力，优先在当前最优规模固化结论
- 每次扩增前更新 data.md 和 scripts/data/ 下的下载记录

---

## 5. 存储空间规划

| 数据集 | 阶段 | 下载内容 | 预估空间 |
|--------|------|---------|---------|
| Brainsets 原生（全部子集） | 阶段一 | spike times + behavior | ~10-30 GB |
| IBL（调试，10-20 sessions） | 阶段二入口 | spike times + behavior | ~5-10 GB |
| IBL（中等规模，100 sessions） | 阶段二/三扩展 | spike times + behavior | ~30-60 GB |
| IBL（完整，459 sessions，不含 LFP） | 阶段三完整实验 | spike times + behavior | ~100-200 GB |
| IBL（预处理后，HDF5） | 阶段二/三 | 转换后格式 | ~50-100 GB |
| Allen Neuropixels（58 sessions NWB） | 阶段四 | spike times + behavior，不含 LFP | ~146.5 GB |
| Allen DINOv2 embeddings（预提取） | 阶段四 | 118 张图像 × ViT-B/L | <1 GB |
| FALCON（所有 sub-tasks） | 补充/可选 | spike times | ~5-20 GB |
| **合计（完整实验，四阶段）** | | | **~350-450 GB** |

> **分阶段存储计划**：
> - 阶段一：仅需 ~30GB（Brainsets），当前服务器空间足够
> - 阶段二前：确认 `/root/autodl-tmp` 剩余 > 60GB（IBL 调试 + 中等规模）
> - 阶段三前：确认剩余 > 200GB（IBL 完整下载）
> - 阶段四前：确认剩余 > 150GB（Allen NWB + 预处理缓存）

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

### 参考对比模型

#### 同类基础模型（主要对比基线）
- **POYO**（Azabou et al., NeurIPS 2023）：spike-level tokenization + Perceiver 编码器，猕猴运动皮层行为解码；NeuroHorizon 的直接代码基础，行为解码 baseline 可直接对比。GitHub: https://github.com/neuro-galaxy/torch_brain
- **POYO+**（Azabou et al., ICLR 2025 Spotlight）：多 session、多任务扩展，在 Allen Brain Observatory 钙成像数据上验证；任务侧重不同（解码 vs 编码），但跨 session 机制有参考价值
- **SPINT**（Liu et al., 2023）：提出通过参考窗口原始神经活动（binned spike counts）生成 unit embedding 的 IDEncoder 思路，实现梯度无关跨 session 泛化；NeuroHorizon IDEncoder 模块的直接灵感来源，跨 session 泛化设计应与其对比
- **Neuroformer**（Antoniades et al., 2023）：首个自回归 spike-level 预测框架，支持多模态条件输入（图像、行为）；与 NeuroHorizon 任务最接近（自回归 spike 编码），是阶段一/二的主要对比对象
- **NDT1 / NDT2**（Ye & Pandarinath, 2021；Ye et al., 2023）：masked spike prediction（类 BERT 范式），binned spike counts 输入；可在 Brainsets 数据上对比 PSTH 预测质量和行为解码 R²
- **NDT3**（Wang et al., 2023）：大规模跨 session 预训练，在 IBL Repeated Site 上有公开结果，是阶段二/三 IBL 扩展后的主要跨 session 泛化对比基线

#### 在 IBL 上有公开结果（阶段二/三 IBL 扩展后的直接对比目标）
- **NDT3**（Wang et al., 2023）：IBL Repeated Site 上的跨 session 基线
- **NEDS**（Neural Encoding and Decoding at Scale, arXiv:2504.08201）：同时包含 spike 预测（编码）和行为解码任务，在 IBL Repeated Site 上对比了 POYO+ 和 NDT3 基线；与 NeuroHorizon 对比场景最接近

---

*最后更新：2026-02-28*
