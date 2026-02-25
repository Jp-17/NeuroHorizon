# NeuroHorizon: 跨Session鲁棒的长时程神经脉冲数据预测编码模型

## 研究计划书

**项目名称**：NeuroHorizon — 面向跨Session鲁棒性与长时程预测的可扩展神经脉冲编码模型

**研究领域**：计算神经科学 / 神经数据建模 / 脑机接口

**目标会议/期刊**：NeurIPS / ICLR / Nature Methods

**日期**：2026年2月13日

---

## 目录

1. [研究背景](#1-研究背景)
2. [问题定义](#2-问题定义)
3. [相关工作](#3-相关工作)
4. [研究创新点](#4-研究创新点)
5. [方法设计](#5-方法设计)
6. [实验设置](#6-实验设置)
7. [数据集](#7-数据集)
8. [预期结果与实验](#8-预期结果与实验)
9. [创新性与局限性讨论](#9-创新性与局限性讨论)
10. [参考文献](#10-参考文献)
11. [执行计划](#11-执行计划)

---

## 1. 研究背景

### 1.1 当前研究现状

大规模神经记录技术（如Neuropixels探针、多电极阵列等）的快速发展使得同时记录数千个神经元成为可能，推动了对神经群体活动的深层理解。然而，如何有效建模这些高维、高时间分辨率的spiking数据，仍然是计算神经科学的核心挑战之一。

近年来，受自然语言处理（NLP）领域Transformer架构成功的启发，研究者开始将基础模型（Foundation Model）的理念引入神经数据建模。一系列"神经数据基础模型"相继涌现：Neural Data Transformer（NDT1/NDT2/NDT3）系列通过masked modeling范式建模binned spike counts；LFADS及其变体利用变分自编码器提取低维潜在动力学；Neuroformer首次提出自回归的spike-level预测框架并支持多模态条件输入；POYO/POYO+通过tokenization of individual spikes实现了高时间分辨率建模，并利用Perceiver架构实现了跨session的数据整合；SPINT通过IDEncoder机制实现了无需梯度更新的跨session泛化。

这些进展虽然令人瞩目，但当前模型仍然面临几个关键瓶颈：(1) 大多数模型（如NDT系列）仍依赖binned spike counts作为输入，丢失了毫秒级时间信息；(2) 跨session泛化仍然困难，不同session中的神经元集合可能完全不同，传统方法需要对每个新session进行微调；(3) 长时程预测（>200ms）几乎未被系统性地探索，现有方法主要关注短时间窗口内的next-token prediction；(4) 多模态信息（行为数据、视觉刺激等）的融合预测并没有充分实现。

### 1.2 研究意义

本研究旨在构建一个同时解决跨session鲁棒性和长时程预测两大核心问题的统一神经编码模型。这一研究具有多层面的重要意义：

**技术层面**：跨session的鲁棒泛化将大幅降低神经数据分析的实验成本——无需为每个新session重新训练或微调模型。长时程预测能力为实时脑机接口（BCI）系统提供了更稳定的解码基础。

**科学层面**：长时程神经活动预测能力将帮助研究者理解神经群体动力学的长程依赖性和时间演化规律。同时，通过在模型中引入多模态信息（行为数据、视觉刺激），可以量化不同输入模态对神经活动的预测贡献，揭示感觉-运动系统中信息编码的深层机制。

**方法论层面**：本研究提出的架构设计（类IDEncoder的神经元身份编码支持跨session机制、长时程的自回归解码器）将为神经数据基础模型的设计提供新的范式参考。

### 1.3 研究动机

当前最先进的神经数据模型（如POYO+、NDT3）虽然在各自的任务上取得了优秀表现，但它们在设计上存在不同的侧重与局限：POYO+擅长高时间分辨率的spike-level建模但缺乏长时程生成能力；NDT3支持大规模数据训练但采用masked prediction范式、不适合自回归生成；Neuroformer支持自回归生成和多模态输入但跨session能力较弱；SPINT提出了优秀的跨session机制但主要应用于解码任务。

这些模型各有优势但缺乏统一。我们的研究动机正是在于：能否设计一个统一的编码模型，同时继承POYO的高时间分辨率tokenization、SPINT的跨session泛化能力、Neuroformer的多模态融合与自回归生成能力，并将预测时间窗口扩展到1秒甚至更长。

---

## 2. 问题定义

### 2.1 核心问题

**形式化定义**：给定来自任意recording session $s$ 的一段时间窗口 $[0, T]$（$T = 1\text{s}$）内的神经群体spiking活动 $\mathcal{S} = \{(n_i, t_i)\}_{i=1}^{N_{\text{spikes}}}$，其中 $n_i$ 为神经元标识、$t_i$ 为spike时间戳，以及可选的多模态条件信息 $\mathcal{C}$（行为数据、视觉刺激等），我们的目标是：

1. **编码**：学习一个通用的神经群体活动表征 $\mathbf{Z} = f_\theta(\mathcal{S}, \mathcal{C})$，该表征捕获神经群体的时空动态模式。

2. **长时程预测**：基于编码表征，通过自回归解码器生成未来时间窗口 $[T, T + \Delta T]$ 内各神经元在固定time bins中的spike counts预测 $\hat{\mathbf{Y}} = g_\phi(\mathbf{Z})$。

3. **跨Session泛化**：模型在新的session $s'$（可能包含完全不同的神经元集合）上无需梯度更新即可泛化工作。

### 2.2 问题的挑战性

**挑战一：跨Session的神经元身份漂移**。不同recording session之间，电极记录到的神经元集合可能完全不同（电极微移、信号漂移等）。传统方法将每个神经元视为固定的输入维度，导致模型与特定session强耦合。即使是同一被试的连续两天记录，神经元集合的重叠率也可能很低。这要求模型具备在不知道具体神经元身份的情况下，从群体活动模式中提取有意义表征的能力。

**挑战二：长时程时序依赖的捕获**。神经活动存在丰富的长程时间依赖性（如oscillatory dynamics、slow drift等），但spike train本身是高度稀疏且随机的（泊松过程或近似泊松过程）。将预测窗口从传统的几十毫秒扩展到1秒，信号的不确定性显著增加，预测难度呈指数级增长。如何在保持预测准确性的同时扩展时间范围，是核心技术挑战。

**挑战三：序列长度与计算效率**。以1ms分辨率处理1秒的数据，单个神经元就产生1000个时间步。对于数百个神经元，原始spike sequence长度可达数万甚至数十万。标准Transformer的 $O(N^2)$ 复杂度使得直接处理如此长的序列在计算上不可行。

**挑战四：多模态异构信息的有效融合**。行为数据（运动轨迹、速度等）是低维连续信号，视觉刺激（图像/视频）是高维结构化信号，而spiking数据是高维稀疏的点过程信号。这些异构模态的有效融合需要精心设计的架构。

### 2.3 问题的范围

本研究聚焦于以下边界条件：输入为侵入式电极记录的spiking数据；预测目标为固定time bin内的spike counts（而非单个spike event的精确时间）；跨session指同一脑区、同一或不同被试的不同recording session；多模态条件信息包括行为数据（如眼动，running speed等）和视觉刺激（如自然图像、光栅等）。

---

## 3. 相关工作

### 3.1 神经数据基础模型

#### 3.1.1 Binned Spike Count 方法

**NDT系列（Neural Data Transformer）**：NDT1 (Ye & Pandarinath, 2021) 首次将Transformer引入神经数据建模，使用masked prediction在binned spike counts上预测神经活动。NDT2/STNDT进一步引入时空分离的attention机制，提升了计算效率。NDT3 (Ye et al., 2024) 将规模扩展到多个数据集和脑区，验证了scaling law在神经数据上的适用性。优点是训练稳定、兼容传统分析流程；缺点是binning丢失了亚毫秒级时间信息，masked prediction范式不支持自回归生成。

**LFADS系列**：LFADS (Pandarinath et al., 2018) 及其变体使用变分序列自编码器（VAE）从binned spike counts中推断低维潜在动力学。优点是提供了可解释的潜在因子；缺点是模型容量有限，难以扩展到大规模数据。

#### 3.1.2 Spike-Level 方法

**Neuroformer** (Antoniades et al., 2024)：首个将自回归Transformer用于spike-level预测的工作。输入为 (neuron_id, timestamp) 的spike event序列，使用GPT-style causal attention进行next-spike prediction，并支持通过cross-attention引入behavior和image条件信息（图像通过对比学习编码）。优点是spike-level建模保留了完整时间信息，支持多模态；缺点是跨session能力较弱，自回归生成效率低（每次仅生成一个spike event）。

**POYO/POYO+** (Azabou et al., 2024)：提出将individual spikes作为tokens，通过unit-specific embedding和RoPE时间编码实现高时间分辨率建模。关键创新是引入Perceiver架构的cross-attention对spike序列进行压缩，将 $O(N_{\text{spikes}})$ 的序列压缩为固定长度的latent array，极大降低了计算成本。POYO+进一步实现了跨session和跨brain area的联合训练。优点是高时间分辨率、计算效率高、支持大规模训练；缺点是主要用于解码任务（predict behavior from spikes），缺乏自回归生成能力。

#### 3.1.3 跨Session方法

**SPINT** (Wei et al., 2024)：专门解决跨session泛化问题。核心创新是IDEncoder模块——通过将每个session中观察到的神经元活动模式编码为unit embedding，使模型能够在新session上无需梯度更新即可工作。IDEncoder利用每个神经元在一小段参考数据上的活动模式，通过一个共享的encoder网络生成该神经元的embedding。这种gradient-free的泛化机制是解决neural identity drift问题的优雅方案。

**Multi-task Masking (MtM)** (Ye et al., 2024)：通过随机mask输入和输出的神经元子集来实现跨session鲁棒性，是一种数据增强式的解决方案。

#### 3.1.4 其他相关工作

**LDNS** (Hurwitz et al., 2024)：将latent diffusion model应用于神经数据，通过去噪过程生成neural population dynamics。

**PopT** (Ye et al., 2024)：基于population-level tokenization的方法。

**POSSM** (Azabou et al., 2025)：POYO的后续工作，将SSM（State Space Model）引入spike sequence建模，用Mamba替代Transformer以处理更长的序列。

### 3.2 现有方法的对比与研究空白

| 特性 | NDT3 | Neuroformer | POYO+ | SPINT | POSSM | **Ours** |
|------|------|-------------|-------|-------|-------|----------|
| 时间分辨率 | Binned | Spike-level | Spike-level | Binned | Spike-level | Spike-level |
| 自回归生成 | ✗ | ✓ | ✗ | ✗ | ✗ | ✓ |
| 跨Session (gradient-free) | ✗ | ✗ | 有限 | ✓ | 有限 | ✓ |
| 长时程预测 (≥1s) | ✗ | 有限 | ✗ | ✗ | ✗ | ✓ |
| 多模态条件 | ✗ | ✓ | ✗ | ✗ | ✗ | ✓ |
| 计算效率 | 中 | 低 | 高 | 中 | 高 | 高 |

**研究空白总结**：目前没有一个模型能够同时实现 (1) spike-level的高时间分辨率输入、(2) gradient-free的跨session泛化、(3) 长时程（≥1s）的自回归预测生成、(4) 灵活的多模态条件输入。NeuroHorizon旨在填补这一空白。

---

## 4. 研究创新点

### 创新点一：基于Feed-Forward IDEncoder的跨Session Unit Embedding学习（主要创新）

借鉴SPINT的IDEncoder思想，但进行关键改进：我们设计一个feed-forward网络，从每个神经元在短暂参考窗口中的放电模式（firing pattern）直接学习该神经元的unit embedding，而非使用手工设计的特征或复杂的encoder。这种方法的优势在于：(1) 推理时无需梯度更新（gradient-free），只需将新session的参考数据通过前馈网络即可获得unit embedding；(2) 通过end-to-end训练，unit embedding能够捕获对下游任务最有用的神经元特征表征；(3) 支持动态适应——即使在同一session内神经元特性发生漂移，也可通过更新参考窗口实时调整embedding。

**创新理由**：SPINT的IDEncoder已证明了gradient-free跨session泛化的可行性，但其主要用于解码任务。我们将这一机制扩展到编码/生成任务中，并通过feed-forward设计简化了推理流程。与POYO+的per-unit learnable embedding相比，我们的方法不需要将新神经元加入训练集即可泛化。

### 创新点二：扩展时间窗口的自回归Spike Count预测（主要创新）

将输入时间窗口扩展至1秒（相比Neuroformer的约200ms），并设计基于cross-attention的decoder进行自回归预测。与Neuroformer逐spike生成的方式不同，我们的decoder在固定time bins中预测spike counts。具体而言：encoder将1秒内的所有spike events编码为latent representation，decoder通过cross-attention attend到encoder输出，并在一系列预定义的time bins上自回归地生成各神经元的spike count预测。

**创新理由**：(1) 预测spike counts而非individual spikes大幅降低了序列长度和生成难度，同时保留了对群体活动模式的捕获能力；(2) 1秒的时间窗口能够捕获更丰富的时间依赖性（如theta oscillation ~4-8Hz的完整周期）；(3) cross-attention解码器允许模型选择性地关注encoder中最相关的时间段和神经元，提升长时程预测的准确性。

### 创新点三：多模态条件注入与科学发现

通过cross-attention机制灵活注入多模态条件信息。对于行为数据（如运动轨迹、速度），使用线性投影后通过cross-attention与neural representation交互；对于视觉刺激（如自然图像），直接使用预训练的DINOv2模型提取image embedding后注入（无需像Neuroformer那样进行额外的对比学习训练），简化了训练流程并利用了DINOv2强大的视觉表征能力。

**创新理由**：(1) 直接使用DINOv2替代对比学习是一个简洁而有效的设计选择——DINOv2已在大规模视觉数据上预训练，其特征空间自然适合表征visual stimuli；(2) 通过分析不同模态对预测准确率的贡献，可以进行有价值的科学发现，如量化视觉刺激vs.内在动力学对V1神经元活动的预测贡献。

### 创新点四（可选）：基于Perceiver的序列压缩与SSM高效推理

借鉴POYO的Perceiver cross-attention机制压缩spike序列长度，降低encoder端的计算复杂度。在需要实时推理的场景下，可选择将Transformer backbone替换为State Space Model（如Mamba），进一步提升长序列处理的效率（$O(N)$ 复杂度替代 $O(N^2)$）。

---

## 5. 方法设计

### 5.1 总体框架

NeuroHorizon采用Encoder-Decoder架构，总体流程如下：

```
输入：Spike Events {(neuron_id, timestamp)} + 可选多模态条件信息
  │
  ▼
[Step 1] Spike Tokenization & Embedding
  │  - 每个spike event → token
  │  - Unit Embedding (via IDEncoder) + Temporal Embedding (RoPE)
  │
  ▼
[Step 2] (可选) Perceiver Cross-Attention 序列压缩
  │  - 将变长spike序列压缩为固定长度latent array
  │
  ▼
[Step 3] Transformer/SSM Encoder
  │  - 编码神经群体时空动态
  │  - Multi-modal cross-attention (behavior/image条件注入)
  │
  ▼
[Step 4] Cross-Attention Decoder (自回归)
  │  - 在固定time bins上预测spike counts
  │  - 逐time bin自回归生成
  │
  ▼
输出：未来时间窗口的spike count predictions {count_{neuron, time_bin}}
```

### 5.2 模块一：Spike Tokenization与Embedding

#### 5.2.1 Token定义

每个spike event $(n_i, t_i)$ 被视为一个独立token，保留了spike-level的时间分辨率。输入序列为：

$$
\mathcal{X} = [(n_1, t_1), (n_2, t_2), \ldots, (n_K, t_K)]
$$

其中按时间戳排序，$K$ 为1秒窗口内的总spike数量（通常数百到数千）。

#### 5.2.2 Unit Embedding (IDEncoder)

对每个神经元 $n$，其unit embedding通过feed-forward IDEncoder网络生成：

$$
\mathbf{u}_n = \text{IDEncoder}(\mathbf{r}_n) = \text{FFN}(\mathbf{r}_n)
$$

其中 $\mathbf{r}_n \in \mathbb{R}^{d_{\text{ref}}}$ 是神经元 $n$ 在参考窗口（如session开始的10-30秒）中的神经活动特征向量。$\mathbf{r}_n$ 可以由以下特征构成：参考窗口内的平均firing rate、ISI（inter-spike interval）分布统计量、自相关函数特征、波形特征（如有）等。

IDEncoder的具体结构为一个共享参数的多层前馈网络：

$$
\text{FFN}: \mathbb{R}^{d_{\text{ref}}} \rightarrow \mathbb{R}^{d_{\text{model}}}
$$

$$
\mathbf{u}_n = W_3 \cdot \text{GELU}(W_2 \cdot \text{GELU}(W_1 \cdot \mathbf{r}_n + b_1) + b_2) + b_3
$$

**关键设计**：IDEncoder的参数在训练时通过反向传播学习，但在推理时对新session的神经元只需做一次前向传播即可获得embedding，实现gradient-free泛化。

#### 5.2.3 Temporal Embedding (RoPE)

借鉴POYO的设计，使用Rotary Position Embedding (RoPE) 编码连续时间戳信息：

$$
\mathbf{p}(t_i) = \text{RoPE}(t_i)
$$

RoPE的优势在于：(1) 能够编码任意精度的连续时间值（不受离散binning限制）；(2) 自然支持位置的相对关系编码（attention score只依赖时间差而非绝对时间）；(3) 可外推到训练中未见过的时间范围。

#### 5.2.4 Token Embedding

每个spike token的完整embedding为：

$$
\mathbf{h}_i^{(0)} = \mathbf{u}_{n_i} + \mathbf{e}_{\text{spike}}
$$

其中 $\mathbf{e}_{\text{spike}}$ 是一个可学习的spike type embedding（与POYO类似，用于区分spike tokens和其他类型的tokens）。RoPE时间编码在attention计算时直接应用于query和key，而非加到token embedding上。

### 5.3 模块二：序列压缩（可选）

当spike序列过长时（如>2000 spikes/秒），可借鉴POYO的Perceiver cross-attention进行压缩：

$$
\mathbf{Z}_{\text{latent}} = \text{CrossAttn}(\mathbf{Q}_{\text{latent}}, \mathbf{K}_{\text{spikes}}, \mathbf{V}_{\text{spikes}})
$$

其中 $\mathbf{Q}_{\text{latent}} \in \mathbb{R}^{M \times d}$ 是 $M$ 个可学习的latent query tokens（$M \ll K$，如 $M=128$ 或 $256$），$\mathbf{K}$ 和 $\mathbf{V}$ 来自spike token embeddings。

这一步将变长的spike序列压缩为固定长度 $M$ 的latent representation，后续encoder的计算复杂度从 $O(K^2)$ 降为 $O(M^2)$。

### 5.4 模块三：Encoder

#### 5.4.1 主干架构

默认采用标准Transformer encoder，由 $L$ 层self-attention + FFN组成：

$$
\mathbf{H}^{(l)} = \text{TransformerLayer}(\mathbf{H}^{(l-1)}), \quad l = 1, \ldots, L
$$

每层包括：
- Multi-Head Self-Attention with RoPE
- Layer Normalization (Pre-LN)
- Feed-Forward Network (SwiGLU activation)
- Residual Connections

#### 5.4.2 可选：SSM替代方案

对于需要处理超长序列或要求实时推理的场景，可将Transformer替换为State Space Model（如Mamba）：

$$
\mathbf{H}^{(l)} = \text{MambaLayer}(\mathbf{H}^{(l-1)})
$$

SSM的优势是 $O(N)$ 的序列长度复杂度和高效的推理（无需KV cache），适合streaming/实时场景。

#### 5.4.3 多模态条件注入

在encoder的特定层（如每隔2层），通过cross-attention注入多模态条件信息：

**行为数据 (Behavior)**：
$$
\mathbf{c}_{\text{beh}} = \text{Linear}(\mathbf{x}_{\text{behavior}}) \in \mathbb{R}^{T_b \times d}
$$
$$
\mathbf{H}^{(l)} = \mathbf{H}^{(l)} + \text{CrossAttn}(\mathbf{H}^{(l)}, \mathbf{c}_{\text{beh}}, \mathbf{c}_{\text{beh}})
$$

**视觉刺激 (Image)**：
$$
\mathbf{c}_{\text{img}} = \text{DINOv2}(\mathbf{I}) \in \mathbb{R}^{N_p \times d_{\text{dino}}}
$$
$$
\mathbf{c}_{\text{img}}' = \text{Linear}(\mathbf{c}_{\text{img}}) \in \mathbb{R}^{N_p \times d}
$$
$$
\mathbf{H}^{(l)} = \mathbf{H}^{(l)} + \text{CrossAttn}(\mathbf{H}^{(l)}, \mathbf{c}_{\text{img}}', \mathbf{c}_{\text{img}}')
$$

其中DINOv2是冻结的预训练视觉模型，仅训练投影层。这避免了Neuroformer中额外的对比学习训练阶段，简化了整体训练流程。

### 5.5 模块四：Cross-Attention Decoder（长时程自回归预测）

#### 5.5.1 预测目标定义

将预测时间窗口 $[T, T+\Delta T]$ 划分为 $B$ 个固定time bins，每个bin宽度为 $\delta$（如 $\delta = 10\text{ms}$ 或 $20\text{ms}$）。预测目标为每个time bin中每个神经元的spike count：

$$
\hat{y}_{n,b} = \text{predicted spike count of neuron } n \text{ in time bin } b
$$

#### 5.5.2 解码流程

Decoder采用自回归方式逐time bin生成预测。对于第 $b$ 个time bin：

1. **Target query构建**：
$$
\mathbf{q}_b = \mathbf{e}_{\text{bin}} + \text{RoPE}(t_b)
$$
其中 $\mathbf{e}_{\text{bin}}$ 是可学习的bin type embedding，$t_b$ 是第 $b$ 个bin的中心时间。

2. **Cross-attention到encoder输出**：
$$
\mathbf{d}_b = \text{CrossAttn}(\mathbf{q}_b, \mathbf{H}_{\text{enc}}, \mathbf{H}_{\text{enc}})
$$

3. **Causal self-attention**（仅attend到已预测的bins）：
$$
\mathbf{d}_b' = \text{CausalSelfAttn}(\mathbf{d}_b, \mathbf{d}_{1:b-1})
$$

4. **Spike count预测**：
$$
\hat{y}_{n,b} = \text{MLP}([\mathbf{d}_b'; \mathbf{u}_n])
$$
输出经过softmax或Poisson parameter预测。

#### 5.5.3 损失函数

主要采用Poisson negative log-likelihood loss：

$$
\mathcal{L}_{\text{pred}} = -\sum_{b=1}^{B} \sum_{n=1}^{N} \left[ y_{n,b} \log \hat{\lambda}_{n,b} - \hat{\lambda}_{n,b} - \log(y_{n,b}!) \right]
$$

其中 $\hat{\lambda}_{n,b}$ 是模型预测的Poisson rate parameter。也可探索zero-inflated Poisson或negative binomial分布以更好地处理spike count数据的over-dispersion特性。

总损失函数：
$$
\mathcal{L} = \mathcal{L}_{\text{pred}} + \alpha \mathcal{L}_{\text{reg}}
$$

其中 $\mathcal{L}_{\text{reg}}$ 包含可选的正则化项（如unit embedding的smooth regularization）。

### 5.6 训练策略

#### 5.6.1 多Session联合训练

收集来自多个session的数据，每个batch可包含来自不同session的样本。IDEncoder在训练时为每个session的每个神经元动态生成embedding，确保模型学习到session-invariant的表征。

#### 5.6.2 数据增强

- **Neuron Dropout**：随机丢弃一定比例（如20%）的神经元，模拟跨session时神经元缺失的情况
- **Temporal Jitter**：对spike时间戳添加小幅度的随机扰动（如±1ms），增强时间鲁棒性
- **Reference Window Augmentation**：随机选择不同的参考窗口用于IDEncoder输入

#### 5.6.3 课程学习

初始训练时使用较短的预测窗口（如100ms），逐步扩展到完整的1秒，帮助模型逐渐学习长程依赖。

---

## 6. 实验设置

### 6.1 计算资源

- **训练硬件**：NVIDIA A100/H100 GPU × 4-8（根据数据规模调整）
- **训练时间预估**：完整训练约1-2周（取决于数据规模和模型大小）
- **推理硬件**：单张GPU即可完成推理

### 6.2 软件环境

- **编程语言**：Python 3.10+
- **深度学习框架**：PyTorch 2.x + PyTorch Lightning
- **关键库**：
  - `einops`：张量操作
  - `transformers` (HuggingFace)：RoPE实现参考
  - `mamba-ssm`：SSM相关模块（可选）
  - `wandb`：实验跟踪
  - `hydra`：配置管理
  - `allensdk`：Allen数据集接口
- **版本控制**：Git + GitHub

### 6.3 模型参数

| 参数 | Small | Base | Large |
|------|-------|------|-------|
| Encoder Layers | 4 | 8 | 12 |
| Decoder Layers | 2 | 4 | 6 |
| Hidden Dim | 256 | 512 | 768 |
| Attention Heads | 4 | 8 | 12 |
| Perceiver Latents | 64 | 128 | 256 |
| IDEncoder Layers | 2 | 3 | 4 |
| 参数量（约） | ~5M | ~30M | ~100M |

### 6.4 训练配置

- **优化器**：AdamW ($\beta_1=0.9, \beta_2=0.999$, weight decay=0.01)
- **学习率**：peak lr=1e-4, warmup 2000 steps, cosine decay
- **Batch Size**：64-256（梯度累积）
- **输入窗口**：1.0秒（1000ms）
- **预测窗口**：0.5-1.0秒
- **Time Bin大小**：10ms或20ms
- **Dropout**：0.1
- **Mixed Precision**：BF16

---

## 7. 数据集

### 7.1 数据集详细介绍

#### 7.1.1 Allen Brain Observatory — Neuropixels Visual Coding

- **来源**：Allen Institute for Brain Science
- **内容**：该数据集是Allen脑科学研究所大规模标准化神经记录项目的核心组成部分。使用Neuropixels高密度硅探针记录清醒、头部固定小鼠的视觉皮层及相关区域（包括V1、LM、AL、PM、AM等视觉区域以及LGN、LP等丘脑区域）的神经活动。实验范式包括多种精心设计的视觉刺激：自然场景图像（118张静态自然图像）、drifting gratings（不同方向和时间/空间频率的移动光栅）、static gratings（静态光栅）、natural movies（自然视频片段，约30秒循环播放）、以及Gabor patches等。每个stimulus block持续时间通常为数十秒到数分钟，刺激间有固定的灰屏间隔。数据包含spike-sorted unit信息（通过Kilosort2处理）、LFP信号、以及running speed和eye tracking等行为数据。
- **规模**：约58个session（来自不同小鼠），每session同时记录约200-700个高质量units，跨越6个视觉皮层区域和2个丘脑核团。每个session的总记录时长约2-3小时。
- **优势**：(1) 极其标准化和可重复的实验范式，所有小鼠经历完全相同的刺激协议，便于跨session对比；(2) 丰富的视觉刺激元数据（每帧图像可用），是验证多模态（neural + image）融合的理想数据集；(3) 多个脑区的同步记录支持跨脑区分析；(4) 提供完善的Python API (AllenSDK)，数据获取和预处理流程成熟；(5) running speed和pupil size等行为数据可作为额外的条件输入。
- **局限**：(1) session数量相对有限（~58个），可能不足以充分测试大规模跨session scaling；(2) 被试为被动观看范式（head-fixed passive viewing），行为变量较少（仅running speed和eye position）；(3) 数据主要集中在视觉通路，不涉及运动皮层等其他脑区。
- **链接**：https://allensdk.readthedocs.io/
- **用途**：主要用于多模态实验（neural + visual stimuli）的核心验证，以及视觉皮层跨session泛化测试

#### 7.1.2 International Brain Laboratory (IBL) Dataset

- **来源**：International Brain Laboratory（国际脑实验室），由全球22个实验室组成的联盟
- **内容**：该数据集来自IBL的"Brain-wide Map"项目，是目前最大规模的标准化全脑Neuropixels记录数据集之一。所有实验室采用完全相同的行为任务范式——一个视觉决策任务（International Brain Laboratory task）：小鼠需要根据屏幕上出现的visual grating的对比度和位置（左/右），通过转动滚轮做出二选一决策。Neuropixels探针插入到全脑多个区域（包括视觉皮层、前额叶、纹状体、丘脑、海马、小脑等），提供了前所未有的全脑覆盖。数据包含高质量的spike-sorted units（经过统一的Kilosort + IBL pipeline处理）、详细的行为数据（wheel position/velocity、stimulus contrast、choice、reaction time、reward等）、以及视觉刺激参数。
- **规模**：数百个recording session，来自数十只小鼠和多个实验室，涵盖全脑超过200个脑区。总计记录了数万个神经元。每个session通常持续约1-2小时，包含数百个行为trial。
- **优势**：(1) **session数量极多**（数百个），是测试跨session泛化能力和data scaling的最佳选择；(2) 标准化的行为任务提供了丰富的、结构化的行为变量（stimulus、choice、reaction time、reward），非常适合neural + behavior多模态实验；(3) 跨实验室数据可测试模型的跨实验室泛化能力（更严格的泛化测试）；(4) 全脑覆盖使得可以验证模型在不同脑区上的通用性；(5) 数据通过统一的质量控制流程处理，一致性好；(6) POYO/POYO+已经在IBL数据集上进行了训练和评估，可直接对比。
- **局限**：(1) 行为任务相对简单（二选一决策），行为变量的复杂度有限；(2) 视觉刺激为简单的光栅（非自然图像），不适合测试复杂visual encoding；(3) 每个session的探针位置不同，脑区覆盖不均匀，部分脑区数据较少。
- **用途**：跨session泛化性核心测试、data scaling实验、neural + behavior多模态实验、跨脑区泛化分析

#### 7.1.3 Neural Latents Benchmark (NLB) Datasets

- **来源**：Neural Latents Benchmark，由Pandarinath Lab等多个课题组联合发起的神经数据建模基准测试
- **内容**：NLB包含多个经典的非人灵长类（NHP, macaque）运动皮层数据集，所有数据集均来自已发表的高影响力研究：
  - **MC_Maze**：猕猴在delayed reaching任务中的运动皮层（M1/PMd）记录。猕猴需要控制光标穿过不同复杂度的迷宫到达目标。包含丰富的运动学变量（hand position, velocity），约182个sorted units，约2500+trials。这是NLB中最常用的benchmark子集。
  - **MC_RTT**：Random Target Task，猕猴不断追踪随机出现的目标。连续的运动行为提供了丰富的运动动力学数据，约130个sorted units。
  - **Area2_Bump**：体感皮层Area 2的记录，猕猴在reaching任务中手臂受到随机力场扰动（bump perturbation）。约50-80个units，可用于研究感觉-运动整合。
  - **DMFC_RSG**：前额叶背内侧皮层记录，猕猴执行时间估计任务（Ready-Set-Go）。约150个units，关注timing和决策相关的神经活动。
- **规模**：每个数据集几十到约200个sorted neurons，每个数据集数百到数千个trials。总体规模相对较小，但质量极高且有标准化的评估协议。
- **优势**：(1) **标准化的benchmark**——NDT1/NDT2/NDT3、LFADS、AutoLFADS等主流方法均在此评测，结果可直接对比，是论文中必须呈现的对比实验；(2) 提供标准化的数据划分（train/val/test splits）和评估代码；(3) NHP数据质量高（well-isolated单units），spike sorting质量有保证；(4) 运动皮层数据具有较强的低维动力学结构，适合验证模型捕获latent dynamics的能力；(5) MC_Maze等数据集包含行为数据（hand kinematics），可用于behavior decoding下游任务评估。
- **局限**：(1) session数量极少（每个数据集通常只有1-2个session），**不适合跨session泛化实验**；(2) 没有视觉刺激信息，不支持image模态实验；(3) 数据规模较小，不适合scaling law研究；(4) 仅限运动皮层，不涉及视觉区域。
- **链接**：https://neurallatents.github.io/
- **用途**：主要用于与baseline方法的定量比较（长时程预测性能、latent quality评估），以及行为解码下游任务

#### 7.1.4 Jia Lab 实验室数据集

- **来源**：Jia Lab内部数据
- **内容**：多session的小鼠视觉皮层（V1为主）Neuropixels记录数据。实验包含多种视觉刺激条件下的神经响应记录，每个刺激条件有大量重复呈现（数十到上百次repeated trials），这是该数据集的核心优势。数据包含spike-sorted units、视觉刺激参数、以及可能的running speed等行为数据。实验设计注重刺激的系统性和重复性，为统计分析和长时程预测提供了坚实的数据基础。
- **规模**：多个session（跨不同小鼠和不同天的记录），每个session包含数百个sorted units和大量重复trials。单session记录时长可达数小时。
- **优势**：(1) **大量重复trials**是长时程预测实验的关键优势——高trial数使得可以可靠地评估单trial level的预测准确性，并计算trial-averaged PSTH作为ground truth参考；(2) 内部数据可灵活使用，不受外部数据集的使用限制；(3) 多session设计天然支持跨session泛化实验；(4) V1数据对视觉刺激有明确的编码关系，预测信号相对较强，适合作为方法验证的初始数据集；(5) 数据获取和预处理可以根据项目需求灵活调整。
- **局限**：(1) 数据未公开，实验结果的可重复性依赖于数据描述的详细程度；(2) session数量相比IBL较少；(3) 仅限视觉皮层，脑区覆盖有限。
- **用途**：长时程预测实验的核心验证数据集、跨session实验、多模态实验（neural + visual stimuli）

#### 7.1.5 Neuroformer / NDT3 使用的其他数据集

作为参考和对比，可能使用 Neuroformer 和 NDT3 论文中使用的数据集子集，确保结果的可比性。包括但不限于：Neuroformer使用的V1 two-photon calcium imaging数据集（用于验证自回归生成能力）、NDT3在大规模多session训练中使用的FALCON Benchmark数据集等。

### 7.2 NeuroHorizon项目的数据集推荐与使用策略

考虑到NeuroHorizon项目的三大核心目标——**长时程预测（最高优先级）、跨session泛化、多模态融合**——以下是推荐的数据集使用策略：

#### 7.2.1 核心推荐：Jia Lab数据集（长时程预测首选）

**推荐理由**：Jia Lab数据集的大量重复trials是长时程预测实验不可替代的优势。对于评估1秒甚至更长时间窗口的预测准确性，需要足够的trial数来可靠估计预测性能的统计显著性。同时，V1神经元对视觉刺激的响应具有较好的可预测性（尤其是stimulus-driven component），为验证长时程预测方法提供了信号较强的测试场景。建议作为长时程预测能力的**主要验证数据集**和**开发迭代的首选数据集**。

#### 7.2.2 核心推荐：IBL数据集（跨session泛化首选）

**推荐理由**：IBL数据集的数百个session是跨session泛化实验的最佳选择，也是POYO/POYO+已经使用的核心数据集，便于直接对比。其标准化的行为任务和丰富的行为变量使其同时适合neural + behavior的多模态实验。大量session还支持data scaling law研究。建议作为跨session泛化能力的**核心评估数据集**。

#### 7.2.3 重要补充：Allen Brain Observatory（多模态融合首选）

**推荐理由**：Allen数据集提供了最丰富、最标准化的视觉刺激信息（natural scenes、gratings、natural movies的完整图像帧），是验证neural + image多模态融合的理想选择。DINOv2 image embedding的有效性需要在具有复杂自然视觉刺激的数据上验证，Allen数据集完美满足这一需求。建议作为**多模态融合实验的主要数据集**。

#### 7.2.4 必要基准：NLB数据集（方法对比基准）

**推荐理由**：NLB是领域内公认的benchmark，论文投稿时必须呈现与现有方法（NDT1/2/3、LFADS等）在此数据集上的对比结果。虽然不适合跨session和多模态实验，但对于长时程预测性能的对比以及downstream behavior decoding任务的评估是必要的。建议作为**论文中baseline对比的标准数据集**。

#### 7.2.5 数据集使用优先级总结

| 优先级 | 数据集 | 核心用途 | NeuroHorizon目标对应 |
|--------|--------|---------|---------------------|
| ★★★ | Jia Lab | 长时程预测核心验证、开发迭代 | 长时程预测（首要） |
| ★★★ | IBL | 跨session泛化、data scaling | 跨session（次要）|
| ★★☆ | Allen Brain Observatory | 多模态融合验证（neural+image） | 多模态（第三） |
| ★★☆ | NLB | Baseline对比、downstream tasks | 方法对比基准 |

### 7.2 数据预处理

1. **Spike Sorting**：对原始数据使用Kilosort等工具进行spike sorting（如数据未预处理）
2. **质量筛选**：按照标准质量指标筛选units（如presence ratio > 0.9, ISI violation rate < 0.5%）
3. **时间对齐**：将spike times与行为事件/刺激呈现时间对齐
4. **参考窗口提取**：为每个session的每个unit提取IDEncoder所需的参考窗口数据
5. **数据划分**：按session划分train/val/test（确保test sessions在训练中未见过）

---

## 8. 预期结果与实验

### 8.1 实验一：跨Session泛化性测试

**目的**：验证IDEncoder机制的跨session泛化能力。

**设计**：
- 在多个session上联合训练模型
- 在完全未见过的新session上直接测试（gradient-free）
- 对比方法：(a) per-session训练的baseline; (b) 直接apply的无IDEncoder模型; (c) fine-tuning方案; (d) SPINT

**评估指标**：
- Co-smoothing bits/spike（参考NDT3）
- Spike count预测的Poisson log-likelihood
- R² of predicted firing rates vs. actual firing rates

**预期结果**：NeuroHorizon在新session上的gradient-free性能应接近或超过per-session训练的baseline，且显著优于无IDEncoder的直接apply方案。

### 8.2 实验二：Data Scaling特性研究

**目的**：研究模型性能随训练数据（session数量）和参数规模的变化规律。

**设计**：
- 固定模型大小，逐步增加训练session数量（10, 50, 100, 200+）
- 固定数据量，变化模型大小（Small/Base/Large）
- 绘制scaling curves

**预期结果**：验证neural data foundation model的scaling law——性能应随数据和参数的增加呈现可预测的提升趋势（类似NLP中的power-law scaling）。

### 8.3 实验三：长时程预测能力评估

**目的**：评估模型在不同预测时间跨度下的预测准确性。

**设计**：
- 固定1秒输入窗口，变化预测窗口长度：100ms, 200ms, 500ms, 1000ms
- 对比方法：Neuroformer（自回归spike生成）、NDT3（masked prediction）、LFADS
- 分析预测准确性随时间跨度的衰减曲线

**评估指标**：
- 不同time bin位置上的Poisson log-likelihood
- PSTH correlation（trial-averaged response prediction quality）
- 预测spike count与真实值的pearson correlation

**预期结果**：NeuroHorizon在长时间跨度（>500ms）上的预测性能显著优于现有方法，且衰减更为缓慢。

### 8.4 实验四：多模态条件贡献分析

**目的**：量化不同模态信息对预测准确率的贡献，进行科学发现。

**设计**：
- 模态组合实验：(a) neural only; (b) neural + behavior; (c) neural + image; (d) neural + behavior + image
- 在不同脑区上分析模态贡献差异（如V1对image模态敏感度 vs. motor cortex对behavior模态敏感度）
- 分析neural variability：计算多模态模型对trial-to-trial variability的解释程度

**评估指标**：
- 各模态组合的预测性能对比
- 模态贡献的归因分析（通过ablation或gradient-based attribution）
- $R^2$ improvement per modality

**预期结果**：
- 在视觉皮层数据上，image模态应显著提升预测性能
- 在运动皮层数据上，behavior模态应提供最大的性能提升
- 多模态融合应捕获更多的neural variability

### 8.5 实验五：下游任务泛化性测试

**目的**：验证预训练模型的通用表征质量。

**设计**：
- 冻结预训练的encoder，在下游任务上训练轻量级decoder
- 下游任务：(a) 行为解码（如reaching方向、手臂速度）; (b) 刺激分类（如呈现的图像类别）
- 对比方法：从头训练的task-specific模型、NDT3、POYO+

**评估指标**：
- 行为解码：$R^2$, velocity MSE
- 刺激分类：accuracy, F1-score

**可选扩展**：如果采用causal attention或SSM架构，可进行在线/实时解码实验，评估latency和streaming accuracy。

### 8.6 实验六：消融研究 (Ablation Study)

**目的**：验证各创新模块的独立贡献。

**消融实验列表**：

| 实验 | 消融内容 | 验证的贡献 |
|------|---------|-----------|
| A1 | 移除IDEncoder，使用random unit embedding | IDEncoder的跨session贡献 |
| A2 | IDEncoder替换为固定统计特征（非学习） | 端到端学习vs.手工特征 |
| A3 | 移除Perceiver压缩 | 序列压缩的效率vs.性能trade-off |
| A4 | 缩短输入窗口至200ms | 长时间窗口的价值 |
| A5 | 移除behavior模态 | 行为信息的贡献 |
| A6 | 移除image模态 | 视觉信息的贡献 |
| A7 | DINOv2替换为对比学习编码 | DINOv2 vs. contrastive learning |
| A8 | Transformer替换为Mamba | 架构选择的影响 |
| A9 | RoPE替换为fixed positional encoding | 时间编码方式的影响 |
| A10 | Poisson loss替换为MSE loss | 损失函数选择的影响 |

---

## 9. 创新性与局限性讨论

### 9.1 创新性总结

本研究的核心贡献在于提出NeuroHorizon——首个同时实现跨session鲁棒泛化和长时程自回归预测的神经脉冲编码模型。具体创新包括：(1) 将IDEncoder机制从解码任务扩展到编码/生成任务，实现gradient-free的跨session泛化；(2) 设计基于cross-attention的decoder用于在固定time bins上的长时程spike count预测，将预测窗口扩展至1秒；(3) 通过DINOv2简化多模态条件注入流程；(4) 提供了neural data scaling law的实证研究。

### 9.2 理论意义

从理论层面，本研究验证了神经数据基础模型的可扩展性假设——即通过增加数据和模型规模，可以持续提升对神经群体活动的建模能力。IDEncoder机制的成功将表明，神经元的功能特性可以从其活动模式中自动推断，无需先验的解剖学或电生理学知识。长时程预测能力将揭示神经群体动力学中的长程结构。

### 9.3 实际应用价值

- **脑机接口 (BCI)**：跨session鲁棒性减少了BCI系统每日重新校准的需求；长时程预测为更稳定的控制信号提供基础。
- **神经科学研究工具**：作为通用的neural encoding model，可用于数据增强、缺失数据插补、实验设计优化等。
- **临床应用**：为慢性植入电极的长期稳定解码提供技术支撑。

### 9.4 局限性

1. **预测粒度**：当前设计预测spike counts in bins而非individual spike times，可能无法捕获spike timing的精细信息（如precise temporal coding）。
2. **计算成本**：尽管引入了Perceiver压缩，大规模训练仍需要较多GPU资源。
3. **数据依赖**：IDEncoder的质量依赖于参考窗口数据的质量和长度，极短或极嘈杂的参考窗口可能导致泛化性能下降。
4. **脑区限制**：当前实验主要在视觉皮层和运动皮层验证，对其他脑区（如海马体、前额叶）的泛化性有待验证。
5. **物种限制**：主要使用小鼠和非人灵长类数据，跨物种泛化能力未知。

### 9.5 未来工作

1. 探索predict individual spike times的可能性（如通过扩散模型或flow-matching）
2. 将NeuroHorizon扩展到更多脑区和物种
3. 探索self-supervised pretraining策略（如masked spike prediction + autoregressive prediction的联合训练）
4. 在真实BCI系统中进行在线验证
5. 结合neural ODE等连续时间模型，进一步改善动力学建模

---

## 10. 参考文献

### 核心参考论文

1. **NDT1**: Ye, J., & Pandarinath, C. (2021). Representation learning for neural population activity with Neural Data Transformers. *NeurIPS*.
2. **NDT2/STNDT**: Ye, J., Collinger, J., Bhatt, S., & Pandarinath, C. (2022). Neural Data Transformers 2: Multi-context pretraining for neural spiking activity.
3. **NDT3**: Ye, J., et al. (2024). Neural Data Transformer 3: Scaling foundation models for neural spiking activity. *NeurIPS*.
4. **LFADS**: Pandarinath, C., et al. (2018). Inferring single-trial neural population dynamics using sequential auto-encoders. *Nature Methods*.
5. **Neuroformer**: Antoniades, A., et al. (2024). Neuroformer: Multimodal and Multitask Generative Pretraining for Brain Data. *ICLR*.
6. **POYO**: Azabou, M., et al. (2024). A Unified, Scalable Framework for Neural Population Decoding. *NeurIPS*.
7. **POYO+**: Azabou, M., et al. (2024). POYO+: Scaling Neural Foundation Models with Multi-Area and Multi-Task Learning.
8. **POSSM**: Azabou, M., et al. (2025). POSSM: State Space Models for Neural Population Spike Data.
9. **SPINT**: Wei, X., et al. (2024). SPINT: Session-invariant neural decoding via gradient-free ID encoding.
10. **MtM**: Ye, J., et al. (2024). Multi-task Masking for Cross-session Neural Decoding.
11. **LDNS**: Hurwitz, C., et al. (2024). Latent Diffusion for Neural Spiking Data.
12. **NEDS**: Karpowicz, B., et al. (2024). Neural Embedding for Data Sharing.
13. **PopT**: Ye, J., et al. (2024). PopT: Population Transformer.
14. **DINOv2**: Oquab, M., et al. (2024). DINOv2: Learning Robust Visual Features without Supervision. *TMLR*.
15. **Perceiver**: Jaegle, A., et al. (2021). Perceiver: General Perception with Iterative Attention. *ICML*.
16. **RoPE**: Su, J., et al. (2024). RoFormer: Enhanced Transformer with Rotary Position Embedding.
17. **Mamba**: Gu, A., & Dao, T. (2024). Mamba: Linear-Time Sequence Modeling with Selective State Spaces. *ICLR*.

### 数据集引用

18. Allen Institute for Brain Science. Allen Brain Observatory — Neuropixels Visual Coding. https://allensdk.readthedocs.io/
19. International Brain Laboratory. IBL Brain-wide Map Dataset. https://www.internationalbrainlab.com/
20. Neural Latents Benchmark. https://neurallatents.github.io/

---

## 11. 执行计划

> **核心原则**：本项目以POYO/POYO+的开源代码库为基础（https://github.com/mehdiazabou/poyo-1），采用**增量式开发**策略。每个阶段在已验证的代码基础上进行最小化修改，确保每一步的改动都可控、可调试、可回滚。避免从零开始重写，最大限度复用POYO的成熟组件（数据加载、tokenization、Perceiver encoder、训练框架等）。

### 11.1 时间规划总览

**总预计时间**：16-20周（约4-5个月）

```
Phase 1: 代码熟悉与基线复现         ████░░░░░░░░░░░░░░░░  [Week 1-4]
Phase 2: 增量式模型改进             ░░░░████████░░░░░░░░  [Week 5-12]
Phase 3: 完整实验与数据集扩展       ░░░░░░░░░░░░████░░░░  [Week 12-16]
Phase 4: 论文撰写与修改             ░░░░░░░░░░░░░░░░████  [Week 16-20]
```

### 11.2 Phase 1: 代码熟悉与基线复现（第1-4周）

**目标**：完全理解POYO代码架构，在目标数据集上复现POYO的baseline性能，建立实验基线。

**Step 1.1：POYO代码库深度阅读与环境搭建（Week 1）**
- Fork POYO/POYO+代码库，搭建完整的开发环境（PyTorch, wandb, hydra等）
- 系统阅读代码结构：重点理解 (a) spike tokenization模块；(b) unit embedding（per-unit learnable embedding）的实现方式；(c) Perceiver cross-attention encoder的实现细节；(d) RoPE时间编码的具体实现；(e) 训练循环、数据加载器、配置管理
- 同步阅读POYO/POYO+论文，将代码实现与论文描述逐一对应
- 整理POYO代码模块依赖关系图，标注后续需要修改的模块
- **关键论文精读**：SPINT（IDEncoder机制）、Neuroformer（自回归生成 + 多模态）

**Step 1.2：数据集获取与适配（Week 2）**
- **IBL数据集**（跨session核心）：通过IBL的ONE API下载数据，理解POYO代码中IBL数据的加载方式，验证数据可以正常流入POYO的training pipeline
- **Jia Lab数据集**（长时程预测核心）：整理内部数据格式，编写数据适配器使其兼容POYO的数据加载框架（实现 `JiaLabDataset` 类，继承POYO的基础数据类）
- **Allen数据集**（多模态核心）：通过AllenSDK下载Neuropixels Visual Coding数据，编写Allen数据适配器，特别注意提取视觉刺激帧与spike times的时间对齐
- **NLB数据集**（基准对比）：下载MC_Maze等子集，编写NLB数据适配器
- 验证方式：所有数据集都能成功加载并通过POYO数据pipeline的sanity check

**Step 1.3：POYO基线复现（Week 3-4）**
- 在IBL数据集上复现POYO的behavior decoding性能（R², velocity MSE），确认与论文报告的数值一致或接近
- 在Jia Lab数据集上运行POYO（可能需要调整输出head），记录baseline性能
- 分析POYO在当前目标任务上的表现：(a) 评估POYO encoder输出的latent representation质量；(b) 测试POYO在不同时间窗口长度下的性能；(c) 初步评估POYO的跨session表现（在held-out session上直接评估）
- **交付成果**：POYO baseline性能报告（各数据集）、代码模块分析文档、明确的增量改进路线图

### 11.3 Phase 2: 增量式模型改进（第5-12周）

> **核心策略**：每个增量步骤只修改一个模块，在验证该改动有效后再进行下一步。每个步骤都有明确的A/B实验对比（改动前 vs 改动后）。优先实现与长时程预测直接相关的改动。

**Increment 1：添加自回归Decoder用于长时程spike count预测（Week 5-7，3周）**

这是NeuroHorizon与POYO最核心的差异——POYO仅用于解码（spikes → behavior），而NeuroHorizon需要预测未来的neural activity。

- **Week 5**：在POYO encoder输出之上，添加cross-attention decoder模块
  - 保持POYO encoder完全不变，仅添加新的decoder head
  - 实现time bin query embedding + RoPE时间编码
  - 实现cross-attention（decoder queries attend to encoder output）
  - 实现causal self-attention mask（确保自回归特性）
  - 实现spike count输出层（MLP: latent → per-neuron Poisson rate）
  - 实现Poisson NLL loss
- **Week 6**：在Jia Lab数据集上训练和调试
  - 先用teacher forcing模式训练（提供ground truth previous bins作为decoder输入）
  - 验证loss是否收敛，检查预测的spike count分布是否合理
  - 逐步开启自回归模式（inference时使用模型自己的预测作为输入）
  - 数据集选择理由：Jia Lab大量重复trials提供充足训练数据，V1响应可预测性较强
- **Week 7**：长时程预测性能评估与调优
  - 实验：固定1s输入窗口，逐步扩展预测窗口（100ms → 200ms → 500ms → 1000ms）
  - 实现课程学习策略：训练初期使用短预测窗口，逐步增长
  - A/B对比：NeuroHorizon (POYO encoder + new decoder) vs. 简单baseline（如线性预测、PSTH-based预测）
  - 验证方式：预测spike count的Poisson log-likelihood、pearson correlation with ground truth firing rates
  - **关键检查点**：长时程预测在Jia Lab数据上是否显著优于简单baseline？如果否，需要分析原因并调整decoder设计（如增加decoder层数、调整time bin宽度等）

**Increment 2：用IDEncoder替换POYO的Per-Unit Learnable Embedding（Week 8-9，2周）**

这是实现跨session泛化的核心改动。

- **Week 8**：实现IDEncoder模块并集成到POYO框架
  - 在POYO代码中定位unit embedding模块（per-unit learnable embedding lookup table）
  - 实现feed-forward IDEncoder网络：输入为参考窗口神经活动特征向量，输出为unit embedding
  - 实现参考窗口特征提取函数：从每个session开始的10-30秒数据中计算每个neuron的特征向量（firing rate, ISI statistics, autocorrelation features等）
  - 将POYO的 `nn.Embedding(num_units, d_model)` 替换为 `IDEncoder(d_ref, d_model)`
  - 确保替换后模型的前向传播正常运行，维度匹配
- **Week 9**：在IBL数据集上验证跨session泛化
  - 训练配置：多session联合训练（训练集包含多个session），实现neuron dropout数据增强
  - 核心评估：在完全未见过的test session上，仅通过IDEncoder前向传播生成unit embedding（gradient-free），测试模型性能
  - A/B对比：(a) NeuroHorizon with IDEncoder vs. (b) POYO原始per-unit embedding（需要为新session添加新embedding并fine-tune） vs. (c) 无IDEncoder的random embedding baseline
  - 同时在Jia Lab数据集上验证IDEncoder不损害长时程预测性能
  - **关键检查点**：IDEncoder的gradient-free跨session性能是否接近per-session fine-tuning？如果差距过大，考虑增加参考窗口长度或引入额外特征

**Increment 3：添加多模态条件注入（Week 10-11，2周）**

- **Week 10**：实现behavior模态注入
  - 在POYO encoder的指定层添加cross-attention模块（encoder hidden states attend to behavior embeddings）
  - 行为数据处理：线性投影 behavior features → d_model维度
  - 在IBL数据集上训练和验证（IBL有丰富的行为变量：wheel velocity, stimulus contrast, choice等）
  - A/B对比：neural only vs. neural + behavior
  - 数据集选择理由：IBL的标准化行为任务提供结构化的behavior variables
- **Week 11**：实现image模态注入
  - 集成DINOv2预训练模型（冻结权重），提取image patch embeddings
  - 添加线性投影层：DINOv2 output dim → d_model
  - 在encoder中添加image cross-attention（与behavior cross-attention类似的接口）
  - 在Allen数据集上训练和验证（Allen有完整的visual stimuli帧数据）
  - A/B对比：neural only vs. neural + image vs. neural + behavior + image
  - 数据集选择理由：Allen数据集的natural scenes和gratings提供了最丰富的视觉刺激信息
  - **关键检查点**：多模态是否在Allen V1数据上带来显著预测性能提升？分析不同visual stimulus类型（natural scenes vs. gratings）下的modality contribution差异

**Increment 4：优化与扩展实验（Week 12，1周）**

- 将Increment 1-3的所有改动整合为完整的NeuroHorizon模型
- 全面测试整合后的模型在各数据集上的性能
- 实现Perceiver压缩的可选开关（对于spike数量特别大的session启用）
- 针对发现的问题进行局部调优（如学习率调整、decoder层数、time bin宽度等）
- 准备完整实验所需的配置文件和实验脚本

### 11.4 Phase 3: 完整实验与数据集扩展（第12-16周）

**Step 3.1：核心实验矩阵（Week 12-14）**

在所有目标数据集上运行完整实验矩阵：

| 实验 | 数据集 | 目标 |
|------|--------|------|
| 长时程预测 (100ms-1s) | Jia Lab (主), NLB MC_Maze (辅) | 验证预测窗口扩展能力 |
| 跨Session泛化 | IBL (主), Jia Lab (辅) | 验证IDEncoder gradient-free泛化 |
| Data Scaling | IBL | 验证scaling law (10/50/100/200+ sessions) |
| 多模态-Behavior | IBL (主) | 量化behavior modality贡献 |
| 多模态-Image | Allen (主), Jia Lab (辅) | 量化image modality贡献 |
| 多模态-全模态 | Allen (neural+image+behavior) | 验证全模态融合效果 |
| Baseline对比 | NLB (必须), IBL, Allen | 与NDT3/LFADS/Neuroformer对比 |
| 下游任务 | IBL (behavior decoding), Allen (stimulus classification) | 验证通用表征质量 |

**Step 3.2：消融实验（Week 14-15）**
- 按照第8.6节的消融实验列表逐一执行
- 所有消融实验在同一数据集（建议IBL或Jia Lab）上进行，确保可比性
- 特别关注：IDEncoder消融（A1, A2）和长时程窗口消融（A4）

**Step 3.3：结果分析与可视化（Week 15-16）**
- 绘制scaling curves、预测性能随时间窗口的衰减曲线、模态贡献归因图
- 生成论文所需的所有图表和表格
- 撰写实验分析报告

**交付成果**：完整实验结果表、性能对比图表、分析报告

### 11.5 Phase 4: 论文撰写与修改（第16-20周）

- **Week 16-17**：撰写论文初稿（方法、实验章节优先）
- **Week 18-19**：内部审阅和修改，补充实验
- **Week 19-20**：完善introduction、related work、discussion，最终定稿，准备补充材料

**交付成果**：完整研究论文（目标NeurIPS/ICLR投稿格式）

### 11.6 里程碑与检查点

| 里程碑 | 时间 | 检验标准 |
|--------|------|---------|
| M1: POYO代码完全理解 | Week 1 | 能绘制完整的模块依赖关系图 |
| M2: 数据Pipeline可用 | Week 2 | 4个数据集均可通过POYO数据流水线加载 |
| M3: POYO基线复现 | Week 4 | IBL上复现POYO报告的behavior decoding性能（R²误差<5%） |
| M4: 长时程预测初步验证 | Week 7 | Jia Lab数据上1s预测显著优于简单baseline |
| M5: 跨Session初步验证 | Week 9 | IBL上gradient-free泛化性能>random baseline且接近fine-tuning |
| M6: 多模态验证 | Week 11 | Allen数据上多模态性能>neural only |
| M7: 完整模型整合 | Week 12 | 所有模块整合后的模型可稳定训练 |
| M8: 完整实验结果 | Week 16 | 所有核心实验和消融实验完成，结果可复现 |
| M9: 论文初稿 | Week 18 | 完整草稿可供内部评审 |
| M10: 论文定稿 | Week 20 | 达到目标会议投稿标准 |

### 11.7 增量式开发的关键原则

1. **最小改动原则**：每个Increment只修改一个模块，保持其他模块不变。这使得每次改动的效果可以清晰归因。
2. **持续测试原则**：每个Increment都有明确的A/B对比实验。改动后模型的性能至少不应低于改动前（除非该改动引入了新的、更难的任务目标）。
3. **代码版本管理**：每个Increment完成后创建Git tag，确保可以随时回退到任何一个稳定版本。
4. **数据集匹配原则**：每个改动选择最适合验证该改动效果的数据集进行初始测试（如长时程预测用Jia Lab，跨session用IBL，多模态用Allen），验证通过后再在其他数据集上泛化测试。
5. **基于POYO代码的具体修改清单**：
   - **不修改**：spike tokenization, RoPE temporal encoding, Perceiver cross-attention, Transformer encoder backbone, 训练框架（PyTorch Lightning, wandb, hydra）
   - **替换**：unit embedding模块（per-unit learnable → IDEncoder feed-forward）
   - **新增**：cross-attention decoder（自回归spike count预测）、多模态cross-attention layers（behavior/image条件注入）、Poisson NLL loss、数据集适配器（Jia Lab, Allen, NLB）

### 11.8 风险评估与应对方案

**风险1：IDEncoder跨session泛化效果不佳**
- 可能原因：参考窗口信息不足以区分神经元功能特性
- 应对方案：(a) 增加参考窗口长度；(b) 引入额外特征（如waveform shape）；(c) 采用更复杂的encoder架构（如小型Transformer而非FFN）；(d) 结合contrastive learning进行预训练

**风险2：长时程预测准确性不足**
- 可能原因：1秒窗口内的spike count过于稀疏/随机，难以预测
- 应对方案：(a) 增大time bin宽度（如50ms instead of 10ms）；(b) 采用概率预测（输出分布而非点估计）；(c) 引入latent variable model捕获neural variability；(d) 缩短预测窗口作为折中

**风险3：计算资源不足**
- 可能原因：大规模多session训练超出可用GPU显存/时间
- 应对方案：(a) 使用gradient accumulation减少显存需求；(b) 采用mixed precision (BF16)训练；(c) 优先使用Perceiver压缩减少序列长度；(d) 从Small模型开始验证

**风险4：多模态融合未带来显著提升**
- 可能原因：DINOv2特征与neural activity的correspondence不够直接
- 应对方案：(a) 添加learnable adapter layers；(b) 尝试不同的DINOv2变体（ViT-S/B/L）；(c) 在visual cortex数据上重点测试（模态匹配度最高）

**风险5：与现有方法对比不公平**
- 可能原因：baseline方法的实现或调参不充分
- 应对方案：(a) 尽量使用原始代码库；(b) 在原论文报告的数据集上先复现原始结果；(c) 联系原作者确认实验设置

**风险6：POYO代码修改导致原有功能回退**
- 可能原因：增量修改时无意中破坏了POYO原有的数据流或计算逻辑
- 应对方案：(a) 每个Increment前先运行POYO原始baseline测试，确认基线性能未变化；(b) 使用Git分支管理，每个Increment在独立分支开发；(c) 编写关键模块的单元测试（如IDEncoder输出维度、decoder因果mask正确性等）

---

## 附录A：符号表

| 符号 | 含义 |
|------|------|
| $\mathcal{S}$ | 输入spike event集合 |
| $(n_i, t_i)$ | 第$i$个spike event（神经元ID, 时间戳） |
| $\mathbf{u}_n$ | 神经元$n$的unit embedding |
| $\mathbf{r}_n$ | 神经元$n$的参考窗口特征向量 |
| $\mathbf{H}$ | Encoder隐层表征 |
| $\hat{\lambda}_{n,b}$ | 预测的Poisson rate（神经元$n$, time bin $b$） |
| $\mathcal{C}$ | 多模态条件信息 |
| $T$ | 输入窗口长度（默认1s） |
| $\Delta T$ | 预测窗口长度 |
| $\delta$ | Time bin宽度 |
| $B$ | 预测time bin数量 |

---

*本研究计划书创建于2026年2月13日，将随研究进展持续更新。*
