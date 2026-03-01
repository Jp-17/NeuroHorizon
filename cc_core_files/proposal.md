# NeuroHorizon: 跨Session鲁棒的长时程神经脉冲数据预测编码模型

**项目名称**：NeuroHorizon — 面向跨Session鲁棒性与长时程预测的可扩展神经脉冲编码模型

**研究领域**：计算神经科学 / 神经数据建模 / 脑机接口

**目标会议/期刊**：NeurIPS / ICLR / Nature Methods

---

## 项目概述

NeuroHorizon 是一个基于 POYO/POYO+（NeurIPS 2023 / ICLR 2025）框架改造的统一神经编码模型，旨在同时实现 **gradient-free 的跨session泛化**、**长时程自回归神经活动预测**，以及**多模态条件融合与可解释性分析**。

> 详细研究背景、研究意义、研究动机与相关工作综述，参见 `cc_core_files/background.md`。

---

## 目录

1. [核心挑战](#1-核心挑战)
2. [问题定义](#2-问题定义)
3. [研究创新点](#3-研究创新点)
4. [方法设计与创新模块实现](#4-方法设计与创新模块实现)
5. [数据集](#5-数据集)
6. [实验设计](#6-实验设计)
7. [可能的风险](#7-可能的风险)
8. [参考文献](#8-参考文献)
附录A：[符号表](#附录a符号表)

---

## 1. 核心挑战

NeuroHorizon 聚焦于当前神经数据基础模型尚未统一解决的三大核心挑战：

1. **跨Session神经元身份漂移**：不同recording session之间，电极记录到的神经元集合可能完全不同。传统方法将每个神经元视为固定输入维度，导致模型与特定session强耦合，需要为每个新session重新训练或微调。
2. **长时程神经活动预测**：当前方法主要关注短时间窗口（几十毫秒）内的预测，长时程预测几乎未被系统性地探索。将预测窗口显著延长时，spike train的稀疏性和随机性使不确定性急剧增加。
3. **多模态融合与可解释性**：行为数据、视觉刺激等多模态条件信息如何有效融合以辅助神经活动预测，以及各模态在不同实验状态下对预测的贡献如何量化，是尚待解决的问题。

> 更多关于构建 Spike Foundation Model 的通用技术挑战，详见 `cc_core_files/background.md` 第2节。

---

## 2. 问题定义

### 2.1 核心问题

**形式化定义**：给定来自任意recording session $s$ 的一段历史时间窗口内的神经群体spiking活动 $\mathcal{S} = \{(n_i, t_i)\}_{i=1}^{N_{\text{spikes}}}$，其中 $n_i$ 为神经元标识、$t_i$ 为spike时间戳，以及可选的多模态条件信息 $\mathcal{C}$（行为数据、视觉刺激等），我们的目标是：

1. **编码**：学习一个通用的神经群体活动表征 $\mathbf{Z} = f_\theta(\mathcal{S}, \mathcal{C})$，该表征捕获神经群体的时空动态模式。

2. **长时程预测**：基于编码表征，通过自回归解码器生成未来预测窗口 $[T, T + \Delta T]$ 内各神经元在固定time bins中的spike counts预测 $\hat{\mathbf{Y}} = g_\phi(\mathbf{Z})$。我们将系统性地探索不同预测窗口长度（从250ms到更长时间尺度）下的预测性能边界，以确定模型的实际长时程预测能力。

3. **跨Session泛化**：模型在新的session $s'$（可能包含完全不同的神经元集合）上无需梯度更新即可泛化工作。

4. **多模态可解释性**：将多模态条件分解为独立子集 $\mathcal{C} = \{\mathcal{C}^{(1)}, \mathcal{C}^{(2)}, \ldots, \mathcal{C}^{(M)}\}$（如 $\mathcal{C}^{(\text{beh})}$ 行为数据、$\mathcal{C}^{(\text{img})}$ 视觉刺激），通过模态消融与归因分析，量化各模态在不同实验状态（脑区、刺激类型、行为阶段等）下对预测性能的贡献。形式化地，定义模态 $m$ 的贡献为：

$$\Delta_m = \mathcal{L}(g_\phi(f_\theta(\mathcal{S}, \mathcal{C}))) - \mathcal{L}(g_\phi(f_\theta(\mathcal{S}, \mathcal{C} \setminus \mathcal{C}^{(m)})))$$

其中 $\mathcal{L}$ 为预测性能度量（如Poisson log-likelihood或 $R^2$）。进一步，该贡献可在不同条件变量 $v$（如脑区、刺激类别、行为状态）上进行条件分解 $\Delta_m(v)$，从而揭示模态-状态交互效应，为理解感觉-运动系统中的信息编码机制提供可量化的科学洞见。

### 2.2 问题范围

本研究聚焦于以下边界条件：输入为侵入式电极记录的spiking数据；预测目标为固定time bin内的spike counts（而非单个spike event的精确时间）；跨session指同一脑区、同一或不同被试的不同recording session；多模态条件信息包括行为数据（如眼动、running speed等）和视觉刺激（如自然图像、光栅等）。

---

## 3. 研究创新点

### 创新点一：基于Feed-Forward IDEncoder的Gradient-Free跨Session泛化（主要创新）

借鉴SPINT (Le et al., NeurIPS 2025)的IDEncoder思想，设计feed-forward网络从每个神经元在参考窗口中的原始放电活动中直接学习unit embedding。与SPINT的关键差异：(1) IDEncoder输出直接**替换**POYO的InfiniteVocabEmbedding作为unit embedding（非SPINT的加法注入），更契合Perceiver架构的token-level设计；(2) 提出**Spike Event Tokenization方案**（方案B）——直接使用raw spike event timestamps + rotary time embedding作为IDEncoder输入，保留精确spike timing信息，与主模型输入表示一致，作为NeuroHorizon的创新点之一。推理时只需一次前向传播即可获得新session的unit embedding（gradient-free）。

### 创新点二：长时程自回归Spike Count预测（主要创新）

提出基于cross-attention的自回归decoder，从历史神经活动预测未来的神经发放。核心设计：(1) **双窗口数据结构**——历史窗口（previous_window）的spike events作为encoder输入提供上下文，预测窗口（target_window）的binned spike counts作为decoder的预测目标；(2) **bin-level自回归**——与Neuroformer逐spike生成不同，我们的decoder在固定time bins（20ms）上预测spike counts，每步同时预测所有神经元在一个time bin的发放数，大幅降低序列长度和生成难度；(3) **causal decoder + per-neuron MLP head**——decoder通过cross-attention获取encoder完整上下文，通过causal self-attention实现时间因果性，per-neuron MLP head将时间步表示与神经元身份结合进行预测；(4) 系统性探索不同预测窗口长度（250ms → 500ms → 更长）下的性能边界。

### 创新点三：多模态条件注入与可解释性分析

通过cross-attention机制灵活注入多模态条件信息：行为数据经线性投影后注入，视觉刺激直接使用预训练DINOv2模型提取embedding后注入（冻结DINOv2权重，无需像Neuroformer那样进行额外的对比学习训练）。通过模态消融实验与归因分析（$\Delta_m$ 和条件分解 $\Delta_m(v)$），量化各模态在不同实验状态下对预测性能的贡献，提供可量化的科学洞见。

---

## 4. 方法设计与创新模块实现

> 本节详细介绍 NeuroHorizon 的架构设计与创新模块实现。更详细的代码级改造方案参见 `cc_core_files/proposal_review.md`；底层代码架构分析参见 `cc_todo/phase0-env-baseline/20260228-phase0-env-code-understanding.md`。

### 4.1 总体架构

NeuroHorizon 基于 POYOPlus 改造（复用 encoder + processor，重写 decoder + tokenize），采用 Encoder-Decoder 架构：

```
数据输入
├── previous_window spike events  [N_spikes]
│   unit_index / timestamps / token_type
│        ↓ unit_emb (InfiniteVocabEmbedding / IDEncoder) + token_type_emb
│   inputs [N_spikes, dim]
│        ↓ enc_atn (Perceiver cross-attn, latents ← inputs)  ← 序列压缩
│   latents [L, dim]
│        ↓ proc_layers (RotarySelfAttention × depth)
│   latents [L, dim]  ← 完整历史上下文表示
│                │
│   bin_queries ─┘  (learnable base + rotary time embed) [T, dim]
│        ↓ [× N_dec autoregressive decoder blocks]
│        │  ├─ cross-attn(bin_queries, latents)    [T, dim]  ← 双向
│        │  ├─ causal self-attn(bin_queries)        [T, dim]  ← causal mask
│        │  └─ FFN (GEGLU)
│   bin_repr [T, dim]
│        ↓ 投影到 dim//2
│   PerNeuronMLPHead: concat(bin_repr[t, dim//2], unit_emb[n, dim//2])
│        ↓ 3层MLP → log_rate [T, N]
│        ↓ Poisson NLL loss
│   vs target_spike_counts [T, N]
```

**复用模块**（来自 POYO/POYOPlus，不修改）：
- **Spike Tokenization**：每个spike event $(n_i, t_i)$ 作为独立token，通过 `unit_emb + token_type_emb` 获取embedding，RoPE编码连续时间戳
- **Perceiver 序列压缩**：通过 `enc_atn`（1层RotaryCrossAttention）将变长spike序列压缩为固定长度 $L$ 的latent array（$L \ll N_{\text{spikes}}$），后续计算复杂度从 $O(K^2)$ 降为 $O(L^2)$
- **Processing Layers**：$depth$ 层 RotarySelfAttention + GEGLU FFN，编码完整历史上下文

**新增/替换模块**（NeuroHorizon 核心创新）：
- Autoregressive Decoder（§4.2）
- IDEncoder（§4.3）
- 多模态条件注入（§4.5）

### 4.2 自回归生成模块

#### 4.2.1 双窗口数据组织

每个训练样本中的神经活动数据分为两个部分，承担不同角色：

| 属性 | previous_window（历史窗口） | target_window（预测窗口） |
|------|---------------------------|-------------------------|
| **用途** | Perceiver encoder 输入（提供上下文） | Autoregressive decoder 的预测目标 |
| **表示格式** | Spike events（离散时间戳，POYO格式） | Binned spike counts（固定时间格上的整数） |
| **数据形状** | `[N_spikes, 3]`（timestamp, unit_id, type） | `[T_pred_bins, N_units]` |
| **复用现有架构** | 完全复用 POYO 的 spike tokenization | 新增 binning pipeline |
| **对应 loss** | 无（不做预测） | PoissonNLL |

```
时间轴：
├── [t_start, t_start + T_hist]  ← previous_window（历史窗口）
│       spike events → encoder 输入
│
└── [t_start + T_hist, t_start + T_hist + T_pred]  ← target_window（预测窗口）
        binned spike counts → decoder 预测目标
```

**Binning pipeline**（在 `tokenize()` 中完成）：将 target_window 内的 spike events 按固定 bin 宽度（默认20ms）统计为 `[T_pred_bins, N_units]` 的 spike counts 张量。Bin 中心时刻用于 rotary time embedding。

#### 4.2.2 Causal Decoder 设计

Decoder 采用 $N_{dec}$ 层（推荐2-4层）decoder block，每个 block 包含：

```
bin_queries [B, T_pred, dim]
     │
     ├─① Cross-Attention（bin_queries attend to encoder_latents）
     │       Q = bin_queries,  K = V = encoder_latents
     │       mask: 无（双向）— latents 来自历史窗口，是完整的上下文信息
     │
     ├─② Causal Self-Attention（bin_queries 彼此 attend）
     │       Q = K = V = bin_queries
     │       mask: causal 下三角 — bin t 只看 bin 0..t（自回归约束）
     │
     └─③ FFN (GEGLU)
```

**Causal 分析总结**：

| 层 | 位置 | 是否需要 causal | 理由 |
|----|------|----------------|------|
| `enc_atn`（encoder cross-attn） | Perceiver encoder | ❌ | latent attend to 历史 spike events，无需因果 |
| `proc_layers`（self-attn × depth） | Processor | ❌ | latents 互相 attend，编码完整历史，双向 |
| Decoder cross-attn | AR Decoder | ❌ | bin_queries attend to encoder_latents（完整历史） |
| **Decoder causal self-attn** | AR Decoder | **✅** | bin t 不能 attend 到 bin t+1..T 的 query 表示 |

**结论：只有 decoder 内部的 self-attention 需要 causal mask**，encoder 和 decoder 的 cross-attention 均保持双向。

**Teacher Forcing 与自回归推理**：
- **训练时（Teacher Forcing）**：所有 $T_{pred}$ 的 bin query 同时送入 decoder，causal mask 确保时间因果性，一次前向传播得到所有步预测
- **推理时（自回归生成）**：逐步生成。注意 bin_query 本身不依赖前一步的预测输出（bin_query 是固定的 learnable base + rotary time embed），causal self-attn 使后续 bin 能 attend 到前序 bin 的**decoder 隐状态**（而非预测输出值），这是标准 Transformer decoder 的自回归模式

#### 4.2.3 Per-Neuron MLP Head

预测目标是 `[T_pred_bins, N_units]` 的 spike counts，需要同时感知**时间信息**（bin 的 decoder 表示）和**神经元身份**（unit embedding）。

**设计**：对每个 (time_bin $t$, unit $n$) 对，将 bin 表示和 unit embedding 拼接后通过共享MLP映射到 log_rate：

```python
class PerNeuronMLPHead(nn.Module):
    def __init__(self, dim):
        # 输入: concat(bin_repr[dim//2], unit_emb[dim//2]) = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, 1),   # 输出标量 log_rate
        )

    def forward(self, bin_repr, unit_embs):
        # bin_repr:  [B, T, dim//2]  ← decoder 输出投影到 dim//2
        # unit_embs: [N, dim//2]     ← unit embedding 投影到 dim//2
        B, T, _ = bin_repr.shape
        N = unit_embs.shape[0]
        combined = torch.cat([
            bin_repr.unsqueeze(2).expand(B, T, N, -1),
            unit_embs.unsqueeze(0).unsqueeze(0).expand(B, T, N, -1),
        ], dim=-1)                          # [B, T, N, dim]
        log_rate = self.mlp(combined).squeeze(-1)  # [B, T, N]
        return log_rate
```

**设计理由**：
- 不同神经元有不同的基础发放率和调谐特性，必须感知 unit identity
- N_units 在不同 session 中可变（从几十到几百），无法用固定维度的线性层
- 所有 (t, n) 对共享同一个 MLP 参数，通过 unit_emb 编码神经元间差异

**行为解码双路径**：NeuroHorizon 同时保留 POYO 原有的行为解码路径（用于 Phase 3 验证预训练迁移效果），encoder + processor 完全共享：

```
共享 encoder + processor (latents)
       │
       ├── [路径A] Spike Count Prediction
       │     bin_queries → AR Decoder → PerNeuronMLPHead → PoissonNLL
       │
       └── [路径B] Behavior Decoding（复用 POYO 设计，Phase 3 启用）
             task_queries → 原 POYO dec_atn → MultitaskReadout → MSE
```

#### 4.2.4 Bin Query 设计

Bin query 编码每个预测时间 bin 的位置信息，采用 **learnable base + rotary time embedding** 方案（与 POYO 的 latent token 机制一脉相承）：

```python
self.bin_emb = nn.Parameter(torch.randn(1, max_T_pred, dim))  # learnable base
# rotary embedding 将绝对时间信息注入 Q/K 的旋转
bin_timestamps = torch.linspace(t_pred_start + bin_size/2,
                                t_pred_end - bin_size/2,
                                T_pred_bins)  # 每个 bin 的中心时刻
```

**优势**：(1) 与 POYO 的 latent token 机制一致，复用现有 rotary embedding 基础设施；(2) RoPE 天然编码时间关系，cross-attn 时 encoder latents 和 bin queries 的时间距离反映在 attention 权重中；(3) 可变预测窗口（250ms/500ms/更长）只需调整 bin 数量，无需重新学习 embedding。

#### 4.2.5 损失函数

主要采用 Poisson negative log-likelihood loss：

$$\mathcal{L}_{\text{pred}} = \sum_{b=1}^{B} \sum_{n=1}^{N} \left[ \exp(\hat{r}_{n,b}) - y_{n,b} \cdot \hat{r}_{n,b} \right]$$

其中 $\hat{r}_{n,b}$ 为模型预测的 log firing rate（经 clamp 至 $[-10, 10]$ 保证数值稳定），$y_{n,b}$ 为真实 spike count。模型输出 log_rate 而非 rate 本身，避免 exp 下溢。

### 4.3 IDEncoder 跨Session模块

#### 4.3.1 架构设计

参考 SPINT (Le et al., NeurIPS 2025) 的 feedforward 架构，IDEncoder 从参考窗口的原始神经活动推断 unit embedding：

$$E_i = \text{MLP}_2\left( \frac{1}{M} \sum_{j=1}^{M} \text{MLP}_1(X_i^{C_j}) \right)$$

```
输入：X_i^C ∈ ℝ^(M × T_ref)  （unit i 的 M 条参考窗口 binned spike counts）

Step 1: MLP₁(X_i^{C_j}): ℝ^T_ref → ℝ^H     每条参考窗口独立映射（3层FC）
Step 2: Mean pooling across M windows          M 条取均值（置换不变）
Step 3: MLP₂: ℝ^H → ℝ^d_model                映射到 unit embedding 维度（3层FC）

输出：E_i ∈ ℝ^d_model  （unit i 的 identity embedding）
```

**关键设计**：IDEncoder 参数在训练时通过反向传播学习，推理时对新 session 的神经元只需一次前向传播即可获得 embedding（gradient-free）。

#### 4.3.2 Identity 注入方式：替换 unit_emb

与 SPINT 的加法注入（$Z = X + E$）不同，NeuroHorizon 将 IDEncoder 输出**直接替换** POYO 的 `InfiniteVocabEmbedding`：

```python
# POYO 原始路径（Phase 1）：
inputs = self.unit_emb(input_unit_index) + self.token_type_emb(input_token_type)

# NeuroHorizon IDEncoder 路径（Phase 2）：
unit_embs = self.id_encoder(ref_data)            # [N_units, d_model]
inputs = unit_embs[input_unit_index] + self.token_type_emb(input_token_type)
```

**设计动机**：POYO Perceiver 架构中每个 spike event 需要独立的 unit embedding 作为"身份标签"。IDEncoder 输出自然填充这个角色——从"查表"变为"从神经活动推断"。两种路径通过 `use_id_encoder` flag 切换，InfiniteVocabEmbedding 的 `tokenizer()`/`detokenizer()`/vocab 管理接口保留不变（被 data pipeline 大量依赖）。

#### 4.3.3 输入 Tokenization 方案

| 方案 | 输入 | 架构 | 定位 |
|------|------|------|------|
| **方案A: Binned Timesteps** | 参考窗口 spike events → 20ms bin → 固定长度 $T_{ref}$ | 纯MLP（与SPINT一致） | **基础实现**，可直接参考SPINT的验证结果 |
| **方案B: Spike Event Tokenization** | Raw spike event timestamps + rotary time embedding → attention pooling | Attention + MLP | **NeuroHorizon创新**，保留精确spike timing |

推荐先实现方案A（降低实现风险），方案B作为后续改进方向（若方案A效果不理想或需要更精细的timing信息）。方案A与方案B的对比实验可验证spike-level identity推断的优势。

### 4.4 Spike Tokenization

复用 POYO 的 spike tokenization 机制：每个 spike event $(n_i, t_i)$ 作为独立 token，embedding 为：

$$\mathbf{h}_i^{(0)} = \mathbf{u}_{n_i} + \mathbf{e}_{\text{spike}}$$

其中 $\mathbf{u}_{n_i}$ 为 unit embedding（来自 IDEncoder 或 InfiniteVocabEmbedding），$\mathbf{e}_{\text{spike}}$ 为可学习的 spike type embedding。RoPE 时间编码在 attention 计算时直接应用于 query 和 key（编码连续时间戳的相对关系），而非加到 token embedding 上。每个 unit 还有 start/end boundary tokens 标记边界。

### 4.5 多模态条件注入

在 encoder 的特定层，通过 cross-attention 注入多模态条件信息：

**行为数据（Behavior）**：
$$\mathbf{c}_{\text{beh}} = \text{Linear}(\mathbf{x}_{\text{behavior}}) \in \mathbb{R}^{T_b \times d}$$
$$\mathbf{H}^{(l)} = \mathbf{H}^{(l)} + \text{CrossAttn}(\mathbf{H}^{(l)}, \mathbf{c}_{\text{beh}}, \mathbf{c}_{\text{beh}})$$

行为数据（运动轨迹、速度、wheel position等）经线性投影到模型维度后，作为 cross-attention 的 key/value。

**视觉刺激（Image）**：
$$\mathbf{c}_{\text{img}} = \text{Linear}(\text{DINOv2}(\mathbf{I})) \in \mathbb{R}^{N_p \times d}$$
$$\mathbf{H}^{(l)} = \mathbf{H}^{(l)} + \text{CrossAttn}(\mathbf{H}^{(l)}, \mathbf{c}_{\text{img}}, \mathbf{c}_{\text{img}})$$

DINOv2 为冻结的预训练视觉模型（ViT-B），仅训练线性投影层。DINOv2 embedding 必须**离线预计算**（非训练时实时提取）。这避免了 Neuroformer 中���外的对比学习训练阶段，简化了整体训练流程并利用了 DINOv2 强大的视觉表征能力。

---

## 5. 数据集

> 详细数据集介绍、选型策略与各阶段适配注意事项，参见 `cc_core_files/dataset.md`。

| 数据集 | 简介 | NeuroHorizon 用途 | 引入阶段 |
|--------|------|------------------|---------|
| **Brainsets（Perich-Miller）** | 猕猴运动皮层，70+ sessions，零配置接入 POYO 框架 | 自回归改造验证、长时程预测、跨session初期 | 阶段一～三核心 |
| **IBL Brain-wide Map** | 小鼠全脑，459 sessions，12 实验室 | 大规模跨session泛化、data scaling（可选扩展） | 阶段二/三可选 |
| **Allen Visual Coding Neuropixels** | 小鼠视觉皮层，58 sessions，丰富视觉刺激 | 多模态融合实验（neural + image） | 阶段四 |
| **NLB / FALCON** | 标准化 benchmark | 与社区方法对比 | 补充/可选 |

---

## 6. 实验设计

> 详细实验方案与验收标准参见 `cc_core_files/proposal_review.md`；分阶段执行计划参见 `cc_core_files/plan.md`。

### 6.1 核心实验

| 实验 | 目标 | 主要数据集 | 对应阶段 |
|------|------|-----------|---------|
| 长时程预测评估 | 评估不同预测窗口长度下的性能边界 | Brainsets | Phase 1 |
| 跨Session泛化 | 验证IDEncoder gradient-free泛化 | Brainsets（必做）/ IBL（可选） | Phase 2 |
| Data Scaling | 性能随训练session数的变化规律 | Brainsets / IBL | Phase 3 |
| 下游任务迁移 | 自回归预训练 → 行为解码迁移增益 | Brainsets | Phase 3 |
| 多模态贡献分析 | 量化各模态对预测的贡献 $\Delta_m(v)$ | Allen | Phase 4 |

### 6.2 消融实验

- IDEncoder 消融：IDEncoder vs 可学习嵌入 vs random embedding
- Decoder 深度消融：$N_{dec}$ = 1 / 2 / 4
- 预测窗口长度消融
- Scheduled sampling 消融
- Causal decoder vs parallel prediction（非自回归）对比
- Poisson NLL vs MSE loss 对比

### 6.3 Baseline 对比

- 简单 baseline：PSTH-based prediction、线性预测
- 模型 baseline：Neuroformer、NDT1/NDT2/NDT3、NEDS
- POYO+ 原始行为解码（作为编码质量参照）

### 6.4 评估指标

- **spike count 预测**：Poisson log-likelihood、PSTH correlation（trial-averaged）、$R^2$
- **行为解���**（下游任务）：$R^2$、velocity MSE
- **跨session泛化**：gradient-free 性能 vs per-session fine-tuning upperbound
- **多模态贡献**：$\Delta_m$ 和条件分解 $\Delta_m(v)$

---

## 7. 可能的风险

**风险1：IDEncoder跨session泛化效果不佳**
- 可能原因：参考窗口信息不足以区分神经元功能特性
- 应对方案：增加参考窗口长度；尝试更复杂架构（如小型Transformer）；引入 Dynamic Channel Dropout（参考 SPINT）

**风险2：长时程预测准确性不足**
- 可能原因：长预测窗口内的 spike count 过于稀疏/随机
- 应对方案：增大 time bin 宽度（20ms → 50ms）；引入 scheduled sampling；采用概率预测；缩短预测窗口作为折中

**风险3：计算资源不足**
- 可能原因：大规模多session训练超出GPU显存/时间
- 应对方案：gradient accumulation；BF16 混合精度；Perceiver 压缩；从 Small 模型开始

**风险4：多模态融合未带来显著提升**
- 可能原因：DINOv2特征与neural activity的对应关系不够直接
- 应对方案：添加 learnable adapter layers；尝试不同 DINOv2 变体；在视觉皮层数据上重点测试

**风险5：预测粒度局限**
- 当前设计预测 spike counts in bins 而非 individual spike times，可能无法捕获 spike timing 的精细信息
- 此为有意的设计权衡——bin-level 预测大幅降低序列长度和生成难度

**风险6：数据依赖**
- IDEncoder 的质量依赖于参考窗口数据的质量和长度
- 极短或极嘈杂的参考窗口可能导致泛化性能下降

---

## 8. 参考文献

### 核心参考论文

1. **NDT1**: Ye, J., & Pandarinath, C. (2021). Representation learning for neural population activity with Neural Data Transformers. *NeurIPS*.
2. **NDT2/STNDT**: Ye, J., Collinger, J., Bhatt, S., & Pandarinath, C. (2022). Neural Data Transformers 2: Multi-context pretraining for neural spiking activity.
3. **NDT3**: Ye, J., et al. (2024). Neural Data Transformer 3: Scaling foundation models for neural spiking activity. *NeurIPS*.
4. **LFADS**: Pandarinath, C., et al. (2018). Inferring single-trial neural population dynamics using sequential auto-encoders. *Nature Methods*.
5. **Neuroformer**: Antoniades, A., et al. (2024). Neuroformer: Multimodal and Multitask Generative Pretraining for Brain Data. *ICLR*.
6. **POYO**: Azabou, M., et al. (2024). A Unified, Scalable Framework for Neural Population Decoding. *NeurIPS*.
7. **POYO+**: Azabou, M., et al. (2024). POYO+: Scaling Neural Foundation Models with Multi-Area and Multi-Task Learning.
8. **POSSM**: Azabou, M., et al. (2025). POSSM: State Space Models for Neural Population Spike Data.
9. **SPINT**: Le, V., et al. (2025). SPINT: Session-invariant neural decoding via gradient-free ID encoding. *NeurIPS*.
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

## 附录A：符号表

| 符号 | 含义 |
|------|------|
| $\mathcal{S}$ | 输入spike event集合 |
| $(n_i, t_i)$ | 第$i$个spike event（神经元ID, 时间戳） |
| $\mathbf{u}_n$ | 神经元$n$的unit embedding |
| $\mathbf{r}_n$ | 神经元$n$的参考窗口特征向量 |
| $\mathbf{H}$ | Encoder隐层表征 |
| $\hat{r}_{n,b}$ | 预测的log firing rate（神经元$n$, time bin $b$） |
| $\mathcal{C}$ | 多模态条件信息 |
| $\mathcal{C}^{(m)}$ | 第$m$个模态的条件信息 |
| $\Delta_m$ | 模态$m$的预测贡献 |
| $\Delta_m(v)$ | 模态$m$在条件变量$v$下的贡献 |
| $T_{\text{hist}}$ | 历史窗口（previous_window）长度 |
| $T_{\text{pred}}$ / $\Delta T$ | 预测窗口（target_window）长度 |
| $\delta$ | Time bin宽度（默认20ms） |
| $B$ | 预测time bin数量 |
| $L$ | Perceiver latent array 长度 |

---

## 补充：基于Perceiver的序列压缩与高效推理

借鉴POYO的Perceiver cross-attention机制压缩spike序列长度，降低encoder端的计算复杂度。当spike序列过长时（如>2000 spikes/秒），$L$ 个可学习的latent query tokens（$L \ll K$，如 $L=128$ 或 $256$）通过cross-attention聚合spike信息，将后续计算复杂度从 $O(K^2)$ 降为 $O(L^2)$。在需要实时推理的场景下，可探索将Transformer backbone替换为State Space Model（如Mamba），实现 $O(N)$ 复杂度的高效推理。
