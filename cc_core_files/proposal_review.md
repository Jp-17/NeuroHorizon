# NeuroHorizon 项目分析文档

> 本文档整理自项目初期分析，包含项目目标、与 POYO 的关键差异、合理性评估及架构设计。
> 供研究背景参考，不作为执行依据。**执行计划参见 `cc_core_files/plan.md`**。

**整理日期**：2026-02-21（原始分析）

---

## 1. 项目目标

NeuroHorizon 旨在构建一个**统一的神经编码模型**，同时解决计算神经科学中的两大核心挑战：

1. **跨 Session 鲁棒性**：实现梯度无关的跨 session 泛化（不同记录会话、不同神经元群体）
2. **长时间跨度预测**：将神经活动预测从传统 ~200ms 扩展到 1 秒以上

**目标发表**：NeurIPS / ICLR / Nature Methods

---

## 2. 与 POYO 的关键差异

| 维度 | POYO/POYO+ | NeuroHorizon |
|------|-----------|-------------|
| **核心任务** | 行为解码 (spikes → behavior) | 神经编码 (spikes → future spikes) |
| **Unit Embedding** | Per-unit 可学习嵌入 (InfiniteVocabEmbedding) | IDEncoder (从参考窗口特征前馈生成) |
| **输出机制** | 单次 cross-attention → 线性读出 | 自回归 cross-attention decoder |
| **时间分辨率** | Spike 级输入 | Spike 级输入 + Binned spike count 输出 |
| **预测跨度** | N/A（解码任务） | 500ms - 1000ms |
| **跨 Session 能力** | 有限（需微调） | 梯度无关（IDEncoder 零样本泛化） |
| **多模态** | 无 | 行为 + 图像 (DINOv2) |
| **损失函数** | MSE / CE | Poisson NLL |

---

## 3. 合理性评估

### 3.1 优势

- **研究方向有价值**：跨 session 泛化和长时间预测是 BCI 和计算神经科学的核心需求
- **代码基础扎实**：POYO+ 框架模块化程度高，Encoder-Processor-Decoder 架构便于扩展
- **数据集选择合理**：
  - Brainsets 原生数据（Perich-Miller 等）→ 快速验证、低门槛
  - IBL (459 sessions, 241 brain regions) → 跨 session 泛化 + scaling law
  - Allen (58 sessions, 多刺激模态) → 多模态融合实验
- **创新点明确**：IDEncoder (SPINT 思路用于生成任务) + 自回归 decoder + 多模态融合 的组合是新颖的

### 3.2 已识别的问题与应对

#### 问题 1: Jia Lab 数据不可用 [已确认]
- **影响**：提案中列为长时程预测核心数据源（★★★，大量重复 trial）
- **应对**：完全使用 Brainsets 原生数据 + IBL 连续记录（策略 B：不对齐 trial 的滑动窗口）+ Allen Natural Movies（30s 连续视频）替代
- **风险等级**：中。Brainsets 和 Allen 数据仍然可以验证长时间预测，但缺少大量重复 trial 下的 PSTH 可靠性评估

#### 问题 2: Allen Natural Scenes 时间间隔
- **问题**：每张图片展示 250ms + 500ms 灰屏间隔
- **影响**：1s 预测窗口会跨越多个刺激边界，预测内容混合了刺激响应和灰屏衰减
- **应对**：优先使用 Natural Movies 刺激集（30s 连续视频，无间隔）

#### 问题 3: 可变输出维度
- **问题**：spike count 输出维度 = 该 session 的 neuron 数，每个 session 不同
- **影响**：现有 MultitaskReadout 假设固定输出维度，无法直接使用
- **应对**：设计 shared per-neuron MLP head，输入 = concat(bin_repr, unit_emb)，输出 = 1 (log-rate)，自然适应任意 n_units

#### 问题 4: 自回归误差累积
- **问题**：20ms bin、1s 预测 = 50 步自回归，误差逐步放大
- **应对**：
  - 训练：Scheduled sampling（逐步减少 teacher forcing 比例）
  - 对照：Non-autoregressive parallel prediction 基线
  - 策略：Coarse-to-fine（先 100ms 分辨率，再精细化）

#### 问题 5: 计算资源
- **当前**：单卡 4090
- **应对**：
  - 开发全程使用 Small (5M params)
  - 正式实验 Base (30M) 需 BF16 + gradient checkpointing
  - Large (100M) 需梯度累积或升级 GPU
  - 后续可扩展资源

#### 问题 6: rotary_attention 不支持 causal mask
- **问题**：现有 `RotarySelfAttention` 的 mask 处理为 `(b, 1, 1, n)` 形状，仅支持 KV masking，不支持自回归所需的 causal mask `(b, 1, n_q, n_kv)`
- **应对**：修改 `rotary_attn_pytorch_func` 中 mask 的 reshape 逻辑，`F.scaled_dot_product_attention` 已原生支持 2D mask

---

## 4. 架构设计

### 4.1 整体架构

```
输入窗口 [0, T_in]                          预测窗口 [T_in, T_in + T_pred]
──────────────────                          ────────────────────────────
    │                                                │
    ▼                                                ▼
[Spike Tokens]                              [Spike Count Targets]
    │                                        (binned, 20ms bins)
    ▼
[IDEncoder] 参考窗口特征 → Unit Embeddings
    │
    ▼
[Token Embedding] = IDEncoder(unit_idx) + TokenType + RoPE
    │
    ▼
[Perceiver Cross-Attention] spikes → latents
    │
    ▼
[Self-Attention Processing] × depth 层
    │  (可选：每隔2层插入 Multimodal Cross-Attention)
    │  (图像 DINOv2 embedding / 行为数据)
    │
    ▼
[Autoregressive Cross-Attention Decoder] × N_dec 层
    │  每层: Cross-Attn(bins→latents) + Causal Self-Attn(bins) + FFN
    │
    ▼
[Per-Neuron MLP Head] concat(bin_repr, unit_emb) → log-rate
    │
    ▼
[Poisson NLL Loss] 对比预测 log-rate 与真实 spike count
```

### 4.2 IDEncoder

替换 `InfiniteVocabEmbedding`，从参考窗口统计特征生成 unit embedding：

- **输入特征** (~33 维)：
  - 平均发放率 (1d)
  - ISI 变异系数 (1d)
  - ISI log-histogram (20d)
  - 自相关特征 (10d)
  - Fano factor (1d)
- **网络结构**：3 层 MLP (Linear + GELU + LayerNorm)
- **核心优势**：新 session 的新 neuron 只需计算统计特征即可获得 embedding，无需微调

### 4.3 自回归解码器

```
对每个预测 time bin b = 1..B:
  1. bin_query = bin_type_emb + RoPE(t_b)
  2. cross_attend(bin_query, encoder_latents)
  3. causal_self_attend(bin_1..bin_b)  // 只看之前的 bins
  4. per_neuron_head: concat(bin_repr, unit_emb) → log_rate
```

### 4.4 模型规模

| 配置 | Encoder层 | Decoder层 | 隐藏维度 | Attention Heads | 参数量 |
|------|-----------|-----------|---------|-----------------|--------|
| Small | 4 | 2 | 256 | 4 | ~5M |
| Base | 8 | 4 | 512 | 8 | ~30M |
| Large | 12 | 6 | 768 | 12 | ~100M |

---

*原始分析日期：2026-02-21；整理入此文档：2026-02-28*
