# proposal.md 审查分析报告

**审查日期**：2026-02-25
**审查对象**：cc_core_files/proposal.md
**审查依据**：POYO 代码库实际实现 + code_research.md 审查结果 + 方法论合理性评估

---

## 总体评价

proposal.md 是一份学术水准较高的研究提案，研究动机明确、创新点逻辑清晰、实验设计全面。但在**方法设计细节**与**POYO 代码库实际实现**的对接上存在若干不一致和技术隐患，部分架构设计决策需要更仔细的论证。以下按章节逐段分析。

---

## 第1-3节：背景、问题定义、相关工作 — 质量好，局部可改进

### 准确且有价值的部分
- 研究现状综述覆盖了核心工作（NDT系列、LFADS、Neuroformer、POYO/POYO+、SPINT）
- 问题定义形式化清晰，四大挑战的分析深入
- 相关工作对比表（Section 3.2）结构清晰

### 建议修正/补充

1. **POSSM 描述可能不准确**：
   > "POSSM (Azabou et al., 2025)：POYO的后续工作，将SSM引入spike sequence建模，用Mamba替代Transformer以处理更长的序列。"

   POSSM 的细节需要再核实。如果确认是 Mamba 替代 Transformer，需要注意 POYO 的核心不是简单的 Transformer，而是 Perceiver + Self-Attention 的组合。Mamba 替代的应该是 Self-Attention 处理层（processor），而非 Perceiver 编码器部分。

2. **对比表中"POYO+ 跨Session = 有限"需要精确化**：
   POYO+ 的跨 session 能力基于 `InfiniteVocabEmbedding` 的动态词汇扩展——对新 session 可以 `extend_vocab()` 添加新 unit，但需要**微调**（fine-tuning）才能学到有效的 embedding。这不是完全没有跨 session 能力，而是需要梯度更新。建议改为"需要微调"。

3. **"计算效率"列的评估依据不清**：
   表中标注 Neuroformer "低"、POYO+ "高"、NeuroHorizon "高"。但 NeuroHorizon 增加了 IDEncoder 计算 + 多层 autoregressive decoder + 多模态 cross-attention，比 POYO+ 的计算量显著增大。标注"高"可能过于乐观，建议改为"中"或在文中说明效率保持的机制。

---

## 第4节：研究创新点 — 逻辑清晰，但需要更严谨的技术对比

### 创新点一：IDEncoder — 核心方向正确，但描述需要精确化

**问题1：与 SPINT 的实际差异不够清晰**

提案说"借鉴SPINT的IDEncoder思想，但进行关键改进"，然后描述了一个 FFN 从参考窗口特征生成 unit embedding 的方案。但实际上 SPINT 的 IDEncoder 本身就是一个从活动模式生成 embedding 的前馈网络。提案需要更清楚地说明**与 SPINT 的具体差异**是什么：
- 输入特征不同？（统计特征 vs 原始活动模式）
- 网络结构不同？
- 应用场景不同？（解码→编码/生成）

如果主要差异只是"从解码任务扩展到编码/生成任务"，这作为独立创新点的新颖性需要更强的论证。

**问题2：参考窗口特征的选择需要论证**

提案列出了"平均 firing rate、ISI 分布统计量、自相关函数特征、波形特征"作为 IDEncoder 输入。但这些手工设计的统计特征有几个隐患：
- 计算自相关特征和 ISI 统计量需要足够长的参考窗口（至少几秒），这在某些实验范式中可能不可用
- 波形特征在很多公开数据集中不可用（如 IBL 的标准下载不包含波形模板）
- 手工特征可能丢失信息——SPINT 原文可能使用的是学习到的特征提取

**建议**：考虑两种 IDEncoder 设计的对比：(a) 手工统计特征 + FFN；(b) 参考窗口原始 spike train + 小型卷积/Transformer encoder。消融实验中可以对比两者。

### 创新点二：自回归 Spike Count 预测 — 方向正确，但有重要设计隐患

**问题1：自回归解码器的具体查询结构设计不完整**

Section 5.5.2 的 decoder 设计描述了 per-bin query `q_b = e_bin + RoPE(t_b)`。但这里有一个关键问题：**每个 time bin 的预测需要为所有 N 个神经元输出 spike count**。

当前设计是：
```
d_b' = CausalSelfAttn(d_b, d_{1:b-1})
y_{n,b} = MLP([d_b'; u_n])  // 对每个神经元 n 拼接 bin 表示和 unit embedding
```

这意味着每个 bin 只有**一个**查询向量 `d_b'`，然后与每个 neuron 的 `u_n` 拼接后通过 MLP 预测。这样的设计有几个问题：

- **信息瓶颈**：整个 bin 内所有神经元的预测都依赖同一个 `d_b'` 向量。如果 `d_b'` 的维度是 512，而需要预测 300 个神经元的发放率，每个神经元分到的"信息通道"非常有限。
- **计算效率**：每个 bin 需要对 N 个神经元分别执行 MLP forward，如果 N=300, B=50（1s / 20ms），则需要 15000 次 MLP forward。虽然可以并行化，但这使得计算量与神经元数量线性增长。
- **缺乏神经元间交互**：每个神经元的预测独立于同一 bin 内其他神经元的预测。但神经元之间存在强烈的同步性和相关性，忽略这种交互可能损失预测准确性。

**替代方案建议**：
- **方案A**：使用 per-neuron per-bin 查询——每个 `(neuron, bin)` 对有一个独立的查询向量 `q_{n,b} = u_n + e_bin + RoPE(t_b)`。但这会使查询序列长度变为 N × B，可能过大。
- **方案B**：在 bin 表示 `d_b'` 之上再加一层 cross-attention，用 unit embeddings 作为 query attend 到 `d_b'`，这样每个 neuron 可以选择性地从 bin 表示中提取自己需要的信息。
- **方案C**：使用 POYO 现有的 per-query 结构——输出查询本身就包含 unit 信息，只是将目标从 behavior 改为 spike counts。这需要解决不同 session 的 unit 数量不同的问题。
- **方案D**（推荐）：参考 plan.md 中已有的 "per-neuron MLP head: concat(bin_repr, unit_emb) → log_rate" 方案，但增加一个 **轻量级的 neuron interaction 层**（如一个小的 self-attention over neurons within each bin）来捕获神经元间交互。

**问题2：自回归的必要性需要更强的论证**

提案假设自回归（逐 bin 预测）是长时程预测的必要方式。但对于 spike count 预测（而非 spike event 预测），**非自回归的并行预测**也是可行的：所有 bin 的查询可以同时 attend 到 encoder 输出，使用 causal mask 限制只看到之前 bin 的预测结果。

在训练时使用 teacher forcing（提供真实的之前 bin 计数），在推理时使用自回归——这是标准做法。但如果非自回归预测效果接近，计算效率会好得多（一次 forward 而非 B 次）。建议将非自回归 parallel prediction 作为消融实验的 baseline。

### 创新点三：多模态条件注入 — 设计合理但需要调整实现位置

**问题1：注入位置与 POYO 架构的冲突**

提案说"在encoder的特定层（如每隔2层），通过cross-attention注入多模态条件信息"。但 POYO 的架构结构是：
1. Perceiver cross-attention（1层）：spikes → latents
2. Self-attention processing（多层）：latents 自注意力
3. Decoder cross-attention（1层）：latents → outputs

多模态 cross-attention 应该加在**第 2 步（processing layers）的特定层间**。但 POYO 的 processing layers 已经是紧密耦合的 Self-Attention + FFN 块。在其间插入 cross-attention 层意味着修改 processor 的结构，这比"不修改 encoder"的增量策略更侵入性。

**建议**：明确说明多模态注入是在 processing layers 之间进行的，并在 plan 中将其列为需要修改的代码路径。或者考虑在 Perceiver 编码之前就将多模态信息融合到输入中（更简单但可能效果不同）。

**问题2：DINOv2 的使用假设**

提案提到"直接使用预训练的 DINOv2 模型提取 image embedding"。这对 Allen 数据集的 natural scenes（灰度图像，918×1174）需要注意：
- DINOv2 预训练在 RGB 自然图像上，灰度图需要复制三通道
- Allen 图像分辨率与 DINOv2 期望的 224×224 差异较大，resize 可能丢失细节
- 更重要的是：Allen 的视觉刺激是高度标准化的（固定亮度、固定呈现时间），与 DINOv2 训练数据的自然图像分布可能有较大差距

这些不是不可克服的问题，但需要在实验中验证 DINOv2 特征在这个特定域上的有效性。

### 创新点四：Perceiver 压缩 — 定位需要修正

提案将 Perceiver 压缩标记为"可选"。但在 POYO 框架中，**Perceiver 压缩是架构的核心组件**，不是可选的。POYO 的整个数据流依赖 Perceiver cross-attention 将变长 spike 序列压缩为固定长度 latent array。如果移除 Perceiver，self-attention processing 层就需要直接在原始 spike token 上操作，序列长度可能达到数千甚至上万，这在计算上不可行。

**建议**：将 Perceiver 从"可选"改为"核心组件（继承自 POYO）"。SSM 替代方案可以作为可选的探索方向。

---

## 第5节：方法设计 — 核心问题汇总

### 5.1 总体框架 — 基本合理但有误导

流程图中的 Step 3 "Transformer/SSM Encoder" 说"编码神经群体时空动态"，但实际上在 POYO 框架中这个位置对应的是 **Self-Attention Processing Layers**（在 Perceiver 压缩之后的处理层），不是传统意义上的"encoder"。将其称为 "Processor" 更符合 POYO 的命名惯例。

### 5.4 Encoder 架构

**问题：SwiGLU vs GEGLU**

> "Feed-Forward Network (SwiGLU activation)"

POYO 实际使用的是 **GEGLU**（`x * GELU(gates)`），而非 SwiGLU（`x * Swish(gates)`）。两者相似但不完全相同。如果计划继承 POYO 的架构，应使用 GEGLU；如果有意改为 SwiGLU，需要说明理由和预期效果差异。

### 5.5 Decoder 架构

除了前面在创新点二中分析的设计隐患外，还有一个具体的实现问题：

**Spike Count 预测的输出分布选择**

提案提到"输出经过softmax或Poisson parameter预测"。这两种方式有本质区别：
- **Softmax**：将 spike count 视为分类问题（0, 1, 2, 3, ... spikes），需要预设最大 count 值
- **Poisson rate**：输出 log-rate 参数 λ，spike count 服从 Poisson(λ)

两种方式的 loss 函数、训练动态、和预测行为都不同。建议**明确选择 Poisson rate 参数化**（与后续的 Poisson NLL Loss 一致），并在方法描述中统一。

### 5.6 训练策略

**Neuron Dropout 与 POYO 现有实现的关系**

提案说"随机丢弃一定比例（如20%）的神经元"。POYO 已有 `UnitDropout` 实现，使用三角分布（而非固定比例）来决定保留的 neuron 数量。建议直接复用 POYO 的 UnitDropout，或说明为什么需要不同的 dropout 策略。

---

## 第6节：实验设置 — 参数合理性分析

### 6.3 模型参数表

| 配置 | 提案参数量 | 合理性评估 |
|------|-----------|-----------|
| Small: dim=256, depth=4+2 | ~5M | POYO-MP (dim=64, depth=6) 为 1.3M。dim=256 的 6 层应该在 5M 左右，**大致合理** |
| Base: dim=512, depth=8+4 | ~30M | POYO-1 (dim=128, depth=24) 为 11.8M。dim=512 的 12 层 + decoder 约 30M，**大致合理** |
| Large: dim=768, depth=12+6 | ~100M | 18 层 dim=768 加上多模态组件，100M **可能偏低估**，建议验证 |

**注意**：如果增加多模态 cross-attention 层（每隔 2 层一个），额外参数量需要计入。每个 cross-attention 层约增加 `4 * dim^2` 参数。

### 6.4 训练配置

**优化器选择：AdamW vs SparseLamb**

提案选择 AdamW，但 POYO 使用 SparseLamb。这两者有重要差异：
- SparseLamb 专为 `InfiniteVocabEmbedding` 的稀疏梯度设计，只更新有梯度的参数行
- 如果用 IDEncoder 替换 InfiniteVocabEmbedding，`unit_emb` 不再需要稀疏更新（因为所有 neuron 的 embedding 都通过同一个 FFN 生成）
- 但 `session_emb` 仍然是 InfiniteVocabEmbedding，仍需要稀疏优化

**建议**：保留 SparseLamb（至少对 session_emb），或者使用 AdamW 但对 session_emb 使用分组优化。

---

## 第7节：数据集 — 与 dataset.md 的一致性

### 7.1.1 Allen Brain Observatory

描述总体准确。

**需要注意的细节**：
- 提案说"Natural Scenes（118张）"，但 Allen 数据集中 natural_scenes 的注册模态是 dim=119（ID 从 0-118），这表示有 119 个类别。需要确认是 118 还是 119 张图像。
- Allen NWB session 文件每个 1.7-3.3 GB，58 个 session 共约 146.5 GB，与 dataset.md 一致。

### 7.1.3 NLB Datasets

描述准确，但需要注意：
- NLB 作为 baseline 对比基准，需要确保实验设计的可比性
- NLB 数据是 binned spike counts（不是 spike events），需要考虑如何用 NeuroHorizon 的 spike-level 输入处理 binned 数据

### 7.1.4 Jia Lab 数据集

plan.md 已确认不可用，但 proposal.md 仍将其列为"★★★核心数据集"。**需要更新 proposal.md 以反映这一变化**，将 IBL + Allen Natural Movies 的替代方案写入正文。

---

## 第8节：预期结果与实验 — 实验设计全面但有可行性风险

### 8.3 长时程预测能力评估

**预测窗口的现实可行性**

提案计划测试 100ms, 200ms, 500ms, 1000ms 的预测窗口。需要考虑：
- 在 20ms bin、1000ms 预测窗口下，decoder 需要自回归 50 步
- 50 步自回归的误差累积是严峻的挑战
- plan.md 中提到的 scheduled sampling 策略是必要的
- **建议增加 coarse-to-fine 策略**：先用 100ms bin 预测粗粒度，再用 20ms bin 细化

### 8.4 多模态条件贡献分析

"在不同脑区上分析模态贡献差异（如V1对image模态敏感度 vs. motor cortex对behavior模态敏感度）"——这要求同时在 V1 数据（Allen）和 motor cortex 数据（NLB/IBL）上训练。但两个数据集的格式、质量、规模完全不同，直接对比需要仔细的实验控制。

### 8.6 消融实验

消融实验列表全面，但 10 个消融实验的计算成本很高。建议：
- 优先完成 A1（IDEncoder）、A3（Perceiver）、A4（输入窗口长度）、A10（Loss 函数）
- A7（DINOv2 vs 对比学习）需要额外训练对比学习模型，成本高，可以后置
- A8（Transformer vs Mamba）是独立的架构探索，可以在 MVP 之后再做

---

## 第9节：局限性讨论 — 补充建议

### 缺少的局限性讨论

1. **IDEncoder 参考窗口的敏感性**：参考窗口的长度和选择位置可能显著影响 IDEncoder 的质量。如果某个 neuron 在参考窗口内几乎不发放（低发放率 neuron），其统计特征将非常不可靠。
2. **Spike count vs spike timing 的信息损失**：提案承认了这一点，但可以进一步讨论——在需要精确 timing 的脑区（如听觉皮层），spike count 预测可能不够有意义。
3. **数据泄露风险**：如果 IDEncoder 的参考窗口与预测窗口时间上重叠或过近，可能存在信息泄露。

---

## 整体架构一致性审查

### Proposal 方法设计 vs POYO 代码实际结构 对比

| 模块 | Proposal 描述 | POYO 实际实现 | 一致性 |
|------|-------------|-------------|--------|
| 输入嵌入 | IDEncoder(r_n) + spike_type_emb | InfiniteVocabEmbedding(unit_id) + token_type_emb | 需要替换，接口变化大 |
| 时间编码 | RoPE | RotaryTimeEmbedding | 一致 |
| 序列压缩 | "可选" Perceiver cross-attention | 核心组件，不可选 | **不一致** |
| 处理器 | "Transformer/SSM Encoder" | Self-Attention Processing Layers | 命名不一致 |
| 激活函数 | SwiGLU | GEGLU | **不一致** |
| 多模态注入 | 每隔2层 cross-attention | 不存在（需新增） | 需明确插入位置 |
| 解码器 | 多层 cross-attn + causal self-attn | 单层 cross-attn + FFN | 需大幅扩展 |
| 读出层 | per-neuron MLP | nn.Linear / MultitaskReadout | 需重新设计 |
| 损失函数 | Poisson NLL | MSE / CrossEntropy | 需新增 |
| 优化器 | AdamW | SparseLamb | **不一致** |

---

## 建议的关键修正清单

| 优先级 | 修正内容 |
|--------|---------|
| **高** | 将 Perceiver 从"可选"改为"核心组件" |
| **高** | 统一激活函数描述：明确使用 GEGLU（与 POYO 一致）还是有意改为 SwiGLU |
| **高** | 解决 decoder 的信息瓶颈问题（per-bin 单查询 vs per-neuron 查询） |
| **高** | 更新 Jia Lab 数据集的不可用状态，调整数据集优先级 |
| **高** | 明确与 SPINT IDEncoder 的具体技术差异 |
| **中** | 调整计算效率自评（从"高"降为"中"） |
| **中** | 统一优化器选择（SparseLamb vs AdamW） |
| **中** | 明确多模态 cross-attention 在 POYO 架构中的具体插入位置 |
| **中** | 精确化 POYO+ 跨 session 能力的描述 |
| **低** | 补充 IDEncoder 参考窗口敏感性的局限性讨论 |
| **低** | 验证模型参数量估算 |
