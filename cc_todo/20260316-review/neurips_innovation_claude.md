# NeuroHorizon 项目创新性评估与 NeurIPS 发表策略分析

> 文档日期：2026-03-16
> 文档性质：内部研究讨论，非公开发表材料
> 目标：客观评估项目当前状态，分析 NeurIPS/ICLR/Nature Methods 发表可行性，制定务实策略

---

## 一、当前项目的实际贡献盘点

在讨论发表策略之前，必须先诚实地盘点项目到目前为止**真正做到了什么**，而非原本计划做什么。

### 1.1 POYO+ Encoder 从 Behavior Decoding 到 Forward Prediction 的成功迁移

这是一个**非平凡的贡献**，尽管看起来简单。POYO+ 原本设计用于行为解码（从神经活动预测行为变量），其 Perceiver 架构（event-level spike tokenization → cross-attention to latent tokens → self-attention processor）并非天然适合前向预测任务。前向预测要求模型理解神经群体动力学的时间演化规律，而不仅仅是提取与行为相关的低维特征。实验证明 POYO+ encoder 在编码任务上同样有效，这暗示 Perceiver 架构捕获的 latent representation 具有比预期更强的通用性。

但必须承认：这一贡献的"天花板"有限。审稿人可能会认为这只是"将别人的模型用于新任务"，除非能提供深入的分析解释**为什么** Perceiver 架构比 vanilla Transformer 更适合神经编码。

### 1.2 PerNeuronMLPHead 设计

PerNeuronMLPHead 采用 T-token 设计（bin_repr D/2 + unit_emb D/2 → shared MLP → log_rate），避免了 O(T×N) 的序列长度爆炸问题。这是一个实用的工程贡献，使得模型在神经元数量较多时仍然可行。然而，这类输出头设计在 neural data 领域并不罕见（NDT2 也有类似的 per-neuron 输出设计），因此其新颖性有限。

### 1.3 系统性 Benchmark Comparison

在**统一框架**下对 4 种模型（NeuroHorizon v2、Neuroformer、IBL-MtM、NDT2）进行了公平对比，覆盖 3 个预测窗口（250ms/500ms/1000ms），使用相同的数据预处理、训练配置和评估指标。这类系统性对比在该领域是稀缺的——大多数论文只在自己的设置下与 1-2 个 baseline 比较。

核心数据：

| 模型 | 参数量 | 250ms fp-bps | 500ms fp-bps | 1000ms fp-bps |
|---|---|---|---|---|
| NeuroHorizon v2 | ~2.1M | 0.2115 | 0.1744 | 0.1317 |
| Neuroformer | ~4.9M | 0.1856 | 0.1583 | 0.1210 |
| IBL-MtM | ~10.7M | 0.1749 | 0.1531 | 0.1001 |
| NDT2 | ~4.8M | 0.1691 | 0.1502 | 0.1079 |

NeuroHorizon v2 以 9-32% 的优势超越所有 baseline，且参数量仅为竞品的 20-44%。这一结果本身是有说服力的。

### 1.4 Exposure Bias 在神经数据中的深入实验分析

这可能是项目中**最独特的贡献之一**。经过 4 轮迭代实验，系统性地探索了显式 AR 反馈的各种策略，最终发现所有显式 AR 方案都未能超越 baseline_v2（无预测反馈、仅因果自注意力）。最佳显式 AR 结果为 0.2004/0.1526/0.1218，显著低于 baseline 的 0.2115/0.1744/0.1317。

这是一个**反直觉的负面结果**：在自然语言处理中，AR 是标准范式且效果显著，但在 20ms bin 级别的神经活动预测中，显式 AR 反馈反而引入了误差累积。这一发现如果被正确框架化，具有相当的学术价值。

### 1.5 详细的 Ablation Studies

项目积累了大量消融实验数据：
- **观测窗口 ablation**：500ms 为最优，过短（250ms）信息不足，过长（1000ms）引入噪声
- **Session scaling**：1 session 时 NeuroHorizon 表现最差（0.089，低于所有 baseline），10 session 时表现最优（0.212）——这揭示了模型对多 session 数据的强依赖
- **训练策略**：continuous sampling 严格优于 trial-aligned
- **1 session 场景下 Neuroformer 领先**（0.150 vs 0.089）

### 1.6 参数效率

2.1M 参数超越 4.8-10.7M 的模型，参数效率优势明显。但需注意，参数量差异也可能是性能差异的混淆因素之一——模型之间的架构差异太大，难以将性能提升归因于任何单一因素。

---

## 二、以 AR 为核心创新叙事的风险分析

### 2.1 核心矛盾

项目最初的叙事是"自回归预测是神经编码的正确范式"，但实验结果恰恰相反：

- **baseline_v2**（表现最好的模型）实际上是 **parallel prediction with causal constraint**——它使用因果自注意力掩码使得每个时间 bin 只能看到之前的 bin，但**不**将自身的预测结果反馈到后续 bin 的输入中
- 所有 4 轮显式 AR 反馈实验都失败了

这意味着：如果论文以"AR 预测"作为核心创新点，审稿人会立刻抓住这个致命矛盾。

### 2.2 "如果去掉 causal mask 会怎样？"

这是一个**必须回答但尚未回答**的关键问题。baseline_v2 使用了 causal self-attention，这引入了某种时间顺序的归纳偏置。但如果将因果掩码替换为完全的双向注意力（fully parallel decoder，所有 bin 都能看到所有 bin），性能会如何变化？

- 如果 **parallel decoder 表现更好或相当**：说明因果约束都不需要，整个 AR 叙事彻底崩塌
- 如果 **parallel decoder 表现显著更差**：说明因果顺序信息确实有价值，即使不需要显式反馈

**这个 ablation 是论文能否站住脚的关键实验，必须在提交前完成。**

### 2.3 baseline_v2 的本质

严格来说，baseline_v2 不是"自回归模型"。它是一个**带因果约束的并行预测模型**。在 NLP 中，GPT 也使用因果掩码，但 GPT 在推理时是逐 token 生成的（真正的 AR）。baseline_v2 在推理时一次性输出所有 bin 的预测，更接近于 BERT 的 masked prediction 而非 GPT 的 autoregressive generation。

如果审稿人追问这一点，你很难辩护说这是一个"AR 模型"。

---

## 三、NeurIPS 审稿人可能提出的关键质疑

### Q1: "Your main contribution is AR prediction, but your best model doesn't use AR feedback. How is this an AR contribution?"

**毁灭性：9/10**

这是最致命的质疑。如果论文标题或摘要中包含"autoregressive"一词，审稿人会期望看到 AR 机制带来的性能提升。而实验结果显示 AR 反馈反而有害。

**可能的回应**：将叙事转向"我们发现因果注意力（而非显式 AR 反馈）是神经预测的正确归纳偏置"。但这个回应较弱——因果注意力不是新技术，Transformer decoder 天然就有。

**评估**：如果坚持 AR 叙事，这个问题几乎无解。必须重新框架化论文的核心贡献。

### Q2: "The performance improvement over baselines could be entirely due to the encoder (POYO+ vs Transformer). Did you control for this?"

**毁灭性：8/10**

这是一个合理且严重的混淆因素。NeuroHorizon 使用 POYO+ Perceiver encoder，而 Neuroformer/NDT2/IBL-MtM 使用不同的 Transformer 变体。性能差异可能完全来自 encoder 的差异，而非 decoder 设计。

**可能的回应**：
- 设计一个 ablation：将 POYO+ encoder 与 Neuroformer 的 decoder 结合，或将 vanilla Transformer encoder 与 NeuroHorizon 的 decoder 结合
- 如果 POYO+ encoder + 简单线性 decoder 就能超越 baseline，则证明性能确实来自 encoder
- 如果替换 encoder 后性能大幅下降，则证明 encoder 和 decoder 都有贡献

**评估**：这个 ablation 技术上可行，但工作量较大（需要实现 encoder-decoder 的交叉组合）。如果不做这个实验，审稿人几乎一定会要求 revision。

### Q3: "You only evaluate on one dataset (Perich-Miller motor cortex). How generalizable are your findings?"

**毁灭性：7/10**

目前所有实验仅在一个数据集上进行。NeurIPS 审稿人通常期望至少 2-3 个不同的数据集（不同脑区、不同动物、不同实验范式）来验证泛化性。

**可能的回应**：
- 增加 IBL 数据集（视觉皮层，不同于运动皮层）
- 增加 Allen Institute 的 Neuropixels 数据
- 如果只有一个数据集，可以在不同 session 子集上做 cross-validation

**评估**：这是一个"可修复但需要时间"的问题。增加 1-2 个数据集是可行的，但需要数据预处理和重新训练的时间。对于 NeurIPS 提交，这是一个中等优先级的待办事项。

### Q4: "The fp-bps numbers seem modest. Are these improvements practically significant for BCI applications?"

**毁灭性：5/10**

fp-bps 数值（0.13-0.21）在绝对值上看起来不大。审稿人可能会质疑这些改进是否有实际意义。

**可能的回应**：
- fp-bps 是以 bits per spike 为单位的对数度量，0.21 vs 0.17 在实际预测精度上的差距可能比数字看起来大
- 提供 spike count prediction 的可视化对比，直观展示预测质量
- 讨论在 BCI 在线解码场景中，这种精度提升的潜在价值
- 引用 NLB 和 POYO 论文中的类似数值范围，说明这是该领域的正常量级

**评估**：这个问题可以通过补充可视化和讨论来化解，不是致命问题。

### Q5: "Neuroformer is the closest competitor and also uses AR. The comparison is confounded by different encoder architectures."

**毁灭性：7/10**

Neuroformer 也使用自回归策略预测神经活动，是最直接的竞争对手。但 NeuroHorizon 和 Neuroformer 之间的差异涉及多个维度：encoder 架构、tokenization 方式、参数量、训练策略。无法将性能差异归因于任何单一因素。

**可能的回应**：
- 设计控制实验：在相同 encoder 下对比不同 decoder 策略
- 分析 Neuroformer 在 1 session 场景下的优势（0.150 vs 0.089），讨论其 event-level tokenization 在小数据场景下的优势
- 承认混淆因素的存在，将贡献定位为"整体系统设计"而非"单一技术创新"

**评估**：如果不做 encoder 控制实验，这个问题与 Q2 叠加将极大削弱论文的说服力。

---

## 四、与 Baseline 对比中的混淆因素分析

当前的 benchmark 对比存在以下无法忽视的混淆因素：

### 4.1 Encoder 架构差异

| 模型 | Encoder 类型 | 核心机制 |
|---|---|---|
| NeuroHorizon | POYO+ Perceiver | Event-level tokenization + cross-attn to latent |
| Neuroformer | Vanilla Transformer | Causal Transformer encoder |
| NDT2 | Masked Transformer | Bin-level tokenization + masked modeling |
| IBL-MtM | Transformer | Multi-task Transformer |

POYO+ Perceiver 的 event-level tokenization 和 cross-attention to latent tokens 机制可能天然比 bin-level tokenization 更高效地捕获稀疏 spike 数据中的信息。这一架构优势可能是性能差异的**主要来源**。

### 4.2 参数量差异

2.1M vs 4.8-10.7M 的参数量差异是一个双刃剑：
- **有利解读**：NeuroHorizon 参数效率更高
- **不利解读**：如果将 baseline 的参数量降到 2.1M，性能差距可能缩小甚至消失；反之，如果将 NeuroHorizon 的参数量增加到 5M，性能可能进一步提升

缺少**参数量控制实验**（iso-parameter comparison）是一个显著弱点。

### 4.3 训练策略差异

Continuous sampling vs trial-aligned、不同的 learning rate schedule、不同的 data augmentation 策略——这些都可能影响最终性能。虽然项目在统一框架下尽量控制了这些变量，但不同模型的原始设计可能对某些训练策略有天然偏好。

### 4.4 结论

**在不做额外控制实验的情况下，当前结果只能证明"NeuroHorizon 作为一个完整系统优于 baseline"，无法证明任何单一技术组件（encoder、decoder、AR 策略）的优越性。** 这对于以某个特定技术创新为核心叙事的论文来说是一个严重问题。

---

## 五、不同 Reframing 方案的可行性分析

### Option A: "Foundation Model Architecture for Neural Encoding"

**核心叙事**：POYO+ encoder 是一个通用的神经数据基础模型架构，不仅适用于行为解码（原论文），也可以成功迁移到前向预测（编码）任务，且在多 session 场景下展现出优越的 scaling 特性。

**优势**：
- 与实验结果高度一致——性能提升的主要来源确实可能是 encoder
- "基础模型"是当前 AI 领域的热点叙事
- 可以诚实地呈现 AR 实验结果作为补充分析

**劣势**：
- POYO+ 是已发表工作（ICLR 2025），本文的增量性质明显
- 审稿人可能认为这是"应用论文"而非"方法论文"
- NeurIPS 审稿人常见批评："What is the technical novelty beyond applying POYO+ to a new task?"

**所需额外实验**：
- Encoder ablation（替换 POYO+ encoder 为 vanilla Transformer）
- 更多数据集验证 encoder 的通用性
- Encoder representation 的可解释性分析

**NeurIPS 可行性**：**中低**。除非能提供深入的 encoder 分析和多数据集验证，否则增量性太强。

**更适合的 venue**：ICLR Workshop 或 Computational Neuroscience 方向的会议（如 COSYNE）。

### Option B: "Systematic Study of Long-Horizon Neural Prediction"

**核心叙事**：提供了神经活动长时程前向预测的首个系统性 benchmark，在统一框架下对比了 4 种代表性方法，揭示了预测窗口、session 数量、训练策略等因素的影响，并发现了自回归反馈在神经数据中失效的现象。

**优势**：
- 最诚实的框架，不需要夸大任何单一创新点
- Benchmark 论文在社区中有持久的影响力（如 NLB benchmark）
- 负面结果（AR 失效）可以自然地融入分析
- 丰富的 ablation 数据支撑

**劣势**：
- NeurIPS 近年来对纯 empirical study 的接受度下降
- "没有提出新方法"可能被视为贡献不足
- 只有一个数据集，benchmark 的代表性受限

**所需额外实验**：
- 至少增加 2 个数据集（不同脑区、不同范式）
- 开源 benchmark 代码和标准化评估流程
- 可能需要增加更多 baseline（如 LFADS, pi-VAE）

**NeurIPS 可行性**：**中等**。如果能做成一个真正的 benchmark（多数据集、多模型、开源代码），有一定机会，特别是作为 Datasets and Benchmarks track 的投稿。

**更适合的 venue**：NeurIPS Datasets and Benchmarks Track、Nature Methods。

### Option C: "Why Autoregressive Prediction Fails for Neural Data"

**核心叙事**：通过系统性实验揭示了自回归预测在 bin-level 神经活动数据中失效的原因——exposure bias 在高噪声、低信噪比的 spike count 数据中被急剧放大。这一发现对神经科学 AI 社区具有重要的指导意义。

**优势**：
- 独特的贡献——目前文献中缺乏对 AR 在神经数据中失效原因的深入分析
- 反直觉的结论具有传播力
- 4 轮迭代实验提供了充分的证据
- 可以自然地引出"因果注意力 vs 显式 AR"的理论讨论

**劣势**：
- NeurIPS 对负面结果论文的接受度历史上较低（虽然近年有所改善）
- "这个方法不行"不如"这个方法很行"有吸引力
- 需要更深入的理论分析（不仅仅是实验现象，还需要解释**为什么**）
- 需要对比不同时间分辨率（5ms vs 20ms vs 100ms bins）下 AR 的表现变化

**所需额外实验**：
- 不同 bin 大小下 AR 性能的系统性比较
- 理论分析：exposure bias 的误差传播模型
- 与 NLP/CV 中 AR 成功场景的对比分析
- 可能的改进方案（如 scheduled sampling 的变种）

**NeurIPS 可行性**：**中低至中等**。需要非常强的分析深度和理论洞察来支撑一篇以"负面结果"为核心的论文。如果分析足够深入，有机会进入讨论环节。

**更适合的 venue**：ICLR（更欢迎深入分析类论文）、NeurIPS Workshop。

### Option D: "Pivot to Latent Dynamics / Diffusion Decoder"

**核心叙事**：保留 POYO+ encoder，将 decoder 替换为基于隐空间动力学模型（如 latent ODE/SDE）或扩散模型的生成器，实现从 deterministic prediction 到 probabilistic generation 的范式转换。

**优势**：
- 如果成功，技术新颖性最强
- 隐空间动力学与神经科学的 dynamical systems 理论天然契合
- 扩散模型在生成任务中的成功有强大先验支持
- 可以同时解决"AR 失效"和"encoder 贡献不清"的问题

**劣势**：
- 需要大量额外工作（新 decoder 设计、训练、调参）
- 不确定性高——可能投入大量时间后效果不佳
- 与当前代码和实验的距离最远
- 如果目标是近期会议截止日（如 NeurIPS 2026 五月截稿），时间极其紧张

**所需额外工作**：
- 设计并实现 latent dynamics decoder 或 diffusion decoder
- 完整的训练和评估流程
- 与现有方法的对比
- 预计需要 2-4 个月的全职工作

**NeurIPS 可行性**：**高（如果成功的话）**。但"如果成功"是一个巨大的条件。

---

## 六、建议的论文叙事框架

### 6.1 各 venue 的最佳策略

**NeurIPS 2026 主会议**：
- 首选 **Option D**（如果时间允许且初步实验有希望）
- 次选 **Option B + Option C 的混合体**——"Long-horizon neural prediction: a systematic benchmark and the failure of autoregressive feedback"
- 必须增加至少 1 个额外数据集

**ICLR 2027**：
- 首选 **Option C** 的深化版——"Understanding when and why autoregressive prediction fails in neural data"
- Option B 也适合 ICLR 的风格

**Nature Methods**：
- 首选 **Option B**——强调方法论的系统性比较和实用指导
- Nature Methods 更看重实际应用价值而非技术新颖性
- 需要更多数据集和更详细的 usage guidelines

### 6.2 如何将负面 AR 结果转化为正面贡献

关键策略：**不要回避负面结果，而是将其升华为洞察**。

建议的叙事结构：
1. **引言**：长时程神经活动预测是 BCI 和计算神经科学的关键挑战；AR 是 NLP/CV 中的主导范式，直觉上也适用于神经数据的时序预测
2. **方法**：介绍统一的 benchmark 框架和多种预测策略（包括显式 AR 和因果并行预测）
3. **结果**：
   - NeuroHorizon（POYO+ encoder + causal parallel decoder）在 10 session 场景下以 9-32% 优势超越所有 baseline
   - **关键发现**：显式 AR 反馈不仅没有帮助，反而有害（详细的 4 轮实验分析）
   - 因果注意力掩码提供的**隐式时间顺序信息**已经足够，显式反馈引入的误差累积抵消了其信息增益
4. **讨论**：分析 AR 在神经数据中失效的原因（spike count 的高噪声、20ms bin 的时间尺度、exposure bias 的放大效应），与 NLP 中 AR 成功的条件对比

### 6.3 所需额外实验（按优先级排序）

1. **[关键] Non-causal decoder ablation**：去掉 causal mask 的 parallel decoder，验证因果约束是否有价值。如果这个实验的结果是因果约束确实有帮助，则强化"implicit temporal order matters"的叙事；如果没有帮助，则需要彻底重新定位
2. **[关键] Encoder ablation**：用 vanilla Transformer encoder 替换 POYO+ encoder，量化 encoder 的贡献
3. **[重要] 增加至少 1 个数据集**（建议 IBL 视觉皮层数据或 Allen Neuropixels 数据）
4. **[重要] 不同 bin 大小下的 AR 性能比较**（5ms, 20ms, 50ms, 100ms）
5. **[有益] 参数量控制实验**（iso-parameter comparison）
6. **[有益] 预测结果可视化**（spike raster 对比、population activity pattern 对比）

### 6.4 时间线评估

假设 NeurIPS 2026 截稿时间为 2026 年 5 月中旬：

- **现在到 3 月底**（2 周）：完成实验 1（non-causal ablation）和实验 2（encoder ablation）
- **4 月上旬**（2 周）：增加新数据集 + bin 大小 ablation
- **4 月中旬到下旬**（2 周）：论文初稿撰写 + 可视化
- **5 月上旬**（2 周）：论文修改 + 审阅 + 最终提交

这个时间线非常紧张但技术上可行，**前提是不选择 Option D（pivot to new decoder）**。如果选择 Option D，建议目标调整为 ICLR 2027。

---

## 七、与同期工作的定位

### 7.1 POYO (NeurIPS 2023) / POYO+ (ICLR 2025)

**关系**：NeuroHorizon 直接构建在 POYO+ 之上，使用其 encoder 架构。

**差异化策略**：
- POYO/POYO+ 解决的是 behavior decoding（从神经到行为），NeuroHorizon 解决的是 forward prediction（从历史神经到未来神经）
- 但这种差异化较弱——审稿人可能认为只是"换了个任务头"
- 需要强调编码任务的独特挑战（时间动力学建模 vs 静态特征提取）

### 7.2 Neuroformer (NeurIPS 2023)

**关系**：最直接的竞争对手，也使用 AR 策略预测神经活动。

**关键差异**：
- Neuroformer 使用 event-level causal Transformer，NeuroHorizon 使用 Perceiver encoder + bin-level decoder
- Neuroformer 在 1 session 场景下表现更好（0.150 vs 0.089），NeuroHorizon 在多 session 场景下大幅领先
- 这一对比可以引出有趣的讨论：event-level tokenization 在小数据量下的优势 vs Perceiver 在大数据量下的 scaling 优势

**风险**：审稿人可能是 Neuroformer 的作者或密切相关人员，需要在论文中给予充分的 credit 和公平的对比。

### 7.3 NDT2 / IBL-MtM

**关系**：Parallel prediction 方法的代表。

**定位**：NeuroHorizon 在所有预测窗口上超越这两个方法，可以作为 baseline 对比。但需要承认参数量和架构的差异可能是混淆因素。

### 7.4 LFADS (Nature Methods 2018)

**关系**：经典的隐空间动力学方法，目前仍是该领域的重要参考。

**定位**：
- LFADS 使用 VAE + RNN 学习 latent dynamics，与 Transformer-based 方法有本质区别
- 如果选择 Option D（latent dynamics decoder），LFADS 将成为直接比较对象
- 即使不选择 Option D，也应该在相关工作中讨论 LFADS，并解释为什么选择了 bin-level prediction 而非 latent dynamics

### 7.5 SPINT (NeurIPS 2025)

**关系**：解决 cross-session generalization 问题，与 NeuroHorizon 的 Innovation 1（IDEncoder）相关但方向不同。

**定位**：
- SPINT 通过 spatial invariance 实现跨 session 泛化
- NeuroHorizon 计划通过 IDEncoder（gradient-free unit embedding inference）实现类似目标，但尚未实现
- 如果 IDEncoder 能在提交前实现并展示初步结果，将显著增强论文的独特性
- 如果不能实现，应避免在论文中过度承诺

---

## 八、总结与建议

### 8.1 项目的真实定位

**NeuroHorizon 当前是一个基于 POYO+ encoder 的高效神经活动前向预测系统，在多 session 场景下展现了 state-of-the-art 性能，但其原本声称的核心创新（自回归预测）在实验中未能证实其价值。**

### 8.2 最务实的发表策略

1. **短期（NeurIPS 2026）**：采用 Option B + C 的混合策略，以"systematic benchmark + understanding AR failure"为核心叙事。必须在提交前完成 non-causal ablation 和 encoder ablation 两个关键实验。增加至少 1 个新数据集。

2. **中期（ICLR 2027 或 NeurIPS 2027）**：如果时间允许，探索 Option D（latent dynamics decoder 或 diffusion decoder），追求更强的技术新颖性。

3. **替代方案（Nature Methods）**：如果 NeurIPS 审稿结果不理想，可以将论文扩展为更完整的 benchmark + 方法论文投稿 Nature Methods，强调实用价值和系统性比较。

### 8.3 必须避免的错误

1. **不要以"autoregressive prediction"作为论文标题或核心卖点**——实验数据不支持这个叙事
2. **不要隐藏负面结果**——审稿人会发现，而且诚实的负面分析反而是加分项
3. **不要在没有 encoder ablation 的情况下声称 decoder 的创新性**——这是最容易被审稿人攻击的弱点
4. **不要过度承诺尚未实现的功能**（IDEncoder、multimodal conditioning）——论文应该只呈现已完成的工作
5. **不要低估 Neuroformer 在 1 session 场景下的优势**——应该公平讨论并分析原因

### 8.4 最后的诚实评估

以当前的实验结果和时间约束，NeuroHorizon 要在 NeurIPS 2026 主会议上发表面临**相当大的挑战**。核心问题不是结果不好（实际上性能数据很强），而是**缺乏一个清晰的、与实验结果一致的创新叙事**。性能提升很可能主要来自 encoder（POYO+），而不是项目原本声称创新的 decoder（AR 预测）。

最可行的路径是：诚实地重新定位论文的贡献，补做 2-3 个关键 ablation 实验以支撑新的叙事，然后根据结果选择最合适的 venue。一篇诚实、深入、系统性的论文，即使不是顶会主会议，也能在社区中产生有价值的影响。

---

> *本文档仅供内部讨论使用，基于截至 2026-03-16 的实验数据和分析。*
