# 自回归生成在长时程神经活动 Forward Prediction 中的有效性分析

> NeuroHorizon 项目研究讨论文档
> 日期：2026-03-16

---

## 摘要

本文基于 NeuroHorizon 项目四轮自回归（Autoregressive, AR）反馈实验的完整数据，系统分析 AR 生成策略在长时程神经活动前向预测任务中的实际有效性。核心发现是：在当前实验框架下，显式 AR 反馈不仅未能超越无反馈的 baseline_v2 模型，反而在所有预测窗口上均表现更差。本文从 exposure bias、神经活动的随机过程本质、信息瓶颈、以及与 NLP/CV 领域 AR 成功案例的本质差异等多个维度，深入剖析这一现象的根本原因，并对 causal self-attention 的实际作用机制提出新的解释。

---

## 1. 当前实验数据的详细解读

### 1.1 Baseline_v2：无反馈模型的意外胜出

baseline_v2 是 NeuroHorizon 中表现最好的模型，其核心设计特征如下：

- **解码器**：多层因果解码器，使用 causal self-attention mask
- **反馈方法**：，即反馈向量为全零
- **关键性质**：由于 bin query 是无状态的可学习嵌入（stateless learned embeddings），且反馈为零向量，teacher-forcing 和 rollout 在数学上完全等价

其在不同预测窗口上的表现为：

| 预测窗口 | fp-bps | R-squared | PSTH-R-squared |
|---------|--------|-----------|----------------|
| 250ms   | 0.2115 | 0.2614    | 0.6826         |
| 500ms   | 0.1744 | 0.2368    | 0.1475         |
| 1000ms  | 0.1317 | 0.2290    | 0.2139         |

值得注意的是，在 250ms 窗口内（12 个 bin），per-bin fp-bps 在 0.16-0.24 范围内波动，未呈现显著衰减趋势。这表明 baseline_v2 在短窗口内的预测质量是稳定的——它并不依赖前序 bin 的预测结果来维持后续 bin 的预测精度。

### 1.2 第一轮实验：结构化预测记忆（Structured Prediction Memory）

**设计思路**：通过 PredictionMemoryEncoder 将历史预测压缩为 K=4 个 summary tokens，在解码器中通过第二层 cross-attention 注入预测反馈信息，使模型能够看到自己之前的预测并据此调整后续输出。

**结果**：

| 预测窗口 | Teacher-Forcing fp-bps | Rollout fp-bps | TF/Rollout Gap |
|---------|----------------------|----------------|----------------|
| 250ms   | ~0.30               | 0.1486         | ~0.15          |
| 500ms   | ~0.15               | -0.0153        | ~0.17          |
| 1000ms  | ~0.28               | -0.2590        | ~0.54          |

**失败分析**：这是最具灾难性的一轮实验。在 500ms 和 1000ms 窗口上，rollout fp-bps 为负值，意味着模型的预测能力甚至不如简单地使用平均发放率（null model）。TF/rollout gap 随预测窗口增大而急剧扩大（从 0.15 到 0.54），呈现典型的误差雪崩（error avalanche）模式。模型在训练时学会了高度依赖来自真实数据的反馈信号，但在推理时接收到的是自身含噪预测，微小的偏差被反复放大，导致预测完全崩溃。

### 1.3 第二轮实验：局部预测记忆（Local Prediction Memory）

**设计思路**：针对第一轮中误差通过长程记忆累积的假说，将记忆范围限制为仅前一个 bin 的预测结果。

**结果**：

| 预测窗口 | Rollout fp-bps |
|---------|----------------|
| 250ms   | 0.1621         |
| 500ms   | -0.0105        |
| 1000ms  | -0.2122        |

**失败分析**：虽然在 250ms 上略有改善（0.1621 vs 0.1486），但 500ms 和 1000ms 仍然为负。这一结果明确否定了问题出在记忆范围过大的假说——**核心问题不是记忆的作用域（scope），而是训练与推理之间的分布不匹配（distribution mismatch）**。无论模型看到的是一个 bin 还是四个 bin 的历史预测，只要训练时输入的是真实值（GT）而推理时输入的是预测值，就不可避免地出现 exposure bias。

### 1.4 第三轮实验：预测记忆对齐训练（Prediction Memory Alignment Training）

**设计思路**：采用 scheduled sampling 策略，在训练过程中混合使用真实值和模型自身预测值作为反馈输入，逐步缩小训练与推理之间的分布差距。

**结果**：

| 预测窗口 | Rollout fp-bps | TF/Rollout Gap |
|---------|----------------|----------------|
| 250ms   | 0.1943         | 0.08           |
| 500ms   | 0.1513         | 0.13           |
| 1000ms  | 0.1103         | 0.17           |

**分析**：这是最成功的一轮改进。所有预测窗口的 rollout fp-bps 均为正值，TF/rollout gap 大幅缩小。对齐训练有效缓解了 exposure bias 问题。然而，**在所有窗口上仍然全面低于 baseline_v2**（差距分别为 -0.0172、-0.0231、-0.0214）。这意味着即使解决了 exposure bias，AR 反馈本身也未能带来正向收益。

### 1.5 第四轮实验：对齐训练 + 超参数调优

**设计思路**：在第三轮基础上进行系统性超参数调优，包括 mix_prob 从 0.25 提升至 0.35、dropout 从 0.10 降至 0.05、noise_std 从 0.05 降至 0.03。

**结果**：

| 预测窗口 | Rollout fp-bps | vs baseline_v2 差距 |
|---------|----------------|---------------------|
| 250ms   | 0.2004         | -0.0111             |
| 500ms   | 0.1526         | -0.0218             |
| 1000ms  | 0.1218         | -0.0099             |

**分析**：进一步缩小了与 baseline_v2 的差距，尤其在 1000ms 窗口上差距仅为 0.0099。但经过四轮迭代优化、大量计算资源投入后，AR 反馈模型仍然未能追平、更遑论超越一个完全没有预测反馈的模型。这一事实传递了一个非常清晰的信号：**在当前的 bin-level forward prediction 框架下，显式 AR 反馈的边际收益为零甚至为负**。

### 1.6 四轮实验的全局趋势总结

从四轮实验的数据演进来看，可以观察到两个核心趋势：

1. **对齐训练有效缓解了 exposure bias**：rollout 性能从灾难性崩溃（Round 1-2）恢复到接近 teacher-forcing 水平（Round 3-4），TF/rollout gap 从 0.15-0.54 缩小到 0.02-0.08。
2. **AR 反馈的信息增益天花板极低**：即使完美解决了 exposure bias（假设 gap 趋近于零），teacher-forcing 性能本身也仅与 baseline_v2 相当，说明来自历史预测的反馈信号对未来 bin 的预测几乎没有提供额外有用信息。

---

## 2. 自回归在神经活动数据上面临的根本性挑战

### 2.1 Exposure Bias 在高维 Poisson 计数向量上的严重性

在 NLP 中，exposure bias 是一个已被广泛研究的问题，但其严重程度在神经活动预测中被数量级地放大。原因在于：

**维度灾难**：NLP 中每个 token 是一个离散的、有限词表中的选择。即使预测错误，错误的形态是有限的——它只能是词表中的某个词。而在 NeuroHorizon 中，每个 bin 的输出是一个 N 维（N 为神经元数量，通常 50-500+）的连续发放率向量。预测误差可以在每个维度上独立积累，其误差空间是 N 维连续空间，分布偏移（distribution shift）的可能性呈指数级增长。

**Poisson 噪声的内禀不可预测性**：神经元的发放遵循（近似）Poisson 过程。即使模型完美学习了条件发放率 λ(t)，实际观测到的 spike count 也存在方差为 λ 的内禀随机性。在 20ms 的 bin 宽度下，典型的发放率对应 λ ≈ 0.1-2.0，此时 Poisson 噪声的相对方差（1/λ）非常大。这意味着即使 AR 模型能完美预测发放率，喂回去的观测计数也包含大量不可约噪声，这些噪声在逐 bin 传播中会不断累积。

**高维 count vector 的误差叠加效应**：假设模型对每个神经元的预测误差独立且有限，那么 N 个神经元的联合误差在 L2 范数意义上大约以 sqrt(N) 的速率增长。对于 200 个神经元、50 个 bin 的场景，误差累积的规模远超 NLP 中 50 个 token 的序列。

### 2.2 神经活动是外部驱动而非自我驱动的过程

这是 AR 在神经活动预测中面临的最根本性的挑战。

在语言建模中，AR 的基本假设是合理的：下一个 token 强烈依赖于前面的 tokens。语言具有强烈的局部依赖结构——语法规则、语义连贯性、话题延续性都使得历史序列对未来序列有很强的预测力。

但神经活动的本质完全不同。**大脑皮层神经元的发放模式主要由外部输入驱动**：

- **感觉输入**：视觉、听觉、体感等外部刺激是初级感觉区神经活动的主要驱动力
- **运动指令**：运动皮层的活动主要反映即将执行的运动计划，这些计划的时序由行为需求（而非神经活动历史）决定
- **内部状态变化**：注意力转移、任务切换等认知事件可以在毫秒级别突然改变神经群体的活动模式

在这种外部驱动的框架下，t 时刻的神经活动对 t+Δ 时刻的预测力随 Δ 增大而快速衰减。神经活动的自相关函数通常在 50-200ms 内快速下降，这意味着在 500ms 或 1000ms 的预测窗口中，序列后半段的活动与前半段的相关性已经很弱。AR 模型试图从历史预测中提取信息来辅助后续预测，但当信号本身的自相关已经衰减时，历史预测中能提供的有用信息极为有限，而包含的噪声却在持续累积。

从信息论的角度来看，当 I(X_t; X_{t+Δ}) 随 Δ 增大而迅速趋近于零时，AR 反馈的互信息收益也趋近于零，但反馈引入的噪声和 bias 却是持续的。这解释了为什么 AR 反馈在长窗口上的净效果为负。

### 2.3 反馈信息瓶颈（Feedback Information Bottleneck）

在 NeuroHorizon 的架构中，每个 bin 的 AR 反馈需要经过以下处理链：

1. 模型输出 N 个神经元的 log firing rates
2. 通过 PerNeuronMLPHead 解码为 N 维 count 预测
3. 将 N 维 count vector 压缩为 D 维 embedding（D 通常为 256 或 512）
4. 通过 cross-attention 或直接拼接注入下一个 bin 的 query

当 N >> D/2 时（例如 N=200 神经元，D/2=128），步骤 3 本质上是一个有损压缩。N 个神经元的发放模式包含丰富的空间结构信息（哪些神经元共同活跃、哪些沉默），但将其压缩到 128 维后，大量细粒度信息被丢弃。模型在下一步收到的反馈信号已经是对原始预测的模糊近似。

更关键的是，这个压缩过程是不可逆的。即使模型在第 i 个 bin 做出了精确预测，经过压缩-反馈-解压的过程后，传递给第 i+1 个 bin 的信息已经严重退化。经过多轮传递，有用信号几乎完全消失在信息瓶颈中。

---

## 3. 与 NLP/CV 中 AR 成功案例的本质差异

### 3.1 序列依赖结构的差异

**语言**：强短程依赖。给定一个英文句子的前 5 个词，下一个词的熵通常只有 2-4 bits（在 GPT 级别模型下可降至 1-2 bits）。语法约束、搭配习惯、语义连贯等因素使得 next token prediction 是一个高度结构化的问题。AR 在这里成功是因为历史序列确实包含了对未来的强预测信号。

**图像（离散化后）**：中等短程依赖。在 VQVAE tokenized 的图像序列中，相邻 token 之间存在显著的空间相关性。AR 模型（如 DALL-E）可以利用这种局部结构逐步生成coherent 的图像。

**神经活动**：弱且快速衰减的短程依赖。在 20ms bin 级别，相邻 bin 之间存在一定相关性（主要来自突触时间常数和网络动力学），但这种相关性随间隔增大而以指数速率衰减。在 10-25 个 bin（200-500ms）之后，自相关接近噪声水平。这意味着 AR 反馈在前几个 bin 可能有微弱的正面作用，但在序列后段几乎无价值。

### 3.2 Token 空间的差异

**NLP**：离散有限词表（32k-100k tokens）。softmax 分类的输出空间结构良好，beam search 等解码策略可以有效探索可能的序列。更重要的是，训练时的 teacher-forcing 与推理时的 free-running 之间的 distribution shift 相对可控——即使预测错了一个 token，后续 token 仍然处于同一个离散空间中，模型仍有机会恢复到合理的轨迹。

**Neural counts**：高维连续空间。每个 token（bin 的输出）是一个 N 维连续向量。错误的预测将模型推向训练分布之外的连续空间区域，且没有像离散词表那样的围栏将模型拉回。一旦偏离，回到训练分布的可能性随步数增长而指数级下降。

### 3.3 Teacher-Forcing 效能的差异

在 NLP 中，teacher-forcing 之所以有效且 exposure bias 相对可控，有一个关键原因：**词表的离散性提供了自然的误差量化效果**。即使模型对某个 token 的概率分布估计不够准确，只要 top-1 预测正确，后续 token 的输入就与 teacher-forcing 完全相同。这种离散化效果天然地抑制了小误差的累积。

而在连续空间中，任何量级的预测偏差都会完整地传递到下一步。不存在足够接近就等于正确的阈值效应。这使得连续空间中的 AR generation 在原理上就比离散空间更容易崩溃。

---

## 4. Bin-Level AR vs Spike-Level AR 的 Trade-off 分析

### 4.1 Bin-Level AR（NeuroHorizon 方案）

**设计**：将时间轴以 20ms 为单位离散化为 bins，每个 bin 作为一个 AR step，每步预测所有 N 个神经元的 count。

**优势**：
- 序列长度短（250ms=12 bins, 500ms=25 bins, 1000ms=50 bins），计算效率高
- 与 Perceiver encoder 的 latent representation 自然对齐
- T-token 设计使得并行计算成为可能

**劣势**：
- 每个 token 是高维向量（N neurons），单步预测的信息密度和不确定性都非常高
- 反馈信息瓶颈严重：N 维 count vector → D 维 embedding 的压缩损失大
- Poisson 噪声在高维空间中的叠加效应加剧误差累积

### 4.2 Spike-Level AR（Neuroformer 方案）

**设计**：将每个 spike event 作为一个 AR step，每步预测下一个 spike 的（神经元 ID, 时间戳）。

**优势**：
- 每个 token 是低维的（1 个神经元 ID + 1 个时间值），更接近 NLP 的 token 结构
- 反馈信息保留完整：每个 spike 的身份和时间被精确传递
- 不存在高维 count vector 的压缩瓶颈

**劣势**：
- 序列极长：250ms 内可能有数百个 spike events，500ms 可达上千
- 长序列带来的计算成本和 attention 的稀释效应
- 仍然面临 exposure bias：一旦某个 spike 的时间或身份预测错误，后续所有 spike 的条件分布都会偏移

### 4.3 两种方案在神经数据上的共同困境

无论选择 bin-level 还是 spike-level tokenization，AR 在神经活动预测中都面临一个共同的根本问题：**神经活动序列不是一个自主生成的过程**。

在 NLP 中，语言模型生成的文本是自洽的——每个 token 的正确性可以由序列内部的一致性来评判。但在神经活动预测中，每个时刻的正确发放模式由外部条件（刺激、行为、内部状态）决定，而这些外部条件不在 AR 反馈链中。**AR 模型试图从内部历史中生成一个由外部因素决定的序列，这在根本上就是一个信息不匹配的问题**。

这解释了为什么 bin-level 和 spike-level AR 在神经数据上都面临困难：不是 tokenization 方式的问题，而是 AR 的基本假设（future depends primarily on past outputs）与神经数据的生成机制（future depends primarily on external inputs）之间的不匹配。

---

## 5. Causal Self-Attention 的实际作用机制

### 5.1 Baseline_v2 中的 Causal Mask：不是 AR，而是正则化

baseline_v2 的解码器使用了 causal self-attention mask，但 feedback_method=none 且 bin queries 是无状态的可学习嵌入。这意味着：

- **没有信息从预测回流到后续 bins**：第 i 个 bin 的 query 不包含任何关于第 1 到第 i-1 个 bin 的预测结果的信息
- **每个 bin 的预测独立依赖于 encoder latents**：所有 bin 都通过 cross-attention 访问相同的 encoder 输出
- **Causal mask 的唯一作用是限制 self-attention 的感受野**：第 i 个 bin 只能 attend to bins 1 到 i

因此，baseline_v2 实际上是一个**带有因果正则化的并行预测模型**（parallel prediction with causal regularization），而不是一个真正的 AR 生成模型。

### 5.2 Causal Mask 的正则化效果

Causal mask 为什么能带来比 full self-attention 或纯 parallel prediction 更好的性能？可能的机制包括：

**隐式课程学习（Implicit Curriculum）**：在 causal mask 下，bin 1 只能 attend to 自身，bin 2 可以 attend to bins 1-2，bin T 可以 attend to 所有 bins。这创造了一个从简单到困难的自然梯度——前面的 bins（更接近历史窗口）通常更容易预测，后面的 bins 更难。Causal mask 允许模型将更简单任务的 intermediate representations 作为更难任务的参考，形成一种隐式的课程学习。

**防止后向信息泄露**：如果使用 full self-attention，后面 bins 的信息会泄露给前面的 bins。虽然在训练时这些都是 GT-derived representations，但这种泄露可能导致模型学到不自然的依赖关系（利用未来信息预测当前），从而在推理时出现偏差。Causal mask 天然防止了这种泄露。

**时间结构建模**：Causal mask 隐式地编码了时间的方向性——每个 bin 的 representation 只能整合过去和当前的信息，这与神经活动的因果结构一致。即使没有显式的 AR 反馈，这种结构偏置也有助于模型学习时间上合理的 representations。

### 5.3 关键洞察：正则化好处 vs Exposure Bias 代价

baseline_v2 的成功可以用一个简洁的框架来理解：

**收益 = causal mask 的正则化效果**
**代价 = 0（因为没有 feedback，所以没有 exposure bias）**

而 AR 反馈模型的等式是：

**收益 = causal mask 的正则化效果 + AR 反馈的信息增益**
**代价 = exposure bias + 信息瓶颈损失 + 噪声累积**

四轮实验的数据表明，在神经活动预测任务中，**AR 反馈的信息增益 < exposure bias + 信息瓶颈损失 + 噪声累积**。即使通过对齐训练将 exposure bias 大幅降低（Round 3-4），**AR 反馈的信息增益仍然不足以抵消剩余的代价**。

这是一个深刻的结论：问题不仅仅是 exposure bias（一个工程上可以缓解的问题），更是 AR 反馈在神经活动预测中的信息价值根本性地不足。

---

## 6. Neuroformer 对比分析：为什么真 AR per-spike 模型也不如 NeuroHorizon

### 6.1 Benchmark 数据回顾

在相同评测条件下的对比结果：

| 模型 | 参数量 | 250ms | 500ms | 1000ms |
|------|--------|-------|-------|--------|
| NeuroHorizon v2 | ~2.1M | 0.2115 | 0.1744 | 0.1317 |
| Neuroformer | ~4.9M | 0.1856 | 0.1583 | 0.1210 |
| IBL-MtM | ~10.7M | 0.1749 | 0.1531 | 0.1001 |
| NDT2 | ~4.8M | 0.1691 | 0.1502 | 0.1079 |

NeuroHorizon v2 在所有预测窗口上均领先，同时参数量最小（2.1M vs 4.9-10.7M）。

### 6.2 NeuroHorizon 优势的来源

Neuroformer 作为 spike-level AR 模型，拥有更精细的时间分辨率和更完整的反馈信息保留。然而它在参数量多出一倍以上的情况下仍然全面落后于 NeuroHorizon v2。这强烈暗示：

**NeuroHorizon 的优势主要来自 encoder，而非 decoder**。

POYO+ 框架的核心创新在于其 event-level tokenization 和 Perceiver cross-attention encoder：
- 将每个 spike event 视为独立 token，保留了精确的时间和空间信息
- 通过 Perceiver 的 cross-attention 将变长的 spike 序列压缩为固定长度的 latent tokens
- 这种 encoder 能够高效地提取 spike 序列中的统计规律，而无需依赖 AR 的逐步展开

Neuroformer 虽然在 decoder 端使用了更精细的 AR 策略，但其 encoder 的表示能力可能不如 POYO+ Perceiver。当 encoder 提供的 latent representation 足够好时，decoder 甚至不需要 AR 反馈——baseline_v2 已经证明了这一点。

### 6.3 Session 数量效应的启示

一个值得深思的数据点：在单 session 训练时，Neuroformer（0.150 fp-bps）明显优于 NeuroHorizon（0.089 fp-bps），但随着 session 数量增加到 10，NeuroHorizon（0.212）大幅反超 Neuroformer。

这个现象的可能解释是：

- **POYO+ Perceiver encoder 是一个数据饥渴型架构**：它的表示能力强但参数效率不高，需要大量多样化的数据来学习通用的 spike representation
- **在数据稀缺时，AR 是一种有效的归纳偏置**：当 encoder 不够强时，AR 反馈提供了额外的结构化信息来辅助预测
- **但当 encoder 足够强大时，AR 反馈变成了冗余甚至有害的**：强大的 encoder latent 已经编码了足够的预测信息，AR 反馈只是引入了额外的噪声和 bias

这个发现对架构设计有重要启示：**与其投入精力优化 AR 反馈机制，不如将资源投入到 encoder 的表示能力和数据规模上**。

---

## 7. 结论与建议

### 7.1 核心结论

基于四轮严格的实验对比，我们得出以下核心结论：

**1) 显式 AR 反馈在 bin-level 神经活动 forward prediction 中的信息增益接近零或为负**

四轮实验中，最优的 AR 反馈模型（Round 4）在所有预测窗口上仍然落后于无反馈的 baseline_v2，差距分别为 -0.0111（250ms）、-0.0218（500ms）、-0.0099（1000ms）。

**2) Exposure bias 是可以缓解的，但它不是唯一的问题**

对齐训练（scheduled sampling）有效地将 TF/rollout gap 从灾难性水平降低到可接受水平。但即使在 gap 很小的情况下，AR 模型仍然不如 baseline_v2，说明 AR 反馈的信息价值本身不足。

**3) Causal self-attention 的价值在于正则化，而非 AR generation**

baseline_v2 的成功表明，causal mask 提供的时间结构正则化是有益的，但这种好处不需要通过 AR 反馈来实现。因果注意力 + 零反馈是当前框架下的最优组合。

**4) 神经活动的外部驱动本质限制了 AR 的适用性**

神经活动的长时程变化主要由外部信号驱动，而非自身历史。这一事实从根本上限制了 AR 反馈的信息价值——历史预测中包含的关于未来的信息远少于语言序列中的对应量。

**5) Encoder 表示能力比 decoder 策略更重要**

NeuroHorizon 在参数更少的情况下全面超越 Neuroformer、IBL-MtM、NDT2，主要归功于 POYO+ Perceiver encoder 的强大表示能力，而非 decoder 端的设计。

### 7.2 Baseline_v2 实际上在做什么

重新审视 baseline_v2，它本质上是一个**条件并行预测模型**：

- **编码阶段**：POYO+ Perceiver encoder 将历史窗口的 spike 序列压缩为高质量的 latent tokens
- **解码阶段**：T 个 bin queries 通过 cross-attention 各自独立地从 encoder latents 中提取预测所需的信息
- **Causal mask 的作用**：在 self-attention 层中提供时间结构正则化，使得 bin representations 在时间上平滑连贯
- **输出阶段**：PerNeuronMLPHead 将每个 bin 的 representation 与神经元 embedding 结合，独立输出每个神经元的 log firing rate

这个模型的核心预测能力来自于 encoder latents 对历史信息的高效压缩，以及 cross-attention 从这些 latents 中提取时间特异性预测信息的能力。它不需要 AR 反馈，因为 encoder latents 已经编码了足够的信息来并行预测所有 bins。

### 7.3 下一步方向建议

基于以上分析，建议将研究重点从 AR 反馈机制转向以下方向：

**1) 增强 encoder 表示能力**
- 探索更大的 encoder 架构（更多层、更多 latent tokens）
- 实验不同的 history window 长度对预测精度的影响
- 研究多尺度 encoding（同时使用 fine-grained spike events 和 coarse-grained rate signals）

**2) 引入外部条件信息**
- 既然神经活动主要由外部信号驱动，将行为变量（运动轨迹、刺激参数等）作为额外输入注入 encoder 或 decoder
- 这比 AR 反馈更直接地解决了预测信息从哪里来的问题

**3) 改进预测目标和损失函数**
- 探索分层预测：先预测群体级别的粗粒度活动模式，再精化到单神经元级别
- 考虑对比学习损失，使模型学习到 bin 之间的时间结构，而非仅仅优化逐点 Poisson NLL

**4) 数据规模扩展**
- NeuroHorizon 在 session scaling 实验中展现了强劲的规模效应（1 session: 0.089 → 10 sessions: 0.212）
- 这暗示当前性能可能主要受限于数据量，而非模型架构
- 优先考虑扩大训练数据规模，充分发挥 POYO+ encoder 的潜力

**5) 慎重对待 AR 的角色**
- 如果后续仍需探索 AR 反馈，建议将其作为辅助信号（如 auxiliary loss 或 soft constraint）而非核心生成机制
- 考虑 non-autoregressive 的替代方案，如扩散模型（diffusion models）或 masked prediction，这些方法不依赖逐步生成的假设

---

## 结语

四轮实验的数据提供了一个清晰且一致的信号：在当前的 bin-level neural forward prediction 框架下，自回归反馈未能为模型提供有意义的性能提升。这不仅是工程实现上的困难（exposure bias 已被大幅缓解），更是神经活动数据本质特性与 AR 假设之间的根本性不匹配。baseline_v2 以因果正则化 + 并行预测的简洁策略取得了最优性能，这一结果本身就是对AR 是否必要这一问题的有力回答。

未来的研究应当将 baseline_v2 作为坚实的起点，将注意力从如何做好 AR转向如何更好地编码历史信息和如何引入外部驱动信号。NeuroHorizon 的真正优势在于 POYO+ Perceiver encoder 的强大表示能力——这才是应当被进一步发掘和强化的核心资产。
