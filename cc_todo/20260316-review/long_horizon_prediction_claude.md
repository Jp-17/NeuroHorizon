# 长时程神经活动预测：问题意义、方法论探索与项目方向评判

> 撰写日期：2026-03-16
> 项目：NeuroHorizon
> 背景：基于 POYO+ encoder + autoregressive bin-level decoder 的神经活动前向预测

---

## 摘要

本文系统性地探讨三个核心问题：（1）长时程神经活动预测（>250ms）是否是一个有意义的研究问题；（2）哪些方法论路径有潜力突破当前性能瓶颈；（3）NeuroHorizon 项目应如何调整方向以最大化研究贡献。文档基于当前项目的实验结果（baseline_v2 在 250ms/500ms/1000ms 窗口的 fp-bps 分别为 0.2115/0.1744/0.1317）、四轮 AR 反馈实验的失败经验，以及对相关文献的综合分析，提出 Latent Dynamics Model 作为最具前景的技术方向，并给出具体的实现路线图和风险评估。

---

## 第一部分：问题意义——为什么长时程预测值得研究

### 1.1 脑机接口（BCI）应用层面的需求

长时程神经活动预测并非纯粹的学术问题，其在多个 BCI 应用场景中具有直接价值：

**闭环神经调控（Closed-loop Neuromodulation）**。深部脑刺激（DBS）等神经调控技术正在从"开环恒定刺激"向"闭环自适应刺激"演进。闭环系统需要预测 500ms-1s 之后的神经状态，以便提前调整刺激参数。考虑到刺激参数计算、硬件响应和神经生理学延迟，系统需要至少 200-500ms 的前瞻能力才能实现真正的"预测性干预"而非"反应性调整"。目前大多数闭环 DBS 系统仅使用实时状态检测（如 beta 功率阈值），而非前向预测，这严重限制了干预的时效性。

**癫痫发作预测（Seizure Prediction）**。癫痫发作预测是神经工程领域的"圣杯问题"之一。临床上有意义的预测需要提前数秒甚至数十秒发出警告，以便患者或自动系统采取预防措施。这要求模型能够捕捉神经活动从正常状态向发作状态过渡的缓慢漂移过程。当前基于特征工程的方法（如频谱特征、同步指数）难以捕捉复杂的时序依赖关系，而基于深度学习的端到端预测方法在长时程上的表现尚未被系统研究。

**运动假体控制（Motor Prosthetics）**。现代运动 BCI（如 BrainGate）通过解码运动皮层神经活动来控制外部设备。系统存在固有的处理延迟（通常 50-100ms），如果能够预测未来 100-500ms 的神经状态，就可以补偿这种延迟，实现更流畅、更自然的控制。特别是对于需要快速反应的任务（如抓取运动物体），前向预测能力至关重要。

**感觉反馈优化（Sensory Feedback）**。双向 BCI 不仅解码运动意图，还需要向感觉皮层注入人工反馈信号。预测感觉皮层对刺激的响应模式，可以优化刺激策略，减少不必要的试错过程。这需要模型能够预测神经群体在接收到特定输入后数百毫秒内的活动演化轨迹。

### 1.2 科学理解层面的价值

**可预测性地平线（Predictability Horizon）作为神经回路的内在属性**。一个神经群体的活动能被准确预测多远，本身反映了该回路的动力学结构。混沌系统的 Lyapunov 指数决定了其可预测性的理论上限；类似地，神经回路的"有效 Lyapunov 指数"决定了预测准确性随时间衰减的速率。从我们的数据来看，fp-bps 从 250ms 到 500ms 下降 17.5%，从 500ms 到 1000ms 下降 24.5%——这种加速衰减模式暗示了运动皮层 M1 在 reaching task 中的内在动力学特性。

**跨脑区、跨任务的可预测性差异**。如果我们能在多个脑区和多种任务上系统性地度量"预测准确性 vs. 预测窗口"曲线，这些曲线的形状差异将揭示重要的计算原理。例如：前额叶皮层（参与高级计划）是否比运动皮层（执行快速运动指令）具有更长的可预测性地平线？延迟期（delay period）的神经活动是否比运动执行期更可预测？这些问题直接关联到"不同脑区如何在不同时间尺度上处理信息"这一核心神经科学问题。

**理论天花板（Theoretical Ceiling）**。长时程预测准确性的理论上限本身就是一个深刻的科学问题。考虑到神经活动的随机性（Poisson-like variability）、未被记录的输入信号（来自其他脑区或感觉输入）、以及系统的混沌特性，在给定记录条件下，最优预测器能达到多高的准确性？定量回答这个问题需要信息论工具（如互信息、传递熵）和动力系统理论的结合。这一理论分析本身就可以构成独立的学术贡献。

### 1.3 当前文献的空白

**缺乏系统性的长时程研究**。当前神经群体活动建模的主流工作——NDT (Ye & Pandarinath, 2021)、NDT2 (Ye et al., 2024)、POYO/POYO+ (Azabou et al., 2023, 2024)、LFADS (Pandarinath et al., 2018) ——几乎全部聚焦于 next-bin prediction 或短窗口（<200ms）的预测/平滑/插补任务。这些模型虽然架构强大，但其评估协议并未涉及"预测准确性如何随窗口长度衰减"这一关键问题。

**Neuroformer 的尝试与局限**。Neuroformer (Antoniades et al., 2024) 是少数涉及长时程预测的工作之一，但其评估仅在几个固定窗口上报告指标，并未系统性地分析衰减曲线，也未探究导致衰减的根本原因（是模型能力不足，还是数据固有的不可预测性？）。

**缺乏公平的方法论比较**。在长时程预测这一特定设定下，不同建模范式（autoregressive vs. parallel vs. latent dynamics vs. generative）之间缺乏公平的对比实验。每个方法的论文使用不同的数据集、不同的评估指标、不同的预测窗口，使得我们无法判断哪种方法论最适合这一问题。

**缺乏对衰减现象的解释**。"预测准确性随时间单调递减"这一现象虽然直觉上显而易见，但缺乏定量的理论分析。衰减的速率取决于什么？是模型架构的限制，还是数据的信噪比，还是任务动力学的内在复杂度？对这些问题的回答将把一个经验性观察提升为具有理论深度的科学发现。

### 1.4 NeurIPS 适配性分析

长时程神经活动预测作为一个 NeurIPS 投稿主题具有良好的适配性，原因如下：

**Well-defined ML Problem**。前向预测是一个标准的序列建模问题，具有清晰的输入输出定义、明确的损失函数（Poisson NLL）和定量评估指标（fp-bps）。这使得 ML 领域的审稿人可以轻松理解问题设定。

**涉及多个 ML 核心议题**。长时程预测天然涉及：(a) sequence modeling 的长程依赖建模；(b) exposure bias 和 distribution shift（AR 模型的核心挑战）；(c) uncertainty quantification（长时程预测的置信度估计）；(d) structured prediction（神经群体的协同活动约束）。这些都是 NeurIPS 社区高度关注的技术方向。

**NeurIPS 对 neuro + ML 交叉的开放度**。近年来 NeurIPS 接收了越来越多 neuroscience + ML 交叉工作：POYO/POYO+ (NeurIPS 2023/2024)、NDT/NDT2、LFADS 的后续工作等。NeurIPS 2025/2026 还新增了 Neuro-AI track。长时程预测作为一个具有明确 ML 挑战的神经科学问题，完全符合这一趋势。

**系统性 benchmark 的贡献**。即使方法上的创新有限，一个系统性的、多方法论的长时程预测 benchmark 本身也具有显著的社区贡献价值——类似于 NLB (Neural Latents Benchmark, Pei et al. 2021) 对 neural population modeling 领域的贡献。

---

## 第二部分：方法论探索

### 方向 A：Latent Dynamics Model（最推荐）

#### 技术原理

核心架构如下：

```
History spikes (0~T_obs)
    -> POYO+ Encoder (Perceiver)
    -> Latent state z_0 in R^{L x D}  (L个latent tokens, D维)
    -> Dynamics Model: z_0 -> z_1, z_2, ..., z_T
    -> PerNeuronMLPHead: z_t -> predicted spike counts (N neurons)
```

这一设计的理论基础是神经群体活动的低维流形假说（low-dimensional manifold hypothesis）。大量实验证据表明，即使同时记录数百个神经元，其群体活动的大部分方差可以被 10-30 个 latent dimensions 解释 (Cunningham & Yu, 2014; Gallego et al., 2017)。LFADS (Pandarinath et al., 2018, Nature Methods) 是这一思路最成功的实现：使用序列变分自编码器（SVAE）推断 latent initial condition，然后通过 GRU 在 latent space 中前向模拟动力学。

#### 具体架构选择

**Neural ODE (Chen et al., 2018)**。将 latent dynamics 定义为连续时间 ODE：dz/dt = f_theta(z, t)，其中 f_theta 是一个神经网络。优势：天然支持任意时间分辨率的预测，可以做 continuous-time 推理；理论上优雅，与动力系统理论直接对接。劣势：ODE solver 的计算开销较大，尤其是 adjoint method 的反向传播内存效率虽高但速度慢；训练不稳定性。实现复杂度：中等，可使用 torchdiffeq 库。

**Neural SDE (Li et al., 2020; Kidger et al., 2021)**。在 ODE 基础上添加随机项：dz = f_theta(z, t)dt + g_theta(z, t)dW。优势：能够显式建模神经活动中的随机性，生成多个轨迹样本以量化不确定性；更符合神经系统的随机特性。劣势：训练更加复杂（需要 SDE-specific loss 或 KL 散度项），推理时需要多次 sampling。实现复杂度：高，但 torchsde 库提供了基本接口。

**Linear Recurrent Model / S4 (Gu et al., 2022)**。使用结构化状态空间模型：z_{t+1} = Az_t + Bx_t, y_t = Cz_t + Dx_t。优势：计算效率极高（可以并行化训练），长程依赖建模能力强（S4 在长序列任务上表现优异）；训练稳定。劣势：linear dynamics 的表达能力有限，可能无法捕捉非线性动力学特征。实现复杂度：低，有成熟的实现。

**推荐选择**：初始实验使用 **Linear RNN / S4** 作为 dynamics model，因为其训练稳定性高、实现简单、计算开销低。如果 linear dynamics 的表达能力不足，再升级到 Neural ODE。实际上，考虑到 M1 在 center-out reaching task 中的 dynamics 被广泛认为是近似线性的（rotational dynamics, Churchland et al. 2012），linear model 很可能就足够了。

#### 与当前项目的结合方式

**Perceiver latent tokens 的处理**。当前 POYO+ encoder 输出 L 个 latent tokens（通常 L=32 或 64），每个 token 是 D 维向量。有两种策略处理这些 tokens：

1. **Token-level dynamics**：对每个 latent token 独立运行 dynamics model。这保留了 Perceiver 学到的 token 结构，但假设 tokens 之间独立演化，可能丢失信息。
2. **Pooling + dynamics**：将 L 个 tokens 通过 attention pooling 或 mean pooling 压缩为一个 D 维向量（或少数几个向量），然后在这个低维空间上运行 dynamics。这更符合"低维流形"的直觉，也降低了 dynamics model 的输入维度。

推荐从方案 2 开始，使用 attention pooling（带 learned query）将 L 个 tokens 压缩为 K 个（K=4~8），然后对这 K x D 维的 concatenated vector 运行 dynamics model。

**与 PerNeuronMLPHead 的兼容性**。当前 PerNeuronMLPHead 接收 bin representation（D 维）和 unit embedding，通过 shared MLP 输出 log_rate。在 latent dynamics 框架下，dynamics model 在每个预测 bin t 输出的 z_t 可以直接替代原来的 bin representation 输入到 PerNeuronMLPHead 中。这意味着 PerNeuronMLPHead 可以完全复用，只需要修改其输入来源。

**训练策略**。推荐分两阶段训练：
- Phase 1：冻结 POYO+ encoder 和 PerNeuronMLPHead，只训练 dynamics model。这利用了预训练 encoder 的强大表征能力，降低了训练复杂度。
- Phase 2：端到端微调所有组件。

#### 预期收益

Latent dynamics model 的核心优势在于：在低维空间中预测动力学，避免了高维 count vector 的 AR 反馈中固有的误差累积问题。理论上，如果 latent space 确实捕捉了神经群体活动的主要变异来源，那么在 latent space 中的预测误差应该比在 observation space 中的误差更小且更稳定。

具体来说，预期长时程预测（500ms-1000ms）的 fp-bps 有显著提升空间。当前 baseline_v2 在 1000ms 的 0.1317 fp-bps 可能部分归因于 parallel decoder 无法有效捕捉跨 bin 的动力学连续性——每个 bin 的预测虽然通过 causal attention 看到了前面的 bins，但没有显式的动力学 inductive bias。Latent dynamics model 引入了"神经活动在低维流形上平滑演化"的强 prior，这对长时程预测应该是有利的。

#### NeurIPS 叙事强度

**强叙事**："我们提出了一个将 foundation model（POYO+）的强大编码能力与 latent dynamics modeling 的物理先验结合的框架。POYO+ 提供 session-agnostic 的 latent state inference，dynamics model 在这个 latent space 中做 forward prediction。这是第一个能够实现跨 session zero-shot 长时程神经活动预测的系统。"

与 LFADS 的关键区别为：LFADS 是 per-session 的 SVAE，需要对每个新 session 重新训练；而本方案利用 POYO+ 的跨 session 泛化能力，实现"一次训练，任意 session 推理"。这一差异化在 NeurIPS 社区中具有很强的吸引力。

#### 风险分析

- **主要风险**：POYO+ encoder 学到的 latent tokens 可能不适合做 dynamics propagation——Perceiver 的 latent tokens 是为了 reconstruction 而优化的，不一定捕捉了 dynamics-relevant 的信息。
- **缓解措施**：端到端训练（Phase 2）可以让 encoder 调整其 latent representation 以适应 dynamics prediction 的需求。
- **次要风险**：M1 在 center-out reaching task 中的 dynamics 过于简单（近似线性旋转），使得 latent dynamics model 的优势不明显。
- **缓解措施**：如果在当前数据上效果有限，可以在更复杂的任务（如 random target pursuit）或更复杂的脑区（如前额叶）上验证。

#### 预估开发时间

- 基础版本（Linear RNN dynamics + frozen encoder）：1-2 周
- 完整版本（Neural ODE + 端到端训练 + 系统性对比实验）：3-4 周

### 方向 B：继续优化 AR 反馈

#### 可尝试的技术

**Scheduled Sampling (Bengio et al., 2015)**。在训练过程中，以逐渐增加的概率 epsilon 用模型自身的预测替代 ground truth 作为下一步的输入。训练初期 epsilon 接近 0（纯 teacher-forcing），训练后期 epsilon 趋向 1（纯 model rollout）。这一方法在 NLP 和 speech synthesis 中被广泛使用，理论上可以完全消除 exposure bias。

**改进 Feedback Encoding**。当前 AR 反馈机制（如果采用）将前一个 bin 的预测压缩为一个 vector。这丢失了 per-neuron 的信息，导致反馈信号过于粗糙。改进方案：保留 per-neuron 的预测 count，通过一个轻量级的 embedding 层将其编码为与 encoder output 同维度的 representation。

**Curriculum Learning on Prediction Length**。先用短窗口（如 100ms, 5 bins）训练至收敛，然后逐步增加窗口长度（200ms, 500ms, 1000ms）。每次增加长度时，只微调新增部分的参数。这种"由近及远"的策略可以避免模型在训练早期就面对长程预测的困难。

**Quantized Count Embedding**。将连续的 spike count 预测离散化为有限的类别（如 0, 1, 2, 3+），使用 learned embedding 表示每个类别。这减少了 continuous prediction 到 feedback 过程中的 distribution shift，但牺牲了精度。

**Consistency Regularization**。在训练时同时运行 teacher-forced 和 free-running 两条路径，添加正则化项鼓励两条路径的 hidden states 保持一致。这可以被看作 scheduled sampling 的一个 soft version。

#### 诚实评估

四轮 AR 实验的结果给了我们重要的信号。虽然 alignment training 在缩小 teacher-forced 和 free-running 之间的 gap，但收敛速度在减缓，且最终性能未能超越 baseline_v2（无 AR 反馈的 parallel prediction）。这一结果暗示了一个根本性问题：

**对于 neural data，AR feedback 的 incremental information 可能本来就很少。** 考虑一下：在 NLP 中，AR 至关重要，因为下一个 word 强烈依赖于前一个 word（语法、语义约束）。但在 neural spiking data 中，相邻 20ms bins 之间的统计依赖可能主要由底层的慢动力学（latent dynamics）驱动，而非 bin-to-bin 的直接因果关系。如果一个 bin 的 spike count 对下一个 bin 的预测只提供微弱的信息增益（相对于 encoder 已经捕捉的信息），那么无论 AR 训练多完美，其收益都将有限。

这一分析得到了理论支持：如果 spikes 是从一个平滑变化的 firing rate 中 Poisson-sampled 的，那么观测到的 spike count（一个 noisy sample）对下一个 bin 的 firing rate（一个缓慢变化的函数值）提供的信息量远小于 history 中所有 spikes 的聚合信息。换言之，encoder 已经从完整的 history 中提取了关于 latent dynamics 的大部分信息，AR feedback 能追加的边际信息微乎其微。

**投入产出比评估**：在所有方向中最低。即使 scheduled sampling 等技术能带来改善，预期改善幅度也有限（可能 1-3% fp-bps），且无法构成强有力的 NeurIPS 叙事。"我们优化了 AR training 使其略好于 parallel prediction"这一结论不够有趣。

#### NeurIPS 叙事强度

弱。"更好的 AR 训练策略"缺乏方法论创新性，更接近于 engineering optimization。除非能给出"为什么 AR 对 neural data 不 work"的深入理论分析，否则难以独立支撑一篇 NeurIPS 论文。但如果将其作为"方法论对比"的一部分（证明 latent dynamics > AR > parallel），则有补充价值。

#### 预估开发时间

- Scheduled sampling 实现与实验：1 周
- 全部改进方案：2-3 周

### 方向 C：Diffusion / Flow Matching 生成模型

#### 技术原理

将整个预测窗口的 spike count matrix X（T 个 time bins x N 个 neurons）视为一个需要"生成"的对象，训练一个条件生成模型：

```
条件：POYO+ encoder output (history representation)
输入：Gaussian noise epsilon (T x N)
输出：predicted spike count matrix X_hat (T x N)
```

训练时对 clean data 逐步添加噪声，模型学习逆转这一过程（denoising）。推理时从纯噪声出发，通过迭代 denoising 生成预测。

#### 技术细节

**Discrete Diffusion (D3PM, Austin et al. 2021)**。由于 spike counts 是非负整数，discrete diffusion 可能更自然。D3PM 定义了离散状态之间的 Markov transition matrix 来实现 forward corruption，模型学习 reverse transition。优势：直接在 count space 操作，不需要连续化近似。劣势：状态空间大小需要截断（如 max count = 10），实现复杂。

**Continuous Diffusion on Transformed Counts**。对 log(1 + count) 做标准 continuous diffusion（DDPM, Song et al. 2021）。这是一个更简单的实现路径，但引入了连续化的近似误差，且 diffusion 的 Gaussian noise 假设与 count data 的离散性不完全匹配。

**Flow Matching (Lipman et al., 2023; Liu et al., 2023)**。相比 diffusion models 的 score matching，flow matching 直接学习从噪声到数据的确定性映射（ODE flow），训练更高效、更稳定。可以与 Optimal Transport (OT) 结合，进一步提高生成质量。近年来在图像、蛋白质结构生成等领域表现突出。

#### 与当前项目的结合方式

条件信号来自 POYO+ encoder 的输出。具体来说，可以通过 cross-attention 将 encoder latent tokens 注入到 denoising network 中。Denoising network 的骨干可以使用 1D U-Net（沿时间维度）或 Transformer。

#### 预期收益

- **完全消除 exposure bias**：训练和推理的输入分布完全一致（都是噪声或部分去噪的样本），不存在 teacher-forcing vs. free-running 的不匹配问题。
- **不确定性量化**：通过生成多个样本，天然获得预测分布而非点估计。这对科学分析尤为重要——可以区分"模型不确定"和"数据固有随机性"。
- **NeurIPS 热度**：Diffusion models 和 flow matching 是 2023-2026 年 NeurIPS 的热门话题，将其应用于 neural data prediction 具有话题吸引力。

#### 劣势与风险

- **推理速度**：标准 diffusion 需要 50-1000 步 denoising，即使使用 DDIM 或 DPM-Solver 也需要 10-50 步。这在 real-time BCI 场景中可能不可接受。
- **Count data 的适配性**：Spike counts 的低维、稀疏、离散特性与图像/文本等 diffusion models 的传统应用场景差异较大。模型是否能有效学习 count data 的分布尚不确定。
- **架构复杂度高**：denoising network + noise scheduling + sampling strategy 引入了大量超参数，调优成本高。
- **过度工程化风险**：对于一个可能用简单方法（如 latent dynamics）就能解决的问题，使用 diffusion models 可能是"杀鸡用牛刀"。

#### NeurIPS 叙事强度

中等偏强。"首次将 diffusion/flow matching 应用于长时程神经活动预测"是一个新颖的切入点，但需要令人信服的实验证据表明 diffusion 的 uncertainty quantification 能力确实带来了独特价值。如果仅仅是性能上与其他方法持平或略优，说服力不足。

#### 预估开发时间

- 基础版本（Continuous diffusion + 简单 U-Net）：2-3 周
- 完整版本（Discrete diffusion + flow matching + 系统对比）：4-6 周

### 方向 D：Conditional Generation with Behavioral Covariates

#### 技术原理

在预测窗口内，除了 history neural activity 之外，还提供 behavioral signals（运动轨迹、速度、任务标记等）作为额外的条件输入：

P(future_spikes | history_spikes, future_behavior)

行为信号（如手臂运动轨迹）通常变化缓慢（低频 dynamics），可以作为"锚点"来约束长时程预测。在 reaching task 中，如果模型知道未来 1 秒的手臂轨迹，那么预测 M1 活动就变得更加受限，因为 M1 活动与运动高度相关。

#### 与 NeuroHorizon 现有设计的兼容性

项目的 Innovation 3 已经规划了 multimodal conditioning（DINOv2 visual stimuli + behavioral data），但目前尚未实现。Perich-Miller 数据集包含详细的运动学数据（手臂位置、速度），可以直接使用。

实现方式：将 future behavior 通过一个 lightweight encoder（如 MLP 或小型 Transformer）编码为 behavior tokens，然后在 decoder 中通过 cross-attention 注入。这与当前 decoder 的 cross-attention 机制（attend to encoder output）完全兼容，只需增加一个额外的 cross-attention 层。

#### 限制与讨论

**BCI 场景的局限**。在 real-time BCI 应用中，future behavior 通常不可用（因为 behavior 就是我们要解码预测的目标）。因此，behavior-conditioned prediction 主要适用于离线科学分析场景。

**但这引出一个有趣的科学问题**。对比 behavior-conditioned 和 behavior-free 两种预测的准确性差距，可以量化"behavioral covariates 能解释多少神经活动的可预测性"。如果加入 future behavior 后 fp-bps 大幅提升，说明长时程预测的主要瓶颈是缺少"来自行为层面的 top-down 信号"；如果提升有限，说明瓶颈在于神经活动的内在随机性。

#### NeurIPS 叙事强度

中等。这个方向更像是"完善现有系统"而非"提出新方法"。但如果能给出上述科学分析（behavior-conditioned vs. free 的对比），可以构成一个有趣的分析贡献。

#### 预估开发时间

- 基础版本（behavioral conditioning）：1 周
- 科学分析完整版本：2 周

---

## 第三部分：各方向对比总结

| 评估维度 | A. Latent Dynamics | B. AR 优化 | C. Diffusion/Flow | D. Behavioral Conditioning |
|---|---|---|---|---|
| **技术创新性** | 5/5 | 2/5 | 4/5 | 3/5 |
| **实现难度** | 中等 | 低 | 高 | 低 |
| **预期性能提升** | 高（尤其长窗口） | 低 | 中-高 | 中（仅限有behavior场景） |
| **与现有代码兼容性** | 高（复用encoder+head） | 最高（修改训练策略即可） | 中（需新增denoising网络） | 高（增加cross-attention层） |
| **NeurIPS 叙事强度** | 5/5 | 2/5 | 4/5 | 3/5 |
| **风险等级** | 中 | 低（但回报也低） | 高 | 低 |
| **预估开发时间** | 3-4 周 | 2-3 周 | 4-6 周 | 1-2 周 |
| **独立成文能力** | 可独立支撑论文 | 不能独立支撑 | 可独立支撑论文 | 作为分析章节 |

**总体排序**（综合考虑创新性、可行性、投入产出比）：

**A >> C > D > B**

方向 A（Latent Dynamics）在几乎所有维度上都是最优选择：创新性最高、与现有架构兼容性好、NeurIPS 叙事最强、实现难度适中。方向 C（Diffusion）虽然话题热度高，但实现复杂度和风险也显著更高，且对 neural count data 的适配性存疑。方向 D 可以作为补充分析。方向 B 的投入产出比最低，但可以作为 ablation study 的一部分。

---

## 第四部分：对当前项目的综合评判

### 4.1 当前项目做对了什么

**POYO+ encoder 的选择是正确的**。POYO+ 是当前最先进的 neural population encoder 之一，其 Perceiver 架构的 event-level spike tokenization 能够处理任意数量的神经元和不规则采样，cross-session 泛化能力强。这一选择为后续任何方法论扩展奠定了坚实基础。

**系统性 benchmark 具有独立贡献价值**。在 250ms/500ms/1000ms 三个窗口上系统性地对比 NeuroHorizon、Neuroformer、IBL-MtM 和 NDT2，这本身就是一个有价值的贡献。目前文献中缺乏这样的多方法论长时程预测对比。

**PerNeuronMLPHead 的 T-token 设计是优雅的**。将 decoder 的 T 个 bin tokens 通过 shared MLP + unit embedding 扩展为 T x N 的预测矩阵，这种设计在保持参数效率的同时实现了 per-neuron 的精细预测。这个设计在 latent dynamics 框架中可以被完整复用。

**四轮 AR 实验的"失败"是宝贵的经验证据**。虽然 AR 反馈未能超越 baseline_v2，但这四轮实验系统性地排除了"简单 AR 训练优化就能解决长时程预测"的假设。更重要的是，这些负面结果暗示了一个重要的 insight：对于 neural spiking data，observation-level AR 反馈的边际信息增益可能远小于在 NLP/speech 中的情形。如果在论文中恰当地呈现，这一 negative result 本身就是一个有价值的 finding。

### 4.2 当前项目的核心问题

**AR 叙事与实验证据不匹配**。如果项目的核心 narrative 是"autoregressive prediction for long-horizon neural forecasting"，但实验证据表明 AR 并不比 parallel prediction 更好，那么这一叙事就面临根本性的挑战。审稿人会问："你的方法的核心创新是 AR，但实验表明 AR 没有帮助，那你的贡献是什么？"这需要通过转变叙事方向来解决——从"AR is the solution"转变为"what is the right approach for long-horizon neural prediction?"

**数据集单一性**。仅使用 Perich-Miller 2018 一个数据集（M1, center-out reaching, 3 macaques）限制了结论的泛化性。M1 在 center-out reaching 中的 dynamics 相对简单（近似线性旋转），其他脑区（如前额叶、海马）和更复杂的任务可能表现出截然不同的可预测性特征。NeurIPS 审稿人很可能要求至少两个数据集的验证。

**项目完成度有限**。Innovation 3（multimodal conditioning）和 IDEncoder（session-specific adaptation）均未实现。如果在投稿时这些组件仍然缺失，项目的完整度会受到质疑。需要在投稿前明确：哪些是核心贡献（必须实现），哪些是未来工作（可以在论文中讨论但不实现）。

### 4.3 推荐的最佳路径

综合考虑研究价值、实现可行性和（假设的）投稿时间压力，推荐以下分阶段路径：

#### Phase 1（1-2 周）：Latent Dynamics Model 基础版本

1. 实现 attention pooling 将 Perceiver latent tokens 压缩为低维 dynamics state
2. 实现 Linear RNN dynamics model（先冻结 encoder 和 head）
3. 在 250ms/500ms/1000ms 三个窗口上评估，与 baseline_v2 对比
4. 如果 Linear RNN 有效，实验 Neural ODE 作为升级

#### Phase 2（1 周）：端到端训练 + Ablation Study

1. 解冻 encoder，端到端训练 latent dynamics 版本
2. 进行系统性 ablation：dynamics model 类型（Linear vs. ODE vs. GRU）、latent dimension、pooling 策略
3. 将 AR 实验结果整合为"方法论对比"（parallel vs. AR vs. latent dynamics）

#### Phase 3（1 周）：科学分析 + 论文叙事

1. 分析"预测准确性 vs. 窗口长度"的衰减曲线，给出动力系统理论的解释
2. 如果可能，添加 behavioral conditioning 实验（方向 D）作为补充分析
3. 构建论文叙事："Long-horizon neural prediction requires latent dynamics, not autoregressive feedback"

#### 论文核心叙事（推荐框架）

**标题方向**：Latent Dynamics for Long-Horizon Neural Population Prediction: Why Autoregressive Feedback Falls Short

**核心论点**：

1. 长时程神经活动预测是一个重要但未被系统研究的问题（gap identification）
2. 朴素的 AR 反馈在 neural data 上不 work，因为 observation-level 的 sequential dependency 弱于 NLP/speech（empirical + theoretical analysis）
3. 正确的做法是在 latent dynamics space 中做 forward prediction，利用神经群体活动的低维流形结构（proposed method）
4. POYO+ encoder + latent dynamics = session-agnostic neural dynamics prediction（unique selling point）
5. 系统性 benchmark 在多个预测窗口上验证了上述论点（experiments）

这一叙事框架的优势在于：(a) AR 的"失败"不再是负面结果，而是论文论点的重要组成部分；(b) latent dynamics 的引入有清晰的动机（不是拍脑袋想的，而是 AR 失败后的 principled alternative）；(c) 与 LFADS 的差异化清晰（session-agnostic）；(d) benchmark 贡献独立于方法贡献。

---

## 参考文献

- Austin, J., et al. (2021). Structured Denoising Diffusion Models in Discrete State-Spaces (D3PM). NeurIPS.
- Azabou, M., et al. (2023). A Unified, Scalable Framework for Neural Population Decoding (POYO). NeurIPS.
- Azabou, M., et al. (2024). POYO+: A Multi-task, Multi-dataset Neural Foundation Model. NeurIPS.
- Bengio, S., et al. (2015). Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks. NeurIPS.
- Chen, R.T.Q., et al. (2018). Neural Ordinary Differential Equations. NeurIPS.
- Churchland, M.M., et al. (2012). Neural population dynamics during reaching. Nature.
- Cunningham, J.P. & Yu, B.M. (2014). Dimensionality reduction for large-scale neural recordings. Nature Neuroscience.
- Gallego, J.A., et al. (2017). Neural Manifolds for the Control of Movement. Neuron.
- Gu, A., et al. (2022). Efficiently Modeling Long Sequences with Structured State Spaces (S4). ICLR.
- Kidger, P., et al. (2021). Neural SDEs as Infinite-Dimensional GANs. ICML.
- Li, X., et al. (2020). Scalable Gradients for Stochastic Differential Equations. AISTATS.
- Lipman, Y., et al. (2023). Flow Matching for Generative Modeling. ICLR.
- Liu, X., et al. (2023). Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow. ICLR.
- Pandarinath, C., et al. (2018). Inferring single-trial neural population dynamics using sequential auto-encoders (LFADS). Nature Methods.
- Pei, F., et al. (2021). Neural Latents Benchmark '21. NeurIPS Datasets and Benchmarks Track.
- Perich, M.G. & Miller, L.E. (2018). Motor cortical dynamics during naturalistic reaching and grasping. Zenodo Dataset.
- Song, Y., et al. (2021). Score-Based Generative Modeling through Stochastic Differential Equations. ICLR.
- Ye, J. & Pandarinath, C. (2021). Representation learning for neural population activity with Neural Data Transformers (NDT). NeurIPS.
- Ye, J., et al. (2024). Neural Data Transformer 2 (NDT2): Multi-context Pretraining for Neural Spiking Activity. NeurIPS.
- Antoniades, A., et al. (2024). Neuroformer: Multimodal and Multitask Generative Pretraining for Brain Data. ICLR.

---

> 本文档由 Claude 生成，作为 NeuroHorizon 项目方向评审的讨论材料。
> 生成日期：2026-03-16
