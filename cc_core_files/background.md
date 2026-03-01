# NeuroHorizon 研究背景与相关工作

> 本文档包含 NeuroHorizon 项目的研究背景（研究现状、研究意义、研究动机）、当前构建 Spike Foundation Model 的核心挑战，以及相关工作综述。
> 项目核心方案与创新点详见 `proposal.md`；技术执行参考详见 `proposal_review.md`。

---

## 1. 研究背景

### 1.1 当前研究现状

大规模神经记录技术（如Neuropixels探针、多电极阵列等）的快速发展使得同时记录数千个神经元成为可能，推动了对神经群体活动的深层理解。然而，如何有效建模这些高维、高时间分辨率的spiking数据，仍然是计算神经科学的核心挑战之一。

近年来，受自然语言处理（NLP）领域Transformer架构成功的启发，研究者开始将基础模型（Foundation Model）的理念引入神经数据建模。一系列"神经数据基础模型"相继涌现：Neural Data Transformer（NDT1/NDT2）系列通过masked modeling范式建模binned spike counts；NDT3进一步将规模扩展到多个数据集和脑区，采用自回归生成范式验证了scaling law在神经数据上的适用性；LFADS及其变体利用变分自编码器提取低维潜在动力学；Neuroformer首次提出自回归的spike-level预测框架并支持多模态条件输入；POYO/POYO+通过tokenization of individual spikes实现了高时间分辨率建模，并利用Perceiver架构实现了跨session的数据整合；SPINT通过IDEncoder机制实现了无需梯度更新的跨session泛化。

这些进展虽然令人瞩目，但当前模型仍然面临几个关键瓶颈：(1) 跨session泛化仍然困难，不同session中的神经元集合可能完全不同，传统方法需要对每个新session进行微调；(2) 长时程预测几乎未被系统性地探索，现有方法主要关注短时间窗口内的预测；(3) 多模态信息（行为数据、视觉刺激等）的融合预测及其对神经活动的贡献分析尚未充分实现。

### 1.2 研究意义

本研究旨在从三个互补的维度推进神经数据建模的前沿：

**跨Session鲁棒泛化**：我们提出一种新的无梯度更新（gradient-free）跨session泛化方法——通过从短暂参考窗口的原始神经活动中动态推断神经元身份表征（IDEncoder），使模型能够在完全未见过的session上直接工作，无需任何微调或梯度更新。这将大幅降低神经数据分析的实验成本，并为脑机接口（BCI）系统的长期稳定运行提供核心技术支撑。

**长时程自回归预测**：我们提出一个新的自回归神经脉冲活动预测模型框架。基于给定的历史神经活动（spike events），通过causal decoder在固定时间bins上逐步生成未来各神经元的spike count预测，系统性地探索模型在不同预测窗口长度下的性能边界。长时程神经活动预测能力将帮助研究者理解神经群体动力学的长程依赖性和时间演化规律，为实时脑机接口系统提供更稳定的解码基础。

**多模态可解释性**：我们设计灵活的多模态条件融合机制，同时引入行为数据和视觉刺激作为条件信息辅助神经活动预测。更重要的是，通过模态消融与归因分析，量化各模态在不同实验状态（脑区、刺激类型、行为阶段等）下对预测性能的贡献，从而为理解感觉-运动系统中的信息编码机制提供可量化的科学洞见。

### 1.3 研究动机

当前最先进的神经数据模型虽然在各自的任务上取得了优秀表现，但它们在设计上存在不同的侧重与局限：

- **POYO+**：擅长高时间分辨率的spike-level建模，通过Perceiver架构实现高效序列压缩，但缺乏长时程生成能力，主要用于行为解码任务
- **NDT3**：支持大规模数据的自回归生成训练，验证了scaling law在神经数据上的适用性，但其跨session泛化仍依赖per-session embedding，gradient-free泛化能力有限
- **Neuroformer**：支持自回归生成和多模态输入，但跨session能力较弱（依赖固定的per-session embedding table），且逐spike生成效率低
- **SPINT**：提出了优秀的gradient-free跨session泛化机制（IDEncoder），但主要应用于解码任务，缺乏自回归生成能力

这些模型各有优势但缺乏统一。我们的研究动机正是在于：能否设计一个统一的编码模型，同时继承POYO的高时间分辨率tokenization与高效压缩、SPINT的gradient-free跨session泛化能力、自回归生成框架的长时程预测能力，以及灵活的多模态条件融合与可解释性分析能力。

---

## 2. 构建 Spike Foundation Model 的核心挑战

当前构建面向spiking数据的基础模型，面临以下核心技术挑战：

**挑战一：跨Session的神经元身份漂移**。不同recording session之间，电极记录到的神经元集合可能完全不同（电极微移、信号漂移等）。传统方法将每个神经元视为固定的输入维度，导致模型与特定session强耦合。即使是同一被试的连续两天记录，神经元集合的重叠率也可能很低。这要求模型具备在不知道具体神经元身份的情况下，从群体活动模式中提取有意义表征的能力。

**挑战二：长时程时序依赖的捕获**。神经活动存在丰富的长程时间依赖性，但spike train本身是高度稀疏且随机的（泊松过程或近似泊松过程）。将预测窗口从传统的几十毫秒显著扩展时，信号的不确定性显著增加，预测难度大幅增长。如何在保持预测准确性的同时扩展预测时间范围，是核心技术挑战。

**挑战三：序列长度与计算效率**。以毫秒级分辨率处理秒级数据时，对于数百个神经元，原始spike sequence长度可达数万甚至数十万。标准Transformer的 $O(N^2)$ 复杂度使得直接处理如此长的序列在计算上不可行，需要高效的序列压缩机制（如Perceiver cross-attention）。

**挑战四：多模态异构信息的有效融合与归因**。行为数据（运动轨迹、速度等）是低维连续信号，视觉刺激（图像/视频）是高维结构化信号，而spiking数据是高维稀疏的点过程信号。这些异构模态的有效融合需要精心设计的架构，而量化各模态对神经活动预测的贡献更增加了一层分析复杂度。

---

## 3. 相关工作

### 3.1 神经数据基础模型

#### 3.1.1 Binned Spike Count 方法

**NDT系列（Neural Data Transformer）**：NDT1 (Ye & Pandarinath, 2021) 首次将Transformer引入神经数据建模，使用masked prediction在binned spike counts上预测神经活动。NDT2/STNDT进一步引入时空分离的attention机制，提升了计算效率。NDT3 (Ye et al., 2024) 将规模扩展到多个数据集和脑区，采用自回归生成范式建模binned spike counts，验证了scaling law在神经数据上的适用性。NDT3支持大规模自回归生成，但其跨session泛化仍依赖per-session embedding，缺乏gradient-free的泛化能力。

**LFADS系列**：LFADS (Pandarinath et al., 2018) 及其变体使用变分序列自编码器（VAE）从binned spike counts中推断低维潜在动力学。优点是提供了可解释的潜在因子；缺点是模型容量有限，难以扩展到大规模数据。

#### 3.1.2 Spike-Level 方法

**Neuroformer** (Antoniades et al., 2024)：首个将自回归Transformer用于spike-level预测的工作。输入为 (neuron_id, timestamp) 的spike event序列，使用GPT-style causal attention进行next-spike prediction，并支持通过cross-attention引入behavior和image条件信息（图像通过对比学习编码）。优点是spike-level建模保留了完整时间信息，支持多模态；缺点是跨session能力较弱，自回归生成效率低（每次仅生成一个spike event）。

**POYO/POYO+** (Azabou et al., 2024)：提出将individual spikes作为tokens，通过unit-specific embedding和RoPE时间编码实现高时间分辨率建模。关键创新是引入Perceiver架构的cross-attention对spike序列进行压缩，将 $O(N_{\text{spikes}})$ 的序列压缩为固定长度的latent array，极大降低了计算成本。POYO+进一步实现了跨session和跨brain area的联合训练。优点是高时间分辨率、计算效率高、支持大规模训练；缺点是主要用于解码任务（predict behavior from spikes），缺乏自回归生成能力。

#### 3.1.3 跨Session方法

**SPINT** (Le et al., NeurIPS 2025)：专门解决跨session泛化问题。核心创新是IDEncoder模块——通过将每个session中观察到的神经元原始活动数据（binned spike counts）编码为unit embedding，使模型能够在新session上无需梯度更新即可工作。IDEncoder采用MLP1 -> mean pool -> MLP2的feedforward架构，从calibration trials的原始spike count序列（非手工统计特征）中端到端学习神经元identity表示。这种gradient-free的泛化机制是解决neural identity drift问题的优雅方案。

**Multi-task Masking (MtM)** (Ye et al., 2024)：通过随机mask输入和输出的神经元子集来实现跨session鲁棒性，是一种数据增强式的解决方案。

#### 3.1.4 其他相关工作

**LDNS** (Hurwitz et al., 2024)：将latent diffusion model应用于神经数据，通过去噪过程生成neural population dynamics。

**PopT** (Ye et al., 2024)：基于population-level tokenization的方法。

**POSSM** (Azabou et al., 2025)：POYO的后续工作，将SSM（State Space Model）引入spike sequence建模，用Mamba替代Transformer以处理更长的序列。

### 3.2 现有方法的对比与研究空白

| 特性 | NDT3 | Neuroformer | POYO+ | SPINT | POSSM | **Ours** |
|------|------|-------------|-------|-------|-------|----------|
| 时间分辨率 | Binned | Spike-level | Spike-level | Binned | Spike-level | Spike-level |
| 自回归生成 | ✓ | ✓ | ✗ | ✗ | ✗ | ✓ |
| 跨Session (gradient-free) | ✗ | ✗ | 有限 | ✓ | 有限 | ✓ |
| 长时程预测 | 有限 | 有限 | ✗ | ✗ | ✗ | ✓ |
| 多模态条件 | ✗ | ✓ | ✗ | ✗ | ✗ | ✓ |
| 多模态归因分析 | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| 计算效率 | 中 | 低 | 高 | 中 | 高 | 高 |

**研究空白总结**：目前没有一个模型能够同时实现 (1) spike-level的高时间分辨率输入、(2) gradient-free的跨session泛化、(3) 长时程的自回归预测生成、(4) 灵活的多模态条件融合与可解释性归因分析。NeuroHorizon旨在填补这一空白。

---

## 参考文献

> 完整参考文献列表详见 `proposal.md`。
