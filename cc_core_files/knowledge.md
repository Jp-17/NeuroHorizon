# NeuroHorizon 知识库（Knowledge Base）

> 本文档收集项目相关的核心概念、技术讨论与设计决策的深度分析。
> 每个条目力求解释充分、逻辑完整，可作为项目技术方案的理论支撑。

---

## 目录

1. [Poisson NLL 与神经活动预测的 Loss 选择](#1-poisson-nll-与神经活动预测的-loss-选择)
2. [Spike 稀疏性与 Loss 统计策略](#2-spike-稀疏性与-loss-统计策略)
3. [Scheduled Sampling：概念、用途与引入时机](#3-scheduled-sampling概念用途与引入时机)

---

## 1. Poisson NLL 与神经活动预测的 Loss 选择

### 1.1 为什么用 Poisson NLL 预测 Binned Firing Rate

#### 神经元发放的统计基础

神经元的动作电位（spike）是离散的二值事件——在某个极短时间点，神经元要么发放、要么不发放。当我们将时间轴划分为固定宽度的 bin（如 20ms），统计每个 bin 内的 spike 数量（spike count），这个计数过程天然符合 **Poisson 过程**的假设：

- 每个 spike 事件独立（在足够短的时间尺度上）
- spike 发生的概率在同一 bin 内近似恒定
- bin 内 spike 数为非负整数

Poisson 分布的概率质量函数为：

$$P(k | \lambda) = \frac{\lambda^k e^{-\lambda}}{k!}$$

其中 $\lambda$ 是 bin 内的期望 spike count（即 firing rate x bin width），$k$ 是实际观测到的 spike count。

#### Poisson 分布的 Mean-Variance 关系

Poisson 分布有一个独特性质：**均值等于方差**（$E[X] = Var[X] = \lambda$）。在神经科学中，这个性质通过 **Fano Factor**（方差/均值之比）来检验：

- Fano Factor 约 1：符合 Poisson 假设（很多皮层神经元的 spike count 在 trial 间表现出接近 1 的 Fano Factor）
- Fano Factor < 1：sub-Poisson，发放比预期更规则
- Fano Factor > 1：super-Poisson，发放比预期更不规则（常见于 burst 放电）

虽然真实神经元不完全服从 Poisson 分布（有不应期、burst 放电等现象），但 Poisson 假设在合理的 bin 宽度（10–50ms）下是一个被广泛接受且有效的近似，已在 LFADS（Pandarinath et al., 2018）、NDT（Ye et al., 2021）、POYO（Azabou et al., 2023）等多个神经数据建模工作中使用。

#### 为什么 Poisson NLL 适合神经活动预测

1. **统计假设匹配**：spike count 是非负整数的计数数据，Poisson 分布正是描述计数数据的经典分布族
2. **自动编码 mean-variance 耦合**：Poisson NLL 隐式假设了「预测的 rate 越高，允许的方差也越大」，这与真实神经元高发放率时波动也大的经验一致
3. **处理零值自然**：大量 bin 中 spike count = 0，Poisson NLL 在 k=0 时有明确定义（L = lambda），不会产生 log(0) 问题
4. **信息论最优**：如果真实数据生成过程是 Poisson 的，Poisson NLL 就是最大似然估计，具有渐近最优性

### 1.2 Poisson NLL 的计算方式

#### 数学推导

给定 Poisson 分布 P(k|lambda) = lambda^k * e^(-lambda) / k!，负对数似然为：

```
-log P(k|lambda) = lambda - k * log(lambda) + log(k!)
```

由于 log(k!) 是常数（不依赖模型参数 lambda），在优化中可以省略，因此实际 loss 为：

```
L_Poisson = lambda - k * log(lambda)
```

#### 实际实现（log-rate 参数化）

直接输出 lambda（firing rate）需要保证非负，这给优化带来麻烦。实践中模型输出的是 **log-rate** r = log(lambda)，则 lambda = exp(r)，loss 变为：

```
L_Poisson = exp(r) - k * r
```

这就是本项目（NeuroHorizon）和 NDT/LFADS 等工作中使用的形式。

**数值稳定性处理**：将 r clamp 到 [-10, 10] 区间，防止 exp(r) 溢出（exp(10) 约 22026，对应的 spike rate 已远超生理范围）。

#### 本项目的实现（torch_brain/nn/loss.py）

```python
class PoissonNLLLoss(Loss):
    def forward(self, input, target, weights=None):
        log_rate = input.clamp(-10, 10)
        loss = torch.exp(log_rate) - target * log_rate
        # ...
```

#### 梯度分析

```
dL/dr = exp(r) - k
```

- 当 exp(r) > k（预测 rate 偏高）时，梯度为正，推动 r 减小
- 当 exp(r) < k（预测 rate 偏低）时，梯度为负，推动 r 增大
- 当 exp(r) = k 时，梯度为零——此时 rate 恰好等于观测 count，是极大似然点

### 1.3 NDT3 的 Cross-Entropy Loss：一种不同的思路

#### NDT3 的做法

NDT3（Ye et al., 2025, "A Generalist Intracortical Motor Decoder"）采用了与 NDT1/NDT2 不同的 loss 策略：

- **NDT1/NDT2**：模型输出 log firing rate，用 Poisson NLL 训练
- **NDT3**：模型输出 spike count 的**类别概率分布**，用 **Categorical Cross-Entropy** 训练

具体来说，NDT3 将 spike count 视为离散类别（如 0, 1, 2, ..., K），模型对每个 spatiotemporal token（20ms x 32 channels patch）输出一个 (K+1) 维的概率分布，训练时用标准的分类交叉熵 loss。

NDT3 论文原文描述："NDT3 is trained with mean-squared error for prediction of behavioral variables, and categorical cross-entropy losses for prediction of neural spike count and return."

#### Cross-Entropy 的计算方式

```
L_CE = -log(p_k)
```

其中 p_c 是模型预测的 spike count = c 的概率（softmax 输出），k 是实际观测值。

#### 两种 Loss 的对比

| 维度 | Poisson NLL | Categorical Cross-Entropy (NDT3) |
|------|-------------|----------------------------------|
| **建模假设** | spike count 服从 Poisson 分布 | 不做分布假设，直接建模离散概率分布 |
| **输出维度** | 1 维（log rate） | (K+1) 维（每个可能 count 一个 logit） |
| **参数效率** | 每个 bin 只需 1 个标量输出 | 每个 bin 需 K+1 个输出 |
| **表达能力** | 只能表达 Poisson 分布（均值=方差） | 可表达任意离散分布（bimodal、zero-inflated 等） |
| **对 count=0 的处理** | loss = exp(r)，自然处理 | 需要足够的 softmax 概率分配给 0 类 |
| **高 count 值** | 连续外推（exp(r) 可取任意正值） | 需要预定义最大类别 K，超出范围的 count 需截断 |
| **与 LLM 范式的兼容性** | 需要特殊 loss | 与标准语言建模一致（next-token prediction） |

#### NDT3 选择 Cross-Entropy 的可能原因

1. **Foundation Model 范式统一**：NDT3 是多模态自回归模型，同时预测 neural spike、behavioral covariates 和 return signal。将 spike count 也离散化为 token，使得神经数据和行为数据都纳入统一的 next-token prediction 框架，简化了架构设计
2. **更强的表达能力**：真实神经元的 spike count 分布未必严格服从 Poisson（可能存在 overdispersion、zero-inflation 等），Cross-Entropy 不做分布假设，能够灵活拟合任意分布形状
3. **Scaling Law 考量**：在大规模预训练中（2000 小时数据、350M 参数），分类 loss 的行为与语言模型相似，更容易利用已有的 scaling 经验

#### 权衡与讨论

- **Poisson NLL 的优势**：参数高效、包含物理先验（mean-variance coupling），在中小规模数据上通常表现更好，因为先验减少了需要学习的自由度
- **Cross-Entropy 的优势**：表达能力更强、与大规模自回归预训练范式天然兼容，但需要更多数据来学好每个类别的概率
- **实际选择取决于场景**：NeuroHorizon 当前使用 Poisson NLL，因为我们的训练数据量相对有限（单个 dataset 而非 2000 小时跨实验室数据），Poisson 先验提供了有效的归纳偏置（inductive bias）

---

## 2. Spike 稀疏性与 Loss 统计策略

### 2.1 神经元 Spike Events 的稀疏性

皮层神经元的平均发放率通常在 1–20 Hz 范围。以 20ms bin width 为例：

- 平均 firing rate = 10 Hz，每个 bin 的期望 spike count = 10 x 0.02 = 0.2
- 即 **80% 以上的 bin 是零**
- 高活跃神经元（50 Hz）：期望 count = 1.0/bin，但仍有约 37% 的 bin 为零（P(0|lambda=1) = e^(-1) 约 0.37）

更窄的 bin（如 1ms）稀疏性更极端——绝大多数 bin 为 0，偶尔为 1，几乎不会出现 >1。

这种极端的稀疏性对模型训练和 loss 选择有深远影响。

### 2.2 稀疏性与 Poisson NLL 的关系

Poisson NLL **天然适配稀疏数据**，原因如下：

1. **零值不引入额外惩罚**：当 target k=0 时，L = exp(r)。模型只需将 log-rate r 推到足够低的值（如 r=-3，对应 rate=0.05），loss 就很小。不需要额外的「零权重」策略
2. **非零值有信息量**：当 k>0 时，loss = exp(r) - k*r，最小值在 r = log(k) 处取得。稀疏的非零观测自然成为学习发放率的关键信号
3. **自动平衡零与非零**：Poisson 分布本身就编码了「大多数时刻不发放」的信息——当 lambda 小时，P(0|lambda) 自然就大

因此，**Poisson NLL 与稀疏性是兼容的**，但它本身并不是「因为稀疏所以选 Poisson」——选择 Poisson 的根本原因是 spike count 的生成过程近似 Poisson。稀疏性是 Poisson 过程在低 lambda 下的自然结果。

### 2.3 不同 Loss 统计粒度的对比

在神经活动预测中，有多种 loss 统计策略，它们在时间粒度和建模假设上各有不同：

#### 方案 A：逐 Spike Event 预测（Point Process / Temporal Point Process）

**做法**：不做 binning，直接预测下一个 spike 发生的时刻（continuous time）。

**数学框架**：条件强度函数 lambda*(t)，loss 为：

```
L_PP = -sum_i log(lambda*(t_i)) + integral_0^T lambda*(t) dt
```

**优点**：
- 保留了完整的时间精度，不损失 spike timing 信息
- 理论上最完整的建模方式，可以捕捉精细的时间结构（如精确同步、振荡相位锁定）
- 不需要选择 bin width 这个超参数

**缺点**：
- 计算成本高：积分项通常需要数值近似（如蒙特卡洛积分、分段线性近似）
- 对多神经元（hundreds of neurons）场景扩展性差：每个神经元是一个独立的 point process，联合建模的复杂度很高
- 实际中很多下游任务（解码、BCI）并不需要 ms 级精度，10–20ms 的 binned rate 已足够

**代表工作**：Neural Hawkes Process（Mei & Eisner, 2017）、SPINT（Le et al., NeurIPS 2025）

#### 方案 B：Binned Spike Count + Poisson NLL

**做法**：将时间轴分为固定宽度的 bin（如 20ms），统计每个 bin 的 spike count，模型输出 log-rate，用 Poisson NLL 训练。

**优点**：
- 统计假设合理且参数高效（每 bin 每神经元 1 维输出）
- 天然处理稀疏性（大量 zero count 不是问题）
- 计算高效，与 Transformer 架构兼容性好
- 在中小规模数据上由 Poisson 先验提供正则化

**缺点**：
- bin width 是需要调节的超参数（太宽损失时间精度，太窄加剧稀疏性且增加序列长度）
- Poisson 假设的局限：无法表达 overdispersion（实际 Fano Factor > 1 时）或 zero-inflation
- 均值-方差耦合是强约束，如果真实分布不满足，可能导致系统性偏差

**代表工作**：NDT（Ye et al., 2021）、NDT2（Ye et al., 2023）、LFADS（Pandarinath et al., 2018）、POYO（Azabou et al., 2023）

#### 方案 C：Binned Spike Count + Cross-Entropy（分类任务）

**做法**：将 binned spike count 视为离散类别（0, 1, 2, ..., K），模型输出 (K+1) 维 logits，用 categorical cross-entropy 训练。

**优点**：
- 不做分布假设，能拟合任意分布（包括 bimodal、zero-inflated 等）
- 与语言建模的 next-token prediction 框架统一，适合大规模自回归预训练
- 可以利用 softmax temperature、top-k/top-p sampling 等成熟的生成策略

**缺点**：
- 参数量膨胀：每 bin 每 token 需要 K+1 维输出（如 K=20，输出维度从 1 扩大到 21）
- 需要预定义最大 count K，超出 K 的观测被截断（信息损失）
- 将计数数据视为无序类别，丧失了「count=3 比 count=2 大 1」的序数信息（除非使用 ordinal loss）
- 在小数据量下容易过拟合：需要学习每个类别的完整概率分布，自由度远大于 Poisson（1 个参数 vs K+1 个参数）

**代表工作**：NDT3（Ye et al., 2025）

#### 方案 D：Negative Binomial（负二项分布）Loss

**做法**：用 Negative Binomial 替代 Poisson，允许 overdispersion（方差 > 均值）。

```
Var[X] = mu + mu^2 / theta
```

其中 theta 是 dispersion 参数，theta 趋向无穷时退化为 Poisson。

**优点**：
- 比 Poisson 更灵活，能处理 overdispersion
- 仍然是参数化的分布，保留了参数效率
- 在 RNA-seq 等生物计数数据中已被广泛验证

**缺点**：
- 多一个需要学习或预设的参数 theta
- 如果数据确实是 Poisson 的，额外参数带来不必要的复杂度

**代表工作**：在神经科学中 Negative Binomial 较少用于 spike count 建模，但在 scRNA-seq 中是标准选择（scVI, Lopez et al., 2018）。

#### 方案 E：Zero-Inflated Poisson（ZIP）

**做法**：混合模型——以概率 pi 生成「结构性零」（神经元完全静默），以概率 (1-pi) 从 Poisson(lambda) 采样。

```
P(k=0) = pi + (1-pi) * e^(-lambda)
P(k>0) = (1-pi) * lambda^k * e^(-lambda) / k!
```

**优点**：
- 显式建模了「真实的零」vs「随机的零」，对极端稀疏数据可能更准确
- 保留了 Poisson 核心结构

**缺点**：
- 额外需要学习 pi 参数
- 对于大多数皮层神经元，标准 Poisson 已经足够描述稀疏性，ZIP 的额外复杂度可能不必要

### 2.4 综合讨论：如何看待这个问题

#### 关键权衡轴

1. **时间粒度 vs 计算效率**：逐 event（方案 A）最精细但最贵，binned（方案 B/C/D/E）是实用妥协
2. **分布假设强度 vs 数据需求**：Poisson（强假设、少参数）到 Neg. Binomial / ZIP（中等假设）到 Cross-Entropy（无假设、需要更多数据）
3. **与下游任务的匹配**：如果目标是 BCI 解码（需要的是 latent representation 而非精确 spike count），那么 loss 选择对最终性能的影响可能小于预期——latent 质量才是关键

#### NeuroHorizon 的选择逻辑

NeuroHorizon 使用 **Poisson NLL**（方案 B），理由是：

1. **数据规模有限**：我们使用单个或少量 datasets，不是 NDT3 那样的 2000 小时跨实验室数据，Poisson 先验提供了有效的正则化
2. **bin width = 20ms**：这是 spike count 建模的主流选择，稀疏性可控（非极端），Poisson 近似合理
3. **评估指标一致**：NDT/POYO 的 baseline 都使用 Poisson NLL，用相同的 loss 可以做公平对比
4. **参数效率**：每 bin 每神经元只需 1 维输出，这对我们的 Small 模型（4.2M 参数）尤为重要

如果后续发现 Poisson 假设不足（如某些神经元 Fano Factor 显著偏离 1），可以考虑升级到 Negative Binomial。

---

## 3. Scheduled Sampling：概念、用途与引入时机

### 3.1 什么是 Scheduled Sampling

**Scheduled Sampling**（Bengio et al., NeurIPS 2015）是一种课程学习（curriculum learning）策略，用于缓解自回归序列预测模型中的 **exposure bias**（暴露偏差）问题。

#### Exposure Bias 问题

自回归模型的训练和推理存在根本性不一致：

- **训练时（Teacher Forcing）**：每一步的输入是 **ground truth**（真实的前一步数据）
  - 输入序列：[x1_gt, x2_gt, x3_gt, ...] -> 预测 [x2_hat, x3_hat, x4_hat, ...]
- **推理时（Autoregressive Generation）**：每一步的输入是 **模型自己上一步的预测**
  - 输入序列：[x1_gt, x2_hat, x3_hat, ...] -> 预测 [x2_hat, x3_hat, x4_hat, ...]

这意味着：
- 训练时模型从未见过「自己的错误」作为输入
- 推理时一旦某步预测出错，后续步骤的输入分布就偏离了训练分布
- 错误逐步累积（error accumulation），生成质量随步数增加而退化

这就是所谓的 **exposure bias**——模型在训练期间只「暴露」（exposed）在 ground truth 的分布下，从未体验过自身预测误差造成的偏移分布。

#### Scheduled Sampling 的核心思想

在训练过程中，**逐渐用模型自己的预测替换 ground truth 输入**，让模型在训练阶段就学会应对自己的预测误差。

具体机制：在训练的每一步，以概率 epsilon 使用 ground truth 作为输入，以概率 (1-epsilon) 使用模型自己的预测。epsilon 随训练进程从 1（纯 teacher forcing）衰减到接近 0（纯自回归）。

常见的衰减策略：
- **线性衰减**：epsilon(i) = max(epsilon_min, 1 - i/N)
- **指数衰减**：epsilon(i) = k^i，其中 k < 1
- **逆 Sigmoid 衰减**：epsilon(i) = k / (k + e^(i/k))

#### 训练流程示意

```
Epoch 1-50:   epsilon = 1.0     (纯 Teacher Forcing，模型先收敛)
Epoch 51:     epsilon = 0.95    (5% 的步骤用模型自己的预测)
Epoch 60:     epsilon = 0.80    (20% 的步骤用模型自己的预测)
Epoch 80:     epsilon = 0.50    (一半用 GT，一半用预测)
Epoch 100:    epsilon = 0.10    (接近纯自回归训练)
```

每个训练步内，对每个时间步独立地以概率 epsilon 决定用 GT 还是用模型预测——这模拟了推理时模型可能在任何步骤出错的情况。

### 3.2 为什么要考虑引入 Scheduled Sampling

#### 在 NeuroHorizon 中的意义

NeuroHorizon 使用自回归 decoder 预测未来 spike count 序列。以 1000ms 预测窗口（50 个 20ms bin）为例：

- 推理时需要**自回归生成 50 步**
- 每一步的预测误差都会成为下一步的输入误差
- 如果训练只用 teacher forcing，模型对自身误差完全没有鲁棒性

**误差累积的严重性**：假设每步预测引入 5% 的噪声，50 步后累积的偏差可能使预测完全偏离真实轨迹。这在 proposal_review.md 的风险表中被列为「高可能性、高影响」风险。

#### Scheduled Sampling 的具体收益

1. **提高长程生成质量**：让模型学会在「不完美输入」上做出合理预测，减缓误差累积
2. **缩小 train-test gap**：训练条件更接近推理条件，验证指标更可靠（不会出现「训练 loss 很低但实际生成很差」的情况）
3. **隐式学习纠错能力**：模型看到自己的错误预测后仍需做出合理输出，这鼓励了一种「纠偏」能力

### 3.3 什么时候需要考虑引入

#### NeuroHorizon 项目中的分层策略

根据 proposal_review.md 的设计，不同预测窗口对 scheduled sampling 的需求不同：

| 预测窗口 | 自回归步数 | 是否需要 Scheduled Sampling | 理由 |
|----------|-----------|---------------------------|------|
| 250ms | 12 步 | **不需要** | 步数少，误差累积有限；先建立纯 TF 基线 |
| 500ms | 25 步 | **视 250ms 结果决定** | 如果 250ms 的 AR 生成已出现明显退化，则引入 |
| 1000ms | 50 步 | **需要引入** | 50 步几乎必然出现误差累积；同时引入非自回归并行预测作为对照 |

#### 判断是否需要引入的信号

- **AR 推理的误差随步数衰减曲线**：如果 PSTH correlation 在 step 30+ 急剧下降，说明误差累积严重，需要 scheduled sampling
- **TF 与 AR 的性能 gap**：如果 teacher forcing 下 R2=0.5 但自回归推理 R2=0.1，说明模型严重依赖真实输入，scheduled sampling 能缩小这个 gap
- **预测窗口扩大时性能骤降**：从 250ms 扩展到 500ms 时如果性能不成比例地下降，说明误差累积是主要瓶颈

#### 引入时的具体策略（proposal_review.md 方案）

1000ms 窗口训练方案：
- 前 N epoch：100% teacher forcing（先让模型收敛到合理水平）
- 之后 20–50 epoch：epsilon 从 1.0 线性衰减至约 0.1
- 最终阶段：维持 epsilon 约 0.1（保留少量 teacher forcing 以稳定训练）

### 3.4 Scheduled Sampling 的局限性与替代方案

#### 局限性

1. **理论不一致性**：Huszar (2015) 指出 scheduled sampling 训练的目标函数与任何一致的概率模型都不对应——它既不是 teacher forcing 的 MLE，也不是纯自回归的 MLE，而是两者的混合
2. **对 Transformer 不太自然**：原始 scheduled sampling 为 RNN 设计，每步顺序决策。Transformer 的并行计算特性使得标准 scheduled sampling 无法高效实现——需要逐步 unroll 而非并行计算
3. **超参数敏感**：衰减速度、初始 epoch、最终 epsilon 值都需要调节

#### 替代方案

| 方案 | 思路 | 优缺点 |
|------|------|--------|
| **Professor Forcing** (Lamb et al., 2016) | 用对抗训练对齐 TF 和 AR 的 hidden state 分布 | 更有原则性，但训练不稳定 |
| **Parallel Scheduled Sampling** | 适配 Transformer 的并行变体 | 保持并行效率，但实现更复杂 |
| **Coarse-to-Fine** | 先用大 bin（如 100ms）预测粗粒度，再细化到小 bin（20ms） | 减少细粒度上的 AR 步数，但需要两阶段架构 |
| **非自回归并行预测** | 去掉 causal mask，一次性预测所有 bins | 完全避免误差累积，但丧失了时间因果结构 |

NeuroHorizon 在 plan.md 1.3.3 中设置了非自回归并行预测作为消融对照，用于验证自回归结构相比并行预测是否真正带来收益。如果自回归带来的收益不足以抵消误差累积的代价，那么 scheduled sampling 就不是「修补问题」而是在提示我们重新考虑架构。

---

## 参考文献

- Bengio, S., Vinyals, O., Jaitly, N., & Shazeer, N. (2015). Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks. *NeurIPS 2015*.
- Ye, J., et al. (2021). Representation learning for neural population activity with Neural Data Transformers. *bioRxiv*.
- Ye, J., et al. (2023). Neural Data Transformer 2: Multi-context Pretraining for Neural Spiking Activity. *NeurIPS 2023*.
- Ye, J., et al. (2025). A Generalist Intracortical Motor Decoder (NDT3). *bioRxiv / NeurIPS 2025*.
- Pandarinath, C., et al. (2018). Inferring single-trial neural population dynamics using sequential auto-encoders (LFADS). *Nature Methods*.
- Azabou, M., et al. (2023). A unified, scalable framework for neural population decoding (POYO). *NeurIPS 2023*.
- Le, T., et al. (2025). SPINT: Spike-based Point-process Interaction Network Transformer. *NeurIPS 2025*.
- Mei, H., & Eisner, J. (2017). The Neural Hawkes Process: A Neurally Self-Modulating Multivariate Point Process. *NeurIPS 2017*.
- Lopez, R., et al. (2018). Deep generative modeling for single-cell transcriptomics (scVI). *Nature Methods*.
- Lamb, A., et al. (2016). Professor Forcing: A New Algorithm for Training Recurrent Networks. *NeurIPS 2016*.
- Huszar, F. (2015). How (not) to train your generative model: Scheduled sampling, likelihood, adversary? *arXiv*.

---

*文档创建时间：2026-03-02*
*最后更新：2026-03-02*
