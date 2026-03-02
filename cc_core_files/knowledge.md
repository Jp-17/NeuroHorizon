# NeuroHorizon 知识库（Knowledge Base）

> 本文档收集项目相关的核心概念、技术讨论与设计决策的深度分析。
> 每个条目力求解释充分、逻辑完整，可作为项目技术方案的理论支撑。

---

## 目录

1. [Poisson NLL 与神经活动预测的 Loss 选择](#1-poisson-nll-与神经活动预测的-loss-选择)
2. [Spike 稀疏性与 Loss 统计策略](#2-spike-稀疏性与-loss-统计策略)
3. [Scheduled Sampling：概念、用途与引入时机](#3-scheduled-sampling概念用途与引入时机)
4. [各阶段评估指标统一整理](#4-各阶段评估指标统一整理)
5. [各阶段 Baseline 统一整理](#5-各阶段-baseline-统一整理)
6. [数据组织考量](#6-数据组织考量)
7. [结果呈现与 Figure 规划](#7-结果呈现与-figure-规划)

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


## 4. 各阶段评估指标统一整理

### 4.1 指标总览表

下表汇总 NeuroHorizon 各 Phase 使用的评估指标。标注说明：**T** = 训练目标（loss），**P** = 主要评估指标，**A** = 辅助参考指标，**-** = 不使用。

| 指标 | Phase 0 | Phase 1 | Phase 2 | Phase 3 | Phase 4 |
|------|---------|---------|---------|---------|---------|
| Poisson NLL | - | **T/P** | **T/P** | T | **T/P** |
| R²（行为解码） | **P** | A | **P** | **P** | A |
| R²（spike 预测） | - | **P** | **P** | A | A |
| PSTH Correlation | - | **P** | A | A | A |
| Error Decay Curve | - | **P** | - | - | - |
| Zero-shot R² | - | - | **P** | A | - |
| Embedding Clustering | - | - | A | - | - |
| Scaling Curve | - | - | - | **P** | - |
| Transfer Gain | - | - | - | **P** | - |
| Delta_m（模态贡献） | - | - | - | - | **P** |
| Spike Rate 合理性 | - | A | A | - | A |

### 4.2 逐指标详解

#### 4.2.1 Poisson NLL（Poisson Negative Log-Likelihood）

**公式**（log-rate 参数化，详见第 1 章）：

```
L_Poisson = exp(r) - k * r
```

其中 r 是模型输出的 log firing rate，k 是观测到的 binned spike count。

**使用场景**：
- **Phase 1–4 的训练目标**：自回归 decoder 输出 log-rate，用 Poisson NLL 作为 loss 函数训练
- **Phase 1/2 的主要评估指标**：在 held-out trials 上计算平均 Poisson NLL，值越低越好
- **跨模型可比性**：NDT1/2、LFADS、POYO 都使用 Poisson NLL，使得结果可以直接对比

**注意**：Poisson NLL 的绝对值受神经元发放率影响——高发放率神经元的 NLL 天然更大。跨数据集对比时需注意归一化（如使用 bits/spike，见 4.2.6）。

#### 4.2.2 R²（Coefficient of Determination）

**公式**：

```
R^2 = 1 - SS_res / SS_tot = 1 - sum((y - y_hat)^2) / sum((y - y_mean)^2)
```

**两种用法**：

1. **R²（行为解码）**：
   - 预测变量：行为输出（如 cursor velocity 2D）
   - 用途：评估模型 latent representation 对下游任务的信息量
   - Phase 0 基线值：0.807（POYO+ 在 Perich-Miller 上的表现）
   - 贯穿所有 Phase，作为「模型改造不应退化行为解码能力」的锚点

2. **R²（spike 预测）**：
   - 预测变量：binned spike count
   - 用途：直接评估 spike count 预测质量
   - Phase 1 目标：R² > 0.3（在 held-out session 上）
   - Phase 2 zero-shot 目标：R² > 0.2

**解读注意**：
- R² 可以为负数（当模型预测比均值预测还差时）
- 对低发放率神经元敏感——如果某神经元几乎不发放，y_mean 约 0，稍有偏差 SS_res/SS_tot 就很大
- 建议同时报告 per-neuron R² 分布和 population-averaged R²

#### 4.2.3 PSTH Correlation（Peri-Stimulus Time Histogram Correlation）

**定义**：在多 trial 数据上，将同一条件（如相同 reach target）的 trials 对齐后平均，得到 trial-averaged 预测和 trial-averaged 真实 firing rate，计算两者的 Pearson 相关系数。

**公式**：

```
PSTH_corr = Pearson_r(mean_trials(y_hat), mean_trials(y))
```

**为什么比 single-trial R² 更稳定**：
- Single-trial spike count 噪声很大（Poisson 噪声），R² 会被随机波动压低
- Trial-averaged 后噪声被平均掉，暴露的是模型是否捕捉到了真实的 firing rate 动态
- PSTH correlation 更接近「模型是否学到了正确的 tuning」的评估

**使用场景**：
- Phase 1 主要指标：用于评估自回归生成质量
- Phase 1 Error Decay Curve 的基础指标（见 4.2.4）
- 不适用于 zero-shot 跨 session 场景（不同 session 的 trial 条件可能不同）

#### 4.2.4 Error Decay Curve（误差随时间步衰减曲线）

**定义**：在自回归生成过程中，逐步计算每一步的 PSTH correlation（或 R²），绘制「预测质量 vs 预测步数」的曲线。

**计算方式**：
1. 对 held-out trials，自回归生成 T 步（如 50 步 = 1000ms）
2. 在每个步数 t 处，计算 PSTH correlation(step=t)
3. 绘制 t vs correlation 曲线

**用途**：
- **Phase 1 特有指标**：诊断自回归生成的误差累积速度
- 理想情况：曲线缓慢衰减（步数增加，质量微降）
- 问题信号：曲线急剧下降（如 step 20 后 correlation 跌到 0）→ 需要 scheduled sampling 或其他缓解措施
- 对比用途：TF vs AR 的 decay curve 差异、scheduled sampling 前后的 decay curve 差异

#### 4.2.5 Co-smoothing

**定义**：NLB（Neural Latents Benchmark）的核心指标。将神经元随机分为 held-in（75%）和 held-out（25%），模型只看 held-in 神经元的 spike 数据，推断 latent state，再从 latent 预测 held-out 神经元的 firing rate。

**计算方式**：
1. 随机划分神经元为 held-in / held-out
2. 模型编码 held-in 神经元 → latent
3. Linear readout 从 latent → 预测 held-out 神经元 firing rate
4. 对 held-out 预测做 Gaussian 平滑（sigma = 50ms）
5. 计算 R² 或 Poisson NLL

**意义**：
- 直接评估 latent representation 的质量——好的 latent 应该包含整个 population 的信息，而不仅仅是输入的那些神经元
- 避免了「模型只是记住了每个神经元的 PSTH」的情况

**NeuroHorizon 是否需要**：
- Phase 0–1：不必须，因为还在验证基础功能
- Phase 2+：建议作为辅助指标，特别是在跨 session 评估中，co-smoothing 可以检验 IDEncoder 推断的 embedding 是否真正捕捉了神经元的功能角色
- Phase 5 论文：如果在 NLB benchmark 数据上评估，co-smoothing 是必须报告的

#### 4.2.6 Bits/spike

**定义**：信息论指标，衡量模型在每个 spike 上提供了多少比特的预测信息（相比于 baseline 的均值预测模型）。

**计算方式**：

```
bits/spike = (L_baseline - L_model) / (N_spikes * log(2))
```

其中 L_baseline 是 homogeneous Poisson model（每个神经元用其平均 firing rate 作为预测）的 Poisson NLL，L_model 是待评估模型的 Poisson NLL，N_spikes 是总 spike 数。

**意义**：
- **归一化了发放率差异**：不同数据集、不同神经元的 Poisson NLL 绝对值差异很大，bits/spike 提供了一个「相对于 trivial baseline 提升了多少」的标准化度量
- **跨论文可比**：NDT1/2、LFADS 等都报告 bits/spike，使得不同工作可以直接比较
- 值越高越好；典型范围为 0.1–0.5 bits/spike

**NeuroHorizon 建议**：在 Phase 1+ 的结果报告中增加 bits/spike，增强与文献的可比性。实现简单——只需额外计算一个 homogeneous Poisson baseline 的 NLL。

#### 4.2.7 Zero-shot R²

**定义**：在完全未见过的 session（训练集中不包含该 session 的任何 trial）上，通过 IDEncoder 推断神经元 embedding 后直接做预测，不进行任何梯度更新。

**Phase 2 核心指标**：
- 目标阈值：R² > 0.2（vs InfiniteVocabEmbedding lookup 的约 0）
- 跨动物 R² degradation < 30%
- 这是 NeuroHorizon 跨 session 泛化能力的直接证据

#### 4.2.8 Scaling Curve

**定义**：Phase 3 核心指标。以训练使用的 session 数量为横轴，R²（或 Poisson NLL）为纵轴，观察性能随数据规模的增长趋势。

**分析方式**：
- 至少 4 个数据点（如 5/10/20/40/70 sessions）
- 绘制带 error bar 的曲线
- 拟合 power law：R² = a * N^b + c，分析 scaling exponent b
- 与 POYO+ 论文的 scaling curve 对比

#### 4.2.9 Delta_m（模态贡献度）

**定义**：Phase 4 核心指标。通过消融实验量化每个模态（neural / behavior / image）对预测质量的贡献。

**计算方式**：

```
Delta_m = Loss(full_model) - Loss(ablate_modality_m)
```

如果 Delta_m > 0，说明去掉模态 m 后 loss 变差，即模态 m 有正贡献。

**条件分解** Delta_m(v)：按脑区、刺激类型、行为状态分别计算，揭示模态贡献的空间和条件依赖性。

预期：Delta_image 在视觉皮层（V1）最大，在运动皮层最小。

### 4.3 主流模型的指标选择对比

| 模型 | Poisson NLL | R²(行为) | R²(spike) | PSTH Corr | Co-smoothing | Bits/spike | 其他 |
|------|-------------|----------|-----------|-----------|--------------|------------|------|
| NDT1 | **T/P** | P | - | - | - | P | 推理延迟 |
| NDT2 | **T/P** | P | - | - | - | P | 跨 context 泛化 |
| NDT3 | - (用 CE) | **P** | - | - | - | - | 下游任务多样性 |
| POYO | - | **P** | - | - | - | - | NLB 排名 |
| POYO+ | - | **P** | - | - | - | - | Scaling curve |
| SPINT | - | **P** | - | - | - | - | Zero-shot R², FALCON |
| LFADS | **T/P** | P | - | - | P(NLB) | P | 单 trial 去噪 |
| Neuroformer | P | P | - | - | - | - | 多模态 attribution |

**总结**：
- **Poisson NLL + R²（行为解码）** 是最通用的指标组合
- **Bits/spike** 在 NDT 系列中常用，提供了跨工作可比性
- **Co-smoothing** 是 NLB benchmark 的核心指标，但并非所有论文都报告
- NDT3 使用 Cross-Entropy 而非 Poisson NLL，因此其 loss 值不能与其他模型直接对比

### 4.4 对 NeuroHorizon 的建议

1. **当前方案合理**：Poisson NLL + R² + PSTH Correlation 覆盖了训练和评估的核心需求
2. **建议补充 bits/spike**：实现成本极低（一次额外的 baseline NLL 计算），但显著提升与 NDT1/2、LFADS 等经典工作的可比性
3. **Phase 2+ 建议引入 co-smoothing 作为辅助指标**：用于验证 IDEncoder 推断的 embedding 是否真正捕捉了神经元的功能角色，也为后续在 NLB benchmark 上评估做准备
4. **Phase 5 论文中建议报告完整指标矩阵**：Poisson NLL、R²、PSTH Correlation、bits/spike，覆盖与 NDT/POYO/SPINT 的可比性

---

## 5. 各阶段 Baseline 统一整理

### 5.1 Baseline 总览表

#### Phase 0 Baselines

| Baseline | 方法 | 对比目的 | 指标 | 参考值 |
|----------|------|---------|------|--------|
| POYO+ 行为解码 | dim=128, depth=12, 约 8M 参数 | 建立行为解码锚点 | R²(velocity) | **0.807** |

#### Phase 1 Baselines

| Baseline | 方法 | 对比目的 | 实现复杂度 |
|----------|------|---------|-----------|
| PSTH Prediction | trial-averaged firing rate | 最低性能下限 | 极低 |
| Linear Regression | history bins 线性映射 future counts | 消融非线性价值 | 低 |
| Smoothed Firing Rate | Gaussian kernel 平滑历史 firing rate 外推 | 简单统计 baseline | 低 |
| Non-AR Parallel Prediction | 去掉 causal mask，并行预测所有 bins | 消融自回归的必要性 | 中（修改模型配置） |
| **Neuroformer** (外部) | 逐 spike event 自回归预测 | 方法论对比：binned vs event-level AR | 中–高 |
| NDT1/2 (外部) | Masked modeling + Poisson NLL | 经典参照：masked vs autoregressive | 引用论文数值 |

#### Phase 2 Baselines

| Baseline | 方法 | 对比目的 | 实现复杂度 |
|----------|------|---------|-----------|
| InfiniteVocabEmbedding Lookup | Per-session 可学习 embedding（POYO 原始方式） | IDEncoder 相对于 lookup 的优势 | 已有实现 |
| IDEncoder Scheme A | 类 SPINT 方式：binned spike count 输入 | Scheme B 的对照 | 中 |
| IDEncoder Scheme B | NeuroHorizon 创新：spike event + RoPE pooling | 核心创新验证 | 中 |
| Within-Session 90/10 Split | 同 session 内 train/test | 性能上界 | 低 |
| **SPINT IDEncoder** (外部) | 原始 SPINT 的 gradient-free 方案 | 展示改进 | **必须复现** |
| **POYO IVE** (外部) | POYO 的 InfiniteVocabEmbedding | 基础框架对比 | 已有实现 |

#### Phase 3 Baselines

| Baseline | 方法 | 对比目的 | 实现复杂度 |
|----------|------|---------|-----------|
| From-scratch 行为解码 | 随机初始化 → 行为解码 | 预训练的价值基线 | 低 |
| Frozen Transfer | 预训练 encoder 冻结 → 新 behavior head | 预训练表征质量 | 中 |
| Fine-tuned Transfer | 预训练 encoder 小 lr 微调 → 新 head | 微调的增量价值 | 中 |
| Few-shot (10%/25%/50%) | 不同标注量下的 Transfer vs Scratch | 数据效率分析 | 中 |
| **POYO+ Scaling** (外部) | POYO+ 论文报告的 scaling curve | 数据效率对比 | 引用论文数值 |

#### Phase 4 Baselines（消融矩阵）

| 条件 | Neural Only | +Behavior | +Image | +Both |
|------|-------------|-----------|--------|-------|
| Natural Movies | baseline | Delta_beh | Delta_img | Full |
| Natural Scenes | baseline | Delta_beh | Delta_img | Full |
| 期望：V1 | 中 | 小提升 | **大提升** | 最佳 |
| 期望：Motor | 中 | **大提升** | 小提升 | 最佳 |

### 5.2 逐 Baseline 详解

#### PSTH Prediction（最简 Baseline）

**方法**：对于每个 trial 条件（如 reach 到某个 target），计算训练集中同条件 trials 的平均 firing rate，将该平均值作为所有 test trials 的预测。

**实现**：
```python
# 按条件分组，计算 trial-averaged firing rate
for condition in conditions:
    train_trials = get_trials(train_set, condition)
    psth = mean(train_trials, axis=0)  # [T_bins, N_units]
    predictions[condition] = psth
```

**意义**：如果模型不能超过 PSTH prediction，说明模型没有从 single-trial 数据中提取到有用的信息。

#### Linear Regression

**方法**：将历史窗口的 binned spike counts（或 population firing rate vector）作为特征，用线性回归预测未来窗口的 spike counts。

**意义**：消融模型非线性建模能力的价值。如果 NeuroHorizon 只比线性回归好一点，说明 Transformer 架构的复杂度可能不值得。

#### Smoothed Firing Rate（建议新增）

**方法**：用 Gaussian kernel（sigma = 50–100ms）平滑历史 spike train，然后线性外推到未来窗口。

**意义**：比 PSTH 更好但比 Linear Regression 更简单的 baseline，提供中间参考点。在 NLB benchmark 中也常用作 simple baseline。

#### Non-AR Parallel Prediction

**方法**：使用与 NeuroHorizon 相同的模型架构，但去掉 decoder 的 causal mask，允许所有 bins 互相 attend。

**意义**：
- 如果 parallel > AR：说明自回归的误差累积损害超过了因果建模的收益，可能需要更短的预测窗口或更好的误差控制
- 如果 AR > parallel：说明时间因果结构对预测有帮助，验证了自回归设计的必要性

### 5.3 主流模型的 Baseline 设计对比

不同的 spike foundation model 在论文中选择的 baseline 反映了各自的 positioning：

| 模型 | 论文中使用的 Baselines | 评估框架 |
|------|----------------------|---------|
| **NDT1** | AutoLFADS, LSTM, standard RNN | 单数据集，强调推理速度 |
| **NDT2** | AutoLFADS, LSTM, NDT1 | 多 context pretraining，展示跨 context 泛化 |
| **NDT3** | Linear, from-scratch NDT, NDT2 | FALCON benchmark，强调 foundation model 泛化 |
| **POYO** | Wiener Filter, MLP, GRU, AutoLFADS+Linear, NDT+Linear, NDT-Sup, EIT | NLB benchmark 排名，全面对比 |
| **POYO+** | POYO, single-session baselines | 展示 multi-task + scaling 的增量价值 |
| **SPINT** | Zero-shot baselines, few-shot unsupervised, test-time alignment | FALCON benchmark，强调 gradient-free 跨 session |
| **LFADS** | Linear models | 经典工作，定义了 baseline 层级 |
| **NLB Benchmark** | Linear, LSTM, LFADS, Transformer (分层级) | 标准化层级 |
| **FALCON Benchmark** | Held-in/held-out session 设计 | 标准化跨 session 评估 |

**Baseline 设计的通用原则**：
1. **层级化**：从最简单（linear）到经典模型（LFADS）到 SOTA（NDT/POYO），展示每一层的增量
2. **消融**：去掉自己的核心组件，展示每个组件的必要性
3. **外部 SOTA**：在共享 benchmark 上与同时期最好的模型对比
4. **公平性**：相同数据、相同指标、相同评估协议

### 5.4 外部 SOTA 模型作为 Baseline 的讨论

#### 为什么仅靠自身 Ablation 不够

自身消融（去掉 causal mask、去掉 IDEncoder 等）只能说明各组件的必要性——「这个模块有用」。但 reviewer 关心的另一个核心问题是：「相比现有 SOTA，你的方法总体上更好吗？」

这需要与外部模型做直接对比。

#### 按 Phase 的外部 Baseline 建议

| Phase | 推荐外部 Baseline | 理由 | 实现方式 |
|-------|-------------------|------|----------|
| Phase 1 | **Neuroformer** | 同为自回归 spike prediction，但预测粒度不同（逐 event vs binned），形成方法论对比 | 在 Perich-Miller 上复现或引用论文数值 |
| Phase 1 | NDT1/2（Poisson NLL） | 经典参照，masked modeling vs autoregressive 对比 | 引用论文在 NLB 上的数值 |
| Phase 2 | **SPINT** | IDEncoder 直接源自 SPINT，必须展示改进（Scheme B vs SPINT 原始方案） | 在相同数据上复现 SPINT IDEncoder |
| Phase 2 | **POYO**（IVE） | 基础框架，展示 IDEncoder 相比 lookup table 的优势 | 已有实现，直接对比 |
| Phase 3 | POYO+ scaling curve | 展示在相同数据上 NeuroHorizon 的数据效率是否更高 | 引用论文 figure 数值 |
| Phase 5 | NDT3, POYO+, SPINT, Neuroformer | 论文必须的全面对比表 | 论文报告数值 + 共享 benchmark |

#### 操作策略

**直接复现**（高优先）：
- SPINT IDEncoder：Phase 2 核心创新点，必须在相同数据上直接对比。SPINT 的 IDEncoder 用 MLP 处理 binned reference counts，实现相对简单
- Neuroformer：如果开源代码可用且能在 Perich-Miller 上运行

**论文数值引用**（中优先）：
- NDT1/2/3、POYO+ 在 NLB/FALCON/Perich-Miller 上的报告数值
- 注意标注数据来源和评估条件差异

**不建议完全复现**：
- NDT3：350M 参数 + 2000h 数据预训练，算力需求不现实且量级差异太大
- 如需数值对比，应在论文中明确说明定位差异：NeuroHorizon 侧重跨 session 泛化 + 长时程生成，NDT3 侧重大规模预训练 + 运动解码

#### 关于公平对比的注意事项

- **数据量级对齐**：NDT3 用 2000h 数据，NeuroHorizon 用 10 sessions（约几小时）。直接数值对比不公平，应在论文中说明
- **任务对齐**：POYO 主要做行为解码，NeuroHorizon 主要做 spike 预测——不同任务的 R² 不能直接比较
- **评估协议对齐**：不同论文的 train/test split、random seed、preprocessing 可能不同。最可靠的对比是在完全相同的评估协议下重新跑
- **NLB/FALCON benchmark 的价值**：标准化的数据 + 评估协议，最大程度保证公平性

### 5.5 对 NeuroHorizon 的建议

1. **Phase 1 增加 smoothed firing rate baseline**：实现极简，提供有意义的中间参考
2. **Phase 2 必须复现 SPINT IDEncoder**：这是创新核心的直接对比，无法回避
3. **Phase 1 建议在 Perich-Miller 上与 Neuroformer 对比**：如果 Neuroformer 开源代码可用，在相同数据上运行；否则引用论文数值并标注条件差异
4. **Phase 5 论文阶段补充 LFADS 作为经典对照**：LFADS 是神经数据建模的经典 baseline，reviewer 期望看到与其对比
5. **考虑 NLB/FALCON benchmark**：如果有精力，在标准化 benchmark 上提交结果可显著增强论文说服力

---

## 6. 数据组织考量

### 6.1 时间区间选择：Hold vs Reach

#### Center-out Reaching 任务的时间结构

Perich-Miller 2018 数据集的典型 trial 时间线：

```
|--- Hold Period ---|-- Go Cue --|--- Reach Period ---|--- Target Acquire ---|
|     约 676ms      |            |     约 1090ms      |                      |
|  手保持在中心不动  |   视觉信号  |  手向目标方向移动   |   到达并停在目标上    |
```

#### 为什么用 Hold 作为 Encoder Input、Reach 作为 Prediction Target

1. **Hold 期间的神经活动包含准备信息**：motor planning 阶段，神经元 firing rate 反映了即将执行的运动方向。作为 encoder input 提供了丰富的上下文
2. **Reach 期间是运动执行阶段**：firing rate 动态变化，包含速度、位置等连续变化信息，是更有意义的预测目标
3. **时间上的因果关系**：hold 在 reach 之前，用历史预测未来是自然的因果方向
4. **实际 BCI 应用场景**：在闭环 BCI 中，hold 期间的数据可用于预测接下来的运动意图

#### 统计特性（Phase 0 分析结果）

- **Hold 期间**：均值 676ms，87% 的 trials > 250ms → 支持 250ms 编码器输入窗口
- **Reach 期间**：均值 1090ms，100% > 500ms，75% > 1s → 支持 250/500/1000ms 预测窗口

#### Scheme A（Trial-aligned）vs Scheme B（Sliding Window）

| 维度 | Scheme A（trial-aligned） | Scheme B（sliding window） |
|------|--------------------------|--------------------------|
| **输入** | hold 期间完整 spike events | 固定长度滑动窗口的 spike events |
| **预测目标** | reach 期间前 T ms 的 binned counts | 滑动窗口后 T ms 的 binned counts |
| **优点** | 利用任务结构（hold→reach 因果关系）；trial 间自然对齐 | 不依赖 trial 边界，泛化性更强；数据量更大（更多窗口） |
| **缺点** | 受 trial 结构约束；不适用于无 trial 结构的自由行为数据 | 可能跨 trial 边界引入噪声；失去了 hold-reach 因果语义 |
| **适用场景** | Phase 1 基线验证（Perich-Miller 有明确 trial 结构） | Phase 3+ 扩展到连续记录数据（如 Allen、IBL） |

**项目策略**：Phase 1 以 Scheme A 为主，250ms 窗口建立基线；500ms 窗口时 A/B 对比决定后续方向；1000ms 窗口转向 Scheme B（trial 内不一定有这么长的 reach）。

### 6.2 Bin Width 选择：20ms 的理由

#### 主流选择范围

文献中使用的 bin width 范围为 2–50ms，不同选择的权衡：

| Bin Width | 1000ms 内的 bins 数 | 稀疏性（10Hz 神经元） | 典型应用 |
|-----------|--------------------|--------------------|---------|
| 1ms | 1000 | 99% 为零 | 精确 spike timing 分析 |
| 5ms | 200 | 95% 为零 | 高时间精度解码 |
| 10ms | 100 | 90% 为零 | NDT1/2 默认 |
| **20ms** | **50** | **80% 为零** | **SPINT, LFADS, NeuroHorizon** |
| 50ms | 20 | 50% 为零 | 粗粒度分析 |

#### 20ms 的平衡点

1. **时间精度 vs 序列长度**：20ms 在 1000ms 窗口内产生 50 个 bins，对 Transformer 的序列长度是可接受的（不需要特殊的长序列处理）；10ms 则 100 个 bins，计算量翻倍
2. **稀疏性 vs 信息量**：20ms bin 下大多数 bin 的 spike count 在 0–2 之间，Poisson 近似合理；1ms bin 下几乎全是 0 或 1，退化为 Bernoulli
3. **NDT1 的实验**：NDT1 在 2–20ms 范围内测试，结果差异很小（「similar results for bin sizes varying from 2ms to 20ms」）
4. **与下游任务的匹配**：BCI 解码和行为分析通常不需要 < 10ms 的时间精度

#### Bin Width 对 Spike Count 分布的影响

```
bin=1ms:   P(k=0) = 99%,  P(k=1) = 1%,  P(k>=2) 极少 → 接近 Bernoulli
bin=5ms:   P(k=0) = 95%,  P(k=1) = 5%,  P(k>=2) 约 0.1%
bin=20ms:  P(k=0) = 80%,  P(k=1) = 16%, P(k=2) = 3%, P(k>=3) = 1%
bin=50ms:  P(k=0) = 60%,  P(k=1) = 30%, P(k=2) = 8%, P(k>=3) = 2%
```

（以 10Hz 发放率的典型皮层神经元为例）

20ms bin 下分布有足够的「结构」（不是简单的 0/1 二值），使得 Poisson NLL 有意义的梯度信号。

### 6.3 Previous Window（History）Length

#### 编码器输入格式

NeuroHorizon 继承 POYO 的设计，编码器输入是**原始 spike event 序列**（连续时间戳），而非 binned counts：

```
input = [(t1, unit_3), (t2, unit_7), (t3, unit_3), ...]
```

每个 spike event 是一个 token，通过 RoPE 编码时间戳，通过 unit embedding 编码神经元 identity。

#### 典型长度选择

- **Phase 1 默认**：hold period（约 250–700ms，取决于 trial）
  - 优点：包含完整的 motor planning 信息
  - 缺点：长度可变，需要 padding 或截断

- **IDEncoder 参考窗口**（Phase 2）：2s（100 bins @ 20ms），M=20–50 个参考窗口
  - 更长的参考窗口提供更稳定的 firing rate 估计
  - 多窗口（M=20–50）通过平均或 attention pooling 降低单窗口噪声

#### 历史窗口长度对编码质量的影响

- **太短**（< 100ms）：上下文不足，encoder 难以推断当前的 population state
- **太长**（> 2s）：
  - 计算量增大（spike event 数量与时间成正比）
  - 远距离的 spike 可能与当前状态关联减弱（attention 稀释）
  - 但 RoPE 的相对位置编码在一定程度上缓解了长距离衰减
- **最佳范围**：250ms–1s，与 trial 结构对齐

### 6.4 跨 Session / 跨 Trial 数据统一使用

#### 核心挑战

| 挑战 | 描述 | 影响 |
|------|------|------|
| 神经元数量不同 | Session A 有 71 个神经元，Session B 有 45 个 | 模型需要处理可变大小的 population |
| 神经元 identity 不同 | 不同 session 的电极可能记录到不同的神经元 | 无法使用固定的 neuron embedding lookup |
| 发放率分布不同 | 不同 session 的 baseline firing rate 可能差异很大 | loss 权重可能不平衡 |
| Trial 数量不同 | 有的 session 100 trials，有的 300 trials | DataLoader 需要 balanced sampling |

#### 不同模型的解决方案

**POYO 方案（InfiniteVocabEmbedding Lookup）**：
- 每个（session, unit）对有一个可学习的 embedding
- 优点：简单、per-unit 表达能力强
- 缺点：新 session 的神经元没有 embedding → zero-shot 泛化失败

**SPINT 方案（IDEncoder）**：
- 从参考窗口的 binned spike counts 通过 MLP 推断 unit embedding
- 优点：gradient-free，新 session 只需前向传播
- 缺点：依赖 binned counts，丢失了 spike timing 信息

**NeuroHorizon 方案（IDEncoder Scheme B）**：
- 从参考窗口的 spike event tokens + RoPE 通过 attention pooling 推断 embedding
- 优点：保留 spike timing 信息，可能获得更好的表征
- 缺点：实现更复杂，计算量更大

#### Trial 间数据组织

每个 trial 是一个独立样本，包含：
- 输入：该 trial 的 hold period spike events
- 目标：该 trial 的 reach period binned spike counts
- 元数据：session ID、trial condition（reach target direction）、trial 时间戳

DataLoader 内不同 trial 混合：
- 同一 session 的 trials 自然混合
- 跨 session 训练时，建议 **session-balanced sampling**（每个 batch 均匀采样各 session），避免大 session 主导梯度

#### Session 间混合训练的注意事项

1. **Batch 构成**：建议每个 batch 包含来自多个 session 的 trials，让 IDEncoder 每步都看到多种神经元组合
2. **Loss 归一化**：不同 session 的神经元数量不同，需要注意 loss 是 per-neuron 平均还是 per-sample 平均
3. **数据不平衡**：如果某些 session 有更多 trials，需要通过 sampling weight 或 epoch 内循环平衡
4. **Evaluation split**：cross-session 评估必须确保 test session 的 **所有 trials** 都在 test set 中，不能有数据泄漏

### 6.5 数据组织对模型效果的影响

#### Bin Width 的影响

| 效应 | 太窄（如 1–5ms） | 适中（10–20ms） | 太宽（如 50–100ms） |
|------|-----------------|----------------|-------------------|
| 稀疏性 | 极端，>95% 为零 | 可控，70–90% 为零 | 较低，<60% 为零 |
| 序列长度 | 很长（1000ms→200–1000 tokens） | 适中（50–100 tokens） | 很短（10–20 tokens） |
| 时间精度 | 高（ms 级） | 中（20ms 级） | 低（50–100ms 级） |
| Poisson 近似 | 退化为 Bernoulli | 合理 | 开始偏离（overdispersion） |
| 计算成本 | O(T^2) attention 很贵 | 可接受 | 很低 |

**建议**：Phase 1 用 20ms，Phase 1 完成后做 10ms vs 20ms vs 50ms 敏感性测试。

#### History Length 的影响

- **太短**（< 100ms）：encoder 上下文不足，spike count 预测 R² 下降
- **适中**（250ms–1s）：覆盖 hold period，信息充分
- **太长**（> 2s）：计算量增大但边际收益递减，远距离 spike 的信息量有限

#### Trial Alignment 的影响

- **Trial-aligned（Scheme A）**：利用任务结构，hold→reach 因果关系明确，但泛化性受限
- **Sliding window（Scheme B）**：不依赖 trial 结构，数据量更大，但可能跨 trial 边界引入不连续性
- **实际影响**：对于有明确 trial 结构的数据（Perich-Miller），两者差异可能不大；对于连续记录数据（Allen、IBL），Scheme B 是必须的

#### Cross-Session Mixing 的影响

- **正面**：增加数据多样性，提高模型泛化能力
- **负面**：引入 neuron identity 问题，如果 IDEncoder 不够好，混合训练可能比单 session 训练更差
- **关键**：cross-session mixing 的收益取决于 IDEncoder 的质量——这也是为什么 Phase 2 的 IDEncoder 实验是整个项目的关键节点

### 6.6 主流模型的数据组织对比

| 模型 | Bin Width | Time Window | Cross-Session 策略 | Trial 处理 |
|------|-----------|-------------|-------------------|-----------|
| **NDT1** | 10ms | 250ms pre → 450ms post movement | 单 session | Trial-aligned（运动任务） |
| **NDT2** | 10ms | 类似 NDT1 | Multi-context pretraining（共享 encoder） | Trial-aligned |
| **NDT3** | 20ms | 连续序列 | Foundation model pretraining（2000h） | 连续 token stream |
| **POYO** | N/A（spike event） | 连续时间戳 | IVE per-session | Spike event tokenization |
| **POYO+** | N/A（spike event） | 连续时间戳 | IVE + multi-task | 同 POYO |
| **SPINT** | 20ms | 参考窗口 2s | IDEncoder gradient-free | FALCON 协议 |
| **LFADS** | 20ms | Trial duration | 单 session（或 stitching） | Trial-aligned |
| **Neuroformer** | N/A（逐 spike） | 连续时间戳 | Multi-session pretraining | 逐 spike autoregressive |

**关键观察**：

1. **Bin width 趋势**：早期工作（NDT1）用 10ms，近期工作（NDT3、SPINT、LFADS）倾向 20ms。POYO/Neuroformer 直接在 spike event 层面操作，跳过了 binning
2. **Cross-session 策略演化**：单 session → multi-session pretraining（NDT2）→ foundation model（NDT3）→ gradient-free（SPINT）
3. **Trial 处理趋势**：从 trial-aligned 向连续序列 / spike event 序列演化，更通用

#### NLB Benchmark 标准化格式

NLB 提供了 7 个标准化数据集，统一的数据格式包括：
- Spike trains（原始 spike timestamps）
- Binned spike counts（5ms bins）
- 行为数据（与 neural data 时间对齐）
- 标准化的 train/valid/test split（按 trial 和 neuron 两个维度划分）

#### FALCON Benchmark 设计

FALCON 专门评估跨 session 泛化，设计包括：
- **Held-in sessions**：完整数据发布，用于训练和 within-session 评估
- **Held-out sessions**：只发布 calibration splits（少量标注数据），用于 zero-shot / few-shot 评估
- 评估指标：R²（variance-weighted），报告 mean +/- std across sessions

### 6.7 对 NeuroHorizon 的建议

1. **当前方案合理**：20ms bin + hold/reach 分离是主流选择，与 SPINT、LFADS 一致
2. **Phase 1 完成后建议做 bin width 敏感性测试**：10ms vs 20ms vs 50ms，验证结果是否对 bin width 鲁棒
3. **Scheme A/B 对比是关键决策点**：500ms 窗口的 A/B 对比结果将决定 Phase 2+ 的数据组织方向
4. **跨 session 训练时使用 session-balanced sampling**：避免大 session 主导训练，确保 IDEncoder 看到均衡的神经元组合
5. **考虑向 NLB/FALCON 格式对齐**：如果后续计划在 benchmark 上评估，早期就按其格式组织数据可以减少后续工作量
6. **注意 Allen 数据的特殊性**：Allen Visual Coding 是连续记录（非 trial-based），必须用 Scheme B（sliding window），且刺激时间对齐需要特殊处理

---

---

## 7. 结果呈现与 Figure 规划

### 7.1 主流 Spike Foundation Model 论文的结果呈现方式

通过系统调研 NDT1/2/3、POYO/+、SPINT、Neuroformer、LFADS、MTM 共 9 篇核心论文，归纳出以下 figure 类型和呈现规律。

#### 7.1.1 各论文 Figure 概览

**NDT1**（Ye et al., 2021, 5 figures）：
- Fig 1：架构示意 + firing rate 线图
- Fig 4：predicted vs ground truth 线图对比（NDT vs AutoLFADS）；超参数扫描散点（likelihood vs R²）
- Fig 5：trial-averaged inferred rates；2D 轨迹预测图；kinematic R² vs 训练集大小（data efficiency 曲线）

**NDT2**（Ye et al., 2023, 5+ figures）：
- Fig 3：Bar chart + SEM error bar——多预训练条件下 NLL 和 R² 对比
- Fig 4：**Scaling curves**——R² 和 NLL vs 预训练数据量；pretrained vs from-scratch 收敛对比
- Fig 5：Offline 解码对比（0-shot / supervised / from-scratch）；online BCI 到达时间对比

**NDT3**（Ye et al., 2025, 5+ figures）：
- Fig 1B：**Crossover line plot**——pretrained vs from-scratch 随数据量交叉（约 1.5h 处）
- Fig 3：多尺度评估，**p-value heatmap**（FDR 校正的 pairwise t-test）；模型规模对比（45M vs 350M）
- Fig 4：跨 session/subject 泛化失败分析；PCA-LDA 可视化
- Fig 5：分布偏移鲁棒性（temporal / posture / spring load）的散点图 + 边际直方图

**POYO**（Azabou et al., 2023, 5 figures）：
- Fig 2：true vs predicted 轨迹叠加；scatter plot（单 session vs POYO-mp R²）
- Fig 3：**Scaling curves**——R² vs 模型深度（L=6/14/26）和数据量
- Fig 4：sample/compute efficiency 曲线（held-out session 和 new animal）
- Fig 5：PCA of session embeddings（按数据来源和任务着色）

**POYO+**（Azabou et al., 2025, 4+ figures）：
- Fig 2：Scaling curve——性能 vs session 数
- Fig 3：多任务收益 bar chart（POYO+ vs baselines）
- Fig 4：**UMAP/PCA latent embeddings** 按 stimulus/Cre-line 着色；**confusion matrix**（脑区/细胞类型解码）；层次聚类树状图

**SPINT**（Le et al., 2025, 4 figures）：
- Fig 3：**Bar chart + error bar**——跨 session R² vs 校准 trial 数/训练天数/population 减少比例（FALCON M1/M2/H1）
- Fig 4：消融实验 bar chart——位置编码策略和 dynamic channel dropout 贡献

**Neuroformer**（Antoniades et al., 2024, 13 figures）：
- Fig 2：**Raster plot**（predicted vs true spikes）+ 连接性热力图（attention-derived vs ground truth）
- Fig 4：速度预测 scatter plot
- Fig 5：Few-shot 学习曲线（pretrained vs non-pretrained）
- Fig 6：Progressive ablation scatter——逐步添加组件后性能提升
- Fig 9：连接性矩阵对比（attention vs partial correlation vs Granger causality vs ground truth）

**LFADS**（Pandarinath et al., 2018, 6 figures）：
- Fig 2：condition-averaged + single-trial rates 线图；**t-SNE** of initial conditions（按 target angle 着色）；kinematic trajectory 叠加；R² 对比
- Fig 3：**jPCA** 旋转动力学可视化
- Fig 4：44-session dynamic neural stitching——跨 session factor trajectory 一致性
- Fig 5：推断输入（inferred inputs）捕捉 perturbation 时间和方向

**MTM**（Multi-Task-Masking, 2024, 9 figures）：
- Fig 2：trial-averaged raster map + scatter（bps 和 behavior decoding 对比）
- Fig 4：**Scaling curves**——co-smoothing/forward prediction 等指标 vs 预训练 session 数
- Fig 5：脑区级别 heatmap——choice accuracy 和 whisker motion 按脑区着色
- Fig 7：单神经元分析 scatter——bps / PSTH R² / trial-average R² 按 firing rate 着色

#### 7.1.2 跨论文通用 Figure 类型（出现频率排序）

| 排名 | Figure 类型 | 出现论文数 | 代表示例 | 必要性 |
|------|-----------|-----------|---------|--------|
| 1 | 架构示意图 | **9/9** | 所有论文 Fig 1 | **必须** |
| 2 | Bar chart + error bar（方法对比） | **7/9** | NDT2 Fig 3, SPINT Fig 3 | **必须** |
| 3 | Scaling curves（线图） | **7/9** | NDT3 Fig 3, POYO Fig 3, MTM Fig 4 | **必须**（对 foundation model） |
| 4 | Scatter plot（pairwise 方法对比） | **6/9** | NDT1 Fig 4b, MTM Fig 3 | **推荐** |
| 5 | 预测 vs 真实轨迹叠加 | **5/9** | POYO Fig 2, LFADS Fig 2 | **推荐**（解码任务） |
| 6 | Firing rate 时间序列线图 | **5/9** | NDT1 Fig 5a, LFADS Fig 2 | **推荐** |
| 7 | Latent embedding 可视化（t-SNE/PCA/UMAP） | **4/9** | LFADS Fig 2, POYO+ Fig 4 | **推荐** |
| 8 | Raster plot（spike activity） | **3/9** | LFADS Fig 1, Neuroformer Fig 2 | 可选 |
| 9 | 消融实验 bar chart | **3/9** | SPINT Fig 4, Neuroformer Fig 6 | **推荐** |
| 10 | 统计显著性矩阵/热力图 | **2/9** | NDT3 Fig 3E, POYO+ Fig 4B | 可选但加分 |
| 11 | Few-shot / data efficiency 曲线 | **3/9** | NDT3 Fig 1B, Neuroformer Fig 5 | **推荐** |
| 12 | Confusion matrix | **2/9** | POYO+ Fig 4F | 可选 |

#### 7.1.3 论文中特别有效的可视化手法

1. **NDT3 Crossover Plot（Fig 1B）**：一条简洁的线图，展示 pretrained 与 from-scratch 在约 1.5h 数据处交叉——直观传达 foundation model 的价值主张。NeuroHorizon 可借鉴此手法展示 IDEncoder 预训练 vs 从头训练的交叉点。

2. **NDT3 p-value Heatmap（Fig 3E）**：FDR 校正的 pairwise t-test 热力图，比单纯的 error bar 更严谨。NeuroHorizon 在 Phase 5 论文中应考虑类似的统计检验矩阵。

3. **POYO+ Confusion Matrix + Dendrogram（Fig 4）**：展示 latent space 自动捕获脑区和细胞类型层级结构——说明 representation 不仅能解码，还蕴含生物学语义。NeuroHorizon 在 Phase 4 可类比展示 latent 是否区分视觉/运动皮层。

4. **Neuroformer 连接性推断（Fig 9）**：将 attention 权重与已知连接性对比——独特的可解释性分析。NeuroHorizon 可考虑类似的 attention pattern 分析。

5. **LFADS jPCA 旋转动力学（Fig 3）**：在 jPCA 空间展示单 trial 动力学——计算神经科学领域的标志性可视化，适合 NeuroHorizon 的 latent dynamics 展示。

6. **MTM 脑区级 Heatmap（Fig 5）**：按脑区着色的解码性能 scatter——NeuroHorizon Phase 4 的 Delta_m 条件分解可参考此风格。

---

### 7.2 NeuroHorizon 当前的 Figure 规划现状

根据 proposal_review.md 和 plan.md 的梳理，当前各 Phase 已规划/隐含的分析输出如下：

#### Phase 0（已完成）

| 已规划 | 类型 | 状态 |
|--------|------|------|
| 数据集概览统计图 | Multi-panel（神经元数、trial 时长、发放率分布） | 已完成 |
| 神经统计 + 自回归可行性图 | Multi-panel（spike count 分布、hold/reach 统计） | 已完成 |
| Latent PCA 图 | PCA scatter | 已完成 |
| **输出位置** | `results/figures/data_exploration/` | |

#### Phase 1

| 已规划 | 类型 | 来源 |
|--------|------|------|
| Poisson NLL 训练曲线 | Line plot（loss vs epoch） | proposal_review §二 |
| Error Decay Curve | Line plot（PSTH correlation vs AR step） | proposal_review §二, plan 1.3.4 |
| Spike count 分布验证 | Histogram（predicted vs GT） | proposal_review §二 |
| 预测窗口性能衰减曲线 | Line plot（指标 vs 窗口长度 250/500/1000ms） | plan 1.3.4 |

#### Phase 2

| 已规划 | 类型 | 来源 |
|--------|------|------|
| IDEncoder embedding t-SNE/PCA | Scatter/cluster plot | proposal_review §三 |
| Zero-shot R² 对比 | Bar chart（IDEncoder vs IVE lookup） | proposal_review §三 |

#### Phase 3

| 已规划 | 类型 | 来源 |
|--------|------|------|
| Scaling Curve | Line plot + error bar（session 数 vs R²） | proposal_review §四, plan 3.1.3 |
| Transfer Learning 对比 | Multi-line（from-scratch vs frozen vs fine-tuned） | proposal_review §四 |
| Few-shot 曲线 | Line plot（% 标注数据 vs R²） | proposal_review §四 |

#### Phase 4

| 已规划 | 类型 | 来源 |
|--------|------|------|
| 模态贡献消融矩阵 | Heatmap（Δ_m） | proposal_review §五 |
| 条件分解 Δ_m(v) | Bar/heatmap（按脑区/刺激类型/行为状态） | proposal_review §五 |

#### Phase 5（论文）

| 已规划 | 类型 | 来源 |
|--------|------|------|
| 完整预测窗口矩阵 | 表格/multi-bar | plan 5.1 |
| 跨 Session 泛化核心图 | 综合对比图 | plan 5.1 |
| Scaling Law 图 | Line plot | plan 5.1 |
| 消融实验矩阵 | Bar chart 组 | plan 5.2 |
| 模态归因图 | Heatmap/bar | plan 5.1 |

---

### 7.3 Gap 分析：当前规划与论文级标准的差距

将 §7.1 中的"标准 figure 集"与 §7.2 中的当前规划对比，识别以下缺口：

#### 已覆盖（无需新增）

| 标准 Figure | 当前覆盖情况 |
|-------------|-------------|
| 架构示意图 | Phase 5 论文阶段会绘制 |
| Scaling curve | Phase 3 已规划 |
| 消融实验 bar chart | Phase 5 已规划（decoder 深度、IDEncoder、causal mask 等） |
| Latent embedding 可视化 | Phase 2 IDEncoder t-SNE 已规划 |

#### 缺失或不足（建议补充）

| # | 缺失 Figure | 重要性 | 对应 Phase | 建议 |
|---|------------|--------|-----------|------|
| 1 | **方法对比 bar chart（vs 外部 baseline）** | **必须** | Phase 1/2/5 | 当前只有自身消融，缺少与 NDT/POYO/SPINT/Neuroformer 的直接 bar chart 对比。Phase 5 论文必须有一张综合性 bar chart，横轴为各方法，纵轴为 R²/Poisson NLL/bits per spike |
| 2 | **Predicted vs actual firing rate 时间序列** | **推荐** | Phase 1 | 展示 2-3 个代表性神经元的预测 vs 真实 firing rate 对比线图（trial-averaged）。直观展示自回归生成的时序质量。5/9 论文包含此类图 |
| 3 | **Predicted vs actual behavior trajectory** | **推荐** | Phase 3 | 2D 轨迹叠加图（解码的手臂速度/位置 vs 真实），5/9 论文使用此类图提供直观证据 |
| 4 | **Data efficiency / crossover 曲线** | **推荐** | Phase 3 | 类似 NDT3 Fig 1B 的 pretrained vs from-scratch crossover 图。当前 few-shot 曲线部分覆盖，但缺少明确的 crossover 分析 |
| 5 | **统计显著性检验可视化** | **可选但加分** | Phase 5 | p-value heatmap 或显著性标注。当前验收标准只要求"3 seeds, variance < 0.05"，但论文级呈现应包含 FDR 校正的统计检验 |
| 6 | **Raster plot（predicted vs true spikes）** | **可选** | Phase 1 | 类似 Neuroformer Fig 2 的 raster 对比，展示单 trial 预测精度。当前只有 spike count 分布验证，缺少时序层面的视觉证据 |
| 7 | **Attention pattern / 连接性分析** | **可选但创新** | Phase 4 | 类似 Neuroformer Fig 9，分析自回归 decoder 的 cross-attention 权重是否捕捉了神经元间的功能连接 |
| 8 | **Error distribution 分析** | **可选** | Phase 1 | 预测误差的时空分布——哪些神经元预测更准/更差，是否与 firing rate 相关（类似 MTM Fig 7 的单神经元分析） |
| 9 | **跨 session embedding 一致性** | **推荐** | Phase 2 | 类似 POYO Fig 5D 的 session embedding PCA——展示不同 session 在 embedding 空间的组织方式（按动物/任务着色） |
| 10 | **Confusion matrix（跨 session）** | **可选** | Phase 2/4 | 类似 POYO+ Fig 4F，展示 IDEncoder 推断的 embedding 是否能正确区分不同功能类型的神经元 |

---

### 7.4 各 Phase 建议的完整 Figure 清单

基于当前规划和 gap 分析，以下是各 Phase 建议的 figure 清单。标注 **[已规划]** / **[建议新增]** / **[论文必须]**。

#### Phase 1：自回归改造验证

| # | Figure | 类型 | 目的 | 状态 |
|---|--------|------|------|------|
| 1.1 | 训练收敛曲线 | Line plot（train/val loss vs epoch） | 验证 Poisson NLL 收敛 | [已规划] |
| 1.2 | Error Decay Curve | Line plot（PSTH corr vs AR step, 多窗口叠加） | 诊断自回归误差累积 | [已规划] |
| 1.3 | 预测窗口对比 | Grouped bar chart（250/500/1000ms × R²/NLL/PSTH corr） | 确定最优预测窗口 | [已规划] |
| 1.4 | TF vs AR 性能对比 | Bar chart（Teacher Forcing vs Autoregressive） | 量化 exposure bias | [已规划，隐含] |
| 1.5 | **Predicted vs true firing rate 时间序列** | Multi-panel line plot（3-5 个代表性神经元） | 直观展示预测质量 | **[建议新增]** |
| 1.6 | **Predicted vs true spike raster** | Raster plot（selected trials, 对比面板） | 展示单 trial 时序精度 | **[建议新增]** |
| 1.7 | Spike count 分布验证 | Histogram（predicted vs GT 的 count 分布） | 验证分布合理性 | [已规划] |
| 1.8 | AR vs Non-AR 消融 | Bar chart（causal vs parallel decoder） | 验证自回归必要性 | [已规划] |
| 1.9 | **Per-neuron R² 分布** | Histogram / violin plot | 分析哪些神经元预测好/差 | **[建议新增]** |
| 1.10 | **Baseline 对比 bar chart** | Bar chart（PSTH / Linear / Smoothed / NeuroHorizon） | 展示模型相对 baseline 的增量 | **[建议新增]** |

**Phase 1 Figure 呈现建议**：

- **Fig 1.5 的具体做法**：选 3-5 个不同 firing rate 的神经元（高/中/低活跃），trial-averaged 后画 predicted rate（红线）vs ground truth rate（黑线）± trial-to-trial std（灰色 shaded band），x 轴为时间（0-1000ms），y 轴为 firing rate。LFADS Fig 2 和 NDT1 Fig 5a 是此类图的经典范例。
- **Fig 1.2 Error Decay Curve 的增强**：建议在同一图中叠加多条曲线——250ms/500ms/1000ms 窗口，以及 TF baseline（水平虚线），和 Non-AR parallel 的 decay（用于对比）。NDT3 的 multi-condition 叠加风格可借鉴。
- **Fig 1.6 的具体做法**：左列为 ground truth raster，右列为 model-generated raster，每列 10 个 trials，y 轴为 neuron index（按 peak activity 时间排序），x 轴为时间。LFADS Fig 1 和 Neuroformer Fig 2 的 raster 对比是参考范例。

#### Phase 2：跨 Session 泛化

| # | Figure | 类型 | 目的 | 状态 |
|---|--------|------|------|------|
| 2.1 | IDEncoder embedding t-SNE/PCA | Scatter plot（按 session/animal/function 着色） | 验证 embedding 质量 | [已规划] |
| 2.2 | Zero-shot R² 对比 | Bar chart（IDEncoder vs IVE lookup vs SPINT） | IDEncoder 核心验证 | [已规划] |
| 2.3 | **跨 session pairwise scatter** | Scatter plot（x=within-session R², y=zero-shot R²） | 量化 zero-shot 泛化 gap | **[建议新增]** |
| 2.4 | **Session embedding 空间** | PCA scatter（每个 session 一个点，按 animal 着色） | 类似 POYO Fig 5D | **[建议新增]** |
| 2.5 | Scheme A vs Scheme B 对比 | Bar chart | 决定 IDEncoder tokenization | [已规划] |
| 2.6 | **Per-session R² bar chart** | Grouped bar chart（每个 session 一组，IDEncoder vs IVE） | 展示逐 session 泛化稳定性 | **[建议新增]** |
| 2.7 | 跨动物泛化 | Bar chart（同动物 vs 跨动物 R²） | 量化跨动物 degradation | [已规划] |

**Phase 2 Figure 呈现建议**：

- **Fig 2.1 t-SNE 的着色策略**：建议制作 2 个版本——(a) 按 session ID 着色（验证同 session 神经元 clustering）；(b) 按功能属性着色（如 preferred direction），验证跨 session 功能相似的神经元确实靠近。POYO+ Fig 4 的多着色策略是良好范例。
- **Fig 2.3 Pairwise scatter**：类似 MTM Fig 3 的风格，x 轴为 within-session baseline R²，y 轴为 IDEncoder zero-shot R²，每个点是一个 held-out session，对角线为等效线。点在对角线以下的程度直观展示 zero-shot gap。
- **Fig 2.6 重要性**：SPINT Fig 3 通过逐 dataset（M1/M2/H1）展示结果，POYO Fig 2 通过逐 dataset scatter 展示结果——reviewer 期望看到逐 session 而非仅平均值的结果。

#### Phase 3：数据 Scaling + 迁移学习

| # | Figure | 类型 | 目的 | 状态 |
|---|--------|------|------|------|
| 3.1 | **Scaling Curve（核心）** | Line plot + error bar（log-x, session 数 vs R²） | Foundation model 核心论证 | [已规划] |
| 3.2 | Transfer Learning 三级对比 | Multi-line plot（from-scratch / frozen / fine-tuned） | 预训练价值 | [已规划] |
| 3.3 | Few-shot 曲线 | Line plot（% 标注 vs R², 多条线） | 数据效率 | [已规划] |
| 3.4 | **Crossover 分析** | Line plot（pretrained vs from-scratch vs session 数交叉） | 类似 NDT3 Fig 1B | **[建议新增]** |
| 3.5 | **Power-law 拟合** | Log-log plot + 拟合线 | 量化 scaling exponent | **[建议新增]** |
| 3.6 | **Predicted vs actual trajectory** | 2D trajectory overlay | 直观解码质量 | **[建议新增]** |

**Phase 3 Figure 呈现建议**：

- **Fig 3.1 Scaling Curve 规范**：建议使用 semi-log 或 log-log 坐标，至少 5 个数据点（5/10/20/40/70 sessions），每个点 3 seeds 的 mean ± std。POYO Fig 3 和 NDT2 Fig 4 是经典参考。如果同时有 Brainsets 和 IBL 数据，建议在同一图中用不同颜色/标记展示。
- **Fig 3.4 Crossover 分析**：NDT3 Fig 1B 是最有影响力的 foundation model figure 之一——一条线展示 pretrained 在小数据时大幅领先 from-scratch，随数据增加差距缩小。建议 NeuroHorizon 在此图中同时画 frozen transfer 和 fine-tuned transfer 两条线，与 from-scratch 交叉。
- **Fig 3.5 Power-law**：fitting R² = a * N^b + c，报告 scaling exponent b 及其置信区间，与 POYO+ 报告的 scaling exponent 对比。

#### Phase 4：多模态融合

| # | Figure | 类型 | 目的 | 状态 |
|---|--------|------|------|------|
| 4.1 | 模态贡献消融矩阵 | Heatmap（Neural/+Beh/+Img/+Both × 指标） | 量化各模态贡献 | [已规划] |
| 4.2 | 条件分解 Δ_m(v)——按脑区 | Grouped bar chart（V1/LM/AM × Δ_image/Δ_beh） | 脑区特异性 | [已规划] |
| 4.3 | 条件分解 Δ_m(v)——按刺激 | Bar chart（Natural Movies vs Gratings） | 刺激依赖性 | [已规划] |
| 4.4 | **Predicted vs true firing rate（视觉区域）** | Multi-panel line plot | 展示多模态预测质量 | **[建议新增]** |
| 4.5 | **脑区级 heatmap** | Brain region × metric 热力图 | 类似 MTM Fig 5 | **[建议新增]** |
| 4.6 | **Cross-attention pattern 分析** | Attention weight 可视化 | 可解释性创新 | **[建议新增]** |
| 4.7 | **DINOv2 stimulus embedding PCA** | Scatter plot + 样本图像 | 验证视觉特征质量 | **[建议新增]** |

**Phase 4 Figure 呈现建议**：

- **Fig 4.1 消融矩阵的呈现方式**：建议用 grouped bar chart 而非纯 heatmap——横轴为 4 种条件（Neural only / +Behavior / +Image / +Both），每组内 2-3 个指标（R²/NLL/PSTH corr），附 error bar。Neuroformer Fig 6 的 progressive addition scatter 也可参考。
- **Fig 4.5 脑区级 heatmap**：类似 MTM Fig 5，每行一个脑区（V1/LM/AM 等），每列一个模态条件，颜色深浅表示 Δ_m 大小。预期 V1 行的 Δ_image 列颜色最深。
- **Fig 4.6 Attention 分析**：这是潜在的创新亮点。分析自回归 decoder 的 cross-attention 权重：当加入视觉刺激后，V1 神经元是否更多地 attend 到 image tokens？这种分析在 Neuroformer 中已有先例（Fig 3 spatial attention overlaid on stimulus），NeuroHorizon 可做类似的模态-attention 归因。
- **Fig 4.7 DINOv2 PCA**：在 PCA 空间展示 118 张 natural scene 的 embedding，每个点标注缩略图，验证语义相近的图像在 embedding 空间也相近。这为后续分析（Δ_image 与图像语义的关联）提供基础。

#### Phase 5：论文级 Figure 总览

综合以上各 Phase 的 figure，Phase 5 论文需要的核心 figure 集：

| 优先级 | Figure | 对应 Phase 结果 | 论文位置 |
|--------|--------|---------------|---------|
| **必须** | 架构示意图 | 全局 | Main Fig 1 |
| **必须** | Scaling curve | Phase 3 Fig 3.1 | Main Fig 2 or 3 |
| **必须** | 方法对比 bar chart（vs 外部 SOTA） | Phase 1/2 综合 | Main Fig 3 or 4 |
| **必须** | 消融实验总图 | Phase 1/5 消融汇总 | Main Fig 4 or 5 |
| **必须** | IDEncoder embedding 可视化 | Phase 2 Fig 2.1 | Main Fig |
| **推荐** | Error Decay Curve | Phase 1 Fig 1.2 | Main or Supp |
| **推荐** | Predicted vs actual firing rate | Phase 1 Fig 1.5 | Main or Supp |
| **推荐** | Transfer learning / few-shot 曲线 | Phase 3 Fig 3.2-3.3 | Main Fig |
| **推荐** | Predicted vs actual trajectory | Phase 3 Fig 3.6 | Main or Supp |
| **推荐** | 模态贡献分析 | Phase 4 Fig 4.1-4.3 | Main Fig（如论文包含 Phase 4） |
| **可选** | p-value heatmap | Phase 5 统计检验 | Supp |
| **可选** | Raster plot 对比 | Phase 1 Fig 1.6 | Supp |
| **可选** | Per-neuron R² 分析 | Phase 1 Fig 1.9 | Supp |
| **可选** | Attention pattern 分析 | Phase 4 Fig 4.6 | Supp 或创新亮点 |

---

### 7.5 对 NeuroHorizon 的建议

#### 7.5.1 最高优先级补充

1. **Phase 1 新增 predicted vs true firing rate 时间序列图（Fig 1.5）**：这是 5/9 论文使用的标准可视化，实现简单（trial-averaged 后画线图），但提供了 bar chart/数值无法传达的时序质量直观感受。建议在每次预测窗口实验完成后都生成此图。

2. **Phase 1 新增 baseline 对比 bar chart（Fig 1.10）**：当前 Phase 1 的验收标准只有"收敛 + R²"，缺少与内部 baseline（PSTH prediction / linear regression）的对比图。这是几乎所有论文的标配。

3. **Phase 2 新增 per-session R² bar chart（Fig 2.6）**：reviewer 期望看到逐 session 结果，而非仅平均值。SPINT 和 POYO 都按 dataset/session 展示结果。

#### 7.5.2 中等优先级补充

4. **Phase 3 新增 crossover 分析（Fig 3.4）**：NDT3 Fig 1B 是近年最有影响力的 foundation model 可视化之一，建议 NeuroHorizon 也做此分析。

5. **Phase 4 新增 attention 模式分析（Fig 4.6）**：如果 NeuroHorizon 声称多模态融合提供了可解释性（创新点四），那么 attention 可视化是最直接的证据。

6. **Phase 5 新增统计显著性可视化**：至少在论文 supplementary 中包含 pairwise t-test 或 Wilcoxon signed-rank test 的结果矩阵。

#### 7.5.3 实施建议

- **建议建立统一的绘图工具模块**：如 `scripts/analysis/plot_utils.py`，定义标准配色方案、字体大小、figure 尺寸等，保证所有 figure 风格一致。推荐使用 matplotlib + seaborn，论文投稿时考虑 Nature/NeurIPS 的排版宽度（单栏 3.25 英寸，双栏 6.75 英寸）。

- **建议每个 Phase 完成时生成完整 figure 集**：不要等到 Phase 5 才开始绘图——每个 Phase 的实验结果应立即可视化，存放于 `results/figures/phase{N}/` 目录。这既有助于实时监控进展，也为论文积累素材。

- **建议配色方案**：
  - NeuroHorizon 主色：蓝色系（#1f77b4）
  - Baseline/对照：灰色系（#7f7f7f）
  - 外部 SOTA 对比：各自固定颜色（如 POYO=#ff7f0e, SPINT=#2ca02c, NDT=#d62728）
  - 模态着色：Neural=蓝色, Behavior=绿色, Image=橙色

- **建议统一的 error bar 策略**：所有需要 error bar 的 figure 使用 mean ± SEM（standard error of the mean），minimum 3 random seeds。如果 seed 数量足够（≥5），也可用 95% CI。在 figure caption 中明确标注 error bar 含义。


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

- Karpowicz, B., et al. (2024). NEDS: Neural Embedding for Data Sharing. *NeurIPS 2024*.
- Antoniades, A., et al. (2024). Neuroformer: Multimodal and Multitask Generative Pretraining for Brain Data. *ICLR 2024*.
- Pei, F., et al. (2021). Neural Latents Benchmark (NLB). *NeurIPS 2021 Datasets and Benchmarks*.
- Karpowicz, B., et al. (2024). FALCON Benchmark: Standardized Cross-Session Neural Decoding. *NeurIPS 2024 Datasets and Benchmarks*.
- Hurwitz, C., et al. (2024). LDNS: Latent Diffusion for Neural Spike Data. *NeurIPS 2024*.
- Williams, R. J. & Zipser, D. (1989). A Learning Algorithm for Continually Running Fully Recurrent Neural Networks. *Neural Computation*.

*文档创建时间：2026-03-02*
*最后更新：2026-03-02*
