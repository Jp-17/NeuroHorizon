# Option D 展开：隐空间动力学模型与扩散模型解码器的详细实施方案

> 日期：2026-03-16
> 项目：NeuroHorizon
> 背景：基于 neurips_innovation_claude.md 中 Option D 的展开，结合 POSSM、LDNS 等前沿工作
> 目标：形成两个可直接执行的技术方案 + 补充实验清单

---

## 第一部分：方案 1 — 隐空间动力学模型（Latent Dynamics Decoder）

### 1.1 方案概述与动机

**核心思路**：保留 POYO+ Perceiver encoder 不变，将当前的 AutoregressiveDecoder（causal self-attention + optional prediction memory）替换为隐空间动力学模型（Latent Dynamics Model），在低维连续空间中进行时间演化预测。

**理论基础**：

大量实验证据表明，M1 运动皮层的神经群体活动在低维流形上演化（Cunningham & Yu 2014）。特别是在 center-out reaching 任务中，M1 群体活动呈现准线性的旋转动力学模式（Churchland et al. 2012），可用 10–30 维的 latent state 良好描述。这意味着：

1. **在 latent space 做预测比在 observation space 做预测天然更合理** — 几百个神经元的 spike count vector 中大部分变异性来自 Poisson 噪声，而真正可预测的信号集中在低维流形上
2. **误差累积问题大幅缓解** — 在 D_latent=16–32 维空间中外推，比在 N=407 维 count space 中做 AR 反馈，噪声放大效应小得多
3. **物理可解释性** — latent dynamics 对应真实的 neural population dynamics，具有神经科学意义

**与 LFADS 的本质区别**：

| 方面 | LFADS (Nature Methods 2018) | 本方案 (NeuroHorizon-LD) |
|---|---|---|
| Encoder | Sequential VAE（per-session） | POYO+ Perceiver（session-agnostic） |
| 推理 | 双向（需完整 trial） | 因果（实时可用） |
| 跨 session | 不支持，需重新训练 | 支持（POYO+ 的 unit embedding 机制） |
| 动力学 | RNN generator + controller | SSM（GRU/Mamba/S4D）|
| 训练目标 | ELBO（重建 + KL） | Poisson NLL（直接优化预测） |
| 输出 | Latent trajectories（需后处理）| 直接预测 spike counts |

**创新叙事**："Foundation model meets neural dynamics — 将 POYO+ 的 session-agnostic spike encoder 与 learned latent dynamical system 结合，实现跨 session 的长时程神经活动预测。"

### 1.2 POSSM 启发的 SSM 引入

POSSM（POYO-SSM，NeurIPS 2025，arXiv:2506.05320）是最直接的参考。它将 POYO 的 spike tokenization + cross-attention encoder 与 recurrent SSM backbone 结合，实现了实时神经解码。但 POSSM 做的是 **behavior decoding**（spikes→behavior），而本方案做的是 **forward prediction**（past spikes→future spikes），这带来一个关键差异：

- **POSSM**：SSM 的每一步都接收新的 spike chunk 作为输入（h_t = f(z_t, h_{t-1})），是 **有输入驱动** 的
- **本方案**：SSM 需要在没有新 spike 输入的情况下 **自主演化**（h_t = f(h_{t-1}, c)），是 **autonomous dynamics**

这意味着不能简单照搬 POSSM 的架构，需要针对 autonomous dynamics 做特殊设计。

#### 1.2.1 GRU 变体

**POSSM 中的表现**：GRU 在 NHP reaching 任务上表现最佳（R²=0.9587），优于 S4D 和 Mamba。

**适配设计**：
```
标准 GRU: h_t = GRU(x_t, h_{t-1})  -- 需要输入 x_t
Autonomous GRU: h_t = GRU(h_{t-1}, c)  -- 无外部输入
```

具体实现：
- 将 GRU 的输入 gate 的外部输入替换为 **条件向量 c**（从 encoder latents 提取的全局上下文）
- c 通过 attention pooling 或 mean pooling 从 S=32 个 encoder latent tokens 中获得
- 每一步：`h_t = GRU(c, h_{t-1})` 或 `h_t = GRU(proj(h_{t-1}), h_{t-1})`（将 previous hidden state 的投影作为 "pseudo-input"）

**优势**：实现最简单，POSSM 已验证有效，训练稳定
**风险**：GRU 的 hidden state 维度固定，可能限制 latent space 的表达能力

#### 1.2.2 Mamba 变体

**POSSM 中的表现**：参数量最大，NHP 上表现好但不如 GRU。

**适配设计**：
- Mamba 的核心是 **input-dependent state transition**（选择性 SSM），每步需要输入来调节 state transition
- Forward prediction 模式：将 previous step 的 **decoded output**（或 hidden state 的投影）作为 Mamba 的输入
- 这实际上构成了 **latent space 中的自回归**：z_t → Mamba → z_{t+1}
- 与 observation space AR 的关键区别：z 是 D_latent=16–32 维的连续向量，而非 N=407 维的 discrete count vector

**优势**：O(n) 复杂度，长序列建模能力强，input-dependent gating 可能捕获更丰富的 dynamics
**风险**：实现较复杂，需要从 torchmamba 或 mamba-ssm 引入依赖

#### 1.2.3 S4D 变体

**POSSM 中的表现**：参数最少，在 human handwriting 跨物种迁移中表现最好。

**适配设计**：
- S4D 是对角化线性 SSM：`x_{k+1} = diag(A) * x_k + B * u_k`
- 对 autonomous dynamics：`x_{k+1} = diag(A) * x_k`（纯线性演化）
- 条件注入：将 encoder latents 作为初始状态 x_0

**优势**：
- 线性动力学可能 **完全足够** 描述 M1 reaching dynamics（Churchland 旋转动力学是准线性的）
- 训练最稳定，参数最少
- 理论可分析（eigenvalues of A 对应 latent dynamics 的 time constants）

**风险**：如果 dynamics 有显著非线性成分（如 reach initiation/termination），线性 S4D 可能不够
**建议**：**作为起始点优先实验**。如果线性不够，再升级到 GRU/Mamba

#### 1.2.4 三种变体的对比总结

| 维度 | S4D | GRU | Mamba |
|---|---|---|---|
| 复杂度 | 最低 | 中 | 最高 |
| 非线性表达 | 线性（有限） | 门控非线性 | 选择性非线性 |
| Autonomous dynamics 适配 | 天然支持 | 需改造 | 需改造 |
| POSSM 中最佳 | 跨物种迁移 | NHP reaching | 参数大时 |
| 推荐优先级 | **1（首选）** | 2 | 3 |

### 1.3 具体架构设计

```
History spikes (0 – T_obs, e.g. 500ms)
    |
    v
POYO+ Encoder (unchanged)
    |
    v
Latent tokens [B, S=32, D=128]
    |
    v
Attention Pooling: K=4 learned queries attend to S=32 tokens
    |
    v
z_0 [B, K=4, D_pool=128] --> flatten --> [B, K*D_pool=512]
    |                                         |
    v (Strategy A: per-token)           v (Strategy B: global, 推荐)
K independent SSMs                    Single SSM, hidden_dim=512
z_0^k -> z_1^k -> ... -> z_T^k      z_0 -> z_1 -> ... -> z_T
    |                                         |
    v                                         v
pool/concat -> [B, T, D=128]         Linear proj -> [B, T, D=128]
    |
    v
PerNeuronMLPHead (unchanged)
    |
    v
log_rate [B, T, N_units]
```

**组件详解**：

**Attention Pooling Layer**（新增）：
- K=4 个 learned query tokens [K, D]
- Multi-head attention (num_heads=4): queries attend to encoder latent tokens
- 输出：K 个 pooled tokens [B, K, D]，每个 token 捕获 encoder latent 的不同方面
- 参数量：约 K*D*3 (QKV projections) ≈ 200K

**Dynamics Model**（新增）：
- 输入维度：K*D = 4*128 = 512（Strategy B）
- Hidden 维度：512（与输入相同）
- 对每个 prediction bin (20ms)，SSM 前进一步
- 输出维度：512 per step

**Step Projection**（新增）：
- Linear(512, 128)：将 SSM hidden state 映射为 bin representation
- 这样 PerNeuronMLPHead 可以直接复用（期望输入 dim=128）

**PerNeuronMLPHead**（不变）：
- bin_proj(bin_repr) → [B,T,64]
- unit_proj(unit_embs) → [B,N,64]
- concat → MLP → log_rate [B,T,N]

**预估参数量**：
- Attention pooling: 约 200K
- S4D dynamics (512 hidden): 约 50K
- GRU dynamics (512 hidden): 约 800K
- Mamba dynamics (512 hidden): 约 500K
- Step projection: 约 65K
- 总计（含 encoder + head）: 约 2.3–2.9M（与当前 2.1M 相当）

### 1.4 训练策略

**推荐：两阶段训练**

**Phase 1：冻结 encoder + head，训练 dynamics 组件**
```
冻结: POYO+ encoder (所有参数), PerNeuronMLPHead
训练: Attention Pooling, Dynamics Model, Step Projection
Loss: Poisson NLL（与 baseline_v2 相同）
Optimizer: AdamW, lr=1e-3, weight_decay=1e-4
Schedule: OneCycleLR, pct_start=0.3
Epochs: 200
```

**Phase 2：端到端微调**
```
训练: 全部参数
Loss: Poisson NLL
Optimizer: AdamW, lr=1e-4（Phase 1 的 1/10）
Schedule: CosineAnnealingLR
Epochs: 100–200
```

**为什么两阶段**：
1. POYO+ encoder 在 behavior decoding 上已经学到了好的 spike representation，Phase 1 让 dynamics model 学会在这个 representation 上做外推
2. Phase 2 的微调让 encoder 适应 forward prediction 任务的特殊需求（可能需要编码更多 dynamics-relevant 信息）
3. 避免一开始就端到端训练导致 encoder 表征被破坏

### 1.5 预期性能分析

| 预测窗口 | baseline_v2 fp-bps | 预期 NeuroHorizon-LD fp-bps | 预期变化 |
|---|---|---|---|
| 250ms | 0.2115 | 0.20–0.22 | 持平（短窗口两种方法差异不大） |
| 500ms | 0.1744 | 0.18–0.20 | +3–15%（dynamics 外推优于 parallel） |
| 1000ms | 0.1317 | 0.15–0.18 | +14–37%（最大收益区间） |

**乐观情景**：如果 encoder latents 包含丰富的 dynamics 信息，且 S4D 的线性动力学足以描述 reaching dynamics，则 1000ms 可能达到 0.18+ fp-bps。

**悲观情景**：如果 encoder latents 主要编码 "当前快照" 而非 "动力学趋势"，dynamics model 可能只学到平滑的均值回归（mean reversion），250ms 以外的改善有限。

**关键验证点**：Phase 1 训练的 250ms 结果。如果 frozen encoder + dynamics model 能达到 baseline_v2 的 80%+，说明 encoder latents 确实包含可用信息。

### 1.6 实现计划

| 阶段 | 内容 | 预计时间 |
|---|---|---|
| Week 1 前半 | 实现 AttentionPooling + S4D dynamics module | 2 天 |
| Week 1 后半 | 实现 GRU dynamics module + 训练 pipeline 适配 | 2 天 |
| Week 2 前半 | 250ms smoke test（S4D + GRU） | 2 天 |
| Week 2 后半 | 三窗口正式训练（250/500/1000ms, S4D） | 2–3 天 |
| Week 3 | S4D vs GRU vs Mamba 对比 + 端到端微调 | 4–5 天 |
| Week 4 | 与 baseline_v2 + benchmark 对比，可视化 | 3–4 天 |

**总计**: 约 3–4 周

### 1.7 风险与缓解

| 风险 | 可能性 | 影响 | 缓解策略 |
|---|---|---|---|
| Encoder latents 不含 dynamics 信息 | 中 | 高 | Phase 2 端到端微调让 encoder 适应 |
| S4D 线性不够 | 低-中 | 中 | 升级到 GRU/Mamba |
| Attention pooling 信息损失 | 低 | 中 | 增加 K（4→8→16）|
| 训练不稳定 | 低 | 中 | 两阶段训练 + gradient clipping |


---

## 第二部分：方案 2 — 扩散模型生成器（Diffusion/Flow Decoder）

### 2.1 方案概述与动机

**核心思路**：将预测窗口的 T x N spike count matrix 视为一个整体生成目标，使用条件扩散模型（或 flow matching）从 POYO+ encoder 的输出条件生成未来的 spike count 矩阵。

**关键创新机会**：根据文献调研，**目前没有论文使用 diffusion/flow matching 进行 spike forward prediction**。现有工作中最接近的是 LDNS（NeurIPS 2024 Spotlight），但它做的是 unconditional/conditional **generation**（从行为变量生成 spike 样本），而非 **prediction**（从历史 spikes 预测未来 spikes）。这是一个明确的研究空白。

**为什么 diffusion 可能有效**：
1. **完全消除 exposure bias** — 训练和推理都是从噪声到数据的 denoising 过程，不存在 teacher-forcing / rollout 分布偏移
2. **整体生成** — 一次性生成整个 T x N 矩阵，而非逐 bin AR，避免误差累积
3. **内置不确定性量化** — 多次采样可得到预测分布的置信区间
4. **NeurIPS 话题热度** — diffusion models 是当前 ML 领域最热门的方向之一

### 2.2 文献调研：Spike 数据的扩散/生成模型

#### 2.2.1 LDNS — Latent Diffusion for Neural Spiking Data

**论文**：Kapoor et al., NeurIPS 2024 Spotlight (arXiv:2407.08751)
**机构**：Macke Lab, University of Tubingen / Max Planck Institute

**架构**：两阶段框架
1. **Autoencoder 阶段**：Encoder 用双向 S4 层，将 discrete spikes 映射到 smooth continuous latent (8–32 维)；Decoder 用逐点 MLP，latents 映射到 Poisson firing rates
2. **Diffusion 阶段**：DDPM (1000 timesteps, epsilon-parameterization, linear noise schedule)，S4-based temporal layers

**条件生成**：无条件、角度条件、轨迹条件、可变长度

**Latent 维度**：8 dims（128 neurons, Lorenz）、16 dims（182 neurons, monkey reaching）、32 dims（128 neurons, human speech）

**对本方案的关键启示**：
- 证明了 spikes → continuous latents → diffusion 的技术路径可行
- S4 层同时用于 autoencoder 和 denoising network，是 spike 数据的良好 backbone
- **差异**：LDNS 的条件是行为变量；本方案的条件是 POYO+ encoder 输出的 history latent tokens

#### 2.2.2 Neural Timeseries Diffusion

**论文**：Vetter et al., Patterns (Cell Press) 2024

**创新**：使用 Ornstein-Uhlenbeck (OU) noise（有色噪声）替代标准 Gaussian noise，更好匹配 neural data 的 1/f 功率谱特性

**启示**：OU noise schedule 可能更适合 neural data 的时间相关性

#### 2.2.3 EAG — Energy-Based AR for Neural Populations

**论文**：arXiv:2511.17606, November 2025

**核心**：能量基 Transformer + masked AR training（随机 masking ratio 0.7–1.0）

**启示**：masking 策略可整合到 diffusion 的条件机制中

#### 2.2.4 RLPP — RL-Based Spike Generation

**论文**：Wu et al., Nature Computational Science 2026

**启示**：如果扩散模型生成的 spikes 不仅要 "统计正确" 还要 "功能正确"，可引入 downstream task reward 作为辅助训练信号

#### 2.2.5 其他相关基础模型

- **NEDS** (ICML 2025 Spotlight, arXiv:2504.08201)：多任务 encoding+decoding 基础模型，IBL 83 只动物预训练
- **Foundation Model of Neural Activity** (Nature 2025)：视觉皮层响应预测基础模型，可泛化到新 stimulus 类型

### 2.3 具体架构设计

#### 方案 2A: Latent Diffusion（推荐）

**训练阶段 1: Count Autoencoder**

```
Target spike counts [B, T, N]
    |
    v
Count Encoder:
  - per-neuron: count -> Linear(1, D) -> GELU -> Linear(D, D)
  - attention pooling / mean-pool across neurons -> [B, T, D_latent]
  - 1D Conv layers along time (temporal smoothing)
    |
    v
Continuous latent [B, T, D_latent=32]
    |
    v
Count Decoder:
  - 复用 PerNeuronMLPHead (latent 作为 bin_repr)
    |
    v
Reconstructed log_rate [B, T, N]
Loss: Poisson NLL + L2 reg + temporal smoothness
```

**训练阶段 2: Conditional DDPM**

```
Continuous latent target [B, T, D_latent=32]   (from frozen encoder)
    |   + Gaussian noise (linear schedule, 1000 steps)
    v
Noisy latent [B, T, D_latent]
    |   + Condition: POYO+ encoder latents [B, S=32, D=128]
    v
Denoising Network (1D U-Net):
  - Down blocks: 1D Conv + GroupNorm + SiLU, stride=2
  - Mid block: 1D Conv + Cross-Attention to encoder latents
  - Up blocks: 1D TransposedConv + skip connections
  - Timestep embedding via sinusoidal + MLP
    |
    v
Predicted noise epsilon_hat [B, T, D_latent]
```

**推理**：

1. POYO+ encode history → latents（1 次前向）
2. Sample z_1000 ~ N(0,I) [B, T, D_latent]
3. 逐步 denoise（DDIM 加速到 50–100 步）
4. Count decoder: z_0 → log_rate [B, T, N]
5. 可选：采样 M=10–50 次做不确定性量化

#### 方案 2B: Direct Count-Space Flow Matching

```
Target: x_1 = log(1 + counts) [B, T, N]  (continuous relaxation)
Source: x_0 ~ N(0, I) [B, T, N]

Flow matching: learn v_theta(x_t, t, cond)
  where x_t = (1-t)*x_0 + t*x_1

推理: ODE integration x_0 -> x_1 (Euler, 10-50 steps)
```

**2A vs 2B 对比**：2A 有 LDNS 文献支持，更可靠但需要额外 autoencoder；2B 更简单但在 discrete count data 上未经验证。**推荐先做 2A**。

### 2.4 训练策略

**Stage 1: Count Autoencoder** — AdamW, lr=1e-3, 100 epochs
**Stage 2: Conditional DDPM** — 冻结 autoencoder + encoder，AdamW lr=1e-4, 300–500 epochs
**Stage 3 (optional)**: 端到端微调，小学习率，50–100 epochs

### 2.5 预期性能

- **250ms**: 0.19–0.22 fp-bps（短窗口 diffusion 优势不明显，autoencoder 引入少量误差）
- **500ms**: 0.17–0.19 fp-bps（整体生成避免误差累积）
- **1000ms**: 0.14–0.17 fp-bps（最大收益区间，无 AR 误差累积）

**推理时间**：50 步 DDIM 约 100ms（vs baseline_v2 的 5ms），对离线分析可接受

### 2.6 实现计划

总计约 4–6 周：
- Week 1: Count autoencoder（3–4 天）
- Week 1–2: 1D U-Net + conditioning（3–4 天）
- Week 2–3: DDPM 训练调通 + 250ms smoke test（4–5 天）
- Week 3–4: 三窗口正式训练（4–5 天）
- Week 4–5: 对比评估 + 不确定性可视化（3–4 天）

### 2.7 风险与缓解

- **Autoencoder 重建损失大**：增大 D_latent (32→64)；用 PerNeuronMLPHead 替代简单 MLP
- **DDPM 推理太慢**：DDIM 加速（1000→50 步）；Consistency Model 一步生成
- **Count data 的 discrete 性质**：在 log(1+count) 空间操作；Poisson 后处理
- **条件注入不充分**：多层 cross-attention；classifier-free guidance

---

## 第三部分：方案对比与推荐

### 3.1 多维对比

| 维度 | 方案 1: Latent Dynamics | 方案 2: Diffusion/Flow |
|---|---|---|
| 核心创新 | POYO+ encoder + learned dynamics | Conditional latent diffusion for spikes |
| 技术成熟度 | 高（LFADS/POSSM 验证） | 中（LDNS 验证生成，未验证预测） |
| 实现难度 | 中（主要改 decoder） | 高（需 autoencoder + denoising net） |
| 代码兼容性 | 高（PerNeuronMLPHead 完全复用） | 中（需新 count encoder/decoder） |
| Exposure bias | 完全消除 | 完全消除 |
| 推理速度 | 快（单次前向 + T 步 SSM） | 慢（50–100 步 denoising） |
| 不确定性量化 | 需额外设计（ensemble/SDE） | 内置（多次采样） |
| NeurIPS 叙事 | 强（Foundation model + dynamics） | 很强（首个 diffusion spike prediction） |
| 250ms 预期 | 持平或略超 baseline_v2 | 持平 baseline_v2 |
| 1000ms 预期 | 显著超越 | 超越 |
| 参数量 | 约 2.3–2.9M | 约 3–4M |
| 开发时间 | 3–4 周 | 4–6 周 |
| 风险等级 | 中 | 高 |

### 3.2 推荐实施路线

**Phase A (Week 1–4): 优先做方案 1（Latent Dynamics）**
- 理由：实现更快、风险更低、与现有代码兼容性高、POSSM/LFADS 已验证类似思路
- 目标：在 1000ms 上超越 baseline_v2
- SSM 优先级：S4D（最简单）→ GRU → Mamba
- 如果成功：这就是论文的核心结果

**Phase B (Week 4–8, optional): 方案 2（Diffusion）作为补充/对比**
- 如果方案 1 成功：diffusion 作为 ablation/comparison 增加论文厚度
- 如果方案 1 效果不够好：diffusion 提供不同的技术路径
- 优先 2A（latent diffusion），2B（flow matching）作为 further ablation

**Phase C (parallel with A/B): 补充实验（6.3 节）**
- 可与方案 1/2 并行，因为这些实验用 baseline_v2 即可

---

## 第四部分：6.3 所需额外实验的详细设计

以下实验来自 `neurips_innovation_claude.md` 第 6.3 节，按优先级排列并提供完整实验方案。

### 实验 1: [CRITICAL] Non-Causal Decoder Ablation

**目的**：验证 causal self-attention 是否真的有贡献。如果去掉 causal mask 后性能不变或更好，说明 "temporal ordering" 不重要。

**实验设计**：
- 使用 baseline_v2 架构（无 prediction feedback）
- 修改点：`torch_brain/nn/autoregressive_decoder.py` 中将 `create_causal_mask()` 返回全 True mask
- 对比条件：causal (baseline_v2) vs non-causal (bidirectional)
- 训练配置：10 sessions, continuous sampling, 500ms obs, 300 epochs
- 评估窗口：250ms, 500ms, 1000ms
- 指标：fp-bps, R-squared, PSTH-R-squared, per-bin fp-bps decay curve

**预期结果**：
- non-causal 大于等于 causal：说明 causal mask 只是 regularization → 强化 Option D 必要性
- non-causal 小于 causal：说明 temporal ordering 有价值 → 支持因果建模

**实现难度**：低（改一行 mask 代码）
**实验时间**：约 3 天（3 窗口训练+评估）

### 实验 2: [CRITICAL] Encoder Ablation

**目的**：量化 POYO+ Perceiver encoder vs vanilla Transformer encoder 的贡献。

**实验设计**：
- 保留 baseline_v2 的 decoder + head
- 替换 encoder：Perceiver cross-attention → 标准 Transformer encoder
  - 将 spike events 做 bin-level 聚合（如 NDT2 的 spatiotemporal patches）
  - 6 层 self-attention, 调整 dim/heads 保持参数量相当
- 对比：POYO+ Perceiver vs vanilla Transformer

**实现难度**：中（需实现 vanilla Transformer encoder wrapper）
**实验时间**：约 5 天（2 天实现 + 3 天训练）

### 实验 3: [IMPORTANT] Additional Dataset

**目的**：验证跨脑区/任务/物种的泛化性。

**推荐数据集**：IBL Brain-Wide Map（多脑区 Neuropixels，有 NDT2/IBL-MtM 现成结果）

**实验设计**：
- 下载 IBL 数据 → 预处理为 NeuroHorizon 格式
- 训练 baseline_v2 + 最佳新方案
- 与 NDT2 / IBL-MtM 已有结果对比

**实现难度**：中-高（数据预处理 + dataloader 适配）
**实验时间**：约 1 周

### 实验 4: [IMPORTANT] Bin Size Ablation

**目的**：确定预测效果与时间分辨率的关系。

**实验设计**：
- Bin sizes: 5ms, 10ms, 20ms (当前), 50ms, 100ms
- 固定预测窗口 500ms（对应 bins: 100, 50, 25, 10, 5）
- 每个 bin size 训练 baseline_v2
- 评估 fp-bps（归一化到相同 spike count）

**预期**：较大 bin size (50–100ms) 可能有更高的 bin-to-bin 信号噪声比

**实现难度**：低-中
**实验时间**：约 3–5 天

### 实验 5: [BENEFICIAL] Iso-Parameter Comparison

**目的**：排除参数量差异的混淆。

**实验设计**：
- NeuroHorizon 放大到约 5M 参数（dim 128→192, enc_depth 6→8, dec_depth 2→4）
- NeuroHorizon 缩小到约 1M 参数
- 对比 1M / 2.1M / 5M 的 scaling behavior

**实现难度**：低
**实验时间**：约 3 天

### 实验 6: [BENEFICIAL] Prediction Visualization

**目的**：让性能差异直观可理解。

**可视化内容**：
1. Spike raster: GT vs predicted（per-neuron spike count 对比）
2. Population activity pattern（PCA/UMAP of predicted vs GT 神经轨迹）
3. Per-bin fp-bps decay curve（所有模型叠加）
4. Prediction uncertainty band（diffusion 方案多次采样的 95% CI）
5. Error accumulation heatmap（neuron x time bin 的误差空间分布）

**实现难度**：中
**实验时间**：约 2–3 天

### 6.3 实验优先级与时间线总结

| 优先级 | 实验 | 时间 | 可并行 |
|---|---|---|---|
| 1 | Non-causal ablation | 3 天 | 可与方案 1 并行 |
| 2 | Encoder ablation | 5 天 | 可与方案 1 并行 |
| 3 | Additional dataset | 1 周 | 需方案 1 结果后决定 |
| 4 | Bin size ablation | 3–5 天 | 可与方案 2 并行 |
| 5 | Iso-parameter comparison | 3 天 | 可随时做 |
| 6 | Prediction visualization | 2–3 天 | 需全部实验完成后 |

---

## 第五部分：参考文献

- POSSM: Ryoo et al. 2025, "Generalizable, real-time neural decoding with hybrid state-space models", NeurIPS 2025 (arXiv:2506.05320)
- LDNS: Kapoor et al. 2024, "Latent Diffusion for Neural Spiking Data", NeurIPS 2024 Spotlight (arXiv:2407.08751)
- LFADS: Pandarinath et al. 2018, "Inferring single-trial neural population dynamics using sequential auto-encoders", Nature Methods
- POYO: Azabou et al. 2023, NeurIPS 2023
- POYO+: Azabou et al. 2025, ICLR 2025
- Neuroformer: Antoniades et al. 2023, NeurIPS 2023
- NDT2: Ye et al. 2023, NeurIPS 2023
- NEDS: ICML 2025 Spotlight (arXiv:2504.08201)
- EAG: arXiv:2511.17606, November 2025
- Neural Timeseries Diffusion: Vetter et al. 2024, Patterns (Cell Press)
- RLPP: Wu et al. 2026, Nature Computational Science
- FlowFA: NeurIPS 2021
- Flow Matching: Lipman et al. 2023
- Mamba: Gu & Dao 2023
- S4: Gu et al. 2022, "Efficiently Modeling Long Sequences with Structured State Spaces", ICLR 2022
- Churchland et al. 2012, "Neural population dynamics during reaching", Nature
- Cunningham & Yu 2014, "Dimensionality reduction for large-scale neural recordings", Nature Neuroscience
- Foundation Model of Neural Activity: Wang et al. 2025, Nature
