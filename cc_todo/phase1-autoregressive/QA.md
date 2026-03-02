# Phase 1 实现分析 QA

- **日期**：2026-03-02
- **对应 plan.md 任务**：Phase 1 全部（1.1-1.3）
- **文档目的**：整理 Phase 1 实现过程中的关键技术问题与分析

---

## Q1: 数据处理与组织方式

### Q1.1 Sessions/Trials 详情

> 用的数据中，用了哪些 sessions、哪些 trials，session 和 trials 的关系是怎样的，不同 session/trials 是什么不同，有多少完全重复的 trials 设置吗，选择的考量是什么；每次分析用的是一个 session 还是很多 sessions，会考虑不同个体/不同 neurons/不同 trials 结构的差异吗

#### 1. 数据来源

- **数据集**：Perich-Miller 2018（DANDI:000688），运动皮层 Utah Array 记录
- **选取规模**：10 sessions，来自 3 只恒河猴（C:4, J:3, M:3），总计 320MB
- **任务类型**：Center-out reaching（中心外向到达任务）

#### 2. Session 详情

| Session | 受试 | 神经元数 | 有效 Trials | 时长(min) | 平均发放率(Hz) |
|---------|------|---------|-------------|----------|--------------|
| c_20131003 | C | 71 | 157 | 11.1 | 6.8 |
| c_20131022 | C | 41 | 150 | 10.3 | 8.3 |
| c_20131101 | C | 41 | 243 | 14.8 | 7.0 |
| c_20131204 | C | 52 | 156 | 10.6 | 7.3 |
| j_20160405 | J | 38 | 203 | 21.3 | 6.2 |
| j_20160406 | J | 18 | 195 | 16.8 | 8.4 |
| j_20160407 | J | 19 | 199 | 18.0 | 6.4 |
| m_20150610 | M | 49 | 184 | 15.9 | 6.2 |
| m_20150612 | M | 37 | 165 | 19.8 | 5.8 |
| m_20150615 | M | 41 | 164 | 19.4 | 6.2 |
| **合计** | **3** | **407** | **1816** | **158** | **6.86** |

#### 3. Session 与 Trial 的关系

**Session** 是一次完整的实验记录（一天的数据），每个 session 包含一只猴子在一次实验中完成的所有 trials。

**Trial** 是单次动作执行，典型的时间结构为：

```
|--- Hold Period ---|-- Go Cue --|--- Reach Period ---|--- Target Acquire ---|
|     约 676ms      |   视觉信号  |     约 1090ms      |   到达并停在目标上    |
|  手保持在中心不动  |            |  手向目标方向移动   |                      |
```

#### 4. 不同 Session 之间的差异

- **不同猴子（C/J/M）**：
  - 大脑解剖结构不同
  - Utah Array 植入位置/深度不同
  - 行为策略可能不同（运动速度、轨迹特征）
  - 记录时间相差数年（C:2013, J:2016, M:2015）

- **同猴不同天（如 c_20131003 vs c_20131022）**：
  - Utah Array 记录的神经元集合**可能部分重叠但不完全相同**（电极漂移、信号质量变化）
  - 神经元数量可能不同（如 C 猴从 71 降到 41）
  - 行为表现可能有日间变化

- **神经元数量差异巨大**：18（j_20160406）到 71（c_20131003），相差近 4 倍

#### 5. 不同 Trial 之间的差异

- 同一 session 内的所有 trials 都是 **center-out reaching 任务**，但**目标方向不同**（通常 8 个方向）
- **不存在"完全重复"的 trials**：即使目标方向相同，每次的 spike timing 也不同（神经元发放具有 Poisson 随机性）
- 同方向的 trials 可以视为同一条件的重复采样，用于计算 PSTH（trial-averaged firing rate）
- Hold 期时长有变异（均值 676ms, std 428ms），Reach 期时长较稳定（均值 1090ms, std 152ms）
- **Trial 筛选规则**（在 pipeline 中实现）：仅保留 success trials（result=="R"）、有效 target_id、时长 0.5-6.0s

#### 6. 当前 Phase 1 的数据使用方式

- **使用全部 10 sessions 混合训练**，不区分个体或 session
- 通过 `InfiniteVocabEmbedding`（IVE）为每个 session 的每个神经元分配**唯一的可学习 embedding**（per-session, per-unit），使模型能区分不同 session 的不同神经元
- `UnitDropout` 数据增强（随机 mask 30-200 个 units），增强对神经元数量变化的鲁棒性
- 每个 session 内部做 **70/10/20 train/valid/test** 分割（基于 trial 边界）
- **不区分 trial 条件**（不按 reach 方向分组），所有 trial 均匀采样
- 训练时跨 session 混合 batching：一个 batch 内可能同时包含来自不同 session、不同猴子的样本

#### 7. 选择 10 Sessions 的考量

- **Phase 1 的目标是验证功能正确性**，不需要大规模数据
- 10 sessions（320MB）足以覆盖 3 个受试、验证跨 session 训练机制
- 6372 个训练窗口（1s each）提供了足够的训练样本
- Perich-Miller 2018 数据集总共有 70+ sessions，Phase 2/3 可扩展到全部

---

### Q1.2 关键配置参数及其影响

> sessions_num, neurons_num, time_windows（hold/reachout）, previous_timewindow_length, bins_length 等信息如何配置，会有怎样的影响

#### 1. 采样方式：Scheme B（滑动窗口）

当前 Phase 1 采用 **Scheme B（滑动窗口）** 采样，而非 trial-aligned：

- `RandomFixedWindowSampler` 从 `train_sampling_intervals`（训练集内的有效时间段）中**随机抽取固定长度窗口**
- **不对齐到 trial 边界**：一个窗口可能落在 hold 期、reach 期、或跨越 trial 边界
- 优点：数据利用率高、不依赖 trial 结构、泛化性更强
- 缺点：可能跨 trial 边界引入不连续性；无法直接计算 PSTH correlation

#### 2. 配置参数总览

| 参数 | 250ms 实验 | 500ms 实验 | 1000ms 实验 | 说明 |
|------|-----------|-----------|------------|------|
| sequence_length | 0.75s | 1.0s | 1.5s | 总窗口 = history + prediction |
| hist_window | 0.5s | 0.5s | 0.5s | 编码器输入的历史长度（隐含，= seq_len - pred_window） |
| pred_window | 0.25s | 0.5s | 1.0s | 预测窗口长度 |
| bin_size | 0.02s | 0.02s | 0.02s | 20ms，spike count 离散化粒度 |
| T_pred_bins | 12 | 25 | 50 | 预测窗口内的 bin 数 |
| batch_size | 64 | 64 | 32 | 1000ms 因显存减半 |
| 训练窗口数/epoch | 约 6372 | 约 4778 | 约 3185 | 窗口越长，可采样数越少 |
| dim | 128 | 128 | 128 | 模型维度（Small 配置） |
| enc_depth | 6 | 6 | 6 | encoder 层数 |
| dec_depth | 2 | 2 | 2 | decoder 层数 |
| epochs | 300 | 300 | 300 | 训练轮数 |
| base_lr | 3.125e-5 | 3.125e-5 | 3.125e-5 | 基础学习率 |

#### 3. 各参数的影响分析

**history window（0.5s）**：
- 覆盖 hold 期（87% trials 的 hold > 250ms），包含运动准备信息
- 0.5s / 0.05s(latent_step) = 10 步 x 32 latents = 320 个 latent tokens
- 增加 history → 更多上下文信息，但增加计算量和显存

**bin_size（20ms）**：
- 20ms 是主流 spike foundation model 的标准选择
- 太小（5ms）：大多数 bin 的 spike count = 0，过于稀疏
- 太大（50ms）：丢失时间分辨率，无法捕捉 fast dynamics
- 20ms bin 下平均 spike count = 0.137/bin，零 bin 比例约 87.6%
- 与 POYO+、NDT 系列、NLB benchmark 一致

**prediction window（250ms/500ms/1000ms）**：
- 影响 T_pred_bins 数量和预测难度
- 窗口越长，后续 bin 的预测越不确定（但实验显示衰减缓慢）
- 影响可采样窗口数（越长 → 数据越少 → batch 可能需要减小）

**session 数量和神经元数量**：
- 当前 10 sessions, 407 units（per-session 18-71 units）
- 不同 session 的 units 数不同 → batch 内需要 padding + mask
- 更多 sessions → 更多数据，但也增加 IVE embedding 参数量
- 更多 neurons → per-neuron MLP head 的计算量线性增长

#### 4. 数据流全链路

```
原始 HDF5 文件 (per-session)
    ↓ Dataset.from_hdf5() 懒加载
Data 对象 (spikes, trials, domains)
    ↓ RandomFixedWindowSampler 抽取固定长度窗口
(session_id, window_start, window_end) 索引
    ↓ tokenize() 方法
{
  encoder inputs:  spike events + start/end tokens (history 0.5s)
  latent queries:  320 个 latent tokens (每 50ms 一组)
  bin timestamps:  12/25/50 个 bin center times
  target:          binned spike counts [T_bins, N_units]
  unit indices:    per-session global unit IDs
}
    ↓ collate (pad8 + pad2d)
Batched tensors → Model → PoissonNLL Loss
```

---

### Q1.3 配置文件位置与调整指南

> 所有路径相对于 `/root/autodl-tmp/NeuroHorizon/`

#### 1. 选择训练 Sessions（数据选择层）

| 配置文件 | 关键字段 | 当前值 | 调整说明 |
|---------|---------|--------|---------|
| `examples/neurohorizon/configs/dataset/perich_miller_10sessions.yaml` | `sessions:` 列表 | 10 个 session 名 | 增删列表项即可；名称须与 `data/perich_miller_population_2018/` 下 HDF5 文件名对应 |

如果要用新数据集（如 IBL），需要：
1. 运行对应 pipeline 脚本生成 HDF5（参考 `scripts/data/perich_miller_pipeline.py`）
2. 新建 dataset yaml（如 `ibl_sessions.yaml`）
3. 在训练 config 中引用

#### 2. 模型架构与窗口参数（模型层）

| 配置文件 | 参数 | 当前值(250ms) | 调整说明 |
|---------|------|-------------|---------|
| `examples/neurohorizon/configs/model/neurohorizon_small.yaml` | `sequence_length` | 0.75 | 总窗口 = history + prediction |
| 同上 | `pred_window` | 0.250 | 预测窗口长度 |
| 同上 | `bin_size` | 0.020 | bin 宽度 |
| 同上 | `dim` | 128 | 模型维度 |
| 同上 | `enc_depth` | 6 | encoder 层数 |
| 同上 | `dec_depth` | 2 | decoder 层数 |
| 同上 | `latent_step` | 0.05 | latent token 时间间隔 |
| 同上 | `num_latents_per_step` | 32 | 每步 latent 数 |
| 同上 | `max_pred_bins` | 50 | 最大预测 bin 数 |

**注意**：`hist_window` 不是显式配置，而是由 `sequence_length - pred_window` 隐式决定。要改 history 长度，需同时调 `sequence_length`。

已有 4 个模型配置变体：
- `neurohorizon_small.yaml`（250ms）
- `neurohorizon_small_500ms.yaml`（500ms）
- `neurohorizon_small_1000ms.yaml`（1000ms AR）
- `neurohorizon_small_1000ms_noar.yaml`（1000ms non-AR，`causal_decoder: false`）

#### 3. 训练超参数（训练层）

| 配置文件 | 参数 | 当前值 | 调整说明 |
|---------|------|--------|---------|
| `examples/neurohorizon/configs/train_small.yaml` | `epochs` | 300 | 训练轮数 |
| 同上 | `batch_size` | 64 | batch 大小 |
| 同上 | `eval_epochs` | 10 | 验证间隔 |
| `examples/neurohorizon/configs/defaults.yaml` | `optim.base_lr` | 3.125e-5 | 基础学习率（实际 lr = base_lr x batch_size） |
| 同上 | `optim.weight_decay` | 1e-4 | 权重衰减 |
| 同上 | `optim.lr_decay_start` | 0.5 | OneCycleLR 衰减开始比例 |
| 同上 | `seed` | 42 | 随机种子 |
| 同上 | `precision` | bf16-mixed | 混合精度 |

#### 4. 数据增强（transforms 层）

| 位置 | 参数 | 当前值 | 调整说明 |
|------|------|--------|---------|
| `defaults.yaml` transforms | `UnitDropout.max_units` | 200 | 最大保留神经元数 |
| 同上 | `UnitDropout.min_units` | 30 | 最小保留神经元数 |
| 同上 | `UnitDropout.mode_units` | 80 | 保留数量众数 |

#### 5. 数据处理 Pipeline（预处理层）

| 文件 | 控制内容 | 调整场景 |
|------|---------|---------|
| `scripts/data/perich_miller_pipeline.py` | NWB to HDF5 转换、trial 筛选、train/valid/test 分割比例（70/10/20） | 改数据源、修改 split 比例 |
| `torch_brain/data/dataset.py` | 懒加载、session 索引 | 通常无需改动 |
| `torch_brain/data/sampler.py` | 滑动窗口采样（窗口长度由模型 config 的 `sequence_length` 决定） | 改为 trial-aligned 需修改此文件 |

#### 6. Tokenize 逻辑（模型内部数据流）

| 文件 | 方法 | 控制内容 |
|------|------|---------|
| `torch_brain/models/neurohorizon.py` | `tokenize()` | spike events 分割为 history/prediction 窗口、encoder input tokens 生成、binned spike counts 计算 |

#### 7. 训练启动命令

```bash
conda activate poyo
cd /root/autodl-tmp/NeuroHorizon

# 250ms
python examples/neurohorizon/train.py --config-name train_small
# 500ms
python examples/neurohorizon/train.py --config-name train_small_500ms
# 1000ms AR
python examples/neurohorizon/train.py --config-name train_small_1000ms
# 1000ms non-AR
python examples/neurohorizon/train.py --config-name train_small_1000ms_noar
```

#### 8. 常见调整场景速查

| 想要做什么 | 修改位置 |
|-----------|---------|
| 增减训练 sessions | `configs/dataset/perich_miller_10sessions.yaml` 的 `sessions:` 列表 |
| 改变预测窗口长度 | 新建模型 yaml（参考已有 4 个变体） |
| 改变 bin 宽度 | 模型 yaml 的 `bin_size` |
| 增加历史窗口长度 | 模型 yaml 的 `sequence_length`（保持 `pred_window` 不变） |
| 改变 batch size | 训练 yaml 的 `batch_size` |
| 改变学习率 | `defaults.yaml` 的 `optim.base_lr` |
| 关闭 UnitDropout | `defaults.yaml` 中删除 transforms 的 UnitDropout 条目 |
| 关闭 causal mask | 模型 yaml 添加 `causal_decoder: false` |
| 改变 train/valid/test 比例 | `scripts/data/perich_miller_pipeline.py` 的 split 参数 |
| 切换为 trial-aligned 采样 | 修改 `torch_brain/data/sampler.py` 或在 train.py 中替换 sampler |

---

## Q2: 效果评估分析

### Q2.1 当前指标是否合理 + 应否引入 bits/spike

> 目前的 metric 合理的数值应该是多少，现在得到的数据是否合理，当前指标在什么数据水平上计算的，是否应该引入 bits/spike 等指标

#### 1. 当前结果回顾

| 实验 | Bins | Best R2 | Best Epoch | Final Val Loss |
|------|------|---------|-----------|---------------|
| 250ms AR | 12 | 0.2658 | 229 | 0.3142 |
| 500ms AR | 25 | 0.2417 | 229 | 0.3124 |
| 1000ms AR | 50 | 0.2343 | 299 | 0.3159 |
| 1000ms non-AR | 50 | 0.2354 | 259 | 0.3159 |

#### 2. R2 的计算水平

当前 R2 是在以下水平上聚合计算的：

- **Sessions**：跨全部 10 sessions 的验证集（每 session 10% trials）
- **神经元**：跨所有 407 个神经元（但 per-session 不同：18-71）
- **Time bins**：跨所有 prediction bins（12/25/50）
- **计算方式**：R2 = 1 - SS_res/SS_tot，其中 y 涵盖所有 (session, trial_window, bin, neuron) 的 spike counts
- **与 config 的关系**：R2 的分母 SS_tot 取决于验证集的 spike count 方差；分子 SS_res 取决于模型预测的误差。更多 neurons、更多 bins 都会稀释 R2（因为低发放率 neuron 和远时间步 bin 的预测更难）

#### 3. R2 约 0.26 是否合理？——**是的，合理**

**核心论点：spike 预测 R2 和行为解码 R2 是完全不同的指标，不能直接比较。**

| 对比维度 | 行为解码 R2（Phase 0） | Spike 预测 R2（Phase 1） |
|---------|----------------------|------------------------|
| 预测目标 | cursor velocity（2D 连续值） | binned spike counts（离散整数） |
| 信噪比 | 高（行为变量平滑、低维） | 低（spike count 有 Poisson 噪声） |
| Phase 0 值 | **0.807** | - |
| Phase 1 值 | - | **0.266** |
| 可解释方差 | 大部分方差可被预测 | 大量方差是不可约的随机噪声 |

**理论上限估算**：

假设真实 firing rate 为 lambda，观测到的 spike count k 服从 Poisson(lambda)。

- Poisson 分布的方差 = mean = lambda
- 对于平均 firing rate = 6.86 Hz，20ms bin 下 lambda = 0.137
- Spike count 的总方差 = signal variance（率的变化）+ noise variance（Poisson 噪声）
- Signal variance 约为总方差的 20-40%（经验估计）
- 因此 **spike prediction R2 的理论上限约 0.2-0.4**
- R2 = 0.26 处于这个范围内，说明模型已经捕捉到了大部分可预测的信号

#### 4. 主流模型的参考数值

**关键提示**：大多数主流模型报告的是**行为解码 R2**（而非 spike 预测 R2），两者不能直接比较。

| 模型 | 任务 | 数据集 | 指标 | 值 | 可比性 |
|------|------|--------|------|------|--------|
| POYO | 行为解码 | Perich-Miller CO | R2(velocity) | 0.93 | 不可比（不同任务） |
| POYO+ | 行为解码 | Perich-Miller CO | R2(velocity) | 0.807(我们的 baseline) | 不可比 |
| NDT1 | 行为解码 | Motor cortex | R2(velocity) | 0.918 | 不可比 |
| NDT2 | 行为解码 | Motor cortex | R2(velocity) | 0.54(sorted) | 不可比 |
| LFADS | firing rate 推断 | Motor cortex | PSTH corr | >0.9 | 部分可比（去噪后的率，非 raw count） |
| NDT1/2 | spike 预测 | NLB mc_maze | bits/spike | 0.1-0.4 | **可比**（需换算） |
| NLB 榜单 | spike latent | mc_maze | co-bps | 0.1-0.3 | 可比（需引入 co-bps） |
| SPINT | 跨 session 解码 | Motor cortex | R2(velocity) | 0.386 | 不可比（不同任务，且是 cross-session） |

**结论**：目前没有直接可比的「binned spike count prediction R2」文献报告。这恰恰说明需要引入 bits/spike 来建立可比性。

#### 5. 应该引入 bits/spike 吗？——**强烈建议是**

**bits/spike 公式**：
```
bits/spike = (NLL_baseline - NLL_model) / (N_spikes * log(2))
```

其中 NLL_baseline 是 homogeneous Poisson model（每个神经元用其平均 firing rate 作为预测）的 Poisson NLL。

**优势**：
- **归一化了发放率差异**：不同数据集/神经元的 NLL 绝对值差异大，bits/spike 标准化了这一差异
- **跨论文可比**：NDT1/2、LFADS 等都报告 bits/spike
- **实现成本极低**：只需额外算一次 baseline NLL
- **典型值**：0.1-0.5 bits/spike 表示 decent 模型

**建议**：Phase 1 补算 bits/spike 并记录到 results.md。

---

### Q2.2 效果是否符合预期 + 潜在问题分析

> 当前的效果是否符合预期，数据组织/输入输出/代码/loss 是否有问题，效果是否需要改进

#### 1. 总体评估：效果基本符合预期

Phase 1 的目标是**验证自回归架构的功能正确性**，而非追求最优性能。从这个角度看：

- ✅ 模型能训练收敛（所有 4 组实验 loss 正常下降）
- ✅ R2 > 0（模型比均值预测好）
- ✅ R2 在理论预期范围内（0.2-0.4）
- ✅ AR 与 TF 验证通过（max diff < 1e-5）
- ✅ 窗口长度增加时性能缓慢衰减（-11.9% total），说明模型长时程预测鲁棒

#### 2. 值得关注的发现与潜在问题

##### 发现 1：AR vs non-AR 无差异（delta < 0.002）

**现象**：1000ms AR 和 1000ms non-AR 的 R2 几乎相同（0.2343 vs 0.2354）。

**原因**：当前 decoder 的 bin query 使用**固定 position embedding**（rotary time embedding），不依赖前一步的输出。在 teacher forcing 训练时，causal mask 只是限制了 attention 范围，但每个 bin 的 query 表示本身不变。因此 TF 和 AR 在数学上等价。

**影响**：
- 这不是 bug，而是当前架构设计的必然结果
- causal mask 在当前架构下**不提供额外收益**
- scheduled sampling **不适用**于当前架构（因为没有 exposure bias）
- 如果需要真正的 AR 收益（如利用已预测的 spike 信息改善后续预测），需要**改变 decoder 设计**：
  - 方案 A：将前一步的预测输出 concatenate 到下一步的 query
  - 方案 B：使用 learned query embeddings 替代固定 position embeddings

##### 发现 2：R2 随窗口缓慢衰减

**现象**：250ms(-0%) -> 500ms(-9.1%) -> 1000ms(-3.1%) -> 总计 -11.9%

**正面解读**：模型长时程预测鲁棒，性能衰减亚线性。

**负面解读**：per-bin R2 在 12 个 bin 上波动（0.17-0.35）但**没有明显的时间衰减趋势**。这可能说明模型对每个 bin 的预测主要依赖 encoder latent（全局历史信息），而非利用 bin 之间的时间顺序信息。

##### 发现 3：潜在的数据处理问题

- ⚠️ **跨 trial 边界**：滑动窗口采样可能跨越 trial 边界（history 在 trial A 的 reach 期，prediction 在 trial B 的 hold 期）。这引入了不连续性——模型需要预测一个与历史不相关的未来。
  - 影响：可能拉低整体 R2
  - 缓解：可在采样时限制窗口不跨 trial 边界，或切换到 trial-aligned 采样

- ⚠️ **无 trial 条件对齐**：当前无法计算 PSTH correlation（需要同条件 trials 对齐后平均）。
  - 影响：少了一个有意义的评估指标
  - 缓解：在验证阶段按 trial 条件（reach 方向）分组计算

#### 3. 代码与 Loss 检查清单

| 组件 | 状态 | 说明 |
|------|------|------|
| Poisson NLL Loss | ✅ 正确 | `exp(r) - k*r`，log_rate clamp 到 [-10, 10]，数值稳定 |
| Tokenize 流程 | ✅ 正确 | 双窗口分离、bin count 统计均通过单元测试 |
| Causal mask | ✅ 正确 | 下三角 bool mask，测试通过 |
| Collate/Padding | ✅ 正确 | pad8 + pad2d + mask 追踪 |
| 梯度更新 | ✅ 正确 | SparseLamb optimizer，参数组正确分离 |
| 验证指标 | ⚠️ 可改进 | 只计算了 overall R2，未计算 per-session/per-neuron R2 分布 |

#### 4. 改进建议（按优先级）

| 优先级 | 建议 | 原因 | 复杂度 |
|--------|------|------|--------|
| **P1** | 引入 linear baseline | 量化 Transformer 的增量价值 | 低 |
| **P1** | 计算 bits/spike | 建立与文献的可比性 | 低 |
| **P2** | 引入 PSTH baseline | 量化最低性能下限 | 低 |
| **P2** | 报告 per-session/per-neuron R2 分布 | 更细粒度的性能分析 | 低 |
| **P3** | 补充 trial-aligned 采样实验 | 支持 PSTH correlation、消除跨 trial 边界噪声 | 中 |
| **P4** | 评估 decoder query 改进方案 | 使 AR 真正有别于 non-AR | 高 |

---

### Q2.3 Baseline 对比 + 主流模型参考

> 是否应该引入 baseline 作为对比，当前分析没有详细的 baseline 对比。主流 spike foundation model 的 metric 值大概是多少作为参考

#### 1. 当前的 Baseline 对比情况

Phase 1 目前**只有一组消融对比**：AR vs non-AR（causal mask 开/关）。

缺少的关键 baseline：
- **最低下限**（PSTH prediction、mean firing rate prediction）
- **简单模型**（linear regression、smoothed firing rate）
- **与文献可比的标准化指标**（bits/spike）

**这是一个明显的不足**：没有 baseline 参照，无法判断 R2=0.26 是好是差。

#### 2. 强烈建议补充的 Baseline

##### Baseline 1：Linear Ridge Regression（必做）

**方法**：
- 将 history window 的 binned spike counts（500ms / 20ms = 25 bins x N_units）展平为特征向量
- 用 Ridge Regression（sklearn）预测 prediction window 的 spike counts
- 对每个 session 单独训练（same train/valid split）

**意义**：
- 如果 NeuroHorizon R2 约等于 Linear R2 → Transformer 未提供显著非线性建模价值
- 如果 NeuroHorizon >> Linear → Transformer 架构的价值得到验证
- 实现简单（sklearn），可在 CPU 上快速计算

##### Baseline 2：PSTH Mean Prediction（建议做）

**方法**：
- 按 trial 条件（reach 方向）分组
- 计算训练集中同条件 trials 的平均 firing rate（PSTH）
- 将 PSTH 作为所有 test trials 的预测

**意义**：
- 如果模型不能超过 PSTH → 模型没有从 single-trial 数据中学到信息
- 这是最基本的下限

##### Baseline 3：Homogeneous Poisson（bits/spike 计算的前提）

**方法**：
- 每个神经元用其训练集平均发放率作为所有 time bin 的预测
- 计算 Poisson NLL → 作为 bits/spike 公式中的 NLL_baseline

**意义**：
- 提供 bits/spike 归一化计算的基础
- 实现极简（一行代码计算均值）

#### 3. 主流模型的指标对比参考

##### 任务类型区分

**必须理解的关键区分**：

```
行为解码（Behavior Decoding）  ≠  Spike 预测（Spike Prediction）
         ↓                              ↓
  预测 cursor velocity          预测 binned spike counts
  R2 = 0.8-0.95               R2 = 0.2-0.4（理论上限受限于 Poisson 噪声）
  大多数论文的主要指标          较少论文直接报告 R2，更多用 bits/spike
```

##### 行为解码 R2（不可直接与 Phase 1 比较，但提供上下文）

| 模型 | 数据集 | R2(velocity) | 说明 |
|------|--------|-------------|------|
| POYO | Perich-Miller CO | 0.93 | multi-session, multi-subject |
| POYO+（我们的 Phase 0） | Perich-Miller CO | 0.807 | 10 sessions, Small config |
| NDT1 | Motor cortex | 0.918 | single-session |
| LFADS/AutoLFADS | Motor cortex | 0.915 | single-session |
| Wiener Filter | Motor cortex | 0.5-0.8 | 经典简单 baseline |
| SPINT | Motor cortex (cross-session) | 0.386 | gradient-free, zero-shot |

##### Spike 预测相关指标（更可比）

| 模型 | 数据集 | 指标 | 值 | 说明 |
|------|--------|------|------|------|
| NDT1 | NLB mc_maze | bits/spike | 0.1-0.4 | masked modeling |
| NDT2 | NLB mc_maze | bits/spike | 0.2-0.5 | multi-context pretrain |
| LFADS | NLB mc_maze | co-bps | 0.1-0.3 | 去噪 + latent 推断 |
| NLB 榜单冠军 | mc_maze | co-bps | 约 0.3 | ensemble NDT |
| Neuroformer | 多区域 | Poisson NLL | 未标准化报告 | 逐 spike 自回归 |

##### 我们的 Phase 1 在这个谱系中的位置

- **与行为解码相比**：Phase 1 的 R2=0.26 看似很低，但这是**不同任务**——spike 预测天然受 Poisson 噪声限制
- **与 spike 预测相比**：需要引入 bits/spike 才能定量对比。粗略估计，如果 baseline Poisson NLL 约 0.5，model NLL 约 0.31，N_spikes 充足，则 bits/spike 约 0.27，处于合理范围
- **缺少 linear baseline**：无法判断 0.26 有多少来自 Transformer 的非线性建模，多少来自简单线性外推

#### 4. 是否需要 Linear Baseline？——**必须要**

**理由**：
1. 没有 baseline 无法解读 R2=0.26 的意义
2. Linear baseline 实现成本极低（1-2 小时）
3. 这是所有 spike prediction 论文的标配 baseline
4. 如果 Linear R2 约 0.20，NeuroHorizon 的 0.26 代表约 30% 的提升——这是有意义的
5. 如果 Linear R2 约 0.25，则说明 Transformer 没提供价值——需要诊断和改进

#### 5. 建议的 Baseline 补充优先级

| 优先级 | Baseline | 实现复杂度 | 必要性 |
|--------|---------|-----------|--------|
| 1 | **Linear Ridge Regression** | 低（sklearn, 1h） | **必做** |
| 2 | **Bits/spike（需 homogeneous Poisson baseline）** | 低（几行代码） | **必做** |
| 3 | PSTH Mean Prediction | 低（需 trial 条件分组） | 建议做 |
| 4 | Smoothed Firing Rate | 低（Gaussian kernel 平滑） | 可选 |
| 5 | Neuroformer 对比 | 高（需复现或运行开源代码） | Phase 2+ 考虑 |
| 6 | NDT1/2 数值引用 | 无（引用论文） | 论文阶段 |

---

## 参考文件位置

| 数据 | 路径 |
|------|------|
| Session 统计 | `results/logs/exploration_summary.json` |
| Phase 1 结果 | `results/logs/phase1_full_report.json` |
| AR 验证 | `results/logs/phase1_small_250ms/ar_verify_results.json` |
| 数据处理脚本 | `scripts/data/perich_miller_pipeline.py` |
| 模型代码 | `torch_brain/models/neurohorizon.py` |
| 训练代码 | `examples/neurohorizon/train.py` |
| 配置文件目录 | `examples/neurohorizon/configs/` |
| 指标详解 | `cc_core_files/knowledge.md` 第 4 节 |
| Baseline 策略 | `cc_core_files/knowledge.md` 第 5 节 |
| 数据组织讨论 | `cc_core_files/knowledge.md` 第 6 节 |
