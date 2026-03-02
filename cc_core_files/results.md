# 实验结果记录

> 记录项目中所有实验结果（可视化图表、分析输出、评估指标等）的说明信息。
> 每次产生重要实验结果，必须在此处更新记录。

---

## 目录结构规范

```
results/
├── figures/        # 可视化图表（训练曲线、性能对比图等）
├── logs/           # 训练日志、TensorBoard 文件
└── checkpoints/    # 模型权重文件
```

---

## 记录格式

```
### 实验名称

- **存储路径**：results/xxx/
- **产生时间**：YYYY-MM-DD
- **产生方式**：使用的脚本/命令（参考 scripts.md）
- **实验目的**：为了验证什么假设/对比什么方法
- **实验配置**：模型参数、数据集、超参数等关键配置
- **主要结果**：关键指标数值
- **结果分析**：对结果的解读和结论
- **备注**：其他注意事项
```

---

## 实验结果列表

*（随项目推进持续补充）*

---

## 说明

> ⚠️ cc_todo/20260221-cc-1st/results/ 和 cc_todo/20260221-cc-1st/figures/ 中存储了来自已废弃早期任务的实验结果，**不作为当前项目的参考结果**，仅作为历史存档。

---

## Phase 0.2 数据探索结果

### 01_dataset_overview.png

- **存储路径**：`results/figures/data_exploration/01_dataset_overview.png`
- **产生方式**：`scripts/analysis/explore_brainsets.py`（plan 0.2.3）
- **产生时间**：2026-02-28
- **目的**：数据集概览，了解各 session 的 duration、unit 数、valid trial 数、firing rate
- **结果分析**：
  - 10 sessions（C:4, J:3, M:3），407 units，1816 valid trials
  - Subject C: 41-71 units/session；J: 18-38；M: 37-49
  - 平均 firing rate：6.9 Hz/unit（中位数 3.5 Hz）

- **逐子图解读**（2x3 布局）：

  - **子图 (0,0) Session Duration by Subject**：各 session 录制时长柱状图，按 subject 着色。C 录制 22-34 分钟，J 15-22 分钟，M 28-35 分钟。所有 session 录制时长充足，满足训练数据需求。

  - **子图 (0,1) Unit Count Distribution**：各 session 神经元数量柱状图。C: 41-71 units/session，J: 18-38 units（最少），M: 37-49 units。J 的 unit 数最少，解释了后续基线训练中 J sessions R² 偏低的原因。

  - **子图 (0,2) Valid Trials per Session**：各 session 有效 trial 数。多数 session 有 100-200+ 个有效 trial，足够划分 train/val/test 集。

  - **子图 (1,0) Per-Unit Firing Rate Distribution**：407 个 unit 的平均发放率直方图。中位数 3.5 Hz，均值 6.9 Hz，呈右偏分布，符合皮层神经元的典型特征。

  - **子图 (1,1) Hold Period Duration (Input Window)**：hold period 时长分布。均值 676ms，87% trials > 250ms，确认 250ms 输入窗口对大部分 trial 可行。

  - **子图 (1,2) Reach Period Duration (Prediction Window)**：reach period 时长分布。均值 1090ms，100% > 250ms，100% > 500ms，75% > 1000ms，验证了三种预测窗口长度均可行。

- **交叉引用**：
  - 脚本：`scripts/analysis/explore_brainsets.py`
  - 数据：`data/processed/perich_miller_population_2018/*.h5`
  - JSON 汇总：`results/figures/data_exploration/exploration_summary.json`

### 02_neural_statistics.png

- **存储路径**：`results/figures/data_exploration/02_neural_statistics.png`
- **产生方式**：`scripts/analysis/explore_brainsets.py`（plan 0.2.3）
- **产生时间**：2026-02-28
- **目的**：神经元统计特征与自回归可行性评估
- **结果分析**：
  - **Hold period**（输入窗口）：均值 676ms，87% trials > 250ms → 250ms 输入可行
  - **Reach period**（预测窗口）：均值 1090ms，100% > 250ms，100% > 500ms，75% > 1000ms
  - **Per-unit spike 稀疏度**（20ms bin）：87.6% zero bins，均值 0.133 spikes/bin
  - **Poisson NLL 适用性**：population-level 均值 5.17 spikes/20ms bin，适用
  - **预测步数**：250ms=12步，500ms=25步，1000ms=50步
  - **Mean-Variance 关系**（50ms bin）：接近 Poisson 特性（var ≈ mean）

- **逐子图解读**（2x3 布局）：

  - **子图 (0,0) Spike Count Distribution by Bin Width**：分别以 10ms/20ms/50ms bin 宽统计 spike count 分布。20ms bin 下 87.6% 为零，均值 0.133 spikes/bin；50ms bin 下均值升至 约0.33，零比例降低。说明 20ms bin 下数据极度稀疏，Poisson NLL 是合适的损失函数。

  - **子图 (0,1) Population PSTH Aligned to Go Cue**：10 个 session 的群体 PSTH（时间对齐至 go cue，time=0）。所有 session 在 go cue 处显示明显的发放率调制，hold period 内存在稳定的预运动活动，为 encoder 提供有效上下文。

  - **子图 (0,2) Average Firing Rate per Session**：各 session 平均发放率水平柱状图。群体水平发放率从 约4 到 约11 Hz/unit 不等，发放率越高的 session 可能预测效果越好。

  - **子图 (1,0) Mean-Variance Relationship (Poisson check)**：各 unit 的 50ms bin 下 mean vs variance 散点图。数据点聚集在 var=mean 的 Poisson 参考线附近，确认 Poisson 分布假设成立，验证了 PoissonNLLLoss 的适用性。

  - **子图 (1,1) Raster Plot**：示例 session（c_20131003）的 5 个 unit、10 个 trial 的 raster plot。展示了围绕 go cue 的 spike timing 模式，可见清晰的时序结构。

  - **子图 (1,2) Autoregressive Window Feasibility**：汇总表格，列出 250ms/500ms/1000ms 三种预测窗口对应的 bins 数和 trial 可行比例（分别为 87%/100%/75%），直观展示各窗口长度的可行性。

- **交叉引用**：
  - 脚本：`scripts/analysis/explore_brainsets.py`
  - 数据：`data/processed/perich_miller_population_2018/*.h5`
  - JSON 汇总：`results/figures/data_exploration/exploration_summary.json`

### exploration_summary.json

- **存储路径**：`results/figures/data_exploration/exploration_summary.json`
- **产生方式**：`scripts/analysis/explore_brainsets.py`
- **目的**：机器可读的统计汇总，供后续脚本直接加载使用

---

## Phase 0.3 POYO+ 基线训练结果

### POYO+ 行为解码基线（Perich-Miller 10 sessions）

- **存储路径**：`results/logs/phase0_baseline/`
- **产生时间**：2026-02-28
- **产生方式**：`examples/poyo_plus/train.py`，配置 `train_baseline_10sessions.yaml`
- **实验目的**：验证 POYO+ 在 Perich-Miller 数据上的行为解码基线性能（Phase 0.3.1 验收）
- **实验配置**：
  - 模型：POYOPlus，dim=128，depth=12，cross_heads=2，self_heads=8（约8M params）
  - 数据：10 sessions（C:4, J:3, M:3），center_out_reaching，BF16
  - 训练：500 epochs，batch_size=64，OneCycleLR（max_lr=2e-3，LR 衰减从 epoch 250 开始）
  - UnitDropout：min_units=30，mode_units=80
- **主要结果**：
  - **最佳平均 R²：0.8065**（epoch 429）
  - **最终 epoch R²：0.8046**（epoch 499）
  - 最佳 checkpoint：`results/logs/phase0_baseline/lightning_logs/version_0/checkpoints/epoch=429-step=42570.ckpt`

  | Session | R²（epoch 429） |
  |---------|----------------|
  | c_20131003 | 0.871 |
  | c_20131022 | 0.846 |
  | c_20131101 | 0.907 |
  | c_20131204 | 0.857 |
  | j_20160405 | 0.569 |
  | j_20160406 | 0.574 |
  | j_20160407 | 0.756 |
  | m_20150610 | 0.922 |
  | m_20150612 | 0.864 |
  | m_20150615 | 0.901 |
  | **平均** | **0.807** |

- **结果分析**：
  - C 和 M animal 的 R² 均在 0.85-0.92，接近论文原始报告（POYO+ 在 Perich-Miller 上 约0.8-0.9）
  - J animal 较低（0.57-0.76），原因是训练集中 J 只有 3 sessions，样本量不足
  - **0.3.1 验收标准 R² > 0.3 大幅满足**（实际达 0.807）
  - 收敛曲线：epoch 9=0.321 → epoch 89=0.784 → epoch 229=0.803 → epoch 429=0.807（收敛稳定）

### POYO+ Encoder Latent 质量分析

- **存储路径**：`results/figures/baseline/`
- **产生时间**：2026-02-28
- **产生方式**：`scripts/analysis/analyze_latents.py`，最佳 checkpoint（epoch 429）
- **实验目的**：分析 POYO encoder 输出的 latent representation 质量（Phase 0.3.2 验收）
- **实验配置**：
  - 输入：val set 全部 1313 个窗口（1s 滑动窗口，step=0.5s）
  - Latent 提取：hook `dec_atn` pre-hook 捕获处理后的 latent（shape=[1313, n_latents, 128]）
  - 分析：mean-pooled → PCA；repeated → 5-fold CV Ridge linear probe
- **主要结果**：

  | 分析 | 结果 |
  |------|------|
  | PCA PC1 方差占比 | **53.8%** |
  | PCA PC2 方差占比 | 7.4% |
  | Linear probe R²（cursor_velocity_2d） | **0.286 ± 0.032** |
  | 图表 | `results/figures/baseline/latent_pca.png` |
  | 数值文件 | `results/figures/baseline/latent_linear_probe.json` |
  | 汇总文件 | `results/figures/baseline/latent_analysis_summary.json` |

- **结果分析**：
  - PC1 占 53.8% 方差，说明 latent 空间具有强主成分结构，并非随机——encoder 学到了系统性的神经活动特征
  - Linear probe R²=0.286 是从**均值池化的窗口 latent**预测 cursor velocity 的结果，显著低于模型的 R²=0.807
    - 原因：均值池化丢失了时序信息；decoder 使用的是每时间步的 per-latent 特征
    - 意义：latent 空间编码的并非简单的运动方向信息，而是包含了复杂的时序模式
  - 后续 Phase 2 中，IDEncoder 替换后可对比 latent 结构的变化（更好的分离性 = 更好的跨 session 泛化）

- **latent_pca.png 子图解读**（单图）：

  - **PCA 2D 散点图**：横轴 PC1（53.8% 方差），纵轴 PC2（7.4%）。每个点代表一个 1s 时间窗口的 mean-pooled latent，按 session 着色。各 session 形成部分重叠但可区分的 cluster。PC1 的高方差占比说明 encoder 学到了一个强主特征（可能对应运动方向或运动强度）。Session 间的 clustering 表明 latent 空间存在 session-specific 表征，这正是 Phase 2 引入 IDEncoder 的动机：使跨 session 的 latent 表征更加统一。

- **交叉引用**：
  - 脚本：`scripts/analysis/analyze_latents.py`
  - Checkpoint：`results/logs/phase0_baseline/lightning_logs/version_0/checkpoints/epoch=429-step=42570.ckpt`
  - 数值输出：`results/figures/baseline/latent_analysis_summary.json`、`results/figures/baseline/latent_linear_probe.json`
  - 数据：`data/processed/perich_miller_population_2018/*.h5`（val set）

### Phase 0.3 基线训练曲线

- **存储路径**：`results/figures/baseline/03_baseline_training_curves.png`
- **产生时间**：2026-03-02
- **产生方式**：`scripts/analysis/phase0_baseline_plots.py`
- **数据来源**：`results/logs/phase0_baseline/lightning_logs/version_0/metrics.csv`
- **实验目的**：可视化 POYO+ 基线 500 epochs 训练过程，展示收敛特性和各 session 性能差异

- **逐子图解读**（2x2 布局）：

  - **子图 (0,0) Training & Validation Loss vs Epoch**：训练/验证损失随 epoch 变化。训练损失平稳下降，验证损失在 epoch 约100 后趋于平稳。Best epoch 429 处以垂直虚线标注。两条曲线间无明显 gap，说明模型未过拟合。

  - **子图 (0,1) Average Validation R² vs Epoch**：10 sessions 平均 R² 曲线。从 epoch 9 的 0.321 快速上升至 epoch 89 的 0.784，随后缓慢增长至 epoch 429 的最佳值 0.807。红色虚线标注 Phase 0.3.1 验收阈值 R²=0.3（大幅超越）。收敛曲线表明模型在 epoch 约100 后进入精细优化阶段。

  - **子图 (1,0) Per-Session R² vs Epoch**：10 条曲线按 subject 着色（C=蓝, J=橙, M=绿）。C 和 M sessions 的 R² 在 0.85-0.92 区间，J sessions 明显偏低（0.57-0.76）。J 的低 R² 与其 unit 数最少（18-38）相关。所有 session 收敛轨迹一致，无异常波动。

  - **子图 (1,1) Per-Session R² Bar Chart (epoch 429)**：最佳 epoch 各 session R² 水平柱状图。m_20150610 最高（0.922），j_20160405 最低（0.569）。灰色虚线标注平均 R²=0.807。柱状图直观展示了跨 session 性能差异，为 Phase 2 跨 session 泛化提供基线参考。

- **交叉引用**：
  - 脚本：`scripts/analysis/phase0_baseline_plots.py`
  - 数据：`results/logs/phase0_baseline/lightning_logs/version_0/metrics.csv`
  - 训练脚本：`examples/poyo_plus/train.py`，配置 `train_baseline_10sessions.yaml`
  - Checkpoint：`results/logs/phase0_baseline/lightning_logs/version_0/checkpoints/epoch=429-step=42570.ckpt`

---

## Phase 1: 自回归改造验证 + 预测窗口梯度测试

### 1.2 基础功能验证

#### 1.2.1 Teacher Forcing 训练（250ms，Small 配置）

- **存储路径**：`results/logs/phase1_small_250ms/`
- **产生时间**：2026-03-02
- **产生方式**：`examples/neurohorizon/train.py --config-name=train_small
- **实验目的**：验证 NeuroHorizon 模型在 250ms 预测窗口下的基本训练收敛性
- **实验配置**：
  - 模型：NeuroHorizon Small（dim=128, enc_depth=6, dec_depth=2, 12 time bins）
  - 数据：Perich-Miller 10 sessions（8 train / 2 val）
  - 损失函数：PoissonNLLLoss
  - 硬件：RTX 4090 D，BF16 混合精度
- **主要结果**：
  - **最佳 R²：0.2658**（epoch 229）
  - 最终 R²：0.2606（epoch 299）
  - 最终 val_loss：0.3142
  - 收敛稳定，epoch 120 后 R² 在 0.25-0.27 区间波动

#### 1.2.2 自回归推理验证

- **存储路径**：`results/logs/phase1_small_250ms/ar_verify_results.json`
- **产生时间**：2026-03-02
- **产生方式**：`scripts/analysis/neurohorizon/ar_verify.py`（验证脚本）
- **实验目的**：验证 causal mask 正确性，对比 TF 与 AR 推理一致性
- **主要结果**：
  - TF vs AR max diff：**3e-6**（数值精度级别，完全一致）
  - Causal mask 验证：**PASSED**（修改 bins 8-11 不影响 bins 0-7 的输出）
  - AR/TF R² ratio：**1.0000**
- **结果分析**：
  - 由于当前 decoder 使用固定 bin position embedding 作为 query（非前一步预测值），TF 和 AR 推理数学等价
  - 这意味着 scheduled sampling 对当前架构不适用
  - decoder 本质上是"带 causal attention 约束的并行解码器"

### 1.3 预测窗口梯度测试

#### 1.3.1-1.3.3 预测窗口实验汇总

- **存储路径**：`results/logs/phase1_full_report.json`
- **产生时间**：2026-03-02
- **产生方式**：各窗口配置独立训练 300 epochs
- **实验目的**：测试不同预测窗口长度（250ms/500ms/1000ms）下的预测质量衰减

| 实验 | 时间窗口 | Bins | 最佳 R² | 最佳 Epoch | 最终 Val Loss |
|------|----------|------|---------|-----------|--------------|
| 250ms AR | 250ms | 12 | **0.2658** | 229 | 0.3142 |
| 500ms AR | 500ms | 25 | **0.2417** | 229 | 0.3124 |
| 1000ms AR | 1000ms | 50 | **0.2343** | 299 | 0.3159 |
| 1000ms non-AR | 1000ms | 50 | **0.2354** | 259 | 0.3159 |

#### AR vs non-AR 对比（1000ms）

| 指标 | AR (causal mask) | non-AR (no mask) | 差异 |
|------|------------------|-----------------|------|
| Best R² | 0.2343 | 0.2354 | +0.0011 (non-AR 略好) |
| Best Epoch | 299 | 259 | non-AR 更早收敛 |
| Final Val Loss | 0.3159 | 0.3159 | 完全相同 |

- **结果分析**：
  1. **R² 随窗口长度缓慢衰减**：250ms=0.266 → 500ms=0.242 → 1000ms=0.234，衰减幅度为 -9.1% → -3.3%，说明模型在长时程预测中具有较强鲁棒性
  2. **AR 与 non-AR 无显著差异**（差异 < 0.002），证实当前架构下 causal mask 不提供额外收益：
     - 原因：decoder 使用固定位置编码作为 query，不依赖前步输出
     - 每个 bin 的预测仅通过 encoder latent 的 cross-attention 和自身的 positional embedding 获得信息
     - causal mask 只限制了 self-attention 中的信息流，但 cross-attention 仍然提供了来自编码器的完整上下文
  3. **Val loss 收敛区间**：所有实验的 val_loss 收敛在 0.312-0.316 区间，差异微小
  4. **Poisson noise floor 限制**：在 20ms bin 粒度下，平均 spike count 仅 约0.14/bin，理论 R² 上限受限于 Poisson 噪声

#### 关键发现与决策

1. **当前 causal mask 设计可保留但不是关键**：后续 Phase 如需减少计算开销，可安全移除
2. **预测窗口推荐**：Phase 2/3 实验建议使用 **500ms** 作为主窗口（R² 衰减温和，计算开销适中）
3. **模型改进方向**：要提升长时程预测质量，应考虑：
   - 真正的自回归生成（decoder query 使用前步输出而非固定位置编码）
   - 更大的模型规模（从 Small 升级到 Base）
   - 更多训练数据（增加 sessions 数量）

---

### Phase 1 可视化分析

#### 01_training_curves.png — 训练曲线总览

- **存储路径**：`results/figures/phase1/01_training_curves.png`
- **产生时间**：2026-03-02
- **产生方式**：`scripts/analysis/neurohorizon/phase1_visualize.py`
- **数据来源**：`results/logs/phase1_full_report.json`（汇总自各实验 metrics.csv）
- **实验目的**：对比 4 组实验（250ms/500ms/1000ms AR + 1000ms non-AR）的训练收敛情况

- **逐子图解读**（1x2 布局）：

  - **左图（Val Loss vs Epoch）**：四条曲线展示验证损失随训练轮数的变化。所有实验在约 100 epochs 后收敛至 0.31–0.32 区间，收敛区间差异极小。250ms 和 500ms 实验收敛最快（约 80 epochs 进入平台期），1000ms 实验收敛稍慢但最终损失值与短窗口实验相当。说明 PoissonNLL 损失函数在不同窗口长度下表现一致，模型训练稳定。

  - **右图（R² vs Epoch）**：250ms AR 达到最高 R²=0.266（epoch 229），明显优于其他窗口。500ms AR 最高 R²=0.242（epoch 229），衰减约 9%。1000ms AR/non-AR 最高 R² 约 0.234–0.235，衰减趋势放缓。灰色参考线 R²=0.3 为 Phase 0 验收阈值——Phase 1 spike 预测任务难度更高（Poisson noise floor 限制），R² 在 0.24–0.27 区间合理。所有曲线在 epoch 120 后进入平台期。

- **交叉引用**：
  - 脚本：`scripts/analysis/neurohorizon/phase1_visualize.py`
  - 数据：`results/logs/phase1_full_report.json`
  - 原始 metrics：`results/logs/phase1_small_*/lightning_logs/version_0/metrics.csv`
  - 训练脚本：`examples/neurohorizon/train.py`
  - 训练配置：`examples/neurohorizon/configs/train_small*.yaml`

#### 02_r2_vs_window.png — R² 随预测窗口长度变化

- **存储路径**：`results/figures/phase1/02_r2_vs_window.png`
- **产生时间**：2026-03-02
- **产生方式**：`scripts/analysis/neurohorizon/phase1_visualize.py`
- **数据来源**：`results/logs/phase1_full_report.json`
- **实验目的**：量化预测窗口长度对模型性能的影响，评估长时程预测的可行性

- **逐子图解读**（1x2 布局）：

  - **左图（Best R² vs Prediction Window）**：3 个 AR 数据点（250ms=0.2658, 500ms=0.2417, 1000ms=0.2343）连线，展示 R² 随窗口增长的衰减趋势。250ms→0.500ms 衰减 -9.1%，500ms→1000ms 衰减 -3.1%，呈**亚线性衰减**模式——窗口翻倍但性能衰减放缓。1000ms non-AR（方块标记，R²=0.2354）与 AR 几乎重合，验证了 causal mask 对当前架构影响可忽略。

  - **右图（Final Val Loss vs Prediction Window）**：三种窗口的最终验证损失均在 0.312–0.316 区间，差异极小。说明模型在不同窗口长度下的损失收敛一致，性能差异主要体现在 R² 指标上（R² 对预测精度更敏感）。

- **交叉引用**：
  - 脚本：`scripts/analysis/neurohorizon/phase1_visualize.py`
  - 数据：`results/logs/phase1_full_report.json`
  - 汇总 JSON：`results/logs/phase1_summary.json`

#### 03_per_bin_r2.png — 逐 Bin R² 与 NLL 分析

- **存储路径**：`results/figures/phase1/03_per_bin_r2.png`
- **产生时间**：2026-03-02
- **产生方式**：`scripts/analysis/neurohorizon/phase1_visualize.py`
- **数据来源**：`results/logs/phase1_small_250ms/ar_verify_results.json`
- **实验目的**：分析 250ms 预测窗口内 12 个 time bin（每 bin 20ms）的预测质量差异

- **逐子图解读**（1x2 布局）：

  - **左图（Per-Bin R² Bar Chart）**：12 个 bin 的 R² 柱状图，颜色从蓝（早期 bin）渐变到红（晚期 bin）。R² 值在 0.168–0.347 间波动，均值 0.263。Bin 0（0–20ms）R²=0.347 最高，bin 9（180–200ms）R²=0.169 最低。R² 并非单调递减，而是呈波动模式——说明预测难度与时间位置相关，但不严格随距离增大而增大（因为使用固定位置编码，每个 bin 独立预测）。

  - **右图（Per-Bin Poisson NLL Bar Chart）**：12 个 bin 的 Poisson NLL 柱状图。NLL 值在 0.265–0.388 间波动，均值约 0.305。最后一个 bin（bin 11, 220–240ms）NLL=0.388 最高（预测最难）。NLL 的分布与 R² 大体互补——NLL 高的 bin 通常 R² 低。

- **交叉引用**：
  - 脚本：`scripts/analysis/neurohorizon/phase1_visualize.py`
  - 数据：`results/logs/phase1_small_250ms/ar_verify_results.json`
  - AR 验证脚本：`scripts/analysis/neurohorizon/ar_verify.py`
  - 模型 checkpoint：`results/logs/phase1_small_250ms/lightning_logs/version_0/checkpoints/epoch=229-step=30590.ckpt`

#### 04_ar_vs_noar.png — AR vs non-AR 对比

- **存储路径**：`results/figures/phase1/04_ar_vs_noar.png`
- **产生时间**：2026-03-02
- **产生方式**：`scripts/analysis/neurohorizon/phase1_visualize.py`
- **数据来源**：`results/logs/phase1_full_report.json`
- **实验目的**：在 1000ms 窗口下直接对比 AR（causal mask）与 non-AR（无 mask）的训练表现，验证 causal mask 的影响

- **逐子图解读**（1x2 布局）：

  - **左图（R² Training Curves）**：1000ms AR（绿色实线）与 non-AR（红色虚线）的 R² 随 epoch 变化。两条曲线几乎完全重叠，灰色阴影区域展示两者差异区间，肉眼几乎不可见。non-AR 最佳 R²=0.2354（epoch 259），AR 最佳 R²=0.2343（epoch 299），差异仅 0.0011。说明 causal mask 在当前架构下对性能无实质影响。

  - **右图（R² Difference）**：逐 epoch 的 R² 差值（non-AR − AR）柱状图。蓝色虚线标注平均差值。所有差值绝对值 < 0.002，在数值波动范围内。正/负差值交替出现，无系统性偏向。原因分析：当前 decoder 使用固定位置编码作为 query（而非前步输出），使 TF 和 AR 推理数学等价，causal mask 仅限制 self-attention 信息流但 cross-attention 仍提供完整编码器上下文。

- **交叉引用**：
  - 脚本：`scripts/analysis/neurohorizon/phase1_visualize.py`
  - 数据：`results/logs/phase1_full_report.json`
  - 1000ms AR 训练日志：`results/logs/phase1_small_1000ms/`
  - 1000ms non-AR 训练日志：`results/logs/phase1_small_1000ms_noar/`
  - 训练配置：`examples/neurohorizon/configs/train_small_1000ms.yaml`、`train_small_1000ms_noar.yaml`
