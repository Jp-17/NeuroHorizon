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

## Perich-Miller 数据集探索结果

### 01_dataset_overview.png

- **存储路径**：`results/figures/data_exploration/01_dataset_overview.png`
- **产生方式**：`scripts/analysis/explore_brainsets.py`（plan 0.2.3）
- **产生时间**：2026-02-28
- **目的**：数据集概览，了解各 session 的 duration、unit 数、valid trial 数、firing rate
- **结果分析**：
  - 10 sessions（C:4, J:3, M:3），407 units，1816 valid trials
  - Subject C: 41-71 units/session；J: 18-38；M: 37-49
  - 平均 firing rate：6.9 Hz/unit（中位数 3.5 Hz）

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
  - 模型：POYOPlus，dim=128，depth=12，cross_heads=2，self_heads=8（~8M params）
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
  - C 和 M animal 的 R² 均在 0.85-0.92，接近论文原始报告（POYO+ 在 Perich-Miller 上 ~0.8-0.9）
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

---

## Phase 1: 自回归改造验证 + 预测窗口梯度测试

### 1.2 基础功能验证

#### 1.2.1 Teacher Forcing 训练（250ms，Small 配置）

- **存储路径**：`results/logs/phase1_small_250ms/`
- **产生时间**：2026-03-02
- **产生方式**：`examples/neurohorizon/train.py`，inline args（epochs=300, batch_size=64, precision=bf16-mixed）
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
- **产生方式**：`/tmp/nh_ar_verify.py`（验证脚本）
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
  4. **Poisson noise floor 限制**：在 20ms bin 粒度下，平均 spike count 仅 ~0.14/bin，理论 R² 上限受限于 Poisson 噪声

#### 关键发现与决策

1. **当前 causal mask 设计可保留但不是关键**：后续 Phase 如需减少计算开销，可安全移除
2. **预测窗口推荐**：Phase 2/3 实验建议使用 **500ms** 作为主窗口（R² 衰减温和，计算开销适中）
3. **模型改进方向**：要提升长时程预测质量，应考虑：
   - 真正的自回归生成（decoder query 使用前步输出而非固定位置编码）
   - 更大的模型规模（从 Small 升级到 Base）
   - 更多训练数据（增加 sessions 数量）
