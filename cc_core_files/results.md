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
  - PC1 占 53.8% 方差，说�� latent 空间具有强主成分结构，并非随机——encoder 学到了系统性的神经活动特征
  - Linear probe R²=0.286 是从**均值池化的窗口 latent**预测 cursor velocity 的结果，显著低于模型的 R²=0.807
    - 原因：均值池化丢失了时序信息；decoder 使用的是每时间步的 per-latent 特征
    - 意义：latent 空间编码的并非简单的运动方向信息，而是包含了复杂的时序模式
  - 后续 Phase 2 中，IDEncoder 替换后可对比 latent 结构的变化（更好的分离性 = 更好的跨 session 泛化）
