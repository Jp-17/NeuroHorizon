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
