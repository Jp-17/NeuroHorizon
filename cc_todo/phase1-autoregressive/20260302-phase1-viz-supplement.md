# 20260302-phase1-viz-supplement — Phase 1 可视化补充

- **日期**：2026-03-02
- **对应 plan.md 任务**：Phase 1.2/1.3 可视化补充（回溯补全）
- **任务目标**：为 Phase 1 创建所有缺失的可视化图表，并在 results.md 中补充逐子图解读

## 执行记录

### 1. 缺失可视化识别

Phase 1 完成了 4 组实验（250ms/500ms/1000ms AR + 1000ms non-AR），有完整的 JSON 数据和 metrics.csv，但**没有任何可视化图表**。

### 2. 创建可视化脚本

**脚本**：`scripts/analysis/neurohorizon/phase1_visualize.py`

```bash
conda activate poyo
cd /root/autodl-tmp/NeuroHorizon
python scripts/analysis/neurohorizon/phase1_visualize.py
```

**生成 4 张图**：

| 图 | 路径 | 内容 |
|----|------|------|
| 01 | `results/figures/phase1/01_training_curves.png` | 4 组实验 Val Loss + R2 训练曲线 |
| 02 | `results/figures/phase1/02_r2_vs_window.png` | R2 随窗口长度衰减 + Val Loss 对比 |
| 03 | `results/figures/phase1/03_per_bin_r2.png` | 250ms 逐 bin R2 和 NLL 分析 |
| 04 | `results/figures/phase1/04_ar_vs_noar.png` | 1000ms AR vs non-AR 详细对比 |

### 3. results.md 逐子图解读

为 4 张新图添加了完整的逐子图解读和交叉引用：
- 01: 左图 Val Loss 收敛分析，右图 R2 收敛分析
- 02: 左图 R2 亚线性衰减模式，右图 Val Loss 一致性
- 03: 左图 per-bin R2 波动模式，右图 NLL 分布
- 04: 左图 AR/non-AR 曲线重叠，右图差值 < 0.002

### 4. scripts.md 更新

添加了 `phase1_visualize.py` 条目。

## 结果

- Phase 1 从 0 张可视化图增加到 4 张
- 涵盖：训练收敛、窗口长度影响、逐 bin 分析、AR vs non-AR 对比
- 所有图均有逐子图解读 + 交叉引用
