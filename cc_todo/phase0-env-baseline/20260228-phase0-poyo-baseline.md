# Phase 0.3 POYO+ 基线复现

**日期**：2026-02-28
**对应 plan.md 任务**：Phase 0 → 0.3.1 / 0.3.2 / 0.3.3
**任务目标**：在 Perich-Miller 10 sessions 上运行 POYO+ 行为解码，验证 R² > 0.3；分析 encoder latent 质量；记录基线性能报告

---

## 执行记录

### 0.3.1 POYO+ 训练（cursor_velocity_2d，Perich-Miller 10 sessions）

#### 数据情况
- 已处理的 10 个 sessions：4 C + 3 J + 3 M（center_out_reaching）
- 路径：`data/processed/perich_miller_population_2018/`
- 每个 session 约 41 units，总训练窗口约 6372 个（1s each）

#### 新建文件
- 数据集配置：`examples/poyo_plus/configs/dataset/perich_miller_10sessions.yaml`
- 模型配置：`examples/poyo_plus/configs/model/poyo_baseline.yaml`（dim=128, depth=12）
- 训练配置：`examples/poyo_plus/configs/train_baseline_10sessions.yaml`

#### 训练参数
- 模型：POYOPlus，dim=128，depth=12，cross_heads=2，self_heads=8，~8M params
- epochs：500，batch_size=64，BF16，lr=3.125e-5（scaled by batch_size）
- UnitDropout：min_units=30，max_units=200，mode_units=80
- Wandb：禁用
- 输出目录：`results/logs/phase0_baseline/`

#### 训练命令
```bash
cd /root/autodl-tmp/NeuroHorizon
conda run -n poyo --no-capture-output python examples/poyo_plus/train.py \
    --config-name=train_baseline_10sessions \
    hydra.job.chdir=false \
    > results/logs/phase0_baseline/train.log 2>&1 &
```

#### 训练过程（持续更新）

| epoch | avg R² | 备注 |
|-------|--------|------|
| 9 | 0.321 | 初始验证 |
| 89 | 0.784 | LR 加热完成 |
| 229 | 0.803 | LR 衰减开始（epoch 250） |
| **429** | **0.807** | **最佳 checkpoint** |
| 499 | 0.805 | 训练完成 |

---

### 0.3.2 Encoder Latent 质量分析

（训练完成后执行）

- 脚本：`scripts/analysis/analyze_latents.py`
- 方法：从 val set 提取 encoder latent，PCA 可视化 + 线性解码探针（R²）
- 输出：`results/figures/baseline/latent_pca.png`，`results/figures/baseline/linear_probe_r2.txt`

---

### 0.3.3 基线性能报告

**已完成** — 结果记录于 cc_core_files/results.md，plan.md 已打勾

---

## 遇到的问题及解决方法

（持续更新）

---

## 还有什么没有做

- [x] 0.3.2 latent 分析
- [x] 0.3.3 性能报告
