# NeuroHorizon 模型架构演进

> 本文档记录模型架构的演进历程和每次改进想法的讨论分析。
> 每个改进想法都有独立小节，标注日期和状态。
>
> **相关文档**：
> - 执行计划：`cc_core_files/plan.md` §1.9
> - 实验效果追踪：`cc_todo/phase1-autoregressive/1.9-module-optimization/results.tsv`
> - 优化趋势图：`results/figures/phase1-autoregressive-1.9-module-optimization/optimization_progress.png`

---

## 模型版本总结

### v1 — 基础自回归架构 (2026-03-02)

> Git commit: `bb9439d` | 分支: main
> 对应 plan.md: §1.1.1–§1.1.6
> 变更: 7 files changed, 669 insertions, 15 deletions

**核心模块**：
- `torch_brain/nn/loss.py` — PoissonNLLLoss（Poisson 负对数似然损失）
- `torch_brain/nn/autoregressive_decoder.py` — AutoregressiveDecoder + PerNeuronMLPHead
- `torch_brain/models/neurohorizon.py` — NeuroHorizon 主模型（384 行，encoder+processor+AR decoder+tokenize+generate）
- `torch_brain/nn/rotary_attention.py` — causal mask 支持（2D/3D/4D mask + create_causal_mask）
- `torch_brain/registry.py` — spike_counts 模态注册

**架构概要**：

```
输入: spike events (timestamps + unit_index)
  ↓
POYO+ Encoder-Processor（预训练，提取上下文表征）
  ↓ context embeddings [B, S, D]
AutoregressiveDecoder
  ├─ bin_query: learned positional embeddings [T_pred, D]（每个时间步一个 query）
  ├─ Cross-Attention: bin_query attends to context embeddings
  ├─ Causal Self-Attention: bin_query 之间因果注意力（每步只能看到过去）
  └─ 多层堆叠
  ↓ decoded features [B, T_pred, D]
PerNeuronMLPHead
  ├─ 每个 neuron 独立的 2-layer MLP
  └─ 输出 log firing rate
  ↓
输出: log_rate [B, T_pred, N_units]
```

**训练方式**：
- Teacher Forcing: bin_query 是固定的 learned embedding，不包含上一步预测信息
- 损失函数: PoissonNLLLoss，在预测窗口的所有 bins 上计算
- 训练时所有 bins 并行计算（causal mask 确保因果性）

**推理方式**：
- 与训练完全相同（bin_query 无状态，不依赖上一步输出）
- 所有预测 bins 一次并行生成

**已知问题 — TF=AR**：
- bin_query 是无状态的 learned embedding，causal mask 仅限制 hidden state 可见性
- 推理时模型并未利用自己之前步骤的预测结果
- 本质是"带因果掩码的并行预测"，而非"真正的自回归生成"
- 这限制了模型在长时程预测中利用自身预测反馈的能力

### v2 — AR 修复框架 + 评估增强 (2026-03-11)

> Git commit: `e5dea0a` | 分支: main
> 对应 plan.md: §1.1.7–§1.1.9
> 变更: 16 files changed, 1338 insertions, 93 deletions

**新增模块**：
- `torch_brain/utils/neurohorizon_metrics.py`（341 行）— fp-bps, PSTH-R², r2_score, null model 计算
- `torch_brain/nn/prediction_feedback.py`（169 行）— 4 种预测反馈编码方案
- `torch_brain/data/trial_sampler.py`（95 行）— TrialAlignedSampler
- `torch_brain/data/dataset.py` — get_trial_intervals() 扩展
- `scripts/analysis/neurohorizon/eval_psth.py`（194 行）— PSTH 分析脚本

**架构改动**：
- `autoregressive_decoder.py`: forward() 接受 feedback 参数，支持将预测反馈注入 bin_query
- `neurohorizon.py`: forward() 支持 target_counts 教师强制反馈；generate() 支持逐步预测反馈
- `train.py`: 集成 fp-bps 到验证步骤，支持 trial-aligned 数据加载

**Prediction Feedback 方案**（已实现框架，待对比验证）：

| 方案 | 描述 | 实现状态 |
|------|------|---------|
| Scheme A — Query Augmentation | 将上一步预测/真值编码后拼接到 bin_query | 已实现 |
| Scheme B — Hidden State Injection | 将反馈注入 decoder 中间层 | 未实现 |
| Scheme C — Input Concatenation | 将反馈作为额外输入通道 | 未实现 |

**Scheme A 的 4 种编码方法**：
- `mlp_pool`: MLP 编码 + 全局池化，将 N 维 spike counts 压缩为 D 维
- `rate_weighted`: 以 firing rate 加权平均 neuron embedding
- `cross_attn`: 交叉注意力机制聚合 neuron 信息
- `none`: 不反馈（退化为 v1）

**数据加载增强**：
- `TrialAlignedSampler`: 按 trial 对齐采样，以 go_cue_time 为锚点
- `get_trial_intervals()`: 从数据集中提取 trial 级别的时间区间
- 支持两种训练模式: 连续滑动窗口 / trial-aligned

**评估指标**：
- `fp_bps()`: forward prediction bits per spike（相对 null model 的信息增益）
- `psth_r2()`: 群体 PSTH 决定系数（8 方向 trial-averaged）
- `r2_score()`: 单 trial R²
- `compute_null_rates()`: 从训练集计算 per-neuron 平均发放率

**实验结果（1.3.4 v2, 连续训练模式, obs=500ms, 10 sessions, 300 epochs）**：

| pred_window | fp-bps | R² | PSTH-R² |
|-------------|--------|------|---------|
| 250ms | 0.2115 | 0.2614 | 0.6826 |
| 500ms | 0.1744 | 0.2368 | 0.1475 |
| 1000ms | 0.1317 | 0.2290 | 0.2139 |

**Benchmark 对比（同条件: obs=500ms, 连续采样, 10 sessions）**：

| Model | 250ms fp-bps | 500ms fp-bps | 1000ms fp-bps | Params |
|-------|-------------|-------------|--------------|--------|
| **NeuroHorizon v2** | **0.2115** | **0.1744** | **0.1317** | ~2.1M |
| Neuroformer | 0.1856 | 0.1583 | 0.1210 | ~4.9M |
| IBL-MtM | 0.1749 | 0.1531 | 0.1001 | ~10.7M |
| NDT2 | 0.1691 | 0.1502 | 0.1079 | ~4.8M |

**NeuroHorizon v2 优势**：
- vs NDT2: +25.1% / +16.1% / +22.0%
- vs Neuroformer: +14.0% / +10.2% / +8.9%
- vs IBL-MtM: +20.9% / +13.9% / +31.6%
- 以最小参数量（~2.1M）在所有预测窗口上取得最佳表现

---

## 模型改进记录

（按时间倒序排列，新的改进想法添加在此处）

<!-- 模板（每次新优化想法提出时，复制此模板并填写）:

### {YYYY-MM-DD} — {改进名称}

> 状态: 提出 / 实施中 / 验证中 / 已合并 / 已放弃
> 分支: `dev/{YYYYMMDD}_{module_name}`
> cc_todo: `cc_todo/phase1-autoregressive/1.9-module-optimization/{YYYYMMDD}_{module_name}.md`
> commit: （实施后填写）

**想法描述**：
（简要描述改进的核心想法）

**动机与目的**：
（为什么要做这个改进？解决什么问题？预期效果？）

**相比现有方案的改动点**：
（具体哪些代码/模块需要改动？改动幅度估计？）

**批判性分析**：
- 优点：...
- 缺点/风险：...
- 替代方案：...
- 预期影响：...

**修改方案**：
（详细的代码修改方案，包括涉及的文件和关键逻辑）

**基本功能验证方案**：
（如何验证代码改动功能正常？最小测试用例？）

**实验结果摘要**：（验证后填写）
| pred_window | fp-bps | R² | vs baseline |
|-------------|--------|------|------------|
| 250ms | | | |
| 500ms | | | |
| 1000ms | | | |

-->
