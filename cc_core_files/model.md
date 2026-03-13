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

### 2026-03-13 — Local Prediction Memory Decoder

> 状态: 已放弃
> 分支: `dev/20260313_local_prediction_memory`
> cc_todo: `cc_todo/phase1-autoregressive/1.9-module-optimization/20260313_local_prediction_memory.md`
> commit: `22faac6`

**想法描述**：
保留 structured prediction memory 的思想，但将其收缩为“local-only”反馈：每个 query bin 只访问紧邻上一步的 `K` 个 prediction memory tokens，不再检索整段历史 memory。

**动机与目的**：
- `20260312_prediction_memory_decoder` 已证明：高容量全历史 memory 在 teacher forcing 下很好用，但 rollout 稳定性明显差于 `baseline_v2`。
- 下一轮优化优先做“收缩显式 feedback 通路”，而不是继续增强它。
- 目标是在保留结构化上一时刻 population state 表征的同时，降低 exposure bias 和 error accumulation。

**相比上一版的关键变化**：
- 保留 `PredictionMemoryEncoder` 和 `K=4` summary tokens
- query `t` 只允许访问 block `t`，其中：
  - `memory[t] = encode(counts[t-1])`
  - 更早历史交由 causal self-attention 负责
- local memory 的时间嵌入改为与 source bin 对齐，而不是沿用 query 当前 bin 的时间

**为什么这样可能更合理**：
- 上一版真正的问题不是“有没有 prediction memory”，而是“prediction memory 过强且能访问整个历史”。
- local-only memory 更像一个结构化的 `c_{t-1}` 条件输入，而不是整段 teacher-forced 侧信道。
- decoder 原本就有 causal self-attention，可以负责更早时间的历史累积；显式 memory 没必要重复承担这个职责。

**批判性分析**：
- 优点：
  - 保留 structured feedback，但显著降低信息通路容量
  - 更接近经典 AR 的局部条件建模
  - 改动集中在 memory mask 和 time alignment，工程风险较低
- 风险：
  - 如果主要问题是 GT counts 与 predicted expected counts 的分布不一致，local mask 只能部分缓解
  - 如果 decoder 仍强依赖 local memory，本质的 rollout 偏移仍可能存在
- 替代方案：
  - 在 memory 输入上加 noise / dropout
  - 在训练中混入 predicted counts，而不是纯 teacher forcing
  - 回退到更强 bottleneck 的 query augmentation

**修改方案**：
- 新增 `decoder_variant='local_prediction_memory'`
- 为 local 版本单独构造 block-diagonal prediction-memory mask
- 为 local 版本使用 shift-right 后的 source-bin time embedding
- 新增独立配置和验证脚本，先做功能验证和 250ms smoke run

**基本功能验证方案**：
- local mask 满足：query `t` 只能访问第 `t` 个 memory block
- `shift-right` 语义保持不变
- `forward()` 与 `generate()` 仍不等价
- 250ms smoke run 能完成训练、保存 checkpoint、跑 rollout eval

**正式实验结果（300 epochs, rollout eval）**：

| pred_window | teacher-forced fp-bps | rollout fp-bps | vs baseline_v2 | vs 20260312 |
|-------------|-----------------------|----------------|----------------|-------------|
| 250ms | 0.2869 | 0.1621 | -0.0494 | +0.0135 |
| 500ms | 0.2846 | -0.0105 | -0.1849 | +0.0048 |
| 1000ms | 0.2732 | -0.2122 | -0.3439 | +0.0468 |

**关键观察**：
- local-only memory 相比 `20260312_prediction_memory_decoder` 确实带来了小幅 rollout 改善，尤其 `1000ms` 从 `-0.2590` 提升到 `-0.2122`。
- 但改进幅度不足以改变结论：相对 `baseline_v2`，三个窗口仍然全部退化。
- teacher-forced / rollout gap 仍然很大：
  - `250ms`: `0.1248`
  - `500ms`: `0.2951`
  - `1000ms`: `0.4853`
- long horizon 的 rollout 仍明显崩塌：
  - `500ms` 从 bin `12` 开始转负
  - `1000ms` 从 bin `11` 开始转负
  - `250ms` 虽然没有转负，但后段已明显衰减到接近零增益

**为什么它仍然比 baseline_v2 差**：
- 第一，local-only mask 只削弱了“全历史 memory 检索”，但没有消除最核心的 train/inference mismatch：训练时 memory 输入仍是 `log1p(GT counts)`，推理时仍是 `log1p(predicted expected counts)`。
- 第二，decoder 仍然存在一条显式、容量不小的 prediction-memory 通路。即使只看上一步，模型依旧会倾向依赖这条局部 teacher-forced 侧信道，而不是仅依赖 history latents 和 hidden state。
- 第三，结果说明“full-history retrieval” 不是唯一问题。把 memory 收缩到 local block 后，rollout 稳定性有所改善，但核心 exposure bias 仍然存在，尤其在 `500ms/1000ms` 长窗口里仍会持续累积。
- 第四，`baseline_v2` 的单向量 bottleneck 虽然表达力较弱，但反而更稳；当前显式 structured memory 仍然没有把“更强的反馈通路”转化成“更强的自由 rollout”。

**结论**：
- `Local Prediction Memory Decoder` 比上一轮 `20260312_prediction_memory_decoder` 更合理，但仍不足以替代 `baseline_v2`。
- 本轮 1.9 迭代结论是：local-only structured memory 只能部分缓解 error accumulation，不能根治 prediction feedback 的分布偏移问题。
- 该方案保留为已验证但放弃的迭代记录，不合并为主线。
- 下一轮如果继续推进，应优先动训练/推理对齐策略，而不是继续只在 memory 可见性上做结构收缩。

### 2026-03-12 — Structured Prediction Memory Decoder

> 状态: 已放弃
> 分支: `dev/20260312_prediction_memory_decoder`
> cc_todo: `cc_todo/phase1-autoregressive/1.9-module-optimization/20260312_prediction_memory_decoder.md`
> commit: `ebb59fa`

**想法描述**：
将 Phase 1 的主线 AR decoder 正式定为 `event-based POYO encoder + time-bin autoregressive decoder + structured prediction memory`。保持 history 侧输入仍为 spike events，不引入 spike-event 级 decoder；输出固定为未来 `T bins x N units` 的 spike counts。decoder 在时间维做 bin-by-bin 自回归，在每个 bin 内并行预测全部 neuron。

**动机与目的**：
- 明确问题定义：NeuroHorizon 当前要解决的是未来时间窗内的 `binned spike counts` 预测，而不是逐 spike event 生成。
- 保留 POYO 优势：history 侧继续使用 event-based encoder，不为了 AR 破坏现有输入建模方式。
- 解决 v2 的信息瓶颈：v2 虽然引入了 `feedback_method`，但主线实现本质上仍是把上一时刻全 population 的反馈压成单个向量后再加到 bin query 上，结构上偏“挤”。
- 给后续迭代留接口：第一版先把“结构化 prediction memory”接通，后续再决定是否增大 `K`、是否在 decoder 更深层使用 memory、是否再做 scheduled sampling。

**现有 v2 baseline 回顾：旧 `feedback_method` 逻辑是什么**
- v2 的 decoder 主体仍是 `learnable bin query + history cross-attn + causal self-attn + PerNeuronMLPHead`。
- 新增的 `feedback_method` 是一个可选反馈编码器接口，训练时对 `target_counts` 做 `shift-right`：
  - `feedback[t] = encode(counts[t-1])`
  - `feedback[0] = 0`
- 然后将得到的 `[B, T, D]` 反馈向量直接加到 `bin_query` 上。这就是当前文档中的 Query Augmentation。
- `feedback_method` 提供 4 个编码方案：`mlp_pool` / `rate_weighted` / `cross_attn` / `none`。
- 这套逻辑不是错误，而是一个合理的 v2 baseline / ablation：它验证“显式 prediction feedback 是否有价值”，但它把 `N` 个 neuron 的上一时刻状态压缩成了单个 `D` 维向量，信息瓶颈偏强。

**为什么旧 `feedback_method` 现在还有用**
- 旧逻辑不删除，保留为 baseline / ablation。
- `decoder_variant='query_aug'` 时，继续复用旧 `feedback_method` 路径，便于和 `prediction_memory` 做一对一比较。
- 在新模型中，旧逻辑不再代表主线架构，只用于：
  - 回归测试：确认本次重构没有破坏 v2 baseline；
  - 消融实验：比较“单向量反馈”与“结构化 memory”谁更有效。

**主线新架构**

```
History spike events
  -> POYO event encoder / processor
  -> history latents

Future bin queries
  -> history cross-attn
  -> prediction-memory cross-attn
  -> causal self-attn
  -> FFN
  -> PerNeuronMLPHead
  -> log_rate[t, n]
```

- 输入端保持不变：history window 仍然输入 spike events、timestamp、unit id。
- 输出端固定为未来 `T bins x N units` 的 spike counts，不做 spike-event decoder。
- 自回归只沿时间维展开，不做 neuron-by-neuron AR。
- 每个 future bin 仍保留一个 learnable bin query 和 rotary time embedding。
- 新增 `prediction_memory` 作为第二条 memory 通路，而不是只把上一时刻 feedback 压成一个向量后直接加到 query 上。

**`shift-right` 到底是什么意思**
- 目标是保证 query bin `t` 看到的永远是 `< t` 的信息，而不是当前步真值。
- 在训练时，teacher forcing 使用 GT counts，但必须先做 `shift-right`：
  - memory slot `0` 用零向量初始化；
  - memory slot `t` 编码的是 `counts[t-1]`，而不是 `counts[t]`。
- 这样 query `t` 访问 memory `t` 时，实际拿到的是上一 bin 的 population counts。
- 对应到 rollout：
  - 第一步只有零初始化 memory；
  - 预测出 bin `0` 后，再把其 predicted expected counts 编码成下一步可访问的 memory。
- 这样 teacher forcing 和 generate 的因果结构一致，避免把当前步 GT 直接泄漏给当前步。

**`prediction memory` 的设计**
- 对上一 bin 的每个 neuron 构造：
  - `h_{t-1,n} = MLP([unit_emb_n ; log1p(count_{t-1,n})])`
- 然后使用 `K=4` 个 learned pooling queries 对 `{h_{t-1,n}}` 做 attention pooling，得到 `K=4` 个 summary tokens。
- 这 4 个 tokens 共同表示“上一 bin 的 population state”，比单个反馈向量保留更多结构信息，但又不至于退化成完整的 per-neuron memory 序列。
- 本次固定 `K=4`，原因：
  - 先控制计算量和实现复杂度；
  - 给后续 1.9 迭代保留清晰的超参数扩展方向；
  - 对 Phase 1 当前的 10-session 规模已经足够做概念验证。

**为什么不选另外两条路**
- 不选 event-level decoder：
  - 当前目标是 count/rate prediction，event-level 生成会显著增加目标空间和训练不稳定性；
  - 与 Poisson NLL、fp-bps、PSTH-R2 的对齐也更差。
- 不选“把预测 counts 重新编码成 spike tokens 再送回 encoder”：
  - 计算太重；
  - 语义不自然；
  - 会把已经清晰定义的 count prediction 问题重新复杂化。
- Query Augmentation 保留，但降级为 baseline，不再作为最终主线。

**训练 / 推理数值语义**
- 模型被定义为 `autoregressive mean-rate predictor`，不是 stochastic spike generator。
- 训练时：
  - 使用 `shift-right` 后的 GT counts；
  - 进入 feedback / prediction memory encoder 前统一做 `log1p(count)` 压缩。
- 推理时：
  - 使用上一 bin 的 `predicted expected counts = exp(log_rate)`；
  - 同样先做 `log1p(pred_count)` 再编码；
  - 不做 Poisson sampling。
- 第一版不引入 scheduled sampling，避免把架构收益和训练 trick 混在一起。

**相比现有方案的改动点**
- `torch_brain/models/neurohorizon.py`
  - 新增 `decoder_variant`，区分 `query_aug` 与 `prediction_memory`
  - 新增 prediction-memory 的训练/rollout 组装逻辑
- `torch_brain/nn/prediction_memory.py`
  - 新增 `PredictionMemoryEncoder`
- `torch_brain/nn/autoregressive_decoder.py`
  - decoder 层内顺序调整为：
    - history cross-attn
    - prediction-memory cross-attn
    - causal self-attn
    - FFN
- `examples/neurohorizon/train.py`
  - 按 `model.requires_target_counts` 自动传入 `target_counts`
- `scripts/analysis/neurohorizon/eval_phase1_v2.py`
  - 支持在 teacher-forced / rollout 两种模式下评估

**批判性分析**
- 优点：
  - 比 v2 单向量 feedback 更接近真正的结构化 AR；
  - 保留 POYO event encoder，不推翻现有主干；
  - 输出仍是 count prediction，训练和评估路径清楚；
  - 保留旧 `feedback_method`，方便做 baseline 和回归测试。
- 缺点/风险：
  - 训练验证如果只看 teacher forcing，可能高估真实 rollout 质量；
  - `K=4` 是工程上先固定的折中值，未必最优；
  - 新增一条 memory cross-attn 后，长窗口训练的显存和时延会略增。
- 替代方案：
  - 继续用 Query Augmentation，仅做更强的 feedback encoder；
  - 做 hidden-state injection；
  - 做 event-level decoder 或 spike-token re-encoding（当前不推荐）。
- 预期影响：
  - 如果 AR feedback 真有价值，提升应主要体现在 longer horizon，尤其是 500ms / 1000ms 的后段 bins。

**修改方案**
- 模型实现：引入 `PredictionMemoryEncoder` 和 `decoder_variant='prediction_memory'`
- 配置实现：新增 250/500/1000ms 三套 prediction-memory train config
- 验证脚本：新增最小功能验证脚本，检查 shape、shift-right、TF/AR 不再等价
- 实验脚本：新增批量训练与结果汇总脚本

**基本功能验证方案**
- `PredictionMemoryEncoder` 输出 shape 应为 `[B, 4, D]`
- 改动 `target_counts[:, 0, :]` 应只影响 `bin >= 1` 的 teacher-forced 输出，不影响 `bin 0`
- `forward()` 与 `generate()` 在同一 batch 上应不再数值等价
- rollout 中修改某一步 predicted counts 后，应只影响之后 bins

**实验计划**
- 数据：`examples/neurohorizon/configs/dataset/perich_miller_10sessions.yaml`
- 采样：连续滑动窗口
- 观察窗口：500ms
- 预测窗口：250ms / 500ms / 1000ms
- 核心指标：fp-bps / R2 / PSTH-R2 / Poisson NLL
- 主对比对象：`baseline_v2`（v2 query augmentation baseline）和 benchmark 三模型

**实验结果摘要**：

| pred_window | teacher-forced fp-bps | rollout fp-bps | vs baseline_v2 |
|-------------|-----------------------|----------------|----------------|
| 250ms | 0.2979 | 0.1486 | -0.0629 |
| 500ms | 0.2832 | -0.0153 | -0.1897 |
| 1000ms | 0.2776 | -0.2590 | -0.3907 |

**关键观察**：
- teacher forcing 下，prediction memory 版本显著高于 `baseline_v2`，说明模型容量和训练拟合能力都不是瓶颈。
- rollout 下，性能随着预测窗口增长快速恶化：
  - `250ms` 从第 `10` 个 bin 开始出现负 fp-bps
  - `500ms` 从第 `11` 个 bin 开始出现负 fp-bps，并在后段持续恶化
  - `1000ms` 从第 `9` 个 bin 开始出现负 fp-bps，后半段大范围为负
- teacher-forced 与 rollout 的 gap 随窗口快速放大：
  - `250ms`: `0.1493`
  - `500ms`: `0.2985`
  - `1000ms`: `0.5366`

**为什么它比 baseline_v2 更差**
- 第一，当前 structured prediction memory 的信息通路太强。相比 `baseline_v2` 的单向量 Query Augmentation，这版允许 decoder 通过 cross-attn 访问高容量的多 token population memory。训练时模型很容易学会依赖“上一时刻 GT population state”的强提示，因此 teacher-forced 指标很高。
- 第二，训练 / 推理分布偏移被放大。训练时 memory 输入是 `log1p(GT integer counts)`；推理时输入是 `log1p(predicted expected counts)`。前者是稀疏离散、后者是平滑连续，memory encoder 在两种输入分布上的行为差异比 `baseline_v2` 更大。
- 第三，当前设计让 query 可以访问完整历史 memory 序列，而不仅是紧邻上一步。这个高容量历史检索在 teacher forcing 下是优势，但在 rollout 下会把早期误差持续传播并放大，尤其在 `500ms / 1000ms` 长窗口中表现明显。
- 第四，decoder 已经有 causal self-attention 负责时间累积，再叠加全历史 prediction-memory cross-attn 后，模型更容易把显式 memory 当成主通路，而不是把 history latents 和自身 hidden state 当成主通路。结果是训练时“看起来更强”，自由 rollout 时却更脆。

**结论**：
- `Structured Prediction Memory Decoder` 作为主线方案不成立，原因不是“无法训练”，而是“teacher forcing 拟合很强，但 rollout 稳定性显著劣于 baseline_v2”。
- 这版保留为失败但有价值的 1.9 迭代记录，不合并为主线。
- 下一轮优化应优先收缩或约束 prediction feedback 通路，降低对 GT memory 的依赖，而不是继续增加 memory 容量。


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
