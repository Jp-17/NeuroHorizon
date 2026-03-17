# 2026-03-17 QA 调研记录（Claude）

## 任务范围

独立调研并详细分析以下三个问题：

1. 当前项目 `fp-bps` 指标在 neuron / time / batch / session 维度的处理方式，以及 train / val 阶段的计算流程
2. Neuroformer 原始 repo 的 train / inference 范式（teacher forcing vs rollout），观察窗口/预测窗口大小，是否存在 exposure bias
3. IBL-MtM 原始 paper 报告的 forward prediction bps (约0.4) 与我们 base v2 (约0.21) 之间差异的根因分析
4. NeuroHorizon 的数据组织、训练分布、评估/metric 细节、模型架构（AR decoder 和 PerNeuronMLPHead）

## 参考材料

### 项目内代码

| 文件 | 用途 |
|------|------|
| `torch_brain/utils/neurohorizon_metrics.py` | fp-bps 核心实现 |
| `examples/neurohorizon/train.py` | 训练/验证循环中的 metric 计算 |
| `scripts/analysis/neurohorizon/eval_phase1_v2.py` | 离线评估脚本 |
| `neural-benchmark/benchmark_models/neuroformer/neuroformer_train.py` | Neuroformer 训练入口 |
| `neural-benchmark/benchmark_models/neuroformer/neuroformer_inference.py` | Neuroformer 推理入口 |
| `neural-benchmark/benchmark_models/neuroformer/neuroformer/simulation.py` | Neuroformer autoregressive generation |
| `neural-benchmark/benchmark_models/neuroformer/neuroformer/model_neuroformer.py` | Neuroformer 模型和 loss |
| `neural-benchmark/benchmark_models/neuroformer/neuroformer/data_utils.py` | Neuroformer 数据处理 |
| `neural-benchmark/benchmark_models/neuroformer/neuroformer/trainer.py` | Neuroformer 训练循环 |
| `neural-benchmark/benchmark_models/neuroformer/configs/Visnav/lateral/mconf_predict_all.yaml` | Neuroformer 配置文件 |
| `neural-benchmark/benchmark_models/ibl-mtm/src/utils/eval_utils.py` | IBL-MtM 评估工具（含 bits_per_spike） |
| `neural-benchmark/benchmark_models/ibl-mtm/src/models/ndt1.py` | NDT1 模型 |
| `neural-benchmark/benchmark_models/ibl-mtm/src/models/masker.py` | IBL-MtM masking 机制 |
| `neural-benchmark/benchmark_models/ibl-mtm/src/configs/ndt1.yaml` | NDT1 基础配置 |
| `neural-benchmark/benchmark_models/ibl-mtm/src/configs/ndt1_prompting.yaml` | NDT1 prompt 配置 |
| `neural-benchmark/benchmark_models/ibl-mtm/src/configs/trainer_ndt1.yaml` | NDT1 训练器配置 |
| `neural-benchmark/benchmark_models/ibl-mtm/src/finetune_eval_multi_session.py` | IBL-MtM forward pred 评估入口 |
| `cc_core_files/plan.md` | 项目计划（含 base v2 benchmark 表） |
| `torch_brain/models/neurohorizon.py` | NeuroHorizon 主模型（含 tokenize、generate） |
| `torch_brain/nn/autoregressive_decoder.py` | AR Decoder 和 PerNeuronMLPHead |
| `torch_brain/nn/prediction_feedback.py` | Feedback encoder（4 种变体） |
| `torch_brain/nn/prediction_memory.py` | PredictionMemoryEncoder |
| `torch_brain/data/dataset.py` | NeuralDataset（lazy-load HDF5） |
| `torch_brain/data/sampler.py` | RandomFixed / SequentialFixed WindowSampler |
| `torch_brain/data/trial_sampler.py` | TrialAlignedSampler |
| `torch_brain/data/collate.py` | Collation（padding + mask） |

### 外部论文/仓库

- Neuroformer：arXiv 2311.00136
- IBL-MtM：NeurIPS 2024，GitHub `colehurwitz/IBL_foundation_model`

---

## Q1. `fp-bps` 指标的维度处理与 train/val 计算方式

### 1.1 核心公式

`fp-bps` 的定义为：

```
fp-bps = (NLL_null - NLL_model) / (total_spikes * ln2)
```

其中：
- `NLL_model = Sum PoissonNLL(log_rate[b,t,n], target[b,t,n])`，对所有有效 (b,t,n) 求和
- `NLL_null = Sum PoissonNLL(null_log_rate[n], target[b,t,n])`，同样对所有有效 (b,t,n) 求和
- `total_spikes = Sum target[b,t,n]`
- `PoissonNLL(log_lambda, k) = exp(log_lambda) - k * log_lambda`

实现位于 `neurohorizon_metrics.py`：
- `poisson_nll_elementwise()` 计算逐元素 Poisson NLL
- `fp_bps_stats()` 返回三个充分统计量：`nll_model_sum, nll_null_sum, total_spikes`
- `finalize_fp_bps_from_stats()` 做最终除法

### 1.2 Neuron 维度的处理

**关键结论：不做 per-neuron 平均，而是全局 spike-weighted 聚合。**

具体机制：

1. **模型输出** `log_rate` 形状为 `[B, T, N]`，`N` 是 padded neuron 数量
2. **unit_mask** 形状为 `[B, N]`，标记哪些 neuron 是真实的、哪些是 padding
3. 在 `fp_bps_stats()` 中，mask 扩展到 `[B, T, N]` 后，对所有 `True` 位置直接 `.sum()`
4. 这意味着 **firing rate 高、spike 总量大的 neuron 对最终 fp-bps 贡献更大**
5. 低 firing rate neuron 的贡献被 spike 计数自然下压

**null model 是 per-neuron 的**：
- `compute_null_rates()` 遍历训练集，对每个 neuron 统计：
  - 总 spike 数 / 总 bin 数 -> `mean_count_per_bin`
  - 取 `log(max(mean_count_per_bin, 1e-6))`
- neuron 通过 `model.unit_emb.tokenizer` 映射到 **global unit id**（隐含了 session 前缀），不同 session 的同编号 neuron 不会混淆
- 存入 lookup tensor，通过 `target_unit_index` 索引

**与 NLB 社区标准的差异**：NLB（Neural Latents Benchmark）的 co-bps 采用 per-neuron 计算然后取均值。我们当前实现更接近 **spike-weighted global** 口径，这在高低 firing rate neuron 混合时会产生不同结果。

### 1.3 Time 维度的处理

**整体 fp-bps：不区分时间步，直接跨时间累加。**

- `null_log_rates` 从 `[B, N]` unsqueeze 到 `[B, T, N]`，每个 time bin 使用同一个 null rate
- NLL 在所有 T 个 bin 上直接 sum
- 早期 bin（离观察窗口近，更容易预测）和晚期 bin（离观察窗口远，更难预测）对最终数字的影响只取决于各 bin 的 spike 数量

**per-bin fp-bps：按时间步拆开单独计算。**

- `fp_bps_per_bin_stats()` 对每个 `t` 分别累加 `nll_model / nll_null / spikes`
- 返回 `[T]` 维度的 per-bin 值
- 这用于观察 fp-bps 随预测步数增加的衰减趋势

**训练脚本中的 per-bin 日志有精度差异**：
- `train.py` 在 `_shared_eval_step()` 里直接对当前 batch 调 `fp_bps_per_bin()` 然后 `self.log()`
- 这让 Lightning 对 batch 做 epoch 聚合（类似 batch-mean），不是严格的全局充分统计量累计
- `eval_phase1_v2.py` 则正确地对 `fp_bps_per_bin_stats()` 跨 batch 累加后再 finalize
- 因此离线 eval 的 per-bin 值比训练 dashboard 上的更精确

### 1.4 Batch 维度的处理

**不是 batch mean。**

训练脚本中：
- `on_validation_epoch_start()` 初始化 7 个 `float64` 累加器
- 每个 `validation_step()` 调用 `fp_bps_stats()` / `r2_stats()` 获取充分统计量，**累加**到 epoch state
- `on_validation_epoch_end()` 执行 `all_reduce(SUM)` 后调 `finalize_fp_bps_from_stats()`

这意味着：
- 一个包含 128 个 spike 的 batch 和一个包含 512 个 spike 的 batch 不是等权的
- spike 多的 batch 对最终数字贡献更大
- **这是正确的做法**：充分统计量的跨 batch 累加等价于在全数据集上一次性计算

### 1.5 Session 维度的处理

**没有 per-session 平均，也没有 session balancing。**

当前实现：
- sampler 不做 session balancing——哪个 session 可采样窗口多，它贡献的样本就多
- fp-bps 在所有 session 的样本上直接全局累加
- 唯一的 session 隔离在于 null rate：不同 session 的 neuron 有不同的 global unit id -> 不同的 null rate

**潜在影响**：
- 如果某个 session 有 200 个 neuron 且 firing rate 高，另一个 session 只有 50 个 neuron 且 firing rate 低
- 前者对最终 fp-bps 的影响远大于后者
- 如果需要 session-fair 比较，应另外实现 per-session fp-bps 再取平均

### 1.6 Train 阶段

**训练阶段不计算 fp-bps。**

`training_step()` 只计算：
- Poisson NLL loss（masked）-> 反向传播
- `train/mean_pred_rate` 和 `train/mean_target_count`（仅监控）

理由充分：train fp-bps 的意义有限——它会受到 data shuffling、dropout 等影响，且 fp-bps 的 null model 来自 train split 自身，在 train 上计算相当于"用同分布数据评估 vs 同分布均值"，容易过拟合性膨胀。

### 1.7 Val / Test 阶段

完整流程：

1. **null model 构建**（一次性）：
   - 遍历训练集所有 recording，统计 per-neuron spike count / bin count
   - 取 log(mean_count_per_bin) -> `null_rates` dict
   - 构建 lookup tensor，形状 `[max_global_unit_id + 1]`

2. **逐 batch 评估**：
   - forward 得到 `log_rate [B, T, N]`
   - 用 `target_unit_index [B, N]` 索引 null lookup -> `null_log_rates [B, N]`
   - 调 `fp_bps_stats()` 获得 3 个标量
   - 累加到 epoch state

3. **epoch 结束 finalize**：
   - all_reduce 后除以 total_spikes * ln2
   - 输出单一标量 `val/fp_bps` 或 `test/fp_bps`

continuous 和 trial-aligned 模式的**公式完全相同**，只是 sampler 不同：
- continuous: `SequentialFixedWindowSampler` 按连续时间滑窗
- trial-aligned: `TrialAlignedSampler` 按 go cue 对齐

### 1.8 小结

| 维度 | 处理方式 | 备注 |
|------|----------|------|
| Neuron | 全局 spike-weighted 聚合，不做 per-neuron mean | null model 是 per-neuron 的 |
| Time | 全局累加，不区分 bin 位置 | per-bin 版本另有实现 |
| Batch | 充分统计量跨 batch 累加 | 不是 batch mean |
| Session | 无 session balancing | 通过 global unit id 隔离 null rate |
| Train | 不计算 fp-bps | 仅计算 Poisson NLL loss |
| Val/Test | 全局累计 finalize | null 来自 train split |

### 1.9 合理性分析与改进建议

基于 1.1–1.8 的维度分析，本节逐维度评估当前 fp-bps 实现的合理性，并讨论是否需要与 IBL-MtM 的 metric 口径对齐。

#### 1.9.1 Neuron 维度：spike-weighted 聚合 vs per-neuron mean

**当前做法**：全局 spike-weighted 聚合——将所有 (b,t,n) 位置的 NLL 和 spike 数直接求和，高频 neuron 对最终 fp-bps 贡献更大。

**合理性评估**：

spike-weighted 聚合的优势：
- 严格遵循 bits-per-spike 的语义定义：平均每个 spike 携带的可预测信息增益
- 信息论解释清晰：如果模型对 spike 的预测整体更准，该指标直接反映
- 计算简洁，不需要分 neuron 统计

spike-weighted 聚合的潜在问题：
- 高频 neuron 主导结果：如果少数高频 neuron 的 fp-bps 很高，即使大量低频 neuron 预测很差，全局数字仍可能偏高
- 不同 session 间不可比：neuron 组成不同 → spike 分布不同 → 加权基准不同
- 与 NLB / IBL-MtM 社区标准不一致，妨碍跨论文比较

per-neuron mean 的优势：
- 每个 neuron 等权，反映模型对多样化 neuron 的平均预测能力
- 低频 neuron 的预测质量不被淹没
- 与 NLB co-bps 和 IBL-MtM forward-pred bps 口径一致

per-neuron mean 的劣势：
- 低频 neuron 的 bps 估计方差大（spike 少 → 分母小 → 数值不稳定）
- 需要对零 spike neuron 做特殊处理

**建议**：
- **保留 spike-weighted 为主指标**——它是当前训练 loss（Poisson NLL）的自然对应
- **增加 per-neuron mean 为辅助指标**——用于跨论文可比性和检测是否存在 neuron 偏倚
- 具体实现：在 `fp_bps_stats()` 同时返回 per-neuron 的 (nll_model, nll_null, spikes)，在 finalize 时分别计算 per-neuron bps 再取 mean

#### 1.9.2 Time 维度：flat 聚合的合理性

**当前做法**：所有 time bin 的 NLL 直接累加，不区分早期 bin（靠近 history，容易预测）和晚期 bin（远离 history，较难预测）。

**合理性评估**：

- flat 聚合作为 summary metric 是合理的：它给出"在整个预测窗口上，模型平均每个 spike 的预测增益"
- 已有 `fp_bps_per_bin_stats()` 提供 per-bin 版本，可以观察衰减趋势
- 不同预测窗口（250ms / 500ms / 1000ms）下的 flat fp-bps 已经天然反映了"远期预测更难"

**无需修改**：当前 flat + per-bin 的双轨设计已经足够。

#### 1.9.3 Session 维度：不做 balancing 的合理性

**当前做法**：所有 session 的样本在 fp-bps 计算中直接混合，长 session / 多 neuron 的 session 贡献更多。

**合理性评估**：

- 对于**联合训练评估**（"这批 session 整体表现如何"），不做 balancing 是合理的——这反映了模型在实际数据分布上的表现
- 对于**跨 session 泛化分析**（"模型在每个 session 上是否均匀好"），需要 per-session 报告

**建议**：
- 主指标（联合评估）保持全局聚合
- 在 evaluation 报告中增加 per-session fp-bps 表格，用于诊断模型是否在某些 session 上显著偏弱

#### 1.9.4 Null model 来源：train split vs eval split

**当前做法**：null rate 来自 train split 的 per-neuron 均值（`compute_null_rates()` 遍历训练 DataLoader）。

**IBL-MtM 做法**：null rate 来自 eval 数据本身（`np.nanmean(spikes, axis=...)`）。

**合理性分析**：

train-split null 的优势：
- 更原则正确：null model 不接触任何 eval 数据，无信息泄露
- 严格遵循"null model 可以在预测之前确定"的假设

eval-split null 的特点：
- 存在轻微信息泄露：null rate 基于 eval 数据的 marginal 统计
- 在 trial-aligned 场景下，eval null 可能更贴合当前 trial 的 firing rate 分布
- 实际影响通常很小：对于 evaluation set 足够大的场景，train 和 eval 的 per-neuron 均值趋于一致

**建议**：
- **保持 train-split null 为默认**——原则上更严谨
- **增加 eval-split null 的可选模式**——在需要与 IBL-MtM 直接对比时使用
- 实现方式：在 `compute_null_rates()` 中增加 `split='train'/'eval'` 参数

#### 1.9.5 与 IBL-MtM 口径对齐

为了支持跨论文 benchmark 比较，建议实现一个 `ibl_style_bps()` 函数：

| 维度 | IBL-MtM 口径 | 当前 fp-bps |
|------|-------------|-------------|
| Neuron 聚合 | per-neuron 独立计算 → mean | 全局 spike-weighted |
| Null model | eval target 本身的均值 | train split 均值 |
| 评估范围 | 仅 held-out future bins | 全部 prediction bins |
| 输出 | 标量（per-neuron mean） | 标量（global） |

实现要点：
1. 对每个 neuron `n`，取其 eval 数据上的均值作为 null rate
2. 对每个 neuron `n`，独立计算 `bps_n = (nll_null_n - nll_model_n) / (spikes_n * ln2)`
3. 返回 `mean(bps_n)` 和 `std(bps_n)`

**重要说明**：`ibl_style_bps()` 仅用于跨论文比较，不替代当前主指标。两个口径应同时报告，以便读者理解差异。

#### 1.9.6 总结

| 维度 | 当前实现 | 合理性 | 改进建议 |
|------|----------|--------|----------|
| Neuron | spike-weighted | 合理（符合 bps 定义），但妨碍跨论文比较 | 增加 per-neuron mean 辅助指标 |
| Time | flat 聚合 | 合理，已有 per-bin 补充 | 无需修改 |
| Session | 无 balancing | 联合评估合理 | 增加 per-session 报告 |
| Null model | train split | 更严谨，无信息泄露 | 增加 eval null 可选模式 |
| 跨论文比较 | 无对齐 | 阻碍比较 | 实现 `ibl_style_bps()` |

---

## Q2. Neuroformer 原始 repo 的 train/inference 范式

### 2.1 总结

| 特性 | 详情 |
|------|------|
| **训练方式** | Teacher forcing（next-token CE） |
| **推理方式** | Autoregressive token-level rollout |
| **当前预测窗口** | 50ms（`window.curr = 0.05`） |
| **历史窗口** | 50–150ms（`window.prev = 0.05 – 0.15`） |
| **时间分辨率** | 10ms（`resolution.dt = 0.01`） |
| **Exposure bias** | 确定存在，且有双层传播路径 |

### 2.2 训练：Teacher Forcing

Neuroformer 不是一个 binned-count regression 模型，而是一个 **spike event token sequence model**。

**数据表示**：
- 每个时间窗口内的 spike 被 token 化为 `(neuron_ID, dt)` 对
- `ID` token: 哪个 neuron 发了 spike
- `dt` token: spike 在当前窗口中的时间位置（离散化到 10ms 分辨率）
- `block_size.id = 100`: 单窗口最多保留 100 个 event token（不是 100 个 time bin）

**训练循环**（`trainer.py` L236）：

```python
preds, features, loss = model(x, y)
```

- `x` 包含 `id, dt, id_prev, dt_prev, frames` 等输入
- `y` 包含 `id, dt` 的右移目标序列
- 模型一次性接收完整的 gold prefix，输出每个位置的 next-token logits

**Loss 计算**（`model_neuroformer.py` L1248-1259）：

```python
loss_id = F.cross_entropy(id_logits.view(-1, ...), targets['id'].view(-1), ...)
loss_time = F.cross_entropy(dt_logits.view(-1, ...), targets['dt'].view(-1), ...)
loss['id'] = (3/5) * loss_id * (1 - 1/n)
loss['time'] = (2/5) * loss_time * (1 - 1/n)
```

- ID 和 dt 各自独立做 CE loss
- ID loss 权重 3/5，dt loss 权重 2/5
- 分母中的 `n` 取决于是否启用 contrastive loss（不启用时 `n=inf`，乘数退化为 1）

**这是标准的 teacher forcing**：训练时每一步都以真实 token 作为输入，模型永远不会看到自己的预测错误。

### 2.3 推理：Autoregressive Rollout

推理实现在 `simulation.py::generate_spikes()`，核心是一个嵌套循环：

**窗口内 rollout**（主循环）：

```python
for i in range(T_id - 1):  # 逐 token 生成
    logits, features, _ = model(x, y)
    logits['id'] = logits['id'][:, i]  # 取第 i 个位置的 logits
    # top-k/top-p filtering
    logits['id'] = top_k_top_p_filtering(logits['id'], ...)
    # sampling or argmax
    if sample:
        ix = torch.multinomial(F.softmax(logits['id']/temp), num_samples=1)
    # write back to sequence
    x['id'][:, i + 1] = ix.flatten()
    x['dt'][:, i + 1] = ix_dt.flatten()
    if ix.flatten() >= stoi['EOS']:
        break
```

每一步：
1. 把整个已生成序列送入模型
2. 取当前位置 logits
3. 采样/选择
4. 写入下一个位置
5. 直到生成 EOS 或达到 `T_id`

**窗口间 predicted-past rollout**（外层）：

当 `true_past=False`（默认值）时：

```python
if true_past is False:
    if it > (window_prev / window):
        # rebuild past context from model predictions in data buffer
        x['id_prev'], x['dt_prev'], pad_prev = dataset.get_interval(
            prev_id_interval, float(x['trial']), T_id_prev, data=df)
```

- 模型在窗口 w 生成的 token 被缓存到 data dict
- 在窗口 w+1 中，这些缓存 token 被用作 `id_prev / dt_prev`
- 因此，不仅窗口内是 rollout，**跨窗口的 past context 也会逐渐被模型预测替换**

### 2.4 窗口大小

从配置文件中确认：

**`configs/Visnav/lateral/mconf_predict_all.yaml`**:
- `window.curr = 0.05` -> **当前生成窗口 50ms**
- `window.prev = 0.05` -> **过去 history 50ms**
- `window.frame = 0.15` -> 视觉帧上下文 150ms
- `resolution.dt = 0.01` -> **时间分辨率 10ms**

**`configs/V1AL/mconf.yaml`**（另一数据集配置）:
- `window.curr = 0.05` -> 50ms
- `window.prev = 0.15` -> **150ms history**
- `window.frame = 0.2` -> 200ms 帧上下文

**与我们项目的对比**：
- NeuroHorizon base v2: 500ms observation + 250/500/1000ms prediction
- Neuroformer: 50–150ms observation + 50ms prediction
- **尺度差了约一个数量级**

一个容易混淆的点：`block_size.id = 100` 不是"100 个 time bin"。在 50ms 窗口 + 10ms 分辨率下，理论上只有 5 个时间 bin。`block_size=100` 是 token 容量上限——一个 50ms 窗口里如果有很多 neuron 同时发放，可能产生远超 5 个 event token。

### 2.5 Exposure Bias 分析

**结论：Neuroformer 确定存在 exposure bias，且有两层传播路径。**

**第一层：窗口内 token-level exposure bias**
- 训练时：位置 i 的输入永远是 ground truth token x_1, ..., x_{i-1}
- 推理时：位置 i 的输入是模型自己生成的 x_hat_1, ..., x_hat_{i-1}
- 一旦前面某步生成错误，后续所有步的条件分布都会偏移
- 这与 NLP 中 GPT 的 exposure bias 完全同构

**第二层：窗口间 predicted-past exposure bias**
- 当 `true_past=False` 时，窗口 w+1 的 past context 来自窗口 w 的预测
- 如果窗口 w 的预测有误，窗口 w+1 的初始条件就已经被污染
- 误差跨窗口级联传播

**代码中没有任何缓解机制**：
- 没有 scheduled sampling（逐渐用预测替换 gold）
- 没有 professor forcing（用 discriminator 对齐 teacher/free-running 分布）
- 没有 sequence-level RL / minimum risk training
- 没有 DAgger 式数据增强

**为什么论文中这个问题不突出**：

1. **窗口极短**：50ms 内的 event token 数量有限（几十到一百个），rollout 链条短，误差来不及严重积累
2. **研究重点不同**：Neuroformer 论文关注 spike generation + multimodal decoding + contrastive representation，不是 long-horizon forecasting
3. **评估方式不同**：论文主要看 precision/recall/F1 和行为解码，不是逐 bin fp-bps 衰减曲线

而我们的项目在 250–1000ms 尺度上做 binned count forward prediction，如果采用类似的 teacher forcing 训练 + rollout 推理，exposure bias 效应会被显著放大——这已经在 plan.md 的 1.9 模块优化实验中被验证（rollout 在 500ms/1000ms 上严重退化）。

---

## Q3. IBL-MtM paper 报 约0.4+ bps vs 我们 base v2 的 约0.21：差异根因

### 3.1 首先纠正数字误读

IBL-MtM NeurIPS 2024 论文 Table 9（single-session forward prediction）中，数字远非统一的"约0.4"：

| 模型 | temporal mask | all mask + prompt |
|------|-------------|-------------------|
| NDT1 | 0.3092 | **0.7375** |
| NDT2 | 0.2488 | 0.5092 |

关键观察：
- **NDT2 temporal = 0.2488** 与我们 base v2 的 0.2115 已在同一量级
- 高达 0.7375 的数字来自 **all mask + prompt**，这是一个完全不同的任务设定
- 论文中的"约0.4"是一个跨设定的模糊印象，不能作为单一参考

### 3.2 差异来源一：任务定义根本不同

**IBL-MtM forward prediction 任务**：

从 `finetune_eval_multi_session.py` L284 可以确认：
```python
'held_out_list': list(range(90, 100)),  # NLB uses 200 ms for fp
```

- 输入：100 个 time bin 的 trial-aligned 序列（按行为事件对齐）
- 前 90 bins 完全可见
- 预测最后 10 bins
- 如果每个 bin 是 20ms，则：**1.8s 可见 history -> 预测最后 200ms**

**我们的 base v2 任务**：
- `sequence_length = 0.75s`
- `pred_window = 0.25s`，`hist_window = 0.5s`
- 即：**0.5s history -> 预测 0.25s**

差异总结：

| 维度 | IBL-MtM forward pred | NeuroHorizon base v2 |
|------|---------------------|---------------------|
| 可见 history | 约1.8s（90 bins） | 0.5s |
| 预测窗口 | 约200ms（10 bins） | 250ms |
| 对齐方式 | trial-aligned（行为事件锚定） | continuous（连续滑窗） |
| history/pred ratio | 9:1 | 2:1 |

**history 长度对 forward prediction 的影响是决定性的**：更长的 history 提供更丰富的 population dynamics 上下文，模型可以更好地估计当前 neural state -> 更准确地预测近未来。9:1 的 ratio 本身就让任务显著更容易。

### 3.3 差异来源二：Metric 定义不同

**IBL-MtM 的 `bits_per_spike`**（`eval_utils.py` L1165-1193）：

```python
def bits_per_spike(rates, spikes):
    nll_model = neg_log_likelihood(rates, spikes)
    null_rates = np.tile(
        np.nanmean(spikes, axis=tuple(range(spikes.ndim - 1)), keepdims=True),
        spikes.shape[:-1] + (1,),
    )
    nll_null = neg_log_likelihood(null_rates, spikes, zero_warning=False)
    return (nll_null - nll_model) / np.nansum(spikes) / np.log(2)
```

关键：**null rate 来自当前评估数据本身**（`np.nanmean(spikes, axis=...)`），不是来自训练集。

而且，在 `forward_pred` 分支中（`eval_utils.py` L339-343）：
```python
for n_i in tqdm(range(len(target_neuron_idxs)), desc='co-bps'):
    bps = bits_per_spike(pred_held_out[:,:,[n_i]], gt_held_out[:,:,[n_i]])
    bps_result_list[target_neuron_idxs[n_i]] = bps
```

- 对**每个 neuron 单独**调用 `bits_per_spike()`
- 每个 neuron 的 null rate 来自**该 neuron 在 held-out 时间步上的均值**
- 最终 paper 报告的是 **per-neuron mean bps**

**我们的 fp-bps**：
- null rate 来自 **train split** 的 per-neuron 均值
- 聚合方式是 **global spike-weighted**（不是 per-neuron mean）

**具体差异表**：

| 维度 | IBL-MtM bps | NeuroHorizon fp-bps |
|------|------------|-------------------|
| Null model 来源 | eval target 数据本身 | train split |
| Null 粒度 | per-neuron（对 trial+time 取均值） | per-neuron（对 train bins 取均值） |
| 聚合方式 | per-neuron 独立计算再取 mean | 全局 spike-weighted 一次性聚合 |
| 评估片段 | 仅 held-out future bins | 全部 prediction bins |

**这些差异对结果的影响方向**：

1. **null 来自 eval 本身 vs train split**：eval 内 null 更贴合当前数据分布（特别是 trial-aligned 时，特定试次组合的均值更准），通常会让 null NLL 更低 -> null 更强 -> 模型要跑赢更强的 null -> bps 反而可能**更低**。但 IBL-MtM 的 null 是基于 held-out 时间步 (最后10 bins) 而非完整 trial 均值，样本量很小时均值估计可能更不稳定。
2. **per-neuron mean vs spike-weighted**：per-neuron mean 给每个 neuron 等权；spike-weighted 让高频 neuron 主导。如果模型对高频 neuron 预测更好（常见情况），spike-weighted 可能**偏高**。反之亦然。

这两个口径差异意味着即使底层预测完全相同，算出来的数字也会不同。

### 3.4 差异来源三：训练目标完全不同

**IBL-MtM 的训练方式**：

从 `masker.py` 和配置文件可以看到，IBL-MtM 是一个 **multi-task masked modeling** 框架：

训练时的 masking 模式（`ndt1.yaml`）：
- `mode: temporal`，`ratio: 0.3`
- 随机选 30% 的 time bins mask 掉
- 模型看到 70% 的时间步，重建 30%
- 这是一个 **BERT 式 denoising objective**

Prompting 版本（`ndt1_prompting.yaml`）：
- `mode: all`
- `use_prompt: true`
- 模型在不同 masking 任务间切换，prompt embedding 指示当前任务类型

评估时切到 `forward_pred`：
- `masker.mode = 'forward_pred'`
- mask 最后 10 个 time bins
- 模型看前 90 bins -> 预测后 10 bins
- **注意**：`context.forward = -1, context.backward = -1`，即 attention 没有因果约束，前 90 bins 之间可以双向 attend

当 `masking_mode == 'causal'` 时（`ndt1.py` L483-486）：
```python
if masking_mode == 'causal':
    self.context_forward = 0
    self.context_mask = create_context_mask(...)
```
- 此时变成 causal attention
- 但在 forward_pred 模式下，前 90 bins 仍然是**双向 attention**，只是后 10 bins 被 zeroed out

**我们 base v2 的训练方式**：
- 直接的 supervised future prediction
- loss = Poisson NLL on predicted future bins
- 没有 masking / denoising
- 没有 task prompt

**影响分析**：
- multi-task masked modeling 本质上是一种更强的自监督预训练，让模型学会 neural population dynamics 的丰富表征
- 切到 forward_pred 评估时，模型利用的是更强的 latent representation
- 单一 forward prediction 训练则只优化了"给定 history 预测 future"这一条路径
- all mask + prompt 的 **0.7375** 之所以远高于 temporal-only 的 **0.3092**，正是因为多任务联合训练提供了更丰富的表征

### 3.5 差异来源四：数据集与 population 规模

**IBL-MtM 数据**：
- IBL（International Brain Laboratory）标准化数据集
- trial-aligned sessions（对齐到行为事件）
- 每 session 平均约 **668 neurons**（从 `ndt1.yaml` 的 `n_channels: 668` 可以确认）
- 有跨多脑区的记录（VISp, CA1, DG, PO 等）

**NeuroHorizon base v2 数据**：
- Perich-Miller 10 sessions（NHP 运动皮层）
- 每 session 几十到一两百个 neuron
- 连续记录，无显式 trial 结构的强对齐

**影响**：
1. **population 冗余**：668 个 neuron 提供的群体信号冗余远多于几十个 neuron，让 masked 位置更容易从周围 neuron 推断
2. **trial 结构**：IBL 的行为任务（视觉决策）提供了强的 trial-to-trial 结构，late-trial bins 有较强的可预测性
3. **脑区多样性**：跨脑区记录可以利用区间信息流

### 3.6 差异来源五：Attention 机制的关键细节

一个容易忽略但非常重要的点：

IBL-MtM 在 `forward_pred` 评估时的 attention 模式是**双向的**：
- `context.forward = -1, context.backward = -1` 意味着没有因果约束
- 前 90 个 bins 之间可以互相 attend
- 模型利用了**整个可见 history 的双向信息**

而我们的 base v2 使用的是 Perceiver 架构：
- encoder 通过 cross-attention 把 history 压缩到 latent
- decoder 通过 cross-attention 从 latent 解码 future
- 虽然 encoder 内部也是双向的，但 latent bottleneck 会损失信息

IBL-MtM 的**无瓶颈双向 Transformer** 在 infilling/reconstruction 任务上有天然优势：模型可以直接看到 bin 89 的完整信息来预测 bin 90，信息路径最短。

### 3.7 差异来源六：当前仓库的 IBL-MtM-like wrapper 不是原始模型

`plan.md` 中 benchmark 表的 IBL-MtM 行（fp-bps = 0.1749）来自项目内的 simplified adapter，**不是原始 IBL-MtM**。

原始 IBL-MtM 包含：
- 完整的 NDT1 encoder + masker + prompt system
- 多任务训练（temporal / neuron / inter-region / intra-region / all）
- session stitching
- forward_pred 评估 pipeline

当前仓库 wrapper 只是：
- simple causal transformer
- future input zeroing
- single linear head

所以 benchmark 表中 0.1749 只反映 simplified baseline 在 Perich-Miller 上的水平，不能推导出"原始 IBL-MtM 也只有这个水平"。

### 3.8 公平比较需要统一的维度

如果要回答"在相同条件下 IBL-MtM 是否真的比 NeuroHorizon 强"，需要统一以下变量：

| 维度 | 需要统一为 |
|------|-----------|
| 数据集 | 同一数据集（Perich-Miller 或 IBL） |
| 窗口配置 | 相同 history / prediction 长度 |
| Null model | train-split null 或 eval null，二选一 |
| 聚合方式 | global spike-weighted 或 per-neuron mean，二选一 |
| 训练目标 | 如果比较架构 -> 统一为 forward pred only；如果比较方法论 -> 允许各自最优训练 |

在这些未统一之前，**0.4+ vs 0.21 的直接比较没有科学意义**。

### 3.9 更接近的参考数字

如果只看与 base v2 设定**最接近**的原始结果（temporal mask, 单 session, 无 prompt）：

| 模型 | bps | 备注 |
|------|-----|------|
| NDT2 temporal | 0.2488 | 最接近 base v2 设定 |
| NDT1 temporal | 0.3092 | 仍有 history 差 |
| NeuroHorizon base v2 | 0.2115 | 250ms continuous |

NDT2 temporal 的 0.2488 与我们的 0.2115 差距约 0.037 bps。考虑到 history 长度差异（IBL: 约1.8s vs 我们: 0.5s），这个差距完全可以被 history 优势解释。

### 3.10 综合结论

造成"MTM paper 约0.4+ vs base v2 约0.21"表面差距的因素，按重要性排序：

1. **任务定义不同**（最主要）
   - IBL: 1.8s 可见 -> 预测 200ms（ratio 9:1）
   - 我们: 0.5s 可见 -> 预测 250ms（ratio 2:1）

2. **Metric 口径不同**
   - IBL: per-neuron mean bps，null 来自 eval target
   - 我们: global spike-weighted fp-bps，null 来自 train split

3. **训练目标不同**
   - IBL: multi-task masked modeling（+ prompt conditioning）
   - 我们: direct Poisson forward prediction

4. **数据集不同**
   - IBL: 668 neurons, trial-aligned, 多脑区
   - 我们: 几十到一两百 neurons, continuous, 运动皮层

5. **模型架构差异**
   - IBL: 无瓶颈双向 Transformer，直接 attend 前 90 bins
   - 我们: Perceiver latent bottleneck，信息需经过压缩

6. **当前仓库 wrapper 不是原始 IBL-MtM**
   - benchmark 表的 0.1749 来自 simplified adapter

**最准确的结论**：

> 原始 IBL-MtM paper 报告的高 bps 来自更长 history（9:1 ratio）、不同 metric 口径（per-neuron mean + eval null）、多任务预训练和不同数据的综合作用。在更接近的 temporal-only 设定下（NDT2 temporal = 0.2488），与 base v2（0.2115）的差距已大幅缩小，且剩余差距可被 history 长度差异合理解释。在当前变量未统一的情况下，不应将两个数字作为架构优劣的直接证据。

---

## Q4. NeuroHorizon 实现细节补充

### 4.1 数据组织方式

#### 4.1.1 Dataset 与存储格式

数据存储采用 **lazy-load HDF5** 格式，每个 session 一个独立 `.h5` 文件：

| 组件 | 实现 |
|------|------|
| Dataset 类 | `torch_brain/data/dataset.py::NeuralDataset` |
| 存储格式 | HDF5（`.h5`） |
| 加载方式 | Lazy-load：只在 `__getitem__` 时读取对应时间窗口的数据 |
| Recording ID | 格式 `{dataset_name}/{session_name}`，全局唯一 |

每个 HDF5 文件包含：
- `spikes/timestamps`：spike 时间戳数组（秒）
- `spikes/unit_index`：对应 neuron 的局部索引
- `trials/`：trial 元信息（go_cue_time, choice, reward 等）
- 元数据：session_id, recording_duration, n_units

#### 4.1.2 三种 Sampler

| Sampler | 用途 | 采样方式 |
|---------|------|----------|
| `RandomFixedWindowSampler` | 训练 | 在每个 recording 的有效范围内随机采样固定长度窗口，支持 jitter |
| `SequentialFixedWindowSampler` | val/test continuous | 从起点开始以固定步长顺序滑窗，覆盖整个有效范围 |
| `TrialAlignedSampler` | trial-aligned eval | 以 go_cue 为锚点，窗口对齐到行为事件 |

`RandomFixedWindowSampler` 细节（`sampler.py`）：
- `window_length`：采样窗口总长度（= hist_window + pred_window）
- 每个 epoch 重新随机化所有采样起点
- 同一 batch 可能包含来自不同 session 的样本

`SequentialFixedWindowSampler` 细节：
- 用于评估，确保 deterministic 和完整覆盖
- `step_size` = window_length 时无重叠，< window_length 时有重叠

`TrialAlignedSampler` 细节（`trial_sampler.py`）：
- 从 HDF5 读取 trial 表，按 go_cue_time 对齐
- 每个 trial 产生一个样本，窗口以 go_cue 为中心（或指定 offset）
- 用于 PSTH-R² 评估

#### 4.1.3 Train / Val / Test 划分

**per-session temporal split**：每个 session 内部按时间分割。

```
session_duration = [0 ──────────────────── T]
                   [  train  |  val  | test ]
```

- `DataModule`（`train.py`）中通过 `recording.set_split()` 设置
- 比例通常为 70/15/15 或 80/10/10（具体由配置决定）
- **关键**：同一 session 的 train/val/test 不会在时间上重叠，防止数据泄露
- 每个 session 独立分割，不是全局划分

#### 4.1.4 Tokenize Transform

`tokenize()` 方法（`neurohorizon.py` L619-682）将原始 spike 数据转换为模型输入：

1. **History spikes 提取**：
   - 从 `[t_start, t_start + hist_window]` 时间段提取所有 spike event
   - 添加 `START` 和 `END` special tokens 作为边界标记
   - 输出 `input_timestamps` 和 `input_unit_index`

2. **Latent grid 构建**：
   - 在 `[t_start + hist_window, t_end]` 的预测窗口上等间距放置 latent query timestamps
   - 间距由 `latent_step` 参数控制

3. **Target binning**：
   - 预测窗口按 `bin_size` 切分为若干 bin
   - 统计每个 bin 内每个 neuron 的 spike count → `target_counts [T_pred, N]`
   - 记录 `target_unit_index [N]`（global unit id）

#### 4.1.5 Collation

`collate_fn`（`collate.py`）将 batch 内不同长度的样本统一：

- **Neuron 维度 padding**：`pad2d` 将所有样本 pad 到 batch 内最大 neuron 数，生成 `unit_mask [B, N]`
- **Spike event padding**：`pad8` 将 spike 序列 pad 到 8 的倍数（高效 attention 对齐）
- **Mask tracking**：所有 padding 位置通过 mask tensor 标记，在 loss 和 metric 计算时排除

### 4.2 Sessions / Neurons / Time 在训练中的分布

#### 4.2.1 Sessions 分布

- 所有 session 的训练数据被 **concatenate** 到同一个 sampler 中
- **无 session balancing**：长 session（更多可采样窗口）自然贡献更多训练样本
- 同一个 mini-batch 可以包含来自不同 session 的样本
- session 身份信息通过 `session_id` embedding 传递给模型（如果配置启用）

#### 4.2.2 Neurons 分布

- 不同 session 的 neuron 数量不同（几十到一两百个）
- batch 内通过 padding + `unit_mask` 统一到相同维度
- **等权对待**：loss 和 metric 对每个真实 neuron 位置等权（spike-weighted 发生在 metric 层面，不在 loss 层面）

**Unit dropout augmentation**（训练时数据增强）：
- 随机丢弃部分 neuron，模拟不同 population 子集
- 分布：三角分布，`min=30, mode=80, max=200`
- 即大部分训练样本保留约 80 个 neuron，范围在 30–200 之间
- 目的：提升模型对 neuron 子集变化的鲁棒性，防止过拟合到特定 neuron 组合

#### 4.2.3 Time 分布

- **训练**：`RandomFixedWindowSampler` 随机采样起始时间，覆盖 session 的整个训练时间段
- **评估**：`SequentialFixedWindowSampler` 顺序滑窗，确保确定性和完整覆盖
- 预测窗口长度（250ms / 500ms / 1000ms）是关键超参数
- 不同预测窗口下的 fp-bps 显著不同（`plan.md`：250ms=0.2115, 500ms=0.1744, 1000ms=0.1317）

#### 4.2.4 Batch 组成

一个训练 batch 的典型组成：

```
Batch [B=32]:
  sample_0:  session_A, t=12.3s, N=87  neurons (padded to N_max)
  sample_1:  session_C, t=45.1s, N=142 neurons (padded to N_max)
  sample_2:  session_A, t=8.7s,  N=87  neurons (padded to N_max)
  ...
  sample_31: session_B, t=23.5s, N=65  neurons (padded to N_max)
```

- `N_max = max(N_i for i in batch)`（或被 unit dropout 截断后的值）
- 每个样本携带自己的 `unit_mask`、`target_unit_index`
- 同一 session 的多个样本共享相同的 neuron 排列（但如果 unit dropout 启用，每个样本的 neuron 子集可能不同）

### 4.3 Evaluation / Metric 计算细节

#### 4.3.1 训练时的 Evaluation（在线 eval）

在 `train.py` 的 validation loop 中：

```
on_validation_epoch_start():
    初始化 7 个 float64 累加器:
    - fp_bps: nll_model_sum, nll_null_sum, total_spikes (3 个)
    - r2: ss_res_sum, ss_tot_sum, y_sum, n_sum (4 个)

validation_step(batch):
    model.eval()
    log_rate = model(batch)                        # forward pass
    stats = fp_bps_stats(log_rate, targets, ...)   # 充分统计量
    epoch_state[nll_model_sum] += stats[0]
    epoch_state[nll_null_sum] += stats[1]
    epoch_state[total_spikes] += stats[2]

on_validation_epoch_end():
    all_reduce(SUM) across GPUs                    # DDP 同步
    fp_bps = finalize_fp_bps_from_stats(...)       # 最终除法
    self.log('val/fp_bps', fp_bps)
```

关键特性：
- 充分统计量累加 → 数学上等价于全数据集一次性计算
- `all_reduce(SUM)` 保证多 GPU 一致性
- `float64` 累加器防止大数据集上的精度损失

#### 4.3.2 离线 Evaluation（eval_phase1_v2.py）

提供两种评估模式：

**Continuous 模式**（`evaluate_continuous()`）：
- Sampler: `SequentialFixedWindowSampler`
- 顺序滑窗覆盖整个 test split
- 计算全局 fp-bps + per-bin fp-bps 衰减曲线
- 支持 rollout 模式（当 `model.requires_target_counts` 时）

**Trial-aligned 模式**（`evaluate_trial_aligned()`）：
- Sampler: `TrialAlignedSampler`
- 以 go_cue 为锚点对齐每个 trial
- 计算 PSTH-R²：按 `(session_id, target_id)` 分组 → trial-average → Gaussian smooth → 全局 R²
- 用于评估模型对 trial-averaged neural dynamics 的捕捉能力

#### 4.3.3 PSTH-R² 计算

`psth_r2()`（`neurohorizon_metrics.py`）的完整流程：

1. 收集所有 trial 的 `(session_id, target_id, predicted_rate, actual_count)`
2. 按 `(session_id, target_id)` 分组
3. 对每组内的 trial 取平均 → trial-averaged PSTH
4. 对 trial-averaged PSTH 应用可选的 Gaussian smoothing（`sigma` 参数）
5. 计算全局 R²：`1 - SS_res / SS_tot`，其中 SS 在所有 (session, target, neuron, time) 上累加

#### 4.3.4 Edge Cases 处理

| 场景 | 处理方式 | 代码位置 |
|------|----------|----------|
| Padding neuron | `unit_mask` 排除，不计入 NLL 或 spike count | `fp_bps_stats()` |
| 零 spike neuron（eval 中某 neuron 无 spike） | 该 neuron 的 nll_null ≈ exp(null_rate) ≈ null_rate_count（很小），nll_model 类似；bps 贡献约 0 | 自然处理 |
| `log_rate` clamp | `poisson_nll_elementwise()` 中 `log_rate.clamp(-10, 10)`，防止数值溢出 | `neurohorizon_metrics.py` |
| Null rate 下界 | `log(max(mean_count, 1e-6))`，确保 log 有定义 | `compute_null_rates()` |
| 全零 bin（某 time bin 所有 neuron 均无 spike） | NLL 仍正常计算（exp(log_rate) - 0 * log_rate = exp(log_rate)），但该 bin 不贡献 spike count | 自然处理 |

#### 4.3.5 Null Model 构建细节

`compute_null_rates()`（`neurohorizon_metrics.py`）的完整逻辑：

```
for each batch in train_dataloader:
    tokenize(batch)
    for each neuron n (unmasked):
        global_id = target_unit_index[n]
        spike_counts[global_id] += target_counts[:, n].sum()
        bin_counts[global_id] += T_pred  # 该 neuron 的有效 bin 数

null_rate[global_id] = log(max(spike_counts[global_id] / bin_counts[global_id], 1e-6))
```

`build_null_rate_lookup()`：
- 创建 tensor `[max_global_unit_id + 1]`
- 直接通过 `target_unit_index` 索引：`null_log_rates = lookup[target_unit_index]`
- 全局 unit id 由 `UnitTokenizer` 分配，格式 `{session_prefix}_{local_index}`，确保跨 session 不冲突

### 4.4 模型架构：AR Decoder 和 PerNeuronMLPHead

#### 4.4.1 三阶段 Pipeline 概览

NeuroHorizon 的完整前向传播分为四个阶段：

```
Input Spikes [events]
    |
    v
+---------------------+
|   Encoder           |  Cross-attention: spike events -> latent grid
|   (history -> latent)|  输入: spike timestamps + unit embeddings
+----------+----------+  输出: latent representations [B, L, D]
           |
           v
+---------------------+
|   Processor         |  Self-attention on latent grid
|   (latent -> latent) |  多层 Transformer encoder
+----------+----------+  输出: refined latents [B, L, D]
           |
           v
+---------------------+
|   AR Decoder        |  Causal cross-attention: latent -> prediction bins
|   (latent -> bins)   |  逐 bin 或并行解码
+----------+----------+  输出: per-bin representations [B, T, D]
           |
           v
+---------------------+
|   PerNeuronMLPHead  |  Per-neuron readout: bin repr -> log_rate per neuron
|   (bins -> rates)    |  输出: log_rate [B, T, N]
+---------------------+
```

维度说明：`B` = batch, `L` = latent grid size, `D` = model dim, `T` = prediction bins, `N` = neurons

#### 4.4.2 AR Decoder 层结构

每个 AR Decoder layer（`autoregressive_decoder.py`）包含四个子层：

```
Input: query [B, T, D]
    |
    +-- 1. hist_cross_attn     Cross-attention to encoder output (history latents)
    |       query: prediction bins
    |       key/value: latent grid from Processor
    |
    +-- 2. pred_cross_attn     Cross-attention to prediction memory (if enabled)
    |       query: prediction bins
    |       key/value: prediction memory tokens
    |       (仅在 decoder_variant=prediction_memory 时激活)
    |
    +-- 3. causal_self_attn    Causal self-attention among prediction bins
    |       mask: causal (bin t 只能 attend to bins <= t)
    |       确保 autoregressive 性质
    |
    +-- 4. FFN                 Feed-forward network
            标准 Transformer FFN: Linear -> GELU -> Linear
```

每个子层配有 LayerNorm（pre-norm）和残差连接。

#### 4.4.3 三种 Decoder Variant

| Variant | 核心特征 | pred_cross_attn | 适用场景 |
|---------|----------|-----------------|----------|
| `query_aug` | 直接将 history info 注入 query | 无 | 默认（base v2） |
| `prediction_memory` | 额外维护 K 个 summary token | 有，attend to prediction memory | 需要跨 bin 全局信息 |
| `local_prediction_memory` | 局部窗口的 prediction memory | 有，attend to local window | 长序列高效版 |

**`query_aug`**（base v2 默认）：
- 最简单的变体
- prediction bins 的初始 query 通过 cross-attention 从 history latent 获取上下文
- 然后 causal self-attention 在 bins 间传递信息
- 不需要额外的 prediction memory

**`prediction_memory`**：
- 维护 `K` 个可学习的 summary token
- 这些 token 通过 `PredictionMemoryEncoder` 接收已解码 bin 的预测 counts
- 下一个 bin 可以通过 cross-attention 查看前面所有 bin 的预测概要
- 支持"预测依赖于之前预测"的场景

#### 4.4.4 Teacher Forcing vs Rollout 的等价性

**关键性质**：在 base v2 默认配置下（`decoder_variant=query_aug`, `feedback=none`），teacher forcing 与 autoregressive rollout **数学上等价**。

原因：
- `query_aug` 不使用 prediction memory → 没有"前面 bin 的预测结果"作为后面 bin 的输入
- `feedback=none` → 不将前面 bin 的 predicted counts 反馈到输入
- Causal self-attention 保证 bin `t` 只依赖 bins `<= t` 的**表征**（不依赖 bins `< t` 的**预测输出**）
- 因此，parallel forward（一次性算所有 bins）≡ sequential rollout（逐 bin 算）

**何时不等价**：
- 当 `feedback != none` 时：前面 bin 的 predicted spike counts 被反馈到后续 bin 的输入 → rollout 依赖于预测值，TF 使用 ground truth
- 当 `decoder_variant = prediction_memory` 时：prediction memory 接收前面 bin 的预测/真实 counts → 同理

`generate()` 方法（`neurohorizon.py` L538-617）实现真正的 autoregressive rollout：
- 逐 bin forward
- 每步取 predicted log_rate → 转为 counts（exp(log_rate)）
- 将 counts 反馈给 prediction memory / feedback encoder
- 用于需要 feedback 或 prediction memory 的变体

#### 4.4.5 PerNeuronMLPHead 结构

`PerNeuronMLPHead`（`autoregressive_decoder.py`）将 decoder 输出映射到每个 neuron 的 log firing rate：

```
Input: decoder_output [B, T, D], unit_embeddings [B, N, D]
    |
    +-- bin_proj: Linear(D -> D/2)     # 投影 time bin 表征
    |       [B, T, D] -> [B, T, D/2]
    |
    +-- unit_proj: Linear(D -> D/2)    # 投影 neuron 嵌入
    |       [B, N, D] -> [B, N, D/2]
    |
    +-- broadcast + concat:
    |       bin_repr:  [B, T, 1, D/2] -> [B, T, N, D/2]
    |       unit_repr: [B, 1, N, D/2] -> [B, T, N, D/2]
    |       concat:    [B, T, N, D]
    |
    +-- MLP: Linear(D -> D) -> GELU -> Linear(D -> 1)
    |       [B, T, N, D] -> [B, T, N, 1]
    |
    +-- squeeze: [B, T, N, 1] -> [B, T, N]
            = log_rate per (batch, time_bin, neuron)
```

设计动机：
- **分离 bin 和 neuron 的表征空间**：bin_proj 捕捉"这个时间步的 population state"，unit_proj 捕捉"这个 neuron 的特性"
- **拼接后 MLP 做交互**：让模型学习 "state x neuron → rate" 的非线性映射
- **D/2 + D/2 = D**：维持参数量与 decoder 维度匹配
- **逐 neuron 独立**：head 对每个 neuron 独立输出，不存在 neuron 间的交叉（neuron 间交互已在 encoder/decoder 的 population-level attention 中完成）

#### 4.4.6 Feedback Methods

当 `feedback != none` 时，前面 bin 的预测 counts 被反馈到后续 bin 的输入。四种方法（`prediction_feedback.py`）：

| Method | 机制 | 输出维度 |
|--------|------|----------|
| `none` | 不使用 feedback（base v2 默认） | — |
| `mlp_pool` | counts 经 MLP 编码后 mean-pool 到一个向量 | [B, T, D] |
| `rate_weighted` | 用 predicted rates 加权 unit embeddings | [B, T, D] |
| `cross_attn` | 用 bin repr 作 query，counts-weighted units 作 key/value | [B, T, D] |

`mlp_pool`：最简单，将 `[B, T, N]` counts 通过 per-neuron MLP 映射后 pooling

`rate_weighted`：`sum(rate_n * unit_embedding_n) / sum(rate_n)`，类似 attention 的 soft weighting

`cross_attn`：最复杂，允许模型动态选择关注哪些 neuron 的 feedback

#### 4.4.7 PredictionMemoryEncoder

当 `decoder_variant = prediction_memory` 时，`PredictionMemoryEncoder`（`prediction_memory.py`）维护 `K` 个 summary token：

- **输入**：已解码 bin 的 counts `[B, t, N]`（前 `t` 个 bin 的预测或真实 counts）
- **处理**：counts → embedding → cross-attention（summary tokens as query, count embeddings as key/value）
- **输出**：`K` 个更新后的 summary token `[B, K, D]`
- **作用**：在 AR decoder 的 `pred_cross_attn` 子层中，prediction bins attend to 这些 summary token

`K` 通常设为较小值（如 4 或 8），起到信息瓶颈和压缩的作用。

#### 4.4.8 Base v2 默认配置

| 参数 | 值 | 说明 |
|------|-----|------|
| `decoder_variant` | `query_aug` | 不使用 prediction memory |
| `feedback` | `none` | 不使用 prediction feedback |
| `n_decoder_layers` | 4 | AR decoder 层数 |
| `n_heads` | 8 | Multi-head attention 头数 |
| `model_dim` | 512 | 隐藏维度 D |
| `sequence_length` | 0.75s | 总窗口长度 |
| `hist_window` | 0.5s | 历史观察窗口 |
| `pred_window` | 0.25s | 预测窗口 |
| `bin_size` | 0.02s | 20ms bins |
| `latent_step` | — | Latent grid 间距 |

在此配置下：
- TF ≡ rollout（数学等价）
- 预测 bins 数 = 250ms / 20ms = 12–13 bins
- 模型复杂度主要来自 encoder + processor，decoder 相对轻量


---

## 附录：四个问题的交叉关联

1. **Q1 的 metric 实现理解是 Q3 比较的前提**：不理解我们自己的 fp-bps 是 global spike-weighted + train null，就无法准确判断与 IBL-MtM per-neuron mean + eval null 的差异

2. **Q1.9 的改进建议衔接 Q3 和 Q4**：section 1.9 提出的 `ibl_style_bps()` 直接解决 Q3 中识别出的 metric 口径不一致问题；per-neuron mean 辅助指标的实现需要依赖 Q4 中描述的 null model 构建和 evaluation pipeline

3. **Q2 的 exposure bias 分析与 Q3 有间接关联**：Neuroformer 的 teacher-forcing + rollout 分离，与 IBL-MtM 的 masked modeling 都不是我们 base v2 的 direct supervised prediction 范式。不同训练范式对同一 metric 的表现会有系统性偏移

4. **Q4 的架构细节解释了 Q3 中的架构差异因素**：section 4.4 中 AR Decoder 的 causal self-attention + Perceiver latent bottleneck 架构，与 Q3 section 3.6 中分析的 IBL-MtM 无瓶颈双向 Transformer 形成对比。base v2 的 TF≡rollout 等价性（4.4.4）也意味着不存在 Q2 讨论的 exposure bias 问题

5. **如果后续要做 faithful reproduction benchmark**：
   - 对 Neuroformer：需保留原始 spike tokenization + autoregressive generation，然后 re-bin 到统一 fp-bps
   - 对 IBL-MtM：需跑原始 NDT1 + masker + forward_pred 评估，然后统一 null model 和聚合方式
   - 两者都不能用当前仓库的 simplified adapter 代替
   - 对 NeuroHorizon：Q4 中描述的 evaluation pipeline（4.3）和 null model 构建（4.3.5）提供了实现基准
