# 2026-03-17 QA 调研记录（Claude）

## 任务范围

独立调研并详细分析以下三个问题：

1. 当前项目 `fp-bps` 指标在 neuron / time / batch / session 维度的处理方式，以及 train / val 阶段的计算流程
2. Neuroformer 原始 repo 的 train / inference 范式（teacher forcing vs rollout），观察窗口/预测窗口大小，是否存在 exposure bias
3. IBL-MtM 原始 paper 报告的 forward prediction bps (约0.4) 与我们 base v2 (约0.21) 之间差异的根因分析

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

## 附录：三个问题的交叉关联

1. **Q1 的 metric 实现理解是 Q3 比较的前提**：不理解我们自己的 fp-bps 是 global spike-weighted + train null，就无法准确判断与 IBL-MtM per-neuron mean + eval null 的差异

2. **Q2 的 exposure bias 分析与 Q3 有间接关联**：Neuroformer 的 teacher-forcing + rollout 分离，与 IBL-MtM 的 masked modeling 都不是我们 base v2 的 direct supervised prediction 范式。不同训练范式对同一 metric 的表现会有系统性偏移

3. **如果后续要做 faithful reproduction benchmark**：
   - 对 Neuroformer：需保留原始 spike tokenization + autoregressive generation，然后 re-bin 到统一 fp-bps
   - 对 IBL-MtM：需跑原始 NDT1 + masker + forward_pred 评估，然后统一 null model 和聚合方式
   - 两者都不能用当前仓库的 simplified adapter 代替
