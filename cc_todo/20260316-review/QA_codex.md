# 2026-03-17 QA 调研记录（Codex）

## 任务范围

本记录回答以下 3 个问题，并将结论落到当前远程仓库的代码与原始论文/原始仓库对照上：

1. 当前项目里 `fp-bps` / `bp-fps` 指标的实现，到底如何处理 neuron / time / batch / session 维度；train / val 阶段又是怎么计算的。
2. Neuroformer 原始 repo 里，训练与推理到底是 teacher forcing、rollout，还是别的形式；它的观察窗口和预测窗口大概是多少；它是否也存在 exposure bias（暴露偏差）问题。
3. IBL-MtM / MTM 原始 paper 和原始代码里，为什么能报出明显高于我们 `base v2` 的 forward prediction bits-per-spike；这背后到底是训练方式、模型架构、metric 定义，还是数据处理口径不同。

---

## 参考材料

### 当前项目内代码 / 文档

- `torch_brain/utils/neurohorizon_metrics.py`
- `examples/neurohorizon/train.py`
- `scripts/analysis/neurohorizon/eval_phase1_v2.py`
- `examples/neurohorizon/configs/model/neurohorizon_small.yaml`
- `examples/neurohorizon/configs/model/neurohorizon_small_1000ms.yaml`
- `neural-benchmark/adapters/neuroformer_adapter.py`
- `neural-benchmark/adapters/ibl_mtm_adapter.py`
- `neural-benchmark/benchmark_models/neuroformer/README.md`
- `neural-benchmark/benchmark_models/neuroformer/neuroformer_train.py`
- `neural-benchmark/benchmark_models/neuroformer/neuroformer_inference.py`
- `neural-benchmark/benchmark_models/neuroformer/neuroformer/data_utils.py`
- `neural-benchmark/benchmark_models/neuroformer/neuroformer/model_neuroformer.py`
- `neural-benchmark/benchmark_models/neuroformer/neuroformer/simulation.py`
- `neural-benchmark/benchmark_models/ibl-mtm/README.md`
- `neural-benchmark/benchmark_models/ibl-mtm/src/configs/ndt1.yaml`
- `neural-benchmark/benchmark_models/ibl-mtm/src/configs/ndt1_prompting.yaml`
- `neural-benchmark/benchmark_models/ibl-mtm/src/configs/trainer_ndt1.yaml`
- `neural-benchmark/benchmark_models/ibl-mtm/src/models/ndt1.py`
- `neural-benchmark/benchmark_models/ibl-mtm/src/models/masker.py`
- `neural-benchmark/benchmark_models/ibl-mtm/src/utils/eval_utils.py`
- `cc_core_files/plan.md`
- `cc_core_files/results.md`
- `cc_todo/1.8-benchmark_model/20260316_benchmark_legacy_audit_codex.md`

### 官方来源

- Neuroformer 官方仓库：<https://github.com/a-antoniades/Neuroformer>
- Neuroformer paper / arXiv：<https://arxiv.org/abs/2311.00136>
- Neuroformer project page：<https://a-antoniades.github.io/Neuroformer_web/>
- IBL-MtM 官方仓库：<https://github.com/colehurwitz/IBL_foundation_model>
- IBL-MtM / NeurIPS 2024 paper：<https://proceedings.neurips.cc/paper_files/paper/2024/hash/934eb45b99eff8f16b5cb8e4d3cb5641-Abstract-Conference.html>
- IBL-MtM project page：<https://ibl-mtm.github.io/>

---

## Q1. 当前 `fp-bps` 实现到底怎么处理 neuron / time / batch / session / train-val

## 1.1 公式层面

当前项目里 `fp-bps` 的核心实现是：

```text
fp-bps = (NLL_null - NLL_model) / (N_spikes * ln 2)
```

对应实现入口：

- `torch_brain/utils/neurohorizon_metrics.py::fp_bps_stats`
- `torch_brain/utils/neurohorizon_metrics.py::finalize_fp_bps_from_stats`

内部等价于：

```text
NLL_model_sum = Σ_{b,t,n in valid} PoissonNLL(log_rate[b,t,n], target[b,t,n])
NLL_null_sum  = Σ_{b,t,n in valid} PoissonNLL(null_log_rate[b,n], target[b,t,n])
total_spikes  = Σ_{b,t,n in valid} target[b,t,n]
fp-bps        = (NLL_null_sum - NLL_model_sum) / (total_spikes * ln 2)
```

关键点：**整体 fp-bps 不是先按 neuron / time / batch / session 各自算分再平均，而是直接对所有有效元素做一次全局累计。**

---

## 1.2 neuron 维度怎么处理

### 当前实现

- `log_rate` / `target` 的形状是 `[B, T, N]`。
- `mask` / `unit_mask` 的形状是 `[B, N]`，表示当前 batch 里哪些 unit 是真实神经元，哪些只是 padding。
- 在 `fp_bps_stats()` 里，`mask` 会先扩展成 `[B, T, N]`，然后对所有 `True` 位置一次性求和。

### 这意味着什么

1. **整体 fp-bps 不做 per-neuron 平均**。  
   高频神经元、spike 多的神经元，对最终分数权重更大；低 firing rate 神经元权重更小。

2. **padding neuron 不参与计算**。  
   这由 `unit_mask` 保证。

3. **null model 是 per-neuron 的，但最终指标不是 per-neuron 平均**。  
   每个 unit 的 null rate 不同，但最终分数仍然是全局 spike-weighted aggregation。

### null model 如何拿到 per-neuron rate

见 `compute_null_rates()`：

- 遍历训练集每个 `recording_id` 的训练区间；
- 统计每个 unit 在 train split 中的总 spike 数；
- 再除以该 unit 对应 train 区间的总 bin 数；
- 最后取 `log(mean_count_per_bin)`，写入 `null_rates`。

这里的键不是 local unit index，而是 **`model.unit_emb.tokenizer` 给出的 global unit id**。  
因此：

- 同一个 session 内不同神经元不会混；
- 不同 session 里“编号同名”的 local unit 也不会混；
- null model 实际上是 **per-session, per-neuron** 的训练均值，只是通过 global id lookup 来索引。

---

## 1.3 time 维度怎么处理

### 整体 fp-bps

在 `fp_bps_stats()` 中，`null_log_rates` 先从 `[B, N]` 扩展到 `[B, T, N]`，然后对 `T` 维所有时刻直接求和。

这意味着：

- **整体 fp-bps 不区分“早期 bin”和“晚期 bin”**；
- 只要在 mask 内，就和 neuron / batch 一样直接累计到总 NLL 里；
- 时间维的唯一权重来自该时间 bin 里 spike 的多少。

换句话说，整体 fp-bps 本质上是：

> “把整个验证集 / 测试集里的所有有效 `(time, neuron)` 位置摊平以后，模型比 null model 好多少。”

### per-bin fp-bps

项目也提供了 `fp_bps_per_bin_stats()` / `finalize_fp_bps_per_bin_from_stats()`：

- 先固定 `t`；
- 再对该 bin 下所有 batch 和 neuron 累计；
- 得到 `[T]` 长度的 per-bin fp-bps。

因此 per-bin 版本是：

- **按时间 bin 拆开**；
- 但在每个 bin 内仍然对所有 neuron 和 batch 做全局累计；
- 不是 per-neuron-per-bin 再平均。

---

## 1.4 batch 维度怎么处理

### 训练脚本中的 val / test 主指标

见 `examples/neurohorizon/train.py`：

- `on_validation_epoch_start()` / `on_test_epoch_start()` 会初始化全局累加器；
- 每个 `validation_step` / `test_step` 调 `fp_bps_stats()` 拿到 `nll_model_sum / nll_null_sum / total_spikes`；
- 然后累加到 epoch state；
- `on_validation_epoch_end()` / `on_test_epoch_end()` 再统一 `all_reduce` 后 finalize。

因此：

- **整体 `val/fp_bps` 和 `test/fp_bps` 不是 batch mean**；
- 而是全 epoch、全设备的全局累计版本。

这点是正确的，也是目前主指标最值得信任的实现方式。

### 一个细节 caveat：训练脚本里的 per-bin 日志不是全局累计

`examples/neurohorizon/train.py` 里虽然会在 `validation_step` 中计算：

- `val/fp_bps_bin{t}`
- `val/poisson_nll_bin{t}`

但这里做法是：

1. 先对当前 batch 调 `fp_bps_per_bin(...)`；
2. 再直接 `self.log(...)`。

这意味着这些 `val/fp_bps_bin{t}` 更接近于：

- “每个 batch 先算一个 per-bin 分数”
- “再让 Lightning 对 batch 结果做 epoch 聚合”

它**不是**像整体 `val/fp_bps` 那样，通过跨 batch 的充分统计量做严格全局累计。

这个差别后来在 `scripts/analysis/neurohorizon/eval_phase1_v2.py` 里已经被修正：  
`eval_phase1_v2.py` 会对 `fp_bps_per_bin_stats()` 的三个充分统计量跨 batch 累加，再统一 finalize，所以 **evalfix 结果里的 per-bin fp-bps 比训练期间 dashboard 上的 per-bin 更严格。**

---

## 1.5 session 维度怎么处理

### 当前 fp-bps 没有显式 “per-session average”

无论在 `train.py` 还是 `eval_phase1_v2.py`，整体 fp-bps 都没有做：

- 先每个 session 算一个 fp-bps；
- 再对 session 取 mean。

它做的是：

- sampler 产出多少个窗口，就累计多少个窗口；
- 哪个 session spike 多、窗口多、unit 多，它对最终分数影响就更大。

因此当前主指标是：

> **window-weighted + spike-weighted 的全局指标，不是 session-balanced 指标。**

### session 信息在什么地方进入指标

session 不直接进入公式，但会通过两条路径进入：

1. **null rate lookup**
   - `target_unit_index` 是 global unit id；
   - global unit id 已经隐含 session 前缀；
   - 因此不同 session 的神经元用的是不同 null rate。

2. **dataloader / sampler**
   - continuous 模式：`SequentialFixedWindowSampler`
   - trial 模式：`TrialAlignedSampler`
   - 哪个 session 可采样窗口更多，它就对最终 global metric 贡献更多。

如果以后要做“per-session 公平比较”，需要另加一个指标：

- 先按 session 拆；
- 各自累计 `nll_model / nll_null / spikes`；
- 最后对 session 取平均或报告分布。

当前主 fp-bps **不是这个口径**。

---

## 1.6 train / val / test 实际怎么计算

### train

当前 `training_step()` 只算：

- `PoissonNLLLoss`
- `train/mean_pred_rate`
- `train/mean_target_count`

**没有 train fp-bps。**

原因很直接：

- train 用的是随机采样窗口；
- 训练过程中 batch 分布不断变化；
- 如果直接记 train fp-bps，解释性很差，也容易把优化目标和监控目标混在一起。

### val / test

当前主流程里，`val/test` 的 fp-bps 是：

1. 先构建 train-split null model；
2. 在 valid 或 test loader 上逐 batch forward；
3. 用 `target_unit_index` 查 null rate；
4. 用 `unit_mask` 去掉 padding；
5. 全局累计 `nll_model / nll_null / total_spikes`；
6. epoch 结束时统一 finalize。

### continuous vs trial-aligned

两种模式**公式完全一样**，差别只在 sampler：

- continuous：按连续时间滑窗评估；
- trial-aligned：按 go cue 对齐的 trial 窗口评估。

所以 continuous / trial 的 fp-bps 差异，本质上来自：

- 样本分布不同；
- 窗口对齐方式不同；
- 不是公式不同。

---

## 1.7 对 Q1 的结论

当前项目的整体 fp-bps 实现，本质是一个：

- **per-neuron null model**
- **global spike-weighted aggregation**
- **跨 batch / 跨 time / 跨 session 一次性累计**
- **用 train split 估 null，用 valid/test split 做最终分数**

的指标。

它的几个最重要性质是：

1. **不是 per-neuron mean**
2. **不是 per-session mean**
3. **不是 batch mean**
4. **整体主指标是严格全局累计版**
5. **训练过程不算 train fp-bps**
6. **训练期 dashboard 里的 per-bin fp-bps 比离线 evalfix 的 per-bin 口径略松**

---

## 1.8 这个主指标是否合理；是否需要调整；是否要补一个 IBL-MtM 对齐指标

## 1.8.1 我的判断：当前实现适合作为 NeuroHorizon 的主指标，不建议直接替换

如果问题是：

> “对于当前 NeuroHorizon 的 continuous forward prediction 任务，这个主 fp-bps 的定义是否合理？”

我的答案是：

> **合理，而且应继续保留为主指标。**

原因不是“它唯一正确”，而是它和当前项目真正优化、真正评估的对象是一致的：

1. **和 loss 口径一致**
   - 训练目标本来就是对所有有效 `(batch, time, neuron)` 位置的 Poisson NLL 做总体优化；
   - 主 fp-bps 只是把同一个总体 NLL 改写成“相对 null baseline 提升多少 bits/spike”的形式。

2. **避免 eval leakage**
   - null baseline 来自 train split per-neuron mean；
   - valid/test 只用于最终打分；
   - 这是最干净的评估分工。

3. **和当前 continuous benchmark 的语义一致**
   - 现在项目里真正关心的是“整套系统在 held-out 时间窗上的总体 spike prediction 改善量”；
   - 不是“每个 neuron 拿到同样投票权后的平均表现”。

4. **实现层面也最严格**
   - 主 `val/test fp-bps` 用的是跨 batch、跨 device 的充分统计量累计；
   - 它比 dashboard 上 batch-level 平均出来的变体更可信。

所以，如果只允许保留一个榜单指标，我仍然会选当前这个主指标，而不是改成 per-neuron mean 或 eval-split null。

---

## 1.8.2 逐维度来看，哪些地方合理，哪些地方只适合作为辅助视角

### neuron 维度

当前实现：

- 不做 per-neuron mean；
- 而是让 spike 多的 neuron 自动拥有更高权重。

这个选择对“总体预测系统”来说是合理的，因为：

- Poisson likelihood 的自然单位本来就是 event / spike；
- 高频 neuron 的误差会更多地影响整体 NLL；
- 从工程角度看，它更接近“模型总体解释了多少 spike variability”。

但它也有代价：

- 低 firing-rate neuron 很容易被淹没；
- 某些只在一小部分 neuron 上失败的模式，不会在主指标里显得很明显。

因此我的建议不是替换主指标，而是：

- **主指标继续用 global spike-weighted**
- **辅助再报一个 per-neuron mean 指标**

### time 维度

当前实现：

- 所有 prediction bins 摊平累计；
- 整体 fp-bps 不区分早期 bin 与晚期 bin。

这对“整体窗口质量”是合理的，但不适合分析 long-horizon decay。  
因此：

- 主指标保留整体累计；
- 时间退化分析继续看严格累计版 per-bin fp-bps。

### batch 维度

当前实现：

- 主 `val/test fp-bps` 不是 batch mean；
- 而是 epoch-level sufficient statistics accumulation。

这是当前实现里最不应改动的部分，因为它恰好避免了：

- batch size 不均；
- batch 内 spike 数不均；
- DDP 多卡时不同 worker 样本量不均；

这些因素带来的偏差。

### session 维度

当前实现：

- 不做 per-session mean；
- session 的权重来自可采样窗口数、神经元数和 spike 数。

这对“整体验证集性能”是合理的，但它不是 session-balanced benchmark。  
所以这里的判断是：

- 作为主指标，合理；
- 作为 session 公平比较，不够。

如果后续要回答“跨 session 鲁棒性是否提升”，应该单独补：

- per-session fp-bps 分布
- 或 session-mean fp-bps

而不是改写当前主指标。

### null baseline 该不该在 val/test 上重算

这个问题最关键。

我认为：

- **主指标不应该改成 val/test null**
- **train-null 应继续保留为正式评估基线**

理由很直接：

1. 如果在 eval split 上重算 null，本质上就在用评估集边缘统计量定义 baseline；
2. 这对横向比较可能方便，但对项目内 model selection 不够干净；
3. 一旦 split 变化，baseline 语义也会变化，主指标的时序可比性会变差。

所以在项目内，“train-derived null + eval score” 是更稳的设计。

---

## 1.8.3 是否要对齐 Q3 里 IBL-MtM 的 metric 逻辑

我的判断是：

> **要补一个对齐 IBL-MtM 的对比指标，但不应让它替换 NeuroHorizon 现有主指标。**

也就是说，最合理的方案不是二选一，而是双口径并列：

1. **主指标**
   - global spike-weighted fp-bps
   - per-neuron train null
   - 用于项目内主结论、模型选择、版本追踪

2. **对比指标**
   - per-neuron mean fp-bps
   - per-neuron eval-split null
   - 用于和 IBL-MtM / NLB 风格结果做 compatibility comparison

这两个指标服务的目标不同：

- 主指标回答：“这个模型在当前 NeuroHorizon 任务里整体是否更好”
- 对比指标回答：“如果用更接近 IBL-MtM 的口径去看，数值会往哪边偏”

---

## 1.8.4 我建议补充的 IBL-MtM 对齐指标，应该怎么定义

如果要在文档里把这个建议写到可实现的程度，我建议定义成：

```text
对每个 neuron n：
  bps_n = (NLL_null,n - NLL_model,n) / (spikes_n * ln 2)

其中：
  NLL_model,n = Σ_{b,t in valid for n} PoissonNLL(log_rate[b,t,n], target[b,t,n])
  NLL_null,n  = Σ_{b,t in valid for n} PoissonNLL(log(null_eval_rate_n), target[b,t,n])
  null_eval_rate_n = 该 neuron 在当前 eval split 上的 mean count per bin

最后：
  comparison_bps = mean_n bps_n
```

实现时再加两个边界条件：

1. 只对 `spikes_n > 0` 的 neuron 计入平均；
2. 仍然使用当前的 `unit_mask` / global unit id 对齐，不改变 padding 与 session 索引逻辑。

这个定义和当前主指标的差异只在两点：

1. **aggregation 从 global spike-weighted 改成 per-neuron mean**
2. **null 从 train-split mean 改成 eval-split mean**

这正好对应 Q3 里和 IBL-MtM 口径最显著的两个差别。

---

## 1.8.5 这个对齐指标的价值和边界

我认为它值得加，但必须在文档中把边界写死：

### 它的价值

1. **帮助解释数值差距到底有多少来自 metric 口径**
   - 如果主指标是 `0.22`，对齐指标变成 `0.30+`，那说明口径差异本身就占了不小一部分。

2. **更容易与 IBL-MtM / NLB 类论文做定性比较**
   - 至少不会再把 global spike-weighted 和 per-neuron mean 直接混着比。

3. **更容易暴露难 neuron 的建模短板**
   - 因为 per-neuron mean 会把低 firing-rate、难预测 neuron 的权重抬上来。

### 它的边界

1. **它不适合替代主指标**
   - 因为 eval-split null 带有使用评估集边缘统计的性质。

2. **它也不能把任务差异抹平**
   - 即便 metric 对齐了，IBL-MtM 和 NeuroHorizon 在窗口、上下文、trial 组织、训练目标上仍然不同。

3. **它不是 session-balanced 指标**
   - per-neuron mean 会减弱 session 大小效应，但并不会自动变成 per-session 公平比较。

---

## 1.8.6 最后的建议：主指标保留，辅助并列一个 comparison metric

如果让我给项目层面的建议，我会这样写：

1. **保留当前主 fp-bps 不变**
   - 继续作为 NeuroHorizon 的正式主指标。

2. **新增一个 IBL-MtM 对齐 comparison metric**
   - `per-neuron mean + eval-split null`
   - 只用于外部 benchmark 对照与敏感性分析。

3. **如果以后要做更完整的诊断**
   - 再追加 per-session fp-bps；
   - 以及严格累计版 per-bin fp-bps。

这三者合起来，才比较完整：

- 主指标回答总体系统性能；
- per-neuron mean 回答 unit-level 公平性；
- per-session 指标回答跨 session 鲁棒性；
- per-bin 指标回答 long-horizon decay。

---

## Q2. Neuroformer 原始 repo：train / inference 是 teacher forcing 还是 rollout；窗口多大；是否有 exposure bias

## 2.1 先说结论

Neuroformer 原始实现是一个 **event-level / token-level 的自回归生成模型**，而不是 binned-count regression。

- **训练**：本质是 **teacher forcing** 的 next-token cross-entropy
- **推理**：本质是 **autoregressive rollout**
- **窗口**：官方开源配置大多是 **50ms 当前生成窗口**，配 **50ms 或 150ms 的过去 spike history**
- **暴露偏差**：**有，而且是结构性存在**

---

## 2.2 原始训练到底怎么做

Neuroformer 的 dataloader 不是把 `[T, N]` binned count 直接送进 Transformer；它先把 spike event token 化：

- `ID` token：哪个 neuron 发了 spike
- `dt` token：该 spike 在当前窗口里的时间位置

见：

- `neuroformer/data_utils.py::get_interval`
- `neuroformer/data_utils.py::__getitem__`

在 `__getitem__()` 里：

- `x['id_prev'] / x['dt_prev']` 是过去窗口的 token 序列
- `x['id'] / x['dt']` 是当前窗口的输入前缀
- `y['id'] / y['dt']` 是当前窗口的**右移一个 token 之后的目标**

这就是标准 teacher forcing：

- 输入给模型的是 gold prefix
- 目标是预测下一个 token

模型 loss 见 `model_neuroformer.py`：

- `loss_id = cross_entropy(id_logits, targets['id'])`
- `loss_time = cross_entropy(dt_logits, targets['dt'])`
- 最后按 `3/5` 和 `2/5` 做加权

也就是说，哪怕 README 把它描述为 spike causal masking / SCLM，本质落到代码上仍然是：

> **用真实 token 前缀训练 next-token prediction。**

这不是 rollout training，也没有看到 scheduled sampling / professor forcing / DAgger 一类机制。

---

## 2.3 原始推理到底怎么做

见 `neuroformer/simulation.py::generate_spikes`。

推理逻辑是：

1. 进入当前窗口；
2. 对 token 位置 `i = 0..T_id-2` 逐步循环；
3. 每一步都跑一次 `model(x, y)`；
4. 取当前位置的 `logits['id']` 和 `logits['dt']`；
5. 再通过：
   - `torch.multinomial` 采样，或者
   - `topk(argmax)` 选择
6. 把刚生成的 token 接回当前序列，再继续下一个 token。

这就是**标准 autoregressive rollout**。

更重要的是，Neuroformer 不只在当前 50ms 窗口内 rollout。  
它默认还会跨窗口滚动地把**模型预测出来的上一窗口 spike** 回填到 `id_prev / dt_prev` 里：

- `default_args.py` 里 `true_past=False`
- `simulation.py` 里当 `true_past is False` 时，会把缓存中的预测结果重建成上一窗口的 past state，再送入下一窗口

所以它在 inference 时有两层 rollout：

1. **窗口内 token rollout**
2. **窗口间 predicted past rollout**

---

## 2.4 它的观察窗口 / 预测窗口大概是多少

从开源配置看，Neuroformer 常用窗口明显短于我们当前 Phase 1 的 250ms / 500ms / 1000ms 预测任务。

### 官方 repo 常见配置

#### `configs/V1AL/mconf.yaml`

- `window.curr = 0.05`
- `window.prev = 0.15`
- `window.frame = 0.2`
- `resolution.dt = 0.01`

即：

- **当前生成窗口 50ms**
- **过去 history 150ms**
- **视觉帧上下文 200ms**

#### `configs/Visnav/lateral/mconf_pretrain.yaml`

- `window.curr = 0.05`
- `window.prev = 0.05`
- `window.frame = 0.15`
- `resolution.dt = 0.01`

即：

- **当前生成窗口 50ms**
- **过去 history 50ms**

### 一个容易混淆的点

`block_size.id = 100` / `block_size.prev_id = 100` 表示的是 **token 容量**，不是 100 个时间 bin。

因为一个 50ms 窗口里可以有多个 spike event token；  
`block_size` 控的是最多保留多少 event token，而不是窗口长度。

所以对 Neuroformer 更准确的理解是：

> 它是“短窗口内的逐 spike 生成”，而不是“长窗口的按 bin rate forecast”。

---

## 2.5 它是否存在 exposure bias / 暴露偏差

**有，而且非常标准。**

原因非常直接：

### 训练时

- 用的是 gold prefix
- loss 是 teacher-forced next-token CE

### 推理时

- prefix 变成模型自己刚生成的 token
- 默认 `true_past=False`，连跨窗口 history 也会逐渐替换成预测结果

所以只要模型在前面某一步生成错了：

- 当前窗口剩余 token 的条件分布就被污染；
- 下一窗口的 `id_prev / dt_prev` 也会被污染；
- 误差会跨 token、跨窗口传播。

### 有没有明显的缓解机制

在当前开源代码里，我没有看到：

- scheduled sampling
- professor forcing
- sequence-level RL / minimum risk training
- rollout-aware loss

因此它的 exposure bias 不是“可能有”，而是**训练-推理范式天然带来的确定性问题**。

### 为什么论文里这个问题没有在代码上显得特别突出

我倾向于认为主要有两个原因：

1. **窗口很短**  
   常见配置只生成 50ms 当前窗口，生成链条没有我们 250-1000ms 那么长。

2. **它的任务是 event generation，不是长时程 binned count forecasting**  
   论文主要关注 token generation / multimodal pretraining / decoding，而不是像我们这样系统研究 250ms、500ms、1000ms 的 long-horizon fp-bps 衰减。

所以它当然有 exposure bias，但它的实验主问题设定没有把这个问题放大到我们这里这么显著。

---

## 2.6 对 Q2 的结论

Neuroformer 原始 repo 的训练 / 推理范式可以概括为：

- **训练**：teacher forcing
- **推理**：真实 rollout
- **时间尺度**：典型是 50ms 预测窗口，50-150ms 的过去历史
- **风险**：确定存在 exposure bias，只是因为窗口较短、任务口径不同，所以没有像我们 250-1000ms forward prediction 那样被系统性放大

因此，如果要把 Neuroformer faithful 接到我们项目里：

1. 不能把它退化成 `binned counts -> GPT -> log-rate` 的 wrapper；
2. 必须保留它的 spike tokenization；
3. 必须保留它的 autoregressive generation；
4. 最后再把生成的 spike events re-bin 回统一 fp-bps 口径。

这也正是 `20260316_benchmark_legacy_audit_codex.md` 里已经指出的问题：  
当前仓库中的 `neural-benchmark/adapters/neuroformer_adapter.py` 只是一个 **simplified Neuroformer-like causal transformer**，并不是原始 Neuroformer。

---

## Q3. 为什么 MTM 原始 paper / 原始代码的 forward prediction 能到约 0.4，而我们的 base v2 只有约 0.2

## 3.1 先纠正一个常见误读：原文不是“固定 0.4”，而是不同设置差很多

NeurIPS 2024 论文 Table 9（single session forward prediction）里，不同模型 / masking 方案差异非常大。

按论文表格整理：

| 模型 | temporal mask | all mask + prompt |
|------|---------------|-------------------|
| NDT1 | 0.3092 | 0.7375 |
| NDT2 | 0.2488 | 0.5092 |

所以：

- “约 0.4” 只是一个模糊中间印象；
- 原始 paper 里 single-session best 甚至能到 **0.5 - 0.7**；
- 但如果看更接近 basic temporal masking 的版本，其实是 **0.25 - 0.31** 量级。

这点非常关键。  
因为我们的 `base v2` 在 `plan.md` 里的 250ms continuous `fp-bps=0.2115`（后来 evalfix test 已到 `0.2223`），和 **NDT2 temporal 的 0.2488** 已经是同一量级了；并不是“原文 0.4，我们这里只有 0.2，所以差了一倍”那么简单。

---

## 3.2 原始 MTM paper / code 的任务设定，与我们的 `base v2` 不是同一个 benchmark

这是最核心的结论。

### 原始 IBL-MtM forward prediction 是什么任务

从 `finetune_eval_multi_session.py` 和 `eval_utils.py` 看，原始代码的 `forward_pred` 评估是：

1. 用一个 **100 time bins** 的 trial-aligned 序列；
2. 在 `forward_pred` 模式下把 `held_out_list = range(90, 100)` 这些未来时间步全部 mask 掉；
3. 也就是只预测**最后 10 个 bin**；
4. 代码里还专门注释：`# NLB uses 200 ms for fp`。

也就是说，原始任务更接近：

> **给定前 90 个 time bins 的完整上下文，预测最后 10 个 bins（约 200ms）**。

如果 binsize 是 20ms，那么模型看到的是：

- **约 1.8s 历史上下文**
- 预测 **最后 200ms**

而我们的 `base v2` 250ms 配置是：

- `sequence_length = 0.75s`
- `pred_window = 0.25s`
- 因此只有 **0.5s history + 0.25s prediction**

这两者难度根本不在一条线上。  
原始 MTM forward prediction 拥有**更长的可见历史**。

---

## 3.3 metric 定义也不一样，不能把两个数字硬放在一张坐标轴上

### 原始 MTM / IBL-MtM 的 `bits_per_spike`

见 `ibl-mtm/src/utils/eval_utils.py::bits_per_spike`：

- 输入是某个 target slice 上的 `rates` 和 `spikes`
- null model 用的是：
  - 对 **当前 evaluation spikes**
  - 沿着 `trial` 和 `time` 维求均值
  - 得到每个 neuron 的 null rate

也就是说，它的 null baseline 是：

> **用“当前评估目标片段本身”的均值来做 per-neuron null。**

而不是像我们现在这样：

> **先在 train split 上估一个 per-neuron null，再固定到 valid/test 去评估。**

### 原始 forward_pred 的聚合方式

在 `eval_utils.py` 的 `forward_pred` 分支里，代码会：

1. 对每个 neuron 单独算一次 `bits_per_spike(pred_held_out[:,:,[n_i]], gt_held_out[:,:,[n_i]])`
2. 把这些 per-neuron bps 存进 `bps_result_list`
3. 后续结果表里再对 neuron 取平均

所以原始 paper 的 bps 更接近：

> **per-neuron mean bps on held-out future bins**

而我们的 `fp-bps` 是：

> **全局 spike-weighted fp-bps**

因此两边至少有 3 个口径差异：

1. **null baseline 不同**
2. **聚合方式不同**
3. **评估时间片不同**

这已经足够说明：

> 不能直接把 “MTM paper 里的 0.4/0.5” 和 “我们 plan.md 里的 0.2115” 当作同义指标来比较。

---

## 3.4 数据处理也不一样

### 原始 IBL-MtM

从论文和 repo 看，原始工作使用的是 **IBL aligned session** 数据：

- 单 session forward prediction 表格里，作者明确写了：
  - 平均约 **676 neurons**
  - 序列长度是 **100 time bins**
- 任务是 trial-aligned / event-aligned 的；
- 评估的是固定 late-trial bins 的预测。

### 我们当前 base v2

当前 `base v2` 用的是：

- `Perich-Miller` 10 sessions
- 每 session 大约几十到一两百个神经元
- continuous forward prediction 是按连续时间滑窗评估
- 主结果是 `250ms / 500ms / 1000ms` 三个窗口

这会带来两个很实际的差别：

1. **原始 IBL task 结构更强**  
   trial 对齐 + 重复行为结构 + 长历史上下文，本身就更利于 late-trial future bins 的预测。

2. **population 规模更大**  
   IBL 的 676-neuron 平均规模，与 Perich-Miller 的 session 规模明显不是一个量级。  
   更大的群体冗余有时会提高 masked reconstruction / forward prediction 的可预测性。

---

## 3.5 训练目标也完全不同：MTM 是 multi-task masking，我们的 base v2 是 direct future prediction

### 原始 IBL-MtM 在做什么

从 `ndt1.py`、`masker.py`、`ndt1_prompting.yaml` 可以看到，原始 IBL-MtM 的核心不是“只做 forward prediction”。

它的强项是：

- masked modeling
- 多种 mask 类型
  - temporal
  - neuron
  - inter-region
  - intra-region
- 以及 `all mask + prompt`

特别是 `use_prompt: true` 时，模型会 prepend 一个 prompt embedding，告诉模型当前 mask 任务是什么。

论文里最好的 single-session forward prediction 数字，恰恰来自：

- **all mask + prompt**

而不是 simplest temporal-only baseline。

这意味着原始高分不是纯靠“架构更强”拿到的，而是靠：

1. **更丰富的自监督任务族**
2. **prompt-conditioned routing**
3. **统一 masked modeling 训练后，再切到 forward_pred 评估**

### 我们当前 base v2 在做什么

`base v2` 的训练目标非常单纯：

- 直接 supervised future prediction
- loss 是 Poisson NLL
- 没有 multi-task masking
- 没有 task prompt
- 没有 neuron-mask / region-mask / temporal-mask 的联合预训练

所以如果拿原始 MTM 的 **all-mask+prompt** 数字来对比我们的 **single-objective future prediction**，那本来就不是 apples-to-apples。

---

## 3.6 模型架构也不同，但这不是唯一原因

### 原始 IBL-MtM / NDT1

repo 里的 NDT1 主体是：

- linear spike embedding
- 5-layer Transformer
- hidden size 512
- 多 masking 模式
- 评估时还支持 prompting / stitching / session conditioning 等分支

论文还明确指出：

> 在 IBL 的 676-neuron 平均规模下，NDT1 比 NDT2 更合适；NDT2 的 patching 假设在这个 setting 上不是最佳。

### 我们当前 base v2

`base v2` 是：

- POYO+/Perceiver 风格 encoder
- latent bottleneck（`num_latents_per_step=32`）
- `dim=128`
- decoder depth 2
- 约 `2.1M` 参数量

它的 inductive bias 更像：

> “先把历史观测压成 latent，再解出未来的 binned count”

而不是 MTM 的 masked reconstruction encoder。

### 这里真正重要的判断

不能把差异全部归因到“MTM 架构更强”。

因为当前 gap 同时混着：

- 任务定义不同
- 窗口不同
- context 长度不同
- metric 不同
- null model 不同
- 训练目标不同
- 数据集不同

在这些变量没统一之前，单谈架构优劣没有意义。

---

## 3.7 当前仓库里“IBL-MtM benchmark”其实不是原始 IBL-MtM

这点必须单独强调。

当前项目里用于旧 1.8.3 benchmark 的：

- `neural-benchmark/adapters/ibl_mtm_adapter.py`

并没有真正调用原始 IBL-MtM 的：

- `NDT1`
- `Masker`
- `prompt`
- `stitching`
- `forward_pred eval pipeline`

它只是一个项目内重写的 simplified wrapper：

- future input 置零
- causal mask
- `nn.TransformerEncoder`
- 线性 head

这在 `20260316_benchmark_legacy_audit_codex.md` 里已经被定性为：

> **legacy simplified baseline**

所以如果问题是“为什么原始 MTM 0.4+，而我们这里才 0.17/0.18/0.2”，首先要明白：

> **我们旧 benchmark 里的 IBL-MtM-like 根本不是原始 MTM。**

它低，不足以推出原始 MTM 在 Perich-Miller 上也低；  
它高，也不足以推出原始 MTM 在 Perich-Miller 上会同样高。

---

## 3.8 更公平的对比方式是什么

如果尽量把 comparison 拉近，我认为更合理的 apples-to-apples 顺序是：

### 第一层：先看更接近的 paper baseline

与我们当前 `250ms fp-bps ≈ 0.21 - 0.22` 更接近的，其实不是 paper 里最好的 `all mask + prompt` 数字，而是：

- `NDT2 temporal mask = 0.2488`
- `NDT1 temporal mask = 0.3092`

这样看，差距其实没有想象中夸张。

### 第二层：统一 metric

要公平，就至少要统一：

- null model：train mean 还是 eval mean
- aggregation：global spike-weighted 还是 per-neuron mean
- split：held-out test 还是 validation / train-session
- slicing：continuous windows 还是 trial-aligned held-out late bins

### 第三层：统一 task

至少应该做一个更像原始 MTM 的 setting：

- 100 bins aligned trial
- 前 90 bins 可见
- 后 10 bins forward prediction
- trial-aligned bps

再让 NeuroHorizon 跟 faithful IBL-MtM 在同一协议上比。

在这之前，用 `0.4 vs 0.2` 直接讨论“谁更强”是不成立的。

---

## 3.9 对 Q3 的结论

我认为造成“MTM paper 看起来是 0.4+，而我们 base v2 是 0.2 左右”的主要原因，按重要性排序如下：

1. **不是同一个任务**
   - 原始 MTM forward prediction：100-bin aligned 序列里预测最后 10 bins
   - 我们 base v2：0.5s history 预测未来 0.25/0.5/1.0s 连续窗口

2. **不是同一个 metric 口径**
   - 原始：per-neuron mean bps，null 来自 eval target mean
   - 我们：global spike-weighted fp-bps，null 来自 train split mean

3. **不是同一个训练目标**
   - 原始：multi-task masked modeling + prompt
   - 我们：direct Poisson future prediction

4. **不是同一个数据集 / 数据组织**
   - 原始：IBL aligned single session，约676 neurons，结构强
   - 我们：Perich-Miller continuous windows，神经元数更少，任务结构不同

5. **当前仓库里的 IBL-MtM-like 结果不是原始 MTM**
   - 旧 1.8.3 wrapper 已被审计降级为 legacy simplified baseline

因此，现阶段最保守也最准确的结论不是：

> “MTM 原始模型显著强于我们的 base v2。”

而是：

> “原始 paper 报告的是一个更强监督/更长上下文/不同聚合口径/不同数据组织下的 forward prediction 结果；这个数字不能直接和我们现在 plan.md 里的 `fp-bps=0.2115` 做一对一比较。”

如果要真正回答“在 Perich-Miller + 统一 fp-bps 下，faithful IBL-MtM 会不会明显高于 base v2”，只能靠下一步的 faithful reproduction。

---

## Q4. NeuroHorizon 自身的数据组织、evaluation 细节与模型实现补充

## 4.1 数据集是怎么组织进训练图里的

当前 NeuroHorizon 训练脚本里的 dataset 入口在 `examples/neurohorizon/train.py::DataModule.setup_dataset_and_link_model()`。

实际装配顺序是：

1. 建 `train_dataset = Dataset(..., split="train", transform=train_transform)`
2. 用 train dataset 初始化：
   - `model.unit_emb` 的 vocab
   - `model.session_emb` 的 vocab
3. 再建：
   - `val_dataset = Dataset(..., split="valid", transform=eval_transform)`
   - `test_dataset = Dataset(..., split="test", transform=eval_transform)`
4. 用 train dataset 统计 null model rate

这说明三件事：

1. **unit / session vocabulary 是 train-first 的**
2. **null baseline 也是 train-first 的**
3. **valid/test 不负责定义索引体系，只负责在既定索引下做评估**

当前用的数据配置是 `examples/neurohorizon/configs/dataset/perich_miller_10sessions.yaml`，只选了 Perich-Miller population brainset 的 10 个 session，且：

- `multitask_readout: []`

说明这个 experiment family 里并没有启用行为 readout 或其它 multitask 输出，主要任务就是 spike prediction。

---

## 4.2 train / valid / test 的划分，不是 dataloader 临时切的

`torch_brain.data.dataset.Dataset` 的 `split` 不是一个仅用于命名的字符串，它会直接决定 sample 可以从哪里取。

### continuous mode 的采样域

`Dataset.get_sampling_intervals()` 会读取：

- `train_domain`
- `valid_domain`
- `test_domain`

也就是说，continuous 训练和评估可用的时间区间，来自底层 HDF5 data object 里预定义的 split domain，而不是训练脚本临时随机切分。

### trial-aligned mode 的 trial 信息

`Dataset.get_trial_intervals(split=...)` 会优先尝试读取：

- `{split}_domain.go_cue_time`
- `{split}_domain.target_id`

如果底层 domain 没带这些 trial metadata，再退回：

- `trials.go_cue_time`
- `trials.target_id`
- `trials.train_mask / valid_mask / test_mask`
- `trials.is_valid`

因此 trial-level split 也不是运行时现算，而是数据层预先定义好的。

这一点很重要，因为它决定了：

- 主指标里 window / trial 的权重分布，本质上是 data-defined，而不是 trainer 随机定义的。

---

## 4.3 session / neuron / time / trial 在训练里各自怎么参与

### session

session 是 dataset 的顶层组织单元。  
每个 `recording_id` 对应一个完整 recording，dataset 通过多个 recording 组成一个多-session 集合。

`Dataset.get_unit_ids()` 和 `get_session_ids()` 会给 id 加前缀：

- unit id 前缀：`brainset/session/`
- session id 前缀：`brainset/`

所以不同 session 中 local unit 编号相同也不会冲突。  
这也是 null lookup 能做到真正 per-session, per-neuron 的原因。

### neuron

在 `NeuroHorizon.tokenize()` 里：

- 当前 slice 中所有 unit 都会进入 `target_unit_index`
- 这些 local unit index 会通过 tokenizer 映射成 global unit index
- `target_unit_mask` 负责在 batch collate 后标记真实 unit 与 padding unit

因此模型不是“每次抽一部分 neuron”，而是：

- **对当前 session slice 里的全部 units 同时预测**
- batch 之间如果 unit 数不同，再通过 padding + mask 对齐

### time

时间先由 sampler 决定窗口边界，再由 `tokenize()` 拆成：

- history window：`[0, hist_window)`
- prediction window：`[hist_window, sequence_length)`

在 history 部分：

- 输入给 encoder 的是 start/end unit tokens 加上 history spike events；

在 prediction 部分：

- 只保留 binned target counts；
- 每个 future bin 的中心时间组成 `bin_timestamps`；
- 模型要预测的是 `[T_pred, N_units]` 的 future spike counts。

因此当前 NeuroHorizon 不是 event-token rollout 到全窗口，而是：

- **history 用 event-style spike tokens 编码**
- **future 用 binned count autoregression 解码**

### trial

trial 只有在 `trial_aligned=True` 时才成为显式采样单元。

- continuous 模式：trial 结构通常不进入 sampler 主逻辑
- trial-aligned 模式：每个 sample 等于一个 trial，对齐到 go cue

这就是为什么同一个 metric 公式，在 continuous 与 trial-aligned 下会得到很不同的数值分布。

---

## 4.4 sampler 如何决定训练分布

### continuous train：`RandomFixedWindowSampler`

训练时，如果不是 trial-aligned，采样器是：

- `RandomFixedWindowSampler(window_length=sequence_length)`

它对每个 sampling interval 大致生成：

- `floor(interval_length / window_length)` 个窗口

并再加随机 left offset。  
这意味着：

1. **长 recording / 长 interval 的 session 会产出更多 sample**
2. **训练分布不是 session-balanced**
3. **同一个 epoch 内更接近“按可用时长加权”的窗口池采样**

因此主指标天然更偏向：

- 数据量更大的 session
- spike 更多的 session / neuron

这与前面 Q1 里看到的 global metric 口径是完全一致的。

### continuous valid/test：`SequentialFixedWindowSampler`

验证和测试时用的是：

- `SequentialFixedWindowSampler(window_length=sequence_length)`

它会顺序覆盖每个 split interval，必要时补最后一个重叠窗口来保证覆盖尾部。  
所以 valid/test 更接近“全覆盖评估”，而不是训练式随机抽窗。

### trial-aligned：`TrialAlignedSampler`

trial-aligned 模式下，一个 sample 直接对应一个 trial：

- 窗口是 `[go_cue_time - obs_window, go_cue_time + pred_window]`
- train 可 shuffle
- valid/test 默认不 shuffle

因此在这个模式下，trial 才是样本分布的第一组织单元。

---

## 4.5 evaluation / metric 的实际计算链路

### 训练阶段

`training_step()` 只做三件事：

1. forward 得到 `log_rate[B, T, N]`
2. 用 `target_unit_mask` 扩成 `[B, T, N]` 做 masked Poisson NLL
3. 记录：
   - `train_loss`
   - `train/mean_pred_rate`
   - `train/mean_target_count`

**没有 train fp-bps。**  
这说明训练时项目刻意避免把“随机抽窗下的即时 metric”当作正式模型质量指标。

### 验证 / 测试阶段主指标

`validation_step()` / `test_step()` 中：

1. forward 得到 `log_rate`
2. 算 masked loss
3. 用 `r2_stats()` 累计：
   - `ss_res`
   - `target_sum`
   - `target_sq_sum`
   - `count`
4. 用 `fp_bps_stats()` 累计：
   - `nll_model_sum`
   - `nll_null_sum`
   - `total_spikes`
5. epoch end 再统一 `all_reduce + finalize`

因此：

- 主 `val/test fp-bps` 是严格 global accumulated；
- 主 `val/test r2` 也是严格 global accumulated。

### per-bin 指标

训练脚本中的 `val/fp_bps_bin{t}` 是 batch-local 算完再 log，口径略松。  
而 `scripts/analysis/neurohorizon/eval_phase1_v2.py` 会对：

- `fp_bps_per_bin_stats()` 的 `model/null/spike` 统计量

做真正的跨 batch 累计，所以离线 per-bin 曲线更可信。

### teacher forcing vs rollout

这个项目内部也存在两条评估路径：

1. **默认主路径**
   - `eval_phase1_v2.py::run_model(..., rollout=False)`
   - 直接调 `model(**inputs)`
   - 对于需要 feedback / prediction-memory 的 decoder，会把 `target_counts` 喂进去

2. **free-running 路径**
   - `run_model(..., rollout=True)`
   - 调 `model.generate(**inputs)`

因此当前项目里：

- 训练与默认验证更接近 teacher-forced forward；
- 真正的 rollout 退化主要依赖 `generate()` 路径做离线分析。

---

## 4.6 AR decoder 与 `PerNeuronMLPHead` 在模型里各自负责什么

### 先看总分工

`NeuroHorizon` 的整体分工是：

1. encoder / processor：把 history spike token 序列压成时间 latent
2. AR decoder：只在 future **时间 bin** 维上做自回归建模
3. `PerNeuronMLPHead`：把每个 future bin 的共享表示变成每个 neuron 的 log-rate

这一点很关键，因为它说明：

- decoder 的主战场是 **时间**
- head 的主战场是 **按 neuron 条件读出**

### `requires_target_counts` 的含义

`NeuroHorizon.__init__()` 中：

- 如果 `decoder_variant` 是 `prediction_memory` / `local_prediction_memory`
- 或 `feedback_method != "none"`

则：

- `self.requires_target_counts = True`

这表示在 `forward()` 里，decoder 可以拿到 target counts 的 shift-right 版本来构造：

- query augmentation feedback
- 或 prediction memory

因此训练时部分 decoder 变体本身就带有 teacher-forced 条件输入。

### `AutoregressiveDecoder`

`torch_brain/nn/autoregressive_decoder.py` 里的 decoder 设计非常明确：

每一层顺序都是：

1. cross-attention 到 history latents
2. optional cross-attention 到 prediction memory
3. causal self-attention over future bins
4. FFN

而且它操作的 token 只有 `T_pred` 个 future bins，不包含 neuron 维展开。  
如果没传 `causal_mask`，模块会自己构造一个 `T_pred x T_pred` causal mask。

这意味着 decoder 学的是：

- “未来各个时间 bin 之间如何递推”

而不是：

- “每个 neuron 自己单独有一条 transformer 序列”

### `PerNeuronMLPHead`

`PerNeuronMLPHead` 的实现是：

1. 对 `bin_repr[B, T, D]` 做 `bin_proj`
2. 对 `unit_embs` 做 `unit_proj`
3. 把 bin 表示扩到 `[B, T, N, *]`
4. 把 unit 表示也扩到 `[B, T, N, *]`
5. 拼接后过一个共享 MLP，输出 `[B, T, N]` 的 `log_rate`

因此这个 head 的本质不是“每个 neuron 一个独立头”，而是：

> **一个共享的、由 unit embedding 条件化的 per-neuron readout**

这有两个直接含义：

1. **不同 neuron 的差异主要来自 unit embedding**
2. **显式 neuron-neuron mixing 并不发生在 head 内**

也就是说，如果模型要表达复杂的 population-level 相关结构，主要还是靠：

- history encoder 产生的共享 latent
- future decoder 产生的共享时间表示

而不是靠 head 本身再做一层 neuron-to-neuron attention。

### `generate()` 为什么重要

`NeuroHorizon.generate()` 会按 future bins 逐步 rollout：

1. 第 `t` 步只解到当前 `t`
2. 取最后一个 bin 的 `latest_repr`
3. 通过 `head` 得到当前 bin 对所有 units 的 log-rate
4. 再把 `exp(log_rate_t)` 作为后续步的 predicted counts feedback / memory

所以只要走 `generate()`，项目就真的进入：

- free-running future-bin rollout

这也是 exposure bias / rollout decay 会出现的地方。

---

## 4.7 对 Q4 的结论

NeuroHorizon 当前的数据和模型组织方式，有三个关键特征：

1. **数据层面**
   - split 是底层数据定义的；
   - continuous 训练分布按可采样时间长度自然加权；
   - session 不做均衡采样。

2. **评估层面**
   - 主 `fp-bps` / `r2` 都是严格全局累计；
   - per-bin 与 rollout 更多是诊断指标，而不是主榜单指标。

3. **模型层面**
   - history 用 event token 编码；
   - future 用时间 bin 自回归解码；
   - neuron 维通过 `PerNeuronMLPHead` 做条件读出，而不是在 decoder 内显式展开。

这三点合在一起，也正好解释了为什么 Q1 里的主指标会呈现出：

- spike-weighted
- session-size sensitive
- 对 low-rate neuron 不够“公平”

这样的性质。它不是偶然写成这样，而是和当前 NeuroHorizon 的 dataflow、loss 和 decoder/readout 设计是一致的。

---

## 最终总结

### Q1

当前项目的主 fp-bps 是：

- per-neuron train null
- global spike-weighted aggregation
- 不做 per-neuron / per-session 平均
- val/test 按 epoch 全局累计 finalize

训练期间不算 train fp-bps；  
训练脚本里的 per-bin 日志比离线 evalfix 的 per-bin 口径略松。

我认为这个口径应继续保留为 NeuroHorizon 的主指标。  
如果要补和 IBL-MtM 更接近的 comparison metric，最合理的是并列新增：

- per-neuron mean fp-bps
- eval-split per-neuron null

但它只能作为对比指标，不应替换当前主指标。

### Q2

Neuroformer 原始 repo：

- 训练是 teacher forcing 的 token-level CE
- 推理是 token rollout + 默认 predicted-past rollout
- 常见窗口大约是 50ms 当前窗口，50-150ms 历史窗口
- exposure bias 明确存在，只是它研究的问题不是我们这种 250-1000ms long-horizon fp-bps

### Q3

MTM paper 的高 bps 不能直接和我们 `base v2` 的 `0.21` 比较，因为：

- metric 口径不同
- context 长度不同
- task 不同
- 数据集不同
- 训练目标不同
- 旧项目内 IBL-MtM-like wrapper 也不是原始模型

如果硬要找更接近的数字，paper 里 `NDT2 temporal = 0.2488`、`NDT1 temporal = 0.3092` 反而比“0.5+ 的 all-mask+prompt”更接近当前 `base v2` 的量级。

### Q4

NeuroHorizon 的数据组织和模型实现可以概括为：

- split / trial metadata 来自底层 dataset domain，而不是 dataloader 临时切分；
- continuous train 采样按可用时间长度自然加权，不做 session-balanced sampling；
- 主 `fp-bps` / `r2` 是严格全局累计；
- 模型是“history token encoder + future bin autoregressive decoder + per-neuron conditional readout”的组合；
- `PerNeuronMLPHead` 负责按 neuron 条件读出，不负责显式 neuron-neuron mixing。
