# plan.md 1.2 / 1.3 v2 代码严格审查

> 日期：2026-03-16  
> 审查对象：`cc_core_files/plan.md` 中 `1.2 基础功能验证` 与 `1.3 预测窗口实验` 对应的真实代码、评估脚本与已落盘结果  
> 审查方式：只基于仓库内实现与结果文件做严格技术审计，不新增实验，不修改代码

## 1. 审查范围与事实来源

本次 review 主要核查以下文件是否真正支撑 `plan.md` 在 `1.2` 和 `1.3` 中的叙述：

- `torch_brain/models/neurohorizon.py`
- `torch_brain/nn/autoregressive_decoder.py`
- `torch_brain/utils/neurohorizon_metrics.py`
- `examples/neurohorizon/train.py`
- `torch_brain/data/sampler.py`
- `torch_brain/data/trial_sampler.py`
- `torch_brain/data/dataset.py`
- `scripts/analysis/neurohorizon/eval_phase1_v2.py`
- `scripts/analysis/neurohorizon/eval_psth.py`
- `scripts/tests/test_decoder.py`
- `scripts/tests/test_1_2_4_metrics_verification.py`
- `results/logs/phase1_v2_*/lightning_logs/eval_v2_results.json`

我这次不是在问“这个想法好不好”，而是在问“当前代码到底做了什么、文档是否准确、结果能否按当前写法成立”。

## 2. 总体判断

我的总体判断分三句：

- **是否合理**：合理。当前 v2 baseline 的数据流、模型结构、Poisson 目标和 fp-bps 指标形成了一个基本自洽、可运行、可比较的 forward prediction baseline。
- **是否正确**：大体正确，但 `plan.md 1.2/1.3` 对“自回归预测”的表述强于真实代码做到的程度。当前主结果更准确地说是 **teacher-forced 条件下的 causal future-bin prediction**，而不是严格的 open-loop long-horizon autoregressive benchmark。
- **是否最优**：不是。最主要的问题不在于代码跑不通，而在于 **评估口径、rollout 报告、PSTH 指标定义、以及 per-neuron 输出头的表达能力** 都还偏保守或不够严格。

如果只保留一句话，那就是：**当前 v2 baseline 更像“带 causal mask 的并行未来 bin 预测器”，而不是已经被严格验证过的长时程自由 rollout 自回归生成器。**

## 3. 数据处理 / 数据获取 / 数据使用审查

### 3.1 `tokenize()` 的真实数据流是清楚的，而且整体合理

`torch_brain/models/neurohorizon.py::tokenize()` 的逻辑很明确：

- `history` 侧只保留 `[0, hist_window)` 内的 spike events，作为 encoder 输入。
- `future` 侧只把 `[hist_window, sequence_length)` 内的 spikes 聚成固定 bin 的 `target_spike_counts`，作为 decoder 监督目标。
- `bin_timestamps` 是未来 bins 的中心时间点，而不是边界。
- `target_unit_index` 直接覆盖当前窗口中的全部 units，输出张量形状固定为 `[T_pred, N_units]`。

这条数据流和 proposal 里的“event history -> binned future counts”是一致的，作为 baseline 是合理的。

### 3.2 连续模式的 sampler 与文档叙述并不完全一致

`examples/neurohorizon/train.py` 在连续模式下使用的是 `RandomFixedWindowSampler`，其真实行为来自 `torch_brain/data/sampler.py`：

- 每个 interval 每个 epoch 只按 `window_length` 切窗。
- 会加入一个随机左偏移 `left_offset`。
- 不是严格的 sequential full coverage。
- 也不是 `plan.md` 某些地方写的那种“50% overlap 的连续滑动窗口评估”。

这意味着：

- 训练时这样做没问题，属于随机窗口采样。
- 但验证和主结果报告也沿用同一种 random fixed sampler，会让“窗口实验”更像随机抽样估计，而不是对整个 valid split 的确定性覆盖评估。

**批判性建议 1**：如果 1.3 要被当成正式窗口实验，validation/test 应改成 `SequentialFixedWindowSampler` 或显式固定 step 的 deterministic sampler，而不是继续沿用训练式随机窗口策略。

### 3.3 Trial-aligned sampler 的对齐思路合理，但边界控制不够严格

`torch_brain/data/trial_sampler.py` 的 `TrialAlignedSampler` 很直接：

- 每个 sample = 一个 trial
- 窗口为 `[go_cue_time - obs_window, go_cue_time + pred_window]`

这个定义本身合理，也和“hold -> reach” 的神经科学解释一致。

但它有两个技术问题：

- sampler 自身不检查窗口是否越界到 split/domain 边界之外。
- sampler 自身也不检查这个窗口是否跨越了不希望混入的 trial 边界。

当前实现实际上把这些正确性假设都压给了底层 `trials.train_mask/valid_mask` 和数据本身的 `go_cue_time` 标注质量。

**批判性建议 2**：如果后续还要强调 `trial-aligned` 的神经科学解释，需要单独做一个边界审计，至少统计：

- `go_cue - obs_window` 是否落出允许采样域
- `go_cue + pred_window` 是否超出允许采样域
- `obs_window=1000ms` 这类长窗口有多少比例跨越了前一 trial 的末尾

### 3.4 Null model 的计算思路基本正确

`torch_brain/utils/neurohorizon_metrics.py::compute_null_rates()` 从训练集遍历所有允许采样 interval，按 neuron 统计总 spikes / 总 bins，然后取 `log(mean_count_per_bin)` 作为 null model。

这在定义上是合理的，也符合 fp-bps 的基本思想。这里没有看到明显错误。

它的局限主要不是“算错”，而是：

- null model 是跨所有时间点的单一每神经元均值，不含 trial phase、direction 或慢变量条件。
- 因此它适合作为统一弱 baseline，不适合作为“更强条件化 null”对照。

这一点目前可以接受，但文档中应把它明确定位为 **global per-neuron mean null**。

## 4. 评估指标与聚合流程审查

### 4.1 `fp_bps()` 实现本身基本正确

`fp_bps()` 和 `fp_bps_per_bin()` 的实现逻辑是标准的：

- `poisson_nll_elementwise(log_rate, target)`
- 计算 null 与 model 的 NLL 差值
- 再除以总 spikes 和 `ln 2`

`scripts/tests/test_1_2_4_metrics_verification.py` 里对 null model、random model、oracle model 以及 NLB 风格实现做了交叉检查，这部分是这条 v2 路线里实现最扎实的一段。

我的判断是：**fp-bps 公式层面没有大问题。**

### 4.2 但 1.3 主表里的 `fp-bps / R²` 聚合方式不够严格

`scripts/analysis/neurohorizon/eval_phase1_v2.py::evaluate_continuous()` 当前是这样做的：

- 每个 batch 先算一个 batch-level `fp_bps`
- 每个 batch 先算一个 batch-level `R²`
- 最后对 batch 结果做简单平均

这不是最严格的总体聚合方式。更严格的做法应该是：

- `fp-bps` 用全数据级别累计的 `NLL_null - NLL_model` 与全数据级别 `total_spikes` 直接计算一次。
- `R²` 用全数据级别累计的 `ss_res` 和 `ss_tot` 直接计算一次。

当前这种“先 batch 算，再简单平均”的问题是：

- spikes 更少的 batch 和 spikes 更多的 batch 权重相同。
- 不同 batch 的有效 neuron 数不同，但最终权重一样。
- 对 `fp-bps` 这种非线性 ratio 指标，这不是严格等价的。

**批判性建议 3**：把 `1.3` 所有主表指标都改成全局累计版，不再使用 batch mean 作为正式结果。

### 4.3 一个更关键的问题：1.3 主表默认不是 rollout

这是我认为本次审查里最重要的事实。

`eval_phase1_v2.py` 的 `run_model()` 逻辑非常明确：

- 默认 `rollout=False`
- 只有显式传 `--rollout` 时才调用 `model.generate()`
- 否则就调用 `model(**inputs)`，即 teacher-forced / parallel forward path

而当前落盘的：

- `results/logs/phase1_v2_250ms_cont/lightning_logs/eval_v2_results.json`
- `results/logs/phase1_v2_500ms_cont/lightning_logs/eval_v2_results.json`
- `results/logs/phase1_v2_1000ms_cont/lightning_logs/eval_v2_results.json`

对应的结果分别是：

- `250ms-cont fp-bps = 0.2115`
- `500ms-cont fp-bps = 0.1744`
- `1000ms-cont fp-bps = 0.1317`

这些文件里的 `rollout` 字段没有被明确保存为 true，而且脚本默认不启用 rollout。结合代码逻辑，最保守也最合理的解释是：**1.3.4 主表记录的是 teacher-forced evaluation，而不是 free-running rollout evaluation。**

这件事非常重要，因为它改变了结果的解释边界：

- 如果你说“v2 baseline 在 1000ms 上还能有 0.1317 fp-bps”，这句话在 teacher-forced 语境下成立。
- 如果你说“v2 baseline 已经证明了 1000ms 的 open-loop autoregressive 生成能力”，这句话目前证据不够。

**批判性建议 4**：后续文档必须把 `teacher-forced` 和 `rollout` 分成两张主表，不要继续合称为“自回归预测性能”。

### 4.4 训练阶段的 per-bin 记录不完整

`examples/neurohorizon/train.py::validation_step()` 只记录 `min(T, 12)` 个 bins 的：

- `val/fp_bps_bin{t}`
- `val/poisson_nll_bin{t}`

这意味着：

- 250ms 条件下 12 bins 刚好覆盖完整 horizon
- 500ms 条件下只有前 12 / 25 bins 被记录
- 1000ms 条件下只有前 12 / 50 bins 被记录

所以如果文档里把这些曲线解释成“完整长时程衰减曲线”，这是不准确的。当前训练日志最多只能支持“前段衰减趋势”。

**批判性建议 5**：对 500ms / 1000ms 条件，训练期和离线评估都应记录完整 horizon 的 per-bin fp-bps，而不是只截前 12 个 bin。

### 4.5 PSTH-R² 的定义和落地实现并不一致

这里存在一个非常值得明确指出的口径问题。

`torch_brain/utils/neurohorizon_metrics.py::psth_r2()` 的定义是：

- 输入 `{target_id: [n_trials, T, N]}`
- 先对每个方向按 trial 平均得到 `[T, N]`
- 再对所有方向和所有神经元一起 flatten 做 R²

这个定义是“保留神经元维度”的 population PSTH-R²。

但 `eval_phase1_v2.py::evaluate_trial_aligned()` 并没有这么算。它实际做的是：

- 先对每个 trial 在 neuron 维做 masked mean，得到 `[T]`
- 再按 target_id 聚合 trials
- 最后算的是基于 `[n_trials, T]` 的 **population-mean PSTH-R²**

这会带来两个后果：

- 它比真正的 per-neuron PSTH-R² 更宽松，信息损失更大。
- 它容易把“单神经元错很多，但总体均值轨迹还行”的模型评得过高。

这也正好解释了为什么 `results.md` 中会出现一种“trial 模型 continuous fp-bps 很差，但 PSTH-R² 反而很高”的现象。那不一定只是“过拟合但保留了群体结构”，也可能是因为指标先把 neuron 维平均掉了，导致难度被显著降低。

更进一步，`scripts/analysis/neurohorizon/eval_psth.py` 又走了另一套逻辑：它确实把完整 `[n_trials, T, N]` 喂给 `psth_r2()`。也就是说，**仓库里现在至少有两种不同口径的 PSTH-R²**。

**批判性建议 6**：PSTH 指标必须拆分成两个明确名称：

- `population_mean_psth_r2`
- `per_neuron_psth_r2`

然后在 `1.3/1.4/1.5` 全部统一口径，否则这个指标现在不适合作为强结论依据。

## 5. Decoder / generate / PerNeuronMLPHead 审查

### 5.1 当前 decoder 是合理的高效设计，但不是“真正 AR 已经落地”的充分证据

`torch_brain/nn/autoregressive_decoder.py` 里的 decoder 结构是：

1. history cross-attention
2. prediction memory cross-attention（可选）
3. causal self-attention
4. FFN

这个设计本身是合理的，也符合 proposal 的“decoder 只沿时间维建模”的效率诉求。

但需要澄清一件事：

- `causal self-attention` 只说明未来 bin 不能看未来 bin 的 hidden state。
- 它不自动等于“模型已经在训练和主评估中做了 free-running autoregressive generation”。

当前主线代码里，这两者仍然是不同概念。

### 5.2 `generate()` 确实是逐 bin rollout，但主训练和主表并不依赖它

`torch_brain/models/neurohorizon.py::generate()` 的逻辑很明确：

- 逐步从 `t=0` 到 `t=T-1`
- 每一步都重新跑一遍当前前缀长度的 decoder
- 用上一步预测的 rate 作为下一步 feedback / memory 的来源

这是真正的 rollout 路径。

但当前主训练和 `1.3.4` 的主表并不是围绕这条路径设计的。也就是说：

- `generate()` 存在，不代表主结果已经在测它。
- `forward()` 的 teacher-forced 性能和 `generate()` 的 rollout 性能不能直接等同。

这也是为什么后面几轮 `prediction_memory` 分支会暴露出明显的 TF-rollout gap：它们正是在显式测这个差距。

### 5.3 `PerNeuronMLPHead` 的实现是合理 baseline，但明显不是最优输出头

当前 `PerNeuronMLPHead` 的结构非常简单：

- `bin_repr -> bin_proj(dim -> dim/2)`
- `unit_emb -> unit_proj(dim -> dim/2)`
- 拼接后走 3 层 MLP 输出一个标量 `log_rate`
- 最后 clamp 到 `[-10, 10]`

这个头有三个优点：

- 计算简单，参数量小。
- 对可变 neuron 数量天然兼容。
- 避免在 decoder 主体里显式展开 `T x N` 自注意力，效率很好。

但它也有三个很明确的局限：

1. **输出条件独立性太强**：每个 neuron 的输出只共享同一个 `bin_repr`，神经元间相关结构没有在 readout 阶段被显式建模。
2. **表达力偏保守**：只有 `concat(bin_repr, unit_emb)` 再过 MLP，本质上是一个共享模板加 neuron identity 调制。
3. **对长时程 population state 的刻画可能不足**：如果未来动力学中 neuron-neuron 协同结构很重要，这个头未必是最优归纳偏置。

因此我的结论是：**这个输出头适合作为 baseline，不适合被表述成“已经相对最优”的 readout 设计。**

**批判性建议 7**：至少应该补一个更强 readout 对照，例如：

- low-rank factorized head
- neuron-conditioned bilinear head
- mixture/readout factor head

否则目前很难知道 long-horizon 衰减有多少来自 dynamics 本身，有多少来自输出头容量不足。

### 5.4 另外一个工程问题：`generate()` 是前缀重算，复杂度并不优雅

当前 `generate()` 每步都重跑整个当前前缀，复杂度接近 `O(T^2)` 次 decoder 计算。对 `T<=50` 的 Phase 1 还可接受，但如果以后扩到更长窗口或更细 bin，它会很快变成瓶颈。

这不是 correctness 问题，但它说明当前实现更像“研究原型”，还不是可扩展的正式 rollout 基础设施。

## 6. 对 1.2 基础功能验证的严格评价

### 6.1 `1.2.4` 的指标验证是扎实的，但范围仍然偏基础

优点：

- 对 `fp-bps` 做了 null / random / oracle / NLB cross-check
- 对 trial sampler 做了结构与对齐验证

不足：

- `Test 3b` 名义上写“真实 checkpoint fp-bps > 0”，但实际上只验证了 checkpoint 存在且可加载，并没有真的把真实模型推理后算出 fp-bps。
- 这意味着 `1.2.4` 并没有真正完成“真实模型级别的指标闭环验证”，只是完成了公式与数据结构验证。

所以 `1.2.4` 应被表述为：**指标实现正确性验证**，而不是“评估闭环已经完全验证”。

### 6.2 `1.2.3` 的“AR 修复验证”还不能算完全闭环

当前仓库确实实现了：

- `prediction_feedback.py`
- `feedback_method`
- `prediction_memory` / `local_prediction_memory`
- `generate()` 中的 rollout feedback

但 `1.2.3` 想回答的核心问题其实是两个：

- TF 和 AR 是否真的不再等价
- 修改第 t 步预测是否真的影响 t+1 之后的输出

从代码层面看，后续 `1.9` 的分支已经间接说明“是的，显式 feedback 路径确实生效了”，因为 TF-rollout gap 被大量暴露出来了。

但从测试层面看，仓库里并没有一个足够强的 automated regression test，把这个性质在主线里固定下来。

`scripts/tests/test_decoder.py` 主要验证的是 causal self-attention，不是 prediction feedback 生效性。

**批判性建议 8**：给 `1.2.3` 补一组真正的回归测试：

- 对 `generate()` 做 step intervention test
- 改写第 `t` 步预测，验证 `t+1...T` 的输出显著变化
- 对 `query_aug` / `prediction_memory` / `local_prediction_memory` 都做同样测试

## 7. 对 1.3 预测窗口实验的严格评价

### 7.1 1.3.4 作为 baseline 窗口实验是有价值的

就 baseline 而言，`1.3.4` 的实验设计并不差：

- 250 / 500 / 1000ms 三个预测窗口
- continuous / trial-aligned 两种采样方式
- fp-bps / R² / PSTH-R² 三类指标

这些维度足够形成一个基础实验矩阵。

### 7.2 但它的命名和解释需要收紧

当前最准确的命名不应是“长时程自回归预测实验”，而应更接近：

**teacher-forced multi-bin forward prediction benchmark with causal decoder**

原因前面已经说了：

- 主表默认不是 rollout
- continuous eval 不是 deterministic full coverage
- PSTH-R² 口径也不统一

因此目前能成立的强结论是：

- 在当前 teacher-forced 口径下，v2 baseline 在 250/500/1000ms 上的 fp-bps 分别为 `0.2115 / 0.1744 / 0.1317`
- 性能会随 horizon 增大而下降，但下降相对平滑

目前还不能同样强地说：

- 这个模型已经在 open-loop 自由 rollout 上同样稳定

### 7.3 Trial-aligned 结果的解释也需要收紧

现在 `results.md` 中把 `trial-aligned continuous fp-bps 为负，但 PSTH-R² 反而很高` 解释为“过拟合但保留了群体模式”，这个解释方向不算错，但不完整。

更完整的解释应该是：

- 该模型很可能确实在 sample-level generalization 上表现差
- 同时 PSTH 指标被 neuron mean 降维后，难度下降，导致群体均值曲线仍然容易被拟合出来

因此当前 `trial-aligned` 的高 PSTH-R² 不能被过度当成强证据。它更像一个提示：

- 模型抓到了一些 direction-level 低频结构
- 但没有抓到足够强的 per-trial、per-neuron 预测能力

### 7.4 `1.3` 的主要短板其实是评估协议，不是模型完全无效

这一点也需要讲清楚。当前问题不是“v2 baseline 没价值”，而是：

- baseline 是有价值的
- 代码也大体正确
- 但评估协议还没有严格到足以支撑最强版本的论文表述

这是一个很典型的“工程上能跑、研究上还需收紧口径”的状态。

## 8. 最终 verdict 与优先级建议

### 8.1 最终 verdict

- **合理**：合理。当前 v2 代码作为 Phase 1 baseline 完全成立，数据流、loss、null model、主要评估脚本都能工作。
- **正确**：基本正确，但文档对“自回归预测”与“teacher-forced causal prediction”的区分不够严格，PSTH-R² 也存在实现口径漂移。
- **最优**：不最优。最需要优先修的不是 decoder 主体，而是评估协议、rollout 报告方式、PSTH 指标统一，以及输出头表达能力对照。

### 8.2 我认为优先级最高的 8 条改进建议

1. **把 teacher-forced 与 rollout 主结果彻底分栏**。今后所有 `fp-bps / R² / PSTH-R²` 表格都必须明确写清楚是哪一种。
2. **把 `1.3` 主表改成全局累计版指标**，不要再用 batch-level 简单平均作为正式数值。
3. **validation/test 改为 deterministic sequential coverage sampler**，避免当前 random fixed window 口径混入正式 benchmark。
4. **统一 PSTH 指标定义**：分成 `population_mean_psth_r2` 和 `per_neuron_psth_r2` 两个指标，不再混写成一个统称。
5. **对 500ms/1000ms 记录完整 horizon 的 per-bin fp-bps**，不要只截前 12 个 bins。
6. **把 rollout-only 结果提升为主报告对象之一**，尤其是 250/500/1000ms 三个窗口的 open-loop 曲线。
7. **给 `PerNeuronMLPHead` 补一个更强 readout 对照**，否则无法排除是输出头限制了 long-horizon 表现。
8. **回收 `plan.md/results.md` 中过强的 AR 表述**。当前最准确的定位仍是 v2 causal forward prediction baseline，而不是已经严格证实的长时程 AR 生成系统。

## 9. 一句话总结

当前 `plan.md 1.2/1.3` 对应的 v2 代码不是“错”，而是“**baseline 成立，但研究口径说得比代码严格性更靠前了一步**”。如果接下来要把这条线继续做强，最先该补的不是新的 decoder 花样，而是 **更严格的 rollout-first evaluation protocol**。
