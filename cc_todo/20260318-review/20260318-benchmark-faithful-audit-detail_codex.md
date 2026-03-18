# 2026-03-18 Benchmark Faithful 审计补充报告（Codex）

## 审查范围

本报告是对 1.8 主线的补充审查，重点不是重复旧 1.8.3 benchmark audit 对 legacy pipeline 的否定，而是回答下面三个当前更关键的问题：

1. 当前 NDT2 / IBL-MtM / Neuroformer 的 faithful 线，代码是否真的已经对接到上游模型核心，而不是继续使用项目内简化 wrapper。
2. 当前 faithful 线与 plan.md 1.3.7 所规定的 NeuroHorizon 默认数据与指标标准，到底对齐到了什么程度。
3. 从代码和结果看，IBL-MtM 与 Neuroformer 现在是否值得继续推进 250ms gate，应该怎样推进才更合理。

本报告的事实依据主要来自：

- cc_core_files/plan.md
- cc_core_files/results.md
- cc_todo/phase1-autoregressive/20260312-phase1-1.8-benchmark.md
- neural_benchmark/adapters/*.py
- neural-benchmark/repro_protocol.py
- neural-benchmark/faithful_ndt2.py
- neural-benchmark/faithful_ibl_mtm.py
- neural-benchmark/faithful_neuroformer.py
- results/logs/phase1_benchmark_protocolfix_*/results.json
- results/logs/phase1_benchmark_repro_faithful_*/results.json

## 总体结论

当前 1.8 已经明显分成三条不能混写的线：

1. legacy simplified baselines
   - 代码入口是 neural_benchmark/adapters/* 与 neural_benchmark/benchmark_train.py
   - 这条线已经不应再被称为 NDT2 / IBL-MtM / Neuroformer 的正式 benchmark，只能叫项目内简化 Transformer baseline

2. protocol-fix legacy reevaluation
   - 代码入口是 neural-benchmark/repro_protocol.py 与 neural-benchmark/benchmark_protocol_repair.py
   - 它的价值是把旧 simplified checkpoint 放回更统一的 held-out protocol 下重评估，得到更严格的 legacy internal reference

3. faithful bridges / runners
   - 代码入口是 neural-benchmark/faithful_ndt2.py、faithful_ibl_mtm.py、faithful_neuroformer.py
   - 这条线的意义是尽量保留上游模型核心，只统一数据入口、窗口协议和评估方式

因此，当前最准确的阶段判断是：

- 旧 1.8.3 benchmark 已经被成功降级为 legacy internal reference
- faithful 线已经完成桥接打通，但仍未完成正式 benchmark 结果收口
- 现在不能再根据旧 1.8.3 的强结论去写 Neuroformer 是最强竞争者、并行预测不如自回归、NeuroHorizon 已严格优于这些原始模型

## 1. 统一标准基准：1.3.7 对 benchmark 的约束

### 1.1 当前应视为硬约束的统一标准

按 plan.md 1.3.7，NeuroHorizon 默认标准至少包括：

- 数据划分：统一来自 torch_brain 的 train / valid / test split，不允许 benchmark 自造分割
- continuous 正式主线：默认使用 continuous，而不是 trial-aligned 作为 benchmark 主协议
- continuous train / valid / test 都应围绕同一 canonical window 定义展开；差异只能体现在是否保留上游训练语义，不应改变 split 来源
- valid/test continuous：必须是 deterministic coverage，而不是训练式随机 eval
- 主指标：global spike-weighted fp-bps + train-split null
- 主结果：默认以 held-out test 为正式结果，valid 主要用于 checkpoint selection 或中间诊断
- ibl_mtm_bps：仅作为 comparison metric，不替代主 fp-bps
- per_neuron_psth_r2：仅作为 trial-aligned 分析指标，不进入 continuous benchmark 主结论

### 1.2 train / valid / test 与 sampler 使用的当前正确理解

这一点必须说清，因为当前 1.8 争议很多都来自把训练方式统一和评估方式统一混成一件事。

- train
  - NeuroHorizon 主线允许使用自己的训练 sampler 与训练目标
  - faithful baseline 不应强迫改成和 NeuroHorizon 完全相同的训练 sampler；否则 faithful 的意义会被破坏
  - 当前更合理的要求是：训练保留上游原生语义，差异必须文档化

- valid / test continuous
  - 这是最应该统一的部分
  - faithful runner 即便不直接调用 SequentialFixedWindowSampler，也至少要在语义上等价于 deterministic no-overlap continuous coverage
  - repro_protocol.py 当前采取的正是这种 canonical continuous windows 语义

- trial-aligned
  - 当前只应作为补充分析
  - 可以统一 per_neuron_psth_r2、trial_fp_bps 等 trial 指标
  - 但不应继续被写成 benchmark 主协议的中心

### 1.3 当前 faithful 线与 1.3.7 的对齐程度

结论应写成三档，而不是简单说已统一或没统一：

- 已满足：
  - 同一 train / valid / test split 来源
  - deterministic continuous valid/test
  - train-split raw-event null
  - held-out test continuous 主评估
  - per_neuron_psth_r2 口径

- 语义对齐但实现不同：
  - faithful runner 使用的是 repro_protocol.py 生成的 canonical windows，而不是直接调用 NeuroHorizon 的 sampler 类
  - 这在评估语义上可以接受，但在实现路径上不应写成完全相同 sampler

- 当前不应强行统一：
  - 训练目标
  - 上游模型原生输入组织方式
  - 上游模型内部推理程序

因此，我对当前 faithful 线与 1.3.7 的判断是：

当前已经基本对齐了 continuous eval semantics，但并没有、也不应该被描述成训练和评估都与 NeuroHorizon 完全同构。

## 2. 每个模型的数据结构适配审计

### 2.1 NDT2

faithful NDT2 与 NeuroHorizon 一样，起点都是 torch_brain 提供的 Perich-Miller raw recording，按 canonical window 切成 fixed observation + prediction windows，再 bin 成 T x N spike counts。这说明在数据源、split、窗口定义这一层，它已经和 NeuroHorizon 对齐。

faithful_ndt2.py 并不把 T x N 直接喂给模型，而是进一步转成上游 BrainBertInterface 所需的 flat token 表达，包括 DataKey.spikes、DataKey.time、DataKey.position、MetaKey.session、LENGTH_KEY、CHANNEL_KEY。这一步的目标是尽量恢复上游 flat tokenization 语义，而不是继续沿用项目内 simplified NDT2Wrapper 的直接吃 counts 的 MAE-like Transformer。

目前已确认并修掉的重要偏差包括：旧 bridge 把所有 session 强行扩成全局 72-channel flat token layout；旧 bridge 的 pad token 选择会触发 mixed-session position 越界；variable-length tokenization 与上游 serve_tokenized_flat 语义不一致。这些问题修复后，当前数据结构已经明显更接近上游。

当前剩下的主要矛盾已经不是输入结构错了，而是：当前 benchmark 目标是 canonical obs 到 pred forecasting，而上游 NDT2 核心目标是 ShuffleInfill。因此，NDT2 的当前问题更像 objective-level mismatch，而不是 adapter 仍然没有把输入接对。

### 2.2 IBL-MtM

faithful IBL-MtM 的起点同样是 canonical windows 的 T x N binned counts，和 NeuroHorizon 的 continuous 协议保持一致。faithful_ibl_mtm.py 将 canonical window 映射成上游 NDT1 + stitching + session prompting 需要的张量契约，包括 spikes_data、target、time_attn_mask、space_attn_mask、spikes_timestamps、spikes_spacestamps、eid、session_id、neuron_regions。这一点和旧 IBLMtMWrapper 已经本质不同。旧 wrapper 只是项目内 counts Transformer；现在则确实进入了上游 NDT1 主路径。

IBL-MtM 当前的最大结构性缺口不是 tensor shape，而是 metadata 生态。这里的关键不是“字段有没有塞进去”，而是“上游模型原本依赖的 metadata 语义有没有被保住”。从当前代码看，tensor contract 本身已经能跑：`session_id / eid` 都被赋成 `recording_id`，`SessionBatchSampler` 也按 recording 纯 session 组织 batch，stitching 与 prompting 的最小语义因此仍然存在；真正缺口在于上游 IBL 生态里那些会影响任务定义和表示学习的 metadata 没有被完整复原。最明显的是三类东西。第一，`eid` 在原始 IBL 代码里不只是字符串键，它对应的是 session identity、split 组织和 prompting/stitching 所依赖的多 session 语境；当前用 `recording_id` 运行时替代后，桥是打通了，但和原始 IBL session 生态并不等价。第二，`neuron_regions` 在当前 bridge 里直接退化成 `['unknown'] * n_units`，这意味着 region-aware 的任务家族在当前数据上基本失活。第三，上游配置和训练器还显式依赖 `mask_regions / target_regions / brain_region / train_session_eid / use_prompt / use_session` 这一整套 metadata 组织，Perich-Miller 只能部分承接其中与 session identity 相关的部分，不能恢复 region-conditioned masking 与 IBL 原始多任务上下文。

所以，region 缺失并不等于 IBL-MtM 就完全没法适配；你说“没有 region 时至少还能保留 neuron + causal”是对的，而且当前代码确实就是这样退化运行的。但这也正是为什么这里要强调 metadata 生态，而不只是 region 字段本身。当前 bridge 已经不再卡在 tensor shape；真正没有完整对齐的是 session identity 的原始语境、region-conditioned 任务族，以及 IBL 特有的多 session / 多区域 metadata 组织。这条线因此更准确地应被叫做“保留 NDT1 + stitching + prompting 核心的受限 faithful”，而不是原始 IBL 生态下的满配 faithful。

### 2.3 Neuroformer

faithful Neuroformer 的起点依然来自 canonical window，但它和前两个模型最大的区别是：真正进入模型的不是 binned counts，而是从 raw spike events 派生出的 token sequence。faithful_neuroformer.py 把当前窗口转成上游 Tokenizer + ID/dt token 流，包括 x.id_prev、x.dt_prev、x.id、x.dt、y.id、y.dt，并保留当前窗口的 re-binned counts 作为统一 benchmark eval 的后处理目标。

这里需要把“模态差异”这件事说得更精确一些。原 repo 确实可以接视觉 / 行为相关模态，但它也可以在纯 neural 条件下运行；当前 faithful bridge 把 `config.modalities = None`、`predict_behavior = False`、视觉分支关闭，这本身不构成 Neuroformer 无法在当前数据上成立的决定性障碍。更准确的说法应是：Neuroformer 当前已经把 spike tokenization 对齐到了比较 faithful 的程度，模态缺失仍然是场景差异之一，但它不是现在最核心的 blocker。真正更需要关心的是：如何把上游 token-level teacher-forced / rollout 推理语义，稳定地落到当前 canonical held-out benchmark 与 count-based 指标上。

所以当前更合理的判断是：Neuroformer 的 spike-event tokenization、teacher-forced train 和 autoregressive generation 都已经基本接通；当前最大的现实问题，不是“没有视觉模态所以不成立”，而是 formal evaluation path 还没有被高效、稳定地收口。

## 3. 每个模型的训练方式与配置审计

### 3.1 NDT2

当前 faithful NDT2 的训练方式已经不再是项目内自写训练 loop 驱动的 counts model，而是直接实例化上游 BrainBertInterface，沿用 ShuffleInfill 路径。直接使用上游模型核心、保留 ShuffleInfill、保留 causal 和 mask_ratio 等关键设定，这些都属于训练方式基本忠实。

存在妥协的部分是：Perich-Miller 不是上游原始 dataset/task harness；为了在当前 benchmark 上运行，需要自行桥接 canonical windows 和 token batch；后续做过 optimizer 和 scheduler 的对齐实验，也做过 warmup 缩放。我的判断是：这条线已经是训练方式基本忠实，但目标不匹配的状态。现在继续微调 optimizer 的信息增益很低，NDT2 更适合作为 objective mismatch 反例保留，而不是当前优先推进对象。

### 3.2 IBL-MtM

当前 faithful IBL-MtM 的训练方式比旧 debug 明显更像上游实现：保留上游 NDT1，保留 stitching + session prompting，保留 ssl combined multi-mask。这里需要把“combined 在当前数据上到底意味着什么”说清楚。按照 faithful_ibl_mtm.py 里的 `choose_training_masking_mode()`，当 `train_mask_mode` 设为 `combined` 或 `all` 时，基础采样集合一定包含 `neuron` 和 `causal`；只有在 `all` 且 batch 内真的存在有效 region annotation 时，才会进一步加入 `intra-region / inter-region`。而当前 Perich-Miller bridge 把 `neuron_regions` 固定成 `unknown`，所以在现实执行中，combined 基本就是 `neuron + causal`。

这说明你的判断有一半是对的：当前 IBL-MtM 的训练里确实保留了 causal 相关任务，因此它和 one-step forward prediction 不是毫无关系；至少模型仍然在学“利用过去上下文推断被遮蔽的未来/因果相关活动”。但另一半也必须说清：这里的 causal masking 仍然是 upstream SSL 多任务体系里的一个训练子任务，不等于 canonical benchmark 里那种“观察窗口全可见、预测窗口整段遮挡、只对未来窗口做统一 forward-pred 评分”的直接优化。换句话说，当前 IBL-MtM 是“训练保留 upstream combined neuron+causal，评估再单独转成 canonical held-out forward prediction”，而不是训练和评估都完全等同于 one-step forward prediction。

所以更准确的判断是：当前训练方式已经达到“部分忠实但有关键场景缺口”。region 缺失后，保留 neuron + causal 仍然是合理且必要的；但这只能说明 bridge 保住了 upstream 训练语义里最接近 forward prediction 的那部分，不意味着已经完整恢复了 IBL-MtM 原始多任务语境。当前 multimask_e1 为负，因此只能说明这个受限 faithful 版本还没有在当前 benchmark 上成立，不能直接否定论文里 forward prediction 的潜力。

### 3.3 Neuroformer

当前 faithful Neuroformer 训练已经保留了最关键的原生部分：upstream tokenizer、id/dt cross-entropy teacher forcing、autoregressive generation、true_past=False / True 双模式。这里需要对原文做一个更精确的修正：当前 faithful bridge 里的 `true_past` 并不是“没有接通，只做了一个近似替代”，而是已经显式接通了 oracle-history 与 rollout 两种 held-out 模式。区别在于，实现路径不是直接复用上游 notebook / simulation 脚本，而是为了接当前 canonical held-out windows 与统一 `fp-bps` 评估，重写了一条 benchmark bridge。具体来说，rollout 模式下，代码会逐样本自回归生成 `ID/dt` token；true_past 模式下，代码先做一次 teacher-forced forward，拿到 `preds['id'] / preds['dt']`，再在 oracle-history 条件下把这些 token logits 解码回 predicted events/counts。语义上它对应的是 `true_past`，只是工程实现并不等同于直接调用上游 simulation.py。

因此，当前真正需要说清的妥协是两点，而不是一句“true_past 只是近似”就带过去。第一，视觉 / 行为分支被关闭，这使它成为一个 neural-only restricted faithful，但这不是致命问题。第二，formal dual-mode eval 的成本非常高：当前 held-out 评估需要在 canonical windows 上分别跑 rollout 和 true_past 两套路径，而 rollout 本身又是逐样本、逐 token、自回归生成，再把生成事件重新 bin 回 20ms count 矩阵。这个成本一部分来自 Neuroformer 原生生成范式本身，一部分来自 faithful benchmark 为了统一 count-based 指标必须做的 event-to-count 后处理，还有一部分来自当前 bridge 仍然是 Python 级逐样本 generation，没有做更深的向量化优化。

我的判断因此要收紧成：训练语义已经比较忠实，当前最大问题不是 `true_past` 没接通，而是 full-data dual-mode held-out eval 的 runtime 还没有被工程上压到可稳定运行的程度。在 250ms formal dual-mode eval 都跑不顺之前，不应该进入 500ms / 1000ms 扩展。

## 4. 模型输出、loss 与 evaluation 路径审计

### 4.1 NDT2

NDT2 原始输出和 loss 仍是 masked / infill 相关预测；当前训练尽量沿用上游 `BrainBertInterface + ShuffleInfill` 语义。这里需要明确回答一个实现细节：当前 faithful NDT2 的 eval，并不是手写成“观察窗口全部可见、预测窗口全部遮挡，然后直接让模型只预测这整段被遮蔽 future bins”的独立 forward-pred runner。代码实际做的是：先沿用上游 task pipeline 拿到 flat `Output.logrates`，再按 `time / position` 把 token-level输出重新拼回 canonical full-window `[T, N]` 预测，最后只截取 prediction window 这一段去计算统一 `fp-bps / R2 / Poisson NLL / per_neuron_psth_r2`。

因此，当前 NDT2 更准确的描述是“训练语义原生，评估语义统一”，而不是“训练和评估都已经被改写成 canonical forward prediction”。这也是为什么它当前的负结果更像 objective mismatch：评估看的是真正的 obs-to-pred forecasting，但训练主体仍然主要在优化 ShuffleInfill，而不是显式优化整段 prediction window 的 held-out forecast。

### 4.2 IBL-MtM

IBL-MtM 原始输出和 loss 是 SSL multi-mask 训练语义；当前训练尽量保留这套语义；当前 eval 则在 held-out 端单独走 canonical forward-pred。这一段和你描述的形式已经非常接近，而且当前代码里确实是显式实现的。`heldout_forward_pred()` 会把 observation bins 保持可见，把 prediction bins 置零，并显式构造 `eval_mask`；随后 `run_ibl_eval_forward()` 将这个 masked input 送回上游 NDT1 核心，最后只取 `preds[:, obs_bins:obs_bins+pred_bins, :]` 去计算统一指标。也就是说，当前 held-out eval 的语义基本就是“观察窗口可见、预测窗口整段遮挡、评估未来窗口预测”。

但这里仍然要把训练和评估分开。训练阶段并不是只做 canonical forward prediction，而是保留 upstream SSL multi-mask；在当前数据上，这个 multi-mask 会主要退化成 `neuron + causal`。所以更准确的说法是：当前 IBL-MtM 的 causal 训练子任务与 one-step forward prediction 明显相关，但它不等于直接优化 canonical held-out forward prediction 本身。当前 faithful 线的真实结构是“训练保留 upstream combined neuron+causal，评估再单独转成 canonical one-step forward prediction”。

这正是为什么当前负结果不能被简单解读成模型无能，更可能意味着上游训练目标到当前 benchmark 主指标之间的可转移性还不够强，或者 metadata 场景差异仍然在吞噬效果。

### 4.3 Neuroformer

Neuroformer 原始输出和 loss 是 token-level ID / dt cross-entropy；当前训练尽量保留原始 token loss；当前 eval 则需要先 autoregressive generate spike events，再 re-bin 成 20ms spike counts，再用统一 fp-bps。当前 faithful bridge 已经把两种 held-out 模式都接上了：rollout 模式下，代码逐样本、逐 token 生成未来 `ID/dt`；true_past 模式下，代码先做 teacher-forced forward，再在 oracle-history 条件下把未来 token logits 解码成事件，最后同样 re-bin 成 count 矩阵。也就是说，这条路线在原理和实现上都已经能做 forward prediction 风格评估，不存在“只能预测 token、不能统一算 bps”的问题。

但这里仍然不能把它简化成“只要 token generation 准，count-based Poisson / PSTH 指标自然就会好，所以只是一个数学转换问题”。原因是这层转换不是无损等价。第一，token-level cross-entropy 优化的是下一个 `ID/dt` 分类是否正确，不是 bin-level Poisson rate 是否校准；同样一个 token 错误，落到 binned counts 后的代价可能差很多。第二，事件级误差在 re-binning 后会发生结构化放大：时间戳稍微偏一个 dt token，可能就把 spike 从正确 bin 推到相邻 bin；重复事件、漏事件、过早 EOS、错误 unit token，都会直接改变多个 bin 的 count。第三，count-based Poisson NLL 和 PSTH-R2 关心的是 rate calibration 与 trial-averaged temporal profile，而 token argmax 正确率高并不自动保证 count-level calibration 好。换句话说，token generation 和 count metrics 高度相关，但不是一个可以视为无损同构的目标。

因此，当前问题不在于这条路径是否可能，而在于它是否已经被公平、高效、稳定地做成 formal benchmark。答案目前仍然是否定的，主要瓶颈不是语义不通，而是 full-data dual-mode held-out generation 的 runtime 仍然过高。

### 4.4 Neuroformer full-data dual-mode held-out generation 为什么贵

这部分成本来源需要拆开看，而不是笼统说“模型慢”或者“适配写差了”。第一，模型原生生成范式本身就贵。上游 Neuroformer 的 simulation 也是按 token step 循环展开，faithful bridge 当前的 rollout 路径同样是逐 token argmax 生成，不是一次性并行输出整个 future count tensor。第二，faithful benchmark 为了统一到当前 `fp-bps / Poisson NLL / PSTH` 口径，必须把生成的 `ID/dt` 事件重新映射回 global unit ids，再 re-bin 成 `[pred_bins, N]` 的 count 矩阵；这一步是当前 canonical count-based benchmark 额外引入的后处理成本。第三，当前桥接实现还有自己的工程成本：`generate_neuroformer_logrates()` 在 batch 内仍然逐样本做 Python 级循环，rollout 和 true_past 两种 held-out 模式又要分别各跑一遍 full-data valid/test。

因此，当前 runtime blocker 既有模型本身的自回归生成复杂度，也有统一 benchmark 时 event-to-count 后处理的客观成本，还有 bridge 目前没有深度向量化优化的实现成本。更准确的结论不是“全是模型的问题”或“全是适配的问题”，而是两者叠加后，当前 full-data 250ms dual-mode formal eval 还没有被工程上压到足够稳定、足够便宜。

## 5. 对当前文档分析的批判性评价

### 5.1 我认同的部分

1. IBL-MtM 和 Neuroformer 理论上都能做 forward prediction，这一点是对的。
2. 尽量保留原代码，只统一数据入口和指标，是正确路线。
3. 当前不应继续围绕 NDT2 做过多后续扩展，这也是合理的。

### 5.2 我认为需要收紧的部分

1. 不能把理论可做直接写成 formal benchmark 已经成立。当前最多到桥接打通，还没有到正式 benchmark 收口完成。
2. IBL-MtM 论文里的高 bps 不能直接拿来要求当前 bridge 也应立刻达到相同量级。训练语义、数据集生态、metadata 条件、任务定义都不完全一样。当前负结果说明 bridge 还没有证明可迁移性，不说明 paper 的结果错误。
3. Neuroformer 现在的主问题不是它不适合做 future prediction，而是 formal eval 的成本还没压下来。这类 blocker 应先工程收口，再谈科学结论。

### 5.3 当前更合理的阶段判断

- legacy simplified baselines：只保留为内部参考，不再写正式 benchmark 结论
- NDT2：当前更像 objective mismatch 已暴露，不再优先推进
- IBL-MtM：值得继续 250ms short formal run
- Neuroformer：值得继续，但先解决 250ms full-data dual-mode runtime blocker

## 6. 我的建议

### 6.1 对 plan.md / results.md / 1.8 记录的文档建议

必须统一回收这些说法：

- 条件完全一致
- Neuroformer 是最强竞争者
- 并行预测方式在长时程任务上不如自回归
- NeuroHorizon 在所有窗口上优于 benchmark

更准确的说法应改成：legacy simplified baseline 的 protocol-fix internal reference 显示，NeuroHorizon 在这些项目内对照上仍更强；但这不构成对原始 NDT2 / IBL-MtM / Neuroformer 的正式公平 benchmark。

### 6.2 对 IBL-MtM 的建议

继续做 250ms short formal run。目标不是立刻超过 NeuroHorizon，而是先看结果是否从当前显著负值向零靠拢。若更多 epoch 后仍稳定显著为负，再把结论落到 objective/domain mismatch stronger than expected。

### 6.3 对 Neuroformer 的建议

当前优先级应高于继续扩 NDT2。先做 runtime unblock，使 250ms dual-mode formal eval 至少能稳定产出完整 results.json。若 250ms 这一关都过不去，不进入 500ms / 1000ms。

### 6.4 对统一标准的建议

benchmark 线应继续尽量向 1.3.7 靠拢，但只统一到下面这层：

- 同一数据源
- 同一 split
- 同一 canonical continuous valid/test eval
- 同一主指标
- 同一 null 口径

不要错误追求：所有模型训练 sampler 完全一致、所有模型训练目标完全一致、所有模型推理程序完全一致。那样会把 faithful reproduction 重新变成项目内重写 baseline。

## 7. 接下来 benchmark 还要继续做什么

### 7.1 NDT2：当前先不继续跟踪

这一条我现在的判断比较明确：NDT2 可以先停，不再作为 1.8 后续优先对象。原因不是“它绝对不行”，而是 faithful runner 已经打通，250ms 下的负结果也已经比较稳定，当前更像 objective mismatch 已经暴露，而不是还差一两个实现 bug 就能翻正。在 IBL-MtM 和 Neuroformer 都还没有完成 250ms gate 之前，继续往 NDT2 上投时间，信息增益太低。

所以更实际的结论是：NDT2 只保留现状记录，不进入新的 250ms 正式训练，也不扩 500ms / 1000ms。后续如果再回来看它，也应该是为了整理 objective mismatch 的反例，而不是继续把它当主 benchmark 竞争者。

### 7.2 IBL-MtM：为什么当前效果差，接下来该怎么推

#### 7.2.1 训练时的 causal masking 与 eval held-out forward prediction 是否真的对齐

这里需要先排除一个容易误判的点：当前 faithful IBL-MtM 的 train sample 时长和 eval sample 时长本身并没有明显不一致。代码里 train 和 eval 都使用同一个 `BenchmarkProtocolSpec`，即 `obs_window_s = 0.5`、`pred_window_s = 0.25`、`bin_size_s = 0.02`。所以问题不是“训练只见过别的窗口时长，评估突然换成 500ms->250ms”。

真正更像问题的是 mask geometry 不对齐，而不是窗口秒数不对齐。当前训练里如果采样到 `causal`，上游 `Masker(mode='causal')` 走的仍然是 temporal/random-token 那一类按 ratio 选时间步的随机 mask 逻辑；faithful bridge 只是把 `causal` 下的 ratio 提到 `0.6`。而 eval 端的 `heldout_forward_pred()` 则是完全显式的：observation bins 全可见，prediction bins 全置零，再对整段 future window 打分。这两者语义相关，但不是同一个训练目标。

因此，当前更值得写进文档的结论不是“训练窗口和评估窗口可能不一样”，而是：训练里的 `causal` 任务并没有精确模拟 eval 时那种“整段 future window held-out”的结构。如果后续继续推 IBL-MtM，我认为最有信息增益的对照不是再换窗口时长，而是补一个更贴近 eval geometry 的训练变体，例如显式 fixed future-window mask 或 forward-pred-like masking。

#### 7.2.2 是不是只是数据规模的问题，或者 cross-session 没适配好

这里只能说“部分可能”，但不能写成唯一解释。当前 faithful IBL-MtM 已经使用了 canonical `train / valid / test` continuous windows，也覆盖全部 recording，不是单 session 或超小数据子集；但 batch 是 `SessionBatchSampler` 组织的 session-pure batching。这样做的好处是尽量保住上游 `eid / session prompting / stitching` 语义，代价是每个优化 step 里跨 session 混合信号被弱化。

所以更准确的判断是：问题不是“没有做跨 session”，而是“跨 session 的利用方式与 NeuroHorizon 主线完全不同”。此外，当前 IBL-MtM 还有一个更强的限制：它是从零训练，不是加载上游已经学到多 session 表示的预训练权重。现有关键结果又主要还是 `e1` 级别试跑，这意味着“还没训练够”和“跨 session 语义利用偏弱”都仍然是合理候选。

如果要排序，我会把这类原因放在窗口时长之前：`from-scratch + session-pure batching + metadata 生态不完整`，比“训练窗口时长不一样”更像当前的主要解释。

#### 7.2.3 eval bps 指标是不是存在尺度不对齐

这里确实存在口径差异，但我不认为它是当前负结果的主因。当前 faithful benchmark 主指标是 NeuroHorizon 的 `fp-bps`：global spike-weighted 聚合，null 来自 train split raw-event mean count per bin。上游 IBL-MtM 常见的 `bits_per_spike / co-bps` 则更接近 per-neuron mean，null 来自 eval data 的 mean firing rate。

这会导致绝对数值不能直接横比，也就是为什么不能把 paper 里 `0.3-0.6` 那类数字直接拿来要求当前 faithful `fp-bps`。但反过来讲，这种尺度差异也不太可能单独解释当前 `-2.95` 这种量级的失败。它更像是在提醒我们“不要做跨论文绝对数值硬对标”，而不是“把指标改一下，结果就会从显著负变成合理正值”。

所以这条应该写成：指标尺度不对齐存在，但它更多影响外部数值比较，不是当前失败的核心来源。

#### 7.2.4 其他我认为更可能的原因

除了上面三条，我认为还有几个更强的候选原因，应该在文档里明确排在前面。

1. 当前是 `from scratch` 训练，没有加载上游多 session 预训练权重。
2. metadata 生态不完整，不只是 region，还有 `eid / session identity / task family` 这些语义缺口。
3. 训练保留 upstream `combined neuron+causal`，评估却看 canonical full future-window forward prediction，本来就存在 train-eval mismatch。
4. 当前最有代表性的结果仍然是 `multimask_e1` 这一类早期试跑，不能据此判定 IBL-MtM 已经收敛后仍然一定无效。

因此，我对 IBL-MtM 的更实际建议是：下一步最值得做的不是改指标，而是补一版比 `multimask_e1` 更正式的 250ms 短训练，并且增加一个更贴近 eval mask geometry 的训练对照。这样才能把“训练不够”和“目标错位”分开。

### 7.3 Neuroformer：为什么当前效果差，接下来该怎么推

#### 7.3.1 训练时的 observation/prediction 窗口和 eval 是否一致

这一点和 IBL-MtM 类似，也要先把错误怀疑排掉。当前 faithful Neuroformer 的 train 和 eval 同样由同一个 `spec` 驱动，代码里明确把 `config.window.prev = spec.obs_window_s = 0.5`，`config.window.curr = spec.pred_window_s = 0.25`，dataset 里也是按 `prev_mask = rel_timestamps < obs_window`、`curr_mask = rel_timestamps >= obs_window` 去切 history 和 future。所以当前没有证据表明 train/eval 的窗口时长本身不一致。

真正的 mismatch 还是目标层面的：训练优化的是 token-level `ID/dt` teacher forcing，而 eval 看的是 rollout 或 true_past 后 re-bin 得到的 count-based `fp-bps`。所以这里更准确的结论不是“训练没见过同样长度的窗口”，而是“虽然窗口时长一致，但训练目标和最终 benchmark 指标不是同一个东西”。

#### 7.3.2 是不是只是数据规模的问题，或者 cross-session 没适配好

这里也不能简单回答“是”或“不是”。当前 faithful Neuroformer 同样已经用上了 canonical split 的全数据 continuous windows，不是单 session 小 demo；但它的 cross-session 组织方式和 IBL-MtM 又不同。当前 bridge 用的是 global unit vocabulary，decode 时再做 session-constrained masking，防止跨 session 预测到无关 neuron token；可是训练阶段并没有显式 session embedding、session prompting 或 stitching 这类机制。

这意味着它并不是“完全没做跨 session”，而是把跨 session 压进了共享 token vocab，再靠 decode 约束兜底。对一个完全从零训练的 token generator 来说，这可能远比 NeuroHorizon 的 per-neuron count forecast 更难。所以 `from-scratch + 缺少显式 session conditioning` 我会放在非常靠前的位置。

如果后续要继续推进，我认为这一条至少和 runtime 一样重要：即便把 runtime 压下去了，如果 cross-session token vocabulary 本身就学不稳，最终 bps 仍然可能很差。

#### 7.3.3 eval bps 指标是不是存在尺度不对齐

对 Neuroformer 来说，尺度不对齐的问题比 IBL-MtM 还更强，因为这里不只是 null baseline 或 aggregation 口径差异，而是目标层级本身就不同。当前统一 benchmark 里，我们先把生成的 `ID/dt` event 序列 re-bin 成 count，再对这个 count tensor 算 `fp-bps / Poisson NLL / PSTH-R²`。faithful bridge 里的 `collect_predicted_counts()` 实际就是把预测事件累计成 count，并取 `log(count)` 进入统一指标。

所以这里的“尺度不对齐”不能只理解成“null 是 train-split 还是 eval-split”。更强的错位在于：Neuroformer 被训练成 token generator，但我们最终用的是 count-based Poisson benchmark 去评价它。这个错位很可能会显著拖累数值，而且比 IBL-MtM 上的 metric mismatch 更强。

不过，我仍然不会把它写成唯一解释，因为如果模型真的在当前 setting 下把 token generation 学得很好，count-based指标理论上也应该明显改善。当前极差结果更像是：`from-scratch + token/count mismatch + session conditioning不足` 一起叠加。

#### 7.3.4 runtime 的问题是不是只是因为 spike 太多、推理太久

我认为不能简化成“就是 spike 太多”。现有记录里，250ms test event 分布大致是 `mean=64`、`p95=113`、`p99=134.9`、`max=167`，而我们已经把 `max_generate_steps` 收紧到 `192`，这意味着生成步数上限本身并没有明显设错。

真正的 runtime 成本来自三层叠加。第一，token-by-token autoregressive decode 本来就贵。第二，faithful bridge 当前是逐样本 Python 循环生成，没有更深的向量化或 cache 优化。第三，formal eval 不只是跑一次 rollout，还要把 full-data valid/test 的 rollout 和 true_past 两种模式都跑完，再做 event-to-count re-binning。

所以更准确的结论是：spike 数量当然会影响 runtime，但它不是唯一来源，甚至不是最关键的单一解释。当前 runtime blocker 更像是 `自回归生成范式 + 双模式 full-data eval + 当前 bridge 实现成本` 的叠加。

#### 7.3.5 还有哪些别的可能原因

除了上面这些，我认为至少还有三条值得明确写进后续判断。

1. 仍然存在 history token truncation 风险。当前 bridge 用 `prev_id_block_size=512`、`id_block_size=256`，250ms target 侧大概率足够，但 500ms observation 侧是否经常截断，当前还没有单独统计。
2. 当前关键证据仍主要来自 `debug_e1`、`smoke_dualmode_v2` 和 formal run 卡住后的中间状态，而不是一个完整收口的 250ms formal training + dual-mode test。
3. 当前也是从零训练，没有上游预训练权重，这一点的重要性不应低估。

#### 7.3.6 一个值得补的参考实验：先试官方风格的更短窗口

在继续推 canonical `500ms obs + 250ms pred` 之前，我认为 Neuroformer 还值得补一个更短窗口的参考实验，作为 sanity check。这个想法现在有比较明确的代码依据：上游 repo 本地配置里确实存在接近这种设定的窗口，例如 `configs/V1AL/mconf.yaml` 中就有 `window.curr = 0.05`、`window.prev = 0.15`。

因此，我建议在文档里把下面这个实验明确写成 Neuroformer 的参考性后续任务，而不是正式 benchmark 改口径：

- `window.curr = 0.05`，即 50ms prediction window
- `window.prev = 0.15`，即 150ms observation window
- 同样报告 `rollout` 和 `true_past` 两种 inference 的 bps

这个实验的价值不在于直接替代当前 canonical benchmark，而在于提供一个方向性判断。如果 Neuroformer 在更接近官方常见窗口的设置下，双 inference 的 bps 能明显改善，那就说明当前 250ms canonical benchmark 至少有一部分是在惩罚更长 horizon、更高 token density 和更强的 rollout accumulation。如果它在 `150ms/50ms` 下仍然显著为负，那么就更支持 `from-scratch + token/count mismatch + session conditioning不足` 才是主因。

所以我会把这条写成：Neuroformer 的下一步不只是一味去解 250ms formal dual-mode runtime，还可以先补一个 `150ms obs + 50ms pred` 的 reference sanity run，作为低成本、高信息增益的判断。

### 7.4 当前 benchmark 后续优先级

1. `NDT2`：停止继续跟踪，只保留现状记录。
2. `IBL-MtM`：优先做 250ms short formal run，并增加一个更贴近 eval full future-window mask geometry 的训练对照。
3. `Neuroformer`：优先解 250ms formal dual-mode eval runtime，并补 history token truncation 统计。
4. `Neuroformer short-window reference`：补一版 `150ms observation + 50ms prediction` 的 rollout / true_past 参考实验，但明确不替代当前 canonical benchmark。

## 8. 7.4 执行计划（2026-03-19）

按照 7.4 的优先级，本轮执行固定收敛为下面三件事：

1. NDT2：不再新增训练或评估，只保留现状记录。
2. IBL-MtM：补一版比 `multimask_e1` 更正式的 `250ms` 短训练，并增加一个显式 `forward_pred` 训练对照，用来直接测试 train/eval mask geometry mismatch 是否是主因。
3. Neuroformer：先把 `250ms` dual-mode formal eval 做成可稳定完成、可恢复、可单独运行的流程；在此基础上再补一版 `150ms observation + 50ms prediction` 的参考实验。

本轮实现默认采用下面的工程收口方式：

- IBL-MtM：在 `faithful_ibl_mtm.py` 中新增 `train_mask_mode=forward_pred`，训练时直接使用与 held-out eval 相同的 future-window masking geometry；同时保留 `combined` 作为 faithful baseline。
- Neuroformer：在 `faithful_neuroformer.py` 中新增 `eval-only` 模式，并把 `rollout` 与 `true_past` 从训练后绑定式评估中拆开，允许独立重跑 held-out eval；同时补充 token/truncation 统计，帮助判断 runtime 与 token density / truncation 的关系。
- 对比输出：新增 `compare_faithful_ibl_mtm.py` 与 `compare_faithful_neuroformer.py`，统一产出 markdown/json 对比摘要。

## 9. 7.4 执行进展

### 9.1 已完成的代码改动（2026-03-19 第一轮）

1. `faithful_ibl_mtm.py`
   - 新增 `resolve_forward_pred_masking_name()`。
   - 新增 `train_mask_mode=forward_pred`。
   - 训练端现在支持显式 future-window held-out control，而不再只支持 upstream `combined / causal / neuron` 这类 masking family。
   - 结果 JSON 新增 `train_protocol`，明确标注 train mask geometry 与 eval geometry 是否 `exact / partial` 对齐。

2. `faithful_neuroformer.py`
   - 新增 `mode=eval`，支持从 checkpoint 单独跑 held-out eval。
   - 新增 `--checkpoint-path`、`--eval-split`、`--inference-mode`、`--skip-trial-eval`、`--progress-every`。
   - `rollout` 评估不再额外做一遍冗余 teacher-forced forward；`true_past` 仍保留单次 teacher-forced decode 语义。
   - 新增 `token_stats`：`prev/curr tokens mean/p95/max` 与 `prev/curr truncation_rate`。

3. 对比脚本
   - 新增 `neural-benchmark/compare_faithful_ibl_mtm.py`。
   - 新增 `neural-benchmark/compare_faithful_neuroformer.py`。

### 9.2 已完成的小规模验证

#### IBL-MtM `forward_pred` smoke v2

输出目录：`results/logs/phase1_benchmark_faithful_ibl_mtm_forwardpred_smoke_v2/`

关键结果：
- `bridge_config.train_mask_mode = forward_pred`
- `train_masking_mode = forward_pred`
- 说明新增训练控制分支已经真实生效，而不是只改了 CLI 参数。

#### IBL-MtM `forward_pred` 小训练验证（1 epoch, limited windows）

输出目录：`results/logs/phase1_benchmark_repro_faithful_ibl_mtm_250ms_forwardpred_smoke_e1/`

关键结果：
- `best valid fp-bps = -7.3364`
- `test fp-bps = -8.4518`
- `history[0].train_mask_counts = {"forward_pred": 4}`
- `train_protocol.eval_geometry_match = exact`

这说明：
- 新增 `forward_pred` 训练模式不仅能跑通，还已经进入正式 train path；
- 但在极小规模 `e1 + limited windows` 下，它并没有立刻优于旧 debug/multimask，后续必须看 full-data `e10` 的结果再下判断。

#### Neuroformer `eval-only` smoke v1

输出目录：`results/logs/phase1_benchmark_faithful_neuroformer_eval_smoke_v1/`

运行配置：
- checkpoint: `results/logs/phase1_benchmark_repro_faithful_neuroformer_250ms_debug_e1/best_model.pt`
- split: `valid`
- mode: `rollout + true_past`
- windows: `max_valid_windows = 2`
- `skip_trial_eval = true`
- `max_generate_steps = 64`

关键结果：
- `valid rollout fp-bps = -11.6581`, `elapsed_s = 1.4330`
- `valid true_past fp-bps = -11.1578`, `elapsed_s = 0.2153`
- `token_stats.valid.prev_tokens_mean = 201.0`
- `token_stats.valid.curr_tokens_mean = 114.5`
- `token_stats.valid.prev_truncation_rate = 0.0`
- `token_stats.valid.curr_truncation_rate = 0.0`

这说明：
- `eval-only` 和 dual-mode held-out path 已经可独立执行；
- `true_past` 比 `rollout` 更快，且当前小样本上略好；
- 在这 2 个 valid windows 上还没有出现 token truncation，至少当前 runtime blocker 不能简单归因为 block size 太小。

### 9.3 当前后台执行状态（2026-03-19 03:38 CST）

- 中间提交已完成并推送：`cf450b2` `补充 7.4 faithful benchmark 执行入口`
- 正式批次已启动：`screen` 会话 `phase1_benchmark_7_4`
- 日志文件：`results/logs/phase1_benchmark_7_4.log`
- 当前已确认进入第一阶段：`IBL-MtM combined_e10`
- 当前顺序脚本固定按如下阶段运行：
  1. `IBL-MtM combined_e10`
  2. `IBL-MtM forwardpred_e10`
  3. `IBL-MtM compare`
  4. `Neuroformer 250ms formal eval-only`
  5. `Neuroformer 150ms/50ms reference_e3`
  6. `Neuroformer compare`

### 9.4 下一步固定安排

1. 先提交本轮 runner / compare / review 文档改动，形成一个可回滚的中间节点。
2. 启动 IBL-MtM 两组 full-data `250ms e10`：
   - `combined_e10`
   - `forwardpred_e10`
3. 用新的 `eval-only` 流程复跑 Neuroformer `250ms` formal dual-mode held-out eval。
4. 再补 Neuroformer `150ms observation + 50ms prediction` 的参考实验。

## 最终判断

当前 1.8 最值得坚持的路线，不是再证明 legacy 结果多漂亮，而是把下面这句话真正做实：

我们已经把原始 NDT2 / IBL-MtM / Neuroformer 接入统一 benchmark 协议，并正在用 250ms gate 判断它们在当前 Perich-Miller forward prediction setting 下，究竟是 objective mismatch、metadata mismatch、runtime mismatch，还是方法本身真的不适合。

如果这句话做实，1.8 就仍然非常有价值；如果继续混写 legacy 与 faithful，整个 benchmark 叙事会继续失真。
