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

IBL-MtM 当前的最大结构性缺口不是 tensor shape，而是 metadata 生态：Perich-Miller 没有 IBL 风格的 eid 体系，没有可靠的 region annotations，也没有原始 IBL 多任务上下文。当前 bridge 的做法是：用 recording_id 替代运行时需要的 session identity，保留 stitching + session prompting，region 缺失时把 neuron_regions 退化为 unknown。

因此，这条线可以视为保留核心模型结构的部分 faithful，而不是原始 IBL 生态下的满配 faithful。这也是为什么当前负结果不能简单解释成 IBL-MtM 本身做不了 forward prediction。更准确的说法是：该模型的核心已经对上了，但所依赖的上游 metadata 语境并没有被完整复原。

### 2.3 Neuroformer

faithful Neuroformer 的起点依然来自 canonical window，但它和前两个模型最大的区别是：真正进入模型的不是 binned counts，而是从 raw spike events 派生出的 token sequence。faithful_neuroformer.py 把当前窗口转成上游 Tokenizer + ID/dt token 流，包括 x.id_prev、x.dt_prev、x.id、x.dt、y.id、y.dt，并保留当前窗口的 re-binned counts 作为统一 benchmark eval 的后处理目标。

Neuroformer 当前最大的场景差异是模态，而不是 spike tokenization：原 repo 可以接视觉 / 行为相关模态，而当前 Perich-Miller 只有 neural-only 输入，因此当前 faithful bridge 只能做 no-vision / no-behavior 的 restricted faithful。

所以当前更合理的判断是：Neuroformer 的 spike-event tokenization、teacher-forced train 和 autoregressive generation 都已经基本接通；但当前 benchmark 场景不是它原始工作最强的使用方式。

## 3. 每个模型的训练方式与配置审计

### 3.1 NDT2

当前 faithful NDT2 的训练方式已经不再是项目内自写训练 loop 驱动的 counts model，而是直接实例化上游 BrainBertInterface，沿用 ShuffleInfill 路径。直接使用上游模型核心、保留 ShuffleInfill、保留 causal 和 mask_ratio 等关键设定，这些都属于训练方式基本忠实。

存在妥协的部分是：Perich-Miller 不是上游原始 dataset/task harness；为了在当前 benchmark 上运行，需要自行桥接 canonical windows 和 token batch；后续做过 optimizer 和 scheduler 的对齐实验，也做过 warmup 缩放。我的判断是：这条线已经是训练方式基本忠实，但目标不匹配的状态。现在继续微调 optimizer 的信息增益很低，NDT2 更适合作为 objective mismatch 反例保留，而不是当前优先推进对象。

### 3.2 IBL-MtM

当前 faithful IBL-MtM 的训练方式比旧 debug 明显更像上游实现：保留上游 NDT1，保留 stitching + session prompting，保留 ssl combined multi-mask。当前妥协点也必须明写：在 Perich-Miller 上，combined 实际可采样到的主要是 neuron + causal；region-aware mask 基本无法恢复；batch 使用 session-pure 组织，是为了优先保住 prompting 语义。

我对它的判断是：当前训练方式已经达到部分忠实但有关键场景缺口。这个缺口不是随便加一层 wrapper 就能消掉的，而是数据集 metadata 不匹配。因此当前 multimask_e1 为负，只能说明当前受限 faithful 还没有在这个 benchmark 上成立，不能直接否定论文里 forward prediction 的潜力。

### 3.3 Neuroformer

当前 faithful Neuroformer 训练已经保留了最关键的原生部分：upstream tokenizer、id/dt cross-entropy teacher forcing、autoregressive generation、true_past=False / True 双模式。主要妥协有三点：视觉 / 行为分支被关闭；true_past 做了兼容性近似，用 teacher-forced 前向输出来解码 oracle-history 语义；formal dual-mode eval 成本过高，当前还没有稳定跑完 full-data 250ms formal held-out。

我的判断是：训练语义已经比较忠实，当前最大问题不是语义错误，而是 runtime blocker。在 250ms formal dual-mode eval 都跑不完之前，不应该进入 500ms / 1000ms 扩展。

## 4. 模型输出、loss 与 evaluation 路径审计

### 4.1 NDT2

NDT2 原始输出和 loss 仍是 masked / infill 相关预测；当前训练尽量沿用上游 loss 语义；当前 eval 则统一映射回 canonical T x N 预测，再计算 fp-bps、R2、Poisson NLL、per_neuron_psth_r2。这里的风险不在于无法评估，而在于统一评估头看的是真正的 forward prediction，而上游训练目标不直接优化这个目标。所以 current negative result 更像训练目标和 benchmark 目标没对齐，而不是指标脚本不公平。

### 4.2 IBL-MtM

IBL-MtM 原始输出和 loss 是 SSL multi-mask 训练语义；当前训练尽量保留这套语义；当前 eval 则在 held-out 端单独走 one-step forward_pred。这点很关键：IBL-MtM 当前不是训练和评估都按 forward prediction 做，而是训练保留 upstream SSL，评估统一转成 canonical forward prediction。这本身是合理的，因为 faithful reproduction 的重点是尽量不改原训练目标；但同时它也意味着，负结果不能直接被解读成模型无能，更可能意味着上游训练目标对当前 benchmark 主指标的直接可转移性不够强。

### 4.3 Neuroformer

Neuroformer 原始输出和 loss 是 token-level ID / dt cross-entropy；当前训练尽量保留原始 token loss；当前 eval 则需要先 autoregressive generate spike events，再 re-bin 成 20ms spike counts，再用统一 fp-bps。

所以，对当前核心问题，我的回答是明确的：Neuroformer 理论上完全可以接到统一数据接口上，训练预测 spike events，然后再按当前方式 re-bin 成 counts 去算 bps。这条路径在原理上没有问题。当前问题不在这件事是否可能，而在它是否已经被公平、高效、稳定地做成 formal benchmark。答案目前仍然是否定的，原因主要是 token generation 到 count-based Poisson / PSTH 指标之间天然有一层变换，mismatch 比 counts model 更强，而且 full-data dual-mode held-out generation 的 runtime 仍然过高。

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

## 最终判断

当前 1.8 最值得坚持的路线，不是再证明 legacy 结果多漂亮，而是把下面这句话真正做实：

我们已经把原始 NDT2 / IBL-MtM / Neuroformer 接入统一 benchmark 协议，并正在用 250ms gate 判断它们在当前 Perich-Miller forward prediction setting 下，究竟是 objective mismatch、metadata mismatch、runtime mismatch，还是方法本身真的不适合。

如果这句话做实，1.8 就仍然非常有价值；如果继续混写 legacy 与 faithful，整个 benchmark 叙事会继续失真。
