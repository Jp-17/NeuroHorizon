# 20260316 NeuroHorizon Long-Horizon Review (Codex)

## 1. 项目当前状态与事实依据

本评审基于以下现有事实来源：
- `cc_todo/phase1-autoregressive/1.9-module-optimization/results.tsv`
- `cc_core_files/model.md`
- `cc_core_files/plan.md`
- `torch_brain/models/neurohorizon.py`
- `progress.md`

### 1.1 当前最关键的结果事实

截至 `2026-03-14`，当前项目在 `obs=500ms`、`10 sessions`、`300 epochs`、`rollout eval` 条件下的关键结果如下：

| name | 250ms fp-bps | 500ms fp-bps | 1000ms fp-bps | 结论 |
|------|--------------|--------------|---------------|------|
| `baseline_v2` | `0.2115` | `0.1744` | `0.1317` | 当前最强基线 |
| `20260312_prediction_memory_decoder` | `0.1486` | `-0.0153` | `-0.2590` | 显式 prediction memory 初版在长窗口 rollout 崩塌 |
| `20260313_local_prediction_memory` | `0.1621` | `-0.0105` | `-0.2122` | local-only memory 仍未解决长窗口问题 |
| `20260313_prediction_memory_alignment` | `0.1943` | `0.1513` | `0.1103` | alignment 显著修复 rollout 稳定性 |
| `20260313_prediction_memory_alignment_tuning` | `0.2004` | `0.1526` | `0.1218` | 当前显式 memory 路线最佳，但仍未超过 `baseline_v2` |

### 1.2 当前代码对问题的真实说明

从 `torch_brain/models/neurohorizon.py` 与 `torch_brain/nn/autoregressive_decoder.py` 看：
- 当前 `baseline_v2` 的核心优势来自 `POYO/POYO+ encoder + causal forecasting decoder` 这条主干本身，而不是显式 prediction-memory。
- `prediction_memory` / `local_prediction_memory` 路线是把前一步的 predicted counts 编码成 memory tokens，再让 decoder 通过 cross-attention 读取这组 memory。
- 这确实是一种显式输出反馈，但它反馈的是 **count-level summary**，不是更稳定的 latent dynamics state。
- 训练期和推理期的 memory 输入分布天然不一致：训练期是 `GT counts` 或 `GT/pred mix`，推理期是模型自己的 `predicted counts`。
- `Phase 1.9` 的 `mix_prob + noise/dropout` 本质上是在修补这个 train-inference mismatch；结果表明这种修补有用，但还没有强到反超 baseline。

### 1.3 当前阶段的核心结论

当前项目已经得到一个很重要的研究事实：

> 长时程 neural forward prediction 的瓶颈，并不只是“没有把上一时刻预测喂回来”；至少在当前任务设置下，显式 output-side autoregression 不是一个自动带来更优 long-horizon rollout 的充分条件。

这条负面结果本身是有价值的，因为它帮助项目从“默认相信 AR 会更好”转向“重新判断什么才是正确状态变量与正确目标”。

---

## 2. 对输出端自回归 encoding 路线的判断

## 2.1 这个想法本身有没有道理

有道理，但它只在比较强的前提下才会成立：
- 未来神经活动在给定过去活动时具有足够强的条件确定性；
- 上一步预测结果确实是下一步预测所需状态的高质量摘要；
- 误差不会在 rollout 中快速累计；
- 输出端反馈不会变成训练期可依赖、推理期不稳定的 side channel。

对于 motor cortex 的 spike-count forward prediction，尤其到 `500ms-1000ms`，这些前提通常不自动成立。未来 spike pattern 还受到未观测的行为状态、任务阶段、慢变量、trial-to-trial variability 等影响。此时如果直接反馈 predicted counts，模型经常会学到一个脆弱的 shortcut，而不是真正更强的 dynamics model。

## 2.2 当前项目为什么没有兑现“AR 应该更强”的预期

当前结果更支持以下判断：
- 真正的问题不只是“decoder 没看到 previous prediction”，而是 **反馈变量选错层级**。
- `count -> memory tokens -> decoder` 这条路径更像把 noisy output 重新编码回输入，而不是让模型学习一个更稳定的隐状态演化。
- 因为反馈变量本身就不稳定，所以 horizon 越长，反馈误差越容易积累。
- `alignment` 系列虽然把显式 feedback 从“崩掉”修到了“接近 baseline”，但没有证明显式 feedback 本身比 baseline 更好，只证明了 mismatch 是主问题之一。

## 2.3 这条路线是否还值得继续

值得继续，但应该 **降级为待证实子路线**，而不是当前项目的唯一主线。

更具体地说：
- 不建议继续围绕当前 `prediction_memory_train_mix_prob / input_dropout / input_noise_std` 做大量纯超参搜索。
- 建议只做少量高信息增益实验，验证“AR 的问题究竟是 exposure bias，还是反馈表示本身不对”。
- 如果换成更合理的反馈表示后仍然打不过 `baseline_v2`，那么 output-side explicit AR 应视为一条已基本证伪的主路线。

## 2.4 若继续 AR，最值得做的不是哪些细调

不推荐优先做：
- 更多 `mix_prob` 细网格搜索
- 更多 dropout/noise 细网格搜索
- 继续在 count-level memory token 结构上做小修小补

推荐优先做：
- `latent feedback` vs `count feedback`
- `direct multi-horizon supervision` vs `pure rollout training`
- `uncertainty-aware forecast` vs `single deterministic path`

---

## 3. 与 NDT2 / NDT-MTM / Neuroformer / one-step 方法的比较判断

## 3.1 不应预设“AR 一定比 one-step 更好”

当前证据不支持这个预设。对固定窗口 forward prediction：
- `one-step / direct forecast / masked reconstruction` 的优化通常更容易；
- 它们天然没有 rollout 误差累积；
- 在短窗口任务上，这类方法往往更稳，甚至更强。

AR 只有在以下场景下更可能占优：
- 真正的开放环 rollout 是任务核心；
- previous prediction 是高质量状态摘要；
- 未来的层层条件依赖确实需要逐步生成建模；
- 模型对 train-inference mismatch 有足够强的抑制机制。

## 3.2 对 NDT2 / NDT-MTM 的判断

当前项目中的比较更接近以下结论：
- `NDT2 / NDT-MTM` 这种 parallel reconstruction / masking 路线，并不“落后”，它们只是把问题建模成另一类更容易训练的 forecasting 任务。
- 在短窗口和中窗口上，它们完全可能因为优化稳定性更强而占优。
- 你当前项目最有意义的比较点，不是“AR 是不是天然更先进”，而是：
  - 哪种方法在 `open-loop rollout` 下更稳；
  - 哪种方法在 horizon 增大时衰减更慢；
  - 哪种方法更容易跨 session 泛化；
  - 哪种方法更容易保留可解释的 population dynamics。

## 3.3 对 Neuroformer 的判断

Neuroformer 的“真正自回归”意义更强，但它是 spike-level token generation，不等于适合当前任务。当前项目是 population count forecasting，Neuroformer 则是 spike-event generation，两者目标层级不同：
- Neuroformer 的优点是生成语义更纯粹；
- 缺点是步数长、训练和推理都更重，而且不一定适合高密度电生理 population forecasting。

因此对当前项目更合理的结论不是“Neuroformer 更像 AR 所以更值得学”，而是：
- 如果你的目标是固定窗口的 long-horizon population forecast，count/bin-level latent dynamics 很可能比 spike-level token AR 更合适。

## 3.4 当前比较结论

当前项目更支持这样的判断：

> 对 neural forward prediction，真正要比较的不是“AR vs non-AR”这一个标签，而是“什么表征层级、什么训练目标、什么 rollout 对齐方式，能在 horizon 增大时保留更多有效信息”。

---

## 4. NeurIPS 论文潜力评估

## 4.1 什么样的故事不够强

如果论文主故事只是：
- “我们把 POYO 改成了自回归 decoder”
- “我们想做 neural encoding / forward prediction”
- “我们在几个窗口上做了 fp-bps”

我认为这不够强。原因是：
- 工程味较重，方法创新边界不清；
- AR 本身不是新概念；
- 从当前结果看，显式 AR 也还没有带来明确优势；
- 很难支撑到 NeurIPS 级别的方法学新意。

## 4.2 什么样的故事有机会更强

更有机会的论文故事应重构为：

> 现有 neural forecasting 方法在 long-horizon open-loop rollout 上存在系统性的 horizon decay；其原因并不只是模型容量不够，而是状态表示、训练目标与不确定性建模存在错配。我们提出一种更适合长时程 population forecasting 的建模方式，并在 `250/500/1000ms`、跨 session 和多类 baseline 上证明其优势。

这个故事的重点不再是“用了 AR”，而是：
- 你定义了一个更清晰、更重要的问题；
- 你证明了现有方法为什么失败；
- 你提出的方法抓住了失败根因；
- 你有足够系统的对照和消融。

## 4.3 要达到这个标准，需要什么证据链

至少需要以下证据：
- 明确区分 `teacher-forced`、`direct forecast`、`open-loop rollout` 三类评估；
- 系统展示 horizon 增大时，不同方法的衰减曲线；
- 证明新方法不是只在单一窗口、单一 session 或单一数据集上偶然有效；
- 给出机制消融，说明收益来自更合理的状态建模，而不是更多参数或更重 regularization；
- 最好补充 cross-session 或外部数据验证，避免论文只停留在单一 setting 的 engineering optimization。

## 4.4 当前项目距离这个标准还有多远

我认为当前项目已经有不错的基础，但还没有到位：
- 有强 baseline；
- 有系统的负面结果；
- 有比较完整的 rollout 指标；
- 但主创新点还没有收敛，当前显式 AR 路线也还没有形成决定性胜势。

因此当前项目的正确状态不是“已经有了 NeurIPS story”，而是“已经找到了一个值得写成 NeurIPS story 的问题入口”。

---

## 5. 建议的研究转向

## 5.1 总体方向判断

建议把当前项目的主问题重新定义为：

> 如何在 long-horizon open-loop rollout 下稳定预测 neural population activity，并减缓预测性能随 horizon 增大而快速衰减。

在这个定义下，`output-side explicit autoregression` 不再是默认答案，只是候选之一。

## 5.2 推荐方向一：latent-state rollout 优先于 count feedback

最推荐的转向是：
- 不再直接反馈 `predicted counts`；
- 改为让模型维护一个更稳定的 latent state，并对 latent state 做 rollout；
- 最终由 readout 把 latent state decode 成 future firing rates。

原因：
- latent state 更可能携带稳定的 dynamics 信息；
- 它比 raw counts 更适合作为时间递推变量；
- 能减少把 noisy output 直接回灌造成的误差放大。

## 5.3 推荐方向二：multi-horizon / direct + rollout hybrid objective

建议不要只训练“逐 bin rollout 一条路”。更合适的是联合训练：
- 短程保持逐步一致性；
- 中长程增加 `250/500/1000ms` 的 direct forecast supervision；
- 让模型同时学会一步一步走，也学会对更远 horizon 的整体结果负责。

原因：
- 纯 rollout 容易把长程任务变成误差放大问题；
- direct horizon supervision 更接近你真正关心的评估目标。

## 5.4 推荐方向三：uncertainty-aware forecasting

如果目标是 `1000ms` 级别的预测，就不应该默认单一路径 deterministic forecast 足够。建议考虑：
- stochastic latent variable
- heteroscedastic likelihood
- 或至少提供 horizon-dependent uncertainty summary

原因：
- 长时程未来本身更不确定；
- 如果模型必须输出单一最优路径，它容易在长 horizon 上平均化、塌陷或过度平滑。

## 5.5 推荐方向四：引入行为或隐变量辅助目标

如果数据中可用 behavior / kinematics / task-state，建议增加辅助目标或条件建模。原因是：
- 很多“未来神经活动不可预测”的部分，其实来自没有显式建模未来行为状态；
- 对 motor cortex 来说，neural forecast 与 behavior forecast 往往强耦合。

---

## 6. 下一步最小实验矩阵

下面 3 组实验应优先于继续做大量 prediction-memory 超参搜索。

### 6.1 实验 A：latent feedback vs count feedback

目标：判断 AR 失败是不是因为反馈表征层级错误。

建议对比：
- `baseline_v2`
- `count feedback`（当前 best tuning）
- `latent feedback`（反馈 decoder latent 或 compact latent state）

成功判据：
- 至少在 `250/500/1000ms` 中 2 个窗口超过 `baseline_v2`
- teacher-forced / rollout gap 明显缩小
- 不出现 500ms / 1000ms rollout 崩塌

### 6.2 实验 B：direct multi-horizon head vs pure rollout

目标：判断长时程衰减是不是主要来自训练目标错位。

建议对比：
- 当前纯 rollout 训练
- 增加 `250/500/1000ms` direct horizon supervision 的 hybrid objective

成功判据：
- 长窗口 `fp-bps` 至少在 `500ms` 和 `1000ms` 上超过 `baseline_v2`
- horizon 增大时的性能下降斜率减小

### 6.3 实验 C：behavior-conditioned / auxiliary forecast

目标：判断未来行为或慢变量缺失是否是长时程性能下降的重要根因。

建议对比：
- neural-only forecasting
- neural forecasting + behavior auxiliary target
- neural forecasting + behavior-conditioned latent state

成功判据：
- `1000ms` 预测收益最明显
- rollout 曲线变稳，而不是只提升 teacher-forced 指标

---

## 7. 最终判断与优先事项

### 7.1 总结判断

当前项目最重要的结论不是“AR 已经证明有效”，而是：
- 当前问题本身是重要的；
- output-side explicit AR 并不是当前最可信的最终方向；
- 项目已经积累了足够多的事实，支持向更高层级的 long-horizon forecasting 问题定义升级。

### 7.2 建议优先事项

1. 不再继续围绕当前 prediction-memory 路线做大量纯超参搜索。
2. 优先做 3 组高信息增益对照实验：`latent feedback`、`multi-horizon objective`、`behavior/latent-state auxiliary modeling`。
3. 论文故事从“把 POYO 改成 AR decoder”升级为“long-horizon neural forecasting 的失效机制与改进方法”。

### 7.3 对当前项目的定位

当前项目不是失败，而是完成了一个关键研究阶段：
- 它已经提供了强 baseline；
- 已经给出一条重要的负面结果；
- 已经说明继续做 long-horizon neural forecasting 是有意义的；
- 下一步需要的是研究问题与方法层级的升级，而不是继续在现有显式 count-memory 方案上做小修小补。
