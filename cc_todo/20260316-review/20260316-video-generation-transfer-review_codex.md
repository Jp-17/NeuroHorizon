# 20260316 Video-Generation Transfer Review (Codex)

## 1. 评审目的

本评审讨论一个更具体的问题：

> 当前 `NeuroHorizon` 在 long-horizon neural autoregressive forecasting 中暴露出的 `train-test mismatch / rollout drift / exposure bias`，是否可以从自回归流式视频生成领域借鉴思路，例如 `self-forcing`、`rolling forcing`、`global sink frame`、`memory bank / MemFlow` 等机制？

本评审不是泛泛做跨领域类比，而是要回答三件事：
1. 当前神经活动自回归生成尝试，和流式视频生成是不是结构同类问题；
2. 哪些视频方法值得借，哪些不能直接照搬；
3. 如果借，应该怎样翻译成当前 `NeuroHorizon` 框架里可执行的最小实验。

本评审基于以下两类事实来源：
- **项目内事实**：`cc_todo/phase1-autoregressive/1.9-module-optimization/results.tsv`、`cc_core_files/model.md`、`torch_brain/models/neurohorizon.py`、`cc_todo/20260316-review/20260316-neurohorizon-long-horizon-review_codex.md`
- **外部方法来源**：Self Forcing 项目页 / 论文、Rolling Forcing OpenReview、LongLive（frame sink）、MemFlow（adaptive memory）等公开视频生成资料

---

## 2. 当前 NeuroHorizon 尝试与流式视频生成的相似性

## 2.1 相似的核心问题

当前 `prediction_memory` / `local_prediction_memory` 路线与流式视频自回归生成，在以下结构性问题上是同类问题：

1. **训练-推理分布不一致**
   - 训练时模型可见更干净、更稳定的条件输入；
   - 推理时模型必须依赖自己刚生成的结果继续往后滚。

2. **long rollout error accumulation**
   - 早期小误差会在后续步骤中持续传播；
   - horizon 越长，漂移越明显。

3. **memory 设计容易变成 shortcut**
   - 一条容量很强的显式 memory 通路，在 teacher forcing 下可能很好用；
   - 但自由 rollout 时，这条通路反而可能放大漂移，而不是稳定动态状态。

4. **真正困难不只是下一步预测，而是“如何在自生成条件下维持长期一致性”**
   - 视频里是一致的主体、场景和运动轨迹；
   - 神经活动里是一致的 latent dynamics、task state、behavior coupling 和 firing statistics。

## 2.2 当前项目为什么已经部分接近视频领域的问题设定

`Phase 1.9` 实际上已经在做一类弱版的 video-style forcing：
- 训练时做一次 `no-grad rollout bootstrap`；
- 将 `prediction_memory` 输入从 `shift-right GT counts` 部分替换为 `predicted expected counts`；
- 再用 `mix_prob + dropout + noise` 缩小 train-inference gap。

这和视频领域很多“让模型在训练时更早接触自己的生成结果”的思路是同类方法学，只是当前实现仍停留在：
- self-generated input 的层级仍然是 **raw count / expected count**；
- 训练信号仍主要围绕 local memory 对齐，而不是整段 rollout 的稳定性。

因此，当前项目并不缺“是否想到要做 forcing”，而是缺：
- forcing 的状态变量选得是否正确；
- forcing 是不是足够强；
- 模型是不是有真正稳定长程动态的 global anchor 与 latent memory。

---

## 3. 与流式视频生成的关键差异

## 3.1 神经活动预测不是视频像素预测

不能直接照搬视频方法，原因主要有四点：

1. **状态变量不同**
   - 视频里直接生成的是 high-dimensional visual frame / latent frame；
   - 当前项目生成的是 `future bins x units` 的 mean-rate / count-level neural forecast。

2. **全局锚点不同**
   - 视频里天然存在首帧、场景布局、物体 identity 这类强 global anchor；
   - 神经活动里没有“第一帧真相图”，更合理的 global anchor 是 observation history summary、session/task embedding、behavior latent、slow neural state。

3. **不确定性的来源不同**
   - 视频 drift 很多来自外观/几何/运动一致性破坏；
   - neural forecasting 的不确定性更大程度来自未观测行为变量、trial variability、内在 stochasticity 和 session shift。

4. **当前模型是 deterministic mean-rate predictor，不是 diffusion / stochastic generator**
   - 这意味着视频里很多依赖去噪轨迹或 sample diversity 的技术，不能原样迁移；
   - 真正可迁移的是“如何在自生成输入下训练得更稳”，而不是 diffusion 细节本身。

## 3.2 当前项目的直接启示

这也解释了为什么 `prediction_memory` 初版和 local-only 版都失败：
- 它们借的是“显式反馈通路更强”，而不是“更稳定的状态表示”；
- 在神经活动预测里，raw predicted counts 很可能不是一个好的长期递推状态。

因此，如果要借视频方法，必须先把“frame / video latent / sink frame / memory bank”翻译成神经预测里真正对应的对象，而不是把 raw count memory 继续做大。

---

## 4. 各视频方法的逐项迁移判断

## 4.1 Self-Forcing

**外部方法核心机制**：
Self Forcing 的核心不是简单 scheduled sampling，而是更显式地让模型在训练期间使用自己的生成上下文，并直接优化长序列 rollout 质量，从而减少 test-time drift。

**与当前 NeuroHorizon 的对应关系**：
- 当前 alignment 训练已经有弱版雏形；
- 但当前 self-generated 输入只作用在 `prediction_memory` 的 raw count 路径上；
- 训练目标还没有真正对一段 self-generated rollout 负责。

**迁移判断**：**最值得优先借**。

**为什么值得借**：
- 它直接命中当前项目最明确的问题：teacher-forced 很强，rollout 变差；
- 它比继续调 `mix_prob` 更接近根因；
- 它能把“当前只是 input mixing”升级为“真正训练模型在自生成条件下保持稳定”。

**如何翻译到当前项目**：
- 不继续 self-force raw counts；
- 改为 self-force `latent state`、`compressed decoder state` 或 `compact memory state`；
- 训练时显式 rollout `K` 个未来 chunk，再对整段 rollout 施加监督。

**主要风险**：
- 如果仍然用 raw count 做自生成反馈，只会重演当前 `prediction_memory` 的 mismatch；
- 如果 rollout 太长且没有 chunking，训练成本会迅速上升。

## 4.2 Rolling Forcing

**外部方法核心机制**：
Rolling Forcing 强调用滚动窗口或局部联合生成/去噪来减少长链条误差积累，而不是把整段 long horizon 完全压成逐 token 的脆弱链式生成。

**与当前 NeuroHorizon 的对应关系**：
当前模型在 `1000ms` 上若按 `20ms` bin rollout，相当于要滚很多步；每步的 feedback drift 都会累积。这个设定本身就偏脆弱。

**迁移判断**：**值得借，但必须翻译成 chunkwise autoregression**。

**建议的神经版本**：
- 将未来预测划成 `50ms` 或 `100ms` chunk；
- chunk 内部并行预测多个 bins；
- chunk 与 chunk 之间再做自回归；
- 必要时对 chunk 边界做重叠或平滑，减少 block artifact。

**为什么这比直接逐 bin rollout 更合理**：
- 有效缩短 rollout 深度；
- 更贴近 motor cortex 中很多慢变量和行为阶段在 chunk-scale 上演化；
- 更容易把 `250 / 500 / 1000ms` 视为多个层级 horizon，而不是一条极长脆弱链。

**主要风险**：
- chunk 太大可能丢掉 bin-level temporal precision；
- chunk 太小则无法实质缓解 drift。

## 4.3 Global Sink Frame / Attention Sink / Frame Sink

**外部方法核心机制**：
视频领域的 sink / frame sink 思路，本质上是在受限上下文或流式生成中保留一个强而稳定的全局参考系，避免模型在长序列中完全丢失早期全局结构。

**迁移判断**：**不能直接照搬，但概念上值得借**。

**不能直接照搬的原因**：
- 神经活动预测里没有“首帧图像”这类天然全局真值；
- 如果把某个 future target 或局部 count summary 当 sink，很容易演变成 teacher-forced side channel。

**建议的神经版本：global anchor latent**
- 从 observation window 的 encoder latents 池化出一个固定 `global anchor latent`；
- 再与 session embedding、task embedding、若可用则加 behavior summary 组合；
- 所有 future chunk / bins 都允许 cross-attend 这个 anchor；
- anchor 在 rollout 中默认静态，或仅允许低频、因果的 latent refresh。

**这个方向的真正用途**：
- 不是替代 local AR feedback；
- 而是给长序列 rollout 提供一个不随每步 count noise 快速漂移的参考坐标系。

**主要风险**：
- anchor 如果做得太强，可能压制模型学习真正的动态变化；
- anchor 如果被 future GT 污染，就会变成另一条 shortcut。

## 4.4 MemFlow / Memory Bank

**外部方法核心机制**：
MemFlow 的启发不是“记忆越大越好”，而是“长期信息需要被筛选、压缩、检索，而不是无条件全量保留”。

**与当前 NeuroHorizon 的关系**：
当前 `prediction_memory` 已经说明：
- 高容量显式 memory 非常容易在 teacher forcing 下变成捷径；
- rollout 时又因输入分布变化而失稳。

因此，**直接把当前 count-memory 扩成更大 memory bank，大概率会更糟**。

**迁移判断**：**高风险可借，前提是改成 causal latent memory bank**。

**建议的神经版本**：
- bank 只存储 `causal latent summaries`，不存 raw target counts；
- 以 chunk 为粒度写入，而不是每个 bin 都写高容量 token；
- query 时做 sparse retrieval，而不是全历史 full cross-attention；
- bank 的内容只能来自 observed history 与 self-generated latent，不允许 future GT memory。

**为什么这个版本可能有价值**：
- 如果长时程预测真的需要非局部历史状态、phase template 或更慢时间尺度信息，latent bank 比 raw count bank 更可能稳定；
- 它与 `global anchor latent` 组合后，可能形成“全局慢状态 + 局部可检索历史”的双层结构。

**主要风险**：
- 如果 retrieval 设计不当，它仍会演化成更复杂的 shortcut；
- 如果 bank 更新频率过高，又会退化成现在这类高频显式 memory 反馈。

---

## 5. 对当前项目的整体判断

## 5.1 当前尝试是否类似流式视频生成尝试

**是，但只在问题结构层面类似，不在数据形态层面类似。**

更准确地说，当前项目和流式视频生成的共同点在于：
- 都在解决 self-generated context 下的长期稳定性；
- 都在解决 train-test mismatch；
- 都需要设计“哪些状态应该被持续保留、哪些应该被检索、哪些应该被重新生成”。

但当前项目**不像**视频生成的地方在于：
- 你没有自然的全局视觉锚点；
- 你预测的是 stochastic neural dynamics，不是低频稳定的场景外观；
- raw predicted counts 不是一个理想的长期状态表示。

因此，合理结论不是“视频方案可以直接抄”，而是：

> 视频领域提供的是一套处理 long rollout instability 的设计语言，而不是当前神经 forecasting 可直接复用的模块清单。

## 5.2 对当前 NeuroHorizon 的具体启示

当前项目最应该吸收的启示不是“再加强 memory”，而是：
- forcing 应该作用在更合适的 latent 层级；
- rollout 单位应重新设计，不要默认 bin-level 最优；
- 长 horizon 需要稳定 global reference，而不是只依赖局部上一时刻反馈；
- memory bank 必须是 selective latent memory，而不是 enlarged count memory。

---

## 6. 建议的三组最小实验设计

## 6.1 实验 A：Latent Self-Forcing

**目标**：验证 exposure bias 的主问题是否来自“反馈层级错误”。

**核心改动**：
- 引入 `latent self-forcing` 路径；
- 训练时 rollout `K` 个未来 chunk；
- feedback 使用 compact latent state，而不是 raw predicted counts。

**对照组**：
- `baseline_v2`
- 当前 `prediction_memory_alignment_tuning`
- `latent self-forcing`

**成功判据**：
- `500ms` 与 `1000ms` rollout 至少在其中两个窗口上超过 `baseline_v2`；
- teacher-forced / rollout gap 收缩；
- 不再出现长窗口中后段快速转负。

**主要风险**：
- latent 选得不对，会变成“更难训练的 count feedback”；
- 训练成本高于当前 alignment 版。

## 6.2 实验 B：Chunkwise Rolling Forecast

**目标**：验证长时程 drift 是否主要来自 rollout 深度过长。

**核心改动**：
- 将未来窗口划成 chunk；
- chunk 内并行预测，chunk 间自回归；
- 先从 `100ms` chunk 做最小版本。

**对照组**：
- 当前逐 `20ms` bin rollout
- `100ms` chunk rollout
- 如有需要，再补一个 `50ms` chunk 版本

**成功判据**：
- `1000ms` 衰减曲线明显变缓；
- `500ms / 1000ms` rollout 提升明显；
- `250ms` 不因 chunking 出现明显回归。

**主要风险**：
- coarse chunk 导致短时精度下降；
- chunk 边界引入 block artifact。

## 6.3 实验 C：Global Anchor + Causal Latent Bank

**目标**：验证长时程预测是否需要一个稳定 global reference 与可检索历史状态。

**核心改动**：
- 新增 `global anchor latent`，来自 observation history summary；
- 新增小型 `causal latent memory bank`，以 chunk 为单位写入与检索；
- 严禁 bank 直接存 raw target counts。

**对照组**：
- 无 anchor / 无 bank
- 仅 static global anchor
- global anchor + causal latent bank

**成功判据**：
- 长窗口 rollout 更稳；
- teacher-forced 指标不会出现异常虚高；
- 相比当前 count-based memory，更少出现“训练很强、rollout 很脆”的现象。

**主要风险**：
- 设计不当会重复当前 `prediction_memory` 的 shortcut 问题；
- 检索过强会把 bank 用成另一条 teacher-forced 通路。

---

## 7. 优先级建议

当前最建议的优先顺序是：

1. **先做 `latent self-forcing`**
   - 因为它最直接解决当前 exposure bias；
   - 且与已有 `alignment` 路线衔接最自然。

2. **再做 `chunkwise rolling forecast`**
   - 因为它最可能直接缓解 `500ms / 1000ms` 的深链条误差累积。

3. **然后尝试 `global anchor latent`**
   - 作为抗漂移的稳定器，而不是强监督捷径。

4. **最后才考虑 `causal latent memory bank`**
   - 只在前三者验证后再做；
   - 且必须避免扩展当前 raw count memory 失败路线。

---

## 8. 最终结论

本评审的最终判断如下：

1. 当前神经活动自回归预测与流式视频生成，在 **rollout 失稳** 这个层面上是相似问题。  
2. 视频领域**可以借鉴**，但真正该借的是：
   - 自生成上下文训练（self-forcing）
   - 减少深链条误差的 chunkwise rollout（rolling forcing 的神经版本）
   - 提供稳定全局参考系的 global anchor
   - 选择性、因果的 latent memory 检索
3. 当前项目**不应**直接把 `prediction_memory` 做得更大、更强；这大概率会重复已知失败模式。  
4. 如果跨领域借鉴要变成一条有效主线，最靠谱的起点是：
   - `latent self-forcing`  
   - `chunkwise rolling forecast`  
   - `global anchor latent`  
   而不是 raw count memory bank。

---

## 9. 外部参考（简记）

- Self Forcing 项目页：<https://self-forcing.github.io/>
- Self Forcing arXiv：<https://arxiv.org/abs/2407.01392>
- Rolling Forcing OpenReview：<https://openreview.net/forum?id=i7lrZ2wrQd>
- LongLive / frame sink：<https://arxiv.org/abs/2510.19147>
- MemFlow：<https://arxiv.org/abs/2509.24674>

这些资料的用途是提供机制启发，而不是要求当前项目按其具体视频模型结构逐项复刻。
