# 从自回归流式视频生成到神经活动预测：技术迁移可行性深度分析

> 日期：2026-03-16
> 作者：Claude (Opus 4.6) — 基于 NeuroHorizon 项目实验数据与视频生成前沿文献
> 状态：技术分析文档

---

## 第一部分：结构性类比——AR 视频生成与 AR 神经活动预测

### 1.1 域间映射

自回归（AR）流式视频生成与 AR 神经活动预测在架构层面存在深层的结构性类比。两者都面临同一核心困境：训练时使用 teacher forcing（真实数据作为条件），推理时只能使用模型自身的输出（rollout），由此产生的分布偏移（exposure bias）导致质量随时间步累积衰减。下表系统梳理了两个领域的对应关系：

| 维度 | 视频生成 | 神经活动预测（NeuroHorizon） |
|---|---|---|
| 生成单元 | 帧（frame），通常 24–30 FPS | 时间 bin（20ms），每个 bin 包含所有神经元的发放计数 |
| 空间维度 | 像素/patch（H×W），通过 VAE 编码为 latent tokens | 神经元群体（N 个单元），通过 PerNeuronMLPHead 扩展 |
| 时间维度 | 几十到几百帧（1–60 秒） | 12–50 个 bins（250ms–1000ms） |
| 质量漂移表现 | 过饱和、纹理退化、运动停滞（frozen frame） | fp-bps 衰减、发放率漂移、误差累积（500ms 后 fp-bps 变负） |
| 生成范式 | 扩散去噪（Diffusion denoising） | 泊松率估计（Poisson rate estimation） |
| Teacher forcing | 用干净帧（clean frame）做 KV cache 条件 | 用 GT spike counts 做 prediction memory |
| Rollout | 用自生成帧做 KV cache 条件 | 用预测 counts 做 prediction memory |
| Exposure bias | 帧级分布漂移（连续像素空间） | count 向量分布漂移（离散整数空间） |
| 序列长度 | 几十到几百帧 | 12–50 bins |
| Context window | Sliding window / KV cache（通常 16–64 帧） | Encoder latents（S 个 token） + prediction memory（K=4 token/bin） |
| 典型模型规模 | 数十亿参数 | 约数百万参数（D=128，2 层 decoder） |

### 1.2 关键相似性

**训练-推理不匹配是根本矛盾。** 视频生成中，Self-Forcing（Yang et al., 2025）的核心发现是：teacher forcing 训练的模型在 rollout 时会出现"分布雪崩"——前一帧的微小误差被后续帧放大。NeuroHorizon 的 Round 1 实验（Structured Prediction Memory）在 500ms 窗口出现的灾难性崩溃（fp-bps 变为负值）本质上是同一现象。

**因果自注意力的隐式 AR 能力。** 视频领域的一个重要发现是：因果自注意力本身已经提供了强大的 AR 建模能力，不需要显式的帧间反馈通道。这与 NeuroHorizon 的实验结果高度一致——baseline_v2（仅使用 causal self-attention，无 prediction memory）始终是最优模型。

### 1.3 关键差异

**序列长度差异巨大。** 视频生成处理几十到几百帧，长序列漂移是真正的工程难题；而 NeuroHorizon 仅处理 12–50 个 bins。这意味着视频领域中针对长序列设计的技术（如 attention sink、rolling KV cache）的必要性在 NeuroHorizon 中可能大幅降低。

**时间冗余度截然不同。** 相邻视频帧的相关性极高（temporal redundancy），帧间差异通常很小。而 20ms 粒度的神经发放计数（spike counts）具有高度随机性——即使底层发放率平稳变化，实际观测到的 counts 服从泊松分布，噪声方差与均值同阶。这意味着前一步的预测对后一步的信息贡献可能远低于视频场景。

**输出空间性质不同。** 视频生成的输出是连续的 latent 空间或像素值，适合用 GAN/分布匹配损失直接优化。神经发放计数是非负整数，且分布高度偏斜（大量零值），传统的分布匹配损失（如 GAN discriminator）需要针对性改造。

**模型容量差异。** 视频生成模型通常有数十亿参数，可以承受 Self-Forcing 等计算密集型训练策略的开销。NeuroHorizon 的模型更小（D=128，2 层 decoder），训练资源限制更紧。

---

## 第二部分：六大技术的适配性深度分析

### 技术 1: Self-Forcing — 分布匹配训练

**技术原理。** Self-Forcing（Yang et al., NeurIPS 2025 Spotlight）彻底放弃 teacher forcing，训练时让模型完全从自身输出进行 rollout 生成。为解决 rollout 全程反向传播的内存爆炸问题，采用随机梯度截断（stochastic gradient truncation）——在每一帧只对最后一步去噪进行反向传播，KV cache 边界处梯度分离。损失函数不再是逐帧的重建损失，而是对整个生成序列施加视频级分布匹配损失（如 DMD、SiD 或 GAN），让生成序列在统计上接近真实视频的分布。

**与 NeuroHorizon 的映射。** NeuroHorizon 已有 `_bootstrap_prediction_counts` 机制实现训练时 rollout，并通过 `prediction_memory_train_mix_prob`（如 mix_prob=0.35）以一定概率混合 GT 与预测 counts。这本质上是 Self-Forcing 的弱版本——只有部分步骤暴露于自身预测。Round 3/4 的 alignment training 正是沿这条路线推进，但混合比例和损失函数均未到位。

**具体改造方案。**

- **Slot D（mixing logic）**：将 `prediction_memory_train_mix_prob` 从 0.35 提升至 1.0，即训练时 100% 使用 rollout 预测的 counts 构建 prediction memory，完全消除对 GT 的依赖。
- **Slot G（loss function）**：在现有 Poisson NLL 基础上，增加分布匹配损失。具体方案有三条路径：
  - **(a) MMD 损失**：在 rollout 输出的 count 向量上计算 Maximum Mean Discrepancy（参考 Xiao et al., "Rescuing Neural Spike Train Models from Bad MLE"），使用 RBF 核衡量预测 count 分布与真实 count 分布的距离。
  - **(b) Rollout Poisson NLL**：对 rollout 输出也计算 Poisson NLL（而非仅对 teacher-forced 输出计算），`total_loss = alpha * TF_loss + (1-alpha) * rollout_loss`。这是最简单的方案，不需要新的损失函数。
  - **(c) 隐状态一致性损失**：对 teacher-forced 和 rollout 两条路径的 decoder 隐状态计算 L2/cosine 距离，鼓励两者收敛。
- 反向传播策略：由于 NeuroHorizon 序列短（12–50 bins），无需像视频那样做梯度截断——可以完整反向传播 rollout 全程。

**预期收益。** 直接解决 exposure bias，预期可以缩小 explicit AR 与 baseline_v2 的差距。如果 rollout loss 有效，500ms 窗口的 fp-bps 差距（当前 -0.0218）可能缩窄至接近零。

**风险与挑战。** (1) 纯 rollout 训练在初期可能不稳定（模型预测差 → memory 差 → 预测更差的正反馈循环），需要渐进式 curriculum（从低 mix_prob 逐步提升到 1.0）。(2) MMD 等分布匹配损失在离散 count 数据上的效果尚无先例，需要仔细选择核函数和带宽。(3) 计算开销约增加 1.5–2 倍（需要同时做 TF 和 rollout 前向传播）。

**适配难度：** 中。现有 bootstrap 机制可复用，但损失函数设计需要实验验证。

**推荐优先级：1（最高）。** 这是最有可能带来突破的方向，且与之前 4 轮实验的逻辑一脉相承。

---

### 技术 2: Rolling Forcing — 联合 bin 预测

**技术原理。** Rolling Forcing（2025.9）摒弃逐帧严格因果生成，改为在一个滑动窗口（denoising window）内同时处理多帧，窗口内各帧施加递增的噪声水平（t_1 < t_2 < ... < t_T），允许帧间双向信息流动。只有窗口最前端的帧（噪声最低）被 commit 输出，其余帧在后续窗口中继续去噪。这种"渐进承诺"（progressive commitment）机制使得帧在被最终输出前有多轮相互精炼的机会。

**与 NeuroHorizon 的映射。** 这一思路与 baseline_v2 有深层关联：baseline_v2 本质上就是一次性处理所有 bins（通过 causal self-attention），bins 之间可以前向传递信息（前面的 bin 可以被后面的 bin attend）。Rolling Forcing 的创新在于把严格因果松弛为"chunk 内双向、chunk 间因果"。

对 NeuroHorizon 而言，可以将 50 个 bins（1000ms）划分为若干 chunk（如 chunk_size=10），chunk 内允许完整双向自注意力，chunk 间保持因果性。在推理时，逐 chunk 生成，每个 chunk 内部通过多次迭代精炼。

在 prediction memory 方面，chunk 内不同 bin 的 memory 可以施加不同程度的噪声——最早的 bin 使用较可靠的 memory（前一 chunk 的最终预测），最晚的 bin 使用噪声更大的 memory（逼近先验）。

**具体改造方案。**

- **Slot E（generate loop）**：将逐 bin 的 O(T^2) 生成改为逐 chunk 的生成。每个 chunk 内部做 1–3 次精炼迭代。
- **Slot C（prediction memory cross-attn）**：chunk 内使用双向 memory attention mask（而非当前的因果 mask）。
- 新增参数：`chunk_size`（建议 5–10）、`n_refine_iters`（建议 1–2）。

**预期收益。** chunk 内双向注意力可以让相邻 bins 相互校正，可能缓解误差累积。推理效率也有提升——从 O(T^2) 降至 O(T * chunk_size)。

**风险与挑战。** (1) 对于 NeuroHorizon 这样短的序列（12–50 bins），分 chunk 的收益可能有限——baseline_v2 已经是"一个大 chunk"。(2) chunk 边界处仍有信息断裂风险。(3) 训练方式需配合调整（Mixed Training: TF + Rolling Forcing 交替），增加工程复杂度。

**适配难度：** 中。需修改 forward 和 generate 两个核心函数。

**推荐优先级：3。** 理论上有意义，但 baseline_v2 已经接近 Rolling Forcing 的极限形态（全局 chunk），边际收益可能有限。

---

### 技术 3: Global Sink Frame / Deep Sink — 历史锚定

**技术原理。** Deep Forcing（2025.12）发现 AR 视频模型存在"深层注意力沉降"（deep attention sink）——模型不仅在序列最早的 token 上投入大量注意力（如 LLM 中观察到的 initial token sink），还在序列中段的 token 上投入显著注意力。利用这一发现，Deep Forcing 在滑动窗口中保留 40–60% 的持久 sink tokens（来自最早帧的 KV 状态），并通过 temporal RoPE 重对齐使其与当前帧的位置编码兼容。LongLive（ICLR 2026）进一步提出 frame sink——将前 3 帧永久保留在 KV cache 中。

**与 NeuroHorizon 的映射。** NeuroHorizon 的 encoder latents 在功能上已经充当了 global sink——decoder 的所有 bin queries 都通过 cross-attention 无条件地访问 encoder latents（Slot F: history cross-attn）。这提供了稳定的历史上下文锚点。

但在 prediction memory 中，目前没有类似的锚定机制。可以引入 sink memory tokens——将第一个预测 bin 的 memory tokens 永久保留在所有后续 bins 的 prediction memory cross-attention 中，作为初始状态的参考。

**具体改造方案。**

- **Slot F（history cross-attn）**：增强 encoder latents 的作用——可以让 encoder 额外输出 2–4 个 anchor latents（通过额外的 learned queries），这些 anchor latents 在 cross-attention 中获得更高的权重或独立的注意力头。
- **Slot C（prediction memory）**：将 t=0 的 memory tokens（K=4 个）在所有后续步骤中保持不变（concatenate 到每步的 memory tokens 前面）。

**预期收益。** 为长窗口推理提供稳定的参考锚点，可能缓解 1000ms 窗口的 fp-bps 衰减。

**风险与挑战。** (1) NeuroHorizon 的序列极短（12–50 bins），sink 机制在视频中解决的是几百帧的长距离遗忘问题，在短序列上可能过度设计。(2) Deep Forcing 可以在不微调的情况下实现 12 倍序列外推，但 NeuroHorizon 本身不需要序列外推——prediction window 是固定的。(3) 如果 Round 1 的失败不是因为缺乏锚点，而是因为 exposure bias，sink 无法解决根本问题。

**适配难度：** 低。仅需在 memory cross-attention 中拼接额外 tokens。

**推荐优先级：5。** 实现简单但收益预期有限，适合作为其他技术的辅助手段。

---

### 技术 4: EMA-Sink — 动态记忆锚点

**技术原理。** Reward Forcing / EMA-Sink（CVPR 2026）指出静态 sink 的弊端——永久保留初始帧的 KV 会导致"初始帧复制"（首帧内容过度影响后续生成）和"运动停滞"。EMA-Sink 的解决方案是：用指数移动平均（EMA）持续更新 sink tokens，公式为 `sink_t = alpha * sink_{t-1} + (1-alpha) * current_token_t`，使锚点随序列演进而平滑变化。

**与 NeuroHorizon 的映射。** 这直接对应 PredictionMemoryEncoder（Slot H）的改造。当前的 prediction memory 是"无状态的"——每步独立编码当前 bin 的 counts 为 K=4 tokens，不保留历史 memory 的任何信息。EMA-Sink 建议引入"全局摘要 tokens"（如额外 2 个 tokens），在每一步用 EMA 融合历史 memory 信息。

这一技术可能解决 Round 1 实验的一个具体问题：当 prediction memory 完全依赖前一步的预测 counts 时，一旦某一步预测出错，错误信息会直接传入下一步的 memory 编码器。EMA-Sink 通过对多步 memory 做平滑，可以降低单步预测误差的冲击。

**具体改造方案。**

- **Slot H（PredictionMemoryEncoder）**：在现有 K=4 learned summary queries 基础上，新增 K_ema=2 个 EMA summary queries。在 rollout 的每一步 t：
  - 正常计算 memory_tokens_t（K=4 tokens）
  - 更新 EMA tokens：`ema_t = alpha * ema_{t-1} + (1-alpha) * mean(memory_tokens_t)`
  - 将 [ema_t, memory_tokens_t] 拼接后送入 decoder 的 prediction memory cross-attention
- 需新增 EMA 模块（约 20 行代码），修改 `neurohorizon.py` 中的 `PredictionMemoryEncoder` 类。

**预期收益。** 平滑 rollout 误差传播，可能将 500ms/1000ms 窗口的 fp-bps 衰减速度放缓。与 Self-Forcing 训练联合使用时效果可能更佳。

**风险与挑战。** (1) EMA 的衰减系数 alpha 需要仔细调参——alpha 太高则退化为静态 sink，alpha 太低则平滑不足。(2) 在 NeuroHorizon 的短序列上，EMA 可能来不及收敛到有意义的统计量。(3) 增加了 inference 的状态管理复杂度。

**适配难度：** 低。改动集中在 PredictionMemoryEncoder，约 20–30 行代码。

**推荐优先级：4。** 实现简单，风险低，适合与优先级更高的技术搭配使用。

---

### 技术 5: MemFlow Memory Bank — 选择性记忆

**技术原理。** MemFlow（2025.12, 快手/Kling）提出 Narrative Adaptive Memory（NAM），使用文本 prompt 作为 query，通过 cross-attention 对历史帧进行语义相关性评分（Q_text x K_frame），只检索最相关的帧加入 KV cache。Sparse Memory Activation 机制通过 relevance-gated 过滤进一步减少计算开销。

**与 NeuroHorizon 的映射。** 在 NeuroHorizon 中，没有"文本 prompt"这样的外部语义线索。但核心思想可以迁移：不是按因果顺序使用所有历史 bins 的 prediction memory，而是让当前 bin 的 query 动态选择最相关的历史 memory tokens。

"语义相关性"在神经数据中可以替换为"神经活动模式相似性"——如果某个历史 bin 的发放模式与当前预测目标相似（例如都处于某种行为状态），则该 bin 的 memory 应获得更高权重。

**具体改造方案。**

- **Slot C（prediction memory cross-attn）**：在现有 prediction memory cross-attention 之前，增加一个门控选择模块：
  - 计算当前 bin query 与所有历史 memory tokens 的相关性分数
  - 仅保留 top-k（如 k=2–3）最相关的 memory tokens 参与 cross-attention
- 需新增一个 retrieval 模块（约 50–80 行代码）。

**预期收益。** 在长窗口（1000ms, T=50）时，减少不相关 memory tokens 的干扰，可能提升 fp-bps。

**风险与挑战。** (1) NeuroHorizon 序列极短——1000ms 窗口仅 50 个 bins，每个 bin 产生 K=4 个 memory tokens，总共最多 200 个 tokens。在这种规模下，选择性检索的收益可能微乎其微。(2) 没有外部语义信号（如文本），仅靠 query-key 点积的选择可能不够准确。(3) top-k 选择操作不可微，需要用 Gumbel-Softmax 或 straight-through estimator 近似，增加训练难度。(4) MemFlow 的核心场景是几分钟的长视频，memory bank 规模达到数百帧——这与 NeuroHorizon 的应用场景差距太大。

**适配难度：** 高。需新建检索模块，修改 attention mask 逻辑，且收益不确定。

**推荐优先级：6（最低）。** 技术本身很优雅，但与 NeuroHorizon 的短序列场景匹配度最低。

---

### 技术 6: Streaming Long Tuning — 长序列对齐训练

**技术原理。** LongLive（ICLR 2026）的核心发现是：attention sink 单独使用并不能阻止 AR 生成质量崩溃——**必须配合 streaming long tuning**，即训练时模拟推理时的滚动生成过程。具体做法是：训练时不仅在单个固定长度片段上做 teacher forcing，还做多片段连续生成（训练镜像推理），让模型在训练阶段就习惯长距离的误差累积和 KV cache 滚动更新。

**与 NeuroHorizon 的映射。** NeuroHorizon 当前训练仅在固定长度的 prediction window（250ms/500ms/1000ms）上进行，且每个窗口独立采样，模型在训练时从未经历过跨窗口的连续 rollout。这与视频生成中"训练时只看 16 帧，推理时需要生成 256 帧"的问题高度类似。

虽然 NeuroHorizon 的推理窗口本身不需要超长（最长 1000ms），但在 1000ms（50 bins）内的累积误差已经导致明显的 fp-bps 下降。如果训练时让模型经历完整的 50-bin rollout（而非 teacher forcing），模型就能学会在自身预测的基础上继续做出合理预测。

**具体改造方案。**

- **训练循环修改**：在训练时，以一定概率（如 50%）将 2–3 个连续的 prediction window 拼接为一个超长序列，在这个序列上执行完整的 AR rollout。前一个窗口的预测 counts 直接作为后一个窗口的 prediction memory 输入。
- **Data sampler 修改**：需要从数据中采样连续的时间段（而非独立的窗口），确保拼接的窗口在时间上是连续的。
- 可以采用渐进式训练：先在单窗口上 TF 训练至收敛，再加入多窗口 rollout 训练进行 fine-tuning。

**预期收益。** 让模型在训练时就体验到 rollout 的误差分布，从而学会在自身预测（而非 GT）基础上做出稳健的后续预测。这与 Self-Forcing 的目标一致，但实现方式更简单——不需要设计新的损失函数，只需改变数据采样和训练流程。

**风险与挑战。** (1) 多窗口拼接增加了每个 batch 的序列长度，GPU 内存占用可能增加 2–3 倍。(2) 渐进式训练需要额外的 curriculum 设计（何时从单窗口切换到多窗口）。(3) 如果单窗口内的 rollout 误差已经很大，多窗口拼接可能导致训练不收敛。

**适配难度：** 低。主要改动在训练循环和数据采样，不涉及模型架构。

**推荐优先级：2。** 实现难度低，与 Self-Forcing 互补（Self-Forcing 改损失函数，Streaming Long Tuning 改训练流程），两者可以组合使用。

---

## 第三部分：与之前 4 轮 AR 实验的关系

### 3.1 四轮实验的失败模式分析

| 轮次 | 方法 | 结果 | 视频领域对应 |
|---|---|---|---|
| Round 1: Structured Memory | 强 prediction memory 通道（K=4 tokens，完整因果 mask） | 500ms+ 灾难性崩溃（fp-bps 负值） | Teacher forcing + 强 KV cache → 典型 exposure bias 雪崩 |
| Round 2: Local Memory | 限制 memory 可见范围（仅前 1–2 bins） | 仍然负值，略有改善 | 缩小 KV cache 窗口 → 治标不治本 |
| Round 3: Alignment Training | Scheduled mixing（mix_prob=0.35），部分暴露于自身预测 | 全部正值，接近 baseline_v2 | 弱版 Scheduled Sampling → 方向正确但不够彻底 |
| Round 4: Alignment Tuning | 超参优化（噪声、dropout、mix_prob 调优） | 进一步缩小差距，1000ms gap=0.0099 | 超参搜索 → 收益递减 |

### 3.2 视频领域的经验教训对实验的解读

**教训 1：仅靠架构改动不够，训练范式必须同步改变。** Round 1 和 Round 2 都是纯架构改动（调整 memory 结构和可见范围），没有改变 teacher forcing 训练范式，因此无法解决 exposure bias。这与 LongLive 的发现完全一致——attention sink（架构改动）单独使用无法阻止崩溃，必须配合 streaming long tuning（训练范式改变）。

**教训 2：分布匹配损失优于逐步的似然损失。** Round 3/4 的 alignment training 使用的仍然是 Poisson NLL 损失，只是在输入端做了混合。Self-Forcing 的关键创新不仅是 100% rollout 输入，更是使用分布匹配损失——后者直接惩罚整个序列的统计偏移，而非逐 bin 的对数似然。NeuroHorizon 尚未尝试序列级分布匹配损失。

**教训 3：Relaxed causality 比 strict causality 更稳定。** Rolling Forcing 表明，允许一定程度的双向信息流（chunk 内非因果）可以显著提升生成稳定性。baseline_v2 的成功恰恰验证了这一点——它的 causal self-attention 允许所有 bins 并行处理，实质上是一个"超大 chunk"内的准双向架构（虽然有因果 mask，但训练时所有 bins 同时优化）。这也解释了为什么显式 AR rollout（严格逐 bin）反而更差。

### 3.3 从四轮实验到下一步的逻辑路线

四轮实验的演进路径清晰：

```
Round 1 (强 AR, 纯 TF 训练) → 崩溃
      | 弱化 AR
Round 2 (弱 AR, 纯 TF 训练) → 仍崩溃
      | 改训练范式
Round 3 (弱 AR, 部分 rollout 训练) → 接近 baseline
      | 超参优化
Round 4 (弱 AR, 优化后的部分 rollout) → 差距缩小到 0.0099
```

视频领域的技术指向的下一步：

```
Next Step (强 AR, 完整 rollout 训练 + 分布匹配损失) → ?
```

即：**不再弱化 AR 通道，而是通过改变训练范式（Self-Forcing + Streaming Long Tuning）让强 AR 通道也能稳定工作。** 这是视频领域证明有效的路线。

---

## 第四部分：可行性与优先级排序

### 4.1 综合评估表

| 技术 | 适配难度 | 预期收益 | 实现时间 | 与现有代码兼容性 | 推荐优先级 |
|---|---|---|---|---|---|
| Self-Forcing 分布匹配训练 | 中 | 高 | 1–2 周 | 高（现有 bootstrap 机制可复用） | **1** |
| Streaming Long Tuning | 低 | 中–高 | 3–5 天 | 高（改 data sampler 和训练循环） | **2** |
| Rolling Forcing chunk-wise | 中 | 中 | 1 周 | 中（需改 generate 和 forward） | **3** |
| EMA-Sink | 低 | 中 | 3–5 天 | 高（改 PredictionMemoryEncoder） | **4** |
| Global Sink Frame | 低 | 低–中 | 2–3 天 | 高（加 sink tokens 到 memory） | **5** |
| MemFlow Memory Bank | 高 | 低 | 1–2 周 | 中（需新模块） | **6** |

### 4.2 推荐实施路线

**第一阶段（1–2 周）：Self-Forcing + Streaming Long Tuning**

这两项技术互补且可以叠加：
- Self-Forcing 解决"损失函数层面"的 exposure bias：让模型在 rollout 输出上也被优化
- Streaming Long Tuning 解决"数据分布层面"的 exposure bias：让模型在训练时就体验长距离 rollout

具体建议：
1. 先实现纯 Rollout Poisson NLL（方案 b），这是最简单的 Self-Forcing 变体
2. 同步实现多窗口拼接训练
3. 渐进式 curriculum：前 50 epoch 用 TF，后 50 epoch 逐步提高 rollout 比例至 100%
4. 如果方案 b 不够，再尝试 MMD 损失（方案 a）

**第二阶段（3–5 天）：EMA-Sink**

在第一阶段基础上叠加 EMA-Sink，平滑 rollout 过程中的 memory 信号。

**第三阶段（可选）：Rolling Forcing**

如果前两阶段仍无法超越 baseline_v2，尝试 chunk-wise generation，但优先级较低。

### 4.3 代码改动位置汇总

所有改动集中在 `torch_brain/models/neurohorizon.py`（模型定义）和训练脚本：

| 改动 | 文件 | 涉及 Slot |
|---|---|---|
| Rollout loss | neurohorizon.py: forward/loss 函数 | Slot D + G |
| Streaming long tuning | 训练脚本 + data sampler | 训练循环 |
| EMA-Sink | neurohorizon.py: PredictionMemoryEncoder | Slot H |
| Chunk-wise generate | neurohorizon.py: generate() | Slot E + C |

---

## 第五部分：关键讨论——这些技术能否真正解决根本问题？

### 5.1 视频 AR 成功的前提条件

视频生成中 AR 之所以有效，有一个关键前提：**视频帧之间存在极高的短程时间冗余（temporal redundancy）。** 24 FPS 视频中，相邻帧几乎完全相同，差异仅在像素级的微小运动和光照变化。这意味着：

1. 前一帧包含了后一帧的绝大部分信息——AR feedback 的信息量大
2. 即使 AR 预测有小误差，后续帧仍然有足够的正确信号来自我修正
3. 累积误差的增长速率较慢（因为每步变化小）

Self-Forcing 等技术解决的是：在时间冗余已经存在的前提下，如何利用正确的训练范式来充分利用这一冗余。

### 5.2 神经活动预测的根本不同

20ms 粒度的神经发放计数有截然不同的统计特性：

**高随机性。** 神经发放本质上是泊松过程（Poisson process），即使底层发放率（rate）完全确定，观测到的 counts 仍有显著随机波动。方差等于均值——对于发放率为 5 Hz 的神经元，20ms 窗口内的期望 count 仅为 0.1，实际观测到 0 或 1 的概率分别约为 90% 和 10%。在这种噪声水平下，前一个 bin 的 count 对后一个 bin 的 count 的信息量极为有限。

**低时间冗余。** 虽然底层发放率可能在 50–100ms 的时间尺度上平滑变化（通过 latent dynamics 控制），但 20ms 粒度的 count 序列看起来几乎是随机的。这与视频帧间的高度相似性形成鲜明对比。

**信息层次问题。** NeuroHorizon 的 encoder latents（来自 500ms 的历史窗口，通过 POYO+ Perceiver 编码为 S 个 D=128 维的 token）已经捕获了底层发放率的慢变化趋势。prediction memory 能额外提供的信息——即"当前预测窗口内已生成的 counts 与 encoder 所见历史的偏差"——本质上是噪声的偏差，而非信号的偏差。

### 5.3 核心质疑：AR feedback 的增量信息到底有多少？

这是一个必须正视的问题。baseline_v2 的成功暗示了一个可能令人不安的结论：

> **在 20ms 粒度上，显式 AR feedback（前一步的预测 counts）对后续预测的增量信息贡献接近零。encoder latents 通过 causal self-attention 已经提供了近乎最优的预测基础。**

如果这个假设成立，那么无论用多先进的视频技术来解决 exposure bias，最终结果都只能是**让 explicit AR 追平 baseline_v2**，而非超越它。具体来说：

- 当前 explicit AR 与 baseline_v2 的差距（250ms: -0.0111, 500ms: -0.0218, 1000ms: -0.0099）本质上是 exposure bias 造成的**性能损失**
- 通过 Self-Forcing 等技术可以消除这一损失，使 explicit AR 达到 baseline_v2 水平
- 但 AR feedback 本身不携带额外信息，因此无法超越 baseline_v2

这与视频生成有根本区别：在视频中，消除 exposure bias 后，AR 模型可以利用前一帧的丰富信息来显著提升后续帧的生成质量（这就是为什么 Self-Forcing 可以匹配甚至超越双向模型）。

### 5.4 可能的破局方向

如果 observation-space（count 级别）的 AR feedback 信息量有限，一个更有前景的方向可能是**在 latent space 做 AR**：

- 不在 count 空间做 feedback，而在 decoder 隐状态空间做 feedback
- 即：bin t 的 decoder 隐状态直接作为 bin t+1 的输入（而非通过 counts → memory encoder → tokens 的间接路径）
- 隐状态空间的信噪比远高于 count 空间，temporal redundancy 更强
- 这其实就是 baseline_v2 的 causal self-attention 在做的事情——只不过它是通过 attention 机制隐式实现，而非显式的 recurrence

EAG（Energy-based Autoregressive Generation, 2025.11, arXiv:2407.11949）在 Neural Latents Benchmark 上的工作也提供了启发：它使用 masked autoregressive modeling（随机 masking ratio 0.7–1.0）来处理 exposure bias，本质上是在 latent space 做 AR 生成而非 observation space。

### 5.5 小结

视频技术对 NeuroHorizon 的帮助是**有上界的**：

- **乐观情况**：Self-Forcing + Streaming Long Tuning 完全消除 exposure bias，explicit AR 追平 baseline_v2，然后 AR feedback 提供少量（但非零的）增量信息，最终 fp-bps 超过 baseline_v2 约 0.005–0.01
- **中性情况**：消除 exposure bias 后 explicit AR 追平 baseline_v2 但不超过
- **悲观情况**：neural data 的噪声特性使得即使用视频级的训练技术也无法完全消除 exposure bias（因为噪声本身造成了不可约的 distribution shift）

---

## 第六部分：结论与推荐

### 6.1 最推荐的组合策略

基于上述分析，推荐以下三步组合策略：

1. **Self-Forcing 分布匹配训练（优先级最高）**：将 `prediction_memory_train_mix_prob` 提升至 1.0（纯 rollout 训练），并在 Poisson NLL 基础上增加 rollout loss（可选 MMD）。这是最有可能产生突破的单项改动。
2. **Streaming Long Tuning（优先级第二）**：训练时做多窗口拼接的连续 rollout，让模型在训练阶段就体验完整的误差累积过程。
3. **EMA-Sink（辅助性技术）**：在 PredictionMemoryEncoder 中加入 EMA 全局摘要 tokens，平滑 rollout 过程中的 memory 信号波动。

预期总实施时间：2–3 周。

### 6.2 需要正视的现实

即使上述策略全部成功实施，**AR feedback 对神经活动预测的边际收益可能仍然有限**。baseline_v2 的强势表现不是 explicit AR 失败的"反面"，而是一个**独立的积极信号**——它说明 causal self-attention + encoder latents 已经是非常有效的预测架构。

四轮实验的核心教训不是"AR feedback 没用"，而是"在 observation space 做 AR feedback 的成本-收益比很低"——训练不稳定、exposure bias 严重、增量信息有限。

### 6.3 视频技术最大的启示

从 6 项视频技术的系统分析中，最有价值的不是某个具体技术，而是一个方法论层面的启示：

> **"怎么训练"比"怎么设计架构"更关键。**

视频领域的经验表明：Self-Forcing（改变训练范式）的收益远大于 attention sink / memory bank / chunk-wise generation（改变架构细节）。NeuroHorizon 前两轮实验的失败恰恰是因为只改了架构，没改训练范式；Round 3/4 开始改训练范式后才看到改善。

### 6.4 长远展望

如果 Self-Forcing + Streaming Long Tuning 无法让 explicit AR 超越 baseline_v2，这并不意味着 AR 方向应该放弃。而是应该转向：

1. **Latent Dynamics AR**：在 decoder 隐状态空间而非 observation count 空间做 AR feedback——绕过 count 空间的高噪声和低信噪比问题
2. **Multi-scale AR**：不在 20ms 的精细粒度上做 AR，而是在 100–200ms 的粗粒度上做（发放率在这个尺度上更平滑，AR 的信息量更大），然后在精细粒度上做并行预测
3. **Energy-based AR**：借鉴 EAG 的思路，用 masked autoregressive modeling 替代传统的 teacher-forced AR，从根本上避免 exposure bias

这些方向超出了视频技术直接迁移的范围，但视频领域的研究——特别是 Self-Forcing 对训练范式的深刻反思——为 NeuroHorizon 指明了一条清晰的探索路线。

---

## 参考文献

1. Yang et al., "Self-Forcing: Bridging the Train-Inference Gap in Autoregressive Video Generation," NeurIPS 2025 Spotlight. (Adobe)
2. "Rolling Forcing: Streaming Generation with Joint Denoising Window," 2025.9.
3. "MemFlow: Narrative Adaptive Memory for Streaming Video Generation," 2025.12. (快手/Kling)
4. "Deep Forcing: Attention Sink Analysis for Long-Context AR Video," 2025.12.
5. "Reward Forcing / EMA-Sink: Dynamic Memory Anchoring for Streaming Video," CVPR 2026.
6. "LongLive: Frame Sink + Streaming Long Tuning," ICLR 2026.
7. Versteeg et al., "EAG: Energy-based Autoregressive Generation for Neural Population Dynamics," 2025.11. arXiv:2407.11949.
8. Xiao et al., "Rescuing Neural Spike Train Models from Bad MLE," NeurIPS.
