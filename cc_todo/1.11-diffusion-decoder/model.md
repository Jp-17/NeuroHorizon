# 1.11 Diffusion Decoder 路线记录

> 本文档记录 NeuroHorizon 从 1.9 AR decoder 优化线转向 diffusion decoder 主线后的路线分析、实现决策和后续迭代历史。

## 背景与迁移动机

`1.9` 中围绕 autoregressive decoder、prediction memory、alignment training 和 tuning 做了多轮优化尝试，但在正式 rollout 指标上始终未能稳定超越 `baseline_v2`。这一结论在以下文档中已经比较明确：

- `cc_todo/20260316-review/ar_effectiveness_claude.md`
- `cc_todo/20260316-review/long_horizon_prediction_claude.md`
- `cc_todo/20260316-review/option_d_implementation_claude.md`

当前判断是：在 bin-level forward prediction 设定下，显式 AR feedback 的边际信息增益过低，而 train / inference mismatch、Poisson 噪声累积和高维 count 向量的反馈压缩会持续放大 rollout 风险。因此 1.11 不再把“继续打磨 AR feedback”作为主线，而是转向 **整体式 future count generation**。

## 当前主线决策

### 主线

- **方案**：`Option 2A latent diffusion`
- **latent 形态**：`time x factorized latent units`
- **第一版实现**：deterministic factorized count autoencoder + latent rectified flow matching
- **默认实验策略**：`250ms gate-first`
- **默认实施分支**：`dev/diffusion`

### 历史对照路线

- `Option 2B direct count-space flow matching`
  - 保留为历史 baseline / failure case 对照
  - 三轮迭代后不再作为默认继续修正的主线
  - 后续只有在 2A 暴露出更强的实现或表征瓶颈时，才重新评估是否回到 count-space 路线

## 与现有代码的结合原则

1. **保留 POYO+ encoder 主干**：history spikes 的编码方式不变，继续复用当前 `NeuroHorizon` 的 history encoder。
2. **去掉 1.9 运行时专用 decoder 分支**：当前分支的主实现不再继续维护 prediction-memory / alignment 的运行时逻辑，只保留 baseline 对照能力和历史文档。
3. **训练与评估入口优先不变**：
   - 训练入口：`examples/neurohorizon/train.py`
   - 离线正式评估入口：`scripts/analysis/neurohorizon/eval_phase1_v2.py`
4. **协议优先保持 1.3.7**：
   - 10 sessions
   - continuous window
   - obs = 500ms
   - pred = 250 / 500 / 1000ms

## 2026-03-20 — Direct Count-Space Flow Matching with DiT

> 状态：250ms gate 未通过（停止扩窗）
> 分支：`dev/diffusion`
> 对应任务记录：`cc_todo/1.11-diffusion-decoder/20260320_direct_count_flow_dit.md`

### 想法描述

将预测窗口的 `spike count matrix [T, N]` 视为整体生成目标，在 `log1p(count)` 空间中做 rectified flow matching。为避免直接对 `T × N` token 做 full attention 带来的计算爆炸，这一版采用 **count-space summary + 时间维 DiT 主干 + per-unit scalar head** 的结构：

1. history spikes 经过现有 POYO+ encoder，得到 history latents
2. target counts 经过 `log1p` 变换后与 Gaussian source 做线性插值，构造 `x_t`
3. 用 `x_t` 与 unit embedding 做 per-bin summary，形成时间维 token
4. 用 DiT 风格时间主干在 `T` 维上建模，并通过 cross-attention 条件化到 history latents
5. 用共享 per-unit head 回归每个 `(time bin, unit)` 的 velocity
6. 推理时从 Gaussian source 出发，做 Euler integration 得到最终 future count 场

### 选择这一实现的原因

- 它保持了 `Option 2B` 的核心精神：**不引入单独 autoencoder，直接在 count target 上建模**
- 它复用了现有 `unit_emb` 和 `PerNeuronMLPHead` 风格的读出逻辑，能兼容当前变长 unit 设计
- 它避免了 `T × N` flatten token 带来的显存不可控问题，更适合当前代码和实验资源

### 与 2A / 传统 diffusion 的区别

- 不是 latent diffusion：没有单独 count autoencoder
- 不是标准 DDPM：训练目标是 rectified flow matching 的 velocity regression
- 不再做 bin-level AR rollout：生成在每个 Euler step 上都是整体 future count field 的更新

### 预期优点

- 避免 AR feedback 的误差累积
- 推理阶段不存在 teacher forcing / rollout gap
- 结构上更贴合“整体 future field generation”的目标

### 主要风险

- 仅使用 per-bin summary 可能损失部分 unit-level 细节
- flow integration 步数和 validation 成本会高于 baseline_v2
- 直接在 count-space 建模的收敛速度与稳定性仍存在不确定性

### 首轮执行计划

1. 增加 `plan.md` 的 1.11 管理章节
2. 新建 diffusion 主线文档和任务记录
3. 在 `NeuroHorizon` 中挂接 `diffusion_flow` decoder 分支
4. 扩展训练和离线评估脚本
5. 新建 250/500/1000ms 三窗口配置
6. 先做 shape / loss / generate smoke 验证
7. 再执行三窗口正式训练与离线评估

### 当前默认超参原则

- 维持 `1.3.7` 的数据协议不变
- 保持现有 `dim=128`、`enc_depth=6` 的主干规模，先控制变量
- `flow_steps_eval` 初始设为 20，先保证 rollout 质量和评估耗时可接受
- 依赖允许按价值与必要性引入，但首版优先在现有 PyTorch/Lightning 体系内完成

### 当前进展（2026-03-20）

- `plan.md` 的 1.11 管理章节已建立
- `diffusion_flow` 代码主干已挂接到 `NeuroHorizon`
- 训练入口 `examples/neurohorizon/train.py` 已支持 diffusion training loss 和 smoke 专用 batch 限制
- 评估入口 `scripts/analysis/neurohorizon/eval_phase1_v2.py` 已支持 diffusion checkpoint 与 smoke 级 `--max-batches`
- 250ms smoke 已完成：
  - 训练、validation、checkpoint 生成和 best ckpt 选择全部通过
  - 离线 valid smoke 也已成功跑通
  - 当前 smoke 指标显著为负，说明第一版结构只是链路打通，还没有性能结论
- 三窗口 formal 已完成：

  | window | best val fp-bps | best epoch | test fp-bps | vs baseline_v2 test | test R2 | test PSTH-R2 |
  |--------|------------------|------------|-------------|----------------------|---------|--------------|
  | 250ms | -7.3585 | 229 | -7.4950 | -7.7173 | -20.0418 | -13.6578 |
  | 500ms | -7.7572 | 179 | -7.8601 | -8.0341 | -20.7506 | -9.6705 |
  | 1000ms | -8.0657 | 199 | -8.2277 | -8.3625 | -20.7491 | -9.1904 |

### 首轮 formal 结论

- 这轮实验证明 diffusion 主线的工程链路已经打通，但当前 `direct_count_flow_dit` 变体在三个窗口上都属于**结构性失败**，不适合作为主线继续调参。
- 三窗口训练都完整跑满 300 epochs，而 `best val/fp_bps` 却提前停在 `epoch 229 / 179 / 199`，同时 `val_loss` 继续下降到更后面的 epoch。这说明当前 `flow matching velocity loss` 与最终 spike prediction 指标存在明显目标错位。
- `per-bin fp-bps` 在三个窗口的所有预测 bin 上都显著为负，因此问题不是“长窗口尾部误差积累”这么简单，而是当前 `per-bin summary + shared per-unit head` 结构没有保住必要的 unit-level 预测信息。

### 下一轮优先修正方向

1. 不再沿当前 `per-bin summary` 结构做小幅超参微调；当前结果差距过大，价值不高。
2. 如果继续坚持 `Option 2B`，优先改为 **unit-level tokenization + factorized time-unit attention / axial block**，在保留可训练复杂度的同时恢复 unit 维建模能力。
3. `Option 2A latent diffusion` 继续保留为备选，但只在新的 `Option 2B` 结构仍然不能收口时再转向。

## 2026-03-20 — Factorized Unit-Time Flow Tokens

> 状态：formal 完成，当前变体归档为 diffusion 新基线
> 分支：`dev/diffusion`
> 对应任务记录：`cc_todo/1.11-diffusion-decoder/20260320_factorized_unit_time_flow.md`

### 想法描述

保留 `Option 2B` 的 direct count-space flow matching，不再在进入主干前把 `N` 个 unit 汇总成单个 per-bin token，而是改成显式 `(time bin, unit)` token。为了避免 `T x N` full attention 的计算爆炸，这一轮采用 factorized token mixing：

1. 先把每个 `(time bin, unit)` 的 noisy scalar、unit embedding 和 bin position 组合成 token
2. 对 unit 维做 masked pooling，得到 time tokens，并通过 cross-attention 条件化到 history latents
3. 对每个 unit 独立做 time self-attention
4. 对每个 time bin 独立做 unit attention
5. 用 token-level scalar head 直接回归每个 `(time bin, unit)` 的 velocity

### 相比上一轮的关键改动

- 不再使用 `per-bin summary + shared per-unit head`
- 保留 direct count-space flow matching，不引入 autoencoder
- 计算复杂度从 full `O((TN)^2)` 改成 factorized `time-mix + unit-mix`

### 预期优点

- unit-level 信息会显式进入主干，而不是只作为 pooled summary 的读出条件
- 若问题主要来自 summary bottleneck，这一轮应至少能把所有 bin 都显著为负的状态拉回到更合理的区间
- 训练与评估入口不变，便于直接与 `20260320_direct_count_flow_dit` 做对照

### 主要风险

- unit attention 会把 unit 数带回计算图主干，显存和速度开销高于上一轮
- pooled time-token cross-attention 仍然可能不足以提供足够强的 history conditioning
- 如果 smoke 仍然明显不对，说明问题可能不只在 summary bottleneck，而要进一步考虑 `Option 2A` 或更强的 conditioning 路线

### 当前进展（2026-03-20 ~ 2026-03-21）

- factorized unit-time token 版本的 `DiffusionFlowDecoder` 已完成初版实现
- 新增三窗口配置：
  - `examples/neurohorizon/configs/model/neurohorizon_factorized_unit_time_flow_{250,500,1000}ms.yaml`
  - `examples/neurohorizon/configs/train_1p11_factorized_unit_time_flow_{250,500,1000}ms.yaml`
- 250ms 真实数据 smoke 已跑通：
  - 训练 smoke：`train_loss = 1.140`，`val_loss = 1.146`，`val/fp_bps = -15.280`
  - 离线 valid smoke（1 batch）：`fp-bps = -15.2633`，`R2 = -51.9120`，`val_loss = 1.7312`
- 当前解读：
  - 新结构的训练、checkpoint、best ckpt 解析和离线评估入口都可用，说明第二轮结构替换没有破坏工程链路
  - 由于这仍然只是 `2 train steps + 1 valid batch + 1 offline eval batch` 的 smoke，当前负指标只能说明性能尚未显现，不能据此直接否定结构本身
  - 下一步应先提交实现 checkpoint，再决定是否直接启动 `250 / 500 / 1000ms` formal
- 三窗口 formal 已完成：

  | window | best val fp-bps | best epoch | test fp-bps | vs direct-count diffusion | vs baseline_v2 test | test R2 | test PSTH-R2 |
  |--------|------------------|------------|-------------|----------------------------|----------------------|---------|--------------|
  | 250ms | -3.9775 | 39 | -4.0307 | +3.4643 | -4.2530 | -0.6848 | 0.1879 |
  | 500ms | -4.5144 | 19 | -4.5237 | +3.3364 | -4.6978 | -0.7595 | 0.2020 |
  | 1000ms | -4.8550 | 179 | -4.9099 | +3.3178 | -5.0447 | -0.6573 | 0.4186 |

### 第二轮 formal 结论

- 这轮结果表明：**恢复 unit-level tokenization 是正确方向**。相对 `20260320_direct_count_flow_dit`，三窗口的 continuous test `fp-bps` 都稳定提升了 `+3.3 ~ +3.5`，说明上一轮最大的结构性问题确实来自 `per-bin summary` 对 unit 维信息的过早压缩。
- 但这轮还不能视为可用主线。三窗口的 continuous `fp-bps` 仍然全部在 `-4 ~ -5` 区间，和 `baseline_v2` 仍有 `4.25 ~ 5.04 fp-bps` 的明显差距，因此当前变体只能作为 diffusion 路线中的**新基线**，不能直接推进为正式候选模型。
- `250ms / 500ms` 的最佳 checkpoint 分别提前出现在 `epoch 39 / 19`，而训练本身都跑满了 `300 epochs`。这说明问题不再只是“训练不够久”，而是 conditioning 与 flow objective 之间仍然存在明显错位。
- `trial-aligned PSTH-R2` 已经恢复到正值，尤其 `1000ms` 达到 `0.4186`，这说明模型开始学到较粗粒度的时序/条件结构；但 spike-wise continuous `fp-bps` 仍显著为负，说明当前生成结果更像“平滑趋势拟合”，而不是高保真 spike 预测。

### 下一轮优先修正方向

1. 保留 **unit-level tokenization + factorized time/unit mixing**，不再回退到 `per-bin summary` 路线。
2. 下一轮优先增强 conditioning，而不是继续加训练轮数：
   - 让 `(time, unit)` token 更直接地访问 history latents
   - 减少 pooled time-token cross-attention 带来的信息瓶颈
   - 必要时考虑更强的条件注入方式（更密的 cross-attn / FiLM / conditioner stack）
3. 如果在保留 unit-level token 的前提下仍然无法把 continuous `fp-bps` 拉回合理区间，再考虑把 `Option 2A latent diffusion` 从备选提升为下一主线。

## 2026-03-21 — Dense History-Cross Factorized Flow

> 状态：250ms gate 未通过（停止扩窗）
> 分支：`dev/diffusion`
> 对应任务记录：`cc_todo/1.11-diffusion-decoder/20260321_dense_history_cross_factorized_flow.md`

### 想法描述

保留第二轮已经确认有效的 `(time bin, unit)` 显式 token 与 factorized time/unit mixing，不再让 pooled time token 代表整段未来窗口去访问 history，而是让**每个 token 直接 cross-attend 到 history latents**。本轮的核心判断是：第二轮剩余的大 gap 更可能来自 conditioning 信息瓶颈，而不是 unit-level tokenization 本身。

### 相比上一轮的关键改动

- 不再使用 `pooled time-token cross-attention`
- 每个 `(time, unit)` token 直接对 `encoder_latents` 做 cross-attention
- 保留 per-unit time self-attention、per-time unit attention 和现有 flow matching 目标

### 预期优点

- history 信息不再先经过 pooled summary，再分发回所有 unit token
- 有机会提高 spike-wise continuous `fp-bps`，而不是只改善更平滑的 `PSTH-R2`
- 仍然保持结构改动集中，不把优化器、数据协议、入口脚本同时改掉

### 主要风险

- dense token-wise cross-attention 的显存和时间成本高于第二轮
- 如果结果仍然只能改善 `PSTH-R2` 而拉不动 continuous `fp-bps`，说明剩余问题不只在 conditioning 路径
- 若 `250ms` gate 也明显不过线，则 `Option 2B` 可能接近阶段性上限

### 当前执行策略

1. 先完成第三轮文档、配置和脚本建档
2. 先跑 `250ms` smoke，确认新 cross-conditioning 结构没有破坏链路
3. 只跑 `250ms` formal gate
4. 默认 gate 为：`250ms test fp-bps >= -2.5`
5. 只有 gate 通过，才继续扩到 `500 / 1000ms`

### 当前进展（2026-03-21）

- dense token-wise history cross 版本的 `DiffusionFlowDecoder` 已完成初版实现
- 第三轮配置与脚本已建立：
  - `neurohorizon_dense_history_cross_factorized_flow_{250,500,1000}ms.yaml`
  - `train_1p11_dense_history_cross_factorized_flow_{250,500,1000}ms.yaml`
  - `run_dense_history_cross_factorized_flow_250ms_gate.sh`
- 250ms 真实数据 smoke 已跑通：
  - 训练 smoke：`train_loss = 1.139`，`val_loss = 1.145`，`val/fp_bps = -15.258`
  - 离线 valid smoke（1 batch）：`fp-bps = -15.2412`，`R2 = -51.6662`，`val_loss = 1.7292`
- 当前解读：
  - 第三轮 dense cross-conditioning 没有破坏训练、checkpoint 和离线评估链路
  - smoke 相对第二轮只带来了极小幅变化，当前还不能据此判断 gate 是否有希望
  - 下一步应提交实现 checkpoint，并直接启动 `250ms formal gate`

### 正式 gate 结果（2026-03-22）

- 250ms formal gate 已完成：
  - best `val/fp_bps = -4.5783`，对应 `epoch 239`
  - formal valid `fp-bps = -4.5587`
  - formal test `fp-bps = -4.5658`
  - formal test `R2 = -0.5021`
  - formal trial `fp-bps = -4.4743`
  - formal `PSTH-R2 = 0.4590`
- gate 结论：
  - 默认阈值是 `250ms test fp-bps >= -2.5`
  - 当前 `-4.5658 < -2.5`，因此第三轮 gate 明确未通过，按计划停止扩到 `500 / 1000ms`
- 相对比较：
  - 相比第二轮 `20260320_factorized_unit_time_flow` 的 `250ms test fp-bps = -4.0307`，第三轮反而下降 `-0.5351`
  - 相比 `baseline_v2` 当前 `250ms test`，仍落后 `-4.7881`
- 当前判断：
  - dense token-wise history cross 并没有在当前实现下带来预期中的 spike-wise 提升
  - `Option 2B` 的剩余 gap 不能再简单归因于 pooled conditioning，本轮不建议继续沿这条局部改动方向做默认扩展

## 2026-03-22 — Latent Diffusion with Factorized Time-Unit Latents

> 状态：验证中（`250ms` formal gate 已通过）
> 分支：`dev/diffusion`
> 对应任务记录：`cc_todo/1.11-diffusion-decoder/20260322_latent_diffusion_factorized_latent.md`

### 转向动机

前面三轮 `Option 2B` 已经回答了两个关键问题：

1. 直接在 count-space 上做 rectified flow matching 的工程链路是可行的
2. `per-bin summary` 的确是首轮重大瓶颈，恢复 unit-level tokenization 后有稳定提升

但第三轮 `dense_history_cross_factorized_flow` 又说明：剩余 gap 不能再简单归因于 conditioning 过弱。继续在 count-space 主干上叠局部 attention 改动，已经很难回答更本质的问题。当前更合理的下一步是正式切到 `Option 2A latent diffusion`，把“future count field 的高维建模难度”与“history-conditioned generation”拆开：

- 先把 future `log1p(count)` 压缩到更易建模的 latent 空间
- 再在 latent 空间中做 diffusion / flow matching
- 最后由 latent decoder 重建每个 `(time bin, unit)` 的未来发放率

### 第一版设计选择

- latent 形态：`time x factorized latent units`
- autoencoder：deterministic，不加 KL，不做 VAE
- target 空间：`log1p(count)`
- diffusion objective：继续复用当前分支已经跑通的 rectified flow matching
- 训练方式：autoencoder + latent diffusion 联合训练
- 协议：继续遵循 `1.3.7` 默认协议（10 sessions、continuous、obs=`500ms`）
- 验证策略：`250ms gate-first`

### 为什么这样做

- 相比 full-field global latent，这个 latent 仍保留时间轴，不会把整个 future window 过度压成单个向量
- 相比继续在 raw `(time, unit)` count field 上 diffusion，latent denoising 的任务更平滑、更低维，可能更容易优化
- deterministic AE 是首版最小实现，不引入 KL / posterior collapse 等额外不确定性
- `250ms gate-first` 可以先回答“切到 2A 是否值得”，避免一开始就把 `500 / 1000ms` 训练成本全部压上去

### 当前实现（2026-03-22）

- 已新增 `decoder_variant='latent_diffusion'`
- 已实现 `torch_brain/nn/latent_diffusion_decoder.py`
  - `FactorizedCountAutoencoder`
  - `LatentDiffusionDecoder`
- 已扩展：
  - `torch_brain/models/neurohorizon.py`
  - `examples/neurohorizon/train.py`
  - `scripts/analysis/neurohorizon/eval_phase1_v2.py`
- 已新增三窗口配置：
  - `neurohorizon_latent_diffusion_factorized_latent_{250,500,1000}ms.yaml`
  - `train_1p11_latent_diffusion_factorized_latent_{250,500,1000}ms.yaml`
- 已新增脚本：
  - `verify_latent_diffusion_factorized_latent.py`
  - `run_latent_diffusion_factorized_latent_250ms_gate.sh`

### 当前进展（2026-03-22）

- 最小功能验证已通过：
  - `compute_training_loss()` 可返回总 loss、`ae_recon_loss` 与 `diffusion_latent_loss`
  - `generate()` 可输出 `[B, T, N]` 形状的有限 log-rate
- `250ms` 真实数据 smoke 已跑通：
  - 训练 smoke：
    - `train_loss = 1.402`
    - `val_loss = 1.401`
    - `val/fp_bps = -1.440`
  - 离线 valid smoke（1 batch）：
    - `fp-bps = -1.4368`
    - `R2 = -0.1217`
    - `val_loss = 0.4433`
  - 离线 test smoke（1 batch）：
    - `fp-bps = -1.3657`
    - `R2 = -0.1072`
    - `val_loss = 0.4403`
- `250ms` formal gate 已完成：
  - best checkpoint：`epoch 9`
  - best `val/fp_bps = -0.02494`
  - formal valid：
    - `fp-bps = -0.0278`
    - `R2 = 0.1754`
    - `trial fp-bps = -0.0230`
    - `PSTH-R2 = 0.3719`
  - formal test：
    - `fp-bps = -0.0293`
    - `R2 = 0.1756`
    - `trial fp-bps = -0.0261`
    - `PSTH-R2 = 0.4252`

### 当前解读

- 第一版 2A 不只是“值得做 formal gate”，而是已经在 `250ms` 上给出了明确正结果：`test fp-bps = -0.0293`，远高于 gate 阈值 `-2.5`
- 相比第二轮 `factorized_unit_time_flow` 的 `250ms test fp-bps = -4.0307`，本轮提升 `+4.0014`
- 相比第三轮 `dense_history_cross_factorized_flow` 的 `250ms test fp-bps = -4.5658`，本轮提升 `+4.5365`
- 相比 `baseline_v2` 当前 `250ms test` 只落后约 `-0.2516`，说明 latent-space generation 已经把 1.11 主线从“结构性失败区间”拉回到“接近 baseline”的量级
- 同时，best checkpoint 很早出现在 `epoch 9`，而训练到 `epoch 299` 时在线 `val/fp_bps` 已退化到约 `-0.2507`；这提示当前 2A 主线存在明显的早熟收敛 / 后期过训练问题，后续扩窗时需要重点观察

### 主要风险

- joint training 可能让 autoencoder reconstruction 和 diffusion latent denoising 相互牵制
- `time x factorized latent units` 仍可能不足以保住精细 spike-wise information
- 即使 250ms 已经通过，`500 / 1000ms` 仍可能暴露更强的 latent bottleneck 或长窗口退化

### 下一步

1. 将 `250ms` formal 结果正式写回 1.11 文档、`results.tsv` 和 `results.md`
2. 继续扩到 `500 / 1000ms` formal
3. 扩窗时优先检查 best checkpoint 是否仍然极早出现，以及 continuous `fp-bps` 是否继续接近或超越 baseline
