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

- **方案**：`Option 2B`
- **任务形式**：direct count-space flow matching
- **去噪网络**：DiT 风格时间主干
- **默认实施分支**：`dev/diffusion`

### 暂缓方案

- `Option 2A latent diffusion`
  - 保留为备选
  - 不进入本轮第一次任务
  - 只有在 2B 无法稳定训练、效果明显不佳、或 count-space 方案暴露出结构性瓶颈时才切换评估

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

> 状态：首轮 formal 完成，当前变体建议放弃
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
