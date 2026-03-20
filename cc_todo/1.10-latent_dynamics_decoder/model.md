# Phase 1.10 Latent Dynamics Decoder 设计记录

> 本文档独立记录 `1.10 latent dynamics decoder` 方向的架构演进与审查结论，不再继续写入旧的 `cc_core_files/model.md`。
>
> **方向切换依据**：
> - `cc_todo/20260316-review/ar_effectiveness_claude.md`
> - `cc_todo/20260316-review/long_horizon_prediction_claude.md`
> - `cc_todo/20260316-review/option_d_implementation_claude.md`（方案一）
>
> **当前主判断**：
> - observation-space AR feedback 在当前任务上的边际收益持续低于 `baseline_v2`
> - 长时程预测更值得尝试的方向是：保留 POYO+ history encoder，把“未来序列建模”从 high-dimensional count space 转移到 latent space

---

## 方向总原则

1. **尽量复用已有有效模块**：
   - 保留 POYO+ history encoder
   - 保留 tokenization / dataset / sampler / metrics / evaluation 协议
   - 保留 `PerNeuronMLPHead` 作为最终的 spike-count readout
2. **替换当前主线 decoder**：
   - 不再继续把 `prediction_memory / local_prediction_memory` 作为主线方向
   - `1.10` 默认探索 latent dynamics decoder
3. **首轮实现优先低依赖与可跑通**：
   - 当前 `poyo` 环境没有 `s4`、`mamba_ssm`、`torchdiffeq`
   - 因此首轮不引入新 dynamics 依赖，优先落地可直接运行的 GRU latent dynamics 主线
4. **Mamba 保留为后续扩展位**：
   - 首轮代码接口要为后续 `1.10.x` 的 Mamba 变体预留扩展点
   - 但本轮不安装依赖、不实现第二变体

---

## 当前仓库下的复用与替换边界

### 继续复用

- `NeuroHorizon` 的 history encoder 及其 latent token 构造方式
- `examples/neurohorizon/train.py`
- `scripts/analysis/neurohorizon/eval_phase1_v2.py`
- `RandomFixedWindowSampler` / `SequentialFixedWindowSampler`
- `fp-bps` / `per-bin fp-bps` 指标口径
- `PerNeuronMLPHead`

### 本方向替换

- 当前 observation-space decoder 主线
- 当前 `1.9` 的 runtime 级 prediction-memory 代码路径
- 当前 `1.9` 的模块优化汇总路径与结果表

### 结果记录迁移

- 设计记录：本文件
- 每轮任务记录：`cc_todo/1.10-latent_dynamics_decoder/{date}_{module_name}.md`
- 结果追踪：`cc_todo/1.10-latent_dynamics_decoder/results.tsv`
- 运行脚本：`scripts/1.10-latent_dynamics_decoder/{date}_{module_name}/`
- 日志与图表：`results/logs/1.10-latent_dynamics_decoder/...` 与 `results/figures/1.10-latent_dynamics_decoder/...`

---

## 2026-03-20 — GRU Latent Dynamics Decoder

> 状态：验证完成
> 分支：`dev/latent`
> 任务记录：`cc_todo/1.10-latent_dynamics_decoder/20260320_latent_dynamics_decoder.md`

### 前因后果

在 `2026-03-12` 到 `2026-03-13` 的四轮 `1.9` AR feedback 实验后，可以确认：

- prediction memory 方向在 teacher-forced 下虽能得到更高指标，但 rollout 一直落后于 `baseline_v2`
- alignment 与 tuning 只能把差距缩小，未能从根本上改变“显式 feedback 信息增益不足”的结论
- 当前任务更值得测试的是：history encoder 是否已经提取出足够的 dynamics-relevant representation，从而允许我们在 latent space 做未来外推

### 本轮目标

实现一个可运行的 latent dynamics baseline，回答三个核心问题：

1. 在不引入 observation-space feedback 的情况下，latent rollout 能否跑通完整训练/评估链路？
2. 在相同 `1.3.7` 协议下，latent dynamics 是否能在 `500ms / 1000ms` 窗口上优于 `baseline_v2`？
3. 当前 POYO+ encoder 输出是否已经足够支持 latent-space forward prediction？

### 首轮实现方案

**总体思路**：

```
history spikes
  -> POYO+ history encoder
  -> encoder latents
  -> attention pooling
  -> latent initial state
  -> GRU latent dynamics rollout
  -> per-step bin representation
  -> PerNeuronMLPHead
  -> future spike counts
```

**关键实现点**：

- 用 learned pooling queries 从 encoder latents 中抽取少量 pooled tokens
- 把 pooled tokens 压缩为固定维度的 latent initial state
- 用 autonomous GRU 在 prediction bins 上做 rollout
- `forward()` 和 `generate()` 共用同一 latent rollout 逻辑，不再依赖 `target_counts`
- 保留 `query_aug + feedback_method=none` 作为 baseline_v2 兼容路径

### 为什么首轮不用 Mamba / S4D

- 当前环境没有对应依赖
- 首轮更需要先验证“latent dynamics 路线本身”是否成立，而不是先把变量数做大
- 如果 GRU 主线能够在当前协议下取得正收益，再继续做 Mamba 才有清晰的增量解释空间

### 首轮功能验证方案

- `decoder_variant=latent_dynamics` 可正常实例化
- 训练脚本与评估脚本无需新增入口即可跑通
- `forward()` 与 `generate()` 在相同输入下数值一致
- 250ms smoke run 能正常训练、保存 checkpoint、跑离线评估

### 首轮正式实验协议

- 数据：`perich_miller_10sessions`
- 采样：continuous
- obs_window：500ms
- pred_window：250ms / 500ms / 1000ms
- 指标：`fp-bps` / `per-bin fp-bps`
- 补充指标：`val_loss`、`R-squared`（按需要）

### 本轮的成功标准

- 代码与协议层面：
  - 完整走通训练 / checkpoint / eval / 汇总 / 图表 / 文档记录链路
- 结果层面：
  - 至少得到可比较的三窗口正式结果
  - 尤其关注 `500ms / 1000ms` 是否相对 `baseline_v2` 出现明确收益

### 当前实现进展

- `latent_dynamics_decoder.py` 已落地，并接入 `NeuroHorizon`
- `decoder_variant=latent_dynamics` 已能通过功能验证：
  - `output_shape=(2, 12, 6)`
  - `tf_vs_rollout_max_delta=0.000000`
- 250ms smoke run 已通过训练、checkpoint 与离线 continuous valid eval 链路：
  - train loss：`0.417`
  - val loss：`0.406`
  - eval valid fp-bps：`-0.8339`
  - eval valid R2：`-0.0021`

### 首轮正式结果

- `250ms`
  - valid `fp-bps=0.1882`, `R2=0.2493`, 相对 `baseline_v2=0.2115` 差 `-0.0233`
  - test `fp-bps=0.1966`, `R2=0.2511`, 相对 `baseline_v2` 差 `-0.0149`
  - best checkpoint 出现在 `epoch 289`
- `500ms`
  - valid `fp-bps=0.0904`, `R2=0.2041`, 相对 `baseline_v2=0.1744` 差 `-0.0840`
  - test `fp-bps=0.0857`, `R2=0.2033`, 相对 `baseline_v2` 差 `-0.0887`
  - best checkpoint 出现在 `epoch 259`
- `1000ms`
  - valid `fp-bps=0.0674`, `R2=0.1946`, 相对 `baseline_v2=0.1317` 差 `-0.0643`
  - test `fp-bps=0.0667`, `R2=0.1937`, 相对 `baseline_v2` 差 `-0.0650`
  - best checkpoint 出现在 `epoch 289`

### 当前结论

- 当前 GRU latent dynamics baseline 已证明这条路线在现有仓库里可完整落地：训练、best checkpoint、离线 `valid/test`、结果汇总和图表都能稳定产出
- 但它没有达到本轮“重点观察 `500ms / 1000ms` 是否优于 `baseline_v2`”的目标
- `250ms` 已经接近 `baseline_v2`，说明 latent rollout 不是完全不可用
- `500ms / 1000ms` 明显落后，说明仅靠当前的 attention pooling + 紧凑 latent state + autonomous GRU dynamics，还不足以支持更长预测窗
- 由于本实现里 `teacher-forced == rollout`，当前瓶颈更像是 latent state 容量 / pooling 压缩 / dynamics expressiveness 不足，而不是 exposure bias

### 后续扩展位

- `1.10.x` 下一轮优先考虑：
  - 提高 latent pooling 数量与 state dim，减少过强压缩
  - 保持当前接口，尝试更强的 dynamics backbone（优先 Mamba）
  - 增加 context skip / recurrent conditioning，而不是纯 autonomous rollout
  - encoder frozen vs end-to-end 微调对比

---

## 2026-03-20 — Latent Dynamics State Scaling (500ms Gate)

> 状态：实施中
> 分支：`dev/latent`
> 任务记录：`cc_todo/1.10-latent_dynamics_decoder/20260320_latent_dynamics_state_scaling.md`

### 动机

- `20260320_latent_dynamics_decoder` 的正式结果说明：问题不在 epoch 不够，而在当前 latent dynamics 表达能力不足
- 复查实现后可以确认，首轮 decoder 会把任意数量的 pooled tokens 再压回 `dim=128`
- 这意味着仅增加 `num_pool_tokens` 并不会真正增加 latent state 容量，因此下一轮必须先把 `state_dim` 和 `pool_token_dim` 显式做成可调参数

### 本轮方案

1. 在 `LatentDynamicsDecoder` 中引入两个新参数：
   - `pool_token_dim`：控制每个 pooled token 在进入 dynamics 前保留多少信息
   - `state_dim`：控制 GRU dynamics 的真实 hidden size
2. 保持训练入口、评估入口、数据协议不变
3. 先只做 `500ms` gate，避免在 `250ms` 已较接近 baseline 的情况下继续重复低价值长跑

### 初始配置

- `num_pool_tokens = 8`
- `pool_token_dim = 64`
- `state_dim = 256`
- 其他训练协议保持 `1.10.0` 默认值

### 本轮成功标准

- 代码层面：
  - 新 decoder 参数可正常实例化、训练、保存、加载
  - `forward()` 与 `generate()` 仍保持一致
- 实验层面：
  - 跑通 `500ms` gate 的训练和 best-ckpt formal `valid/test`
  - 至少判断“更大 latent state”是否能明显抬高上一轮 `500ms valid fp-bps = 0.0904`

### 当前执行进展

- 新接口已落地：
  - `latent_dynamics_pool_token_dim`
  - `latent_dynamics_state_dim`
- 新的 `500ms` gate 配置为：
  - `num_pool_tokens=8`
  - `pool_token_dim=64`
  - `state_dim=256`
- 功能验证已通过：
  - `output_shape=(2, 25, 6)`
  - `tf_vs_rollout_max_delta=0.000000`
- `500ms` smoke run 已通过训练和离线 continuous valid eval：
  - train loss：`0.412`
  - val loss：`0.393`
  - valid `fp-bps=-0.8411`
- `500ms` formal gate 已于 `2026-03-20 14:57 CST` 在 `screen` 会话 `latent_dyn_state_500` 中启动
