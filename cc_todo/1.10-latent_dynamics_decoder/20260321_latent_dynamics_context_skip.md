# Phase 1.10 模块优化：Latent Dynamics Context Skip

**日期**：2026-03-21
**模块名**：`latent_dynamics_context_skip`
**状态**：已放弃
**分支**：`dev/latent`

## 改进摘要

本轮在 `20260320_latent_dynamics_state_scaling` 的负结果基础上继续推进 `1.10`。重点不再是放大 GRU hidden size，而是给 latent rollout 增加一条显式的 persistent context 路径，验证当前失败点是否主要来自“纯 autonomous latent rollout 缺少条件信息”。

## 前因后果

- `20260320_latent_dynamics_decoder` 证明了 latent dynamics 路线在 NeuroHorizon 中可完整落地，但 `500ms / 1000ms` 明显落后于 `baseline_v2`
- `20260320_latent_dynamics_state_scaling` 进一步说明：单纯增大 latent state 不但没有改善 `500ms`，反而把 formal valid `fp-bps` 降到 `0.0048`
- 因此当前更值得优先验证的是“持续条件注入”，而不是继续在 GRU 上堆 state size

## 本轮设计结论

1. 保留 GRU backbone，暂不引入 Mamba 或其他新依赖
2. 保持 `init_state` 路径不变，同时显式构建 `context_vector`
3. 将 `context_vector` 注入每一步的 GRU 输入，并在 rollout 输出侧加 residual
4. 训练入口和离线正式评估入口保持不变：
   - 训练入口：`examples/neurohorizon/train.py`
   - 离线正式评估入口：`scripts/analysis/neurohorizon/eval_phase1_v2.py`
5. 先只做 `500ms gate`，避免在方向尚未确认前扩展到 `250ms / 1000ms`

## 代码实施清单

- [x] 修改 `LatentDynamicsDecoder`，增加可选 `context_conditioning / context_dim`
- [x] 修改 `NeuroHorizon` 配置接口
- [x] 新增 `500ms gate` 的 model/train config
- [x] 新增本轮 verify / run / collect 脚本
- [x] 完成功能验证
- [x] 跑通 `500ms` smoke
- [x] 启动 `500ms` formal gate run

## 本轮目标配置

- `num_pool_tokens = 4`
- `pool_token_dim = 32`
- `state_dim = 128`
- `context_conditioning = True`
- `context_dim = 128`
- `pred_window = 500ms`
- 其他训练协议保持与 `1.10.0` 默认值一致

## 目录约定

- 脚本：`scripts/1.10-latent_dynamics_decoder/20260321_latent_dynamics_context_skip/`
- 日志：`results/logs/1.10-latent_dynamics_decoder/20260321_latent_dynamics_context_skip/`
- 可视化：`results/figures/1.10-latent_dynamics_decoder/20260321_latent_dynamics_context_skip/`

## 运行命令

```bash
conda activate poyo
cd /root/autodl-tmp/NeuroHorizon
python scripts/1.10-latent_dynamics_decoder/20260321_latent_dynamics_context_skip/verify_latent_dynamics_context_skip.py
bash scripts/1.10-latent_dynamics_decoder/20260321_latent_dynamics_context_skip/run_latent_dynamics_context_skip_500ms.sh
```

## 当前进展

### 功能验证

- `verify_latent_dynamics_context_skip.py` 已通过
- 输出：
  - `output_shape=(2, 25, 6)`
  - `tf_vs_rollout_max_delta=0.000000`

### 500ms smoke run

- 配置：`train_1p10_latent_dynamics_context_skip_500ms.yaml`
- override：`epochs=1 eval_epochs=1 batch_size=256 eval_batch_size=256 num_workers=0`
- 训练日志：
  - `results/logs/1.10-latent_dynamics_decoder/20260321_latent_dynamics_context_skip/500ms_smoke/`
- 关键结果：
  - train loss：`0.412`
  - val loss：`0.392`
  - train-end `val/fp_bps=-0.830`
  - 离线 continuous valid：`fp-bps=-0.8301`, `R2=-0.0002`, `val_loss=0.3958`

### 500ms formal gate

- 启动时间：`2026-03-21 21:25 CST`
- 后台会话：`screen -S latent_dyn_ctx_500`
- 日志：
  - `results/logs/1.10-latent_dynamics_decoder/20260321_latent_dynamics_context_skip/screen_run.log`
- 当前状态：
  - 训练与 best-ckpt formal `valid/test` 已完成
  - best checkpoint 出现在 `epoch 69`
  - final `epoch 299` 的 `val/fp_bps` 回落到 `0.0009`
  - `screen` 会话已正常结束

### 500ms formal 结果

- valid：
  - `fp-bps=0.0047`
  - `R2=0.1791`
  - `val_loss=0.3250`
- test：
  - `fp-bps=0.0048`
  - `R2=0.1790`
  - `val_loss=0.3235`
- 对比：
  - 相对 `baseline_v2=0.1744` 差 `-0.1697`
  - 相对上一轮 `20260320_latent_dynamics_state_scaling` 的 `500ms valid fp-bps=0.0048` 差 `-0.0001`
  - 相对首轮 `20260320_latent_dynamics_decoder` 的 `500ms valid fp-bps=0.0904` 差 `-0.0857`

## 当前判断

- 这一轮的价值不在于“把状态做得更大”，而在于分离 `init_state` 和 persistent `context`
- 当前 smoke 数值与前两轮 `1 epoch` smoke 处于同一量级，说明新路径至少没有立刻破坏训练链路
- 正式结果表明：显式 persistent context 注入并没有解决 `500ms` 问题
- 它几乎复现了上一轮 `state scaling` 的负结果，没有把指标从失败区间拉回到首轮 `0.0904`
- 因此该模块应标记为“已放弃”，不再扩展到 `250ms / 1000ms`
- 下一轮若继续推进 `1.10.x`，应直接转向更强的 dynamics backbone，而不是继续在当前 GRU 主线细调
