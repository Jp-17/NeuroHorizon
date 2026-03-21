# Phase 1.10 模块优化：Latent Dynamics State Scaling

**日期**：2026-03-20
**模块名**：`latent_dynamics_state_scaling`
**状态**：已放弃
**分支**：`dev/latent`

## 改进摘要

本轮在 `20260320_latent_dynamics_decoder` 的首轮正式结果基础上继续推进 `1.10`。目标不是再换 dynamics 类型，而是先把 latent dynamics 的真实状态容量做大，验证上一轮的主要瓶颈是否来自过强的 state compression。

## 前因后果

- 上一轮首个 GRU latent dynamics baseline 已经完整打通训练、formal eval、结果汇总和文档记录
- 但正式结果显示：
  - `250ms` 已接近 `baseline_v2`
  - `500ms / 1000ms` 明显落后
  - best epoch 已出现在 `259–289`，说明不是 epoch 不够
- 复查实现后确认，上一轮 decoder 虽然支持 `num_pool_tokens=4`，但所有 pooled tokens 最终仍被压回 `128` 维 hidden state

## 本轮设计结论

1. 优先解耦 `pool_token_dim` 与 `state_dim`
2. 暂不更换 GRU backbone，先判断“更大 latent state”本身是否有价值
3. 先做 `500ms` gate，不立即扩到 `250ms / 1000ms`
4. 训练入口和离线正式评估入口保持不变：
   - 训练入口：`examples/neurohorizon/train.py`
   - 离线正式评估入口：`scripts/analysis/neurohorizon/eval_phase1_v2.py`

## 代码实施清单

- [x] 修改 `LatentDynamicsDecoder`，增加 `pool_token_dim / state_dim`
- [x] 修改 `NeuroHorizon` 配置接口
- [x] 新增 `500ms gate` 的 model/train config
- [x] 新增本轮 verify / run 脚本
- [x] 完成功能验证
- [x] 启动 `500ms` formal gate run

## 本轮目标配置

- `num_pool_tokens = 8`
- `pool_token_dim = 64`
- `state_dim = 256`
- `pred_window = 500ms`
- 其他训练协议保持与 `1.10.0` 默认值一致

## 目录约定

- 脚本：`scripts/1.10-latent_dynamics_decoder/20260320_latent_dynamics_state_scaling/`
- 日志：`results/logs/1.10-latent_dynamics_decoder/20260320_latent_dynamics_state_scaling/`
- 可视化：`results/figures/1.10-latent_dynamics_decoder/20260320_latent_dynamics_state_scaling/`

## 运行命令

```bash
conda activate poyo
cd /root/autodl-tmp/NeuroHorizon
python scripts/1.10-latent_dynamics_decoder/20260320_latent_dynamics_state_scaling/verify_latent_dynamics_state_scaling.py
bash scripts/1.10-latent_dynamics_decoder/20260320_latent_dynamics_state_scaling/run_latent_dynamics_state_scaling_500ms.sh
```

## 当前进展

### 功能验证

- `verify_latent_dynamics_state_scaling.py` 已通过
- 输出：
  - `output_shape=(2, 25, 6)`
  - `tf_vs_rollout_max_delta=0.000000`

### 500ms smoke run

- 配置：`train_1p10_latent_dynamics_state_scaling_500ms.yaml`
- override：`epochs=1 eval_epochs=1 batch_size=256 eval_batch_size=256 num_workers=0`
- 训练日志：
  - `results/logs/1.10-latent_dynamics_decoder/20260320_latent_dynamics_state_scaling/500ms_smoke/`
- 关键结果：
  - train loss：`0.412`
  - val loss：`0.393`
  - train-end `val/fp_bps=-0.840`
  - 离线 continuous valid：`fp-bps=-0.8411`, `R2=-0.0024`, `val_loss=0.3967`

### 500ms formal gate

- 启动时间：`2026-03-20 14:57 CST`
- 后台会话：`screen -S latent_dyn_state_500`
- 日志：
  - `results/logs/1.10-latent_dynamics_decoder/20260320_latent_dynamics_state_scaling/screen_run.log`
- 当前状态：
  - 训练与 best-ckpt formal `valid/test` 已完成
  - best checkpoint 出现在 `epoch 69`
  - final `epoch 299` 的 `val/fp_bps` 回落到 `0.0006`
  - `screen` 会话已正常结束

### 500ms formal 结果

- valid：
  - `fp-bps=0.0048`
  - `R2=0.1791`
  - `val_loss=0.3250`
- test：
  - `fp-bps=0.0049`
  - `R2=0.1790`
  - `val_loss=0.3235`
- 对比：
  - 相对 `baseline_v2=0.1744` 差 `-0.1696`
  - 相对上一轮 `20260320_latent_dynamics_decoder` 的 `500ms valid fp-bps=0.0904` 差 `-0.0856`

## 当前判断

- 这一轮的价值在于验证“更大 latent state”是否能明显改善 `500ms`
- 当前功能链路已打通，新的 `state_dim / pool_token_dim` 接口没有破坏训练或推理路径
- 结果表明：更大 latent state 不但没有改善 `500ms`，反而显著退化
- 因此该模块应标记为“已放弃”，不再扩展到 `250ms / 1000ms`
- 下一轮若继续推进 `1.10.x`，优先方向应改为更强的 dynamics backbone 或显式 context skip，而不是继续单纯放大 GRU state
