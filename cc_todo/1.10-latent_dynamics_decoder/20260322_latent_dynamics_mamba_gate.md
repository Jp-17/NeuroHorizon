# Phase 1.10 模块优化：Latent Dynamics Mamba Gate

**日期**：2026-03-22
**模块名**：`latent_dynamics_mamba_gate`
**状态**：验证中
**分支**：`dev/latent`

## 改进摘要

本轮不再继续围绕 GRU latent dynamics 做局部结构修补，而是直接把 rollout backbone 切到 `Mamba`，验证更强的 selective SSM 是否能把 `500ms` 从最近两轮的失败区间拉回去。

## 前因后果

- `20260320_latent_dynamics_decoder` 证明了 latent dynamics 路线在 NeuroHorizon 中可完整落地，但 `500ms / 1000ms` 仍明显落后于 `baseline_v2`
- `20260320_latent_dynamics_state_scaling` 说明单纯扩大 latent state 容量无效，`500ms valid fp-bps` 退化到 `0.0048`
- `20260321_latent_dynamics_context_skip` 说明显式 persistent context 注入同样无效，`500ms valid fp-bps` 仅 `0.0047`
- 因此下一轮不再继续围绕 GRU 变体细调，直接切换到更强的 dynamics backbone（优先 `Mamba`）

## 本轮设计结论

1. 保留 POYO+ history encoder、tokenize、`PerNeuronMLPHead`、训练入口和离线正式评估入口
2. 将 `LatentDynamicsDecoder` 从固定 GRU 实现升级为可选 backbone：
   - `latent_dynamics_backbone = gru | mamba`
   - `latent_dynamics_backbone_cfg`
   - `latent_dynamics_input_mode = prev_latent`
3. 本轮 `Mamba` 路线保持最少变量：
   - 不叠加 context skip
   - 不引入 observation-space count feedback
   - 仅验证 latent rollout backbone 本身的收益
4. 先只做 `500ms gate`
   - 若 `valid fp-bps < 0.09`，本轮直接标记为失败，不扩展三窗口
   - 若 `valid fp-bps > 0.12`，再扩展 `250ms / 500ms / 1000ms`

## 代码实施清单

- [x] 将 `LatentDynamicsDecoder` 升级为 `gru / mamba` 双 backbone
- [x] 在 `NeuroHorizon` 增加 `latent_dynamics_backbone / latent_dynamics_backbone_cfg / latent_dynamics_input_mode`
- [x] 新增 `500ms gate` 的 model/train config
- [x] 新增本轮 verify / smoke / formal / collect 脚本
- [x] 在 `pyproject.toml` 中添加 `mamba` optional dependency
- [x] 安装 `mambapy` fallback 并完成最小导入验证
- [ ] 完成 `mamba-ssm` / `causal-conv1d` kernel backend 安装
- [x] 完成本轮 verify
- [x] 跑通 `500ms` smoke
- [ ] 启动 `500ms` formal gate run

## 本轮目标配置

- `pred_window = 500ms`
- `latent_dynamics_backbone = mamba`
- `latent_dynamics_input_mode = prev_latent`
- `num_pool_tokens = 4`
- `state_dim = 128`
- `d_model = 128`
- `d_state = 64`
- `d_conv = 4`
- `expand = 2`
- `output_residual = True`

## 目录约定

- 脚本：`scripts/1.10-latent_dynamics_decoder/20260322_latent_dynamics_mamba_gate/`
- 日志：`results/logs/1.10-latent_dynamics_decoder/20260322_latent_dynamics_mamba_gate/`
- 可视化：`results/figures/1.10-latent_dynamics_decoder/20260322_latent_dynamics_mamba_gate/`

## 运行命令

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate poyo
cd /root/autodl-tmp/NeuroHorizon

# 依赖安装（可运行 fallback）
pip install mambapy

# 依赖安装（更快 CUDA kernels，仍在排查）
TMPDIR=/root/autodl-tmp/tmp_pip MAX_JOBS=4 pip install --no-build-isolation ninja 'causal-conv1d>=1.4.0' 'mamba-ssm>=2.2,<2.3'

# 功能验证
python scripts/1.10-latent_dynamics_decoder/20260322_latent_dynamics_mamba_gate/verify_latent_dynamics_mamba_gate.py

# smoke
bash scripts/1.10-latent_dynamics_decoder/20260322_latent_dynamics_mamba_gate/run_latent_dynamics_mamba_smoke.sh

# formal gate
bash scripts/1.10-latent_dynamics_decoder/20260322_latent_dynamics_mamba_gate/run_latent_dynamics_mamba_500ms.sh
```

## 当前进展

### 已完成

- 代码接口已落地：
  - `LatentDynamicsDecoder(backbone='gru' | 'mamba')`
  - `latent_dynamics_backbone_cfg`
  - `latent_dynamics_input_mode='prev_latent'`
- `gru` 路径构造已确认未被新接口破坏
- `mamba` 路径已支持两级后端：
  - 优先 `mamba-ssm`
  - 缺失时回退到 `transformers + mambapy`
- 新增配置与脚本：
  - `examples/neurohorizon/configs/model/neurohorizon_latent_dynamics_mamba_500ms.yaml`
  - `examples/neurohorizon/configs/train_1p10_latent_dynamics_mamba_500ms.yaml`
  - `scripts/1.10-latent_dynamics_decoder/20260322_latent_dynamics_mamba_gate/`
- verify 已通过：
  - `mamba_backend=transformers_mambapy`
  - `output_shape=(2, 25, 6)`
  - `tf_vs_rollout_max_delta=0.000000`
- `500ms` smoke 已在 fallback backend 下跑通：
  - 配置：`batch_size=16`, `eval_batch_size=16`, `epochs=1`, `eval_epochs=1`, `num_workers=0`
  - train loss：`0.415`
  - val loss：`0.395`
  - train-end `val/fp_bps=-0.829`
  - 离线 continuous valid：`fp-bps=-0.8285`, `R2=0.0001`, `val_loss=0.3956`

### 依赖安装状态

- 首次尝试 `pip install 'mamba-ssm>=2.2,<2.3'`
  - 失败原因：build isolation 重新下载整套 `torch/cu12` 依赖，触发磁盘不足
- 第二次尝试 `--no-build-isolation`
  - 已绕过磁盘问题，但编译耗时很长
  - `mamba-ssm` 的源码构建当前仍未成功产出可用 kernel backend
- 临时 fallback：
  - 已安装 `mambapy`
  - `transformers` 内置 `MambaBlock(use_mambapy=True)` 可正常完成 verify 和 smoke
- 附带问题：
  - fallback backend 在 `batch_size=128` 时 OOM
  - 需要将 smoke batch size 下调到 `16`

## 当前判断

- 代码层的最小改造已经完成，verify 和最小 smoke 也已经打通
- 当前 blocker 已从“能不能跑”变成“能不能拿到足够快的 kernel backend”
- 在 `transformers + mambapy` fallback 下，`500ms` smoke 可跑，但吞吐和显存效率都不足以直接进入 formal gate
- 这意味着正式 `500ms` gate 仍应等待 `mamba-ssm/causal-conv1d` 或其他更高效 backend 到位后再启动
