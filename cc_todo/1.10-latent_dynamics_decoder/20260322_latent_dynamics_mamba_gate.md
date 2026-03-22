# Phase 1.10 模块优化：Latent Dynamics Mamba Gate

**日期**：2026-03-22
**模块名**：`latent_dynamics_mamba_gate`
**状态**：进行中
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
- [x] 删除 `transformers + mambapy` fallback，锁定官方 `mamba-ssm`
- [x] 新增官方 wheel 安装脚本
- [x] 新增官方源码编译安装脚本
- [x] 完成 `mamba-ssm` / `causal-conv1d` 官方 backend 安装
- [x] 完成本轮 verify
- [x] 跑通 `500ms` smoke
- [x] 启动 `500ms` formal gate run

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

# 官方 backend 安装（当前环境实测成功路径）
bash scripts/1.10-latent_dynamics_decoder/20260322_latent_dynamics_mamba_gate/build_official_mamba_from_source.sh

# 官方 wheel 安装（适合作为备用路径）
bash scripts/1.10-latent_dynamics_decoder/20260322_latent_dynamics_mamba_gate/install_official_mamba_wheels.sh

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
- `mamba` 路径已锁定为官方 backend：
  - 仅支持 `mamba-ssm`
  - 缺失依赖时直接报错，不再回退到其他实现
- 新增配置与脚本：
  - `examples/neurohorizon/configs/model/neurohorizon_latent_dynamics_mamba_500ms.yaml`
  - `examples/neurohorizon/configs/train_1p10_latent_dynamics_mamba_500ms.yaml`
  - `scripts/1.10-latent_dynamics_decoder/20260322_latent_dynamics_mamba_gate/`
- 当前环境中的官方 backend 已完成安装：
  - `causal-conv1d==1.6.1`
  - `mamba-ssm==2.3.1`
  - 安装路径：`build_official_mamba_from_source.sh`
  - 编译策略：在当前 `torch 2.10.0+cu128` 环境中，以 `sm_89` 定向源码编译
- verify 已通过：
  - `mamba_backend=mamba_ssm`
  - `output_shape=(2, 25, 6)`
  - `tf_vs_rollout_max_delta=0.000000`
- `500ms` smoke 已在官方 backend 下跑通：
  - 配置：`batch_size=64`, `eval_batch_size=64`, `epochs=1`, `eval_epochs=1`, `num_workers=0`
  - train loss：`0.419`
  - val loss：`0.396`
  - train-end `val/fp_bps=-0.829`
  - 离线 continuous valid：`fp-bps=-0.8290`, `R2=-0.0001`, `val_loss=0.3958`
- `500ms` formal gate 已启动：
  - 启动时间：`2026-03-22 17:32 CST`
  - 进程：`bash scripts/1.10-latent_dynamics_decoder/20260322_latent_dynamics_mamba_gate/run_latent_dynamics_mamba_500ms.sh`
  - 日志：`results/logs/1.10-latent_dynamics_decoder/20260322_latent_dynamics_mamba_gate/500ms/formal_run.log`

### 依赖安装状态

- `pip install` 的 build isolation 会重复拉取整套 `torch/cu12` 依赖，直接路径不稳定
- 最终采用源码编译闭环：
  - `causal-conv1d==1.6.1`
  - `mamba-ssm==2.3.1`
  - 显式设置 `CUDA_HOME=/usr/local/cuda-12.4`
  - 将编译目标裁到 `sm_89`，避免为当前机器无关架构耗时
- 验证结果：
  - `from mamba_ssm import Mamba` 可导入
  - 最小前向：`mamba_forward_shape=(2, 25, 128)`
  - 仓库 verify 已在官方 backend 下通过

## 当前判断

- 代码与环境两侧的官方 backend 已经闭环：仓库不再依赖 fallback
- 官方 backend 下的 verify 与 `500ms` smoke 都已打通，且 smoke 恢复到与正式 train config 对齐的 `batch_size=64`
- `500ms` smoke 数值仍停留在负区间，说明仅靠 backbone 切换并没有立刻改善短训表现
- 因此本轮的关键下一步不是继续补环境，而是等待已经启动的正式 `500ms` gate 给出完整 valid/test 判断
