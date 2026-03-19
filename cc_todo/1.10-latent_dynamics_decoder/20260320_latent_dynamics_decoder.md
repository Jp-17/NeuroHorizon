# Phase 1.10 模块优化：GRU Latent Dynamics Decoder

**日期**：2026-03-20
**模块名**：`latent_dynamics_decoder`
**状态**：实施中
**分支**：`dev/latent`

## 改进摘要

本次 `1.10` 主线从 observation-space autoregressive decoder 转向 latent dynamics decoder。保持 POYO+ history encoder、tokenization、训练/评估协议和 `PerNeuronMLPHead` 不变，将“未来时间窗预测”改为“在 latent space 中 rollout，再映射回 spike counts”。

对应设计说明写入：
- `cc_todo/1.10-latent_dynamics_decoder/model.md` → `2026-03-20 — GRU Latent Dynamics Decoder`

## 前因后果

转向依据来自三份审查结论：

1. `cc_todo/20260316-review/ar_effectiveness_claude.md`
   - observation-space AR feedback 的净收益持续低于 `baseline_v2`
2. `cc_todo/20260316-review/long_horizon_prediction_claude.md`
   - latent dynamics 是当前最值得优先尝试的新方向
3. `cc_todo/20260316-review/option_d_implementation_claude.md`
   - 方案一给出了“保留 POYO+ encoder + latent dynamics rollout”的明确实施路线

## 本轮设计结论

1. 首轮不继续在主线上保留 `prediction_memory / local_prediction_memory` 的运行时分支。
2. 首轮不引入 `mamba_ssm` / `s4` / `torchdiffeq` 等新依赖。
3. 首轮仅实现一个可运行的 GRU latent dynamics 主线版本。
4. 代码接口要为后续 Mamba 变体预留扩展位，但本轮不实现第二变体。
5. 正式实验协议保持与 `1.3.7` 一致，方便直接和 `baseline_v2` 对比。

## 关键实现说明

### 1. latent pooling

- 从 history encoder 输出的 latent tokens 中，用少量 learned queries 做 attention pooling
- pooled tokens 再压缩为固定维度的 latent initial state
- 目的：把“历史上下文”从高维 token 集合转成适合 dynamics rollout 的紧凑状态

### 2. latent dynamics rollout

- 使用 GRU 在 prediction bins 上逐步推进 latent state
- 每一步不再依赖 GT future counts，也不使用显式 observation-space feedback
- 输入是固定上下文加 step embedding，hidden state 表示当前 latent trajectory

### 3. readout 保持不变

- 每一步的 latent state 经过投影后送入 `PerNeuronMLPHead`
- 输出目标仍是 `future T bins x N units` 的 spike counts

### 4. train / eval 入口保持不变

- 训练入口：`examples/neurohorizon/train.py`
- 离线正式评估入口：`scripts/analysis/neurohorizon/eval_phase1_v2.py`
- 需要在文档里显式注明：本轮没有改动训练入口和离线正式评估入口

## 代码实施清单

- [x] 新增 latent dynamics decoder 模块
- [x] 修改 `torch_brain/models/neurohorizon.py`
- [x] 修改 `torch_brain/nn/__init__.py`
- [x] 新增 `1.10` model/train configs
- [x] 新增 `1.10` 脚本目录
- [x] 完成 250ms smoke run
- [x] 启动 250/500/1000ms 正式训练
- [ ] 更新 `results.tsv`
- [ ] 生成趋势图并更新 `results.md`

## 基本功能验证

计划验证项：

- `decoder_variant=latent_dynamics` shape 正确
- `requires_target_counts=False`
- `forward()` 与 `generate()` 输出一致
- 250ms smoke run 能完成训练、保存 checkpoint、离线评估

功能验证脚本：
- `scripts/1.10-latent_dynamics_decoder/20260320_latent_dynamics_decoder/verify_latent_dynamics.py`

## 正式实验配置

- 数据集：`examples/neurohorizon/configs/dataset/perich_miller_10sessions.yaml`
- 采样方式：连续滑动窗口
- obs_window：500ms
- pred_window：250ms / 500ms / 1000ms
- 训练轮数：300 epochs
- 目标指标：`fp-bps` / `per-bin fp-bps`

## 目录约定

- 脚本：`scripts/1.10-latent_dynamics_decoder/20260320_latent_dynamics_decoder/`
- 日志：`results/logs/1.10-latent_dynamics_decoder/20260320_latent_dynamics_decoder/`
- 可视化：`results/figures/1.10-latent_dynamics_decoder/20260320_latent_dynamics_decoder/`

## 运行命令

```bash
conda activate poyo
cd /root/autodl-tmp/NeuroHorizon
python scripts/1.10-latent_dynamics_decoder/20260320_latent_dynamics_decoder/verify_latent_dynamics.py
python examples/neurohorizon/train.py --config-name=train_1p10_latent_dynamics_250ms
python examples/neurohorizon/train.py --config-name=train_1p10_latent_dynamics_500ms
python examples/neurohorizon/train.py --config-name=train_1p10_latent_dynamics_1000ms
```

## 实验结果

### 功能验证

- `decoder_variant=latent_dynamics` 可正常实例化：通过
- `requires_target_counts=False`：通过
- `forward()` 与 `generate()` 输出一致：通过
- 验证脚本输出：
  - `output_shape=(2, 12, 6)`
  - `tf_vs_rollout_max_delta=0.000000`

执行命令：

```bash
conda activate poyo
cd /root/autodl-tmp/NeuroHorizon
python scripts/1.10-latent_dynamics_decoder/20260320_latent_dynamics_decoder/verify_latent_dynamics.py
```

### 250ms smoke run

- 配置：
  - `train_1p10_latent_dynamics_250ms.yaml`
  - override: `epochs=1 eval_epochs=1 batch_size=256 eval_batch_size=256 num_workers=0`
- 训练日志：
  - `results/logs/1.10-latent_dynamics_decoder/20260320_latent_dynamics_decoder/250ms_smoke/lightning_logs/version_0/`
- smoke 结果：
  - train loss：`0.417`
  - val loss：`0.406`
  - val/r2：`-0.002`
  - val/fp_bps：`-0.834`
- 离线 continuous valid 评估：
  - 文件：`results/logs/1.10-latent_dynamics_decoder/20260320_latent_dynamics_decoder/250ms_smoke/eval_v2_valid_results.json`
  - fp-bps：`-0.8339`
  - R2：`-0.0021`
  - val_loss：`0.4079`
  - per-bin fp-bps（前 12 bins）：`-0.862 / -0.828 / -0.800 / -0.840 / -0.840 / -0.875 / -0.835 / -0.852 / -0.859 / -0.856 / -0.850 / -0.750`
- 结论：
  - 训练、checkpoint、离线评估链路均已打通
  - 当前数值仅反映 1 epoch smoke run，不用于与 `baseline_v2` 做正式性能对比

### 正式训练

- 启动时间：`2026-03-20 05:27 CST`
- 启动方式：`screen -S latent_dynamics_1p10`
- 主控脚本：`scripts/1.10-latent_dynamics_decoder/20260320_latent_dynamics_decoder/run_latent_dynamics_experiments.sh`
- 当前状态：
  - `250ms / 500ms / 1000ms` 三个窗口已经并发启动
  - 初步检查显示三路训练均已进入 epoch 1–2，无立即 OOM 或配置错误
  - `screen` 日志：`results/logs/1.10-latent_dynamics_decoder/20260320_latent_dynamics_decoder/screen_run.log`

| pred_window | teacher-forced fp-bps | rollout fp-bps | rollout R2 | vs baseline_v2 | 备注 |
|-------------|-----------------------|----------------|------------|----------------|------|
| 250ms | | | | | |
| 500ms | | | | | |
| 1000ms | | | | | |

## 与 baseline_v2 的对比

`baseline_v2` rollout / continuous fp-bps：
- 250ms: `0.2115`
- 500ms: `0.1744`
- 1000ms: `0.1317`

## 当前判断

- `1.10` 的第一阶段实现已经完成，主线代码、配置和 smoke protocol 均可用。
- `250/500/1000ms` 三窗口正式训练已经启动，当前处于正式结果产出阶段。
- 下一步是在训练完成后汇总 best-ckpt `valid/test`、更新 `results.tsv`，并生成趋势图与结果解读。

## 备注

- 本轮是 `1.10` 第一次任务执行，因此文档会比 `1.9` 的普通增量任务写得更完整，用于固定方向切换的前因后果和协议边界。
