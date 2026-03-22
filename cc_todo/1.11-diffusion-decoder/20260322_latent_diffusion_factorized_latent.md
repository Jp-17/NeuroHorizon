# 2026-03-22 Latent Diffusion with Factorized Time-Unit Latents

> 对应计划：`cc_core_files/plan.md` §1.11  
> 分支：`dev/diffusion`  
> 状态：验证中（`250ms` smoke 已通过，formal gate 待执行）

## 任务背景

`20260321_dense_history_cross_factorized_flow` 的 `250ms formal gate` 已经确认失败，说明 `Option 2B` 剩余 gap 不能再简单归因于 pooled conditioning。继续在 raw count-space diffusion 主干上叠局部 cross-attention 改动，已经很难回答更本质的问题。

因此本轮正式转向 `Option 2A latent diffusion`：

1. 先把 future `log1p(count)` 编码到更低维、更平滑的 latent 空间
2. 在 latent 空间中做 rectified flow matching
3. 再由 latent decoder 重建 per-bin, per-unit 的 future log-rate

首版 latent 形态固定为 `time x factorized latent units`，保留时间轴，同时对 unit 维进行压缩。

## 本轮目标

1. 在 `NeuroHorizon` 中新增 `decoder_variant='latent_diffusion'`
2. 实现 deterministic factorized count autoencoder + latent diffusion decoder
3. 保持训练与离线评估入口不变，只扩展其对 2A 路线的支持
4. 完成最小功能验证与 `250ms` 真实数据 smoke
5. 为下一步 `250ms formal gate` 准备配置与脚本

## 本轮设计选择

- 路线：`Option 2A latent diffusion`
- latent 形态：`time x factorized latent units`
- autoencoder：deterministic，不加 KL，不做 VAE
- target space：`log1p(count)`
- objective：继续复用当前分支的 rectified flow matching
- 训练方式：autoencoder + latent diffusion 联合训练
- 协议：沿用 `1.3.7` 默认协议（10 sessions、continuous、obs=`500ms`）
- 验证策略：`250ms gate-first`

## 涉及改动模块

- `torch_brain/nn/latent_diffusion_decoder.py`
- `torch_brain/nn/__init__.py`
- `torch_brain/models/neurohorizon.py`
- `examples/neurohorizon/train.py`
- `scripts/analysis/neurohorizon/eval_phase1_v2.py`
- `examples/neurohorizon/configs/model/neurohorizon_latent_diffusion_factorized_latent_{250,500,1000}ms.yaml`
- `examples/neurohorizon/configs/train_1p11_latent_diffusion_factorized_latent_{250,500,1000}ms.yaml`
- `scripts/1.11-diffusion-decoder/20260322_latent_diffusion_factorized_latent/verify_latent_diffusion_factorized_latent.py`
- `scripts/1.11-diffusion-decoder/20260322_latent_diffusion_factorized_latent/run_latent_diffusion_factorized_latent_250ms_gate.sh`

## 执行记录

### 2026-03-22

- 已实现 `FactorizedCountAutoencoder`
  - future `log1p(count)` 先转成 `(time, unit)` token
  - 经过 factorized time/unit mixing
  - 用 learned latent queries 按时间步压缩成 factorized latent tokens
  - 再从 latent tokens 重建 per-unit transformed count
- 已实现 `LatentDiffusionDecoder`
  - 对 factorized latent tokens 做 rectified flow matching
  - 使用 pooled time-token cross-attention 到 history latents 做条件注入
  - 采样后通过 autoencoder decoder 重建 future log-rate
- 已在 `NeuroHorizon` 中新增 `decoder_variant='latent_diffusion'`
- 已扩展训练与评估入口：
  - `examples/neurohorizon/train.py`
  - `scripts/analysis/neurohorizon/eval_phase1_v2.py`
- 已新增三窗口配置：
  - `train_1p11_latent_diffusion_factorized_latent_{250,500,1000}ms.yaml`
- 已新增最小验证脚本：
  - `verify_latent_diffusion_factorized_latent.py`

## 运行命令

### 最小功能验证

```bash
cd /root/autodl-tmp/NeuroHorizon
/root/miniconda3/bin/conda run -n poyo python \
scripts/1.11-diffusion-decoder/20260322_latent_diffusion_factorized_latent/verify_latent_diffusion_factorized_latent.py
```

### 250ms smoke 训练

```bash
cd /root/autodl-tmp/NeuroHorizon/examples/neurohorizon
/root/miniconda3/bin/conda run -n poyo python train.py \
  --config-name train_1p11_latent_diffusion_factorized_latent_250ms.yaml \
  epochs=1 eval_epochs=1 batch_size=2 eval_batch_size=2 num_workers=0 \
  +max_steps=2 +limit_train_batches=2 +limit_val_batches=1 \
  log_dir=/root/autodl-tmp/NeuroHorizon/results/logs/1.11-diffusion-decoder/20260322_latent_diffusion_factorized_latent/250ms_smoke
```

### 250ms smoke 离线评估（valid）

```bash
cd /root/autodl-tmp/NeuroHorizon
/root/miniconda3/bin/conda run -n poyo python \
scripts/analysis/neurohorizon/eval_phase1_v2.py \
  --log-dir results/logs/1.11-diffusion-decoder/20260322_latent_diffusion_factorized_latent/250ms_smoke \
  --checkpoint-kind best --split valid --batch-size 2 --skip-trial --max-batches 1
```

### 250ms smoke 离线评估（test）

```bash
cd /root/autodl-tmp/NeuroHorizon
/root/miniconda3/bin/conda run -n poyo python \
scripts/analysis/neurohorizon/eval_phase1_v2.py \
  --log-dir results/logs/1.11-diffusion-decoder/20260322_latent_diffusion_factorized_latent/250ms_smoke \
  --checkpoint-kind best --split test --batch-size 2 --skip-trial --max-batches 1
```

## 当前结果

- 最小功能验证通过：
  - `compute_training_loss()` 可返回总 loss、`ae_recon_loss`、`diffusion_latent_loss`
  - `generate()` 输出 shape 正确，数值有限
- `250ms` smoke 训练：
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

## 当前结论

- `Option 2A` 首版工程链路已经完整跑通
- 相比前三轮 `Option 2B` 的 smoke，当前数值明显更接近可用区间，说明“先压到 latent 再生成”值得继续执行正式 gate
- 但当前仍然只是 smoke，不能直接据此宣告主线切换成功

## 遇到的问题

- 远端非交互 shell 中没有 `python` / `conda` 命令
  - 解决：编译检查使用 `python3`，训练与评估统一使用 `/root/miniconda3/bin/conda run -n poyo python`

## 下一步

1. 提交实现 checkpoint
2. 执行 `run_latent_diffusion_factorized_latent_250ms_gate.sh`
3. 按 `250ms test fp-bps >= -2.5` 判断是否扩到 `500 / 1000ms`
