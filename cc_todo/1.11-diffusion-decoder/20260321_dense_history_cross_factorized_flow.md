# 2026-03-21 Dense History-Cross Factorized Flow

> 对应计划：`cc_core_files/plan.md` §1.11  
> 分支：`dev/diffusion`  
> 状态：实施中

## 任务背景

`20260320_factorized_unit_time_flow` 已经确认：保留 `(time bin, unit)` 显式 token 是正确方向，但第二轮的 history conditioning 仍然太弱。当前 block 仍然先对 unit 维做 masked pooling，再让 pooled time token cross-attend 到 history latents；这一层 pooled conditioning 被认为是第二轮 remaining gap 的核心来源。

第三轮不切到 `Option 2A`，继续沿 `Option 2B direct count-space flow matching` 推进，但把 conditioning 改成**dense token-wise history cross-attention**：

1. 每个 `(time bin, unit)` token 直接 query history latents
2. 保留 factorized time/unit mixing
3. 不同时改优化器、flow step 或训练入口，优先控制变量验证 conditioning 本身

## 本轮目标

1. 将 pooled time-token cross-attention 替换为 dense token-wise history cross-attention
2. 保留第二轮已经证明有价值的 unit-level tokenization 与 factorized time/unit mixing
3. 完成 `250ms` smoke，确认训练、checkpoint 与离线评估链路可用
4. 启动 `250ms` formal gate，并以 `test fp-bps >= -2.5` 作为是否扩到 `500 / 1000ms` 的默认门槛

## 本轮设计选择

### 主方案

- 继续使用 `Option 2B`
- target space 保持 `log1p(count)`
- objective 保持 rectified flow matching velocity regression
- block 结构固定为：
  - dense token-wise cross-attention to history latents
  - per-unit time self-attention
  - per-time unit attention
  - FFN

### 为什么这样改

- 第二轮已经证明 summary bottleneck 不是唯一问题，但 pooled conditioning 很可能仍然压掉了 token-level history 信息
- 直接做 dense token-wise cross-attention，是最小但最关键的结构修正
- 仍然不引入 full `(T*N) x (T*N)` self-attention，也不切到 latent diffusion，保证本轮实验主要回答“conditioning 是否是剩余主瓶颈”

## 预计改动模块

- `torch_brain/nn/diffusion_decoder.py`
- `examples/neurohorizon/configs/model/neurohorizon_dense_history_cross_factorized_flow_{250,500,1000}ms.yaml`
- `examples/neurohorizon/configs/train_1p11_dense_history_cross_factorized_flow_{250,500,1000}ms.yaml`
- `scripts/1.11-diffusion-decoder/20260321_dense_history_cross_factorized_flow/`
- `cc_todo/1.11-diffusion-decoder/model.md`
- `cc_core_files/plan.md`

## 计划命令

### 250ms smoke 训练

```bash
conda run -n poyo --cwd /root/autodl-tmp/NeuroHorizon/examples/neurohorizon \
python train.py --config-name train_1p11_dense_history_cross_factorized_flow_250ms.yaml \
  epochs=1 eval_epochs=1 batch_size=2 eval_batch_size=2 num_workers=0 \
  +max_steps=2 +limit_train_batches=2 +limit_val_batches=1 \
  log_dir=/root/autodl-tmp/NeuroHorizon/results/logs/1.11-diffusion-decoder/20260321_dense_history_cross_factorized_flow/250ms_smoke
```

### 250ms smoke 离线评估

```bash
conda run -n poyo --cwd /root/autodl-tmp/NeuroHorizon \
python scripts/analysis/neurohorizon/eval_phase1_v2.py \
  --log-dir results/logs/1.11-diffusion-decoder/20260321_dense_history_cross_factorized_flow/250ms_smoke \
  --checkpoint-kind best --split valid --batch-size 2 --skip-trial --max-batches 1
```

### 250ms formal gate

```bash
conda activate poyo
cd /root/autodl-tmp/NeuroHorizon
bash scripts/1.11-diffusion-decoder/20260321_dense_history_cross_factorized_flow/run_dense_history_cross_factorized_flow_250ms_gate.sh
```

## 执行记录

### 2026-03-21

- 根据 `20260320_factorized_unit_time_flow` 的 formal 结论建立第三轮迭代文档
- 默认策略确定为 `250ms gate-first`，不先三窗口全开
- 当前目标是以最小结构改动验证 dense token-wise history cross-attention 是否能继续明显抬升 diffusion 主线
- 完成 dense token-wise history cross 版本 `DiffusionFlowDecoder` 的初版实现
- 新增三窗口训练配置：
  - `train_1p11_dense_history_cross_factorized_flow_{250,500,1000}ms.yaml`
- 完成 250ms 真实数据 smoke：
  - 训练命令：
    ```bash
    conda run -n poyo --cwd /root/autodl-tmp/NeuroHorizon/examples/neurohorizon \
    python train.py --config-name train_1p11_dense_history_cross_factorized_flow_250ms.yaml \
      epochs=1 eval_epochs=1 batch_size=2 eval_batch_size=2 num_workers=0 \
      +max_steps=2 +limit_train_batches=2 +limit_val_batches=1 \
      log_dir=/root/autodl-tmp/NeuroHorizon/results/logs/1.11-diffusion-decoder/20260321_dense_history_cross_factorized_flow/250ms_smoke
    ```
  - 训练输出：
    - `train_loss = 1.139`
    - `val_loss = 1.145`
    - `val/fp_bps = -15.258`
    - `best.ckpt / last.ckpt` 正常生成
  - 离线评估输出（1 batch smoke）：
    - `fp-bps = -15.2412`
    - `R2 = -51.6662`
    - `val_loss = 1.7292`
- 当前结论：
  - 第三轮 dense history cross 结构没有破坏训练与离线评估链路
  - smoke 结果相对第二轮只出现极小幅变化，现阶段不能据此判断 formal gate 是否会通过
  - 下一步应提交实现 checkpoint，并直接运行 `250ms formal gate`
