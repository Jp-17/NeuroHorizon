# 2026-03-20 Factorized Unit-Time Flow Tokens

> 对应计划：`cc_core_files/plan.md` §1.11  
> 分支：`dev/diffusion`  
> 状态：实施中

## 任务背景

`20260320_direct_count_flow_dit` 已经完成首轮 formal，并明确暴露出当前 `per-bin summary + shared per-unit head` 结构的问题：模型在进入时间主干之前就把 unit 维信息汇总掉了，导致三个窗口的 `continuous / trial-aligned` 指标全面深度为负。因此第二轮迭代不再继续围绕当前 summary 路线做小幅调参，而是直接恢复 unit-level token。

## 本轮目标

1. 将 diffusion decoder 改成显式 `(time bin, unit)` token 表示
2. 用 factorized 的 time/unit mixing 替代上一轮的 `per-bin summary` 主干
3. 保持 `Option 2B`、训练入口和评估入口不变，先控制变量验证结构改动本身
4. 至少完成一次真实数据 `250ms` smoke，确认训练、checkpoint 和离线评估链路仍然可用

## 本轮设计选择

### 主方案

- 继续使用 `Option 2B`
- target space 仍为 `log1p(count)`
- training objective 仍为 rectified flow matching velocity regression
- 核心结构改为：
  - pooled time-token cross-attention to history latents
  - per-unit time self-attention
  - per-time unit attention

### 为什么这样改

- 上一轮失败并不是优化器或 horizon 单独导致，而是 unit 维细节在主干前被压缩丢失
- 直接上 `T x N` full attention 仍然太重；factorized time/unit mixing 是更现实的折中
- 这种结构仍然保留“整体 future field generation”的 diffusion 思路，但不再把每个时间步的所有 unit 先平均成单个 token

## 预计改动模块

- `torch_brain/nn/diffusion_decoder.py`
- `examples/neurohorizon/configs/model/neurohorizon_factorized_unit_time_flow_{250,500,1000}ms.yaml`
- `examples/neurohorizon/configs/train_1p11_factorized_unit_time_flow_{250,500,1000}ms.yaml`
- `cc_todo/1.11-diffusion-decoder/model.md`
- `cc_core_files/plan.md`

## 计划命令

### 250ms smoke 训练

```bash
conda run -n poyo --cwd /root/autodl-tmp/NeuroHorizon/examples/neurohorizon \
python train.py --config-name train_1p11_factorized_unit_time_flow_250ms.yaml \
  epochs=1 eval_epochs=1 batch_size=2 eval_batch_size=2 num_workers=0 \
  +max_steps=2 +limit_train_batches=2 +limit_val_batches=1 \
  log_dir=/root/autodl-tmp/NeuroHorizon/results/logs/1.11-diffusion-decoder/20260320_factorized_unit_time_flow/250ms_smoke
```

### 250ms smoke 离线评估

```bash
conda run -n poyo --cwd /root/autodl-tmp/NeuroHorizon \
python scripts/analysis/neurohorizon/eval_phase1_v2.py \
  --log-dir results/logs/1.11-diffusion-decoder/20260320_factorized_unit_time_flow/250ms_smoke \
  --checkpoint-kind best --split valid --batch-size 2 --skip-trial --max-batches 1
```

## 执行记录

### 2026-03-20

- 根据 `20260320_direct_count_flow_dit` 的 formal 失败结论建立第二轮迭代文档
- 计划直接替换当前 diffusion decoder 的内部结构，不新增新的训练/评估入口
- 当前重点是验证 factorized unit-time token 表达能否在最小 smoke 下稳定前向、回传和生成
- 完成 factorized unit-time token 版本 `DiffusionFlowDecoder` 的初版实现
- 新增三窗口训练配置：
  - `train_1p11_factorized_unit_time_flow_{250,500,1000}ms.yaml`
- 完成 250ms 真实数据 smoke：
  - 训练命令：
    ```bash
    conda run -n poyo --cwd /root/autodl-tmp/NeuroHorizon/examples/neurohorizon \
    python train.py --config-name train_1p11_factorized_unit_time_flow_250ms.yaml \
      epochs=1 eval_epochs=1 batch_size=2 eval_batch_size=2 num_workers=0 \
      +max_steps=2 +limit_train_batches=2 +limit_val_batches=1 \
      log_dir=/root/autodl-tmp/NeuroHorizon/results/logs/1.11-diffusion-decoder/20260320_factorized_unit_time_flow/250ms_smoke
    ```
  - 训练输出：
    - `train_loss = 1.140`
    - `val_loss = 1.146`
    - `val/fp_bps = -15.280`
    - `best.ckpt / last.ckpt` 正常生成
  - 离线评估命令：
    ```bash
    conda run -n poyo --cwd /root/autodl-tmp/NeuroHorizon \
    python scripts/analysis/neurohorizon/eval_phase1_v2.py \
      --log-dir results/logs/1.11-diffusion-decoder/20260320_factorized_unit_time_flow/250ms_smoke \
      --checkpoint-kind best --split valid --batch-size 2 --skip-trial --max-batches 1
    ```
  - 离线评估输出（1 batch smoke）：
    - `fp-bps = -15.2633`
    - `R2 = -51.9120`
    - `val_loss = 1.7312`
- 当前结论：
  - 第二轮结构替换没有破坏训练和评估链路，说明 factorized unit-time token 版本具备继续做正式实验的工程基础
  - 由于 smoke 步数极少，当前指标只能说明“尚未学到东西”，不能拿来与上一轮 formal 或 baseline_v2 做性能结论
