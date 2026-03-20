# 2026-03-20 Direct Count-Space Flow Matching with DiT

> 对应计划：`cc_core_files/plan.md` §1.11  
> 分支：`dev/diffusion`  
> 状态：首轮 formal 完成，效果不佳，当前变体建议停止继续微调

## 任务背景

`1.9` 中围绕 AR decoder 的多轮优化已经证明：即使显式处理 teacher forcing / rollout mismatch，AR feedback 在当前 NeuroHorizon 任务中的边际收益仍然很低，且长窗口下很容易被噪声累积抵消。基于：

- `cc_todo/20260316-review/ar_effectiveness_claude.md`
- `cc_todo/20260316-review/long_horizon_prediction_claude.md`
- `cc_todo/20260316-review/option_d_implementation_claude.md`

当前决定将 Phase 1 的主线从 AR decoder 迁移到 diffusion decoder，并把 `Option 2B` 作为第一优先级。

## 本轮目标

本轮不是只做架构草图，而是一次完整的 1.11 首轮执行，目标包括：

1. 在 `plan.md` 中建立独立的 `1.11 diffusion decoder` 管理章节
2. 建立 diffusion 主线专用的模型文档 `cc_todo/1.11-diffusion-decoder/model.md`
3. 在当前 `dev/diffusion` 分支中挂接第一版 direct count-space flow matching + DiT 结构
4. 完成基础 smoke 验证
5. 完成 `250ms / 500ms / 1000ms` 三窗口正式训练与离线评估
6. 更新结果记录、图表、进度和汇总文档

## 本轮设计选择

### 主方案

- `Option 2B`
- target space: `log1p(count)`
- training objective: rectified flow matching velocity regression
- denoiser backbone: DiT 风格时间主干

### 为什么不直接做 2A

- 2A 需要额外的 count autoencoder，代码和实验链条更长
- 本轮目标是尽快把 diffusion 主线完整跑通，先验证“整体生成 future count field”这一方向是否可行
- 若 2B 暴露结构性问题，再把 2A 作为备选切入

### 为什么不做 `T × N` full token transformer

- 在 `1000ms` 窗口和完整 unit 集合下，`T × N` token 数会很大
- 直接 full attention 的显存和速度成本过高，不适合作为第一版正式实验
- 当前实现改为：把 noisy count field 汇总成 per-bin token，再通过共享 head 回到 unit 维

## 结合当前仓库实现的执行要点

1. history encoder 继续复用现有 `NeuroHorizon` 主干
2. 运行时只保留：
   - baseline query_aug / no-feedback 对照
   - diffusion_flow 主线
3. 1.9 的 prediction-memory / alignment 路线不再作为当前主实现的活跃逻辑维护
4. 训练与评估入口默认不变：
   - `examples/neurohorizon/train.py`
   - `scripts/analysis/neurohorizon/eval_phase1_v2.py`

## 实验协议

- 数据集：`examples/neurohorizon/configs/dataset/perich_miller_10sessions.yaml`
- 采样方式：continuous
- obs window：500ms
- pred windows：250ms / 500ms / 1000ms
- 核心指标：
  - `fp-bps`
  - `per-bin fp-bps`
  - `R2`
  - diffusion validation loss

## 预计改动模块

- `torch_brain/models/neurohorizon.py`
- `torch_brain/nn/diffusion_decoder.py`
- `examples/neurohorizon/train.py`
- `scripts/analysis/neurohorizon/eval_phase1_v2.py`
- `examples/neurohorizon/configs/model/neurohorizon_diffusion_flow_{250,500,1000}ms.yaml`
- `examples/neurohorizon/configs/train_1p11_diffusion_flow_{250,500,1000}ms.yaml`

## 计划命令

### 训练

```bash
conda run -n poyo --cwd /root/autodl-tmp/NeuroHorizon/examples/neurohorizon \
python train.py --config-name train_1p11_diffusion_flow_250ms.yaml
```

```bash
conda run -n poyo --cwd /root/autodl-tmp/NeuroHorizon/examples/neurohorizon \
python train.py --config-name train_1p11_diffusion_flow_500ms.yaml
```

```bash
conda run -n poyo --cwd /root/autodl-tmp/NeuroHorizon/examples/neurohorizon \
python train.py --config-name train_1p11_diffusion_flow_1000ms.yaml
```

### 正式离线评估

```bash
conda run -n poyo --cwd /root/autodl-tmp/NeuroHorizon \
python scripts/analysis/neurohorizon/eval_phase1_v2.py \
  --log-dir results/logs/1.11-diffusion-decoder/20260320_direct_count_flow_dit/250ms \
  --checkpoint-kind best \
  --split test
```

## 执行记录

### 2026-03-20

- 启动 1.11 路线建档
- 开始将当前运行时主实现切换为 baseline + diffusion_flow 两条主线
- 按 `Option 2B + flow matching + DiT` 挂接第一版代码和三窗口配置
- 完成最小 synthetic 验证：
  - `compute_training_loss()` 可返回有效 loss 和辅助统计
  - `generate()` 可输出 `[B, T, N]` 形状的 `log_rate`
- 完成真实数据 250ms smoke：
  - 训练命令：
    ```bash
    conda run -n poyo --cwd /root/autodl-tmp/NeuroHorizon/examples/neurohorizon \
      python train.py --config-name train_1p11_diffusion_flow_250ms.yaml \
      epochs=1 eval_epochs=1 batch_size=2 eval_batch_size=2 num_workers=0 \
      +max_steps=2 +limit_train_batches=2 +limit_val_batches=1 \
      log_dir=/root/autodl-tmp/NeuroHorizon/results/logs/1.11-diffusion-decoder/20260320_direct_count_flow_dit/250ms_smoke
    ```
  - 训练输出：
    - `train_loss = 1.110`
    - `val_loss = 1.055`
    - `val/fp_bps = -12.975`
    - `best.ckpt` / `last.ckpt` 已生成
  - 离线评估命令：
    ```bash
    conda run -n poyo --cwd /root/autodl-tmp/NeuroHorizon \
      python scripts/analysis/neurohorizon/eval_phase1_v2.py \
        --log-dir results/logs/1.11-diffusion-decoder/20260320_direct_count_flow_dit/250ms_smoke \
        --checkpoint-kind best --split valid --batch-size 2 --skip-trial --max-batches 1
    ```
  - 离线评估输出（1 batch smoke）：
    - `fp-bps = -12.9774`
    - `R2 = -31.2224`
    - `val_loss = 1.5183`
- 当前结论：
  - diffusion 主线的训练、checkpoint、best ckpt 选择、离线评估入口都已经打通
  - 第一版结构目前只有链路验证价值，还没有性能上的积极信号
  - 下一步应转向正式三窗口训练，并结合结果决定是否调整 `flow_steps_eval`、decoder depth、condition 设计或 count summary 方式

### 2026-03-20（formal 三窗口完成 + 结果归档）

- 正式运行命令：
  ```bash
  conda activate poyo
  cd /root/autodl-tmp/NeuroHorizon
  bash scripts/1.11-diffusion-decoder/20260320_direct_count_flow_dit/run_diffusion_flow_windows.sh
  ```
- 结果收集与出图命令：
  ```bash
  conda activate poyo
  cd /root/autodl-tmp/NeuroHorizon
  python scripts/1.11-diffusion-decoder/20260320_direct_count_flow_dit/collect_diffusion_flow_results.py
  ```
- 统一关键超参：
  - 数据协议：Perich-Miller 10 sessions，continuous，obs=500ms，pred=`250 / 500 / 1000ms`
  - optimizer：`SparseLamb + OneCycleLR`
  - `base_lr = 3.125e-5`
  - `max_lr = 0.002 / 0.002 / 0.001`
  - `weight_decay = 1e-4`
  - `flow_steps_eval = 20`
  - `epochs = 300`，`eval_epochs = 10`
  - batch size：`64 / 64 / 32`
  - eval batch size：`16 / 12 / 8`
- 正式结果：

  | window | best val fp-bps | best epoch | best val loss | test fp-bps | vs baseline_v2 test | test R2 | test PSTH-R2 |
  |--------|------------------|------------|---------------|-------------|----------------------|---------|--------------|
  | 250ms | -7.3585 | 229 | 0.3413 | -7.4950 | -7.7173 | -20.0418 | -13.6578 |
  | 500ms | -7.7572 | 179 | 0.3276 | -7.8601 | -8.0341 | -20.7506 | -9.6705 |
  | 1000ms | -8.0657 | 199 | 0.3504 | -8.2277 | -8.3625 | -20.7491 | -9.1904 |

- 生成产物：
  - `cc_todo/1.11-diffusion-decoder/results.tsv`
  - `results/figures/1.11-diffusion-decoder/20260320_direct_count_flow_dit/diffusion_flow_summary.json`
  - `results/figures/1.11-diffusion-decoder/20260320_direct_count_flow_dit/training_curves.png`
  - `results/figures/1.11-diffusion-decoder/20260320_direct_count_flow_dit/fp_bps_vs_window.png`
  - `results/figures/1.11-diffusion-decoder/20260320_direct_count_flow_dit/per_bin_fp_bps.png`
  - `results/figures/1.11-diffusion-decoder/20260320_direct_count_flow_dit/summary_table.png`
- 当前结论：
  - 三窗口都完整跑满到 `epoch 299`，说明这轮失败不是“没跑够”，而是模型结构本身没有学到有效的 future count field。
  - `best val/fp_bps` 出现在 `epoch 229 / 179 / 199`，而 `val_loss` 最低点在更后面，进一步说明当前 flow loss 与真正关心的 spike prediction 指标不对齐。
  - `per-bin fp-bps` 在所有 bin 上都显著为负，不是只在长窗口尾端崩塌；当前问题更像是 `per-bin summary` 丢掉了 unit-level 细节。
  - 这意味着继续在当前实现上微调 `flow_steps_eval`、depth、dropout 的性价比很低；若继续 1.11 主线，应改为 unit-level tokenization 或 factorized time-unit attention。
