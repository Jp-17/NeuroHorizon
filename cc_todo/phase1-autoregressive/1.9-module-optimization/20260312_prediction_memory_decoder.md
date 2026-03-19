# Phase 1.9 模块优化：Structured Prediction Memory Decoder

**日期**：2026-03-12
**模块名**：`prediction_memory_decoder`
**状态**：已放弃
**分支**：`dev/20260312_prediction_memory_decoder`

## 改进摘要

本次 1.9 优化将 Phase 1 的主线 AR decoder 从“Query Augmentation + 单向量 feedback”升级为“Structured Prediction Memory”。保持 POYO event-based encoder 不变，输出仍为未来时间窗内各 neuron 的 spike counts；decoder 在时间维做 bin-by-bin 自回归，并通过结构化的 prediction memory 显式接收上一 bin 的 population state。

对应的设计说明已写入：
- `cc_core_files/model.md` → `2026-03-12 — Structured Prediction Memory Decoder`

## 设计结论

1. 不采用 spike-event level decoder，目标维持为 binned spike counts。
2. 不做 neuron-by-neuron autoregression，只沿时间维做 AR。
3. 保留 learnable bin queries 和 rotary time embedding。
4. 将旧 `feedback_method` / Query Augmentation 降级为 baseline / ablation。
5. 主线新方案为：
   - `history latents` 来自 POYO encoder
   - `prediction memory` 来自上一 bin population counts 的结构化编码
   - decoder 层顺序固定为：
     - history cross-attn
     - prediction-memory cross-attn
     - causal self-attn
     - FFN

## 关键实现说明

### 1. shift-right

- teacher forcing 下不允许当前 bin 直接看到当前步 GT counts。
- 训练时构造：
  - `memory[0] = 0`
  - `memory[t] = encode(counts[t-1])`
- 这样 query `t` 访问的是“上一 bin”的人口状态，而不是当前 bin 真值。

### 2. prediction memory 编码

- 对每个 neuron 构造 `MLP([unit_emb_n ; log1p(count_n)])`
- 使用 `K=4` 个 learned pooling queries 做 attention pooling
- 每个历史 bin 生成 `4` 个 summary tokens
- 当前版本固定 `K=4`，先做概念验证，不扫参

### 3. 旧 feedback 逻辑的保留方式

- `decoder_variant='query_aug'`：继续走旧 `feedback_method`
- `decoder_variant='prediction_memory'`：走新主线
- 旧逻辑保留的目的：
  - 作为 baseline 对照
  - 作为回归测试路径

## 代码实施清单

- [x] 新增 `torch_brain/nn/prediction_memory.py`
- [x] 修改 `torch_brain/nn/autoregressive_decoder.py`
- [x] 修改 `torch_brain/models/neurohorizon.py`
- [x] 修改 `examples/neurohorizon/train.py`
- [x] 修改 `scripts/analysis/neurohorizon/eval_phase1_v2.py`
- [x] 新增 prediction-memory model/train configs
- [x] 新增 1.9 模块脚本目录
- [x] 完成 250/500/1000ms 正式训练
- [x] 汇总 metrics 并更新 `results.tsv`
- [x] 生成趋势图并更新 `results.md`

## 基本功能验证

计划验证项：
- `PredictionMemoryEncoder` shape 正确
- `shift-right` 正确：修改 `counts[t]` 不影响同一步输出，只影响未来步
- `forward()` 与 `generate()` 不再数值等价
- rollout 中修改某一步预测会影响后续 bins

功能验证脚本：
- `scripts/phase1-autoregressive-1.9-module-optimization/20260312_prediction_memory_decoder/verify_prediction_memory.py`

**2026-03-12 验证结果**：
- `PredictionMemoryEncoder` 输出 shape 验证通过：`[B, 4, D]`
- masked units 不影响 memory pooling：验证通过
- `forward()` 与 `generate()` 不再等价：验证通过，`tf_vs_rollout_max_delta=0.000230`
- `shift-right` 验证通过：
  - 修改 `target_counts[:, 0, :]` 后，`bin 0` 输出变化 `0.000000`
  - `bin >= 1` 输出变化 `0.002075`
- prediction-memory causal mask shape 验证通过：`[B, T, T*K]`

执行命令：

```bash
conda activate poyo
cd /root/autodl-tmp/NeuroHorizon
python scripts/phase1-autoregressive-1.9-module-optimization/20260312_prediction_memory_decoder/verify_prediction_memory.py
```

## 正式实验配置

- 数据集：`examples/neurohorizon/configs/dataset/perich_miller_10sessions.yaml`
- 采样方式：连续滑动窗口
- obs_window：500ms
- pred_window：250ms / 500ms / 1000ms
- 训练轮数：300 epochs
- 目标指标：fp-bps / R2 / PSTH-R2 / Poisson NLL

**后台运行状态**：
- 2026-03-12 晚间已启动正式并行实验脚本：
  - `scripts/phase1-autoregressive-1.9-module-optimization/20260312_prediction_memory_decoder/run_prediction_memory_experiments.sh`
- 后台日志：
  - `results/logs/phase1-autoregressive-1.9-module-optimization/20260312_prediction_memory_decoder/run_prediction_memory_experiments.log`
- 监控脚本：
  - `scripts/phase1-autoregressive-1.9-module-optimization/20260312_prediction_memory_decoder/monitor_prediction_memory_progress.py`
  - 状态文件：`results/logs/phase1-autoregressive-1.9-module-optimization/20260312_prediction_memory_decoder/progress_status.md`
- 说明：
  - 该后台任务会并行执行 250ms / 500ms / 1000ms 训练，并在各自完成后自动评估
  - `results.tsv` 需待正式结果产出后再更新

**2026-03-12 22:30 并行运行快照**：
- GPU：RTX 4090 D，三开时显存占用约 `7.7 / 24.0 GiB`
- 主任务 pid：
  - 250ms: `957246`
  - 500ms: `957247`
  - 1000ms: `957249`
- 当前训练进度（按最新 `metrics.csv`）：
  - 250ms：已完成到 `epoch 8`，最近 `epoch_time ≈ 27.73s`，粗略剩余 `约 2h14m`
  - 500ms：已完成到 `epoch 10`，最近 `epoch_time ≈ 20.63s`，粗略剩余 `约 1h39m`
  - 1000ms：已完成到 `epoch 8`，最近 `epoch_time ≈ 27.13s`，粗略剩余 `约 2h11m`
- 自动监控：
  - 每 5 分钟刷新一次 `progress_status.md`
  - 路径：`results/logs/phase1-autoregressive-1.9-module-optimization/20260312_prediction_memory_decoder/progress_status.md`

## 目录约定

- 脚本：`scripts/phase1-autoregressive-1.9-module-optimization/20260312_prediction_memory_decoder/`
- 日志：`results/logs/phase1-autoregressive-1.9-module-optimization/20260312_prediction_memory_decoder/`
- 可视化：`results/figures/phase1-autoregressive-1.9-module-optimization/20260312_prediction_memory_decoder/`

## 运行命令

```bash
conda activate poyo
cd /root/autodl-tmp/NeuroHorizon
python scripts/phase1-autoregressive-1.9-module-optimization/20260312_prediction_memory_decoder/verify_prediction_memory.py
python examples/neurohorizon/train.py --config-name=train_1p9_prediction_memory_250ms
python examples/neurohorizon/train.py --config-name=train_1p9_prediction_memory_500ms
python examples/neurohorizon/train.py --config-name=train_1p9_prediction_memory_1000ms
```

## 实验结果

### 功能验证

- 已完成，见上方“基本功能验证”小节

### 250ms smoke run（1 epoch，仅验证训练/评估链路）

- 配置：
  - `train_1p9_prediction_memory_250ms.yaml`
  - override: `epochs=1 eval_epochs=1 batch_size=256 eval_batch_size=256 num_workers=0`
- 训练日志：
  - `results/logs/phase1-autoregressive-1.9-module-optimization/20260312_prediction_memory_decoder/250ms/lightning_logs/version_0/`
- smoke 结果：
  - train loss: `0.424`
  - val loss: `0.412`
  - val/r2: `0.000`
  - val/fp_bps: `-0.823`
- rollout smoke 评估：
  - 文件：`results/logs/phase1-autoregressive-1.9-module-optimization/20260312_prediction_memory_decoder/250ms/eval_rollout_smoke.json`
  - fp-bps: `-0.8218`
  - R2: `0.0001`
  - val_loss: `0.4132`
- 结论：
  - 训练、checkpoint、rollout evaluation 路径均已走通
  - 该结果仅为 1 epoch smoke run，不用于与 `baseline_v2` 做正式性能对比

### 正式训练

| pred_window | teacher-forced fp-bps | rollout fp-bps | rollout R2 | vs baseline_v2 | 备注 |
|-------------|-----------------------|----------------|------------|----------------|------|
| 250ms | 0.2979 | 0.1486 | 0.2519 | -0.0629 | rollout 从 bin 10 开始转负 |
| 500ms | 0.2832 | -0.0153 | 0.2049 | -0.1897 | rollout 从 bin 11 开始转负 |
| 1000ms | 0.2776 | -0.2590 | 0.1617 | -0.3907 | rollout 从 bin 9 开始转负 |

## 与 baseline_v2 的对比

`baseline_v2` rollout / continuous fp-bps：
- 250ms: `0.2115`
- 500ms: `0.1744`
- 1000ms: `0.1317`

对比结论：
- teacher forcing 下，本次模型三组窗口都明显高于 baseline，说明模型在“使用 GT shift-right memory”时拟合能力很强。
- rollout 下，本次模型三组窗口都低于 baseline，且窗口越长退化越严重。
- 与 baseline 的差值：
  - 250ms: `-0.0629`
  - 500ms: `-0.1897`
  - 1000ms: `-0.3907`

## 为什么比 baseline_v2 更差

1. **高容量显式 memory 造成 teacher-forced 依赖过强**  
   当前 decoder 可以 cross-attend 多个 structured prediction memory tokens。相比 baseline_v2 的单向量 feedback，这条通路更强、更直接，训练时模型很容易过度依赖 GT counts 构造出的 memory。

2. **训练 / 推理 memory 输入分布不一致**  
   训练使用 `log1p(GT integer counts)`，推理使用 `log1p(predicted expected counts)`。二者的统计特性差异明显：前者是稀疏离散脉冲，后者是平滑连续率值。memory encoder 在 rollout 时面对的是更“软”的输入分布，导致 learned mapping 失稳。

3. **全历史 memory 检索放大误差传播**  
   当前 query 可以访问全历史 memory，而不只是紧邻上一步。teacher forcing 下这相当于提供了更强的历史监督；rollout 下则会把早期预测误差编码进 memory 后持续传播到后续所有 bins，长窗口退化尤其明显。

4. **decoder 已有 causal self-attention，再叠加强显式 memory 后主通路失衡**  
   baseline_v2 的信息瓶颈反而起到了正则化作用，使模型更多依赖 history latents 和 hidden state；本次 structured memory 版本则让模型在训练中学会“优先查 memory”，自由运行时因此更脆弱。

## 最终结论

- 本方案**不是代码实现错误导致失败**，而是典型的“teacher forcing 指标高、rollout 稳定性差”的架构问题。
- 该版本不合并为主线，状态标记为**已放弃**。
- 下一轮改进应优先减少或约束 prediction feedback 通路的容量，避免全历史高容量 memory 成为 decoder 的主依赖。

## 备注

- 本次只实现主线 `prediction_memory`，不同时引入 scheduled sampling。
- rollout 反馈使用 predicted expected counts，即 `exp(log_rate)`，不做 Poisson sampling。

## 2026-03-19 补充：training curves 可视化

- 脚本：`scripts/phase1-autoregressive-1.9-module-optimization/plot_optimization_training_curves.py`
- 图表：`results/figures/phase1-autoregressive-1.9-module-optimization/20260312_prediction_memory_decoder/training_curves.png`
- 数据来源：
  - `results/logs/phase1-autoregressive-1.9-module-optimization/20260312_prediction_memory_decoder/250ms/lightning_logs/version_3/metrics.csv`
  - `results/logs/phase1-autoregressive-1.9-module-optimization/20260312_prediction_memory_decoder/500ms/lightning_logs/version_1/metrics.csv`
  - `results/logs/phase1-autoregressive-1.9-module-optimization/20260312_prediction_memory_decoder/1000ms/lightning_logs/version_0/metrics.csv`
- 图表内容：
  - 上排为 `250/500/1000ms` 三个窗口的 epoch-level `train_loss` 与 `val_loss`
  - 下排为训练期 `val/fp_bps` 曲线，并叠加 post-train teacher-forced、rollout 与 `baseline_v2` 参考线
- 观察结论：
  1. 三个窗口的 loss 曲线都能稳定下降，说明训练链路本身不存在发散或数值不稳定。
  2. `500ms/1000ms` 的 `val/fp_bps` 在中后期已经明显进入平台，但 post-train rollout 参考线仍远低于训练末期 teacher-forced 水平，进一步确认核心问题是 exposure bias / rollout drift，而不是训练没收敛。

执行命令：

```bash
conda activate poyo
cd /root/autodl-tmp/NeuroHorizon
python scripts/phase1-autoregressive-1.9-module-optimization/plot_optimization_training_curves.py --module 20260312_prediction_memory_decoder
```
