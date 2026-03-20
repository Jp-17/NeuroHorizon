# 脚本管理记录

> 记录项目中所有脚本（数据处理 / 分析 / 项目运行 / 测试等）的信息。
> 每新建一个脚本，必须在此处更新记录。
> **按 Phase + task_num 顺序排列**，便于追溯。

---

## 记录格式

````
### 脚本名称

- **路径**：scripts/xxx/xxx.py
- **功能用途**：描述该脚本做什么
- **创建时间**：YYYY-MM-DD
- **使用方式**：
  ```bash
  python scripts/xxx/xxx.py --args
  ```
- **输入**：说明输入文件/目录
- **输出**：说明输出文件/目录
- **依赖**：特殊依赖（环境、包等）
- **备注**：其他注意事项
````

---

## POYO 框架原有脚本

### calculate_normalization_scales.py

- **路径**：scripts/calculate_normalization_scales.py
- **功能用途**：计算数据集归一化参数（POYO 框架原有脚本）
- **创建时间**：（POYO 框架原有）
- **备注**：POYO 框架自带，非 NeuroHorizon 新增

---

## Phase 0：环境准备与基线复现

### perich_miller_pipeline.py（0.2.1 数据准备）

- **路径**：`scripts/data/perich_miller_pipeline.py`
- **功能用途**：下载并处理 Perich-Miller Population 2018 数据集的子集（10 sessions）
  - 从 DANDI Archive（DANDI:000688/draft）下载指定 sessions 的 NWB 文件
  - 提取 spike times、cursor 行为数据、trial 结构，转换为 brainsets 标准 HDF5 格式
  - 划分 train/valid/test splits（valid=0.1, test=0.2）
- **创建时间**：2026-02-28
- **使用方式**：
  ```bash
  conda activate poyo
  python -m brainsets.runner scripts/data/perich_miller_pipeline.py \
      --raw-dir=data/raw --processed-dir=data/processed -c4
  ```
- **输入**：DANDI Archive 在线下载（需网络）
- **输出**：
  - raw NWB：`data/raw/perich_miller_population_2018/sub-{C,J,M}/`
  - processed HDF5：`data/processed/perich_miller_population_2018/*.h5`
- **依赖**：poyo conda 环境（dandi>=0.61.2, scikit-learn, temporaldata, brainsets）
- **备注**：
  - 修改自 brainsets 官方 pipeline，通过 `SELECTED_PATHS` 限制为 10 sessions
  - 如需下载更多 sessions，修改 `SELECTED_PATHS` 集合或直接用 `brainsets prepare`
  - 运行需要 `brainsets` 配置（`~/.brainsets.yaml`）或通过 `--raw-dir/--processed-dir` 覆盖

### explore_brainsets.py（0.2.3 数据探索）

- **路径**：`scripts/analysis/explore_brainsets.py`
- **功能用途**：Perich-Miller 2018 数据集深度探索分析
  - 加载全部 10 sessions，统计数据集概览
  - 分析任务结构（hold/reach/return period 时长分布）
  - 计算神经元统计特征（发放率分布、spike count 稀疏度、Poisson 适配性）
  - 评估自回归可行性（各预测窗口 steps 数和可行比例）
  - 生成 PSTH、Raster plot、firing rate 分布等可视化图表
  - 输出 exploration_summary.json 汇总关键统计量
- **创建时间**：2026-02-28
- **使用方式**：
  ```bash
  conda activate poyo
  python scripts/analysis/explore_brainsets.py
  ```
- **输入**：`data/processed/perich_miller_population_2018/*.h5`
- **输出**：
  - `results/figures/data_exploration/01_dataset_overview.png`
  - `results/figures/data_exploration/02_neural_statistics.png`
  - `results/figures/data_exploration/exploration_summary.json`
- **依赖**：poyo conda 环境（temporaldata, h5py, matplotlib, numpy）
- **备注**：对应 plan.md 任务 0.2.3

### analyze_latents.py（0.3.2 Latent 质量分析）

- **路径**：`scripts/analysis/analyze_latents.py`
- **功能用途**：POYO encoder latent 提取 + PCA 可视化 + 线性探针评估
- **创建时间**：2026-02-28
- **使用方式**：
  ```bash
  conda activate poyo
  cd /root/autodl-tmp/NeuroHorizon
  python scripts/analysis/analyze_latents.py
  ```
- **输出**：`results/figures/baseline/`
- **依赖**：poyo conda 环境
- **备注**：对应 plan.md 任务 0.3.2

---


### phase0_baseline_plots.py (0.3 可视化补充)

- **路径**：`scripts/analysis/phase0_baseline_plots.py`
- **功能用途**：POYO+ 基线训练过程可视化
  - 解析 Lightning metrics.csv（处理混合行格式）
  - 绘制 2x2 子图：Train/Val Loss、Average R²、Per-Session R² 曲线、Per-Session R² 柱状图
- **创建时间**：2026-03-02
- **使用方式**：
  ```bash
  conda activate poyo
  cd /root/autodl-tmp/NeuroHorizon
  python scripts/analysis/phase0_baseline_plots.py
  ```
- **输入**：`results/logs/phase0_baseline/lightning_logs/version_0/metrics.csv`
- **输出**：`results/figures/baseline/03_baseline_training_curves.png`
- **依赖**：poyo conda 环境（matplotlib, numpy）
- **备注**：对应 plan.md 任务 0.3 可视化补充


### analyze_data_flow.py（0.3.4 数据流分析）

- **路径**：`scripts/analysis/analyze_data_flow.py`
- **功能用途**：Perich-Miller 数据在 POYO+ 中的数据流分析与可视化
  - 加载单个 session HDF5，提取所有时间尺度属性
  - 可视化时间轴上 domain、trials、movement_phases、train/valid/test_domain 的关系
  - 模拟训练/评估采样窗口的叠加效果
  - 分析 loss weights 和 eval_mask 的分配
- **创建时间**：2026-03-09
- **使用方式**：
  ```bash
  conda activate poyo
  cd /root/autodl-tmp/NeuroHorizon
  python scripts/analysis/analyze_data_flow.py
  ```
- **输入**：`data/processed/perich_miller_population_2018/*.h5`
- **输出**：
  - `results/figures/data_exploration/03_timescale_relationships.png`
  - `results/figures/data_exploration/04_sampling_windows_overlay.png`
  - `results/figures/data_exploration/05_eval_pipeline_flow.png`
  - `results/figures/data_exploration/data_flow_summary.json`
- **依赖**：poyo conda 环境（temporaldata, h5py, matplotlib, numpy）
- **备注**：对应 plan.md 任务 0.3.4


### analyze_nlb_benchmark.py（0.4.1 NLB 数据分析）

- **路径**：`scripts/analysis/analyze_nlb_benchmark.py`
- **功能用途**：NLB Benchmark 数据分析与适配性调查
  - Part A: NLB MC_Maze 数据结构分析（trial/unit/spike 统计、时间轴可视化）
  - Part B: 适配性调查（split 一致性检查、held-in/held-out 机制分析、其他子数据集评估）
  - Part C: NLB 指标对齐（co-bps, fp-bps, PSTH R² 实现难度评估）
- **创建时间**：2026-03-09
- **使用方式**：
  ```bash
  conda activate poyo
  cd /root/autodl-tmp/NeuroHorizon
  python scripts/analysis/analyze_nlb_benchmark.py
  ```
- **输入**：
  - `data/nlb/processed/pei_pandarinath_nlb_2021/jenkins_maze_train.h5`
  - `data/nlb/processed/pei_pandarinath_nlb_2021/jenkins_maze_test.h5`
- **输出**：
  - `results/figures/data_exploration/06_nlb_data_structure.png`
  - `results/figures/data_exploration/07_nlb_split_comparison.png`
  - `results/figures/data_exploration/nlb_analysis_summary.json`
- **依赖**：poyo conda 环境（h5py, matplotlib, numpy）
- **备注**：对应 plan.md 任务 0.4.1
## Phase 1：自回归改造验证 + 长时程生成验证

### train.py — NeuroHorizon 训练脚本（1.1.6）

- **路径**：`examples/neurohorizon/train.py`
- **功能用途**：NeuroHorizon 模型训练（PyTorch Lightning + Hydra 配置）
  - 支持自回归和非自回归两种模式
  - WandB 日志记录
  - 内联 R² 评估指标
- **创建时间**：2026-03-02
- **使用方式**：
  ```bash
  conda activate poyo
  cd /root/autodl-tmp/NeuroHorizon
  # 250ms（默认）
  python examples/neurohorizon/train.py --config-name=train_small
  # 500ms
  python examples/neurohorizon/train.py --config-name=train_small_500ms
  # 1000ms AR
  python examples/neurohorizon/train.py --config-name=train_small_1000ms
  # 1000ms non-AR
  python examples/neurohorizon/train.py --config-name=train_small_1000ms_noar
  ```
- **配置文件**：
  - `examples/neurohorizon/configs/train_small.yaml` — Small 配置（250ms，12 bins）
  - `examples/neurohorizon/configs/train_small_500ms.yaml` — 500ms（25 bins）
  - `examples/neurohorizon/configs/train_small_1000ms.yaml` — 1000ms AR（50 bins）
  - `examples/neurohorizon/configs/train_small_1000ms_noar.yaml` — 1000ms non-AR 对照
- **输出**：`results/logs/phase1_small_*/`（训练日志 + 检查点）
- **依赖**：poyo conda 环境（PyTorch, PyTorch Lightning, Hydra, wandb）
- **备注**：对应 plan.md 任务 1.1.6

### test_causal_mask.py（1.1.3 单元测试）

- **路径**：`scripts/tests/test_causal_mask.py`
- **功能用途**：Causal mask 单元测试
  - 验证 `create_causal_mask()` 正确生成下三角 mask
  - 验证 causal mask 在 RotarySelfAttention 中正确阻止未来信息泄露
- **创建时间**：2026-03-02
- **使用方式**：
  ```bash
  conda activate poyo
  cd /root/autodl-tmp/NeuroHorizon
  python scripts/tests/test_causal_mask.py
  ```
- **依赖**：poyo conda 环境
- **备注**：对应 plan.md 任务 1.1.3；从 `/tmp/neurohorizon_test_causal.py` 迁移

### test_decoder.py（1.1.4 单元测试）

- **路径**：`scripts/tests/test_decoder.py`
- **功能用途**：Decoder 单元测试
  - 验证 AutoregressiveDecoder 的 teacher forcing 和自回归推理模式
  - 验证输出维度正确、梯度可回传
- **创建时间**：2026-03-02
- **使用方式**：
  ```bash
  conda activate poyo
  cd /root/autodl-tmp/NeuroHorizon
  python scripts/tests/test_decoder.py
  ```
- **依赖**：poyo conda 环境
- **备注**：对应 plan.md 任务 1.1.4；从 `/tmp/neurohorizon_test_decoder.py` 迁移

### test_model.py（1.1.5 端到端测试）

- **路径**：`scripts/tests/test_model.py`
- **功能用途**：模型端到端测试
  - 验证 NeuroHorizon 完整 forward 链路（tokenize → encoder → decoder → loss）
  - 验证 teacher forcing 和自回归两种模式均可运行
  - 验证梯度回传到所有参数
- **创建时间**：2026-03-02
- **使用方式**：
  ```bash
  conda activate poyo
  cd /root/autodl-tmp/NeuroHorizon
  python scripts/tests/test_model.py
  ```
- **依赖**：poyo conda 环境
- **备注**：对应 plan.md 任务 1.1.5；从 `/tmp/neurohorizon_test_model.py` 迁移

### plot_optimization_training_curves.py（1.9 训练曲线可视化）

- **路径**：`scripts/phase1-autoregressive-1.9-module-optimization/plot_optimization_training_curves.py`
- **功能用途**：为 Phase 1.9 各轮模块优化补充 epoch-level training curves
  - 从各模块 summary JSON 反查正式 checkpoint 对应的 `metrics.csv`
  - 聚合 `train_loss` / `val_loss` / `val/fp_bps`
  - 生成每轮迭代各自的 `training_curves.{png,pdf}`
  - 在图中叠加 best-ckpt 的 teacher-forced / rollout valid-test 参考线
- **创建时间**：2026-03-19
- **使用方式**：
  ```bash
  conda activate poyo
  cd /root/autodl-tmp/NeuroHorizon
  python scripts/phase1-autoregressive-1.9-module-optimization/plot_optimization_training_curves.py
  python scripts/phase1-autoregressive-1.9-module-optimization/plot_optimization_training_curves.py --module 20260313_prediction_memory_alignment_tuning
  ```
- **输入**：
  - `results/figures/phase1-autoregressive-1.9-module-optimization/*/*_summary.json`
  - `results/logs/phase1-autoregressive-1.9-module-optimization/*/*/lightning_logs/version_*/metrics.csv`
- **输出**：
  - `results/figures/phase1-autoregressive-1.9-module-optimization/*/training_curves.png`
  - `results/figures/phase1-autoregressive-1.9-module-optimization/*/training_curves.pdf`
- **依赖**：poyo conda 环境（pandas, matplotlib）
- **备注**：2026-03-20 后改为从 best-ckpt summary JSON 读取 valid/test 参考线，用于补齐并统一 1.9 的训练曲线记录

### module_result_utils.py（1.9 结果汇总公用工具）

- **路径**：`scripts/phase1-autoregressive-1.9-module-optimization/module_result_utils.py`
- **功能用途**：为四轮 1.9 模块实验提供统一的结果汇总逻辑
  - 读取 best-ckpt `eval_teacher_forced_best_{valid,test}.json` 与 `eval_rollout_best_{valid,test}.json`
  - 从 `metrics.csv` 提取曲线最佳 epoch / 最后一轮指标
  - 统一生成各模块 `*_summary.json`
  - 统一更新 `results.tsv`、趋势图和 training curves
- **创建时间**：2026-03-20
- **使用方式**：由各模块 `collect_*_results.py` 间接调用
- **输入**：
  - `results/logs/phase1-autoregressive-1.9-module-optimization/*/*/eval_*_best_{valid,test}.json`
  - `results/logs/phase1-autoregressive-1.9-module-optimization/*/*/lightning_logs/version_*/metrics.csv`
  - `cc_todo/phase1-autoregressive/1.9-module-optimization/results.tsv`
- **输出**：
  - `results/figures/phase1-autoregressive-1.9-module-optimization/*/*_summary.json`
  - `cc_todo/phase1-autoregressive/1.9-module-optimization/results.tsv`
  - `results/figures/phase1-autoregressive-1.9-module-optimization/optimization_progress.{png,pdf}`
- **依赖**：poyo conda 环境（pandas, torch）
- **备注**：用于避免四个 1.9 汇总脚本各自维护不同的结果口径

### backfill_best_checkpoint_evals.py（1.9 历史 best-ckpt 回填）

- **路径**：`scripts/phase1-autoregressive-1.9-module-optimization/backfill_best_checkpoint_evals.py`
- **功能用途**：为历史 1.9 四轮实验补跑 best-ckpt 的 valid/test 正式评估
  - 按模块和窗口调用 `eval_phase1_v2.py --checkpoint-kind best`
  - 输出 teacher-forced / rollout 的 valid/test JSON
  - 调用各模块汇总脚本刷新 summary JSON、`results.tsv` 和图表
- **创建时间**：2026-03-20
- **使用方式**：
  ```bash
  conda activate poyo
  cd /root/autodl-tmp/NeuroHorizon
  python scripts/phase1-autoregressive-1.9-module-optimization/backfill_best_checkpoint_evals.py
  ```
- **输出**：
  - `results/logs/phase1-autoregressive-1.9-module-optimization/*/*/eval_*_best_{valid,test}.json`
  - `results/figures/phase1-autoregressive-1.9-module-optimization/*/*_summary.json`
  - `cc_todo/phase1-autoregressive/1.9-module-optimization/results.tsv`
- **依赖**：poyo conda 环境
- **备注**：用于 2026-03-20 协议审计后的历史结果补查，不替代未来正式 run 脚本

### verify_prediction_memory.py（1.9 Structured Prediction Memory 功能验证）

- **路径**：`scripts/phase1-autoregressive-1.9-module-optimization/20260312_prediction_memory_decoder/verify_prediction_memory.py`
- **功能用途**：对新的 prediction-memory decoder 做最小功能验证
  - 检查 `PredictionMemoryEncoder` 输出 shape
  - 检查 masked units 不影响 memory pooling
  - 检查 teacher forcing 与 rollout 不再数值等价
  - 检查 `shift-right` 逻辑是否只影响未来 bins
- **创建时间**：2026-03-12
- **使用方式**：
  ```bash
  conda activate poyo
  cd /root/autodl-tmp/NeuroHorizon
  python scripts/phase1-autoregressive-1.9-module-optimization/20260312_prediction_memory_decoder/verify_prediction_memory.py
  ```
- **输出**：控制台验证结果
- **依赖**：poyo conda 环境（PyTorch, torch_brain）
- **备注**：对应 plan.md 1.9.2；用于 prediction-memory decoder 改造后的首轮功能回归

### run_prediction_memory_experiments.sh（1.9 Structured Prediction Memory 批量实验）

- **路径**：`scripts/phase1-autoregressive-1.9-module-optimization/20260312_prediction_memory_decoder/run_prediction_memory_experiments.sh`
- **功能用途**：并行执行 prediction-memory decoder 的 1.9 实验流程
  - 先运行功能验证
  - 并行训练 250ms / 500ms / 1000ms 三个配置
  - 对每个窗口输出 best-ckpt teacher-forced / rollout 的 valid/test 四套评估结果
  - 最后调用汇总脚本生成 summary
- **创建时间**：2026-03-12
- **使用方式**：
  ```bash
  conda activate poyo
  cd /root/autodl-tmp/NeuroHorizon
  bash scripts/phase1-autoregressive-1.9-module-optimization/20260312_prediction_memory_decoder/run_prediction_memory_experiments.sh
  ```
- **输入**：
  - `examples/neurohorizon/configs/train_1p9_prediction_memory_{250ms,500ms,1000ms}.yaml`
  - `scripts/analysis/neurohorizon/eval_phase1_v2.py`
- **输出**：
  - `results/logs/phase1-autoregressive-1.9-module-optimization/20260312_prediction_memory_decoder/{250ms,500ms,1000ms}/`
  - `results/figures/phase1-autoregressive-1.9-module-optimization/20260312_prediction_memory_decoder/prediction_memory_summary.json`
- **依赖**：poyo conda 环境
- **备注**：只覆盖 1.9 必做的连续预测窗口实验，不包含 trial-aligned sweep

### monitor_prediction_memory_progress.py（1.9 Structured Prediction Memory 进度监控）

- **路径**：`scripts/phase1-autoregressive-1.9-module-optimization/20260312_prediction_memory_decoder/monitor_prediction_memory_progress.py`
- **功能用途**：周期性读取三个 prediction-memory 实验的 `metrics.csv` 和 pid 文件，估计当前 epoch 与剩余时间，并写出状态快照
- **创建时间**：2026-03-12
- **使用方式**：
  ```bash
  conda activate poyo
  cd /root/autodl-tmp/NeuroHorizon
  python scripts/phase1-autoregressive-1.9-module-optimization/20260312_prediction_memory_decoder/monitor_prediction_memory_progress.py --interval-sec 600
  ```
- **输入**：
  - `results/logs/phase1-autoregressive-1.9-module-optimization/20260312_prediction_memory_decoder/*/lightning_logs/version_*/metrics.csv`
  - `results/logs/phase1-autoregressive-1.9-module-optimization/20260312_prediction_memory_decoder/*/job.pid`
- **输出**：
  - `results/logs/phase1-autoregressive-1.9-module-optimization/20260312_prediction_memory_decoder/progress_status.md`
  - `results/logs/phase1-autoregressive-1.9-module-optimization/20260312_prediction_memory_decoder/progress_monitor.log`
- **依赖**：Python 标准库
- **备注**：用于长期后台实验的 ETA 跟踪，不修改训练结果本身

### collect_prediction_memory_results.py（1.9 Structured Prediction Memory 结果汇总）

- **路径**：`scripts/phase1-autoregressive-1.9-module-optimization/20260312_prediction_memory_decoder/collect_prediction_memory_results.py`
- **功能用途**：汇总 prediction-memory decoder 的评估 JSON，并自动完成 1.9 收尾
  - 读取 best-ckpt teacher-forced / rollout 的 valid/test 评估结果
  - 与 `baseline_v2` 做对比并生成 summary JSON
  - 追加或更新 `results.tsv`
  - 调用 `plot_optimization_progress.py` 刷新趋势图
  - 调用 `plot_optimization_training_curves.py` 刷新训练曲线
- **创建时间**：2026-03-12
- **使用方式**：
  ```bash
  conda activate poyo
  cd /root/autodl-tmp/NeuroHorizon
  python scripts/phase1-autoregressive-1.9-module-optimization/20260312_prediction_memory_decoder/collect_prediction_memory_results.py
  ```
- **输入**：
  - `results/logs/phase1-autoregressive-1.9-module-optimization/20260312_prediction_memory_decoder/*/eval_*_best_{valid,test}.json`
  - `cc_todo/phase1-autoregressive/1.9-module-optimization/results.tsv`
- **输出**：
  - `results/figures/phase1-autoregressive-1.9-module-optimization/20260312_prediction_memory_decoder/prediction_memory_summary.json`
  - `cc_todo/phase1-autoregressive/1.9-module-optimization/results.tsv`
  - `results/figures/phase1-autoregressive-1.9-module-optimization/optimization_progress.{png,pdf}`
- **依赖**：Python 标准库
- **备注**：`results.tsv` 中 `fp_bps_*` 记录 best-ckpt rollout-valid，`best_val_fp_bps_* / best_test_fp_bps_*` 记录 best-ckpt teacher-forced valid/test

### verify_local_prediction_memory.py（1.9 Local Prediction Memory 功能验证）

- **路径**：`scripts/phase1-autoregressive-1.9-module-optimization/20260313_local_prediction_memory/verify_local_prediction_memory.py`
- **功能用途**：验证 `local_prediction_memory` decoder 的最小正确性
  - 检查 local-only prediction memory mask 是否只允许访问对应 block
  - 检查 `shift-right` 仍然生效
  - 检查 `forward()` 与 `generate()` 不再等价
- **创建时间**：2026-03-13
- **使用方式**：
  ```bash
  conda activate poyo
  cd /root/autodl-tmp/NeuroHorizon
  python scripts/phase1-autoregressive-1.9-module-optimization/20260313_local_prediction_memory/verify_local_prediction_memory.py
  ```
- **输出**：控制台验证结果
- **依赖**：poyo conda 环境（PyTorch, torch_brain）
- **备注**：对应 plan.md 1.9.2 的 `20260313_local_prediction_memory`

### run_local_prediction_memory_smoke.sh（1.9 Local Prediction Memory smoke）

- **路径**：`scripts/phase1-autoregressive-1.9-module-optimization/20260313_local_prediction_memory/run_local_prediction_memory_smoke.sh`
- **功能用途**：执行 local prediction memory 版本的最小 smoke 流程
  - 先跑功能验证
  - 再跑 250ms 1-epoch smoke train
  - 最后跑 rollout smoke eval
- **创建时间**：2026-03-13
- **使用方式**：
  ```bash
  conda activate poyo
  cd /root/autodl-tmp/NeuroHorizon
  bash scripts/phase1-autoregressive-1.9-module-optimization/20260313_local_prediction_memory/run_local_prediction_memory_smoke.sh
  ```
- **输入**：
  - `examples/neurohorizon/configs/train_1p9_local_prediction_memory_250ms.yaml`
  - `scripts/analysis/neurohorizon/eval_phase1_v2.py`
- **输出**：
  - `results/logs/phase1-autoregressive-1.9-module-optimization/20260313_local_prediction_memory/250ms/`
  - `eval_rollout_smoke.json`
- **依赖**：poyo conda 环境
- **备注**：用于第二轮 1.9 架构修正的最小可运行验证

### run_local_prediction_memory_experiments.sh（1.9 Local Prediction Memory 批量实验）

- **路径**：`scripts/phase1-autoregressive-1.9-module-optimization/20260313_local_prediction_memory/run_local_prediction_memory_experiments.sh`
- **功能用途**：并行执行 local prediction memory 版本的 1.9 正式实验流程
  - 先运行功能验证
  - 并行训练 250ms / 500ms / 1000ms 三个配置
  - 对每个窗口输出 best-ckpt teacher-forced / rollout 的 valid/test 四套评估结果
  - 最后调用汇总脚本生成 summary
- **创建时间**：2026-03-13
- **使用方式**：
  ```bash
  conda activate poyo
  cd /root/autodl-tmp/NeuroHorizon
  bash scripts/phase1-autoregressive-1.9-module-optimization/20260313_local_prediction_memory/run_local_prediction_memory_experiments.sh
  ```
- **输入**：
  - `examples/neurohorizon/configs/train_1p9_local_prediction_memory_{250ms,500ms,1000ms}.yaml`
  - `scripts/analysis/neurohorizon/eval_phase1_v2.py`
- **输出**：
  - `results/logs/phase1-autoregressive-1.9-module-optimization/20260313_local_prediction_memory/{250ms,500ms,1000ms}/`
  - `results/figures/phase1-autoregressive-1.9-module-optimization/20260313_local_prediction_memory/local_prediction_memory_summary.json`
- **依赖**：poyo conda 环境
- **备注**：与 20260312 版本保持相同实验协议，便于直接对比 rollout 稳定性

### monitor_local_prediction_memory_progress.py（1.9 Local Prediction Memory 进度监控）

- **路径**：`scripts/phase1-autoregressive-1.9-module-optimization/20260313_local_prediction_memory/monitor_local_prediction_memory_progress.py`
- **功能用途**：周期性读取三个 local prediction memory 实验的 `metrics.csv` 和 pid 文件，估计当前 epoch 与剩余时间，并写出状态快照
- **创建时间**：2026-03-13
- **使用方式**：
  ```bash
  conda activate poyo
  cd /root/autodl-tmp/NeuroHorizon
  python scripts/phase1-autoregressive-1.9-module-optimization/20260313_local_prediction_memory/monitor_local_prediction_memory_progress.py --interval-sec 600
  ```
- **输入**：
  - `results/logs/phase1-autoregressive-1.9-module-optimization/20260313_local_prediction_memory/*/lightning_logs/version_*/metrics.csv`
  - `results/logs/phase1-autoregressive-1.9-module-optimization/20260313_local_prediction_memory/*/job.pid`
- **输出**：
  - `results/logs/phase1-autoregressive-1.9-module-optimization/20260313_local_prediction_memory/progress_status.md`
  - `results/logs/phase1-autoregressive-1.9-module-optimization/20260313_local_prediction_memory/progress_monitor.log`
- **依赖**：Python 标准库
- **备注**：用于正式对比实验期间的 ETA 跟踪，不修改训练结果本身

### collect_local_prediction_memory_results.py（1.9 Local Prediction Memory 结果汇总）

- **路径**：`scripts/phase1-autoregressive-1.9-module-optimization/20260313_local_prediction_memory/collect_local_prediction_memory_results.py`
- **功能用途**：汇总 local prediction memory 的评估 JSON，并自动完成 1.9 收尾
  - 读取 best-ckpt teacher-forced / rollout 的 valid/test 评估结果
  - 与 `baseline_v2` 做对比并生成 summary JSON
  - 追加或更新 `results.tsv`
  - 调用 `plot_optimization_progress.py` 刷新趋势图
  - 调用 `plot_optimization_training_curves.py` 刷新训练曲线
- **创建时间**：2026-03-13
- **使用方式**：
  ```bash
  conda activate poyo
  cd /root/autodl-tmp/NeuroHorizon
  python scripts/phase1-autoregressive-1.9-module-optimization/20260313_local_prediction_memory/collect_local_prediction_memory_results.py
  ```
- **输入**：
  - `results/logs/phase1-autoregressive-1.9-module-optimization/20260313_local_prediction_memory/*/eval_*_best_{valid,test}.json`
  - `cc_todo/phase1-autoregressive/1.9-module-optimization/results.tsv`
- **输出**：
  - `results/figures/phase1-autoregressive-1.9-module-optimization/20260313_local_prediction_memory/local_prediction_memory_summary.json`
  - `cc_todo/phase1-autoregressive/1.9-module-optimization/results.tsv`
  - `results/figures/phase1-autoregressive-1.9-module-optimization/optimization_progress.{png,pdf}`
- **依赖**：Python 标准库
- **备注**：`results.tsv` 中 `fp_bps_*` 记录 best-ckpt rollout-valid，`best_val_fp_bps_* / best_test_fp_bps_*` 记录 best-ckpt teacher-forced valid/test

### verify_prediction_memory_alignment.py（1.9 Alignment 功能验证）

- **路径**：`scripts/phase1-autoregressive-1.9-module-optimization/20260313_prediction_memory_alignment/verify_prediction_memory_alignment.py`
- **功能用途**：验证 mixed-memory 训练与 memory-input regularization 两个核心机制
  - `mix_prob=1.0` 时训练态 forward 不再依赖 GT target counts
  - train-time memory noise/dropout 会实际扰动 memory tokens
- **创建时间**：2026-03-13
- **使用方式**：
  ```bash
  conda activate poyo
  cd /root/autodl-tmp/NeuroHorizon
  python scripts/phase1-autoregressive-1.9-module-optimization/20260313_prediction_memory_alignment/verify_prediction_memory_alignment.py
  ```
- **输出**：终端打印机制验证结果
- **依赖**：poyo conda 环境
- **备注**：对应 plan.md 1.9.2 的 `20260313_prediction_memory_alignment`

### verify_prediction_memory_alignment_tuning.py（1.9 Alignment Tuning 功能验证）

- **路径**：`scripts/phase1-autoregressive-1.9-module-optimization/20260313_prediction_memory_alignment_tuning/verify_prediction_memory_alignment_tuning.py`
- **功能用途**：验证 tuning 版三项超参已正确生效，并复用上一轮 mixed-memory / regularization 机制验证
- **创建时间**：2026-03-13
- **使用方式**：
  ```bash
  conda activate poyo
  cd /root/autodl-tmp/NeuroHorizon
  python scripts/phase1-autoregressive-1.9-module-optimization/20260313_prediction_memory_alignment_tuning/verify_prediction_memory_alignment_tuning.py
  ```
- **输出**：终端打印 tuning 超参值与机制验证结果
- **依赖**：poyo conda 环境
- **备注**：对应 plan.md 1.9.2 的 `20260313_prediction_memory_alignment_tuning`

### run_prediction_memory_alignment_tuning_smoke.sh（1.9 Alignment Tuning Smoke Run）

- **路径**：`scripts/phase1-autoregressive-1.9-module-optimization/20260313_prediction_memory_alignment_tuning/run_prediction_memory_alignment_tuning_smoke.sh`
- **功能用途**：执行 tuning 方案的最小链路验证
  - 运行 tuning 功能验证脚本
  - 跑 250ms 1-epoch smoke train
  - 跑 250ms rollout smoke eval
- **创建时间**：2026-03-13
- **使用方式**：
  ```bash
  conda activate poyo
  cd /root/autodl-tmp/NeuroHorizon
  bash scripts/phase1-autoregressive-1.9-module-optimization/20260313_prediction_memory_alignment_tuning/run_prediction_memory_alignment_tuning_smoke.sh
  ```
- **输出**：
  - `results/logs/phase1-autoregressive-1.9-module-optimization/20260313_prediction_memory_alignment_tuning/250ms/`
- **依赖**：poyo conda 环境
- **备注**：Step 2 checkpoint 前的最小可运行性验证

### run_prediction_memory_alignment_tuning_experiments.sh（1.9 Alignment Tuning 正式实验）

- **路径**：`scripts/phase1-autoregressive-1.9-module-optimization/20260313_prediction_memory_alignment_tuning/run_prediction_memory_alignment_tuning_experiments.sh`
- **功能用途**：并行启动 `250ms / 500ms / 1000ms` 三个 tuning 正式实验，并在训练完成后自动执行 best-ckpt teacher-forced / rollout valid-test eval 与结果收集
- **创建时间**：2026-03-13
- **使用方式**：
  ```bash
  conda activate poyo
  cd /root/autodl-tmp/NeuroHorizon
  bash scripts/phase1-autoregressive-1.9-module-optimization/20260313_prediction_memory_alignment_tuning/run_prediction_memory_alignment_tuning_experiments.sh
  ```
- **输出**：
  - `results/logs/phase1-autoregressive-1.9-module-optimization/20260313_prediction_memory_alignment_tuning/{250ms,500ms,1000ms}/`
  - `results/figures/phase1-autoregressive-1.9-module-optimization/20260313_prediction_memory_alignment_tuning/prediction_memory_alignment_tuning_summary.json`
- **依赖**：poyo conda 环境
- **备注**：沿用 1.9 统一三窗口协议

### monitor_prediction_memory_alignment_tuning_progress.py（1.9 Alignment Tuning 进度监控）

- **路径**：`scripts/phase1-autoregressive-1.9-module-optimization/20260313_prediction_memory_alignment_tuning/monitor_prediction_memory_alignment_tuning_progress.py`
- **功能用途**：周期性读取三窗口 tuning 实验的 `metrics.csv` 和 `job.pid`，估算每个窗口的 epoch 进度与 ETA，并写入统一 `progress_status.md`
- **创建时间**：2026-03-13
- **使用方式**：
  ```bash
  conda activate poyo
  cd /root/autodl-tmp/NeuroHorizon
  python scripts/phase1-autoregressive-1.9-module-optimization/20260313_prediction_memory_alignment_tuning/monitor_prediction_memory_alignment_tuning_progress.py --interval-sec 600
  ```
- **输入**：
  - `results/logs/phase1-autoregressive-1.9-module-optimization/20260313_prediction_memory_alignment_tuning/*/lightning_logs/version_*/metrics.csv`
  - `results/logs/phase1-autoregressive-1.9-module-optimization/20260313_prediction_memory_alignment_tuning/*/job.pid`
- **输出**：
  - `results/logs/phase1-autoregressive-1.9-module-optimization/20260313_prediction_memory_alignment_tuning/progress_status.md`
  - `results/logs/phase1-autoregressive-1.9-module-optimization/20260313_prediction_memory_alignment_tuning/progress_monitor.log`
- **依赖**：Python 标准库
- **备注**：用于正式对比实验期间的 ETA 跟踪，不修改训练结果本身

### collect_prediction_memory_alignment_tuning_results.py（1.9 Alignment Tuning 结果汇总）

- **路径**：`scripts/phase1-autoregressive-1.9-module-optimization/20260313_prediction_memory_alignment_tuning/collect_prediction_memory_alignment_tuning_results.py`
- **功能用途**：汇总 tuning 版评估 JSON，并自动完成 1.9 收尾
  - 读取 best-ckpt teacher-forced / rollout 的 valid/test 评估结果
  - 与 `baseline_v2` 做对比并生成 summary JSON
  - 追加或更新 `results.tsv`
  - 调用 `plot_optimization_progress.py` 刷新趋势图
  - 调用 `plot_optimization_training_curves.py` 刷新训练曲线
- **创建时间**：2026-03-13
- **使用方式**：
  ```bash
  conda activate poyo
  cd /root/autodl-tmp/NeuroHorizon
  python scripts/phase1-autoregressive-1.9-module-optimization/20260313_prediction_memory_alignment_tuning/collect_prediction_memory_alignment_tuning_results.py
  ```
- **输入**：
  - `results/logs/phase1-autoregressive-1.9-module-optimization/20260313_prediction_memory_alignment_tuning/*/eval_*_best_{valid,test}.json`
  - `cc_todo/phase1-autoregressive/1.9-module-optimization/results.tsv`
- **输出**：
  - `results/figures/phase1-autoregressive-1.9-module-optimization/20260313_prediction_memory_alignment_tuning/prediction_memory_alignment_tuning_summary.json`
  - `cc_todo/phase1-autoregressive/1.9-module-optimization/results.tsv`
  - `results/figures/phase1-autoregressive-1.9-module-optimization/optimization_progress.{png,pdf}`
- **依赖**：Python 标准库
- **备注**：`results.tsv` 中 `fp_bps_*` 记录 best-ckpt rollout-valid，`best_val_fp_bps_* / best_test_fp_bps_*` 记录 best-ckpt teacher-forced valid/test

### run_prediction_memory_alignment_smoke.sh（1.9 Alignment Smoke Run）

- **路径**：`scripts/phase1-autoregressive-1.9-module-optimization/20260313_prediction_memory_alignment/run_prediction_memory_alignment_smoke.sh`
- **功能用途**：执行 alignment 方案的最小链路验证
  - 运行功能验证脚本
  - 跑 250ms 1-epoch smoke train
  - 跑 250ms rollout smoke eval
- **创建时间**：2026-03-13
- **使用方式**：
  ```bash
  conda activate poyo
  cd /root/autodl-tmp/NeuroHorizon
  bash scripts/phase1-autoregressive-1.9-module-optimization/20260313_prediction_memory_alignment/run_prediction_memory_alignment_smoke.sh
  ```
- **输出**：
  - `results/logs/phase1-autoregressive-1.9-module-optimization/20260313_prediction_memory_alignment/250ms/`
- **依赖**：poyo conda 环境
- **备注**：Step 2 checkpoint 前的最小可运行性验证

### run_prediction_memory_alignment_experiments.sh（1.9 Alignment 正式实验）

- **路径**：`scripts/phase1-autoregressive-1.9-module-optimization/20260313_prediction_memory_alignment/run_prediction_memory_alignment_experiments.sh`
- **功能用途**：并行启动 `250ms / 500ms / 1000ms` 三个正式 alignment 实验，并在训练完成后自动执行 best-ckpt teacher-forced / rollout valid-test eval 与结果收集
- **创建时间**：2026-03-13
- **使用方式**：
  ```bash
  conda activate poyo
  cd /root/autodl-tmp/NeuroHorizon
  bash scripts/phase1-autoregressive-1.9-module-optimization/20260313_prediction_memory_alignment/run_prediction_memory_alignment_experiments.sh
  ```
- **输出**：
  - `results/logs/phase1-autoregressive-1.9-module-optimization/20260313_prediction_memory_alignment/{250ms,500ms,1000ms}/`
  - `results/figures/phase1-autoregressive-1.9-module-optimization/20260313_prediction_memory_alignment/prediction_memory_alignment_summary.json`
- **依赖**：poyo conda 环境
- **备注**：沿用 1.9 统一三窗口协议

### monitor_prediction_memory_alignment_progress.py（1.9 Alignment 进度监控）

- **路径**：`scripts/phase1-autoregressive-1.9-module-optimization/20260313_prediction_memory_alignment/monitor_prediction_memory_alignment_progress.py`
- **功能用途**：周期性读取三窗口实验的 `metrics.csv` 和 `job.pid`，估算每个窗口的 epoch 进度与 ETA，并写入统一 `progress_status.md`
- **创建时间**：2026-03-13
- **使用方式**：
  ```bash
  conda activate poyo
  cd /root/autodl-tmp/NeuroHorizon
  python scripts/phase1-autoregressive-1.9-module-optimization/20260313_prediction_memory_alignment/monitor_prediction_memory_alignment_progress.py --interval-sec 600
  ```
- **输入**：
  - `results/logs/phase1-autoregressive-1.9-module-optimization/20260313_prediction_memory_alignment/*/lightning_logs/version_*/metrics.csv`
  - `results/logs/phase1-autoregressive-1.9-module-optimization/20260313_prediction_memory_alignment/*/job.pid`
- **输出**：
  - `results/logs/phase1-autoregressive-1.9-module-optimization/20260313_prediction_memory_alignment/progress_status.md`
  - `results/logs/phase1-autoregressive-1.9-module-optimization/20260313_prediction_memory_alignment/progress_monitor.log`
- **依赖**：Python 标准库
- **备注**：用于正式对比实验期间的 ETA 跟踪，不修改训练结果本身

### collect_prediction_memory_alignment_results.py（1.9 Alignment 结果汇总）

- **路径**：`scripts/phase1-autoregressive-1.9-module-optimization/20260313_prediction_memory_alignment/collect_prediction_memory_alignment_results.py`
- **功能用途**：汇总 prediction memory alignment 的评估 JSON，并自动完成 1.9 收尾
  - 读取 best-ckpt teacher-forced / rollout 的 valid/test 评估结果
  - 与 `baseline_v2` 做对比并生成 summary JSON
  - 追加或更新 `results.tsv`
  - 调用 `plot_optimization_progress.py` 刷新趋势图
  - 调用 `plot_optimization_training_curves.py` 刷新训练曲线
- **创建时间**：2026-03-13
- **使用方式**：
  ```bash
  conda activate poyo
  cd /root/autodl-tmp/NeuroHorizon
  python scripts/phase1-autoregressive-1.9-module-optimization/20260313_prediction_memory_alignment/collect_prediction_memory_alignment_results.py
  ```
- **输入**：
  - `results/logs/phase1-autoregressive-1.9-module-optimization/20260313_prediction_memory_alignment/*/eval_*_best_{valid,test}.json`
  - `cc_todo/phase1-autoregressive/1.9-module-optimization/results.tsv`
- **输出**：
  - `results/figures/phase1-autoregressive-1.9-module-optimization/20260313_prediction_memory_alignment/prediction_memory_alignment_summary.json`
  - `cc_todo/phase1-autoregressive/1.9-module-optimization/results.tsv`
  - `results/figures/phase1-autoregressive-1.9-module-optimization/optimization_progress.{png,pdf}`
- **依赖**：Python 标准库
- **备注**：`results.tsv` 中 `fp_bps_*` 记录 best-ckpt rollout-valid，`best_val_fp_bps_* / best_test_fp_bps_*` 记录 best-ckpt teacher-forced valid/test

### ar_verify.py（1.2.2 AR 推理验证）

- **路径**：`scripts/analysis/neurohorizon/ar_verify.py`
- **功能用途**：AR 推理验证（Teacher Forcing vs Autoregressive 对比 + causal mask 检验）
  - 加载训练好的模型，同一数据分别用 TF 和 AR 推理
  - 逐 bin 对比 R² 和 NLL，验证两者等价性
  - 检验 causal mask 正确阻止未来信息泄露
- **创建时间**：2026-03-02
- **使用方式**：
  ```bash
  conda activate poyo
  cd /root/autodl-tmp/NeuroHorizon
  python scripts/analysis/neurohorizon/ar_verify.py
  ```
- **输出**：`results/logs/phase1_small_250ms/ar_verify_results.json`
- **依赖**：poyo conda 环境
- **备注**：对应 plan.md 任务 1.2.2；从 `/tmp/nh_ar_verify.py` 迁移

### full_report.py（1.3.4 实验报告生成）

- **路径**：`scripts/analysis/neurohorizon/full_report.py`
- **功能用途**：完整实验报告生成
  - 遍历所有 Phase 1 实验的 TensorBoard 日志
  - 提取每个实验的 val_loss、R² 随 epoch 变化的完整数据
  - 输出 JSON 格式的完整报告
- **创建时间**：2026-03-02
- **使用方式**：
  ```bash
  conda activate poyo
  cd /root/autodl-tmp/NeuroHorizon
  python scripts/analysis/neurohorizon/full_report.py
  ```
- **输出**：`results/logs/phase1_full_report.json`
- **依赖**：poyo conda 环境（tensorboard）
- **备注**：对应 plan.md 任务 1.3.4；从 `/tmp/nh_full_report.py` 迁移

### summary.py（1.3.4 快速汇总）

- **路径**：`scripts/analysis/neurohorizon/summary.py`
- **功能用途**：快速实验汇总
  - 提取各实验的关键指标（best R²、best epoch、final val loss）
  - 输出简洁的 JSON 汇总
- **创建时间**：2026-03-02
- **使用方式**：
  ```bash
  conda activate poyo
  cd /root/autodl-tmp/NeuroHorizon
  python scripts/analysis/neurohorizon/summary.py
  ```
- **输出**：`results/logs/phase1_summary.json`
- **依赖**：poyo conda 环境（tensorboard）
- **备注**：对应 plan.md 任务 1.3.4；从 `/tmp/nh_summary.py` 迁移

### noise_floor.py（1.3.4 Noise floor 分析）

- **路径**：`scripts/analysis/neurohorizon/noise_floor.py`
- **功能用途**：Poisson noise floor 分析
  - 计算 spike count 数据的理论 Poisson noise floor
  - 评估模型性能相对 noise floor 的位置
- **创建时间**：2026-03-02
- **使用方式**：
  ```bash
  conda activate poyo
  cd /root/autodl-tmp/NeuroHorizon
  python scripts/analysis/neurohorizon/noise_floor.py
  ```
- **依赖**：poyo conda 环境
- **备注**：对应 plan.md 任务 1.3.4；从 `/tmp/nh_noise_floor.py` 迁移
### phase1_visualize.py (1.2/1.3 可视化补充)

- **路径**：`scripts/analysis/neurohorizon/phase1_visualize.py`
- **功能用途**：Phase 1 全部实验可视化（4 张图）
  - 01: 4 组实验训练曲线（Val Loss + R² vs Epoch）
  - 02: R² 随预测窗口长度衰减（+ AR vs non-AR 对比）
  - 03: 逐 bin R² 和 NLL 分析（250ms，12 bins）
  - 04: AR vs non-AR 详细对比（1000ms 窗口）
- **创建时间**：2026-03-02
- **使用方式**：
  ```bash
  conda activate poyo
  cd /root/autodl-tmp/NeuroHorizon
  python scripts/analysis/neurohorizon/phase1_visualize.py
  ```
- **输入**：
  - `results/logs/phase1_full_report.json`
  - `results/logs/phase1_small_250ms/ar_verify_results.json`
- **输出**：
  - `results/figures/phase1/01_training_curves.png`
  - `results/figures/phase1/02_r2_vs_window.png`
  - `results/figures/phase1/03_per_bin_r2.png`
  - `results/figures/phase1/04_ar_vs_noar.png`
- **依赖**：poyo conda 环境（matplotlib, numpy）
- **备注**：对应 plan.md 任务 1.2/1.3 可视化补充


### neurohorizon_metrics.py（1.1.7 评估指标库）

- **路径**：`torch_brain/utils/neurohorizon_metrics.py`
- **功能用途**：NeuroHorizon 评估指标模块
  - fp-bps（Forward Prediction Bits Per Spike）：主要评估指标
  - 全局累计辅助：`fp_bps_stats()` / `finalize_fp_bps_from_stats()`
  - per-bin fp-bps：分析 AR 预测随时间步的衰减
  - `per_neuron_psth_r2`：Trial-averaged 神经元粒度预测质量
  - R²、Firing rate correlation、Poisson NLL
  - `r2_stats()` / `finalize_r2_from_stats()`：全局累计 R² 辅助
  - `compute_null_rates()`：计算训练集 global per-neuron mean null
  - `build_null_rate_lookup()`：构建全局单元 → null rate 查询表
- **创建时间**：2026-03-11
- **使用方式**：
  ```python
  from torch_brain.utils.neurohorizon_metrics import fp_bps, fp_bps_per_bin, compute_null_rates
  val = fp_bps(log_rate, target, null_log_rates, mask=None)  # scalar
  per_bin = fp_bps_per_bin(log_rate, target, null_log_rates)  # [T]
  ```
- **依赖**：torch
- **备注**：对应 plan.md 任务 1.1.7；非独立脚本，作为模块被其他脚本/训练代码引用

### eval_psth.py（1.1.7 PSTH-R² 评估脚本）

- **路径**：`scripts/analysis/neurohorizon/eval_psth.py`
- **功能用途**：PSTH-R² 独立评估脚本
  - 加载训练好的模型 + trial-aligned 数据
  - 按 `(session_id, target_id)` 分组计算 trial-averaged 神经活动
  - 输出主指标 `per_neuron_psth_r2` 与 `trial_fp_bps`
- **创建时间**：2026-03-11
- **使用方式**：
  ```bash
  conda activate poyo
  cd /root/autodl-tmp/NeuroHorizon
  python scripts/analysis/neurohorizon/eval_psth.py
  ```
- **依赖**：poyo conda 环境
- **备注**：对应 plan.md 任务 1.1.7；2026-03-17 起统一到 per-neuron 口径

### test_1_2_4_metrics_verification.py（1.2.4 指标与 Sampler 验证）

- **路径**：`scripts/tests/test_1_2_4_metrics_verification.py`
- **功能用途**：验证 fp-bps 和 Trial-Aligned Sampler 的正确性
  - Test 1-4：fp-bps 正确性（null=0, random<0, trained>0, per-bin shape）
  - Test 5-7：Trial-aligned sampler（HDF5 结构、窗口对齐、索引字段）
- **创建时间**：2026-03-11
- **使用方式**：
  ```bash
  conda activate poyo
  cd /root/autodl-tmp/NeuroHorizon
  python scripts/tests/test_1_2_4_metrics_verification.py
  ```
- **输入**：`data/processed/perich_miller_population_2018/*.h5`
- **依赖**：poyo conda 环境
- **备注**：对应 plan.md 任务 1.2.4；全部 8 项测试通过

### eval_phase1_v2.py（1.3.4 综合评估脚本）

- **路径**：`scripts/analysis/neurohorizon/eval_phase1_v2.py`
- **功能用途**：Phase 1 v2 综合评估脚本
  - 连续模式评估：SequentialFixedWindowSampler + 全局累计版 fp-bps（整体 + per-bin）、R-squared
  - Trial-aligned 评估：`per_neuron_psth_r2`（8 方向 per-target + overall）+ `trial_fp_bps`
  - 自动查找 checkpoint、计算 null rates、输出 JSON
- **创建时间**：2026-03-12
- **使用方式**：
  ````bash
  conda activate poyo
  cd /root/autodl-tmp/NeuroHorizon
  python scripts/analysis/neurohorizon/eval_phase1_v2.py --log-dir results/logs/phase1_v2_evalfix_250ms_cont --split valid
  python scripts/analysis/neurohorizon/eval_phase1_v2.py --log-dir results/logs/phase1_v2_evalfix_250ms_cont --split test
  ````
- **输入**：训练 checkpoint（last.ckpt），训练数据（用于 null rates）
- **输出**：`eval_v2_{valid,test}_results.json`（fp-bps, R2, per-bin fp-bps, `per_neuron_psth_r2`, per-target `per_neuron_psth_r2`）
- **依赖**：poyo conda 环境
- **备注**：对应 plan.md 任务 1.3.4；2026-03-17 起切换到全局累计评估协议

### phase1_v2_visualize.py（1.3.4 可视化脚本）

- **路径**：`scripts/analysis/neurohorizon/phase1_v2_visualize.py`
- **功能用途**：Phase 1 v2 实验可视化（5 张图）
  - Figure 1：fp-bps vs 预测窗口（连续 vs trial-aligned）
  - Figure 2：per-bin fp-bps 衰减曲线
  - Figure 3：`per_neuron_psth_r2` 热力图（8 方向 x 6 条件）
  - Figure 4：连续 vs trial-aligned 对比柱状图
  - Figure 5：训练曲线（val_loss + val_fp_bps vs epoch）
- **创建时间**：2026-03-12
- **使用方式**：
  ````bash
  conda activate poyo
  cd /root/autodl-tmp/NeuroHorizon
  python scripts/analysis/neurohorizon/phase1_v2_visualize.py
  ````
- **输出**：`results/figures/phase1_v2/*.png`（5 张图）
- **依赖**：poyo conda 环境, matplotlib
- **备注**：对应 plan.md 任务 1.3.4；兼容旧 `psth_r2` 和新 `per_neuron_psth_r2` JSON key

### run_phase1_v2_evalfix.sh（1.3.4 evalfix 全量重跑脚本）

- **路径**：`scripts/analysis/neurohorizon/run_phase1_v2_evalfix.sh`
- **功能用途**：顺序执行 Phase 1.3.4 的 6 个 evalfix 配置训练与 valid/test 离线评估
- **创建时间**：2026-03-17
- **使用方式**：
  ```bash
  conda activate poyo
  cd /root/autodl-tmp/NeuroHorizon
  bash scripts/analysis/neurohorizon/run_phase1_v2_evalfix.sh
  ```
- **输出**：`results/logs/phase1_v2_evalfix_{250ms,500ms,1000ms}_{cont,trial}/`
- **依赖**：poyo conda 环境
- **备注**：对应 `cc_todo/phase1-autoregressive/20260317-phase1-1.3.4-evalfix-rerun.md`

### compare_phase1_v2_evalfix.py（1.3.4 legacy vs evalfix 对比）

- **路径**：`scripts/analysis/neurohorizon/compare_phase1_v2_evalfix.py`
- **功能用途**：读取 legacy `phase1_v2_*` 和新 `phase1_v2_evalfix_*` 的 valid/test JSON，汇总对比表
- **创建时间**：2026-03-17
- **使用方式**：
  ```bash
  conda activate poyo
  cd /root/autodl-tmp/NeuroHorizon
  python scripts/analysis/neurohorizon/compare_phase1_v2_evalfix.py
  ```
- **输出**：`results/logs/phase1_v2_evalfix_comparison/comparison.{json,md}`
- **依赖**：poyo conda 环境
- **备注**：对应 `cc_todo/phase1-autoregressive/20260317-phase1-1.3.4-evalfix-rerun.md`；同时兼容 `lightning_logs/` 与 `lightning_logs/version_0/` 两种结果路径

### compare_phase1_v2_ibl_metric.py（1.3.5 IBL-MtM 风格指标对照）

- **路径**：`scripts/analysis/neurohorizon/compare_phase1_v2_ibl_metric.py`
- **功能用途**：汇总 `1.3.4 evalfix` continuous baseline 在 valid/test 上的 `fp-bps`、`ibl_mtm_bps`、`R²`
- **创建时间**：2026-03-18
- **使用方式**：
  ```bash
  conda activate poyo
  cd /root/autodl-tmp/NeuroHorizon
  python scripts/analysis/neurohorizon/compare_phase1_v2_ibl_metric.py
  ```
- **输出**：`results/logs/phase1_v2_metric_extension_comparison/comparison.{json,md}`
- **依赖**：poyo conda 环境
- **备注**：对应 `cc_todo/phase1-autoregressive/20260318-phase1-1.3.5-ibl-metric.md`

### run_phase1_v2_nocausal.sh（1.3.6 non-causal baseline 批量重跑）

- **路径**：`scripts/analysis/neurohorizon/run_phase1_v2_nocausal.sh`
- **功能用途**：顺序执行 `1.3.6` 的 3 个 non-causal continuous 配置训练与 valid/test 离线评估
- **创建时间**：2026-03-18
- **使用方式**：
  ```bash
  conda activate poyo
  cd /root/autodl-tmp/NeuroHorizon
  bash scripts/analysis/neurohorizon/run_phase1_v2_nocausal.sh
  ```
- **输出**：`results/logs/phase1_v2_nocausal_{250ms,500ms,1000ms}_cont/`
- **依赖**：poyo conda 环境
- **备注**：对应 `cc_todo/phase1-autoregressive/20260318-phase1-1.3.6-nocausal-ablation.md`

### compare_phase1_v2_causal_ablation.py（1.3.6 causal vs non-causal 对比）

- **路径**：`scripts/analysis/neurohorizon/compare_phase1_v2_causal_ablation.py`
- **功能用途**：对比 causal evalfix baseline 与 non-causal baseline 在 valid/test 上的 `fp-bps`、`ibl_mtm_bps` 等指标
- **创建时间**：2026-03-18
- **使用方式**：
  ```bash
  conda activate poyo
  cd /root/autodl-tmp/NeuroHorizon
  python scripts/analysis/neurohorizon/compare_phase1_v2_causal_ablation.py
  ```
- **输出**：`results/logs/phase1_v2_nocausal_comparison/comparison.{json,md}`
- **依赖**：poyo conda 环境
- **备注**：对应 `cc_todo/phase1-autoregressive/20260318-phase1-1.3.6-nocausal-ablation.md`

---

## Benchmark 对比实验脚本（1.8）

### benchmark_train.py（1.8.3 统一训练脚本）

- **路径**：`neural_benchmark/benchmark_train.py`
- **功能用途**：统一的 benchmark 模型训练脚本，支持 NDT2/IBL-MtM/Neuroformer 三个模型
  - 加载 Perich-Miller 10 sessions 数据（via torch_brain Dataset）
  - 20ms bin spike events → BenchmarkDataset
  - 训练 + 定期评估（fp-bps, R², Poisson NLL）
  - 保存 best_model.pt + results.json
- **创建时间**：2026-03-12
- **使用方式**：
  ```bash
  conda activate benchmark-env
  cd /root/autodl-tmp/NeuroHorizon
  python3 neural_benchmark/benchmark_train.py --model ndt2 --pred_window 0.25 --epochs 300
  python3 neural_benchmark/benchmark_train.py --model ibl_mtm --pred_window 0.5
  python3 neural_benchmark/benchmark_train.py --model neuroformer --pred_window 1.0
  ```
- **输出**：`results/logs/phase1_benchmark_{model}_{window}ms/`（best_model.pt + results.json）
- **依赖**：benchmark-env conda 环境, torch_brain
- **备注**：对应 plan.md 任务 1.8.3 Part B

### run_all_benchmarks.sh（1.8.3 批量训练）

- **路径**：`neural_benchmark/run_all_benchmarks.sh`
- **功能用途**：批量运行全部 9 个 benchmark 实验（3 模型 × 3 窗口）
- **创建时间**：2026-03-12
- **使用方式**：
  ```bash
  cd /root/autodl-tmp/NeuroHorizon
  nohup bash neural_benchmark/run_all_benchmarks.sh > results/logs/benchmark_all_runs.log 2>&1 &
  ```
- **输出**：同 benchmark_train.py
- **备注**：运行时间约 8-9 小时（300 epochs × 9 实验）

### visualize_benchmarks.py（1.8.3 对比可视化）

- **路径**：`neural_benchmark/visualize_benchmarks.py`
- **功能用途**：训练完成后生成 4 张对比图表
  - Figure 1：fp-bps 柱状图（3 模型 × 3 窗口）
  - Figure 2：per-bin fp-bps 衰减曲线
  - Figure 3：R² 柱状图
  - Figure 4：综合对比表 + 雷达图
- **创建时间**：2026-03-12
- **使用方式**：
  ```bash
  conda activate benchmark-env
  cd /root/autodl-tmp/NeuroHorizon
  python3 neural_benchmark/visualize_benchmarks.py
  ```
- **输出**：`results/figures/phase1_benchmark/*.png`
- **备注**：可自动检测并加载 NeuroHorizon 1.3.4 结果进行对比

### phase1_benchmark_compare.py（1.3.4 Benchmark 对比可视化）

- **路径**：`scripts/analysis/neurohorizon/phase1_benchmark_compare.py`
- **功能用途**：1.3.4 NeuroHorizon vs Benchmark 模型对比可视化
  - 读取 NeuroHorizon v2 eval 结果和 1.8.3 benchmark results.json
  - 生成分组柱状图 + 折线图（4 模型 × 3 窗口）
  - 输出相对优势百分比
- **创建时间**：2026-03-12
- **使用方式**：
  ````bash
  conda activate poyo
  cd /root/autodl-tmp/NeuroHorizon
  python scripts/analysis/neurohorizon/phase1_benchmark_compare.py
  ````
- **输出**：`results/figures/phase1_v2/06_benchmark_comparison.png`
- **依赖**：poyo conda 环境, matplotlib
- **备注**：对应 plan.md 任务 1.3.4 Benchmark 对比


### phase1_14_15_visualize.py（1.4/1.5 对比可视化）

- **路径**：scripts/analysis/neurohorizon/phase1_14_15_visualize.py
- **功能用途**：Phase 1 实验 1.4（obs_window）和 1.5（session count）的对比可视化
  - 1.4 图：fp-bps vs obs_window 对比图（NeuroHorizon + NDT2 + Neuroformer + IBL-MtM，4 条曲线）
  - 1.5 图：fp-bps vs session_count 对比图（4 模型曲线）
- **创建时间**：2026-03-12
- **使用方式**：
  ```bash
  conda activate poyo
  cd /root/autodl-tmp/NeuroHorizon
  python scripts/analysis/neurohorizon/phase1_14_15_visualize.py
  ```
- **输出**：
  - results/figures/phase1_obs_window/（1.4 obs_window 对比图）
  - results/figures/phase1_sessions/（1.5 session count 对比图）
- **依赖**：poyo conda 环境（matplotlib, numpy）
- **备注**：对应 plan.md 任务 1.4、1.5 可视化

### faithful_ndt2.py（1.8.3 NDT2 faithful reproduction runner）

- **路径**：`neural-benchmark/faithful_ndt2.py`
- **功能用途**：以 canonical benchmark protocol 驱动上游 `BrainBertInterface` 的 faithful NDT2 训练/评估
  - 构造 Perich-Miller continuous / trial-aligned canonical windows
  - 将 binned counts 转成 NDT2 `serve_tokenized_flat` 输入
  - 运行 `smoke` / `train` / `best-valid` checkpoint selection / held-out `test` continuous / `trial-aligned` eval
- **创建时间**：2026-03-17
- **最新修正**：2026-03-17
  - 修正 per-session variable-length flat tokenization，不再把所有 session 强行扩到全局 `channel_capacity`
  - 修正 mixed-session batch 的 padding，新增 `pad_token < max_spatial_tokens` 保护
  - 默认配置对齐到上游 `flat_enc_dec/f8`（`token spike embedding`, `dropout=0.1`, `ramp/decay=100/2500`）
- **使用方式**：
  ```bash
  cd /root/autodl-tmp/NeuroHorizon
  /root/miniconda3/bin/conda run -n benchmark-env \
    python neural-benchmark/faithful_ndt2.py --mode smoke

  /root/miniconda3/bin/conda run -n benchmark-env \
    python neural-benchmark/faithful_ndt2.py \
    --mode train --pred-window 0.25 --batch-size 16 --num-workers 4 \
    --epochs 10 --eval-every 1 \
    --output-dir results/logs/phase1_benchmark_repro_faithful_ndt2_250ms_f8align_pad8_e10
  ```
- **输出**：`results/logs/phase1_benchmark_repro_faithful_ndt2_*`（`results.json`, `best_model.pt`, `last_model.pt`）
- **依赖**：benchmark-env conda 环境，`neural_benchmark/repro_protocol.py`
- **关键适配设计**：
  - canonical windows -> per-session variable-length flat token
  - 训练继续保留 upstream `BrainBertInterface + ShuffleInfill`
  - one-step fp 只在 held-out eval 端读取，不改写训练目标
- **当前妥协**：
  - 通过 compatibility bridge 驱动上游核心模型，而不是直接复用上游原生 task pipeline
  - 不强求 train dataloader / sampler 与 NeuroHorizon 完全同构，只统一数据源 / split / eval / metric
- **备注**：当前结果表明 faithful NDT2 在修正后的 canonical protocol 上仍显著为负，现阶段更像 objective mismatch，而不是残余实现 bug


### faithful_ibl_mtm.py（1.8.3 IBL-MtM faithful bridge）

- **路径**：`neural-benchmark/faithful_ibl_mtm.py`
- **功能用途**：
  - 以 canonical benchmark protocol 驱动上游 `NDT1 + stitching + session prompting`
  - 训练端保持 upstream SSL multi-mask 语义；held-out eval 单独使用 one-step `forward_pred`
- **最新修正**：2026-03-18
  - 不再把训练固定成 `forward-pred`
  - 当前默认在 Perich-Miller 上使用 `combined` multi-mask（实际采样 `neuron + causal`）
  - session-pure batching 保持上游 `eid` / session token 语义
- **使用方式**：
  ```bash
  cd /root/autodl-tmp/NeuroHorizon
  /root/miniconda3/bin/conda run -n benchmark-env     python neural-benchmark/faithful_ibl_mtm.py --mode smoke

  /root/miniconda3/bin/conda run -n benchmark-env     python neural-benchmark/faithful_ibl_mtm.py     --mode train --pred-window 0.25 --epochs 1 --batch-size 8     --output-dir results/logs/phase1_benchmark_repro_faithful_ibl_mtm_250ms_multimask_e1
  ```
- **输出**：`results/logs/phase1_benchmark_repro_faithful_ibl_mtm_*`（`results.json`, `best_model.pt`, `last_model.pt`）
- **关键适配设计**：
  - canonical windows -> `NDT1` tensor contract
  - 训练端恢复 upstream `ssl` multi-mask；held-out eval 单独做 one-step `forward_pred`
  - session-pure batching 保持 `eid / session prompting` 语义
- **当前妥协**：
  - Perich-Miller 无 region annotation，`combined` 实际退化为 `neuron + causal`
  - 为保住 prompting fidelity，牺牲了跨 session 混合训练分布
- **当前 blocker / 风险**：
  - 当前已完成 smoke、limited-window debug e1 和 full-data 250ms multimask e1，结果仍显著为负
  - 下一步不是直接扩长窗口，而是先区分“训练轮数不足”还是“benchmark mismatch”

### faithful_neuroformer.py（1.8.3 Neuroformer faithful bridge）

- **路径**：`neural-benchmark/faithful_neuroformer.py`
- **功能用途**：
  - 以 canonical benchmark protocol 驱动上游 `Tokenizer + Neuroformer.forward + autoregressive generation`
  - 关闭视觉 / 行为分支，保留 neural token generation 主体
  - decode 后 re-bin 到 `20ms` counts，再接统一 held-out continuous / trial-aligned eval
  - 同时支持 `rollout(true_past=False)` 与 `true_past=True`
- **最新修正**：2026-03-19
  - 新增 `eval-only` 模式，可从 checkpoint 单独跑 held-out eval
  - 新增 `token_stats`（prev/curr token 统计与 truncation rate）
  - `rollout` 评估不再额外执行冗余 teacher-forced forward
  - `true_past` 继续复用单次 teacher-forced 前向输出解码
- **使用方式**：
  ```bash
  cd /root/autodl-tmp/NeuroHorizon
  /root/miniconda3/bin/conda run -n benchmark-env     python neural-benchmark/faithful_neuroformer.py --mode smoke

  /root/miniconda3/bin/conda run -n benchmark-env     python neural-benchmark/faithful_neuroformer.py     --mode train --pred-window 0.25 --epochs 1 --batch-size 16 --max-generate-steps 192     --output-dir results/logs/phase1_benchmark_repro_faithful_neuroformer_250ms_dualmode_e1_g192

  /root/miniconda3/bin/conda run -n benchmark-env     python neural-benchmark/faithful_neuroformer.py     --mode eval --checkpoint-path results/logs/phase1_benchmark_repro_faithful_neuroformer_250ms_dualmode_e1/best_model.pt     --eval-split both --inference-mode both --batch-size 4 --max-generate-steps 192     --output-dir results/logs/phase1_benchmark_repro_faithful_neuroformer_250ms_formal_eval_v1
  ```
- **输出**：`results/logs/phase1_benchmark_repro_faithful_neuroformer_*`（`results.json`, `best_model.pt`, `last_model.pt`）
- **关键适配设计**：
  - raw spike events -> `ID / dt` token streams
  - 保留 causal teacher forcing 与 autoregressive generation
  - 同时输出 `rollout(true_past=False)` 与 `true_past=True`，再统一 re-bin 到 `20ms` counts
- **当前妥协**：
  - 显式关闭视觉 / 行为分支，只保留 neural generation 主体
  - `true_past` 通过 teacher-forced 输出解码近似 oracle-history 语义
  - 当前用 `max_generate_steps=192` 作为 250ms formal eval 的可执行性上限
- **当前 blocker / 风险**：
  - dual-mode smoke 已稳定，但 250ms formal full-data dual-mode eval 仍被运行成本卡住
  - 当前最大问题是 formal eval 可执行性，而不是继续盲目扩 500ms / 1000ms


### compare_faithful_ibl_mtm.py（1.8.3 IBL-MtM 250ms 对比汇总）

- **路径**：`neural-benchmark/compare_faithful_ibl_mtm.py`
- **功能用途**：对比 faithful IBL-MtM `combined` 与 `forward_pred` 两条 250ms 训练路线，统一输出 markdown/json 摘要
- **使用方式**：
  ```bash
  cd /root/autodl-tmp/NeuroHorizon
  python3 neural-benchmark/compare_faithful_ibl_mtm.py     --baseline-json results/logs/phase1_benchmark_repro_faithful_ibl_mtm_250ms_combined_e10/results.json     --control-json results/logs/phase1_benchmark_repro_faithful_ibl_mtm_250ms_forwardpred_e10/results.json     --output-dir results/logs/phase1_benchmark_repro_faithful_ibl_mtm_250ms_compare
  ```
- **输出**：`comparison.json`, `comparison.md`

### compare_faithful_neuroformer.py（1.8.3 Neuroformer 对比汇总）

- **路径**：`neural-benchmark/compare_faithful_neuroformer.py`
- **功能用途**：对比 canonical `500/250` formal eval 与 `150/50` 参考实验的 rollout / true_past 结果
- **使用方式**：
  ```bash
  cd /root/autodl-tmp/NeuroHorizon
  python3 neural-benchmark/compare_faithful_neuroformer.py     --canonical-json results/logs/phase1_benchmark_repro_faithful_neuroformer_250ms_formal_eval_v1/eval_results.json     --reference-json results/logs/phase1_benchmark_repro_faithful_neuroformer_50ms_reference_e3/results.json     --output-dir results/logs/phase1_benchmark_repro_faithful_neuroformer_compare     --split test
  ```
- **输出**：`comparison.json`, `comparison.md`

---

## Phase 1.10：Latent Dynamics Decoder

### verify_latent_dynamics.py（1.10 功能验证）

- **路径**：`scripts/1.10-latent_dynamics_decoder/20260320_latent_dynamics_decoder/verify_latent_dynamics.py`
- **功能用途**：验证 `decoder_variant=latent_dynamics` 的基础功能
  - 检查实例化与输出 shape
  - 检查 `requires_target_counts=False`
  - 检查 `forward()` 与 `generate()` 数值一致
- **创建时间**：2026-03-20
- **使用方式**：
  ```bash
  conda activate poyo
  cd /root/autodl-tmp/NeuroHorizon
  python scripts/1.10-latent_dynamics_decoder/20260320_latent_dynamics_decoder/verify_latent_dynamics.py
  ```
- **输出**：stdout 验证信息
- **依赖**：poyo conda 环境（PyTorch）
- **备注**：作为 `1.10` 实现 checkpoint 的最小功能回归

### run_latent_dynamics_smoke.sh（1.10 250ms smoke run）

- **路径**：`scripts/1.10-latent_dynamics_decoder/20260320_latent_dynamics_decoder/run_latent_dynamics_smoke.sh`
- **功能用途**：执行 `1.10` 的最小 smoke 流程
  - 先跑功能验证
  - 再跑 250ms、1 epoch 的最小训练
  - 最后执行 continuous valid 离线评估
- **创建时间**：2026-03-20
- **使用方式**：
  ```bash
  conda activate poyo
  cd /root/autodl-tmp/NeuroHorizon
  bash scripts/1.10-latent_dynamics_decoder/20260320_latent_dynamics_decoder/run_latent_dynamics_smoke.sh
  ```
- **输出**：
  - `results/logs/1.10-latent_dynamics_decoder/20260320_latent_dynamics_decoder/250ms_smoke/`
  - `eval_v2_valid_results.json`
- **依赖**：poyo conda 环境
- **备注**：只验证链路，不用于正式性能结论

### run_latent_dynamics_experiments.sh（1.10 三窗口正式实验）

- **路径**：`scripts/1.10-latent_dynamics_decoder/20260320_latent_dynamics_decoder/run_latent_dynamics_experiments.sh`
- **功能用途**：执行 `1.10` 正式三窗口实验
  - 250ms / 500ms / 1000ms 训练
  - best checkpoint 的 continuous `valid/test` 离线评估
  - 训练结束后调用结果汇总脚本
- **创建时间**：2026-03-20
- **使用方式**：
  ```bash
  conda activate poyo
  cd /root/autodl-tmp/NeuroHorizon
  bash scripts/1.10-latent_dynamics_decoder/20260320_latent_dynamics_decoder/run_latent_dynamics_experiments.sh
  ```
- **输出**：`results/logs/1.10-latent_dynamics_decoder/20260320_latent_dynamics_decoder/{250ms,500ms,1000ms}/`
- **依赖**：poyo conda 环境
- **备注**：沿用 `1.3.7` 的 continuous 默认协议

### monitor_latent_dynamics_progress.py（1.10 进度监控）

- **路径**：`scripts/1.10-latent_dynamics_decoder/20260320_latent_dynamics_decoder/monitor_latent_dynamics_progress.py`
- **功能用途**：轮询三窗口训练日志，生成 `progress_status.md`
- **创建时间**：2026-03-20
- **使用方式**：
  ```bash
  conda activate poyo
  cd /root/autodl-tmp/NeuroHorizon
  python scripts/1.10-latent_dynamics_decoder/20260320_latent_dynamics_decoder/monitor_latent_dynamics_progress.py --interval-sec 600
  ```
- **输出**：`results/logs/1.10-latent_dynamics_decoder/20260320_latent_dynamics_decoder/progress_status.md`
- **依赖**：poyo conda 环境（pandas）
- **备注**：便于长跑实验期间追踪 epoch / loss / `val/fp_bps`

### collect_latent_dynamics_results.py（1.10 结果汇总）

- **路径**：`scripts/1.10-latent_dynamics_decoder/20260320_latent_dynamics_decoder/collect_latent_dynamics_results.py`
- **功能用途**：汇总 `1.10` 三窗口评估 JSON，并回写 summary JSON 与 `results.tsv`
- **创建时间**：2026-03-20
- **使用方式**：
  ```bash
  conda activate poyo
  cd /root/autodl-tmp/NeuroHorizon
  python scripts/1.10-latent_dynamics_decoder/20260320_latent_dynamics_decoder/collect_latent_dynamics_results.py
  ```
- **输出**：
  - `results/figures/1.10-latent_dynamics_decoder/20260320_latent_dynamics_decoder/latent_dynamics_summary.json`
  - `cc_todo/1.10-latent_dynamics_decoder/results.tsv`
- **依赖**：poyo conda 环境（pandas）
- **备注**：默认把 latent-dynamics 的 `teacher-forced` 与 `rollout` 视为同一路径

### plot_optimization_training_curves.py（1.10 训练曲线可视化）

- **路径**：`scripts/1.10-latent_dynamics_decoder/plot_optimization_training_curves.py`
- **功能用途**：根据 summary JSON 和 Lightning `metrics.csv` 生成 `1.10` 训练曲线面板
- **创建时间**：2026-03-20
- **使用方式**：
  ```bash
  conda activate poyo
  cd /root/autodl-tmp/NeuroHorizon
  python scripts/1.10-latent_dynamics_decoder/plot_optimization_training_curves.py --module 20260320_latent_dynamics_decoder
  ```
- **输出**：
  - `results/figures/1.10-latent_dynamics_decoder/20260320_latent_dynamics_decoder/training_curves.png`
  - `results/figures/1.10-latent_dynamics_decoder/20260320_latent_dynamics_decoder/training_curves.pdf`
- **依赖**：poyo conda 环境（matplotlib, pandas, numpy）
- **备注**：与 `1.9` 的曲线面板保持同样的信息密度

### plot_optimization_progress.py（1.10 进度趋势图）

- **路径**：`scripts/1.10-latent_dynamics_decoder/plot_optimization_progress.py`
- **功能用途**：读取 `cc_todo/1.10-latent_dynamics_decoder/results.tsv`，绘制窗口级优化趋势图
- **创建时间**：2026-03-20
- **使用方式**：
  ```bash
  conda activate poyo
  cd /root/autodl-tmp/NeuroHorizon
  python scripts/1.10-latent_dynamics_decoder/plot_optimization_progress.py
  ```
- **输出**：
  - `results/figures/1.10-latent_dynamics_decoder/optimization_progress.png`
  - `results/figures/1.10-latent_dynamics_decoder/optimization_progress.pdf`
- **依赖**：poyo conda 环境（matplotlib, pandas）
- **备注**：当前以 `baseline_v2` 为水平参考线

### verify_latent_dynamics_state_scaling.py（1.10 state scaling 功能验证）

- **路径**：`scripts/1.10-latent_dynamics_decoder/20260320_latent_dynamics_state_scaling/verify_latent_dynamics_state_scaling.py`
- **功能用途**：验证更大 latent state 配置下的 `latent_dynamics` 基础功能
  - 检查实例化与输出 shape
  - 检查 `requires_target_counts=False`
  - 检查 `forward()` 与 `generate()` 数值一致
- **创建时间**：2026-03-20
- **使用方式**：
  ```bash
  conda activate poyo
  cd /root/autodl-tmp/NeuroHorizon
  python scripts/1.10-latent_dynamics_decoder/20260320_latent_dynamics_state_scaling/verify_latent_dynamics_state_scaling.py
  ```
- **输出**：stdout 验证信息
- **依赖**：poyo conda 环境（PyTorch）
- **备注**：对应 `20260320_latent_dynamics_state_scaling` 的最小功能回归

### run_latent_dynamics_state_scaling_500ms.sh（1.10 500ms state scaling gate）

- **路径**：`scripts/1.10-latent_dynamics_decoder/20260320_latent_dynamics_state_scaling/run_latent_dynamics_state_scaling_500ms.sh`
- **功能用途**：执行更大 latent state 配置下的 `500ms` gate
  - 先跑功能验证
  - 再执行 `500ms` 正式训练
  - 最后执行 best-ckpt continuous `valid/test` 离线评估
- **创建时间**：2026-03-20
- **使用方式**：
  ```bash
  conda activate poyo
  cd /root/autodl-tmp/NeuroHorizon
  bash scripts/1.10-latent_dynamics_decoder/20260320_latent_dynamics_state_scaling/run_latent_dynamics_state_scaling_500ms.sh
  ```
- **输出**：`results/logs/1.10-latent_dynamics_decoder/20260320_latent_dynamics_state_scaling/500ms/`
- **依赖**：poyo conda 环境
- **备注**：作为 `1.10.x` 的 `500ms` capacity-scaling gate，不直接扩展到 `250ms / 1000ms`
