# 脚本管理记录

> 记录项目中所有脚本（数据处理 / 分析 / 项目运行 / 测试等）的信息。
> 每新建一个脚本，必须在此处更新记录。
> **按 Phase + task_num 顺序排列**，便于追溯。

---

## 记录格式

```
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
```

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
