# NeuroHorizon 工作日志

---

## 2026-02-21

### 完成的工作

#### 1. POYO 代码库分析 ✅
- 完成了对 POYO/POYO+ 代码架构的全面分析
- 理清了 Encoder-Processor-Decoder 架构、数据管线、训练流程、模态注册系统
- 识别了 NeuroHorizon 需要修改/替换的关键接口点
- 文档保存：`cc_todo/2026-02-21-poyo-代码分析.md`

#### 2. NeuroHorizon 项目评估与计划 ✅
- 完成了项目合理性评估，识别了 6 个主要问题及应对方案
- 制定了 6 阶段执行计划（Phase 0-5）
- 确认了最小可行发表路径 (MVP)
- 文档保存：`cc_todo/2026-02-21-neurohorizon-项目分析与执行计划.md`

#### 3. 关键决策确认 ✅
- GPU: 4090 单卡，后续可扩展
- Jia Lab 数据：确认不可用，使用 IBL + Allen Natural Movies 替代
- 执行顺序：数据管线优先 (Phase 0)

#### 4. Phase 0.1: 环境扩展 ✅
- poyo 环境：安装了 ONE-api, ibllib（IBL 数据依赖）
- allen 环境：独立 conda 环境安装 allensdk（与 poyo 依赖冲突，numpy 版本不兼容）
- scipy 已有
- 遇到问题：allensdk 要求 numpy==1.23.5，与 IBL 依赖冲突 → 解决方案：独立环境

#### 5. Phase 0.2: IBL 数据下载与预处理 ✅
- 创建了 `scripts/download_ibl.py` — IBL 数据下载脚本
- 创建了 `scripts/preprocess_ibl.py` — 数据转 HDF5（temporaldata 格式）
- 成功处理 10 个 IBL session（12 个中 2 个失败：1 个 good units < 10，1 个下载异常）
- HDF5 格式包含完整的 temporaldata 元数据属性（object, timekeys, absolute_start 等）
- 遇到问题：
  1. Alyx REST 查询使用复杂 django filter 导致超时 → 简化为 task_protocol + project
  2. SpikeSortingLoader 接口复杂 → 改用直接 ONE API 加载
  3. 初始 HDF5 缺少 temporaldata 元数据属性 → 参照 brainsets 格式添加

**IBL 数据统计（10 sessions）：**
| Session | Spikes | Units | Duration |
|---------|--------|-------|----------|
| a7eba2cf | 31,168,943 | 302 | 5382s |
| c46b8def | 27,860,790 | 294 | 5015s |
| ebce500b | 13,133,018 | 291 | 7200s |
| 5ae68c54 | 12,355,315 | 134 | 5853s |
| 11163613 | 7,638,177 | 138 | 5031s |
| 15b69921 | 6,953,627 | 128 | 5014s |
| 6899a67d | 2,996,599 | 80 | 6499s |
| de905562 | 2,191,468 | 19 | 4019s |
| e6594a5b | 1,586,392 | 70 | 4699s |
| d85c454e | 454,639 | 17 | 3691s |

#### 6. Phase 0.3: Allen 数据下载 (进行中)
- 创建了 `scripts/download_allen.py` — Allen Neuropixels 下载+预处理脚本
- 更新了 HDF5 写入格式为 temporaldata 兼容
- 后台运行中，已处理 1 个 session (allen_715093703)

#### 7. Phase 0.4: 参考特征提取 ✅
- 创建了 `scripts/extract_reference_features.py` — IDEncoder 输入特征提取
- 每个 unit 提取 33 维特征：firing rate(1) + ISI CV(1) + ISI log-histogram(20) + autocorrelation(10) + Fano factor(1)
- 使用 60s 参考窗口从 session 开头提取
- 10 个 IBL session 全部提取完成，特征存储在 HDF5 的 `units.reference_features` 字段

#### 8. Phase 0.5: 数据验证 ✅
- 创建了 `scripts/validate_data.py` — HDF5 完整性验证
- 修复了 session.id 检查（属性 vs 数据集）
- 10 个 IBL HDF5 文件全部通过验证
- `torch_brain.dataset.Dataset` 加载测试通过

#### 9. Phase 2.1: PoissonNLLLoss ✅
- 在 `torch_brain/nn/loss.py` 中添加了 `PoissonNLLLoss(Loss)` 类
- 支持 log-rate 输入 + spike count 目标 + 可选权重

#### 10. Phase 2.2: 注册 wheel_velocity 模态 ✅
- 在 `torch_brain/registry.py` 中注册了 `wheel_velocity` 模态（IBL wheel velocity 解码）

#### 11. Phase 2.3: IDEncoder 实现 ✅
- 创建了 `torch_brain/nn/id_encoder.py`
- 3 层 MLP (Linear + GELU + LayerNorm)，输入 33 维 → 输出 model_dim
- 支持 compute_embeddings() 预计算 + forward() 索引查找
- 更新了 `torch_brain/nn/__init__.py` 导出

#### 12. Phase 2.4: NeuroHorizon 模型 ✅
- 创建了 `torch_brain/models/neurohorizon.py`
- 架构组件：
  - CausalRotarySelfAttention：因果自注意力（decoder 用）
  - PerNeuronHead：共享 per-neuron MLP head（处理可变 n_units）
  - NeuroHorizon：完整模型（Encoder-Processor-Decoder）
- Small 配置：8.1M 参数（dim=256, depth=4, dec_depth=2）
- 合成数据上 forward pass + backward pass 验证通过
- 包含 tokenize() 方法用于数据预处理
- 包含 compute_loss() 方法用于 Poisson NLL 损失计算
- 更新了 `torch_brain/models/__init__.py` 导出

### 当前进行中

#### Phase 0.3: Allen 数据下载
- 后台运行中（已处理 4 个 session: 715093703, 719161530, 721123822, 732592105）
- 修复了 Allen HDF5 中 running 组缺少 domain 的问题

#### Phase 2.5: 训练流程 ✅
- 创建了 `examples/neurohorizon/train.py` — 训练脚本（PyTorch Lightning + Hydra）
- 创建了 Hydra 配置文件：
  - `examples/neurohorizon/configs/defaults.yaml` — 基础训练配置
  - `examples/neurohorizon/configs/model/neurohorizon_small.yaml` — Small 模型配置 (~8M params)
  - `examples/neurohorizon/configs/model/neurohorizon_base.yaml` — Base 模型配置 (~33M params)
  - `examples/neurohorizon/configs/train.yaml` — Hydra defaults 链
- 关键组件：
  - `EagerDataset`: 继承 `torch_brain.dataset.Dataset`，使用 eager loading 解决 temporaldata lazy loading 兼容性问题
  - `neurohorizon_collate()`: 自定义 collate，处理可变 n_units padding + Padded8Object
  - `NHTrainWrapper(L.LightningModule)`: 训练/验证步骤，AdamW + OneCycleLR
  - `NHDataModule(L.LightningDataModule)`: 数据加载，支持 train/valid/test domain 分割
- 遇到问题及解决：
  1. `torch_brain.data.Dataset` vs `torch_brain.dataset.Dataset` 混淆 → 使用新版 `torch_brain.dataset.Dataset`
  2. behavior 组缺少 domain → 修复所有 HDF5 文件 + 更新 preprocess_ibl.py
  3. `LazyIrregularTimeSeries._timestamp_indices_1s` 缺失 → 使用 `Data.from_hdf5(f, lazy=False)` 的 EagerDataset 子类
  4. Padded8Object collation → 使用 `torch_brain.data.collate()` 处理
- 端到端测试通过：batch_size=4, 3 sessions, forward + backward + optimizer step + metrics 全部正常

#### 首次完整训练验证 ✅
- 训练配置：batch_size=16, 10 IBL sessions, bf16-mixed, AdamW + OneCycleLR
- **1 epoch 结果**：
  - Train loss: 0.95 → 0.39（7000 步）
  - Val loss: 0.475
  - Val bits/spike: -1.09（模型尚未超过 null model，训练初期正常）
  - Epoch 耗时: ~51 分钟
- Checkpoint 已保存至 `logs/neurohorizon/lightning_logs/`

#### Phase 2.6: 评估指标 ✅
- 创建了 `torch_brain/utils/neurohorizon_metrics.py`
- 指标：
  - `poisson_log_likelihood()`: 平均 Poisson 对数似然
  - `bits_per_spike()`: (LL_model - LL_null) / (n_spikes * log(2))，null = mean-rate model
  - `firing_rate_correlation()`: 时间平均预测 vs 真实 firing rate 的 Pearson 相关
  - `r2_binned_counts()`: 预测 rate vs 真实 spike counts 的 R²
- 已集成到训练脚本的 validation_step 中

### 待完成
- Phase 1: POYO 基线验证（IBL wheel velocity 解码）
- 完整训练测试（1 epoch 后台运行中）
- Phase 3: 多模态扩展
- Phase 4: 实验
- Phase 5: 分析与论文

### 遇到的问题与解决方案

| 问题 | 解决方案 |
|------|----------|
| allensdk 与 IBL 依赖 numpy 版本冲突 | 创建独立 allen conda 环境 |
| IBL Alyx REST 复杂 django filter 超时 | 简化查询：仅用 task_protocol + project |
| SpikeSortingLoader 接口复杂 | 改用直接 ONE API (one.load_dataset) |
| HDF5 缺少 temporaldata 元数据属性 | 参照 brainsets 格式添加 object/timekeys/absolute_start 属性 |
| validate_data.py 检查 session.id 方式错误 | 修复为同时检查属性和数据集 |
| torch_brain.data.Dataset vs torch_brain.dataset.Dataset | 使用新版 torch_brain.dataset.Dataset（dataset_dir 接口）|
| HDF5 behavior 组缺少 domain → 无法 slice | 修复所有 HDF5 + 更新预处理脚本 |
| LazyIrregularTimeSeries 缺少 _timestamp_indices_1s | 使用 eager loading (lazy=False) 的 EagerDataset 子类 |
| Padded8Object 不是 ndarray/dict | 使用 torch_brain.data.collate() 处理 |

### 版本记录
| 日期 | 版本 | 描述 |
|------|------|------|
| 2026-02-21 | v0.1 | 项目分析完成，计划制定，开始执行 Phase 0 |
| 2026-02-21 | v0.2 | Phase 0 完成：环境搭建、IBL 数据管线（10 sessions）、参考特征提取、数据验证 |
| 2026-02-21 | v0.3 | Phase 2.1-2.4 完成：PoissonNLLLoss、wheel_velocity 模态、IDEncoder、NeuroHorizon 模型（8.1M params） |
| 2026-02-21 | v0.4 | Phase 2.5-2.6 完成：训练流程（EagerDataset + Hydra + Lightning）、评估指标、端到端验证通过、1 epoch 训练成功（loss 0.95→0.39） |

---
