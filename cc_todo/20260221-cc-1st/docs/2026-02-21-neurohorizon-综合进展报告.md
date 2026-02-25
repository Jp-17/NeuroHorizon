# NeuroHorizon 项目综合进展报告

> 日期：2026-02-21
> 状态：Phase 0-3 完成，训练进行中（v1 epoch 19/100, v2 epoch 10/100）

---

## 一、项目概述

NeuroHorizon 基于 POYO/POYO+（NeurIPS 2023 神经群体解码框架）进行改造，目标是构建一个**统一的神经编码模型**。

### 核心创新点

| # | 创新点 | 描述 |
|---|--------|------|
| 1 | **任务转变** | 从解码（spikes → behavior）转为编码（spikes → future spikes） |
| 2 | **IDEncoder** | 替换 per-unit learnable embedding，用 MLP 从参考窗口特征生成 unit embedding，实现跨 session 泛化 |
| 3 | **自回归解码器** | Cross-attention decoder 预测未来 0.5-1s 的 spike count（按 20ms time bin） |
| 4 | **多模态融合** | 行为数据 + 图像（CLIP ViT-L/14 embedding）通过交叉注意力注入编码器 |
| 5 | **长时间跨度预测** | 从传统 ~200ms 扩展到 1 秒 |

### 技术架构

```
输入: spike tokens (IDEncoder embedding + RoPE)
  → Perceiver Cross-Attention (spike tokens → latent tokens)
  → Self-Attention Processing (多层 RotarySelfAttention + FFN)
  → [可选] Multimodal Cross-Attention (行为/图像条件)
  → Causal Self-Attention Decoder (预测未来 time bins)
  → PerNeuronHead (共享 MLP, 输出 log firing rate)
  → Poisson NLL Loss
```

---

## 二、已完成的工作

### Phase 0: 环境与数据准备 ✅

#### 0.1 环境搭建
- **poyo 环境**（conda）：安装了 ONE-api, ibllib（IBL 数据依赖）、scipy 等
- **allen 环境**（独立 conda）：安装 allensdk（与 poyo 环境 numpy 版本冲突，需独立环境）
- 解决了 allensdk 要求 numpy==1.23.5 与 IBL 依赖的冲突

#### 0.2 IBL 数据下载与预处理
- **脚本**：`scripts/download_ibl.py`, `scripts/preprocess_ibl.py`
- **成果**：成功处理 10 个 IBL session（12 个中 2 个失败：1 个 good units < 10，1 个下载异常）
- **HDF5 格式**：包含完整 temporaldata 元数据（object, timekeys, absolute_start 等）
- **数据统计**：

| Session | Spikes | Units | Duration |
|---------|--------|-------|----------|
| a7eba2cf | 31.2M | 302 | 5382s |
| c46b8def | 27.9M | 294 | 5015s |
| ebce500b | 13.1M | 291 | 7200s |
| 5ae68c54 | 12.4M | 134 | 5853s |
| 11163613 | 7.6M | 138 | 5031s |
| 15b69921 | 7.0M | 128 | 5014s |
| 6899a67d | 3.0M | 80 | 6499s |
| de905562 | 2.2M | 19 | 4019s |
| e6594a5b | 1.6M | 70 | 4699s |
| d85c454e | 0.5M | 17 | 3691s |
| **总计** | **~107M** | **~1473** | **~52,403s** |

#### 0.3 Allen 数据下载与预处理
- **脚本**：`scripts/download_allen.py`
- **成果**：5 个 Allen Neuropixels sessions 下载并预处理

| Session | Spikes | Units | Duration |
|---------|--------|-------|----------|
| allen_715093703 | 67.9M | 875 | 9629s |
| allen_719161530 | 64.6M | 751 | 9665s |
| allen_721123822 | 35.1M | 439 | 9811s |
| allen_732592105 | 60.4M | 818 | 9415s |
| allen_737581020 | 44.9M | 566 | 9277s |
| **总计** | **~273M** | **~3449** | **~47,797s** |

#### 0.4 参考特征提取
- **脚本**：`scripts/extract_reference_features.py`
- 每个 unit 提取 33 维特征：
  - firing rate (1d)
  - ISI 变异系数 (1d)
  - ISI log-histogram (20d)
  - autocorrelation (10d)
  - Fano factor (1d)
- 使用 60s 参考窗口从 session 开头提取
- 10 个 IBL + 5 个 Allen session 全���提取完成

#### 0.5 参考特征归一化
- **脚本**：`scripts/normalize_reference_features.py`
- 发现特征尺度差异极大：firing_rate max=224 vs isi_hist max~1
- 实施 z-score 归一化（mean=0, std=1 per feature）
- 原始特征备份到 `reference_features_raw`
- **验证**：归一化后 v2 训练效果显著优于 v1（详见训练结果）

#### 0.6 数据验证
- **脚本**：`scripts/validate_data.py`
- 检查 HDF5 完整性、时间戳排序、unit index 有效性、无 NaN 等
- 15 个 HDF5 文件全部通过验证
- `torch_brain.dataset.Dataset` 加载测试通过

---

### Phase 1: POYO 基线验证 ✅（结论：失败）

#### 实验设计
- **目标**：在 IBL 数据上运行 POYO 模型进行 wheel velocity 解码
- **脚本**：`examples/poyo_baseline/train_baseline.py`
- **配置**：POYO Small (dim=128, depth=8, ~3.5M params), 200 epochs, batch_size=64

#### 训练结果

| Epoch | val_loss | val_r2 |
|-------|----------|--------|
| 9 | 5.043 | -0.149 |
| 19 | 4.569 | -0.067 |
| 29 | 4.617 | -0.137 |
| 39 | **4.495** | **-0.050** (best) |
| 49 | 5.065 | -0.207 |
| 59 | 4.658 | -0.210 |
| 69 | 4.924 | -0.195 |
| 79 | 4.702 | -0.152 |
| 89 | **5.692** | **-0.520** (worst) |
| 99 | 4.934 | -0.221 |

#### 分析
- **所有 epoch 的 val_r2 均为负值**，模型未能超过简单均值预测基线
- **严重过拟合**：train_loss ~1.24 vs val_loss ~4.93
- **Epoch 89 灾难性恶化**：R² 从 -0.15 骤降到 -0.52
- **根本原因**：temporal split 导致 train/valid 行为分布严重不匹配（例：某 session 训练集 wv_mean=0.50, 验证集 wv_mean=-1.52）
- **结论**：POYO 的 per-unit learnable embedding 无法在 temporal split 下泛化
- **处理**：在 epoch 106 终止，释放 14.8 GB GPU 内存

#### 修复的 Bug（4 个）
1. target 维度不匹配（MSELoss 要求 2D）
2. Lightning setup() 覆盖 transform（添加 hasattr guard）
3. Hydra _target_ 类型错误（绕过 hydra.utils.instantiate）
4. R² 计算形状不匹配（.view(-1) 展平）

---

### Phase 2: NeuroHorizon 核心模型 ✅

#### 2.1 PoissonNLLLoss
- 文件：`torch_brain/nn/loss.py`
- 支持 log-rate 输入 + spike count 目标 + 可选权重

#### 2.2 模态注册
- 文件：`torch_brain/registry.py`
- 注册了 `wheel_velocity` 模态（IBL wheel velocity 解码）

#### 2.3 IDEncoder
- 文件：`torch_brain/nn/id_encoder.py`
- 3 层 MLP：Linear(33, dim) → GELU → LayerNorm → Linear → GELU → LayerNorm → Linear
- 支持 `compute_embeddings()` 预计算 + `forward()` 索引查找
- 消融模式：`"idencoder"` (默认), `"random"` (固定随机投影), `"mean"` (简单线性)

#### 2.4 NeuroHorizon 模型
- 文件：`torch_brain/models/neurohorizon.py`
- **架构组件**：
  - `CausalRotarySelfAttention`：因果自注意力（decoder 用）
  - `PerNeuronHead`：共享 per-neuron MLP head（处理可变 n_units）
  - `NeuroHorizon`：完整 Encoder-Processor-Decoder 模型
- **模型配置**：
  - Small: ~8.1M params (dim=256, depth=4, dec_depth=2)
  - Base: ~33M params (dim=512, depth=8, dec_depth=4)
- 包含 `tokenize()`, `compute_loss()` 方法
- 合成数据上 forward + backward pass 验证通过

#### 2.5 训练流程
- 文件：`examples/neurohorizon/train.py`
- **关键组件**：
  - `EagerDataset`: 继承 `torch_brain.dataset.Dataset`，eager loading 解决 lazy loading 兼容性
  - `neurohorizon_collate()`: 自定义 collate，可变 n_units padding + union 语义多模态
  - `NHTrainWrapper(L.LightningModule)`: AdamW + OneCycleLR
  - `NHDataModule(L.LightningDataModule)`: 支持 train/valid/test domain 分割 + 多目录
- **配置文件**：
  - `configs/defaults.yaml` — 基础训练配置
  - `configs/model/neurohorizon_small.yaml` — Small 模型
  - `configs/model/neurohorizon_base.yaml` — Base 模型
  - `configs/train.yaml` — Hydra defaults 链

#### 2.6 评估指标
- 文件：`torch_brain/utils/neurohorizon_metrics.py`
- 指标：Poisson log-likelihood, bits/spike, firing rate correlation, R² of binned counts

---

### Phase 3: 多模态扩展 ✅

#### 3.1 多模态模块
- 文件：`torch_brain/nn/multimodal.py`
- `MultimodalCrossAttention`: 投影模态 embedding + RoPE 时间对齐交叉注意力
- `MultimodalEncoder`: 封装图像和行为模态交叉注意力层 (~2.3M 参数)

#### 3.2 模型集成
- 在 `neurohorizon.py` 中集成 MultimodalEncoder
- 新参数：`use_multimodal`, `multimodal_every`, `image_dim`, `behavior_dim`
- 在编码器 self-attention 层间插入多模态交叉注意力（每 N 层一次）
- 向后兼容：`use_multimodal=False` 默认保��现有配置
- 支持优雅降级：多模态模型可在无多模态输入时正常运行

#### 3.3 图像 Embedding 提取与注入
- **脚本**：`scripts/extract_image_embeddings.py`, `scripts/inject_image_embeddings.py`
- 使用 CLIP ViT-L/14（替代 DINOv2，因网络访问受限）
- 输出维度：1024
- Allen 刺激帧处理：
  - Natural Movie 1: 900 frames → (900, 1024)
  - Natural Movie 3: 3600 frames → (3600, 1024)
  - Natural Scenes: 118 images → (118, 1024)
- 5 个 Allen sessions 全部注入完成

#### 3.4 Collate 改进
- union 语义处理多模态 key：缺失样本零填充 + False mask
- `_collate_multimodal_keys()` 函数
- 混合 batch（有/无 behavior/image）正确处理

#### 3.5 多目录 Dataset 支持
- EagerDataset 新增 `dataset_dirs` 参数，支持同时加载 IBL + Allen 数据
- 向后兼容：`dataset_dir` 单目录参数仍可用

#### 3.6 多模态配置
- `neurohorizon_small_beh.yaml` — 行为条件模型 (~10.2M params)
- `neurohorizon_small_mm.yaml` — 全模态模型 (~12.8M params, image_dim=1024)
- `train_v2.yaml` — IBL+Allen 15 sessions + behavior
- `train_v2_ibl.yaml` — IBL-only 归一化特征对照
- `train_v2_norm.yaml` — IBL-only 归一化特征 200 epochs
- `train_v2_mm.yaml` — 全模态配置

---

### 实验基础设施 ✅

| 脚本 | 功能 |
|------|------|
| `scripts/evaluate_neurohorizon.py` | NeuroHorizon 后训练评估 |
| `scripts/evaluate_poyo_baseline.py` | POYO 基线后训练评估 |
| `scripts/evaluate_cross_session.py` | 跨 session 泛化评估 |
| `scripts/evaluate_horizons.py` | 不同预测时间跨度评估 |
| `scripts/run_ablations.py` | 消融实验运行器 |
| `scripts/collect_results.py` | 综合结果收集器 |
| `scripts/training_queue.sh` | 顺序训练队列 |

---

## 三、训练结果

### NeuroHorizon v1（未归一化特征，当前运行中）

- **配置**：batch_size=16, 10 IBL sessions, 100 epochs, bf16-mixed, ~8.1M params
- **当前进度**：epoch 19/100

| Epoch | val_loss | val_bits_per_spike | 改善 |
|-------|----------|-------------------|------|
| 4 | 0.4670 | -1.0075 | — |
| 9 | 0.4214 | -0.7242 | +28% |
| 14 | 0.3982 | -0.5729 | +21% |

- **train_loss**：从 0.52 (epoch 0) 下降到 ~0.40 (epoch 19)
- **趋势**：bits/spike 持续改善，-1.01 → -0.72 → -0.57
- **无过拟合**：val_loss ≈ train_loss
- **预测**：按当前改善速率，预计 epoch 30-40 BPS 转正（超过 null model）

### NeuroHorizon v2 IBL（归一化特征，当前运行中）

- **配置**：batch_size=16, 10 IBL sessions, 100 epochs, bf16-mixed, ~8.1M params
- **改进**：使用 z-score 归一化参考特征 + 更多验证指标
- **当前进度**：epoch 10/100

| Epoch | val_loss | val_bps | val_fr_corr | val_r2 |
|-------|----------|---------|-------------|--------|
| 4 | 0.4138 | -0.679 | 0.733 | 0.320 |
| 9 | 0.3937 | -0.552 | 0.789 | 0.319 |

#### v1 vs v2 对比（归一化特征的影响）

| 指标 | v1 epoch 4 | v2 epoch 4 | 改善 |
|------|------------|------------|------|
| val_loss | 0.4670 | 0.4138 | -11.4% |
| val_bps | -1.008 | -0.679 | +32.6% |
| val_fr_corr | N/A | 0.733 | — |
| val_r2 | N/A | 0.320 | — |

| 指标 | v1 epoch 9 | v2 epoch 9 | 改善 |
|------|------------|------------|------|
| val_loss | 0.4214 | 0.3937 | -6.6% |
| val_bps | -0.724 | -0.552 | +23.8% |
| val_fr_corr | N/A | 0.789 | — |
| val_r2 | N/A | 0.319 | — |

**关键发现**：
1. **参考特征归一化显著提升模型性能**，v2 在相同 epoch 下 BPS 比 v1 好 24-33%
2. **v2 firing rate correlation = 0.79**：模型预测的发放率与真实发放率高度相关
3. **v2 R² = 0.32**：模型能解释约 32% 的 spike count 变异
4. v2 BPS 预计更早转正（epoch 20-25 附近）

### POYO 基线（已终止）

- **结论：失败** — 所有 10 次验证的 R² 均为负值
- 最佳 checkpoint: epoch 39 (val_r2=-0.050)
- 严重过拟合 + temporal distribution shift
- 在 epoch 106 终止，释放 14.8 GB GPU

---

## 四、代码文件清单

### 新建文件

| 文件路径 | 描述 |
|----------|------|
| `torch_brain/nn/id_encoder.py` | IDEncoder 模块 (33d → dim MLP) |
| `torch_brain/nn/multimodal.py` | 多模态交叉注意力模块 |
| `torch_brain/models/neurohorizon.py` | NeuroHorizon 完整模型 |
| `torch_brain/utils/neurohorizon_metrics.py` | 评估指标 (BPS, FR corr, R²) |
| `examples/neurohorizon/train.py` | 训练脚本 (Lightning + Hydra) |
| `examples/neurohorizon/configs/` | 所有 Hydra 配置文件 (7+ yaml) |
| `examples/poyo_baseline/train_baseline.py` | POYO 基线训练脚本 |
| `examples/poyo_baseline/configs/` | POYO 基线配置文件 |
| `scripts/download_ibl.py` | IBL 数据下载 |
| `scripts/preprocess_ibl.py` | IBL 数据预处理 → HDF5 |
| `scripts/download_allen.py` | Allen 数据下载+预处理 |
| `scripts/extract_reference_features.py` | 参考特征提取 (33d/unit) |
| `scripts/normalize_reference_features.py` | 参考特征 z-score 归一化 |
| `scripts/validate_data.py` | HDF5 完整性验证 |
| `scripts/extract_image_embeddings.py` | CLIP ViT-L/14 embedding 提取 |
| `scripts/inject_image_embeddings.py` | Embedding 注入 Allen HDF5 |
| `scripts/evaluate_neurohorizon.py` | NeuroHorizon 后训练评估 |
| `scripts/evaluate_poyo_baseline.py` | POYO 基线后训练评估 |
| `scripts/evaluate_cross_session.py` | 跨 session 泛化评估 |
| `scripts/evaluate_horizons.py` | 预测时间跨度评估 |
| `scripts/run_ablations.py` | 消融实验运行器 |
| `scripts/collect_results.py` | 结果收集器 |
| `scripts/training_queue.sh` | 训练队列脚本 |

### 修改文件

| 文件路径 | 修改内容 |
|----------|----------|
| `torch_brain/nn/loss.py` | 添加 PoissonNLLLoss |
| `torch_brain/nn/__init__.py` | 导出 IDEncoder, MultimodalEncoder 等 |
| `torch_brain/registry.py` | 注册 wheel_velocity 模态 |
| `torch_brain/models/__init__.py` | 导出 NeuroHorizon |

### 数据文件

| 路径 | 描述 |
|------|------|
| `autodl-tmp/datasets/ibl_processed/` | 10 个 IBL HDF5 文件 (~107M spikes) |
| `autodl-tmp/datasets/allen_processed/` | 5 个 Allen HDF5 文件 (~273M spikes) |
| `autodl-tmp/datasets/allen_stimuli/` | Allen 刺激帧 (numpy) |
| `autodl-tmp/datasets/allen_embeddings/` | CLIP embedding 文件 |

---

## 五、遇到的问题与解决方案

| # | 问题 | 解决方案 |
|---|------|----------|
| 1 | allensdk 与 IBL numpy 版本冲突 | 创建独立 allen conda 环境 |
| 2 | IBL Alyx REST 复杂查询超时 | 简化为 task_protocol + project |
| 3 | SpikeSortingLoader 接口复杂 | 改用直接 ONE API 加载 |
| 4 | HDF5 缺少 temporaldata 元数据属性 | 参照 brainsets 格式添加 |
| 5 | LazyIrregularTimeSeries 缺少 _timestamp_indices_1s | 使用 eager loading (lazy=False) |
| 6 | POYO MSELoss target 维度不匹配 | unsqueeze target |
| 7 | Lightning setup() 覆盖 transform | 添加 hasattr guard 防止重复创建 |
| 8 | Hydra _target_ 被外部还原为 POYOPlus | 绕过 hydra.utils.instantiate |
| 9 | 参考特征尺度差异极大 | z-score 归一化 |
| 10 | EagerDataset deepcopy 瓶颈 | 覆写 __getitem__ 避免 deepcopy |
| 11 | collate 多模态 key intersection 导致丢失 | 改用 union 语义 + 零填充 |
| 12 | DINOv2 无法下载（网络受限） | 使用已缓存 CLIP ViT-L/14 替代 |
| 13 | Allen 刺激数据格式与预期不匹配 | 展开 frame index → per-presentation embeddings |
| 14 | image_embeddings HDF5 group 缺少元数据 | ��加 _unicode_keys, timekeys, domain 属性 |

---

## 六、待完成的工作

### 短期（当前训练完成后）

1. **NH v1 训练完成分析** — 100 epoch 后评估 bits/spike 是否转正
2. **NH v2 IBL 训练完成分析** — 100 epoch 后与 v1 对比，量化归一化特征的贡献
3. **v2_beh 训练** — IBL+Allen 15 sessions + 行为条件，200 epochs
4. **v2_mm 训练** — IBL+Allen 15 sessions + 行为 + 图像，200 epochs

### 中期（Phase 4: 实验）

5. **实验 1: 跨 Session 泛化**（核心贡献）
   - IDEncoder vs per-unit embedding vs random embedding
   - 指标：co-smoothing bits/spike, Poisson LL, R²
6. **实验 2: 长时间预测**
   - 预测跨度：100ms, 200ms, 500ms, 1000ms
7. **实验 3: 多模态消融**
   - neural only vs +behavior vs +image vs +both
8. **实验 4: IDEncoder 消融**
   - idencoder vs random vs mean embedding mode
9. **实验 5: 数据 Scaling Laws**（可选）
   - 训练 session 数：10, 50, 100, 200, 459

### 长期（Phase 5: 论文）

10. **结果收集与可视化**
11. **论文写作** — 目标投稿 NeurIPS / ICLR / Nature Methods

### 最小可行发表路径 (MVP)

如果时间/资源有限，核心工作：
1. ~~IBL 数据管线~~ ✅
2. ~~核心 NeuroHorizon 模型~~ ✅
3. NH v1/v2 训练结果（进行中）
4. 实验 1: 跨 session 泛化（主要贡献点）
5. 关键消融（IDEncoder, loss function）

---

## 七、GPU 资源使用

- **硬件**：NVIDIA RTX 4090 D (24 GB VRAM)
- **当前占用**：
  - NH v1 (PID 34659): ~3.8 GB
  - NH v2 IBL (PID 87666): ~3.8 GB
  - 其他项目 (PID 61917): ~3.9 GB
  - **总计 ~11.5 GB / 24 GB**
- POYO 已终止，释放了 14.8 GB

---

## 八、版本历史

| 版本 | 描述 |
|------|------|
| v0.1 | 项目分析完成，计划制定 |
| v0.2 | Phase 0 完成：环境搭建、IBL 数据管线 |
| v0.3 | Phase 2.1-2.4 完成：Loss、IDEncoder、NeuroHorizon 模型 |
| v0.4 | Phase 2.5-2.6 完成：训练流程、评估指标、端到端验证 |
| v0.5 | Allen 数据完成，POYO 基线就绪，100-epoch 训练启动 |
| v0.6 | POYO 基线 bug 修复，200-epoch 训练启动 |
| v0.7 | 参考特征归一化、评估脚本、多模态模块 |
| v0.8 | 训练监控：NH epoch 10, POYO epoch 51 |
| v0.9 | Phase 3 完成：多模态集成、实验基础设施 |
| v1.0 | 多模态训练准备：union collate、多目录 Dataset |
| v1.1 | POYO 终止、NH v1 epoch 14 bps=-0.57、v2 启动 |
| v1.2 | CLIP embedding 注入 Allen、全模态测试通过 |
| v1.3 | Phase 4 实验脚本完成 |
| **v1.4** | **综合进展报告：NH v1 epoch 19, v2 epoch 10** |

---

## 九、cc_todo 目录文件索引

| 文件名 | 内容 |
|--------|------|
| `poyo_setup_log.md` | POYO 环境搭建与 MC Maze 测试日志 |
| `2026-02-21-poyo-代码分析.md` | POYO/POYO+ 代码架构全面分析 |
| `2026-02-21-neurohorizon-项目分析与执行计划.md` | 项目评估 + 6 阶段执行计划 |
| `2026-02-21-neurohorizon-工作日志.md` | 持续更新的工作日志 (v1.3) |
| `2026-02-21-neurohorizon-综合进展报告.md` | 本文档 — 综合进展总结 |
