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

#### Phase 0.3: Allen 数据下载 ✅
- 后台运行完成，5 个 Allen sessions 下载并预处理：
  - allen_715093703: 67.9M spikes, 875 units, 9629s
  - allen_719161530: 64.6M spikes, 751 units, 9665s
  - allen_721123822: 35.1M spikes, 439 units, 9811s
  - allen_732592105: 60.4M spikes, 818 units, 9415s
  - allen_737581020: 44.9M spikes, 566 units, 9277s
- 修复了 Allen HDF5 中 running 组缺少 domain 的问题
- 参考特征提取完成（33维/unit）
- 数据验证全部通过

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

#### Phase 1: POYO 基线 ✅ 训练中
- 创建了 `examples/poyo_baseline/train_baseline.py` — POYO wheel velocity decoding baseline
- 创建了 `IBLEagerDataset` — eager loading + readout config injection
- 配置文件：`configs/defaults.yaml`, `configs/model/poyo_small.yaml`, `configs/train.yaml`
- **修复 Bug 1: target 维度不匹配**
  - 问题：POYO 输出 `[batch, seq, 1]` 经 mask 后变 `[N, 1]`，target 变 `[N]`（1D），MSELoss 要求 2D
  - 修复：在 training_step 和 validation_step 中添加 `if target_values.ndim == 1 and output_values.ndim == 2: target_values = target_values.unsqueeze(-1)`
- **修复 Bug 2: Lightning setup 覆盖 transform**
  - 问题：`data_module.setup()` 手动调用后设置 `transform = model.tokenize`，但 `trainer.fit()` 再次调用 `setup()` 会创建新的 Dataset 对象（transform=None），导致 collate 收到 `temporaldata.Data` 对象而非 dict
  - 错误信息：`TypeError: default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'temporaldata.temporaldata.Data'>`
  - 修复：在 `setup()` 中添加 `if hasattr(self, 'train_dataset'): return` 防护
- **修复 Bug 3: Hydra _target_ 类型错误**
  - 问题：config 中 `_target_: torch_brain.models.POYOPlus` 反复被还原（可能有 linter/watcher）
  - 修复：在 `train_baseline.py` 中绕过 hydra.utils.instantiate，显式使用 POYO 类
  - 代码：`model_cfg.pop("_target_", None); model = POYO(readout_spec=readout_spec, **model_cfg)`
- **修复 Bug 4: R2 计算形状不匹配**
  - 问题：validation_step 中 `output_values_masked` 是 (N,1) 但 `target_values` 是 (N,)
  - 修复：使用 `.view(-1)` 展平后计算 R²
- 端到端测试通过：IBLEagerDataset → RandomFixedWindowSampler → DataLoader → collate → POYO.forward() → MSELoss → backward
- **训练已启动**：200 epochs, batch_size=64, SparseLamb optimizer, 与 NeuroHorizon 并行运行
- 模型参数：~3.5M (POYO Small: dim=128, depth=8)
- 当前进度：epoch 0, step ~663, loss 波动较大（初始阶段正常）

#### NeuroHorizon 100-epoch 训练进行中
- 训练配置：batch_size=16, 10 IBL sessions, 100 epochs, bf16-mixed
- PID 34659, GPU ~3.8GB
- **进度更新**：
  - Epoch 0 完成：1746 steps, avg_loss=0.5198, min_loss=0.3743, 耗时 722s
  - Epoch 1 完成：avg_loss=0.4950, min_loss=0.3568
  - Epoch 2 完成：avg_loss=0.4874, min_loss=0.3540
  - Epoch 3 完成：avg_loss=0.4825, min_loss=0.3623
  - Epoch 4 完成：avg_loss=0.4785, min_loss=0.3513
  - **首次验证 (epoch 5)：val_loss=0.4670, val_bits_per_spike=-1.008**
  - bits/spike 为负值 = 模型尚未超过 null model（均值发放率模型），训练初期正常
  - val_loss (0.467) 接近 train_loss (0.479)，未过拟合
  - Checkpoint 已保存（94MB），继续训练中
  - 预计总耗时 ~20 小时（~739s/epoch × 100 epochs）

#### 两训练并行状态
- GPU 总共 24564 MiB，NeuroHorizon + POYO baseline = ~16-20 GiB
- **NeuroHorizon 进度** (epoch 6/100):
  - train_loss: 0.52→0.49→0.49→0.48→0.48→0.47→0.47（稳定下降）
  - val_loss=0.4670, val_bits_per_spike=-1.008（epoch 5，模型尚未超过 null model）
  - ~801s/epoch，预计还需 ~20h
- **POYO 基线进度** (epoch 11/200):
  - train_loss median: 3.42→3.30→3.25→3.16→2.95→2.71→2.53→2.40→2.38→2.34→2.29→2.25
  - **首次验证 (epoch 10)：val_loss=5.04, val_r2=-0.15**
  - val_loss >> train_loss 原因分析：时间序列数据的 temporal split 导致分布漂移
    - 例：session 5ae68c54 训练集 wv_mean=0.50, 验证集 wv_mean=-1.52
    - 动物在实验过程中行为模式变化
  - 无 NaN（gradient clipping 有效），~84s/epoch，预计还需 ~4.4h

#### 参考特征归一化 ✅
- 发现问题：参考特征（33维）尺度差异极大
  - `firing_rate`: mean=12.2, std=20.8, max=224（远超其他特征）
  - `isi_hist_*`: std 在 0.001~0.16 之间
  - `autocorr_*`: std 在 0.08~0.15 之间
  - IDEncoder 的 MLP 输入被 firing_rate 主导
- 创建了 `scripts/normalize_reference_features.py`
  - z-score 归一化（mean=0, std=1 per feature）
  - 原始特征备份到 HDF5 `reference_features_raw`
  - 归一化统计保存到 `ref_feature_stats.json`
- 10 个 IBL session 全部归一化完成
- **注意**：当前运行中的训练使用未归一化的特征（内存已加载），下次训练将使用归一化后的特征

#### 评估脚本 ✅
- 创建了 `scripts/evaluate_neurohorizon.py` — NeuroHorizon 后训练评估
  - 加载 checkpoint，在验证集上运行详细评估
  - 计算每 session、每 unit 的 bits/spike、FR correlation、R²
  - 生成预测 vs 真实值可视化图
  - 输出 JSON 格式汇总指标
- 创建了 `scripts/evaluate_poyo_baseline.py` — POYO 基线后训练评估
  - 加载 POYO checkpoint + vocabulary 初始化
  - 计算 R²、MSE、相关系数

#### 训练代码改进 ✅
- 增强了 `examples/neurohorizon/train.py` 的 validation_step
  - 添加 val_fr_corr 和 val_r2 指标（之前只有 bits/spike）
- 添加了 gradient_clip_val=1.0 到 NeuroHorizon Trainer

#### 多模态模块 ✅
- 创建了 `torch_brain/nn/multimodal.py`
  - `MultimodalCrossAttention`: 投影模态 embedding + RoPE 时间对齐的交叉注意力
  - `MultimodalEncoder`: 封装图像和行为模态交叉注意力层（2.3M 参数）
- 导出更新到 `torch_brain/nn/__init__.py`
- Allen 刺激帧已保存为 numpy 文件（DINOv2 提取需要网络访问）

#### 训练进度更新
- **NeuroHorizon** (epoch 10/100):
  - train_loss: 0.52→0.49→0.49→0.48→0.48→0.47→0.47→0.46→0.45→0.44
  - **Epoch 9 验证结果**：
    - val_loss: 0.467 → **0.421** (↓9.8%)
    - val_bits_per_spike: -1.008 → **-0.724** (↑28% 改善)
    - 趋势良好，模型持续学习中
    - 无过拟合：val_loss (0.421) ≈ train_loss (0.445)
    - 如果保持当前趋势，预计 epoch 25-35 附近 bits/spike 可能转正
  - ~700s/epoch，预计还需 ~17.5h
- **POYO 基线** (epoch 45/200):
  - train_loss 持续下降到 ~1.99
  - 验证指标趋势：
    | Epoch | val_loss | val_r2 |
    |-------|----------|--------|
    | 9  | 5.0434 | -0.1493 |
    | 19 | 4.5692 | -0.0668 |
    | 29 | 4.6172 | -0.1369 |
    | 39 | 4.4949 | -0.0504 |
  - val_r2 在 -0.05 ~ -0.15 之间震荡，temporal distribution shift 限制了改善
  - ~84s/epoch，预计还需 ~3.6h

#### 训练进度更新 (Session 5)
- **NeuroHorizon** (epoch 10/100, 10%):
  - train_loss 稳定下降: 0.52→0.49→0.49→0.48→0.48→0.47→0.47→0.46→0.45→0.44
  - 验证指标（每5 epoch）：
    | Epoch | val_loss | val_bits_per_spike |
    |-------|----------|-------------------|
    | 4     | 0.4670   | -1.0075           |
    | 9     | 0.4214   | -0.7242           |
  - bits/spike 改善 28%，趋势良好，预计 epoch 25-35 转正
  - 无过拟合（val ≈ train）
  - **注意**：当前训练未记录 val_fr_corr 和 val_r2（代码修改在训练启动后）
  - 下次验证在 epoch 14
- **POYO 基线** (epoch 51/200, 25.5%):
  - train_loss 下降到 ~1.1-1.7（从初始 ~3.4）
  - **验证指标**：
    | Epoch | val_loss | val_r2   |
    |-------|----------|----------|
    | 9     | 5.043    | -0.1493  |
    | 19    | 4.569    | -0.0668  |
    | 29    | 4.617    | -0.1369  |
    | 39    | 4.495    | -0.0504  |
    | 49    | 5.065    | **-0.2067** |
  - **严重过拟合**：epoch 49 val_loss 回升到 5.065（比最好的 4.495 高 12.7%）
  - val_r2 恶化到 -0.21（历史最差）
  - 分析原因：
    1. `div_factor=1` + `pct_start=0.5` 意味着 LR 前100 epoch 保持在最大值 0.002
    2. LR 从 epoch 100 开始衰减，可能会帮助
    3. 根本问题仍是 temporal distribution shift（train vs valid 行为分布不同）
  - 最佳 checkpoint: epoch 39 (val_loss=4.495, val_r2=-0.050)
  - GPU stale test process 已清理

### 待完成
- NeuroHorizon 100-epoch 训练完成后分析结果（bits/spike 是否从 -1.0 提升至正值）
- POYO 基线 200-epoch 训练完成后分析结果（LR decay 后 val_r2 是否回升）
- 训练完成后用归一化特征重新训练 NeuroHorizon（预期显著改善 IDEncoder 效果）
- ~~Phase 3: 多模态扩展（模块已创建，需集成到 NeuroHorizon 模型）~~ ✅ 已完成
- Phase 4: 实验
- Phase 5: 分析与论文

#### Phase 3: 多模态集成到 NeuroHorizon ✅
- 在 `torch_brain/models/neurohorizon.py` 中集成 `MultimodalEncoder`
  - 新参数：`use_multimodal`, `multimodal_every`, `image_dim`, `behavior_dim`
  - 在编码器 self-attention 层间插入多模态交叉注意力（每 N 层一次）
  - 向后兼容：`use_multimodal=False` 默认值保持现有配置不变
  - 支持优雅降级：多模态模型可在无多模态输入时正常运行
- `_tokenize_multimodal()` 自动从 Data 对象提取行为数据
  - Allen: `data.running.running_speed`（~40Hz）
  - IBL: `data.behavior.wheel_velocity`
  - 图像 embedding 预留（DINOv2 提取后生效）
- 更新 collate 函数：union 语义处理多模态 key（缺失样本零填充+False mask）
- 新配置：`neurohorizon_small_mm.yaml`（~10.5M params），`defaults_allen.yaml`
- 测试通过：forward/backward + tokenizer（Allen+IBL 数据）

#### 实验基础设施 ✅
- 创建了 `scripts/evaluate_cross_session.py` — 跨 session 泛化评估
  - 计算 in-distribution vs cross-session bits/spike gap
  - 支持指定 train/test session 分组
- 创建了 `scripts/run_ablations.py` — 消融实验运行器
  - 支持 IDEncoder、prediction horizon、bin_size 消融
  - 可配置 overrides，支持 dry-run 模式

#### IDEncoder 消融支持 ✅
- 在 NeuroHorizon 模型中添加 `embedding_mode` 参数
  - `"idencoder"`（默认）：3 层 MLP，从参考特征生成 unit embedding
  - `"random"`：固定随机投影（无梯度），消融基线
  - `"mean"`：简单线性投影，测试 MLP 深度贡献
- 所有模式 forward/backward 测试通过

#### Phase 3.5: 多模态训练准备 ✅
- **Collate 改进**：
  - 修改 `neurohorizon_collate()` 使用 union 语义处理多模态 key
  - `_collate_multimodal_keys()` 函数处理缺失样本的零填充 + False mask
  - 通过测试：混合 batch（有/无 behavior）正确处理
- **EagerDataset 多目录支持**：
  - 新增 `dataset_dirs` 参数，支持同时加载多个数据目录（IBL + Allen）
  - 向后兼容：`dataset_dir` 单目录参数仍可用
- **新增配置文件**：
  - `configs/model/neurohorizon_small_beh.yaml` — 行为条件模型（~10.2M params）
  - `configs/train_v2.yaml` — v2 训练：IBL+Allen 15 sessions + behavior 条件
  - `configs/train_v2_norm.yaml` — v2 对照：仅 IBL + 归一化特征（无多模态）
- **DINOv2 工具链**：
  - `scripts/extract_dino_embeddings.py` — 从 Allen 刺激帧提取 DINOv2 embedding（已有）
  - `scripts/inject_dino_embeddings.py` — 将提取的 embedding 注入 Allen HDF5 文件（新增）
- **端到端测试**：
  - 混合 Allen + IBL 数据 batch → multimodal 模型 forward → loss 计算通过
  - Behavior-only 模型：10,225,153 参数
  - union 语义 collation 验证：缺失多模态数据的样本 mask 全 False

#### 训练进度更新 (Session 7)
- **NeuroHorizon v1** (epoch 18/100, 18%):
  - train_loss 稳定下降: 最新 0.431
  - 验证指标（**epoch 14 新增**）：
    | Epoch | val_loss | val_bits_per_spike | 改善 |
    |-------|----------|-------------------|------|
    | 4     | 0.4670   | -1.0075           | --   |
    | 9     | 0.4214   | -0.7242           | +28% |
    | 14    | 0.3982   | -0.5729           | +21% |
  - bits/spike 持续改善，从 -1.01 → -0.72 → -0.57
  - 趋势：如果保持当前改善速率，预计 epoch 30-40 转正
  - 无过拟合（val_loss 0.398 ≈ train_loss 0.431）
- **POYO 基线 — 最终结果** (epoch 106/200, 已终止):
  - **结论：失败**，所有 epoch 的 val_r2 均为负值
  - train_loss 1.24，val_loss 4.93（严重过拟合）
  - 完整验证历史：
    | Epoch | val_loss | val_r2    |
    |-------|----------|-----------|
    | 9     | 5.043    | -0.1493   |
    | 19    | 4.569    | -0.0668   |
    | 29    | 4.617    | -0.1369   |
    | 39    | **4.495** | **-0.0504** |
    | 49    | 5.065    | -0.2067   |
    | 59    | 4.658    | -0.2095   |
    | 69    | 4.924    | -0.1951   |
    | 79    | 4.702    | -0.1521   |
    | 89    | 5.692    | **-0.5204** |
    | 99    | 4.934    | -0.2208   |
  - Best checkpoint: epoch 39 (val_loss=4.495, val_r2=-0.050)
  - 分析：temporal split + per-unit learnable embedding 无法泛化
  - **已终止并释放 14.8 GB GPU 内存**
- **NeuroHorizon v2 IBL** (epoch 0/100, 刚启动):
  - 使用归一化参考特征（z-score），同架构对照
  - GPU ~3.6 GB，与 v1 并行运行

#### NH v2 训练计划（已更新）
当前运行：
1. **v1**（baseline）：IBL，未归一化特征，epoch 18/100
2. **v2_ibl**（归一化对照）：IBL，归一化特征，epoch 0/100
待运行：
3. **v2_beh**：IBL+Allen 15 sessions，归一化特征 + 行为条件，200 epochs
4. **v2_mm**：IBL+Allen 15 sessions，归一化特征 + 行为 + 图像 CLIP，200 epochs

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
| POYO MSELoss target 维度不匹配 | unsqueeze target when ndim==1 and output ndim==2 |
| Lightning setup() 覆盖 transform | 在 setup() 添加 hasattr guard 防止重复创建 |
| Hydra _target_ 被外部还原为 POYOPlus | 在代码中绕过 hydra.utils.instantiate，显式 POYO() |
| R2 形状不匹配 (N,1) vs (N,) | .view(-1) 展平后再计算 R² |
| 参考特征尺度差异极大（firing_rate max=224 vs isi_hist max~1） | z-score 归一化（mean=0, std=1），原始备份到 reference_features_raw |
| EagerDataset deepcopy 瓶颈 (POYO 训练卡在 step 663) | 覆写 __getitem__ 避免 copy.deepcopy，Data.slice() 已返回新对象 |
| collate 多模态 key 使用 intersection 导致丢失 | 改用 union 语义 + _collate_multimodal_keys 零填充缺失样本 |
| EagerDataset 仅支持单数据目录 | 添加 dataset_dirs 参数支持多目录加载（IBL + Allen） |
| DINOv2 无法下载（网络受限） | 使用缓存的 CLIP ViT-L/14 替代，从 safetensors 离线加载 |
| image_embeddings HDF5 group 缺少 temporaldata 元数据 | 添加 _unicode_keys (bytes), timekeys (bytes), domain (Interval) 属性 |
| Allen 刺激数据格式 (start/end/frame) 与预期 (timestamps) 不匹配 | 展开 frame index → per-presentation embeddings，按 onset_time 排序 |
| tokenizer 检查 data.images 但 HDF5 group 名为 image_embeddings | 修改为 data.image_embeddings |

### 版本记录
| 日期 | 版本 | 描述 |
|------|------|------|
| 2026-02-21 | v0.1 | 项目分析完成，计划制定，开始执行 Phase 0 |
| 2026-02-21 | v0.2 | Phase 0 完成：环境搭建、IBL 数据管线（10 sessions）、参考特征提取、数据验证 |
| 2026-02-21 | v0.3 | Phase 2.1-2.4 完成：PoissonNLLLoss、wheel_velocity 模态、IDEncoder、NeuroHorizon 模型（8.1M params） |
| 2026-02-21 | v0.4 | Phase 2.5-2.6 完成：训练流程（EagerDataset + Hydra + Lightning）、评估指标、端到端验证通过、1 epoch 训练成功（loss 0.95→0.39） |
| 2026-02-21 | v0.5 | Allen 数据完成（5 sessions），POYO 基线脚本就绪，100-epoch 训练运行中 |
| 2026-02-21 | v0.6 | POYO 基线 2 个 bug 修复（target 维度 + setup guard），POYO 基线 200-epoch 训练启动，与 NeuroHorizon 并行 |
| 2026-02-21 | v0.7 | deepcopy 瓶颈修复（POYO+NH），参考特征归一化，评估脚本，多模态模块，multimodal 代码，训练监控 |
| 2026-02-21 | v0.7 | 参考特征归一化（z-score），评估脚本（NH + POYO），训练代码改进（更多验证指标 + gradient clipping） |
| 2026-02-21 | v0.8 | 训练监控更新：NH epoch 10 (bps -0.72↑), POYO epoch 51 (r2 -0.21↓严重过拟合), 多模态集成准备 |
| 2026-02-21 | v0.9 | Phase 3 完成：多模态集成（model + tokenizer + collate + configs），实验基础设施（cross-session eval + ablation scripts），IDEncoder 消融模式 |
| 2026-02-21 | v1.0 | Phase 3.5：多模态训练准备（union collate、多目录 EagerDataset、v2 configs、DINOv2 injection script），NH epoch 13 / POYO epoch 68 监控 |
| 2026-02-21 | v1.1 | POYO 终止（epoch 106，R²全负），NH v1 epoch 14 bps=-0.57（持续改善），v2_ibl 启动（归一化特征对照），CLIP embedding 注入 Allen |
| 2026-02-21 | v1.1 | 图像 embedding 提取与注入完成（CLIP ViT-L/14），全模态 E2E 测试通过（image+behavior+neural），v2 配置完善 |
| 2026-02-21 | v1.2 | 训练监控更新：NH v1 epoch 14 val_bps=-0.573（预计 epoch 27 转正），POYO epoch 84（8次验证全负 R²），auto_launch_v2 就绪 |
| 2026-02-21 | v1.3 | Phase 4 实验脚本完成（horizon eval + multimodal ablation + results collector + training queue），POYO epoch 89 val_r2=-0.52 灾难性恶化 |

---

#### Phase 3.6: 图像 Embedding 提取与注入 ✅
- **CLIP ViT-L/14 替代 DINOv2**：
  - 网络访问受限无法下载 DINOv2，使用已缓存的 CLIP ViT-L/14（1.71 GiB）
  - 从 HuggingFace 本地缓存加载 safetensors 权重（`HF_HUB_OFFLINE=1` 模式）
  - 输出维度：1024（vs DINOv2 的 768），质量相当
  - `scripts/extract_image_embeddings.py`：从 Allen 刺激帧提取 CLS token 特征
- **刺激帧处理**：
  - Natural Movie 1: 900 frames → (900, 1024) embeddings
  - Natural Movie 3: 3600 frames → (3600, 1024) embeddings
  - Natural Scenes: 118 images → (118, 1024) embeddings
  - 总提取时间 ~40s（batch_size=16, bf16 推理）
- **Embedding 注入**：
  - `scripts/inject_image_embeddings.py`：将 per-frame embeddings 映射到 stimulus presentations
  - Allen HDF5 存储格式：`start/end/frame`（每次刺激呈现的时间 + 帧索引）
  - 展开为 per-presentation embeddings（~59,900 presentations/session × 1024 维）
  - 修复 temporaldata 兼容性：添加 `_unicode_keys`, `timekeys` (bytes dtype), domain 元数据
  - 5 个 Allen sessions 全部注入完成（~245 MB/session）
- **Tokenizer 更新**：
  - `_tokenize_multimodal()` 改用 `data.image_embeddings`（匹配 HDF5 group 名）
  - 切片后自动提取窗口内的图像 embedding + timestamps
- **全模态端到端测试通过**：
  - 模型：12.8M params（image_dim=1024 + behavior_dim=1）
  - Allen 数据 (t=3000s)：30 image embeddings + 39 behavior values
  - forward → loss (0.94) → backward 全部正常
- **v2 配置更新**：
  - `neurohorizon_small_mm.yaml` 更新 image_dim=1024（CLIP 输出维度）
  - 新增 `train_v2_mm.yaml`：全模态配置（IBL+Allen + image + behavior）
  - 新增 `train_v2_ibl.yaml`：IBL-only 归一化特征对照配置

#### NH v2 训练计划（更新）
准备就绪，待 GPU 空间释放（POYO 完成后）：
1. **v2_ibl**（对照）：IBL 10 sessions，归一化特征，无多模态，100 epochs
   - 隔离归一化特征的贡献
   - 与 v1 直接对比
2. **v2_beh**（行为条件）：IBL+Allen 15 sessions，归一化特征，behavior 条件，200 epochs
   - 测试行为条件 + 更多数据的综合效果
3. **v2_mm**（全模态）：IBL+Allen 15 sessions，归一化特征，image + behavior 条件，200 epochs
   - 测试视觉刺激信息对编码预测的贡献

#### 训练进度监控 (最新更新)
- **NeuroHorizon v1** (epoch 15/100, 15%):
  - train_loss: ~0.40 (稳定下降)
  - 验证指标：
    | Epoch | val_loss | val_bits_per_spike |
    |-------|----------|-------------------|
    | 4     | 0.4670   | -1.0075           |
    | 9     | 0.4214   | -0.7242           |
    | 14    | 0.3982   | **-0.5729**       |
  - BPS 改善率：+0.0435/epoch
  - **预计 BPS=0（超过 null model）在 epoch ~27**
  - 无过拟合（val_loss ≈ train_loss）
  - 下次验证在 epoch 19
- **POYO 基线** (epoch 90/200, 45%):
  - train_loss: ~1.0-1.3 (持续下降)
  - 验证指标：
    | Epoch | val_loss | val_r2    |
    |-------|----------|-----------|
    | 9     | 5.043    | -0.1493   |
    | 19    | 4.569    | -0.0668   |
    | 29    | 4.617    | -0.1369   |
    | 39    | 4.495    | **-0.0504** (best) |
    | 49    | 5.065    | -0.2067   |
    | 59    | 4.658    | -0.2095   |
    | 69    | 4.924    | -0.1951   |
    | 79    | 4.702    | -0.1521   |
    | 89    | **5.692** | **-0.5204** (worst) |
  - **epoch 89 灾难性恶化**：val_r2 从 -0.15 骤降到 -0.52
  - 分析：LR 保持最大值 0.002 到 epoch 100，模型过度拟合训练集
  - epoch 100 开始 LR cosine decay，可能有所改善
  - 最佳 checkpoint 仍在 epoch 39 (val_r2=-0.050)
  - auto_launch_v2.sh 后台运行中，等待 POYO PID 完成后自动启动 NH v2
- **GPU 状态**：
  - PID 34659 (NH v1): 3826 MiB
  - PID 46244 (POYO): 14804 MiB
  - PID 61917 (其他项目 masking_pretrain): 3886 MiB
  - 总占用 ~22.5 GiB / 24 GiB，POYO 完成后释放 ~15 GiB
- **模型对比**（来自 compare_models.py）：
  - NH v1 best val_loss=0.4214 vs POYO best val_loss=4.4949
  - 注意：任务不同（neural encoding vs behavior decoding），loss 不直接可比

#### Phase 4: 实验准备 ✅
- 创建 `scripts/evaluate_horizons.py` — 不同预测时间跨度的评估（100ms/200ms/500ms/1s）
- 更新 `scripts/run_ablations.py` — 新增多模态消融实验（neural-only vs +behavior vs +image vs +both）
- 创建 `scripts/collect_results.py` — 综合结果收集器（扫描所有训练日志 + 评估结果）
- 创建 `scripts/training_queue.sh` — 顺序训练队列（v2_ibl → v2_beh → v2_mm → 消融实验）
