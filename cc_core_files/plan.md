# NeuroHorizon 项目分析与执行计划

**日期**：2026-02-21
**项目**：NeuroHorizon — 统一神经编码模型
**基础代码**：POYO/POYO+ (NeurIPS 2023)
**代码库**：/root/autodl-tmp/NeuroHorizon

---

## 1. 项目目标

NeuroHorizon 旨在构建一个**统一的神经编码模型**，同时解决计算神经科学中的两大核心挑战：

1. **跨 Session 鲁棒性**：实现梯度无关的跨 session 泛化（不同记录会话、不同神经元群体）
2. **长时间跨度预测**：将神经活动预测从传统 ~200ms 扩展到 1 秒以上

**目标发表**：NeurIPS / ICLR / Nature Methods

---

## 2. 与 POYO 的关键差异

| 维度 | POYO/POYO+ | NeuroHorizon |
|------|-----------|-------------|
| **核心任务** | 行为解码 (spikes → behavior) | 神经编码 (spikes → future spikes) |
| **Unit Embedding** | Per-unit 可学习嵌入 (InfiniteVocabEmbedding) | IDEncoder (从参考窗口特征前馈生成) |
| **输出机制** | 单次 cross-attention → 线性读出 | 自回归 cross-attention decoder |
| **时间分辨率** | Spike 级输入 | Spike 级输入 + Binned spike count 输出 |
| **预测跨度** | N/A（解码任务） | 500ms - 1000ms |
| **跨 Session 能力** | 有限（需微调） | 梯度无关（IDEncoder 零样本泛化） |
| **多模态** | 无 | 行为 + 图像 (DINOv2) |
| **损失函数** | MSE / CE | Poisson NLL |

---

## 3. 合理性评估

### 3.1 优势

- **研究方向有价值**：跨 session 泛化和长时间预测是 BCI 和计算神经科学的核心需求
- **代码基础扎实**：POYO+ 框架模块化程度高，Encoder-Processor-Decoder 架构便于扩展
- **数据集选择合理**：
  - IBL (459 sessions, 241 brain regions) → 跨 session 泛化 + scaling law
  - Allen (58 sessions, 多刺激模态) → 多模态融合实验
- **创新点明确**：IDEncoder (SPINT 思路用于生成任务) + 自回归 decoder + 多模态融合 的组合是新颖的

### 3.2 已识别的问题与应对

#### 问题 1: Jia Lab 数据不可用 [已确认]
- **影响**：提案中列为长时间预测核心数据源（★★★，大量重复 trial）
- **应对**：完全使用 IBL 连续记录（策略 B：不对齐 trial 的滑动窗口）+ Allen Natural Movies（30s 连续视频）替代
- **风险等级**：中。IBL 和 Allen 数据仍然可以验证长时间预测，但缺少大量重复 trial 下的 PSTH 可靠性评估

#### 问题 2: Allen Natural Scenes 时间间隔
- **问题**：每张图片展示 250ms + 500ms 灰屏间隔
- **影响**：1s 预测窗口会跨越多个刺激边界，预测内容混合了刺激响应和灰屏衰减
- **应对**：优先使用 Natural Movies 刺激集（30s 连续视频，无间隔）

#### 问题 3: 可变输出维度
- **问题**：spike count 输出维度 = 该 session 的 neuron 数，每个 session 不同
- **影响**：现有 MultitaskReadout 假设固定输出维度，无法直接使用
- **应对**：设计 shared per-neuron MLP head，输入 = concat(bin_repr, unit_emb)，输出 = 1 (log-rate)，自然适应任意 n_units

#### 问题 4: 自回归误差累积
- **问题**：20ms bin、1s 预测 = 50 步自回归，误差逐步放大
- **应对**：
  - 训练：Scheduled sampling（逐步减少 teacher forcing 比例）
  - 对照：Non-autoregressive parallel prediction 基线
  - 策略：Coarse-to-fine（先 100ms 分辨率，再精细化）

#### 问题 5: 计算资源
- **当前**：单卡 4090
- **应对**：
  - 开发全程使用 Small (5M params)
  - 正式实验 Base (30M) 需 BF16 + gradient checkpointing
  - Large (100M) 需梯度累积或升级 GPU
  - 后续可扩展资源

#### 问题 6: rotary_attention 不支持 causal mask
- **问题**：现有 `RotarySelfAttention` 的 mask 处理为 `(b, 1, 1, n)` 形状，仅支持 KV masking，不支持自回归所需的 causal mask `(b, 1, n_q, n_kv)`
- **应对**：修改 `rotary_attn_pytorch_func` 中 mask 的 reshape 逻辑，`F.scaled_dot_product_attention` 已原生支持 2D mask

---

## 4. 架构设计

### 4.1 整体架构

```
输入窗口 [0, T_in]                          预测窗口 [T_in, T_in + T_pred]
──────────────────                          ────────────────────────────
    │                                                │
    ▼                                                ▼
[Spike Tokens]                              [Spike Count Targets]
    │                                        (binned, 20ms bins)
    ▼
[IDEncoder] 参考窗口特征 → Unit Embeddings
    │
    ▼
[Token Embedding] = IDEncoder(unit_idx) + TokenType + RoPE
    │
    ▼
[Perceiver Cross-Attention] spikes → latents
    │
    ▼
[Self-Attention Processing] × depth 层
    │  (可选：每隔2层插入 Multimodal Cross-Attention)
    │  (图像 DINOv2 embedding / 行为数据)
    │
    ▼
[Autoregressive Cross-Attention Decoder] × N_dec 层
    │  每层: Cross-Attn(bins→latents) + Causal Self-Attn(bins) + FFN
    │
    ▼
[Per-Neuron MLP Head] concat(bin_repr, unit_emb) → log-rate
    │
    ▼
[Poisson NLL Loss] 对比预测 log-rate 与真实 spike count
```

### 4.2 IDEncoder

替换 `InfiniteVocabEmbedding`，从参考窗口统计特征生成 unit embedding：

- **输入特征** (~33 维)：
  - 平均发放率 (1d)
  - ISI 变异系数 (1d)
  - ISI log-histogram (20d)
  - 自相关特征 (10d)
  - Fano factor (1d)
- **网络结构**：3 层 MLP (Linear + GELU + LayerNorm)
- **核心优势**：新 session 的新 neuron 只需计算统计特征即可获得 embedding，无需微调

### 4.3 自回归解码器

```
对每个预测 time bin b = 1..B:
  1. bin_query = bin_type_emb + RoPE(t_b)
  2. cross_attend(bin_query, encoder_latents)
  3. causal_self_attend(bin_1..bin_b)  // 只看之前的 bins
  4. per_neuron_head: concat(bin_repr, unit_emb) → log_rate
```

### 4.4 模型规模

| 配置 | Encoder层 | Decoder层 | 隐藏维度 | Attention Heads | 参数量 |
|------|-----------|-----------|---------|-----------------|--------|
| Small | 4 | 2 | 256 | 4 | ~5M |
| Base | 8 | 4 | 512 | 8 | ~30M |
| Large | 12 | 6 | 768 | 12 | ~100M |

---

## 5. 数据集

### 5.1 IBL Brain-wide Map [主要]
- **规模**：459 sessions, 139 mice, 12 labs
- **神经数据**：621,733 total units, 75,708 good quality units
- **脑区覆盖**：241 个脑区
- **任务**：标准化视觉决策任务（判断光栅对比度，转动滚轮）
- **用途**：跨 session 泛化、scaling law、长时间预测
- **下载**：ONE API (AWS)，免注册

### 5.2 Allen Brain Observatory Neuropixels [次要]
- **规模**：58 sessions
- **刺激类型**：Natural Scenes (118张), Natural Movies (30s), Drifting Gratings 等
- **用途**：多模态（图像）融合实验
- **下载**：AllenSDK (AWS NWB)

### 5.3 存储预估
- IBL (spike + behavior): ~100-200 GB
- Allen (NWB): ~150 GB
- 总计: ~300-350 GB

---

## 6. 执行计划

### Phase 0: 环境与数据准备
1. **0.1 环境扩展**: 安装 ONE-api, ibllib, allensdk, scipy
2. **0.2 IBL 数据下载**: `scripts/download_ibl.py` — 先 10-20 sessions，后扩展
3. **0.3 Allen 数据下载**: `scripts/download_allen.py` — brain_observatory_1.1
4. **0.4 数据预处理**: `scripts/preprocess_ibl.py`, `scripts/preprocess_allen.py` — 转 HDF5
5. **0.5 参考特征提取**: `scripts/extract_reference_features.py` — IDEncoder 输入
6. **0.6 数据验证**: `scripts/validate_data.py`

### Phase 1: POYO 基线验证
1. **1.1 IBL 适配**: 在 IBL 数据上运行现有 POYO+ 模型
2. **1.2 基线指标**: 验证 wheel velocity 解码 R² > 0

### Phase 2: 核心模型实现
1. **2.1 PoissonNLLLoss**: 修改 `torch_brain/nn/loss.py`
2. **2.2 spike_counts 模态**: 修改 `torch_brain/registry.py`
3. **2.3 IDEncoder**: 新建 `torch_brain/nn/id_encoder.py`
4. **2.4 NeuroHorizon 模型**: 新建 `torch_brain/models/neurohorizon.py`
5. **2.5 训练流程**: 新建 `examples/neurohorizon/train.py` + configs
6. **2.6 评估指标**: 新建 `torch_brain/utils/neurohorizon_metrics.py`

### Phase 3: 多模态扩展
1. **3.1 DINOv2 注入**: 编码器层间 multimodal cross-attention
2. **3.2 行为条件**: wheel velocity 等行为数据注入
3. **3.3 Allen Dataset**: 多模态数据集类

### Phase 4: 实验
1. **实验 1**: 跨 Session 泛化（核心贡献）
2. **实验 2**: 数据 Scaling Laws (10/50/100/200/459 sessions)
3. **实验 3**: 长时间预测 (100/200/500/1000ms)
4. **实验 4**: 多模态贡献分析
5. **实验 5**: 下游任务泛化
6. **实验 6**: 消融实验 (10 variants)

### Phase 5: 分析与论文
1. 结果收集与可视化
2. 论文写作

---

## 7. 最小可行发表路径 (MVP)

优先级排序：
1. IBL 数据管线 → 核心 NeuroHorizon 模型 → 跨 session 泛化实验 → 长时间预测实验 → 关键消融

MVP 仅需 IBL 数据 + 核心模型 + 2 个实验 + 消融即可构成可发表的工作。Allen 多模态实验作为补充贡献。

---

## 8. 关键文件清单

### 需要修改的文件
| 文件 | 修改内容 |
|------|----------|
| `torch_brain/nn/loss.py` | 添加 PoissonNLLLoss |
| `torch_brain/nn/__init__.py` | 导出 IDEncoder |
| `torch_brain/nn/rotary_attention.py` | 支持 causal mask |
| `torch_brain/registry.py` | 注册 spike_counts 模态 |
| `torch_brain/models/__init__.py` | 导出 NeuroHorizon |

### 需要新建的文件
| 文件 | 内容 |
|------|------|
| `torch_brain/nn/id_encoder.py` | IDEncoder 模块 |
| `torch_brain/models/neurohorizon.py` | NeuroHorizon 模型 |
| `torch_brain/utils/neurohorizon_metrics.py` | 评估指标 |
| `scripts/download_ibl.py` | IBL 数据下载 |
| `scripts/preprocess_ibl.py` | IBL 数据预处理 |
| `scripts/download_allen.py` | Allen 数据下载 |
| `scripts/preprocess_allen.py` | Allen 数据预处理 |
| `scripts/extract_dino_embeddings.py` | DINOv2 特征提取 |
| `scripts/extract_reference_features.py` | 参考特征提取 |
| `scripts/validate_data.py` | 数据验证 |
| `examples/neurohorizon/train.py` | 训练脚本 |
| `examples/neurohorizon/datasets/ibl.py` | IBL Dataset |
| `examples/neurohorizon/datasets/allen_multimodal.py` | Allen Dataset |
| `examples/neurohorizon/configs/*` | Hydra 配置文件 |

---

## 9. 风险与应对总结

| 风险 | 等级 | 应对策略 |
|------|------|----------|
| Jia Lab 数据不可用 | 高 | IBL + Allen Natural Movies 替代 |
| 50 步自回归误差累积 | 高 | Scheduled sampling + parallel baseline |
| 4090 显存限制 | 中 | Small 开发 + BF16 + gradient checkpointing |
| Allen Natural Scenes 时间间隔 | 中 | 优先 Natural Movies |
| AllenSDK 依赖冲突 | 低 | 独立环境下载，主环境加载 HDF5 |
| IBL 数据下载速度 | 低 | 分批下载，先 10-20 sessions |
