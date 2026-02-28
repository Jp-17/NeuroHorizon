# NeuroHorizon 执行计划

> **本文档是项目的可执行计划主体。**
>
> - 项目背景与架构分析：`cc_core_files/proposal_review.md`
> - 数据集详细规划：`cc_core_files/dataset.md`
> - 完整研究提案：`cc_core_files/proposal.md`
> - 任务执行记录：`cc_todo/{phase-folder}/{date}-{phase}-{task}.md`

---

## 总览

**总预计周期**：16-20 周（约 4-5 个月）

```
Phase 0: 环境准备与基线复现           ████░░░░░░░░░░░░░░░░░░  [Week 1-2]
Phase 1: 自回归改造验证 + 长时程生成   ░░░░████████░░░░░░░░░░  [Week 3-8]
Phase 2: 跨 Session 测试              ░░░░░░░░████████░░░░░░  [Week 7-12]
Phase 3: Data Scaling + 下游任务      ░░░░░░░░░░░░████░░░░░░  [Week 10-14]
Phase 4: 多模态引入                   ░░░░░░░░░░░░░░░░████░░  [Week 13-17]
Phase 5: 完整实验、消融与论文          ░░░░░░░░░░░░░░░░░░████  [Week 16-20]
```

**最小可行发表路径（MVP）**：
Phase 0-1（环境 + 自回归改造）→ Phase 2（跨 session 泛化）→ Phase 1 长时程实验 → Phase 5 消融
> MVP 仅需 Brainsets 数据 + 核心模型 + 跨 session 泛化实验 + 长时间预测实验 + 关键消融即可构成可发表的工作。Allen 多模态实验作为补充贡献。

---

## Phase 0：环境准备与基线复现

> **目标**：验证开发环境完整性，深度理解 POYO 代码架构，在 Brainsets 数据上建立行为解码 baseline。
> **数据集**：Brainsets 原生（Perich-Miller 2018 为主）
> **cc_todo**：`cc_todo/phase0-env-baseline/`

### 0.1 环境验证与代码理解

- [ ] **0.1.1** 确认并验证 POYO conda 环境可用性
  - 服务器上已有 POYO 相关 conda 环境，先 `conda env list` 查看现有环境，尝试直接激活使用
  - 验证核心依赖完整性：PyTorch, wandb, hydra, brainsets；缺失项按需补装而非重建环境
  - 梳理代码模块依赖关系图：spike tokenization → unit embedding → Perceiver encoder → readout → 训练循环

- [ ] **0.1.2** 精读 SPINT（IDEncoder 机制）和 Neuroformer（自回归生成 + 多模态）两篇关键论文

- [ ] **0.1.3** 标注后续需要修改的代码模块（参见本文档附录"关键文件清单"）

### 0.2 数据准备与探索

- [ ] **0.2.1** 检查 `NeuroHorizon/data/` 中已有的 Brainsets 数据
  - 列出 `data/raw/` 和 `data/processed/` 下的内容，判断是否已下载 Perich-Miller 或其他 Brainsets 数据集
  - 若已有数据：确认格式是否符合 brainsets pipeline 要求，可直接复用
  - 若无或不完整：通过 brainsets API 补充下载 `perich_miller_population_2018`（先 5-10 sessions）
  - 记录数据存放位置到 `cc_core_files/data.md`

- [ ] **0.2.2** 数据加载验证
  - 通过 POYO 数据 pipeline 的 sanity check，确认数据可正常流入训练框架

- [ ] **0.2.3** 数据深度探索与可视化分析

  > 目标：建立对 Perich-Miller 数据集的完整数据直觉，为后续输入/输出窗口设计、自回归可行性评估提供依据。
  >
  > **脚本**：新建 `scripts/analysis/explore_brainsets.py`（记录到 `cc_core_files/scripts.md`）
  > **结果**：图表输出至 `results/figures/data_exploration/`（记录到 `cc_core_files/results.md`）

  - **数据格式与结构**
    - brainsets 数据文件格式（HDF5 / .npy / 其他），字段列表，加载接口

  - **数据集概览统计**
    - 总 sessions 数；各动物（C / J / M）的 session 数分布
    - 每个 session 的 trial 数量、总记录时长
    - 每个 session 的 neuron 数量（最小 / 最大 / 中位数，画分布直方图）

  - **任务结构分析**
    - 任务类型确认（Center-out reaching / Random target reaching 等）
    - Trial 阶段划分及各阶段时长分布（hold period / movement period / rest period）；画直方图
    - Trial 总时长分布；inter-trial interval（ITI）是否存在及其时长
    - 确认"输入窗口 = hold period，预测窗口 = reach period"的自然划分是否成立
    - 各阶段时长是否满足 250ms / 500ms / 1s 的窗口需求（列表汇总）

  - **可用模态梳理**
    - 神经数据：spike times 格式、时间分辨率
    - 行为数据：cursor velocity / position、hand position 等字段是否存在，采样率
    - 辅助信���：trial 标签、目标方向、成功/失败标注等

  - **神经元统计特征**
    - 各 session 平均 firing rate 分布（直方图，多 session 叠加）
    - PSTH 示例图（对齐 trial onset，展示 2-3 个典型 session 的群体平均活动，分 hold / reach 阶段）
    - 单神经元 raster plot 示例（2-3 个神经元，展示 spike 模式的代表性与多样性）
    - Spike count 在不同 bin 宽度（20ms / 50ms / 100ms）下的分布（均值、方差、稀疏度）

  - **自回归可行性评估**
    - spike 稀疏性评估：每 20ms bin 内平均 spike count，判断 Poisson NLL 是否合适
    - Session 间神经元重叠度（brainsets 是否有跨 session 的 neuron 对应关系）
    - 滑动窗口方案（方案 B）的可行性：trial 边界是否会引起异常活动

  - **小结与决策建议**（以文字段落总结）
    - 推荐 Phase 1 初期开发使用的数据子集（哪几个 session）
    - 推荐的 input window / prediction window 长度
    - 潜在问题记录（trial 过短、某些 session 神经元数量不足等）

- [ ] **0.2.4** （可选）下载 NLB MC_Maze 数据（brainsets 内）作为改造后的 sanity check 基准

### 0.3 POYO 基线复现

- [ ] **0.3.1** 在 Perich-Miller 数据上运行现有 POYO+ 行为解码，验证 R² > 0，与论文数值比较
- [ ] **0.3.2** 分析 POYO encoder 输出的 latent representation 质量（PCA / 解码探针）
- [ ] **0.3.3** 记录基线性能报告，作为后续改造前后的对比锚点

---

## Phase 1：自回归改造验证 + 长时程生成验证

> **目标**：在 Brainsets 原生数据上实现核心自回归解码器，验证 causal mask 正确性和不同预测窗口下的生成质量。
> **数据集**：Perich-Miller 2018（Brainsets 原生，5-20 sessions）
> **cc_todo**：`cc_todo/phase1-autoregressive/`

### 1.1 核心模块实现

- [ ] **1.1.1** 添加 Poisson NLL Loss
  - 修改 `torch_brain/nn/loss.py`，实现 `PoissonNLLLoss`
  - 处理数值稳定性（log-sum-exp 技巧，避免 NaN）

- [ ] **1.1.2** 注册 spike_counts 输出模态
  - 修改 `torch_brain/registry.py`，添加 `spike_counts` 模态类型

- [ ] **1.1.3** 修改 RotarySelfAttention 支持 causal mask
  - 修改 `torch_brain/nn/rotary_attention.py`
  - 将 `rotary_attn_pytorch_func` 中 mask reshape 从 `(b,1,1,n)` 改为支持 `(b,1,n_q,n_kv)`
  - 编写单元测试，验证 causal mask 的正确性（未来 token 不可见）

- [ ] **1.1.4** 实现自回归 Cross-Attention Decoder
  - 新建 `torch_brain/nn/autoregressive_decoder.py`
  - 实现：time bin query embedding + RoPE → cross-attention(bin_query, encoder_latents) → causal self-attention(bins) → FFN
  - 实现 Per-Neuron MLP Head：`concat(bin_repr, unit_emb) → log-rate`（自然适应任意 n_units）

- [ ] **1.1.5** 组装 NeuroHorizon 模型
  - 新建 `torch_brain/models/neurohorizon.py`
  - 集成 POYO Encoder + 新 Autoregressive Decoder
  - 更新 `torch_brain/models/__init__.py` 导出

- [ ] **1.1.6** 编写训练脚本与评估指标
  - 新建 `examples/neurohorizon/train.py` + Hydra configs（Small / Base 两套配置）
  - 新建 `torch_brain/utils/neurohorizon_metrics.py`：PSTH 相关性、Poisson log-likelihood、R²

### 1.2 基础功能验证

- [ ] **1.2.1** Teacher forcing 模式训练（5-10 sessions，Small 配置）
  - 验证 loss 收敛、预测 spike count 分布合理
  - 与简单 baseline 对比（PSTH-based prediction、线性预测）

- [ ] **1.2.2** 自回归推理验证
  - 验证 causal mask 在推理时正确（未来 token 不被看到）
  - 检查误差传播情况，记录各步的预测质量衰减

### 1.3 预测窗口梯度测试

- [ ] **1.3.1** 250ms 预测窗口实验（10-20 sessions）
  - 作为基线，记录 PSTH 相关性 / R²
  - 使用方案 A（trial 对齐）：输入 = hold period，预测 = reach period 前 250ms

- [ ] **1.3.2** 500ms 预测窗口实验
  - 视 250ms 结果决定是否引入 scheduled sampling
  - 对比：500ms 下 trial 对齐方案 A vs 滑动窗口方案 B

- [ ] **1.3.3** 1000ms 预测窗口实验（~50 步自回归）
  - 实现 scheduled sampling（逐步减少 teacher forcing 比例）
  - 可选：non-autoregressive parallel prediction 作为对照基线

- [ ] **1.3.4** 预测窗口汇总报告
  - 绘制性能随窗口长度的衰减曲线（写入 `cc_core_files/results.md`）
  - 决策：Phase 2/3 实验的主要预测窗口

---

## Phase 2：跨 Session 测试

> **目标**：实现 IDEncoder，在 Brainsets 数据上验证跨 session 零样本泛化；可选扩展至 IBL 大规模数据。
> **数据集**：Perich-Miller 2018（必做）；IBL（可选扩展，详见 `cc_core_files/dataset.md` 第 3.3 节）
> **前提**：Phase 1 的自回归改造已验证 causal mask 正确、loss 收敛
> **cc_todo**：`cc_todo/phase2-cross-session/`

### 2.1 IDEncoder 实现

- [ ] **2.1.1** 实现参考窗口特征提取
  - 新建 `scripts/extract_reference_features.py`
  - 计算每个 neuron 的统计特征（~33d）：平均发放率(1d)、ISI 变异系数(1d)、ISI log-histogram(20d)、自相关特征(10d)、Fano factor(1d)

- [ ] **2.1.2** 实现 IDEncoder MLP 网络
  - 新建 `torch_brain/nn/id_encoder.py`
  - 网络结构：3 层 MLP（Linear → GELU → LayerNorm），输入 ~33d，输出 d_model

- [ ] **2.1.3** 集成到 NeuroHorizon
  - 替换 `InfiniteVocabEmbedding`，更新 `torch_brain/nn/__init__.py`
  - 验证前向传播维度匹配，end-to-end pipeline 正常运行

### 2.2 IDEncoder 基础验证

- [ ] **2.2.1** 在 Perich-Miller 单动物多 session 上验证特征提取质量
  - 检查特征分布是否合理（不同 session 的同功能 neuron 特征可聚类）
  - 可视化 IDEncoder 生成的 embedding 空间（PCA / t-SNE）

- [ ] **2.2.2** End-to-end pipeline 验证（5-10 sessions）
  - 替换后的 NeuroHorizon 正常训练、loss 收敛，性能不低于 Phase 1 基线

### 2.3 Brainsets 跨 Session 测试（必做）

- [ ] **2.3.1** Train/val/test 划分（按动物）
  - 2 只猴（C、J）用于训练，1 只猴（M）held-out 作为 test
  - 使用 70+ sessions 全量训练

- [ ] **2.3.2** 零样本泛化实验
  - test session 的 neuron：仅通过 IDEncoder 前向传播生成 embedding（不微调）
  - 评估：R² / PSTH 相关性

- [ ] **2.3.3** A/B 对比实验
  - IDEncoder（gradient-free）vs 固定嵌入 baseline（POYO 原始）vs per-session 微调 upperbound
  - 结果写入 `cc_core_files/results.md`

- [ ] **2.3.4** 结果汇总与决策
  - IDEncoder 结果是否足够支持 paper 贡献？
  - 是否需要 IBL 扩展（详见 2.4）？

### 2.4 IBL 跨 Session 扩展（可选）

> **前提**：Phase 2.3 结果令人满意，需更大规模验证跨 session 泛化

- [ ] **2.4.1** 安装 ONE API + ibllib，验证数据管线（下载 10-20 sessions 调试，~5-10GB）

- [ ] **2.4.2** 编写 IBL Dataset 类
  - 新建 `examples/neurohorizon/datasets/ibl.py`
  - 实现滑动窗口策略（不对齐 trial）；质量过滤：仅使用 `clusters.label == 1`

- [ ] **2.4.3** IBL 大规模跨 session 实验
  - 逐步扩展：20 → 50 → 100 sessions（视结果动态调整）
  - 按实验室划分 train/test（12 个 labs 跨实验室泛化）

### 2.5 FALCON Benchmark（可选补充）

> **前提**：Phase 2.3 有基本结论后引入，用于外部标准化对比

- [ ] **2.5.1** 注册并下载 FALCON M1/M2 数据
- [ ] **2.5.2** 在 FALCON 上量化跨 session 泛化改进（与 IDEncoder baseline 对比）

---

## Phase 3：Data Scaling + 下游任务泛化

> **目标**：揭示性能随训练数据量（session 数）的 scaling 规律；验证自回归预训练对行为解码下游任务的迁移增益。
> **数据集**：Perich-Miller 2018（必做）；IBL（可选扩展，需 Phase 2.4 管线就绪）
> **前提**：Phase 2 跨 session 泛化已有基本结论
> **cc_todo**：`cc_todo/phase3-scaling/`

### 3.1 Brainsets Scaling 测试（必做）

- [ ] **3.1.1** 准备不同规模的 Perich-Miller 子集（5 / 10 / 20 / 40 / 70+ sessions）
- [ ] **3.1.2** 每个规模独立训练，记录验证集 PSTH 相关性 / R²
- [ ] **3.1.3** 绘制 scaling 曲线，分析是否存在 power-law 关系
- [ ] **3.1.4** 决策：曲线是否仍在增长？是否需要 IBL 大规模 Scaling？

### 3.2 下游任务泛化（必做）

- [ ] **3.2.1** 用自回归预训练的 NeuroHorizon encoder 初始化（冻结 encoder），微调行为解码 head
- [ ] **3.2.2** 与 POYO 从头训练的行为解码对比（R²），记录迁移增益
- [ ] **3.2.3** 验证自回归预训练是否改善下游解码质量

### 3.3 IBL 大规模 Scaling（可选）

> **前提**：IBL 数据管线在 Phase 2.4 已建立

- [ ] **3.3.1** IBL 30 / 50 / 100 / 200 / 459 sessions scaling 实验（动态扩增，视 scaling 曲线决定）
- [ ] **3.3.2** 跨实验室零样本泛化（12 labs train/test split）
- [ ] **3.3.3** 绘制大规模 scaling curve（论文核心图之一），写入 `cc_core_files/results.md`

---

## Phase 4：多模态引入

> **目标**：实现并验证视觉图像（DINOv2）和行为数据的条件注入，量化不同模态的预测贡献。
> **数据集**：Allen Visual Coding Neuropixels（58 sessions），详见 `cc_core_files/dataset.md` 第 3.5 节
> **前提**：Phase 2/3 的自回归预测和跨 session 泛化已有基本结论
> **cc_todo**：`cc_todo/phase4-multimodal/`

### 4.1 Allen 数据准备

- [ ] **4.1.1** 确认存储空间（需 > 150GB），规划数据下载路径
- [ ] **4.1.2** 安装 AllenSDK（独立 conda 环境，避免依赖冲突）
- [ ] **4.1.3** 下载 Allen Visual Coding Neuropixels NWB 数据（58 sessions）
  - 优先下载 Natural Movies 相关数据
  - 记录下载信息到 `cc_core_files/data.md`
- [ ] **4.1.4** 预处理转 HDF5，主环境直接加载（避免 AllenSDK 依赖进主环境）
- [ ] **4.1.5** 离线预提取 DINOv2 embeddings
  - 新建 `scripts/extract_dino_embeddings.py`
  - 灰度图（918×1174）→ 转 RGB → DINOv2 ViT-B → 缓存为 `.pt`（118 张图像，< 1GB）

### 4.2 多模态数据集与模型实现

- [ ] **4.2.1** 编写 Allen Dataset 类
  - 新建 `examples/neurohorizon/datasets/allen_multimodal.py`
  - 支持 Natural Movies（30s 连续，优先）和 Natural Scenes（250ms + 灰屏）

- [ ] **4.2.2** 实现行为条件注入
  - 在 encoder 指定层添加 behavior cross-attention（linear projection → cross-attn）

- [ ] **4.2.3** 实现 DINOv2 图像条件注入
  - 添加 image cross-attention（与 behavior cross-attention 同接口）
  - DINOv2 权重冻结，仅训练投影层

### 4.3 多模态实验

- [ ] **4.3.1** Allen Natural Movies 长时程连续预测基准
  - 250ms / 500ms / 1s 预测窗口梯度测试（30s 连续无间隔，完全支持）
  - 多脑区（V1, LM, AL 等）对比分析

- [ ] **4.3.2** 图像-神经对齐实验（Natural Scenes + DINOv2）
  - 量化 DINOv2 embedding 对刺激响应预测精度的贡献
  - 对比：neural only vs neural + image

- [ ] **4.3.3** 多模态消融
  - neural only / neural + behavior / neural + image / neural + behavior + image
  - 分析不同模态的独立贡献，写入 `cc_core_files/results.md`

---

## Phase 5：完整实验、消融与论文

> **cc_todo**：`cc_todo/phase5-experiments-paper/`

### 5.1 完整实验矩阵

- [ ] **5.1.1** 长时程预测实验（100ms / 200ms / 500ms / 1000ms 预测窗口系统性对比）
- [ ] **5.1.2** 跨 session 泛化核心实验（含 baseline 全面对比）
- [ ] **5.1.3** Data scaling law 实验（Brainsets + 可选 IBL）
- [ ] **5.1.4** 多模态贡献分析（可选，若 Phase 4 完成）

### 5.2 消融实验

- [ ] **5.2.1** IDEncoder 消融
  - A1: IDEncoder（gradient-free）vs 可学习固定 embedding
  - A2: IDEncoder vs random embedding baseline
- [ ] **5.2.2** Decoder 深度消融（N_dec = 1 / 2 / 4）
- [ ] **5.2.3** 预测窗口长度消融（100ms / 250ms / 500ms / 1000ms）
- [ ] **5.2.4** Scheduled sampling 消融（无 / 固定比例 / 逐步衰减）
- [ ] **5.2.5** Causal decoder vs parallel prediction（非自回归并行预测）对比

### 5.3 Baseline 对比

- [ ] **5.3.1** PSTH-based baseline、线性预测 baseline（最简单基线）
- [ ] **5.3.2** Neuroformer（自回归生成 + 多模态，最接近 NeuroHorizon）
- [ ] **5.3.3** NDT1/NDT2（masked spike prediction，binned counts）
- [ ] **5.3.4** NDT3（IBL 上有公开结果，IBL 实验时引入）
- [ ] **5.3.5** NEDS（同时支持 spike 预测 + 行为解码，与场景最接近）

### 5.4 结果可视化

- [ ] **5.4.1** Scaling curves（session 数 vs 性能折线图）
- [ ] **5.4.2** 预测性能随时间窗口衰减曲线
- [ ] **5.4.3** 模态贡献归因图（多模态消融结果）
- [ ] **5.4.4** IDEncoder embedding 空间可视化（跨 session / 跨动物，PCA / t-SNE）
- 所有图表记录到 `cc_core_files/results.md`

### 5.5 论文撰写

- [ ] **5.5.1** 方法章节初稿（Architecture + IDEncoder + Autoregressive Decoder）
- [ ] **5.5.2** 实验章节初稿（含所有核心图表）
- [ ] **5.5.3** Introduction + Related Work + Discussion 完善
- [ ] **5.5.4** 论文定稿，准备目标会议（NeurIPS / ICLR / Nature Methods）投稿

---

## 附录 A：关键文件清单

### 需修改的现有文件

| 文件 | 修改内容 | Phase |
|------|----------|-------|
| `torch_brain/nn/loss.py` | 添加 PoissonNLLLoss | Phase 1 |
| `torch_brain/nn/__init__.py` | 导出 IDEncoder | Phase 2 |
| `torch_brain/nn/rotary_attention.py` | 支持 causal mask（(b,1,1,n) → (b,1,n_q,n_kv)）| Phase 1 |
| `torch_brain/registry.py` | 注册 spike_counts 模态 | Phase 1 |
| `torch_brain/models/__init__.py` | 导出 NeuroHorizon | Phase 1 |

### 需新建的文件

| 文件 | 内容 | Phase |
|------|------|-------|
| `torch_brain/nn/autoregressive_decoder.py` | 自回归解码器 + Per-Neuron MLP Head | Phase 1 |
| `torch_brain/models/neurohorizon.py` | NeuroHorizon 完整模型 | Phase 1 |
| `torch_brain/utils/neurohorizon_metrics.py` | 评估指标（PSTH 相关性、R² 等）| Phase 1 |
| `torch_brain/nn/id_encoder.py` | IDEncoder 模块 | Phase 2 |
| `scripts/analysis/explore_brainsets.py` | Brainsets 数据深度探索脚本 | Phase 0 |
| `scripts/extract_reference_features.py` | 参考窗口特征提取 | Phase 2 |
| `scripts/extract_dino_embeddings.py` | DINOv2 特征离线预提取 | Phase 4 |
| `scripts/data/validate_data.py` | 数据验证 | Phase 0 |
| `scripts/data/download_ibl.py` | IBL 数据下载 | Phase 2（可选）|
| `scripts/data/preprocess_ibl.py` | IBL 数据预处理 | Phase 2（可选）|
| `scripts/data/download_allen.py` | Allen 数据下载 | Phase 4 |
| `scripts/data/preprocess_allen.py` | Allen 数据预处理 | Phase 4 |
| `examples/neurohorizon/train.py` | 训练脚本 | Phase 1 |
| `examples/neurohorizon/configs/` | Hydra 配置文件（Small / Base）| Phase 1 |
| `examples/neurohorizon/datasets/ibl.py` | IBL Dataset 类 | Phase 2（可选）|
| `examples/neurohorizon/datasets/allen_multimodal.py` | Allen 多模态 Dataset 类 | Phase 4 |

---

## 附录 B：风险与应对

| 风险 | 等级 | 应对策略 |
|------|------|----------|
| IDEncoder 跨 session 效果不佳 | 高 | 增加参考窗口长度；引入额外特征（waveform shape）；换更复杂架构（小型 Transformer）|
| 50 步自回归误差累积 | 高 | Scheduled sampling + parallel baseline；缩短窗口作折中 |
| 4090 显存限制 | 中 | Small 配置开发 + BF16 + gradient checkpointing；Large 需梯度累积 |
| Allen Natural Scenes 时间间隔 | 中 | 优先 Natural Movies；Natural Scenes 明确说明预测跨边界 |
| AllenSDK 依赖冲突 | 低 | 独立 conda 环境下载数据，转 HDF5 后主环境加载 |
| IBL 数据下载速度 | 低 | 分批下载，先 10-20 sessions 调试 |
| POYO 代码修改导致功能回退 | 中 | 每次修改前验证 POYO baseline 不变；git branch 独立开发；关键模块单元测试 |
| IBL 无 POYO 公开 baseline | 中 | 自行建立 IBL baseline，与 NDT3 / NEDS 等已有 IBL 结果对比 |

---

*计划创建：2026-02-28；2026-02-28 更新：补充 Phase 0 环境/数据细节*
