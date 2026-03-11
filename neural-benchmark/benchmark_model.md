# Benchmark 对比模型调研文档

> **日期**：2026-03-12
> **目的**：为 NeuroHorizon 1.8 Benchmark 对比实验整理各对比模型的技术信息，评估适配方案
> **参考**：`cc_todo/phase0-env-baseline/20260309-phase0-0.4-benchmark-analysis.md` §0.4.2–§0.4.3

---

## 1. NDT2 (Neural Data Transformer 2)

### 基本信息

| 属性 | 说明 |
|------|------|
| **GitHub** | [joel99/context_general_bci](https://github.com/joel99/context_general_bci) |
| **论文** | NeurIPS 2023 |
| **License** | MIT |
| **架构类型** | Encoder-Decoder Transformer, Masked Autoencoding (MAE) |

### 模型大小

| 配置 | Hidden Size | Heads | Layers | Dropout | 参数量 |
|------|------------|-------|--------|---------|--------|
| Small | 128 | — | 6 | 0.6 | 约 1.5M |
| **Base** | **256** | **4** | **6** | **0.1** | **约 3.7M**（encoder 3M + decoder 0.7M） |
| Large | 512+ | 8+ | 8+ | 0.1 | 8M+（估算） |

### 资源需求

- **训练 GPU**：推荐 40GB+ VRAM（A100 / V100），batch_size=64
- **推理 GPU**：约 2GB（FP16），可部署至边缘设备
- **训练时间**：未明确公布，SLURM 配置 48h time limit

### Inference Code & Pretrained Weights

- **Inference 代码**：✅ 完整的训练和推理脚本
- **预训练权重**：✅ 通过 wandb 提供（Google Drive 链接在 README）
  - Human data checkpoint、Indy multisession RTT checkpoint、FALCON baseline checkpoint
  - 格式：PyTorch `.pt`（Lightning 格式）
  - 加载方式：`model.load_from_checkpoint(<path>)`

### Forward Prediction 支持

- **是否支持**：✅ 是（NLB fp-bps 是其核心评估任务之一）
- **实现方式**：**MAE 并行填充**——将 future bins 全部 mask，一次性重建所有 masked positions 的 firing rates（非逐步自回归）
  - 预训练时使用 50% random masking
  - 推理时将 forward window 的所有位置替换为 mask tokens
  - Encoder 处理可见 tokens，decoder 在 masked positions 重建
- **观察窗口**：默认 2500ms（125 timesteps @ 20ms bins）
- **预测窗口**：可配置，NLB 标准为 200ms（10 bins @ 20ms）
- **可配置性**：✅ 可通过修改 config 调整 observation/prediction 窗口长度

### fp-bps 参考数值

- NDT2 默认 random masking → fp-bps 约 0.50（IBL-MtM 论文消融实验数据）
- NDT2 + 专用 causal masking → fp-bps 约 0.52
- 注意：上述数值来自 IBL repeated-site 数据，非 Perich-Miller 运动皮层

### 数据输入格式

- **Spike 表示**：20ms binned spike counts（50Hz 时间分辨率）
- **输入 tensor shape**：`(Batch, Time, Channels, 1)` 或 tokenized 模式 `(B, T, SpatialTokens, C_per_token)`
- **Spatial patching**：neurons_per_token=32，将多个神经元分组为一个 spatial token
- **Padding**：channel 对齐到 token 边界，时间维 pad_sequence 处理变长 trial

### torch_brain 适配方案

**数据适配**（spike events → NDT2 输入）：
1. 从 torch_brain 读取 `spikes.timestamps` + `spikes.unit_index`
2. `np.histogram` 按 20ms 进行 binning → `(n_units, n_bins)` 矩阵
3. 转置 + unsqueeze → `(T, C, 1)` tensor
4. Spatial patching：按 neurons_per_token 分组
5. 添加 batch 维度

**评估对接**（NDT2 输出 → 统一评估）：
- NDT2 decoder 输出 log_rate/rate at masked positions
- 取 forward window 对应位置 → reshape 为 `[B, T, N]`
- 直接送入 `neurohorizon_metrics.py` 的 `fp_bps()`, `r2_score()`, `psth_r2()`

**估算工作量**：约 100 行 adapter 代码，1–2 天

---

## 2. Neuroformer

### 基本信息

| 属性 | 说明 |
|------|------|
| **GitHub** | [a-antoniades/Neuroformer](https://github.com/a-antoniades/Neuroformer) |
| **论文** | ICLR 2024 |
| **License** | MIT |
| **架构类型** | Decoder-only (GPT-style), Autoregressive Spike Generation |

### 模型大小

| 配置 | Embedding Dim | Heads | Layers | 参数量 |
|------|-------------|-------|--------|--------|
| V1+AL | 256 | 8 | 6 | 约 8–12M（含多模态组件） |
| Visnav | 512 | 8 | 8 | 约 40–100M |

### 资源需求

- **训练 GPU**：分布式多 GPU（5 × GPU，BS=32/GPU，总 BS=160）
- **推理 GPU**：约 2–4GB 单 GPU
- **训练时间**：250 epochs，具体时长未公布

### Inference Code & Pretrained Weights

- **Inference 代码**：✅ 提供 `neuroformer_inference.py`
  - 用法：`python neuroformer_inference.py --dataset lateral --ckpt_path "model_dir" --predict_modes speed phi th`
  - 提供 `load_model_and_tokenizer()` 工具函数
- **预训练权重**：⚠️ 仓库中有 checkpoint，但仅适用于 V1/AL 钙成像数据
  - 格式：PyTorch `.pt`
  - **无运动皮层 electrophysiology 权重**——需从零训练

### Forward Prediction 支持

- **是否支持**：✅ 是（架构原生支持，是其核心训练目标 SCLM）
- **实现方式**：**逐 spike 自回归生成**（GPT-style causal Transformer）
  - 每个 token 是一个 spike event：先预测 neuron ID，再预测 time interval (dt)
  - 推理时自回归逐步生成 spike tokens，直到填满 prediction window
  - 输出为离散 spike events，**非直接的 binned firing rates**
  - 需后处理：spike events → histogram binning → log_rates
- **观察窗口**：可配置，V1AL 默认 current=0.05s + previous=0.15s（总约 0.2s）
- **预测窗口**：开放式生成，无固定上限
- **可配置性**：✅ 通过 YAML config 调整 current_state_window、previous_state_window

### fp-bps 参考数值

- **未使用 NLB fp-bps 标准指标**
- 论文报告指标：Population Vector Correlation (PVC)、PSTH 相关性、行为解码 R²（speed r=0.97）
- 无可与 NLB 标准直接对比的 fp-bps 数值

### 数据输入格式

- **Spike 表示**：tokenized 为 (neuron_ID, time_delta) 对
- **Tokenizer**：自定义 `Tokenizer` 类，含 SOS/EOS/PAD 特殊 tokens
- **时间离散化**：均匀 binning，dt=0.01s（V1AL）或 dt=0.005s（Visnav）
- **输入结构**：dict with `spikes: (n_neurons, n_timesteps)`, `frames`, `behavior`, `train/test_intervals`

### torch_brain 适配方案

**数据适配**（spike events → Neuroformer 输入）：
1. 从 torch_brain 读取 `spikes.timestamps` + `spikes.unit_index`
2. 按时间排序，计算连续 spike 间的 time delta
3. Tokenize：`[neuron_ID_token_1, delta_token_1, ...]`
4. Pad/truncate 到固定序列长度
5. 使用 Neuroformer 的 `Tokenizer` 类处理

**评估对接**（Neuroformer 输出 → 统一评估）：
- 模型生成 spike tokens → 解码为 (neuron_ID, timestamp) 对
- histogram binning（20ms bins）→ spike counts per bin per neuron
- log(counts + epsilon) → log_rates `[B, T, N]`
- 送入统一评估接口

**估算工作量**：约 150 行 adapter 代码，2–3 天

**关键风险**：
- ⚠️ Neuroformer 从未在运动皮层 electrophysiology 数据上测试
- ⚠️ 逐 spike 自回归在高放电率 Neuropixels 数据上步数极多，推理效率低
- ⚠️ 需从零训练，无现成权重可用

---

## 3. IBL-MtM (IBL Multi-task Masking Model)

### 基本信息

| 属性 | 说明 |
|------|------|
| **GitHub** | [colehurwitz/IBL_MtM_model](https://github.com/colehurwitz/IBL_MtM_model) |
| **论文** | NeurIPS 2024 |
| **License** | MIT |
| **架构类型** | Encoder-only Transformer, Multi-task Masking |

### 模型大小

| 变体 | 架构 | 参数量 | 说明 |
|------|------|--------|------|
| NDT1（单 session） | 6-layer encoder | 约 3M | 单 session 基线 |
| NDT1-stitch（多 session） | 6-layer encoder + per-session linear proj | 约 25.55M（34 session） | 含 session 级投影层 |
| NDT2（单 session） | 6-layer encoder + 2-layer decoder | 约 1.09M | 论文中 NDT1 优于 NDT2 |

### 资源需求

- **训练 GPU**：RTX8000 / V100（32GB VRAM）
- **训练时间**：
  - 单 session 微调：3–6 小时
  - 10 session 联合训练：约 1 天
  - 34 session 联合训练：1–3 天（NDT1）/ 约 5 天（NDT2）
- **Batch size**：16
- **优化器**：AdamW, lr=1e-4, cosine scheduler, 1000 epochs

### Inference Code & Pretrained Weights

- **Inference 代码**：✅ 完整的训练、评估、微调脚本
- **预训练权重**：✅ HuggingFace（`ibl-foundation-model` 组织）
  - 可用模型：multi-NDT1-MtM-{10,34}-sessions、multi-NDT2-MtM-{10,34}-sessions、baseline 版本
  - ⚠️ **访问限制**：需要 `ibl-foundation-model` 组织成员权限
  - 格式：通过 HuggingFace `transformers` API 加载
- **替代方案**：可以从零训练，单 session 仅需 3–6 小时

### Forward Prediction 支持

- **是否支持**：✅ 是（temporal causal masking 是 4 种 masking 任务之一）
- **实现方式**：**Temporal causal masking + MAE 并行预测**
  - 将时间窗口末尾 K 个 bins 全部 mask
  - 一次性重建所有 masked future bins（非逐步自回归）
  - 通过可学习 "task prompt" token 区分 masking 任务类型
  - 4 种 masking：temporal / neuron / intra-region / inter-region，联合训练
- **观察窗口**：约 1.3–1.8s（65–90 bins @ 20ms）
- **预测窗口**：约 200ms（10 bins，时间窗口末尾约 10–15%）
- **可配置性**：✅ 可调整 mask 比例和位置，但需修改 masking 逻辑

### fp-bps 参考数值

| 配置 | fp-bps |
|------|--------|
| NDT2 baseline（random masking） | 约 0.50 |
| NDT2 + causal masking | 约 0.52 |
| NDT2 + 4-task MtM（含 causal） | **约 0.54** |
| NDT1 单 session + MtM（prompted） | **0.57** |
| NDT1 多 session 34-pretrained + stitch | 0.46 |

注：上述数值来自 IBL repeated-site Neuropixels 数据（视觉/海马，多脑区）

### 数据输入格式

- **Spike 表示**：20ms binned spike counts
- **输入 tensor shape**：`[batch, time_steps, num_neurons]`
- **时间分辨率**：20ms bins（50Hz）
- **Trial 长度**：2 秒（100 time bins）
- **Token budget**：最大 2K tokens（GPU 显存限制）
- **Session dict**：需包含 session_id、subject_id、task_id（learned embeddings）
- **神经元数量**：每 session 约 200–1000 个，NDT1-stitch 用 per-session linear proj 处理变长

### torch_brain 适配方案

**数据适配**（spike events → IBL-MtM 输入）：
1. 从 torch_brain 读取 `spikes.timestamps` + `spikes.unit_index`
2. `np.histogram` 按 20ms 进行 binning → `(n_bins, n_units)` 矩阵
3. 转为 tensor `[batch, time_steps, num_neurons]`
4. 添加 session context metadata（session_id, subject_id）

**评估对接**（IBL-MtM 输出 → 统一评估）：
- 模型在 masked positions 输出 predicted firing rates
- 取 temporal mask 对应的 forward window → `[B, T, N]`
- 直接送入统一评估接口

**估算工作量**：约 120 行 adapter 代码，2 天

**关键风险**：
- ⚠️ 预训练权重 domain mismatch（IBL visual/hippocampal → Perich-Miller motor cortex）
- ⚠️ Multi-region masking 对单 area 录制不太相关
- ⚠️ HuggingFace 权重需组织成员权限，可能需要从零训练

---

## 模型综合对比

| 维度 | NDT2 | Neuroformer | IBL-MtM |
|------|------|-------------|---------|
| **参数量** | 3.7M（base） | 8–100M | 3M–25.55M |
| **架构** | Encoder-Decoder, MAE | Decoder-only, GPT | Encoder-only, MtM |
| **FP 方式** | 并行填充 | 逐 spike 自回归 | temporal causal mask + 并行 |
| **时间分辨率** | 20ms bins | 10ms dt（可配置） | 20ms bins |
| **观察窗口** | 2.5s（默认） | 0.2s（V1AL 默认） | 1.3–1.8s |
| **预测窗口** | 可配置 | 开放式 | 约 200ms |
| **预训练权重** | ✅ wandb/GDrive | ⚠️ 仅钙成像 | ✅ HuggingFace（需权限） |
| **运动皮层经验** | ✅ RTT, FALCON | ❌ 从未测试 | ❌ 仅 IBL visual/hippocampal |
| **fp-bps 参考** | 约 0.50（IBL 数据） | 无 | 0.46–0.57（IBL 数据） |
| **GPU 需求** | 40GB+（训练） | 多 GPU 分布式 | 32GB（V100/RTX8000） |
| **适配难度** | **低** | **中** | **中** |
| **推荐度** | ⭐⭐⭐ | ⭐ | ⭐⭐ |

---

## 备选模型（待评估）

> 以下模型作为后续可能的对比对象，当前阶段不纳入 benchmark，视需要扩展。

### NEDS (Neural Encoding and Decoding System)
- **状态**：待评估
- **简介**：用于神经数据编解码的统一框架
- **GitHub**：待查

### STNDT (Spatiotemporal NDT)
- **状态**：待评估
- **简介**：在 NDT 基础上增加空间建模能力
- **GitHub**：待查
- **与 NDT2 关系**：同系列，NDT2 已包含 spatiotemporal patching

### NDT3
- **状态**：待评估
- **GitHub**：[joel99/ndt3](https://github.com/joel99/ndt3)
- **论文**：2025, bioRxiv
- **简介**：NDT2 的后续版本，扩展到 100M+ 参数，更强的跨 session 泛化能力
- **关注点**：NDT2 README 推荐关注此版本作为更新、更强的替代

---

## 参考链接

- [NDT2 GitHub](https://github.com/joel99/context_general_bci) | [Paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC10541112/)
- [NDT3 GitHub](https://github.com/joel99/ndt3) | [Paper](https://www.biorxiv.org/content/10.1101/2025.02.02.634313v1)
- [Neuroformer GitHub](https://github.com/a-antoniades/Neuroformer) | [Paper](https://arxiv.org/abs/2311.00136)
- [IBL-MtM GitHub](https://github.com/colehurwitz/IBL_MtM_model) | [Paper](https://arxiv.org/abs/2407.14668)
- [NLB Benchmark](https://neurallatents.github.io/)
- NeuroHorizon 内部参考：`cc_todo/phase0-env-baseline/20260309-phase0-0.4-benchmark-analysis.md`
