# Phase 0 · 0.1 环境验证与代码理解

**日期**：2026-02-28
**对应 plan.md 任务**：Phase 0 → 0.1.1 / 0.1.2 / 0.1.3
**任务目标**：验证开发环境完整性、深度理解 POYO 代码架构、精读 SPINT 和 Neuroformer 论文、输出后续代码改造建议

---

## 0.1.1 POYO conda 环境可用性确认

### 执行结果

服务器现有 conda 环境（通过 `/root/miniconda3`）：

| 环境 | 用途 |
|------|------|
| `base` | /root/miniconda3 |
| `allen` | Allen SDK 专用 |
| `livetalk` | 无关项目 |
| `poyo` | **NeuroHorizon 主开发环境** |

### 核心依赖验证（`poyo` 环境）

| 依赖 | 版本 | 状态 |
|------|------|------|
| PyTorch | 2.10.0+cu128 | ✅ |
| CUDA | cu128 / RTX 4090 D | ✅ |
| wandb | 0.25.0 | ✅ |
| hydra | 1.3.2 | ✅ |
| omegaconf | 2.3.0 | ✅ |
| einops | 0.6.1 | ✅ |
| h5py | 3.15.1 | ✅ |
| numpy | 2.2.6 | ✅ |
| scipy | 1.15.3 | ✅ |
| matplotlib | 3.10.8 | ✅ |
| scikit-learn | 1.7.2 | ✅ |
| torch_brain | dev | ✅ |
| brainsets | 0.2.1.dev4 | ✅ |

**BF16 验证**（RTX 4090 D 支持 BF16）：已通过 CUDA device check 确认。

### 代码模块依赖关系图

```
Spike 输入
    │
    ├── data/  (brainsets → HDF5 → temporaldata.Data)
    │   ├── dataset/dataset.py    ← HDF5 lazy-loading
    │   ├── data/collate.py       ← pad8 / chain / track_mask8
    │   └── transforms/           ← UnitDropout / RandomCrop / TimeScaling
    │
    ├── utils/tokenizers.py       ← create_start_end_unit_tokens
    │                                create_linspace_latent_tokens
    │
    ├── nn/
    │   ├── infinite_vocab_embedding.py  ← unit_emb, session_emb
    │   │   (含 tokenizer / detokenizer / vocab 管理 / state_dict hook)
    │   ├── embedding.py                 ← token_type_emb, latent_emb, task_emb
    │   ├── rotary_attention.py          ← RotaryCrossAttention / RotarySelfAttention
    │   │   └── rotary_attn_pytorch_func / rotary_attn_xformers_func
    │   ├── feedforward.py               ← FeedForward (GEGLU)
    │   ├── multitask_readout.py         ← MultitaskReadout + prepare_for_multitask_readout
    │   └── loss.py                      ← MSELoss / CrossEntropyLoss / MallowDistanceLoss
    │
    ├── registry.py                      ← ModalitySpec / MODALITY_REGISTRY / register_modality
    │
    ├── models/poyo_plus.py              ← POYOPlus (完整 encoder-processor-decoder + tokenize)
    │   Flow: unit_emb+token_type_emb → enc_atn(cross) → proc_layers(self) → dec_atn(cross) → readout
    │
    └── optim.py                         ← SparseLamb (session_emb 专用优化器)
```

**关键代码路径（已阅读验证）**：

- `models/poyo_plus.py:forward()` — 完整前向传播，7步清晰
- `models/poyo_plus.py:tokenize()` — CPU侧 tokenizer，生成 model_inputs 字典
- `nn/rotary_attention.py:rotary_attn_pytorch_func()` — attn_mask 当前只处理 `(B, N_kv)` 1D mask，reshape 为 `b () () n`
- `nn/rotary_attention.py:rotary_attn_xformers_func()` — mask 被 broadcast 为 `(b, h, n, m)`
- `nn/infinite_vocab_embedding.py` — 包含 tokenizer/detokenizer/vocab/state_dict hooks，不能简单替换为 `nn.Embedding`

---

## 0.1.2 SPINT 与 Neuroformer 论文精读

### SPINT（Wei et al., 2024）— IDEncoder 跨 Session 泛化

**核心机制**：
- 不为每个神经元学习固定 ID，而是从神经元的 firing statistics 动态推断身份
- 输入：每个神经元在短暂校准数据（calibration trials）上的统计特征���spike counts、mean firing rate、variance/std）
- 网络结构：**1 层 cross-attention + 2 个三层全连接网络**（比 proposal 中描述的简单 MLP 更复杂）
- 推理时：只需将新 session 的校准数据前向传播一次，无需梯度更新

**关键创新 — Dynamic Channel Dropout**：
- 训练时随机 sample dropout rate，模拟不同 session 神经元数量和组成的变化
- 使模型对 population composition 变化具备鲁棒性
- 是实现跨 session 泛化的重要正则化手段

**与 NeuroHorizon 方案的对比**：

| 方面 | SPINT | NeuroHorizon（计划） |
|------|-------|---------------------|
| 输入特征 | spike counts + firing rate + variance | ~33d（ISI histogram + autocorr + Fano factor） |
| 网络结构 | cross-attn + 2×三层FC | 简单 3层MLP |
| 是否 gradient-free | 是 | 是 |
| 激活函数 | 未指定（标准GELU） | 标准GELU（非GEGLU，因输入维度低）|

> 注：NeuroHorizon 使用更丰富的 33d 统计特征（ISI histogram 等），理论上比 SPINT 的简单 firing stats 更具判别力。若效果不理想，可考虑借鉴 SPINT 的 cross-attention 结构替代简单 MLP。

### Neuroformer（Antoniades et al., ICLR 2024）— 自回归生成 + 多模态

**输入序列格式**：
- 每个神经元分配唯一 token ID（可学习 embedding table）
- Spike 序列分为 "Current State"（近期活动）和 "Past State"（历史活动）两段
- 视觉刺激：3D 卷积 → patch embedding（标准 ViT 路径）
- 行为数据：作为独立模态

**自回归机制（三阶段）**：
1. **多模态对比对齐**：neural + visual + behavior → 共享 latent space（对比学习）
2. **跨模态融合**：级联注意力将神经活动与历史特征、其他模态融合
3. **因果 Spike 解码**：causal-masked Transformer 自回归预测（每步预测"哪个神经元"+"何时发放"）

**关键差异（Neuroformer vs NeuroHorizon）**：

| 方面 | Neuroformer | NeuroHorizon |
|------|-------------|--------------|
| 预测粒度 | 逐 spike event（next spike identity + timestamp） | 固定时间 bin 内的 spike count |
| 自回归步 | 每步生成一个 spike event | 每步预测一个 time bin 所有神经元的 count |
| 多模态注入 | 对比学习对齐（需额外训练阶段） | DINOv2 embedding 直接 cross-attn（更简单） |
| 跨 session | 较弱（per-session embedding table） | IDEncoder（梯度free泛化） |
| 条件注入位置 | cross-attn 引入行为和图像 | 同为 cross-attn（但无需对比学习预训练） |

**性能参考（Neuroformer 论文）**：
- 行为解码 Pearson r：Lateral 0.95，Medial 0.97（vs MLP baseline 0.83-0.85）
- 1% 微调数据的预训练模型优于 10% 数据的非预训练模型

---

## 0.1.3 后续阶段代码改造建议

基于 0.1.1（代码架构精读）和 0.1.2（SPINT + Neuroformer 论文精读）的综合判断。

### Phase 1 改造建议（自回归改造）

#### 1. `torch_brain/nn/loss.py` — 新增 PoissonNLLLoss

**改造方式**：在现有 `MSELoss` 和 `CrossEntropyLoss` 之后追加新类。

**接口设计**（与现有 Loss 基类一致）：
```python
class PoissonNLLLoss(Loss):
    def forward(self, log_rate: Tensor["batch dim"], target: Tensor["batch dim"],
                weights: Optional[Tensor["batch"]] = None) -> Tensor:
```

**关键注意点**：
- 输入 `log_rate`（非 rate），避免 exp 下溢；目标 `target` 为非负 float（spike counts）
- 数值稳定：公式 `loss = exp(log_rate) - target * log_rate`，log_rate 需在 [-10, 10] 内（在 head 输出后 clamp）
- 低发放率神经元（< 1 Hz）的 0-spike bins 不会导致 log(0)，因为公式不含 log(target)
- 与 weights 的乘法：`(weights * loss_noreduce).sum() / weights.sum()`（与 MSELoss 保持一致）

#### 2. `torch_brain/registry.py` — 注册 spike_counts 模态

**改造方式**：在文件末尾添加 `register_modality("spike_counts", ...)` ���用。

**关键注意点**：
- `dim` 参数：等于神经元数量 N——但 N 在不同 session 中可变，需要动态处理。
  建议将 dim 设为 1（per-neuron 输出），在 PerNeuronMLPHead 内处理维度；或通过 per-session readout 动态分配
- `timestamp_key`：`"spikes.timestamps"`（参考 brainsets 字段命名）
- `value_key`：`"spike_counts.counts"`（新字段，需要 binning pipeline 生成）
- `loss_fn`：`PoissonNLLLoss()`

#### 3. `torch_brain/nn/rotary_attention.py` — 支持 causal mask ⚠️ 重点

**关键问题（代码精读发现）**：
- `rotary_attn_pytorch_func` 中：`attn_mask = rearrange(attn_mask, "b n -> b () () n")` — 只处理 1D kv-padding mask
- `rotary_attn_xformers_func` 中：`repeat(attn_mask, "b m -> b h n m", ...)` — 同样只处理 1D mask
- causal mask 为 2D `(N, N)` 上三角矩阵，需要不同的 reshape 路径

**改造方案**：
```python
# rotary_attn_pytorch_func 中
if attn_mask is not None:
    if attn_mask.ndim == 2:          # (B, N_kv) 原有 padding mask
        attn_mask = rearrange(attn_mask, "b n -> b () () n")
    elif attn_mask.ndim == 3:        # (B, N_q, N_kv) causal mask 或 full mask
        attn_mask = rearrange(attn_mask, "b n m -> b () n m")
    # 传入 scaled_dot_product_attention，PyTorch 会正确处理 additive bool mask

# rotary_attn_xformers_func 中
if attn_mask is not None:
    if attn_mask.ndim == 2:          # (B, N_kv) padding
        attn_mask = repeat(attn_mask, "b m -> b h n m", h=num_heads, n=query.size(1))
    elif attn_mask.ndim == 3:        # (B, N_q, N_kv) causal
        attn_mask = repeat(attn_mask, "b n m -> b h n m", h=num_heads)
    attn_bias = attn_mask.to(query.dtype).masked_fill(~attn_mask, float("-inf"))
```

**新增工具函数**：
```python
def create_causal_mask(seq_len: int, device) -> Tensor["seq_len seq_len"]:
    """返回上三角为 False（masked）的 bool mask，下三角（包括对角线）为 True。"""
    return torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).tril()
```

**重要约束**：causal mask 只在**解码器 self-attn** 中使用。Encoder cross-attn、Encoder self-attn（processing layers）、Decoder cross-attn 均保持双向注意力。

#### 4. 新建 `torch_brain/nn/autoregressive_decoder.py`

**核心设计**（方案 D，参考 proposal_review.md 第四节）：
- 输入：`bin_queries [B, T, dim//2]` + `unit_embs [B, N, dim//2]` + `encoder_latents [B, L, dim]`
- concat → `[B, T*N, dim]` → cross-attn(encoder_latents) + causal self-attn（时间维度因果）+ FFN
- 注意：causal mask 需要在 T 维度上应用，N 维度全连接（mask shape = `(T*N, T*N)` 的块对角形式）

**Block causal mask 设计**：
```
T=3, N=2 时：每个时间步包含 N 个 neuron token，因此 causal mask 以 N×N 块为单位：
[ block(0,0) block(-inf) block(-inf) ]
[ block(0,0) block(0,0)  block(-inf) ]
[ block(0,0) block(0,0)  block(0,0)  ]
其中 block(0,0) = N×N 全连接，block(-inf) = N×N 全遮蔽
```

**需要单元测试**：
- Teacher forcing 模式（目标序列作为输入）端到端正确
- 自回归推理模式（循环生成）输出与 teacher forcing 一致（在无噪声条件下���
- causal mask 验证：position t 的输出不依赖 position t+1 及之后的输入

#### 5. 新建 `torch_brain/models/neurohorizon.py`（分 3 步）

- **步骤 a**：复用 POYO encoder + processing layers，预留 IDEncoder 接口（暂用 InfiniteVocabEmbedding），验证 encoder 维度
- **步骤 b**：接入 AutoregressiveDecoder，实现完整 forward()（teacher forcing 模式可运行）
- **步骤 c**：实现 `tokenize()` — 构建 spike count targets（binning spike events），注意 brainsets 字段命名

**binning 注意点**：
- 目标：将 spike event timestamps → bin 化 spike counts（per neuron per bin）
- 工具：`torch_brain/utils/binning.py` 已存在，需确认接口是否可直接用
- output_timestamps：每个 bin 的中心时刻（与 latent timestamps 类似）
- output_values：spike_counts 张量，`dtype=np.float32`（Poisson NLL 接受 float）

#### 6. 新建 `examples/neurohorizon/train.py` + Hydra configs

**Small 配置参数参考**（从 POYOPlus 缩放）：
- `dim=128`, `depth=2`, `cross_heads=1`, `self_heads=4`, `latent_step=0.05`
- Base 配置：`dim=512`, `depth=4`, `cross_heads=2`, `self_heads=8`

**重要**：Base 配置 `cross_heads=2`（非 4），参见 proposal_review.md 勘误。

#### 7. 新建 `torch_brain/utils/neurohorizon_metrics.py`

三个核心指标：
- `psth_correlation(pred_counts, true_counts)` — Pearson r（对齐 trial 后平均）
- `poisson_log_likelihood(log_rate, spike_counts)` — 模型对数似然
- `r2_score(pred_counts, true_counts)` — 行为解码 R²（与 POYO baseline 对比用）

---

### Phase 2 改造建议（IDEncoder）

#### 1. 新建 `scripts/extract_reference_features.py`

**输入特征计算（~33d）**：

| 特征 | 维度 | 计算方法 | 注意事项 |
|------|------|----------|----------|
| 平均发放率 | 1d | total_spikes / total_time | 低发放率（<1 Hz）需特殊处理 |
| ISI 变异系数 | 1d | std(ISI) / mean(ISI) | ISI < 2ms 的 ISI 需过滤（refractory period） |
| ISI log-histogram | 20d | np.histogram(log(ISI), bins=20) | 低发放率时用 KDE 代替直方图（spike 数 < 50） |
| 自相关特征 | 10d | ACF at lags [5,10,...,50ms] | 用 20ms bin 计算 |
| Fano factor | 1d | var(spike_counts) / mean(spike_counts) | 用 100ms bin 计算 |

**SPINT 启示**：SPINT 的输入更简单（仅 spike counts + mean rate + variance），但加了 cross-attention 结构。NeuroHorizon 用更丰富的特征（33d）配合简单 MLP，设计理念不同——更丰富的手工特征 vs 更强的网络结构。若 33d MLP 方案效果不理想，可借鉴 SPINT 加入 cross-attention。

#### 2. 新建 `torch_brain/nn/id_encoder.py`

**网络设计**：
```python
# 输入：~33d 神经统计特征
# 输出：d_model（unit embedding）
class IDEncoder(nn.Module):
    def __init__(self, input_dim=33, output_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),         # 标准 GELU，非 GEGLU（输入维度低，门控机制无必要）
            nn.LayerNorm(128),
            nn.Linear(128, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, output_dim),
        )
```

**关于 SPINT 的 cross-attention 结构**：SPINT 用的是 "1层cross-attn + 2×三层FC"，
这里 cross-attn 的 query 是 unit embedding、key/value 是 calibration spikes（序列形式输入）。
NeuroHorizon 方案是先预计算 33d 统计特征，再用 MLP。如果直接输入 spike 序列（不预计算统计特征），
则需要类似 SPINT 的 cross-attention 结构——可作为 IDEncoder v2 的改进方向。

#### 3. 集成到 NeuroHorizon 模型

**关键约束**：
- `InfiniteVocabEmbedding` 的 `tokenizer()` / `detokenizer()` 接口被 `tokenize()` 和 collate pipeline 大量使用，不能直接删除
- 建议：IDEncoder 作为额外路径，通过 flag `use_id_encoder` 切换，保留原 `InfiniteVocabEmbedding` 路径
- 优化器：IDEncoder 用 AdamW；session_emb（InfiniteVocabEmbedding）保留 SparseLamb

---

### Phase 3 / 4 建议（简要）

- **Phase 3 数据 scaling**：无额外代码改造，主要是训练脚本配置（session 数量参数化）
- **Phase 4 多模态**：DINOv2 embeddings 离线预计算（`scripts/extract_dino_embeddings.py`），
  注入位置：在 encoder cross-attn 之前，通过额外 cross-attn layer 将 DINOv2 features 注入 latents

---

## 文件与结果

**本次创建/修改的文件**：
- `cc_todo/phase0-env-baseline/20260228-phase0-env-code-understanding.md`（本文件）
- `cc_core_files/plan.md`（0.1.3 内容修改 + 乱码修复）

**待完成（0.2 / 0.3 任务）**：
- 数据探索脚本 `scripts/analysis/explore_brainsets.py`（0.2.3）
- POYO+ 基线复现（0.3.1）

---

## 注意事项（经验沉淀）

1. **conda 激活方式**：服务器上需先 `source /root/miniconda3/etc/profile.d/conda.sh` 才能用 conda 命令，直接 ssh 执行不会自动加载
2. **xformers 后端**：服务器有 xformers，所有 CUDA 上的 attention 会走 `rotary_attn_xformers_func` 路径，改 causal mask 时必须同时修改 xformers 路径
3. **PerceiverRotaryLayer 不存在**：`torch_brain.nn.rotary_attention` 中没有 `PerceiverRotaryLayer`，只有 `RotaryCrossAttention` 和 `RotarySelfAttention`
4. **attn_mask 类型**：`rotary_attn_pytorch_func` 中 `F.scaled_dot_product_attention` 的 `attn_mask` 可以是 bool 型（False=masked）或 float 型（加到 attention scores）——两种类型行为不同，causal mask 推荐用 bool 类型（True=保留）
