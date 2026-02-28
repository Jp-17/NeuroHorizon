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



### 三大模型对比分析：POYO vs POYOPlus vs CaPOYO

#### 架构总览

| 特性 | POYO (`poyo.py`) | POYOPlus (`poyo_plus.py`) | CaPOYO (`capoyo.py`) |
|------|-----------------|--------------------------|---------------------|
| **论文** | NeurIPS 2023 | ICLR 2025 | ICLR 2025（同 POYO+） |
| **输入数据类型** | Spike events（点过程） | Spike events（点过程） | Calcium traces（连续信号） |
| **输入 embedding** | `unit_emb(dim)` + `token_type_emb` | 同 POYO | `value_map(1->dim/2)` concat `unit_emb(dim/2)` |
| **Start/End tokens** | 有（每个 unit 2 个） | 有 | **无**（钙信号是连续的） |
| **Task embedding** | **无** | 有 `task_emb` | 有 `task_emb` |
| **Readout** | 单 `nn.Linear(dim, out_dim)` | `MultitaskReadout`（多任务头） | `MultitaskReadout` |
| **Decoder query** | `session_emb` | `session_emb + task_emb` | `session_emb + task_emb` |
| **tokenize() 辅助** | `prepare_for_readout` | `prepare_for_multitask_readout` | `prepare_for_multitask_readout` |
| **预训练加载** | 有 `load_pretrained()` | 无 | 无 |
| **工厂函数** | `poyo_mp()` | 无 | 无 |

#### 核心差异详解

**1. 输入表示差异**

POYO / POYOPlus 处理的是**离散 spike events**：每个 spike 是一个 token，embedding 为 `unit_emb(unit_id) + token_type_emb(type)`，外加每个 unit 的 start/end boundary tokens。输入序列长度取决于 spike 数量，高发放率时序列很长。

CaPOYO 处理的是**连续钙信号**（df/f）：输入是 `(T, N)` 矩阵（T 个时间步 x N 个神经元），展平为 `(T*N)` 序列。每个 token 的 embedding 是 `value_map(df_f_value)` concat `unit_emb(unit_id)`，其中 `value_map` 是 `nn.Linear(1, dim//2)`，将标量钙信号值映射到 `dim//2` 维空间，再与 `unit_emb(dim//2)` 拼接得到 `dim` 维输入。

**关键设计差异**：CaPOYO 的 `unit_emb` 维度是 `dim//2`（不是 `dim`），因为另一半由 `value_map` 提供。这意味着 CaPOYO 的 unit identity 信息和 signal amplitude 信息各占一半容量。

**2. Readout 机制差异**

POYO（单任务）：`forward()` 直接输出 `self.readout(output_latents)`，一个 `nn.Linear` 映射到目标维度。

POYOPlus / CaPOYO（多任务）：`forward()` 通过 `MultitaskReadout` 分派——根据 `output_decoder_index` 将不同 token 路由到对应任务的 linear head。每个任务有独立的 `nn.Linear(dim, task_dim)`。`tokenize()` 中通过 `prepare_for_multitask_readout()` 根据 `data.config["multitask_readout"]` 构建多任务输出序列。

**3. tokenize() 流程差异**

POYO / POYOPlus 的 `tokenize()`：
```
spike events -> create_start_end_unit_tokens -> 拼接 -> unit_emb.tokenizer(unit_ids) 映射全局 index
-> create_linspace_latent_tokens -> prepare_for_(multitask_)readout -> pad8/track_mask8
```

CaPOYO 的 `tokenize()`：
```
calcium_traces.df_over_f (TxN) -> rearrange 为 (T*N, 1) -> repeat unit_index (T*N)
-> 无 start/end tokens -> 相同 latent/output 准备流程
```

**4. Encoder-Processor-Decoder 骨架完全相同**

三个模型共享**完全相同**的 encoder-processor-decoder 骨架：
- **Encoder**：1 层 `RotaryCrossAttention`（latents attend to inputs）+ FFN
- **Processor**：`depth` 层 `RotarySelfAttention`（latents self-attend）+ FFN
- **Decoder**：1 层 `RotaryCrossAttention`（outputs attend to latents）+ FFN

参数配置（`dim`, `depth`, `dim_head`, `cross_heads`, `self_heads`, `rotate_value`）三者一致。差异只在输入端（input embedding 方式）和输出端（readout 方式）。

#### NeuroHorizon 的基底选择建议

**结论：基于 POYOPlus 改造，而非 CaPOYO，也不从头写**

| 候选方案 | 评估 |
|---------|------|
| **基于 POYO** | 缺少多任务支持。NeuroHorizon 需要同时处理 spike_counts 预测和行为解码（Phase 3），单任务 readout 不够用 |
| **基于 CaPOYO** | 输入类型不匹配。CaPOYO 处理连续钙信号（dim//2 value_map），NeuroHorizon 处理离散 spike events |
| **基于 POYOPlus** | **推荐**。输入类型匹配（spike events）+ 多任务支持 + task_emb 可区分不同预测任务 |
| **从头写** | 不必要。encoder + processor 骨架与 POYOPlus 完全一致，无需重复实现 |

**具体改造策略**：

1. **复用部分**（直接继承 POYOPlus 的设计）：
   - `unit_emb` / `session_emb` / `token_type_emb` / `task_emb` / `latent_emb` / `rotary_emb` 全套 embedding
   - `enc_atn` + `enc_ffn`（encoder layer）
   - `proc_layers`（processing layers）
   - `InfiniteVocabEmbedding` 的 tokenizer/detokenizer/vocab 管理机制

2. **替换部分**（NeuroHorizon 需要全新实现）：
   - **Decoder**：POYOPlus 用简单 cross-attn -> linear readout；NeuroHorizon 需要**自回归 decoder**（cross-attn + causal self-attn blocks + per-neuron MLP head），架构完全不同
   - **Readout**：POYOPlus 的 `MultitaskReadout` 是 per-task linear layer；NeuroHorizon 需要 per-neuron MLP head（`concat(bin_repr, unit_emb) -> log_rate`）
   - **Loss**：MSE -> PoissonNLL

3. **重写部分**（接口相似但逻辑不同）：
   - **`tokenize()`**：需要新增 spike binning 逻辑（将 spike events -> 固定时间 bin 的 spike counts），构建自回归 targets（输入窗口 + 预测窗口），处理 causal mask 标记
   - **`forward()`**：前半段（encode + process）与 POYOPlus 相同，后半段（decode）完全不同

4. **Phase 2 额外改造**：
   - `unit_emb`（InfiniteVocabEmbedding）-> IDEncoder（MLP），通过 flag 切换
   - 需保留原 `tokenizer()`/`detokenizer()` 接口供 data pipeline 使用

**实现路径**：新建 `torch_brain/models/neurohorizon.py`，**不继承** POYOPlus 类（因为 forward/tokenize 签名差异太大），但**复制** encoder + processor 的构建代码，写全新的 decoder + tokenize。保持相同的 `nn/` 基础组件依赖。

---

## 0.1.2 SPINT 与 Neuroformer 论文精读

### SPINT（Le et al., NeurIPS 2025）— IDEncoder 跨 Session 泛化

> 参考：[arXiv:2507.08402](https://arxiv.org/abs/2507.08402)

**核心机制**：
- 不为每个神经元学习固定 ID embedding，而是从**原始神经活动数据**（calibration trials 的 binned spike counts）动态推断身份
- 输入：每个 unit 的 M 条 calibration trials 的 binned spike counts（20ms bin），每条 trial 插值到固定长度 T（M1/H1: T=1024, M2: T=100）
- **不使用手工统计特征**（不需要 firing rate、ISI、Fano factor 等），直接从原始 spike count 序列中学习 identity 表示
- 推理时：只需将新 session 的 calibration data 前向传播一次（gradient-free），无需微调

**IDEncoder 架构**：

```
输入：X_i^C ∈ ℝ^(M×T)  （unit i 的 M 条 calibration trials，每条长度 T）

E_i = MLP₂( 1/M × Σ_{j=1}^M MLP₁(X_i^{C_j}) )

Step 1: MLP₁(X_i^{C_j}): ℝ^T → ℝ^H     每条 trial 独立映射到隐层空间（3层FC）
Step 2: Mean pooling across M trials       对 M 条 trial 取均值（置换不变）
Step 3: MLP₂: ℝ^H → ℝ^W                  映射到 identity embedding 维度（3层FC）

输出：E_i ∈ ℝ^W  （unit i 的 identity embedding，W = window size）
```

参数规模（Table A3）：Hidden dim H = 1024 (M1/H1) / 512 (M2)；Output dim W = 100 (M1) / 50 (M2) / 700 (H1)

**Identity 注入方式**：
```
Z_i = X_i + E_i   （直接加法）
```
当前 activity window X_i 加上 identity embedding E_i，形成 identity-aware 表示 Z_i，送入后续 cross-attention 解码。

**解码架构**（单层 cross-attention）：
```
Z_in = MLP_in(Z)                                           # 输入投影
Z̃ = Z_in + CrossAttn(Q, LayerNorm(Z_in), LayerNorm(Z_in))  # Q 是可学习 query（B×W）
Z_out = Z̃ + MLP_attn(LayerNorm(Z̃))                         # FFN
Y = MLP_out(Z_out)                                         # 输出投影 → 行为预测
```
其中 Q ∈ ℝ^(B×W) 是 learnable query matrix，B = 行为维度；cross-attention 聚合所有 N_s 个 unit 的信息。

**关键创新 — Dynamic Channel Dropout**：
- 与经典 dropout（固定比例）不同，每个 training iteration 随机采样 dropout rate（0~1 之间均匀采样）
- 以该 rate 随机移除 neural units（整个 channel 置零）
- 模拟不同 session 间 population 组成的变化，是实现跨 session 鲁棒性的关键正则化手段

**关键设计特点**：
- **置换不变性（Permutation Invariance）**：IDEncoder 中的 mean pooling + cross-attention 的 query-independent 结构保证了对 unit 顺序的不变性
- **端到端训练**：IDEncoder + cross-attention decoder 用 MSE 目标联合训练
- **轻量级设计**：仅 1 层 cross-attn + 2 个 3层FC，面向实时 iBCI 部署

**与 NeuroHorizon 方案的对比**：

| 方面 | SPINT | NeuroHorizon（计划） |
|------|-------|---------------------|
| **Identity 输入** | 原始 binned spike counts（T 维，从 calibration trials 插值） | ~33d 手工统计特征（ISI histogram + autocorr + Fano factor） |
| **特征提取方式** | 端到端学习（MLP₁ 从原始数据中自动提取特征） | 手工设计特征 + MLP 映射 |
| **网络结构** | MLP₁(T→H) → mean pool → MLP₂(H→W)（2个3层FC） | 3层 MLP(33→128→256→dim) |
| **输入维度** | 高维（T=100~1024，取决于 trial 长度和 bin size） | 低维（~33d） |
| **是否 gradient-free（推理时）** | 是 | 是 |
| **Identity 注入方式** | 加法（Z = X + E） | 替换 unit_emb（作为 encoder 输入的一部分） |

> **对比分析**：两种方案代表了不同的设计哲学。SPINT 直接从原始 spike count 序列中学习 identity，特征提取能力更强但输入维度更高（T=100~1024），需要更多 calibration 数据；NeuroHorizon 使用手工统计特征（~33d），输入紧凑，不依赖 trial 结构（可从任意参考窗口提取），但特征设计的质量直接决定上限。
>
> **启发**：若 NeuroHorizon 的 33d 手工特征 + MLP 方案效果不理想，可考虑：(1) 借鉴 SPINT 的思路，直接用参考窗口的 raw spike counts（插值到固定长度）作为 IDEncoder 输入；(2) 或采用混合方案：手工统计特征 + 短窗口 raw counts 拼接。

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

**SPINT 启示**：SPINT 直接将原始 binned spike counts（插值到固定长度 T=100~1024）输入 MLP₁+MLP₂ 端到端学习 identity，不使用手工统计特征。NeuroHorizon 则走手工特征路线（~33d）+ 简单 MLP。两种方案各有利弊：SPINT 的端到端学习特征提取能力更强但需要 trial 结构的 calibration 数据；NeuroHorizon 的手工特征更紧凑且不依赖 trial 结构。若 33d MLP 方案效果不理想，可考虑：(1) 借鉴 SPINT，直接用参考窗口 raw spike counts 作为输入；(2) 混合方案（手工特征 + raw counts 拼接）。

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

**关于 SPINT 的结构**：SPINT 的 IDEncoder 实际上是 `MLP₁(per-trial raw counts → H) → mean pool → MLP₂(H → W)` 两阶段 MLP，cross-attention 在 IDEncoder 之后用于解码（不是 IDEncoder 内部）。
NeuroHorizon 方案是先预计算 33d 统计特征再用 MLP，不依赖 trial 结构。若需要更强的特征提取能力，
可考虑借鉴 SPINT 直接从 raw spike counts 学习（需要参考窗口的 binned counts 输入）——可作为 IDEncoder v2 的改进方向。

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
