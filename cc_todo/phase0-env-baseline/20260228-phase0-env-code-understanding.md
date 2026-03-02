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
- 与经典 dropout（固定比例）不同，每个 training iteration 随机采样 dropout rate（0–1 之间均匀采样）
- 以该 rate 随机移除 neural units（整个 channel 置零）
- 模拟不同 session 间 population 组成的变化，是实现跨 session 鲁棒性的关键正则化手段

**关键设计特点**：
- **置换不变性（Permutation Invariance）**：IDEncoder 中的 mean pooling + cross-attention 的 query-independent 结构保证了对 unit 顺序的不变性
- **端到端训练**：IDEncoder + cross-attention decoder 用 MSE 目标联合训练
- **轻量级设计**：仅 1 层 cross-attn + 2 个 3层FC，面向实时 iBCI 部署

**与 NeuroHorizon IDEncoder 方案的对比**：

| 方面 | SPINT | NeuroHorizon（设计） |
|------|-------|---------------------|
| **Identity 输入** | 原始 binned spike counts（T 维，从 calibration trials 插值） | 原始神经活动（两种 tokenization 方案待选，见下方讨论） |
| **特征提取方式** | 端到端学习（MLP₁ 从 binned counts 自动提取特征） | 端到端学习（参考 SPINT 架构） |
| **网络结构** | MLP₁(T→H) → mean pool → MLP₂(H→W)（2个3层FC） | 参考 SPINT 实现，先用相同的 feedforward 架构 |
| **输出维度** | W（= activity window size，如 100/700） | d_model（= 模型隐层维度，如 512） |
| **Identity 注入方式** | **加法注入**：Z = X + E（作为位置编码加到 activity window 上） | **替换 unit_emb**：E_i 直接作为神经元 embedding（对应 POYO 的 `InfiniteVocabEmbedding`） |
| **是否 gradient-free（推理时）** | 是 | 是 |
| **下游使用方式** | E 加到 X 后送入 cross-attention 解码器 | E 作为 spike event 的 unit embedding 送入 Perceiver encoder |

#### SPINT vs NeuroHorizon：Identity 注入方式的关键差异

两者在 IDEncoder 输出的**使用方式**上存在本质区别：

**SPINT 的加法注入**：
```
E_i ∈ ℝ^W           （W = activity window size）
Z_i = X_i + E_i     （直接加到当前 activity window 上）
```
SPINT 中 E_i 的维度等于 activity window 的长度 W，本质上是一种**上下文相关的位置编码**——它告诉模型"这条活动序列来自哪个神经元"。加法注入后，Z 同时包含 identity 信息和活动信息，再由 cross-attention 聚合所有 unit 进行行为解码。

**NeuroHorizon 的 unit embedding 替换**：
```
E_i ∈ ℝ^d_model      （d_model = 模型隐层维度，如 512）
# 替换原 POYO 中的：
#   inputs = self.unit_emb(input_unit_index) + self.token_type_emb(...)
# 变为：
#   inputs = id_encoder_emb[unit_i] + self.token_type_emb(...)
```
NeuroHorizon 中 E_i 的维度等于模型隐层维度 d_model，直接替换 POYO 的 `InfiniteVocabEmbedding` 输出。在 POYO 架构中，unit_emb 是每个 spike event token 的"身份标签"，与 token_type_emb 相加后送入 Perceiver encoder。IDEncoder 生成的 E_i 承担完全相同的角色，只是从"查表"变为"从神经活动推断"。

**设计动机对比**：
- SPINT 的加法注入适合其架构——SPINT 直接对 activity window 做 cross-attention 解码，E 需要与 X 在同一维度空间
- NeuroHorizon 的 unit_emb 替换适合 Perceiver 架构——POYO 中每个 spike event 需要独立的 unit embedding，IDEncoder 的输出自然地填充这个角色
- 两种方式殊途同归：都是将"神经元身份"信息注入模型，只是注入的位置和形式不同

**为什么不用 SPINT 的加法注入？**
- SPINT 的架构是直接对 activity window 做 cross-attention 解码，E 需要和 X 在同一空间（维度 = W）
- NeuroHorizon 基于 POYO Perceiver 架构，输入是 spike event 序列（每个 event 一个 token），unit_emb 是 per-token 属性
- 若在 NeuroHorizon 中用加法注入，需要将 E 重复加到每个属于该 unit 的 spike event token 上——这在语义上等价于替换 unit_emb，但替换方式更直接

#### IDEncoder 输入的两种 Tokenization 方案讨论

NeuroHorizon 的 IDEncoder 同样以原始神经活动（非手工统计特征）作为输入，但具体的 tokenization 方式有两种候选方案：

**方案 A：Binned Timesteps（SPINT 风格）**

```
参考窗口内每个 unit 的 spike events → binning（20ms bin）→ spike count 序列 ∈ ℝ^T
→ 插值到固定长度 T_ref → MLP₁(T_ref → H) → pool → MLP₂(H → d_model)
```

| 优势 | 劣势 |
|------|------|
| 与 SPINT 验证过的方案一致，可直接复用其架构 | 固定 bin size 丢失精确 spike timing |
| 输入为固定长度向量，网络设计简单（纯 MLP） | 需要插值到固定长度，引入信息损失 |
| 计算效率高（MLP forward pass） | Bin 选择（20ms vs 50ms）可能影响效果 |
| 不依赖 trial 结构（从任意参考窗口提取） | |

**方案 B：Spike Event Tokenization（POYO 风格）**

```
参考窗口内每个 unit 的 spike events
→ 每个 spike event 注入时间位置编码（rotary time embedding）
→ 通过 attention pooling / mean pooling 聚合为固定维度向量 ∈ ℝ^H
→ MLP(H → d_model)
```

| 优势 | 劣势 |
|------|------|
| 保留精确 spike timing（与 POYO 输入表示一致） | 变长输入需要聚合机制（attention pooling 或 mean pooling） |
| 不需要 binning 和插值，无信息损失 | 网络设计更复杂（需要 attention 层） |
| 与 NeuroHorizon 主模型的 spike event 输入风格统一 | 计算开销更大（尤其对高发放率 unit） |
| | 低发放率 unit spike 数过少可能导致 pooling 不稳定 |

**推荐**：先实现**方案 A**（Binned Timesteps），原因：
1. 与 SPINT 已验证的方案一致，可直接参考其架构和超参，降低实现风险
2. 纯 MLP 架构简单，调试容易，作为 IDEncoder v1 快速验证可行性
3. 方案 B 可作为后续 IDEncoder v2 的改进方向（若 v1 效果不理想或需要更精细的 timing 信息）


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

**改造方式**：在文件末尾添加 `register_modality("spike_counts", ...)` 调用。

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
- 自回归推理模式（循环生成）输出与 teacher forcing 一致（在无噪声条件下）
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

> **设计更新（2026-02-28）**：NeuroHorizon 的 IDEncoder 输入确定为**原始神经活动**（非手工统计特征），参考 SPINT 的 feedforward 架构实现。与 SPINT 的关键差异在于 identity 注入方式：NeuroHorizon 将 IDEncoder 输出作为 unit embedding（替换 `InfiniteVocabEmbedding`），而非像 SPINT 那样加到 activity window 上。

#### 1. IDEncoder 架构设计（参考 SPINT）

**架构**：采用 SPINT 的 MLP1 -> mean pool -> MLP2 feedforward 结构，先验证此架构在 NeuroHorizon 框架下的可行性，后续视效果决定是否调整。

```python
class IDEncoder(nn.Module):
    # 从参考窗口的原始神经活动推断 unit embedding
    # 架构参考 SPINT (Le et al., NeurIPS 2025)，但输出用途不同：
    #   SPINT: E_i 加到 activity window 作为位置编码
    #   NeuroHorizon: E_i 替换 InfiniteVocabEmbedding 作为 unit embedding

    def __init__(self, input_dim, hidden_dim, output_dim):
        # input_dim: 参考窗口 tokenization 后的维度（方案 A: T_ref; 方案 B: pooled dim）
        # hidden_dim: 隐层维度 H（SPINT 用 512~1024）
        # output_dim: d_model（模型隐层维度，如 512）
        self.mlp1 = ThreeLayerFC(input_dim, hidden_dim)   # per-trial/per-window 映射
        self.mlp2 = ThreeLayerFC(hidden_dim, output_dim)   # 映射到 unit embedding 空间

    def forward(self, ref_data):
        # ref_data: [N_units, M_windows, input_dim]（每个 unit 的 M 个参考窗口）
        h = self.mlp1(ref_data)           # [N_units, M_windows, hidden_dim]
        h = h.mean(dim=1)                 # [N_units, hidden_dim]  mean pooling
        unit_embs = self.mlp2(h)          # [N_units, output_dim]
        return unit_embs
```

**关于输入 tokenization（方案 A vs B）**：

如 0.1.2 中讨论，推荐先实现**方案 A（Binned Timesteps）**：
- 参考窗口内的 spike events -> 20ms bin -> spike count 序列 -> 插值到固定长度 T_ref
- `input_dim = T_ref`（如 T_ref=100，参考 SPINT M2 设置）
- 方案 B（Spike Event + attention pooling）作为 v2 备选

**超参建议**（初始值参考 SPINT）：
- `input_dim`：100（对应约 2s 参考窗口，20ms bin）
- `hidden_dim`：512（SPINT M2 设置）或 1024（SPINT M1 设置）
- `output_dim`：d_model（与模型隐层维度一致，如 512）

#### 2. Identity 注入方式：替换 unit_emb

**核心改造点**：IDEncoder 输出 `E_i` (d_model 维) 直接替换 POYO 中 `InfiniteVocabEmbedding` 的 unit_emb 输出。

在 POYO/POYOPlus 的 forward() 中：
```python
# 原代码（POYO）：
inputs = self.unit_emb(input_unit_index) + self.token_type_emb(input_token_type)

# NeuroHorizon（IDEncoder 启用时）：
unit_embs = self.id_encoder(ref_data)        # [N_units, d_model]
inputs = unit_embs[input_unit_index] + self.token_type_emb(input_token_type)
```

**与 SPINT 的对比**：
- SPINT：`Z = X + E`（E 加到 activity window 上，维度 = window size W）
- NeuroHorizon：`inputs = E[unit_idx] + token_type_emb`（E 替代 unit_emb，维度 = d_model）
- SPINT 的 E 编码的是"这条活动序列属于哪个神经元"（window-level 位置编码）
- NeuroHorizon 的 E 编码的是"这个 spike event 来自哪个神经元"（token-level 身份标签）
- NeuroHorizon 的方式更符合 Perceiver 架构的 token-level 设计——每个 spike event 独立携带 unit identity

#### 3. 集成到 NeuroHorizon 模型

**新建文件**：`torch_brain/nn/id_encoder.py`

**集成方式**：
```python
class NeuroHorizon(nn.Module):
    def __init__(self, ..., use_id_encoder=False, id_encoder_cfg=None):
        # 保留原 InfiniteVocabEmbedding（用于 tokenizer/detokenizer/vocab 管理）
        self.unit_emb = InfiniteVocabEmbedding(dim, ...)

        # IDEncoder 作为可选路径
        self.use_id_encoder = use_id_encoder
        if use_id_encoder:
            self.id_encoder = IDEncoder(**id_encoder_cfg)

    def forward(self, ..., ref_data=None):
        if self.use_id_encoder and ref_data is not None:
            # IDEncoder 路径：从参考窗口推断 unit embedding
            unit_embs = self.id_encoder(ref_data)   # [N_units, d_model]
            inputs = unit_embs[input_unit_index] + self.token_type_emb(input_token_type)
        else:
            # 原 InfiniteVocabEmbedding 路径
            inputs = self.unit_emb(input_unit_index) + self.token_type_emb(input_token_type)
```

**关键约束**：
- `InfiniteVocabEmbedding` 的 `tokenizer()` / `detokenizer()` / vocab 管理被 `tokenize()` 和 collate pipeline 大量使用，**必须保留**
- `use_id_encoder` flag 控制切换：Phase 1 用原 unit_emb，Phase 2 启用 IDEncoder
- IDEncoder 输出的 unit_embs 通过 `input_unit_index` 索引（与原 unit_emb 的使用方式一致）
- 优化器分组：IDEncoder 用 AdamW；session_emb 保留 SparseLamb

**tokenize() 改造**：
- 增加参考窗口数据的准备逻辑（从 session 的 calibration/reference 时段提取 spike events）
- 方案 A：binning + 插值到 T_ref -> 作为 `ref_data` 传入
- 方案 B（备选）：raw spike events + timestamps -> attention pooling

**训练流程**：
- IDEncoder 与模型其余部分端到端联合训练（与 SPINT 一致）
- 推理时（新 session）：只需新 session 的参考窗口前向传播一次 IDEncoder，得到 unit_embs 后缓存，无需梯度更新（gradient-free）


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

---

## 0.1.3 补充：核心设计点深度讨论

> **写作背景**：本节基于 0.1.1/0.1.2 的代码精读和论文精读结果，对两个核心设计点展开讨论：
> （1）per-neuron MLP head 的设计；
> （2）IDEncoder 通过 flag 切换 InfiniteVocabEmbedding 的设计。
> 同时，结合任务目标（自回归生成神经活动）的数据组织要求，进一步阐述 decoder 改造、输出 head 设计、query 设计和数据组织的完整技术方案。

---

### 一、任务数据组织的双窗口结构

在正式讨论 per-neuron MLP head 之前，必须先明确数据组织，因为 head 的设计直接依赖输入输出的表示方式。

#### 1.1 神经活动的两个角色

本任务的核心目标是：给定一段**历史神经活动**，自回归地预测**未来的神经发放情况**。因此一个训练样本中的神经活动数据天然分为两个部分，承担不同的角色：

```
时间轴：
├── [t_start, t_start + T_hist]  ← previous_window（历史窗口）
│       作为 encoder 输入，提供上下文信息
│       格式：spike events（timestamps + unit_ids）→ 不改变现有 POYO 架构
│
└── [t_start + T_hist, t_start + T_hist + T_pred]  ← target_window（预测窗口）
        作为 decoder 的预测目标
        格式：binned spike counts（逐 time bin × 逐 neuron 的发放数）
        → 新增，需要 binning pipeline 生成
```

| 属性 | previous_window（历史） | target_window（目标） |
|------|------------------------|----------------------|
| **用途** | Perceiver encoder 输入（提供上下文） | Autoregressive decoder 的预测目标 |
| **表示格式** | Spike events（离散时间戳） | Binned spike counts（连续时间格上的整数） |
| **数据形状** | `[N_spikes, 3]`（timestamp, unit_id, type） | `[T_pred_bins, N_units]` |
| **现有架构复用** | ✅ 完全复用 POYO 的 spike tokenization | ❌ 需新增 binning pipeline |
| **对应 loss** | 无（不做预测） | PoissonNLL（log_rate vs spike_counts） |

#### 1.2 binning pipeline 设计

target_window 内的 spike events 需要转换为 `[T_pred_bins, N_units]` 的 spike counts 张量：

```python
# binning 参数（初始推荐值）
bin_size    = 0.020   # 20ms
T_pred      = 0.250   # 预测窗口 250ms（Phase 1 起点）
T_pred_bins = int(T_pred / bin_size)  # = 12 个 time bins

# binning 操作（在 tokenize() 中完成）
# spike_times: List[float]  target_window 内所有 spike 的时间戳
# spike_units: List[int]    对应的 unit global index
bin_edges = linspace(t_pred_start, t_pred_end, T_pred_bins + 1)
spike_counts = zeros(T_pred_bins, N_units, dtype=float32)
for each spike (t, uid):
    bin_idx = floor((t - t_pred_start) / bin_size)
    spike_counts[bin_idx, uid] += 1
```

**在 tokenize() 中的新增字段**：
```python
model_inputs = {
    # ---- 原有字段（previous_window spike events）----
    "input_unit_index":  ...,    # [N_spikes_hist]
    "input_timestamps":  ...,    # [N_spikes_hist]
    "input_token_type":  ...,    # [N_spikes_hist]
    "latent_index":      ...,    # [N_latents]
    "latent_timestamps": ...,    # [N_latents]

    # ---- 新增字段（target_window binned counts）----
    "target_spike_counts":  spike_counts,    # [T_pred_bins, N_units]  float32
    "target_bin_timestamps": bin_centers,    # [T_pred_bins]  每个 bin 的中心时刻
    "target_unit_mask":      unit_mask,      # [N_units]  bool，标记哪些 unit 有 spike（用于 loss weighting）
}
```

#### 1.3 Teacher Forcing vs 自回归推理

**训练时（Teacher Forcing）**：
- 将所有 `T_pred_bins` 的 bin query 同时送入 decoder
- Causal self-attention mask 确保 bin t 的 query 只能 attend 到 bin 0..t-1 的 query
- 一次前向传播得到所有 T 步的预测，计算全局 PoissonNLL loss
- 高效，可并行化

**推理时（自回归生成）**：
```
step 1: bin_query[0] → cross-attn(encoder_latents) → 不做 self-attn（无历史） → head → log_rate[0, :]
step 2: [bin_query[0], bin_query[1]] → cross-attn + causal self-attn → head → log_rate[1, :]
...
step T: [bin_query[0..T-1], bin_query[T]] → head → log_rate[T, :]
```
注意：推理时的 bin_query 不依赖前一步的预测输出（bin_query 是固定的位置编码/时间戳 embedding），而 causal self-attn 使得后续 bin 能够 attend 到前序 bin 的**表示**（decoder 中间层的隐状态，非输出）。这是 Transformer decoder 的标准自回归模式。

---

### 二、per-neuron MLP head 展开讨论

#### 2.1 为什么需要 per-neuron head？

**现有 POYOPlus 的 readout 结构**：
```python
# MultitaskReadout 中：
output_latents[B, N_out, dim]  →  nn.Linear(dim, task_output_dim)  →  任务预测值
```
这是"per-task, all-neuron-shared"的 readout：一个线性层对所有神经元共享，直接将 latent 映射到行为维度。

**为什么 spike count 预测不能用这个结构？**

1. **输出维度问题**：需要预测的是 `[T_pred_bins, N_units]` 的 spike counts。N_units 在不同 session 中不同（从几十到几百），无法用固定维度的线性层。

2. **神经元身份问题**：不同神经元有不同的基础发放率、调谐特性。同一个 time bin 内，神经元 A 可能发 3 次，神经元 B 可能发 0 次。Readout 必须感知神经元 身份（unit embedding）。

3. **组合表示需要**：预测神经元 n 在 time bin t 的 spike count 需要同时利用：
   - **时间信息**：bin t 的解码器隐状态（来自 cross-attn + causal self-attn 后）
   - **神经元身份**：unit n 的 embedding（来自 IDEncoder 或 InfiniteVocabEmbedding）

#### 2.2 per-neuron MLP head 设计

**核心思路**：对每个 `(time_bin, unit)` 对，将 bin 的解码器表示和 unit embedding 拼接后映射到 log_rate。

```
输入：
  bin_repr  [B, T, dim//2]    ← decoder 输出的 bin 表示（投影到 dim//2）
  unit_emb  [N, dim//2]       ← unit embedding（IDEncoder 或 InfiniteVocabEmbedding 输出，投影到 dim//2）

操作：
  # 广播拼接：每个 (t, n) 对
  bin_expanded  = bin_repr.unsqueeze(2).expand(B, T, N, dim//2)   # [B, T, N, dim//2]
  unit_expanded = unit_emb.unsqueeze(0).unsqueeze(0).expand(B, T, N, dim//2)  # [B, T, N, dim//2]
  combined = cat([bin_expanded, unit_expanded], dim=-1)            # [B, T, N, dim]

  # MLP（2-3 层）
  log_rate = mlp_head(combined)   # [B, T, N, 1] → squeeze → [B, T, N]

输出：
  log_rate  [B, T, N]         ← 每个 (时间步, 神经元) 的预测 log firing rate
```

**MLP head 结构**（参考 SPINT 的轻量级设计）：
```python
class PerNeuronMLPHead(nn.Module):
    def __init__(self, dim):
        # 输入 dim = bin_dim//2 + unit_dim//2 = dim//2 + dim//2 = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, 1),   # 输出标量 log_rate
        )

    def forward(self, bin_repr, unit_embs):
        # bin_repr:  [B, T, dim//2]
        # unit_embs: [N, dim//2]
        B, T, _ = bin_repr.shape
        N = unit_embs.shape[0]
        combined = torch.cat([
            bin_repr.unsqueeze(2).expand(B, T, N, -1),
            unit_embs.unsqueeze(0).unsqueeze(0).expand(B, T, N, -1),
        ], dim=-1)                          # [B, T, N, dim]
        log_rate = self.mlp(combined).squeeze(-1)  # [B, T, N]
        return log_rate
```

**与 PoissonNLL loss 的接口**：
```python
log_rate  = per_neuron_head(bin_repr, unit_embs)   # [B, T, N]
target    = batch["target_spike_counts"]            # [B, T, N]  float32

# PoissonNLL: loss = exp(log_rate) - target * log_rate
# clamp log_rate 避免 exp 溢出
log_rate_clamped = log_rate.clamp(-10, 10)
loss = (log_rate_clamped.exp() - target * log_rate_clamped).mean()
```

#### 2.3 行为预测时的 head 复用

NeuroHorizon 在 Phase 3 还需要验证自回归预训练对行为解码的迁移。此时需要一个行为预测 head，与 spike count head 并存：

```
共享的 encoder + processor（latents）
       │
       ├── [spike_count 任务]
       │     bin_queries → 自回归 decoder → PerNeuronMLPHead → PoissonNLL
       │
       └── [behavior_decoding 任务]（复用 POYO 的设计）
             task_queries (session_emb + task_emb) → cross-attn(latents) → nn.Linear → MSE
```

**实现方式**：在 NeuroHorizon 的 forward() 中通过 `task` 参数路由：
```python
def forward(self, ..., task="spike_prediction"):
    latents = self.encode_and_process(inputs)   # 共享的 encoder + processor

    if task == "spike_prediction":
        bin_repr = self.ar_decoder(bin_queries, latents)  # 自回归 decoder
        return self.per_neuron_head(bin_repr, unit_embs)

    elif task == "behavior_decoding":
        out = self.dec_atn(task_queries, latents)         # 原 POYO decoder（保留）
        return self.multitask_readout(out)
```

---

### 三、IDEncoder 集成：是否废弃 InfiniteVocabEmbedding？

#### 3.1 InfiniteVocabEmbedding 的两个功能必须分开看

`InfiniteVocabEmbedding`（以下简称 IVE）在 POYO 中承担**两个相互独立的功能**：

| 功能 | 说明 | 是否可替换 |
|------|------|-----------|
| **① 查表 embedding**（`forward()`）| 根据 unit global index 返回 `[N, dim]` embedding，用于神经元身份表示 | ✅ 可被 IDEncoder 替换 |
| **② Pipeline 管理**（`tokenizer()`/`detokenizer()`/vocab） | 管理 unit_id ↔ global_index 的映射，collate/dataset 大量依赖此接口 | ❌ 不可删除 |

**结论：不废弃 IVE，而是将其 ① 的功能用 IDEncoder 旁路替换，保留 ②。**

#### 3.2 具体实现：flag 切换方案

```python
class NeuroHorizon(nn.Module):
    def __init__(self, ..., use_id_encoder=False, id_encoder_cfg=None):
        # ✅ 保留 IVE（供 tokenizer/detokenizer/vocab 管理使用）
        self.unit_emb = InfiniteVocabEmbedding(dim, ...)

        # IDEncoder 作为可选路径
        self.use_id_encoder = use_id_encoder
        if use_id_encoder:
            self.id_encoder = IDEncoder(**id_encoder_cfg)
            # 注意：IDEncoder 输出维度 = dim（与 IVE 一致，不需要额外投影层）

    def _get_unit_embeddings(self, input_unit_index, ref_data=None):
        """统一的 unit embedding 获取接口，支持两种路径。"""
        if self.use_id_encoder and ref_data is not None:
            # IDEncoder 路径：从参考窗口的神经活动推断 embedding
            # ref_data: [N_units, M_windows, T_ref]
            all_unit_embs = self.id_encoder(ref_data)          # [N_units, dim]
            return all_unit_embs[input_unit_index]             # [N_spikes, dim]
        else:
            # 原 IVE 路径（Phase 1 使用）
            return self.unit_emb(input_unit_index)             # [N_spikes, dim]
            # 注意：self.unit_emb 的 forward() 仍然正常工作
            # tokenizer()/detokenizer() 接口完全不受影响

    def forward(self, input_unit_index, input_token_type, ..., ref_data=None):
        unit_embs = self._get_unit_embeddings(input_unit_index, ref_data)
        inputs = unit_embs + self.token_type_emb(input_token_type)
        # 后续 encode + process + decode 不变
```

#### 3.3 Phase 1 → Phase 2 的切换路径

| 阶段 | `use_id_encoder` | unit embedding 来源 | 备注 |
|------|-----------------|--------------------|----|
| Phase 1 | `False` | IVE `forward()` | 完全复用现有机制，per-session 可学习 embedding |
| Phase 2 | `True` | IDEncoder 前向推断 | 同一接口，只改 `_get_unit_embeddings()` 路径 |

**优化器分组**（Phase 2 需要）：
```python
optimizer = torch.optim.AdamW([
    {"params": model.id_encoder.parameters(),  "lr": 1e-4},  # IDEncoder 用 AdamW
    {"params": model.unit_emb.parameters(),    "lr": 0.0},   # IVE embedding 权重冻结（不再用于 forward，但保留 vocab 管理）
    {"params": model.session_emb.parameters(), "lr": 1e-3,
     "optimizer_class": SparseLamb},                          # session_emb 保留 SparseLamb
    {"params": other_params,                   "lr": 1e-4},
])
```

---

### 四、Decoder 改造：哪些部分需要 causal？

#### 4.1 现有 POYO decoder 结构回顾

```python
# POYOPlus.forward() 中的 decoder：
# Step 6: dec_atn（1 层 RotaryCrossAttention）
output_latents = self.dec_atn(
    query=output_latents,     # [B, N_out, dim]  output queries（session_emb + task_emb）
    context=latents,          # [B, L, dim]       encoder latents（processor 输出）
    attn_mask=None,           # 无 mask，双向
)
# Step 7: readout → MultitaskReadout
```

**关键认识**：POYOPlus 的 decoder 只有**1 层 cross-attn**，没有 self-attn。output_latents 是固定数量的 learnable query（N 个），彼此之间不互相 attend，也不需要 causal。

#### 4.2 NeuroHorizon 自回归 decoder 需要什么？

自回归解码的语义要求：**bin t 的预测只能依赖 bin 0..t-1 的历史信息**，不能看到未来的 bin。

这意味着需要**在 bin query 之间添加 causal self-attn**。

**最终 decoder 结构**（每个 decoder block）：

```
bin_queries [B, T_pred, dim]
     │
     ├─① Cross-Attn（bin_queries attend to encoder_latents）
     │       Q = bin_queries,  K = V = encoder_latents
     │       mask: 无（双向，latents 来自历史窗口，完整信息）
     │       → [B, T_pred, dim]
     │
     ├─② Self-Attn（bin_queries attend to each other）⚠️ 这里需要 causal mask
     │       Q = K = V = bin_queries
     │       mask: causal（下三角），bin t 只看 0..t
     │       → [B, T_pred, dim]
     │
     └─③ FFN
             → [B, T_pred, dim]
```

**顺序问题**：先 cross-attn 还是先 causal self-attn？

| 顺序 | 含义 | 推荐 |
|------|------|------|
| cross → causal self | bin 先从 encoder latents 获取上下文，再做因果推理 | ✅ **推荐**（更自然） |
| causal self → cross | bin 先做因果推理，再从 encoder latents 获取上下文 | 也可行，但信息流顺序不够直觉 |

#### 4.3 完整 causal 分析：哪些层需要改，哪些不需要

| 层 | 位置 | 是否 causal | 理由 |
|----|------|------------|------|
| `enc_atn`（encoder cross-attn） | Perceiver encoder | ❌ 不需要 | latent attend to previous_window spike events，无需因果 |
| `proc_layers`（self-attn × depth） | Processor | ❌ 不需要 | latents 互相 attend，编码完整历史，双向 OK |
| **Decoder cross-attn** | AR Decoder | ❌ **不需要 causal** | bin_queries attend to encoder_latents（完整历史） |
| **Decoder causal self-attn** | AR Decoder | ✅ **需要 causal** | bin t 不能 attend 到 bin t+1..T 的 query 表示 |

**结论**：只有 decoder 内部的 **self-attn** 需要 causal mask，encoder 和 decoder 的 cross-attn 均保持双向。

#### 4.4 rotary_attention.py 的改造范围

Phase 1 改造只涉及 `RotarySelfAttention`（用于 decoder self-attn），`RotaryCrossAttention` 无需修改。

```python
# RotarySelfAttention.forward() 的调用方：decoder block 的 self-attn
# 传入 attn_mask = create_causal_mask(T_pred, device)  # [T_pred, T_pred] bool 下三角
# rotary_attn_pytorch_func 中需要处理 (B, N_q, N_kv) 3D mask 形状
# （现有代码只处理 (B, N_kv) 2D padding mask）
```

#### 4.5 Decoder block 数量

POYOPlus 的 decoder 只有 1 层。NeuroHorizon 推荐 **N_dec = 2–4** 层的 AR decoder block（消融在 Phase 5 中进行）：

- Small 配置（Phase 1 调试用）：N_dec = 2
- Base 配置（Phase 1 完整验证用）：N_dec = 4

---

### 五、输出 head 设计总结

#### 5.1 两条输出路径

```
NeuroHorizon.forward()
       │
       ├── [路径 A] Spike Count Prediction（Phase 1 主目标）
       │         bin_queries [B, T_pred, dim]
       │              ↓ AR Decoder（N_dec 层：cross-attn + causal self-attn + FFN）
       │         bin_repr [B, T_pred, dim]
       │              ↓ 投影到 dim//2
       │         bin_repr_half [B, T_pred, dim//2]
       │              ↓ PerNeuronMLPHead（concat unit_emb, 3层MLP）
       │         log_rate [B, T_pred, N_units]
       │              ↓ PoissonNLL vs target_spike_counts [B, T_pred, N_units]
       │
       └── [路径 B] Behavior Decoding（Phase 3 验证迁移用）
                 task_queries (session_emb + task_emb) [B, N_out, dim]
                      ↓ 原 POYO dec_atn（1 层 cross-attn，双向）
                 output_latents [B, N_out, dim]
                      ↓ MultitaskReadout（原 POYO 设计，保留不改）
                 behavior_pred [B, N_out, behavior_dim]
                      ↓ MSE vs behavior labels
```

#### 5.2 路径 A 与路径 B 的参数共享情况

| 模块 | 路径 A 用 | 路径 B 用 | 共享 |
|------|----------|----------|------|
| `unit_emb` / `id_encoder` | ✅（unit embedding） | ❌ | 否 |
| `token_type_emb` | ✅ | ✅ | ✅ |
| `enc_atn` + `enc_ffn` | ✅ | ✅ | ✅ |
| `proc_layers` | ✅ | ✅ | ✅ |
| `ar_decoder`（新增） | ✅ | ❌ | 否 |
| `dec_atn`（原 POYO） | ❌ | ✅ | 否 |
| `per_neuron_head`（新增） | ✅ | ❌ | 否 |
| `multitask_readout`（原 POYO） | ❌ | ✅ | 否 |

**encoder + processor 完全共享**：预训练获得的 latent representation 可直接用于两个下游任务，这正是 Phase 3 "预训练迁移验证"的基础。

---

### 六、输出神经活动的 cross-attention query 设计

#### 6.1 POYO 原有的 output query 回顾

在 POYOPlus 中，decoder 的 query 来自两个 embedding 的叠加：
```python
output_latents = session_emb(output_session_index)   # [B, N_out, dim]  ← 每个 session 的可学习嵌入
               + task_emb(output_decoder_index)       # [B, N_out, dim]  ← 每个任务的可学习嵌入
```

- `output_session_index`：标识哪个 session（处理跨 session 的输出差异）
- `output_decoder_index`：标识哪个任务（行为维度 x / y / speed 等）
- `N_out`：输出点数量（等于所有任务的输出时间点总数）

这是一个**时间上连续**的 query 设计——通过 `prepare_for_multitask_readout()` 将目标时间戳对应的 session/task embedding 排列好。

#### 6.2 NeuroHorizon 的 bin query 设计

bin query 需要编码**每个预测时间 bin 的位置信息**。有两种实现方式：

**方案 X：Learnable Positional Embedding（固定 T_pred）**
```python
self.bin_pos_emb = nn.Embedding(max_T_pred, dim)  # 可学习位置编码

# 使用时：
bin_queries = self.bin_pos_emb(torch.arange(T_pred, device=device))  # [T_pred, dim]
bin_queries = bin_queries.unsqueeze(0).expand(B, -1, -1)             # [B, T_pred, dim]
```
- 优点：简单，无需时间戳信息
- 缺点：固定长度，不同预测窗口长度需要不同 embedding；缺乏时间绝对位置信息

**方案 Y：Rotary Time Embedding（连续时间戳）—— 推荐**
```python
# 复用现有的 create_linspace_latent_tokens 工具
bin_timestamps = torch.linspace(t_pred_start, t_pred_end, T_pred)  # [T_pred]
# 通过 rotary embedding 将时间戳编码进 Q/K 中（类似 latent tokens 的做法）
# 在 tokenize() 中生成 target_bin_timestamps 字段，forward() 时传入 RotaryCrossAttention

# bin_queries 本身初始化为 learnable（类似 latent_emb），时间信息通过 rotary 注入
self.bin_emb = nn.Parameter(torch.randn(1, max_T_pred, dim))  # learnable base
# rotary embedding 负责将绝对时间信息注入 Q/K 的旋转
```
- 优点：与 POYO 的 latent token 设计一致，可变长度，时间信息精确
- 缺点：需要在 tokenize() 中准备 bin_timestamps

**推荐方案 Y**，理由：
1. 与现有 POYO 的 latent token 机制（`create_linspace_latent_tokens`）一脉相承，无需引入新机制
2. Rotary time embedding 天然编码时间关系，cross-attn 时 encoder_latents 和 bin_queries 的时间距离会反映在 attention 权重中
3. 可变预测窗口（250ms / 500ms / 1s）只需调整 T_pred 数量，不需要重新学习 embedding

#### 6.3 bin query 与 original behavior query 的关系

| 属性 | POYO behavior query | NeuroHorizon bin query |
|------|--------------------|-----------------------|
| **内容** | session_emb + task_emb（可学习） | learnable base + rotary time embedding |
| **数量** | N_out（等于目标时间点数） | T_pred_bins（等于预测 bin 数） |
| **时间信息** | 通过 `output_timestamps` 注入 rotary | 通过 `bin_timestamps` 注入 rotary |
| **跨 session 差异** | session_emb 区分不同 session | 所有 session 共享（时间相对位置决定 bin query） |
| **对应的下游 head** | MultitaskReadout（linear） | PerNeuronMLPHead（per-neuron MLP） |
| **decoder 结构** | 1 层 cross-attn（无 self-attn） | N_dec 层（cross-attn + causal self-attn） |

**核心改造关系**：bin query 是对 POYO output query 的**扩展**，而非替换——将原来的"任务级静态 query"扩展为"时间步级动态 query"，增加了 causal self-attn 来实现自回归。

#### 6.4 tokenize() 中 bin query 的构造

```python
def tokenize(self, data):
    # ... 原有 previous_window 处理（复用 POYO 逻辑）...

    # ---- 新增：target_window bin query 准备 ----
    t_pred_start = data["target_window_start"]
    t_pred_end   = data["target_window_end"]
    T_pred_bins  = int((t_pred_end - t_pred_start) / bin_size)

    bin_centers  = torch.linspace(
        t_pred_start + bin_size / 2,
        t_pred_end   - bin_size / 2,
        T_pred_bins
    )  # [T_pred_bins]  每个 bin 的中心时刻（用于 rotary time embedding）

    # bin_index: 类比 latent_index（指向 self.bin_emb 的哪个位置）
    bin_index = torch.arange(T_pred_bins)  # [T_pred_bins]

    # 构建 spike_counts target（binning）
    spike_counts = self._bin_spike_events(
        data["spikes.timestamps"],
        data["spikes.unit_index"],
        t_pred_start, t_pred_end, bin_size, N_units
    )  # [T_pred_bins, N_units]

    return {
        # ... 原有字段 ...
        "bin_index":           bin_index,           # [T_pred_bins]
        "bin_timestamps":      bin_centers,          # [T_pred_bins]
        "target_spike_counts": spike_counts,         # [T_pred_bins, N_units]
    }
```

---

### 七、设计全景图（Phase 1 完整架构）

```
数据输入
├── previous_window spike events  [N_spikes]
│   unit_index / timestamps / token_type
│        ↓ unit_emb (IVE) + token_type_emb
│   inputs [N_spikes, dim]
│        ↓ enc_atn (cross-attn, latents←inputs)
│   latents [L, dim]
│        ↓ proc_layers (self-attn × depth)
│   latents [L, dim]  ← Perceiver 完整上下文表示
│                │
│                │ ↙ [路径 A: spike prediction]
│   bin_queries ─┘  (learnable base + rotary time embed) [T, dim]
│        ↓ [× N_dec decoder blocks]
│        │  ├─ cross-attn(bin_queries, latents) [T, dim]
│        │  ├─ causal self-attn(bin_queries)    [T, dim]  ← ⚠️ causal mask
│        │  └─ FFN
│   bin_repr [T, dim]
│        ↓ 投影 [T, dim//2]
│   PerNeuronMLPHead: concat(bin_repr[t], unit_emb[n]) → MLP → log_rate[t,n]
│   log_rate [T, N]
│        ↓ PoissonNLL
│   loss vs target_spike_counts [T, N]
│
└── (路径 B，Phase 3 启用)
    task_queries → POYO dec_atn → MultitaskReadout → behavior_pred
```

---
