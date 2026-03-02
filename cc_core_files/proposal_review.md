# NeuroHorizon Proposal Review & 执行参考

> 本文档是 proposal.md 的技术执行补充，面向 plan.md 各阶段实施，提供代码级改造指南、设计决策记录与验收标准。

---

## 文档定位与结构说明

本文档与 proposal.md / plan.md / code_research.md 配合使用：

| 文档 | 职责 | 关键内容 |
|------|------|---------|
| **proposal.md** | What & Why | 项目目标、方法论、创新点、实验设计 |
| **plan.md** | When & Who | 各 Phase 任务分解、时间节点、完成状态 |
| **本文档** | **How** | 具体实现路径、代码改造细节、设计决策记录、验收标准 |
| **code_research.md** | 代码底层参考 | POYO 代码架构分析、模块依赖关系、接口详解 |
| **dataset.md** | 数据参考 | 各数据集详细介绍、选型策略、阶段适配 |
| **background.md** | 研究背景 | 相关工作综述、研究动机、核心挑战 |

**阅读顺序建议**：proposal.md → 本文档对应 Phase 章节 → code_research.md（需查代码细节时）。

**文档结构**（2026-03-01 重构版）：
- §一：Phase 0 概要（环境、代码、数据、基线锚点）
- §二～§五：Phase 1～4 执行参考（代码改造 + 实验设计 + 验收标准）
- §六：风险与应对汇总
- §七：模型规模配置参考
- 附录 A/B/C：项目架构速览、POYO 代码接口参考、关键文件清单

---

## 执行通则

> 以下规则适用于所有 Phase 的任务执行，优先级高于各 Phase 内的具体指引。

1. **显存不足处理**：遇到 OOM 或显存相关错误时，优先自行排查原因（batch size、梯度累积、混合精度、模型规模等）。若确认当前硬件资源确实不够，直接告知用户，用户可提供更多资源。
2. **效果不达标处理**：实验效果未达预期时，优先在现有计划范围内排查和调整（超参数、训练策略、数据预处理等）。若穷尽计划内手段仍无法解决，可以质疑原始 proposal 的方案设计，甚至提出替代方案——但 **必须将分析过程和新方案写入文档**（proposal_review.md 或对应 Phase 的 cc_todo 记录），并 **提前获得用户同意** 后再执行。

---

## 一、Phase 0 概要：环境、代码与数据

> Phase 0 已全部完成（2026-02-28）。本节提炼关键结论，供后续 Phase 快速参考。
> 完整执行记录见 `cc_todo/phase0-env-baseline/` 目录下三个文档。

### 1.1 开发环境概要

| 项目 | 状态 |
|------|------|
| conda 环境 | `poyo`（主开发）/ `allen`（AllenSDK 专用） |
| PyTorch | 2.10.0+cu128 |
| GPU | RTX 4090 D，BF16 已验证 |
| torch_brain | dev（editable install） |
| brainsets | 0.2.1.dev4 |

> 详见 `cc_todo/phase0-env-baseline/20260228-phase0-env-code-understanding.md` §0.1.1。

### 1.2 POYO 代码架构要点

**基底选择**：基于 **POYOPlus** 改造（非 CaPOYO，非从头写）。

| 选择 | 理由 |
|------|------|
| 不用 POYO | 缺少多任务支持（Phase 3 需同时 spike prediction + behavior decoding） |
| 不用 CaPOYO | 输入类型不匹配（CaPOYO 处理连续钙信号，NeuroHorizon 处理离散 spike events） |
| **用 POYOPlus** | 输入匹配 + 多任务 readout + task_emb 可区分不同预测任务 |

**实现策略**：新建 `torch_brain/models/neurohorizon.py`，不继承 POYOPlus 类（forward/tokenize 签名差异太大），但复制 encoder + processor 构建代码，写全新 decoder + tokenize。保持对 `nn/` 基础组件的依赖。

**关键尺寸约定**（Base 配置）：
- `dim = 512`，`cross_heads = 2`（非 4），`self_heads = 8`
- FFN 使用 **GEGLU**（非 SwiGLU，非 GELU）
- Encoder `rotate_value=True`，Decoder `rotate_value=False`（不可混用）

> 完整代码架构分析见 `cc_core_files/code_research.md`；改造建议见 `cc_todo/phase0-env-baseline/20260228-phase0-env-code-understanding.md` §0.1.3。

### 1.3 数据准备关键发现

**已下载数据**：Perich-Miller 2018，10 sessions（4C + 3J + 3M），center_out_reaching。

| 发现 | 数值 | 对设计的影响 |
|------|------|------------|
| Hold period 均值 | 676ms（87% > 250ms） | 可作为 encoder 输入窗口 |
| Reach period 均值 | 1090ms（100% > 500ms，75% > 1s） | 完全支持 250ms/500ms/1s 预测窗口 |
| Population 20ms bin 均值 | 5.17 spikes（4.1% zero bins） | Poisson NLL 适用 |
| Per-unit firing rate 均值 | 6.8Hz（14.5% < 1Hz） | 低发放率 unit 需注意 |
| Mean-Variance 关系 | 接近 Poisson 特性 | 确认 Poisson 分布假设合理 |

**Phase 1 推荐起点**：session `c_20131003`（71 units，最大 C session），250ms 预测窗口（12 bins @ 20ms）。

> 完整数据探索分析见 `cc_todo/phase0-env-baseline/20260228-phase0-data-explore.md`；数据集规划见 `cc_core_files/dataset.md`。

### 1.4 POYO+ 基线结果锚点

在 10 sessions 上训练 POYOPlus（dim=128, depth=12, 约8M params），500 epochs：

| 指标 | 数值 | 备注 |
|------|------|------|
| **最佳 R²（cursor_velocity_2d）** | **0.807** | epoch 429，验证集 |
| 最终 R² | 0.805 | epoch 499 |
| 收敛轨迹 | epoch 9: 0.321 → 89: 0.784 → 229: 0.803 | LR warmup 50% |

此 R²=0.807 作为后续改造的对比锚点：
- Phase 1 自回归改造后行为解码不应大幅退化
- Phase 3 迁移实验的 baseline

> 详见 `cc_todo/phase0-env-baseline/20260228-phase0-poyo-baseline.md`。

---

## 二、Phase 1 执行参考：自回归改造

> **目标**：实现核心自回归解码器，在 Brainsets 上验证 causal mask 正确性和不同预测窗口下的生成质量。
> **对应 plan.md**：Phase 1（1.1 + 1.2 + 1.3）
> **底层代码参考**：`cc_todo/phase0-env-baseline/20260228-phase0-env-code-understanding.md` 第一～七节

### 2.1 双窗口数据组织与 tokenize() 改造

#### 数据的两个角色

每个训练样本中的神经活动数据分为两个部分，承担不同角色：

```
时间轴：
├── [t_start, t_start + T_hist]  ← previous_window（历史窗口）
│       spike events → Perceiver encoder 输入（提供上下文）
│       格式不变：复用 POYO 的 spike tokenization
│
└── [t_start + T_hist, t_start + T_hist + T_pred]  ← target_window（预测窗口）
        binned spike counts → autoregressive decoder 预测目标
        新增：需要 binning pipeline 生成
```

| 属性 | previous_window（历史窗口） | target_window（预测窗口） |
|------|---------------------------|-------------------------|
| **用途** | Perceiver encoder 输入（提供上下文） | Autoregressive decoder 的预测目标 |
| **表示格式** | Spike events（离散时间戳，POYO 格式） | Binned spike counts（固定时间格上的整数） |
| **数据形状** | `[N_spikes, 3]`（timestamp, unit_id, type） | `[T_pred_bins, N_units]` |
| **复用现有架构** | 完全复用 POYO 的 spike tokenization | 新增 binning pipeline |
| **对应 loss** | 无（不做预测） | PoissonNLL |

#### Binning Pipeline 设计

在 `tokenize()` 中将 target_window 内的 spike events 转换为 `[T_pred_bins, N_units]` 的 spike counts 张量：

```python
# binning 参数（初始推荐值）
bin_size    = 0.020   # 20ms
T_pred      = 0.250   # 预测窗口 250ms（Phase 1 起点）
T_pred_bins = int(T_pred / bin_size)  # = 12 个 time bins

# binning 操作
bin_edges = linspace(t_pred_start, t_pred_end, T_pred_bins + 1)
spike_counts = zeros(T_pred_bins, N_units, dtype=float32)
for each spike (t, uid):
    bin_idx = floor((t - t_pred_start) / bin_size)
    spike_counts[bin_idx, uid] += 1
```

#### tokenize() 新增字段

```python
model_inputs = {
    # ---- 原有字段（previous_window spike events）----
    "input_unit_index":    ...,    # [N_spikes_hist]
    "input_timestamps":    ...,    # [N_spikes_hist]
    "input_token_type":    ...,    # [N_spikes_hist]
    "latent_index":        ...,    # [N_latents]
    "latent_timestamps":   ...,    # [N_latents]

    # ---- 新增字段（target_window binned counts）----
    "target_spike_counts":   spike_counts,    # [T_pred_bins, N_units] float32
    "bin_timestamps":        bin_centers,      # [T_pred_bins] 每个 bin 中心时刻
    "bin_index":             bin_index,        # [T_pred_bins] 指向 bin_emb 的位置
    "target_unit_mask":      unit_mask,        # [N_units] bool，标记有效 unit
}
```

### 2.2 PoissonNLLLoss 实现

**修改文件**：`torch_brain/nn/loss.py`

```python
class PoissonNLLLoss(nn.Module):
    """
    Poisson NLL Loss: loss = exp(log_rate) - target * log_rate
    模型输出 log_rate（非 rate），数值更稳定。
    """
    def __init__(self, log_input=True, eps=1e-8, reduction='mean'):
        super().__init__()
        self.log_input = log_input
        self.eps = eps
        self.reduction = reduction

    def forward(self, log_rate: torch.Tensor, target: torch.Tensor,
                weights=None) -> torch.Tensor:
        # log_rate: [B, T, N]，target: [B, T, N]（非负 float）
        if self.log_input:
            loss = torch.exp(log_rate) - target * log_rate
        else:
            rate = log_rate.clamp(min=self.eps)
            loss = rate - target * torch.log(rate)

        if weights is not None:
            loss = loss * weights

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
```

**数值稳定性**：`log_rate` 需在 `[-10, 10]` 范围内，在 PerNeuronMLPHead 输出后 `clamp`。低发放率神经元的 0-spike bins 不导致 `log(0)`（公式不含 `log(target)`）。

### 2.3 spike_counts 模态注册

**修改文件**：`torch_brain/registry.py`

```python
# 在文件末尾添加
register_modality(
    name="spike_counts",
    dim=1,                    # per-neuron 输出，在 PerNeuronMLPHead 内处理维度
    type=DataType.CONTINUOUS,
    timestamp_key="spikes.timestamps",
    value_key="spike_counts.counts",
    loss_fn=PoissonNLLLoss(),
)
```

注意：`dim=1` 因为 N_units 在不同 session 可变，维度通过 PerNeuronMLPHead 动态处理。

### 2.4 RotarySelfAttention causal mask 修改（重点）

**问题**：`rotary_attn_pytorch_func` 的 `attn_mask` reshape 逻辑只处理 1D kv-padding mask（`b n -> b () () n`），不支持 2D causal mask（`N_q × N_kv` 下三角）。两个后端均需修改。

**修改文件**：`torch_brain/nn/rotary_attention.py`

#### pytorch 后端修改

```python
# rotary_attn_pytorch_func 中
if attn_mask is not None:
    if attn_mask.ndim == 2:          # (B, N_kv) 原有 padding mask
        attn_mask = rearrange(attn_mask, "b n -> b () () n")
    elif attn_mask.ndim == 3:        # (B, N_q, N_kv) causal 或 full mask
        attn_mask = rearrange(attn_mask, "b n m -> b () n m")
    # PyTorch F.scaled_dot_product_attention 正确处理 bool mask
```

#### xformers 后端修改

```python
# rotary_attn_xformers_func 中
if attn_mask is not None:
    if attn_mask.ndim == 2:          # (B, N_kv) padding
        attn_mask = repeat(attn_mask, "b m -> b h n m", h=num_heads, n=query.size(1))
    elif attn_mask.ndim == 3:        # (B, N_q, N_kv) causal
        attn_mask = repeat(attn_mask, "b n m -> b h n m", h=num_heads)
    attn_bias = attn_mask.to(query.dtype).masked_fill(~attn_mask, float("-inf"))
```

#### 新增工具函数

```python
def create_causal_mask(seq_len: int, device) -> Tensor:
    """返回下三角（含对角线）为 True 的 bool mask。"""
    return torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).tril()
```

**重要约束**：causal mask **只在 decoder self-attn** 中启用，encoder cross-attn、processor self-attn、decoder cross-attn 均保持双向。

### 2.5 自回归 Decoder 设计（重点）

#### 信息瓶颈问题分析与方案对比

原始方案中，per-bin 单个 query（dim=d_b'）需要服务所有 N 个神经元，解码器必须把所有神经元的预测信息压缩进一个向量——当 N > 100 时形成 bottleneck。

| 方案 | 描述 | 计算复杂度 | 评估 |
|------|------|-----------|------|
| A | Per-neuron-per-bin queries（N×T 个 query） | O(N²×T²) | 显存爆炸，不可行 |
| B | Cross-attn over units after bin-level decode | O(T×N) | 中等，但信息流不自然 |
| C | POYO-style per-query 方式 | 参考 POYO | 中等 |
| **D（最终选择）** | **T-token decoder + PerNeuronMLPHead** | O(T²) + O(T×N) | **推荐：效率最优** |

#### 最终方案：T-token Decoder

> 经深度分析后（详见 code understanding 文档第四～七节），最终采用 T-token decoder 方案：

**核心思路**：Decoder 仅在 T 维度操作（bin_queries `[B, T_pred, dim]`），(T, N) 展开仅在 head 层进行。

```
bin_queries [B, T_pred, dim]
     │
     ├─① Cross-Attention（bin_queries attend to encoder_latents）
     │       Q = bin_queries,  K = V = encoder_latents
     │       mask: 无（双向）— latents 来自历史窗口，是完整上下文
     │
     ├─② Causal Self-Attention（bin_queries 彼此 attend）
     │       Q = K = V = bin_queries
     │       mask: causal 下三角 — bin t 只看 bin 0..t
     │
     └─③ FFN (GEGLU)
```

**效率优势**：
- Decoder cross-attn 和 self-attn 仅在 T 个 token（如 12 个 bin）上计算
- 避免了 T×N 的显存爆炸（原方案 A/D 中 T×N 可达 12×200=2400）
- Causal mask 简化为标准 T×T 下三角，无需 block-diagonal 处理

#### Causal 分析总结

| 层 | 位置 | 是否 causal | 理由 |
|----|------|------------|------|
| `enc_atn`（encoder cross-attn） | Perceiver encoder | ❌ | latent attend to 历史 spike events |
| `proc_layers`（self-attn × depth） | Processor | ❌ | latents 互相 attend，编码完整历史 |
| Decoder cross-attn | AR Decoder | ❌ | bin_queries attend to encoder_latents（完整历史） |
| **Decoder causal self-attn** | AR Decoder | **✅** | bin t 不能 attend 到 bin t+1..T |

**结论：只有 decoder 内部的 self-attention 需要 causal mask。**

#### Teacher Forcing 与自回归推理

**训练时（Teacher Forcing）**：
- 所有 T_pred 的 bin query 同时送入 decoder
- Causal self-attention mask 确保时间因果性
- 一次前向传播得到所有步预测，高效并行

**推理时（自回归生成）**：
- 逐步生成，每步增加一个 bin query
- bin_query 本身不依赖前一步预测输出（固定 learnable base + rotary time embed）
- Causal self-attn 使后续 bin 能 attend 到前序 bin 的 **decoder 隐状态**（非预测输出值）
- 这是标准 Transformer decoder 的自回归模式

### 2.6 Per-Neuron MLP Head

预测目标是 `[T_pred_bins, N_units]` 的 spike counts，需同时感知**时间信息**（bin 的 decoder 表示）和**神经元身份**（unit embedding）。

**设计**：(T, N) 展开仅在本 head 中发生，decoder 仅在 T 维度操作。

```python
class PerNeuronMLPHead(nn.Module):
    def __init__(self, dim):
        # 输入: concat(bin_repr[dim//2], unit_emb[dim//2]) = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, 1),
        )

    def forward(self, bin_repr, unit_embs):
        # bin_repr:  [B, T, dim//2]  ← decoder 输出投影到 dim//2
        # unit_embs: [N, dim//2]     ← unit embedding 投影到 dim//2
        B, T, _ = bin_repr.shape
        N = unit_embs.shape[0]
        combined = torch.cat([
            bin_repr.unsqueeze(2).expand(B, T, N, -1),
            unit_embs.unsqueeze(0).unsqueeze(0).expand(B, T, N, -1),
        ], dim=-1)                              # [B, T, N, dim]
        log_rate = self.mlp(combined).squeeze(-1)  # [B, T, N]
        return log_rate.clamp(-10, 10)
```

**设计理由**：
- N_units 在不同 session 可变（几十到几百），无法用固定维度线性层
- 所有 (t, n) 对共享同一 MLP 参数，通过 unit_emb 编码神经元间差异
- 参考 CaPOYO 的拼接模式：`concat(value[:dim//2], unit_emb[:dim//2]) → dim`

### 2.7 Bin Query 设计

Bin query 编码每个预测时间 bin 的位置信息，采用 **learnable base + rotary time embedding** 方案：

```python
self.bin_emb = nn.Parameter(torch.randn(1, max_T_pred, dim))  # learnable base
# rotary embedding 在 attention 计算时注入绝对时间信息
bin_timestamps = torch.linspace(
    t_pred_start + bin_size / 2,
    t_pred_end   - bin_size / 2,
    T_pred_bins
)  # 每个 bin 的中心时刻
```

**与 POYO latent token 的对比**：

| 属性 | POYO latent query | NeuroHorizon bin query |
|------|------------------|----------------------|
| 内容 | latent_emb（可学习） | bin_emb（可学习 base） |
| 时间信息 | rotary time embedding | rotary time embedding |
| 数量 | N_latents（固定） | T_pred_bins（按预测窗口调整） |
| 下游 | cross-attn → processor | cross-attn → causal self-attn → head |

**优势**：(1) 与 POYO latent token 机制一致，复用 rotary embedding 基础设施；(2) RoPE 天然编码时间关系；(3) 可变预测窗口只需调整 bin 数量。

### 2.8 NeuroHorizon 模型组装（分步骤）

#### 步骤 a：模型骨架 + Encoder 复用

```python
# torch_brain/models/neurohorizon.py
class NeuroHorizon(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # 复用 POYO Encoder（不修改）
        self.unit_emb = InfiniteVocabEmbedding(cfg.dim)
        self.token_type_emb = Embedding(4, cfg.dim)
        self.latent_emb = Embedding(cfg.num_latents, cfg.dim)
        self.rotary_emb = RotaryTimeEmbedding(cfg.dim // cfg.heads)

        # Perceiver encoder（1 层 cross-attn + FFN）
        self.enc_atn = RotaryCrossAttention(
            dim=cfg.dim, heads=cfg.heads,
            cross_heads=cfg.cross_heads,    # 注意：使用 2，非 4
            rotate_value=True,              # Encoder 使用 rotate_value=True
        )
        self.enc_ffn = FeedForward(cfg.dim)

        # Processing layers（depth 层 self-attn + FFN）
        self.proc_layers = nn.ModuleList([
            nn.ModuleList([
                RotarySelfAttention(dim=cfg.dim, heads=cfg.heads, rotate_value=True),
                FeedForward(cfg.dim),
            ]) for _ in range(cfg.enc_depth)
        ])

        # IDEncoder 预留接口
        self.use_id_encoder = cfg.get('use_id_encoder', False)
```

#### 步骤 b：Decoder 集成

```python
        # 自回归 Decoder
        self.bin_emb = nn.Parameter(torch.randn(1, cfg.max_T_pred, cfg.dim))
        self.decoder_layers = nn.ModuleList([
            nn.ModuleList([
                RotaryCrossAttention(dim=cfg.dim, heads=cfg.heads,
                    cross_heads=cfg.cross_heads, rotate_value=False),
                RotarySelfAttention(dim=cfg.dim, heads=cfg.heads, rotate_value=False),
                FeedForward(cfg.dim),
            ]) for _ in range(cfg.dec_depth)
        ])

        # Output heads
        self.bin_proj = nn.Linear(cfg.dim, cfg.dim // 2)
        self.unit_proj = nn.Linear(cfg.dim, cfg.dim // 2)
        self.head = PerNeuronMLPHead(cfg.dim)
```

#### 步骤 c：tokenize() 实现

> 详细 tokenize() 设计参见 code understanding 文档第一节和第六节。

```python
    def tokenize(self, data):
        # ---- 1. previous_window spike events（复用 POYO 逻辑）----
        # create_start_end_unit_tokens → unit_emb.tokenizer(unit_ids) → ...
        # 生成 input_unit_index, input_timestamps, input_token_type

        # ---- 2. latent tokens（复用 POYO 逻辑）----
        # create_linspace_latent_tokens → latent_index, latent_timestamps

        # ---- 3. target_window binning（新增）----
        t_pred_start = data["target_window_start"]
        t_pred_end   = data["target_window_end"]
        T_pred_bins  = int((t_pred_end - t_pred_start) / self.bin_size)

        bin_centers = torch.linspace(
            t_pred_start + self.bin_size / 2,
            t_pred_end   - self.bin_size / 2,
            T_pred_bins
        )
        spike_counts = self._bin_spike_events(
            data["spikes.timestamps"], data["spikes.unit_index"],
            t_pred_start, t_pred_end, self.bin_size, N_units
        )

        return {
            # 原有字段 ...
            "bin_index":           torch.arange(T_pred_bins),
            "bin_timestamps":      bin_centers,
            "target_spike_counts": spike_counts,
            "target_unit_mask":    unit_mask,
        }
```

### 2.9 行为解码双路径设计（Phase 3 预留）

NeuroHorizon 同时保留 POYO 原有的行为解码路径，encoder + processor 完全共享：

```
共享 encoder + processor (latents)
       │
       ├── [路径 A] Spike Count Prediction（Phase 1 主目标）
       │     bin_queries → AR Decoder → PerNeuronMLPHead → PoissonNLL
       │
       └── [路径 B] Behavior Decoding（Phase 3 启用）
             task_queries → 原 POYO dec_atn → MultitaskReadout → MSE
```

**参数共享情况**：

| 模块 | 路径 A | 路径 B | 共享 |
|------|--------|--------|------|
| unit_emb / id_encoder | ✅ | ❌ | 否 |
| token_type_emb | ✅ | ✅ | **✅** |
| enc_atn + proc_layers | ✅ | ✅ | **✅** |
| ar_decoder（新增） | ✅ | ❌ | 否 |
| dec_atn（原 POYO） | ❌ | ✅ | 否 |
| per_neuron_head（新增） | ✅ | ❌ | 否 |
| multitask_readout（原 POYO） | ❌ | ✅ | 否 |

**encoder + processor 完全共享**是 Phase 3 "预训练迁移验证"的基础。

**实现方式**：forward() 中通过 `task` 参数路由：

```python
def forward(self, ..., task="spike_prediction"):
    latents = self.encode_and_process(inputs)
    if task == "spike_prediction":
        bin_repr = self.ar_decoder(bin_queries, latents)
        return self.per_neuron_head(bin_repr, unit_embs)
    elif task == "behavior_decoding":
        out = self.dec_atn(task_queries, latents)
        return self.multitask_readout(out)
```

### 2.10 训练脚本与评估指标

**新建文件**：
- `examples/neurohorizon/train.py` — 基于 POYOPlus 的 `train.py` 修改
- `examples/neurohorizon/configs/` — Hydra 配置（Small / Base 两套）
- `torch_brain/utils/neurohorizon_metrics.py`

**评估指标**：

| 指标 | 用途 | 计算方式 |
|------|------|---------|
| Poisson log-likelihood | 主要指标 | `exp(log_rate) - y * log_rate` |
| PSTH correlation | trial-averaged 预测质量 | Pearson r（对齐 trial 后平均） |
| R² | 与 POYO baseline 对比 | `1 - SS_res / SS_tot` |
| 误差随时间步衰减曲线 | 评估自回归累积误差 | 每步 PSTH correlation |

### 2.11 功能验证与预测窗口梯度测试

#### 基础功能验证（plan.md 1.2）

1. **Teacher forcing 训练**（5-10 sessions，Small 配置）
   - 验证 loss 收敛、预测 spike count 分布合理
   - 与简单 baseline 对比（PSTH-based prediction、线性预测）
2. **自回归推理验证**
   - 验证 causal mask 正确（修改 t+1 输入，t 输出不变）
   - 绘制误差随预测步数的传播曲线

#### 预测窗口梯度测试（plan.md 1.3）

| 窗口 | Bins 数 | 方案 | 关键考量 |
|------|---------|------|---------|
| 250ms | 12 | trial 对齐（方案 A）| Phase 1 基线，无需 scheduled sampling |
| 500ms | 25 | 方案 A + 方案 B 对比 | 视 250ms 结果决定是否引入 scheduled sampling |
| 1000ms | 50 | 方案 B（滑动窗口） | 引入 scheduled sampling；非自回归并行预测作为对照 |

**Scheduled sampling 策略**（1000ms 窗口）：从 100% teacher forcing 线性衰减至 约10%，持续 20-50 epoch。

**非自回归基线**（plan.md 1.3.3）：并行预测所有 bins（去掉 causal mask），作为消融对比，验证自回归的必要性。

### 2.12 验收标准

| 测试项 | 验收条件 | 测试方法 |
|--------|----------|----------|
| Causal mask 正确性 | t 时刻输出只依赖 ≤t 时刻输入 | 修改 t+1 输入，验证 t 输出不变 |
| Poisson NLL 收敛 | 训练 loss 持续下降，无 NaN/Inf | 监控 loss 曲线 |
| 自回归生成 | 50 步生成无误差爆炸（spike rate < 200Hz） | 合成数据 100 trials |
| 基线对比 | R² > 0.3（非 "> 0"） | Brainsets held-out session |
| 非自回归基线 | 并行预测作为 ablation，对比差异 | 同数据同指标 |


---

## 三、Phase 2 执行参考：IDEncoder 与跨 Session

> **目标**：实现 IDEncoder，验证跨 session 零样本泛化；可选扩展至 IBL。
> **前提**：Phase 1 自回归改造已验证 causal mask 正确、loss 收敛。
> **对应 plan.md**：Phase 2（2.1 + 2.2 + 2.3 + 2.4/2.5 可选）

### 3.1 IDEncoder 输入设计（方案 A + 方案 B 对比）

IDEncoder 以**原始神经活动数据**（非手工统计特征）作为输入，参考 SPINT (Le et al., NeurIPS 2025) 设计。

**方案 A：Binned Timesteps（SPINT 风格）—— 基础实现**

```
参考窗口 spike events → binning (20ms) → spike count 序列 → 插值到 T_ref
X_i^ref ∈ ℝ^(M × T_ref)
E_i = MLP₂( mean_pool_M( MLP₁(X_i^ref) ) )
```

- T_ref = 100（约 2s 参考窗口 @ 20ms bin），参考 SPINT M2 设置
- M 个参考窗口从不同 trial 或时段采样

**方案 B：Spike Event Tokenization（POYO 风格）—— NeuroHorizon 创新**

> 这是 NeuroHorizon 提出的创新点之一。

```
参考窗口 spike events: {(t_1), (t_2), ..., (t_K)}
→ 每个 spike 注入 rotary time embedding: emb_k = rotary_emb(t_k)
→ attention pooling / mean pooling: h_i = pool({emb_1, ..., emb_K})
→ MLP(h_i) → E_i ∈ ℝ^d_model
```

| 对比维度 | 方案 A (Binned, 基础) | 方案 B (Spike Event, 创新) |
|---------|----------------------|--------------------------|
| 输入表示 | binned spike counts（固定长度 T_ref） | raw spike event timestamps（变长） |
| 时间分辨率 | 20ms bin（离散化） | spike-level（连续，约0.1ms） |
| 网络结构 | 纯 MLP（SPINT 风格） | Rotary time emb + attention pooling + MLP |
| 与主模型一致性 | 不一致 | **一致**（主模型也用 spike events + rotary） |
| 信息损失 | binning + 插值丢失精确 timing | 无信息损失 |
| 实现复杂度 | 低 | 中 |
| 论文创新性 | 低（复现 SPINT） | **高**（NeuroHorizon 原创） |

**实验计划**：先实现方案 A 验证基本功能，再实现方案 B 对比。A vs B 的对比作为论文消融实验之一。

### 3.2 IDEncoder 网络架构（参考 SPINT）

**新建文件**：`torch_brain/nn/id_encoder.py`

```python
class IDEncoder(nn.Module):
    """从参考窗口的原始神经活动推断 unit embedding。
    架构参考 SPINT，输出替换 InfiniteVocabEmbedding（非 SPINT 的加法注入）。"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        # input_dim: T_ref (方案 A)
        # hidden_dim: 512~1024 (参考 SPINT)
        # output_dim: d_model
        self.mlp1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )  # per-window 映射
        self.mlp2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )  # → unit embedding

    def forward(self, ref_data):
        # ref_data: [N_units, M_windows, T_ref]
        h = self.mlp1(ref_data)        # [N_units, M, hidden_dim]
        h = h.mean(dim=1)              # [N_units, hidden_dim]  mean pool
        return self.mlp2(h)            # [N_units, output_dim]
```

**超参建议**（初始值参考 SPINT）：
- `input_dim`：100（T_ref，约 2s 参考窗口 @ 20ms bin）
- `hidden_dim`：512（SPINT M2）或 1024（SPINT M1）
- `output_dim`：d_model（与模型隐层维度一致）

### 3.3 Identity 注入方式：替换 unit_emb（vs SPINT 加法注入）

**与 SPINT 的关键差异**：

| 方面 | SPINT | NeuroHorizon |
|------|-------|-------------|
| 注入方式 | `Z = X + E`（加到 activity window） | `inputs = E[idx] + token_type_emb` |
| E 维度 | W（activity window size） | d_model（模型隐层维度） |
| 语义 | window-level 位置编码 | **token-level 身份标签** |
| 适配架构 | SPINT cross-attn 直接解码 | POYO Perceiver encoder |

**代码改造**：

```python
# POYO 原路径（Phase 1）：
inputs = self.unit_emb(input_unit_index) + self.token_type_emb(input_token_type)

# NeuroHorizon IDEncoder 路径（Phase 2）：
unit_embs = self.id_encoder(ref_data)            # [N_units, d_model]
inputs = unit_embs[input_unit_index] + self.token_type_emb(input_token_type)
```

**为什么不用 SPINT 的加法注入？**
- POYO Perceiver 架构中每个 spike event 需独立的 unit embedding（token-level）
- 若用加法注入，需将 E 重复加到每个属于该 unit 的 spike event token 上——语义上等价于替换 unit_emb，但替换方式更直接
- 替换方式更契合 Perceiver 的 token-level 设计

### 3.4 InfiniteVocabEmbedding 集成注意事项

`InfiniteVocabEmbedding` 有两个独立功能：

| 功能 | 说明 | 是否可替换 |
|------|------|-----------|
| ① 查表 embedding（`forward()`） | 按 index 返回 embedding | ✅ 可被 IDEncoder 替换 |
| ② Pipeline 管理（`tokenizer()`/`detokenizer()`/vocab） | unit_id ↔ global_index 映射 | ❌ 不可删除 |

**结论：不废弃 IVE，通过 `use_id_encoder` flag 切换路径，保留 ② 供 data pipeline 使用。**

```python
class NeuroHorizon(nn.Module):
    def __init__(self, ..., use_id_encoder=False):
        self.unit_emb = InfiniteVocabEmbedding(dim)  # 保留（pipeline 依赖）
        self.use_id_encoder = use_id_encoder
        if use_id_encoder:
            self.id_encoder = IDEncoder(**id_encoder_cfg)

    def _get_unit_embeddings(self, input_unit_index, ref_data=None):
        if self.use_id_encoder and ref_data is not None:
            all_unit_embs = self.id_encoder(ref_data)      # [N_units, dim]
            return all_unit_embs[input_unit_index]          # [N_spikes, dim]
        else:
            return self.unit_emb(input_unit_index)          # [N_spikes, dim]
```

### 3.5 优化器参数分组

```python
optimizer = torch.optim.AdamW([
    {"params": model.id_encoder.parameters(),  "lr": 3e-4},  # IDEncoder
    {"params": model.encoder.parameters(),     "lr": 1e-4},
    {"params": model.decoder.parameters(),     "lr": 1e-4},
])
# session_emb 保留 SparseLamb（若仍使用）
# IVE embedding 权重冻结（lr=0）—— 不再用于 forward，但保留 vocab 管理
```

### 3.6 IDEncoder 基础验证实验

1. **特征提取质量**（plan.md 2.2.1）
   - 在 Perich-Miller 单动物多 session 上验证
   - 检查 IDEncoder 生成的 embedding 空间（PCA / t-SNE）
   - 预期：不同 session 的功能相似 unit 应在 embedding 空间中聚类

2. **End-to-end pipeline 验证**（plan.md 2.2.2）
   - 替换后正常训练，loss 收敛
   - 性能不低于 Phase 1 基线（同 session 内）

### 3.7 Brainsets 跨 Session 泛化实验

**实验矩阵**（plan.md 2.3）：

| 实验 | 训练集 | 测试集 | 指标 |
|------|--------|--------|------|
| 同 Session 基线 | Session A (90%) | Session A (10%) | R², Poisson NLL |
| 跨 Session（同动物） | Sessions A,B,C | Session D（同动物） | R² 下降幅度 |
| 跨 Session（不同动物） | 2 只猴训练 | 1 只猴 held-out | R² 泛化能力 |
| IDEncoder vs 查表 | 同上 | 同上 | 两者 R² 对比 |
| 方案 A vs 方案 B | 同上 | 同上 | Tokenization 方案对比 |

**训练/测试划分**：2 只猴（C、J）训练，1 只猴（M）held-out 作为 test。使用 70+ sessions 全量训练。

### 3.8 IBL / FALCON 可选扩展实验

**IBL 扩展**（plan.md 2.4，前提：Brainsets 结果令人满意）：
- 安装 ONE API + ibllib，验证数据管线（下载 10-20 sessions 调试）
- 编写 `IBLDataset` 类：滑动窗口策略，质量过滤 `clusters.label == 1`
- 逐步扩展：20 → 50 → 100 sessions
- 按实验室划分 train/test（12 labs 跨实验室泛化）

**FALCON Benchmark**（plan.md 2.5，可选补充）：
- 在 FALCON M1/M2 上量化跨 session 泛化改进

### 3.9 验收标准

| 测试项 | 验收条件 |
|--------|----------|
| IDEncoder embedding 质量 | 不同 session 的功能相似 unit 在 t-SNE 上可聚类 |
| 零样本新 Session | R² > 0.2（vs IVE 零样本约 0） |
| 跨动物泛化 | R² 下降 < 30%（vs 同动物内训练） |
| 收敛稳定性 | 3 个随机种子结果方差 < 0.05 |
| 方案 A vs B | 方案 B 不低于方案 A |

---

## 四、Phase 3 执行参考：Data Scaling + 下游任务泛化

> **目标**：揭示性能随训练数据量的 scaling 规律；验证自回归预训练对行为解码的迁移增益。
> **前提**：Phase 2 跨 session 泛化已有基本结论。
> **对应 plan.md**：Phase 3（3.1 + 3.2 + 3.3 可选）

### 4.1 数据格式统一代码

所有数据集统一到以下格式，以支持混合训练：

```python
{
    'spike_counts':      Tensor[T, N],           # binned spike counts
    'ref_spike_data':    Tensor[N, M, T_ref],    # IDEncoder 输入
    'trial_start':       float,                  # 对齐基准
    'dataset_id':        str,                    # 来源标识（分层采样用）
    'brain_region':      List[str],              # 每个 unit 的脑区标签
}
```

### 4.2 分层采样策略实现

避免某一数据集主导训练：

```python
sampler = WeightedDatasetSampler(
    datasets=[brainsets_ds, ibl_ds],
    weights=[0.5, 0.5],            # 按数据量逆比例加权
    strategy='balanced_trial'
)
```

### 4.3 行为解码微调代码改动（encoder 冻结 + head 微调）

**核心改动**：利用 §2.9 的双路径设计，冻结自回归预训练的 encoder + processor，仅微调行为解码 head。

#### Encoder 权重冻结策略

```python
# 冻结预训练的 encoder + processor
for param in model.enc_atn.parameters():
    param.requires_grad = False
for param in model.enc_ffn.parameters():
    param.requires_grad = False
for param in model.proc_layers.parameters():
    param.requires_grad = False

# 可选：冻结 unit_emb / id_encoder（视实验需要）
if freeze_unit_emb:
    for param in model.unit_emb.parameters():
        param.requires_grad = False

# 仅训练行为解码路径
trainable_params = list(model.dec_atn.parameters()) + \
                   list(model.multitask_readout.parameters())
optimizer = torch.optim.AdamW(trainable_params, lr=1e-4)
```

#### 微调训练流程

```python
# Phase 3 迁移训练
for batch in dataloader:
    latents = model.encode_and_process(batch)  # 冻结的 encoder + processor
    behavior_pred = model.behavior_decode(latents, batch)  # 可训练的 decoder + readout
    loss = mse_loss(behavior_pred, batch['behavior_target'])
    loss.backward()
    optimizer.step()
```

#### 三种对比方案

| 方案 | encoder 初始化 | head 初始化 | 说明 |
|------|-------------|-----------|------|
| **From scratch** | 随机初始化 | 随机初始化 | POYO+ 从头训练 baseline |
| **Frozen transfer** | 自回归预训练（冻结） | 随机初始化 | 纯迁移，encoder 不更新 |
| **Fine-tuned transfer** | 自回归预训练（小 lr 微调） | 随机初始化 | 端到端微调 |

### 4.4 Scaling 实验设计

#### Brainsets Scaling（必做，plan.md 3.1）

```
Session 数量梯度：5 → 10 → 20 → 40 → 70+
每个规模独立训练，记录验证集指标
```

| 步骤 | Session 数 | 说明 |
|------|-----------|------|
| 起点 | 5 | 最小规模，验证 pipeline |
| 扩展 | 10 / 20 | 初步 scaling 趋势 |
| 中量 | 40 | 是否出现拐点 |
| 全量 | 70+ | Brainsets 范围最终结论 |

**分析**：绘制 `session_count vs R² / PSTH correlation` 折线图，分析是否存在 power-law 关系。若曲线平坦，分析瓶颈来源（模型容量 vs 数据异质性）。

#### IBL 大规模 Scaling（可选，plan.md 3.3）

```
Session 数量梯度：30 → 50 → 100 → 200 → 459
前提：IBL 管线在 Phase 2.4 已建立
```

### 4.5 下游任务迁移实验设计

**核心实验**（plan.md 3.2）：

| 实验 | 方法 | 指标 | 预期 |
|------|------|------|------|
| POYO+ baseline | 从头训练行为解码 | R² | 0.807（Phase 0 基线） |
| Frozen transfer | 自回归预训练 encoder（冻结）+ 新 behavior head | R² | > baseline（迁移增益） |
| Fine-tuned transfer | 自回归预训练 encoder（小 lr）+ 新 behavior head | R² | > frozen（微调进一步提升） |
| 少样本迁移 | 仅用 10%/25%/50% 行为标注数据 | R² | frozen/fine-tuned 在少样本下优势更大 |

**关键对比**：
- `Frozen transfer R²` vs `From scratch R²` → 量化预训练迁移增益
- `少样本 transfer R²` vs `少样本 from scratch R²` → 预训练在低数据场景的价值

### 4.6 验收标准

| 测试项 | 验收条件 |
|--------|----------|
| Scaling 趋势 | session 数增加时 R² 持续增长（至少在 5→20 范围内） |
| 迁移增益 | Frozen transfer R² > From scratch R²（统计显著） |
| 少样本优势 | 10% 数据下 transfer R² > 50% 数据下 from scratch R² |
| Scaling 曲线质量 | 至少 4 个数据点，误差条清晰 |


---

## 五、Phase 4 执行参考：多模态引入

> **目标**：实现视觉图像（DINOv2）和行为数据的条件注入，量化不同模态的预测贡献。
> **数据集**：Allen Visual Coding Neuropixels（58 sessions）
> **前提**：Phase 2/3 的自回归预测和跨 session 泛化已有基本结论。
> **对应 plan.md**：Phase 4（4.1 + 4.2 + 4.3）

### 5.1 Allen 数据准备与 Dataset 类实现

#### 数据下载与预处理

```bash
# 在 allen conda 环境中下载（避免与 torch_brain 依赖冲突）
conda activate allen
python scripts/data/download_allen_neuropixels.py \
    --output_dir /root/autodl-tmp/NeuroHorizon/data/raw/allen_neuropixels/ \
    --n_sessions 58

# 转存为 HDF5（主环境加载）
python scripts/data/allen_to_hdf5.py \
    --input_dir data/raw/allen_neuropixels/ \
    --output_dir data/processed/allen_neuropixels/
```

**存储需求**：约146.5 GB（NWB 原始） + 约50 GB（预处理 HDF5）。下载前确认 `/root/autodl-tmp` 剩余 > 200GB。

#### Allen Dataset 类

**新建文件**：`examples/neurohorizon/datasets/allen_multimodal.py`

```python
class AllenMultimodalDataset(torch.utils.data.Dataset):
    """Allen Visual Coding Neuropixels 多模态数据集。
    支持 neural spike data + visual stimulus + behavior data。"""

    def __init__(self, hdf5_dir, dino_dir, stimulus_type="natural_movies",
                 window_size=1.0, pred_size=0.25, bin_size=0.02):
        self.sessions = self._load_sessions(hdf5_dir)
        self.dino_features = self._load_dino(dino_dir)  # 预计算的 DINOv2 embeddings
        self.stimulus_type = stimulus_type

    def __getitem__(self, idx):
        session, t_start = self.index[idx]
        data = session.slice(t_start, t_start + self.window_size + self.pred_size)

        return {
            "spike_events":     data.spikes,                    # previous_window
            "spike_counts":     self._bin(data, self.pred_size), # target_window
            "stimulus_emb":     self._get_dino_emb(data),       # [N_frames, 768]
            "behavior":         data.running_speed,              # [T_beh, 1]
            "stimulus_times":   data.stimulus_timestamps,
        }

    def _get_dino_emb(self, data):
        """获取对应时间窗口内的预计算 DINOv2 embeddings。"""
        frame_ids = self._time_to_frame_ids(data.stimulus_timestamps)
        return self.dino_features[frame_ids]  # [N_frames, 768]
```

**刺激类型选择**：
- **Natural Movies**（首选）：30s 连续无间隔，完全支持任意预测窗口
- **Natural Scenes**：每张图片 250ms + 约500ms 灰屏，预测窗口建议 ≤250ms

### 5.2 DINOv2 离线特征提取

**必须离线预计算**（不可在训练循环中实时推理，否则 OOM）。

**新建文件**：`scripts/extract_dino_embeddings.py`

```python
import torch
from torchvision import transforms

# DINOv2 ViT-B，冻结权重
model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
model.eval().cuda()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_features(images):
    """
    images: List[ndarray] — Allen 灰度图 (918×1174)
    returns: Tensor[N, 768] — DINOv2 CLS token embeddings
    """
    batch = []
    for img in images:
        # 灰度 → 3通道 RGB
        img_rgb = torch.from_numpy(img).float().unsqueeze(0).repeat(3, 1, 1)
        batch.append(transform(img_rgb))

    batch = torch.stack(batch).cuda()
    with torch.no_grad():
        features = model(batch)  # [N, 768] CLS token
    return features.cpu()

# 预提取所有 118 张 Natural Scenes 图像
# 输出：data/processed/allen_neuropixels/dino_embeddings.pt
```

**特征维度**：DINOv2-Base 输出 768d CLS token，或 14×14=196 个 patch tokens。初期使用 CLS token（简单），后续可尝试 patch tokens（空间感知）。

### 5.3 多模态条件注入模块实现（cross-attention 方案）

**推荐注入方式**：将条件信息作为额外 context token 拼接到 encoder 输入序列中，通过现有 Perceiver cross-attention 自然融合。

```python
class MultimodalInjection(nn.Module):
    """多模态条件注入模块：将视觉/行为信号投影后拼接到 encoder 输入。"""

    def __init__(self, dim, dino_dim=768, behavior_dim=1):
        super().__init__()
        self.visual_proj = nn.Linear(dino_dim, dim)    # DINOv2 → model dim
        self.behavior_proj = nn.Linear(behavior_dim, dim)  # behavior → model dim

    def forward(self, spike_tokens, visual_emb=None, behavior=None):
        """
        spike_tokens: [B, N_spikes, dim] — 原始 spike event tokens
        visual_emb:   [B, N_frames, 768] — DINOv2 CLS tokens
        behavior:     [B, T_beh, 1]      — running speed 等
        returns:      [B, N_total, dim]  — 拼接后的 encoder 输入
        """
        tokens = [spike_tokens]

        if visual_emb is not None:
            vis_tokens = self.visual_proj(visual_emb)    # [B, N_frames, dim]
            tokens.append(vis_tokens)

        if behavior is not None:
            beh_tokens = self.behavior_proj(behavior)    # [B, T_beh, dim]
            tokens.append(beh_tokens)

        return torch.cat(tokens, dim=1)  # [B, N_total, dim]
```

**注入位置分析**：

| 方案 | 描述 | 优点 | 缺点 |
|------|------|------|------|
| **输入端拼接**（推荐） | 与 spike tokens 拼接 → Perceiver cross-attn | 简单，模块化，可消融 | 增加 encoder 输入长度 |
| Processing Layer 注入 | 在 processor self-attn 层间插入额外 cross-attn `H^(l) += CrossAttn(H^(l), c)` | 层级化融合，信息注入灵活 | 需修改 processor 架构，增加参数量，消融不便 |
| Cross-attn context | 作为额外 KV 源 | 信息注入更灵活 | 需修改 cross-attn 接口 |
| Latent 初始化 | 用 DINOv2 初始化 latent | 信息密度高 | 实现复杂 |

推荐**输入端拼接**：与 Perceiver 设计一致——所有 context 信息（spike events、visual tokens、behavior tokens）作为 cross-attn 的 KV，latents 作为 Q 从中提取信息。Perceiver 的序列压缩机制天然处理长输入。

### 5.4 行为数据注入

行为数据（运动轨迹、速度、wheel position 等）经线性投影后作为条件 token：

$$\mathbf{c}_{\text{beh}} = \text{Linear}(\mathbf{x}_{\text{behavior}}) \in \mathbb{R}^{T_b \times d}$$

#### Rotary Time Embedding 时间对齐机制

行为数据的采样率（如 50Hz / 100Hz）通常与 spike data 的事件驱动时间戳不同。NeuroHorizon 通过 rotary time embedding（RoPE）统一处理不同采样率的时间对齐：

1. **统一绝对时间坐标系**：行为数据各采样点和 spike events 共享同一绝对时间坐标（以 trial/window 起始时刻为零点），各 token 携带其真实采样时刻 $t_i$
2. **RoPE 自动编码时间关系**：在 Perceiver cross-attention 中，latent queries 通过 RoPE 与所有 encoder input tokens（spike tokens + behavior tokens）的时间距离自动反映在 attention weights 中——时间上相近的 tokens 天然获得更高 attention 权重
3. **无需手动插值**：由于 Perceiver cross-attention 原生支持变长、异构采样率的输入序列（每个 token 独立携带时间戳），不需要将行为数据重采样到 spike data 的时间网格上
4. **与 spike tokens 拼接**：投影后的行为 tokens 与 spike tokens 在序列维度拼接为统一 encoder 输入，Perceiver 的序列压缩机制自然处理增长的输入长度

### 5.5 多模态实验设计

#### 长时程连续预测（Allen Natural Movies）

| 窗口 | Bins | 说明 |
|------|------|------|
| 250ms | 12 | 基线 |
| 500ms | 25 | 中等 |
| 1s | 50 | 长时程（30s 连续视频支持） |

条件设置：`neural only` / `+behavior` / `+image` / `+behavior+image`

#### 图像-神经对齐实验（Allen Natural Scenes）

- 118 张自然图像 + DINOv2 embedding
- 量化 DINOv2 image embedding 对刺激响应预测精度的贡献
- 条件消融：有/无 image embedding

#### 多模态消融矩阵

| 设置 | Neural | Behavior | Image | 目的 |
|------|--------|----------|-------|------|
| Baseline | ✅ | ❌ | ❌ | 纯自回归基线 |
| +Behavior | ✅ | ✅ | ❌ | 行为数据贡献 |
| +Image | ✅ | ❌ | ✅ | 视觉信息贡献 |
| **Full** | ✅ | ✅ | ✅ | 完整多模态 |

### 5.6 模态贡献分析（Δ_m 公式实现）

**模态贡献定义**（来自 proposal.md §2.1）：

$$\Delta_m = \mathcal{L}(g_\phi(f_\theta(\mathcal{S}, \mathcal{C}))) - \mathcal{L}(g_\phi(f_\theta(\mathcal{S}, \mathcal{C} \setminus \mathcal{C}^{(m)})))$$

其中 $\mathcal{L}$ 为 Poisson log-likelihood 或 R²。

**条件分解实现**：

```python
def compute_delta_m(model, test_data, modality_to_ablate):
    """计算模态 m 的预测贡献 Δ_m。"""
    # 完整条件预测
    full_loss = evaluate(model, test_data, all_modalities=True)

    # 移除模态 m 后预测
    ablated_loss = evaluate(model, test_data, ablate=[modality_to_ablate])

    return full_loss - ablated_loss  # 正值 = 该模态有正贡献

def compute_conditional_delta_m(model, test_data, modality, condition_var):
    """条件分解 Δ_m(v)：在不同条件变量 v 下的贡献。
    condition_var: 'brain_region' / 'stimulus_type' / 'behavior_state'
    """
    results = {}
    for v in test_data.unique(condition_var):
        subset = test_data.filter(condition_var == v)
        results[v] = compute_delta_m(model, subset, modality)
    return results  # {v: Δ_m(v)}
```

**实验设计**：
- 按**脑区**分解：V1 vs LM vs AM 等，预期视觉皮层 Δ_image 更大
- 按**刺激类型**分解：Natural Movies vs Drifting Gratings
- 按**行为状态**分解：运动 vs 静止

### 5.7 验收标准

| 测试项 | 验收条件 |
|--------|----------|
| Allen 数据加载 | 58 sessions 正常加载、spike + stimulus + behavior 对齐 |
| DINOv2 特征 | 118 张图像 embedding 预计算完成，维度正确 |
| 多模态训练 | Full 模型 loss 收敛，不低于 neural-only baseline |
| 贡献分析 | Δ_image > 0 在视觉皮层（V1），统计显著 |
| 条件分解 | Δ_m(v) 在不同脑区/刺激条件下有差异化模式 |

---

## 六、风险与应对汇总

| 风险 | Phase | 可能性 | 影响 | 应对措施 |
|------|-------|--------|------|----------|
| 自回归 50 步误差累积 | 1 | 高 | 高 | Scheduled sampling；并行预测作为 ablation；可选 coarse-to-fine 策略（先 100ms bin 预测粗粒度，再 20ms bin 细化） |
| causal mask 维度错误 | 1 | 中 | 高 | 单元测试：修改 t+1 输入，验证 t 输出不变 |
| Poisson NLL 数值不稳定 | 1 | 中 | 中 | log_rate clamp(-10, 10)；监控梯度范数 |
| IDEncoder 输入表示能力不足 | 2 | 中 | 中 | 方案 A/B 对比；增加参考窗口长度；混合方案 |
| 跨数据集格式不统一 | 3 | 高 | 高 | 统一数据格式规范（§4.1）；早期集成测试 |
| IBL 数据适配工作量低估 | 2/3 | 高 | 中 | 预留额外 1 周；优先 Brainsets 验证 |
| 4090 显存不足（T×N×dim） | 1-4 | 中 | 高 | gradient checkpointing；BF16；Small 配置先行 |
| DINOv2 在线推理 OOM | 4 | 高 | 高 | **强制离线预计算**（§5.2） |
| Allen 灰度图颜色统计偏移 | 4 | 低 | 低 | 三通道复制 + 标准 ImageNet normalization |
| 基线 R² 目标过低 | 1 | 低 | 中 | 目标 R² > 0.3（参考论文 80%），非 "> 0" |
| 多模态融合未带来显著提升 | 4 | 中 | 中 | Learnable adapter layers；不同 DINOv2 变体；视觉皮层数据重点测试 |
| 预训练迁移无增益 | 3 | 中 | 中 | 增加预训练数据量；调整微调策略（full vs frozen） |

---

## 七、模型规模配置参考

| 配置 | enc_depth | dec_depth | dim | heads | cross_heads | 估计参数量 |
|------|-----------|-----------|-----|-------|-------------|-----------|
| **Small** | 2 | 2 | 128 | 4 | 1 | 约2M |
| **Base** | 8 | 4 | 512 | 8 | 2 | 约30M |
| **Large** | 12 | 6 | 768 | 12 | 2 | 约100M |

**建议路径**：Small 验证正确性 → Base 正式实验 → Large 仅在资源允许时使用。

**Small 配置参数参考**（从 POYOPlus 缩放）：
- `dim=128`, `enc_depth=2`, `dec_depth=2`, `cross_heads=1`, `self_heads=4`, `latent_step=0.05`

**Base 配置参数参考**：
- `dim=512`, `enc_depth=8`, `dec_depth=4`, `cross_heads=2`（非 4）, `self_heads=8`

**4090 D 显存估算**（BF16）：
- Small（batch=64）：约4 GB → 余量充足
- Base（batch=32）：约12 GB → 可行
- Large（batch=16）：约20 GB → 接近极限，需 gradient checkpointing


---

## 附录 A：项目架构速览

> 从原第一节移入，更新以与最新 proposal.md 对齐。

| 维度 | POYO（原版） | NeuroHorizon（改造目标） |
|------|-------------|------------------------|
| 任务类型 | 连续值解码（行为） | 自回归 spike count 预测 |
| 输出模态 | 连续行为量 | 离散 spike counts（泊松） |
| 时间分辨率 | spike-level（亚毫秒） | 20ms bin（可调，与 SPINT 一致） |
| 跨 Session | InfiniteVocabEmbedding（查表） | IDEncoder（参考窗口神经活动 → unit embedding） |
| 跨数据集 | 单数据集微调 | Brainsets + IBL + Allen 联合训练 |
| 多模态 | 无 | DINOv2 视觉特征（Phase 4） |
| 损失函数 | MSE / NLL | Poisson NLL |
| 模型规模 | POYO-MP（约33M） | Small(约2M) / Base(约30M) / Large(约100M) |

**核心改造点（按 Phase 顺序）**：
1. **Phase 1**：PoissonNLLLoss + spike_counts 模态 + causal mask + 自回归 decoder + PerNeuronMLPHead
2. **Phase 2**：IDEncoder 替换 InfiniteVocabEmbedding（gradient-free 跨 session）
3. **Phase 3**：数据格式统一 + 分层采样 + 行为解码迁移
4. **Phase 4**：DINOv2 多模态注入 + 模态贡献分析

**完整架构图**（Phase 1 最终设计）：

```
数据输入
├── previous_window spike events  [N_spikes]
│   unit_index / timestamps / token_type
│        ↓ unit_emb (IVE / IDEncoder) + token_type_emb
│   inputs [N_spikes, dim]
│        ↓ enc_atn (Perceiver cross-attn, latents ← inputs)  ← 序列压缩
│   latents [L, dim]
│        ↓ proc_layers (RotarySelfAttention × depth)
│   latents [L, dim]  ← 完整历史上下文表示
│                │
│   bin_queries ─┘  (learnable base + rotary time embed) [T, dim]
│        ↓ [× N_dec autoregressive decoder blocks]
│        │  ├─ cross-attn(bin_queries, latents)    [T, dim]  ← 双向
│        │  ├─ causal self-attn(bin_queries)        [T, dim]  ← causal mask
│        │  └─ FFN (GEGLU)
│   bin_repr [T, dim]
│        ↓ 投影到 dim//2
│   PerNeuronMLPHead: concat(bin_repr[t, dim//2], unit_emb[n, dim//2])
│        ↓ 3层MLP → log_rate [T, N]
│        ↓ Poisson NLL loss
│   vs target_spike_counts [T, N]
```

---

## 附录 B：POYO 代码架构与改造接口参考

> 从原第二节移入，路径已确认使用 `torch_brain/`（非 `poyo/`），审核更新。

### B.1 核心组件与接口

```
torch_brain/                          # 实际代码目录
├── data/
│   ├── collate.py                    # pad8 / chain / track_mask8
│   └── sampler.py                    # RandomFixedWindowSampler / StitchingFixedWindowSampler
├── dataset/
│   ├── dataset.py                    # HDF5 lazy-loading（新 API，非 data/dataset.py）
│   └── nested.py                     # NestedSpikingDataset（多数据集组合）
├── transforms/
│   ├── unit_dropout.py               # UnitDropout（TriangleDistribution）
│   ├── random_crop.py                # RandomCrop
│   └── random_time_scaling.py        # TimeScaling
├── utils/
│   ├── tokenizers.py                 # create_start_end_unit_tokens / create_linspace_latent_tokens
│   └── binning.py                    # binning 工具（已存在，可复用）
├── nn/
│   ├── infinite_vocab_embedding.py   # unit_emb, session_emb（含 tokenizer/detokenizer/vocab）
│   ├── embedding.py                  # token_type_emb, latent_emb, task_emb
│   ├── rotary_attention.py           # RotaryCrossAttention / RotarySelfAttention
│   │   └── rotary_attn_pytorch_func / rotary_attn_xformers_func
│   ├── feedforward.py                # FeedForward (GEGLU)
│   ├── multitask_readout.py          # MultitaskReadout + prepare_for_multitask_readout
│   └── loss.py                       # MSELoss / CrossEntropyLoss / MallowDistanceLoss
├── registry.py                       # ModalitySpec / MODALITY_REGISTRY / register_modality
├── models/
│   ├── poyo.py                       # POYO（单任务）
│   ├── poyo_plus.py                  # POYOPlus（多任务，NeuroHorizon 的基底）
│   └── capoyo.py                     # CaPOYO（钙信号，不直接使用）
└── optim.py                          # SparseLamb（session_emb 专用优化器）
```

### B.2 需要保留的接口（不修改）

| 接口 | 位置 | 说明 |
|------|------|------|
| Encoder cross-attn + process layers | `models/poyo_plus.py` | Encoder 完整保留 |
| `RotaryCrossAttention` | `nn/rotary_attention.py` | cross-attn 不变 |
| `UnitDropout` | `transforms/unit_dropout.py` | 直接复用 |
| `TrialSampler` | `data/sampler.py` | 直接复用 |
| `BlockDiagonalMask` forward | `models/poyo_plus.py` | 高效变长 forward 保留 |
| `InfiniteVocabEmbedding.tokenizer()`/`detokenizer()` | `nn/infinite_vocab_embedding.py` | pipeline 依赖，不可删除 |

### B.3 需要替换/修改的组件

| 组件 | 修改方式 | 影响范围 |
|------|----------|----------|
| `RotarySelfAttention` | 添加 causal mask 支持（§2.4） | 仅 decoder self-attn |
| `InfiniteVocabEmbedding.forward()` | IDEncoder 旁路替换（§3.3） | unit embedding 生成 |
| `collate.py` | 添加变长 N 的 padding 函数 | 数据批处理 |
| `loss.py` | 新增 PoissonNLLLoss | 训练循环 |
| `registry.py` | 新增 spike_counts 模态 | 数据 pipeline |

### B.4 已验证的代码细节（勘误）

以下是代码精读发现的关键细节，**执行时以此为准**：

1. **cross_heads = 2，非 4**：POYO-MP cross-attention head 数量为 2，Small 配置应保持比例
2. **Encoder rotate_value=True，Decoder rotate_value=False**：不可混用，实例化时需明确指定
3. **InfiniteVocabEmbedding 不只是 Embedding**：内含 tokenizer、detokenizer 和词表管理逻辑，替换时需完整迁移数据流
4. **Token type 只有 3 个值**（DEFAULT=0, START_OF_SEQUENCE=1, END_OF_SEQUENCE=2），`nn.Embedding(4, dim)` 有 spare slot
5. **CaPOYO 拼接模式参考**：`unit_emb = concat(unit_feat[:dim//2], value_map[:dim//2])` → dim。NeuroHorizon 的 PerNeuronMLPHead 遵循此模式
6. **`torch_brain.data.Dataset` 已废弃**：新 API 为 `torch_brain.dataset.Dataset`，注意参数名变化
7. **xformers 后端**：服务器有 xformers，CUDA 上 attention 会走 `rotary_attn_xformers_func`，修改 causal mask 时**必须同时修改**
8. **attn_mask 类型**：`F.scaled_dot_product_attention` 的 `attn_mask` 可为 bool（False=masked）或 float（加到 scores）——causal mask 推荐 bool 类型（True=保留）

---

## 附录 C：关键文件清单

> 从原第八节移入，路径已更新为 `torch_brain/`。

| 文件路径 | 改动类型 | 改动内容 | Phase |
|----------|----------|----------|-------|
| `torch_brain/nn/rotary_attention.py` | 修改 | `RotarySelfAttention` 添加 causal mask 支持 | 1 |
| `torch_brain/nn/loss.py` | 新增 | PoissonNLLLoss | 1 |
| `torch_brain/registry.py` | 新增 | spike_counts 模态注册 | 1 |
| `torch_brain/data/collate.py` | 修改 | 变长 N（神经元数）padding 函数 | 1 |
| `torch_brain/nn/autoregressive_decoder.py` | **新建** | AutoregressiveDecoder（cross-attn + causal self-attn + FFN） | 1 |
| `torch_brain/nn/per_neuron_head.py` | **新建** | PerNeuronMLPHead | 1 |
| `torch_brain/models/neurohorizon.py` | **新建** | 主模型类（encoder + decoder + head + tokenize） | 1 |
| `torch_brain/nn/id_encoder.py` | **新建** | IDEncoder（方案 A + 方案 B） | 2 |
| `torch_brain/nn/multimodal_injection.py` | **新建** | MultimodalInjection（visual + behavior） | 4 |
| `torch_brain/utils/neurohorizon_metrics.py` | **新建** | PSTH correlation / Poisson LL / R² | 1 |
| `examples/neurohorizon/train.py` | **新建** | 训练脚本 | 1 |
| `examples/neurohorizon/configs/model/small.yaml` | **新建** | Small 配置（dim=128, enc=2, dec=2） | 1 |
| `examples/neurohorizon/configs/model/base.yaml` | **新建** | Base 配置（dim=512, enc=8, dec=4） | 1 |
| `examples/neurohorizon/datasets/ibl.py` | **新建** | IBL Dataset 适配器 | 2 |
| `examples/neurohorizon/datasets/allen_multimodal.py` | **新建** | Allen Multimodal Dataset 适配器 | 4 |
| `scripts/extract_reference_data.py` | **新建** | IDEncoder 参考窗口数据准备 | 2 |
| `scripts/extract_dino_embeddings.py` | **新建** | DINOv2 特征离线预计算 | 4 |

---

*文档重构：2026-03-01；原文档创建：2026-02-28*
*参考文件：proposal.md / plan.md / code_research.md / dataset.md / background.md*
*代码分析来源：cc_todo/phase0-env-baseline/20260228-phase0-env-code-understanding.md*
