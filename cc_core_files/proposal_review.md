# NeuroHorizon Proposal Review & 执行参考

> 本文档是 proposal.md 的技术补充，面向 plan.md 各阶段执行，提供代码级改造指南、设计隐患分析与验收标准。

---

## 文档结构说明

本文档与 proposal.md / plan.md 配合使用：
- **proposal.md**：项目目标、方法论、预期贡献（What & Why）
- **plan.md**：各 Phase 任务分解与时间节点（When & Who）
- **本文档**：具体实现路径、代码改造细节、风险与验收标准（How）

阅读顺序建议：先读 proposal.md 了解全局目标，再按当前执行 Phase 查阅本文档对应章节。

---

## 一、项目架构速览

| 维度 | POYO（原版） | NeuroHorizon（改造目标） |
|------|-------------|------------------------|
| 任务类型 | 连续值解码（行为） | 自回归 spike count 预测 |
| 输出模态 | 连续行为量 | 离散 spike counts（泊松） |
| 时间分辨率 | 1ms bin | 10ms bin（可调） |
| 跨 Session | InfiniteVocabEmbedding（查表） | IDEncoder（参考窗口神经活动 → unit embedding） |
| 跨数据集 | 单数据集微调 | Brainsets + IBL + Allen 联合训练 |
| 多模态 | 无 | DINOv2 视觉特征（Phase 4） |
| 损失函数 | MSE / NLL | Poisson NLL |
| 模型规模 | POYO-MP（~33M） | Small(~5M) / Base(~30M) / Large(~100M) |

核心改造点（按 Phase 顺序）：
1. 损失函数 + spike_counts 模态注册
2. causal mask 修改 + 自回归解码器
3. IDEncoder 替换 InfiniteVocabEmbedding
4. 多数据集 scaling
5. DINOv2 多模态注入

---

## 二、POYO 代码架构与改造接口参考

### 2.1 核心组件与接口

```
poyo/
├── model/
│   ├── perceiver_io.py        # PerceiverIO: encode + decode
│   ├── rotary_attention.py    # RotarySelfAttention / RotaryCrossAttention
│   ├── sequence_model.py      # SequenceModel (主干网络)
│   └── heads.py               # 各任务 head（含 InfiniteVocabEmbedding）
├── data/
│   ├── collate.py             # batch 拼接，需扩展支持变长输出
│   ├── sampler.py             # TrialSampler（可复用）
│   └── dataset.py             # Dataset 基类
├── nn/
│   ├── embedding.py           # InfiniteVocabEmbedding（含 tokenizer/detokenizer）
│   └── dropout.py             # UnitDropout（含 TriangleDistribution，可复用）
└── utils/
    └── config.py              # 模型配置入口
```

**关键尺寸约定**（以 Base 配置为例）：
- `dim = 512`
- Encoder：`depth=8`，cross_attn `heads=8`，`cross_heads=2`（注意：非 4），self_attn `heads=8`
- Decoder：`depth=4`，cross_attn `heads=8`，`cross_heads=2`，self_attn `heads=8`
- FFN：`mult=4`，使用 **GEGLU**（不是 SwiGLU，不是 GELU）

**GEGLU FFN 结构**（重要，与 proposal 中描述不同）：

```python
# POYO 实际使用的 FFN（来自 rotary_attention.py）
class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        inner_dim = int(dim * mult)
        # GEGLU: 投影到 inner_dim * 2，拆分为 gate 和 value
        self.proj_in = nn.Linear(dim, inner_dim * 2, bias=False)
        self.proj_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x):
        x = self.proj_in(x)
        x, gate = x.chunk(2, dim=-1)
        return self.proj_out(x * F.gelu(gate))  # GEGLU
```

### 2.2 需要保留的接口（不修改）

| 接口 | 位置 | 说明 |
|------|------|------|
| `PerceiverIO.encode()` | `perceiver_io.py` | Encoder 完整保留，latent 初始化方式不变 |
| `RotaryCrossAttention` | `rotary_attention.py` | 跨注意力逻辑不变，仅 self-attn 加 causal mask |
| `UnitDropout` | `nn/dropout.py` | 直接复用，无需修改 |
| `TrialSampler` | `data/sampler.py` | 直接复用用于 trial-aligned 采样 |
| `forward_varlen()` 接口 | `sequence_model.py` | 使用 `BlockDiagonalMask` 的高效变长 forward，保留 |

### 2.3 需要替换/修改的组件

| 组件 | 修改方式 | 影响范围 |
|------|----------|----------|
| `RotarySelfAttention` | 添加 causal mask 支持（见 4.3） | 仅解码器 self-attn |
| `InfiniteVocabEmbedding` | 新增 IDEncoder 作为替代路径 | unit 嵌入生成 |
| `collate.py` | 添加变长神经元输出的 padding 函数 | 数据批处理 |
| Loss 函数 | 新增 `PoissonNLLLoss` wrapper | 训练循环 |
| 模态注册 | 在 `registry.py` 添加 `spike_counts` | 数据 pipeline |

### 2.4 已验证的代码细节（勘误）

以下是 code_research_review 发现的与 proposal 不符之处，**执行时以此为准**：

1. **POYO-MP cross_heads = 2，非 4**：cross-attention head 数量影响 KV 计算量，Small 配置应保持比例
2. **Encoder rotate_value=True，Decoder rotate_value=False**：不可混用，实例化时需明确指定
3. **InfiniteVocabEmbedding 不只是 Embedding**：内含 tokenizer、detokenizer 和词表管理逻辑，替换时需完整迁移相关数据流，不能简单替换为 `nn.Embedding`
4. **Token type 只有 3 个值**（DEFAULT=0, START_OF_SEQUENCE=1, END_OF_SEQUENCE=2），`nn.Embedding(4, dim)` 有一个 spare slot，无需扩展
5. **CaPOYO 拼接模式参考**：`unit_emb = concat(unit_feat[:dim//2], value_map[:dim//2])` → 全 dim。NeuroHorizon 解码器的 `concat(bin_repr, unit_emb)` 应遵循此模式

---

## 三、Phase 0 执行参考：环境与数据

### 3.1 conda 环境确认

```bash
# 创建环境（基于 POYO 要求）
conda create -n neurohorizon python=3.10
conda activate neurohorizon

# 安装依赖（顺序重要）
pip install torch==2.1.0 torchvision --index-url https://download.pytorch.org/whl/cu118
pip install einops hydra-core wandb h5py neo quantities

# 安装 POYO（以开发模式安装，便于修改）
cd /path/to/poyo && pip install -e .

# 验证
python -c "import poyo; from poyo.model.perceiver_io import PerceiverIO; print('OK')"
```

**验证检查项**：
- [ ] `torch.cuda.is_available()` 返回 True
- [ ] `torch.cuda.get_device_name()` 确认为 4090
- [ ] BF16 支持：`torch.cuda.is_bf16_supported()` 返回 True

### 3.2 数据探索分析指南

**依赖链（必须顺序执行）**：

```
数据下载 → 格式预处理 → 特征提取 → 完整性验证
```

| 步骤 | 数据集 | 关键操作 | 预期输出 |
|------|--------|----------|----------|
| 下载 | Brainsets | `brainsets download` CLI | `.h5` 文件 |
| 下载 | IBL | ONE API（`one.load_dataset`） | 需自写 Dataset 子类 |
| 下载 | Allen | `allensdk` CLI | NWB 格式 |
| 预处理 | 全部 | bin spikes @ 10ms，对齐 trial | `spike_counts` 张量 |
| 特征提取 | 全部 | 提取 IDEncoder 参考窗口数据（见 5.1） | 每 unit 特征向量 |
| 验证 | 全部 | 统计 unit 数、trial 数、脑区覆盖 | 数据统计报告 |

**IBL 适配注意事项**（计划中低估的工作量）：
- 需要实现 `IBLDataset(torch.utils.data.Dataset)` 子类
- 进行 unit 质量过滤：`noise_cutoff`、`firing_rate > 0.1 Hz`、`contamination < 0.1`
- 将 IBL 的模态命名映射到 POYO 约定（`spikes` → `spike_counts`）

**Allen Natural Movies 选择原因**：Allen Natural Scenes 存在 trial 间隔不均匀问题，而 Natural Movies 提供连续刺激序列，适合长时程预测评估。

---

## 四、Phase 1 执行参考：自回归改造

### 4.1 PoissonNLLLoss 实现

```python
# neurohorizon/losses.py
import torch
import torch.nn as nn

class PoissonNLLLoss(nn.Module):
    """
    Poisson NLL Loss: -log P(k | lambda) = lambda - k*log(lambda) + log(k!)
    模型输出 log_rate（log_lambda），数值更稳定
    """
    def __init__(self, log_input=True, eps=1e-8, reduction='mean'):
        super().__init__()
        self.log_input = log_input
        self.eps = eps
        self.reduction = reduction

    def forward(self, log_rate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # log_rate: [B, T, N]，target: [B, T, N]（非负整数）
        if self.log_input:
            # 数值稳定：log_rate → rate，计算 NLL
            loss = torch.exp(log_rate) - target * log_rate
        else:
            rate = log_rate.clamp(min=self.eps)
            loss = rate - target * torch.log(rate)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
```

**数值稳定性注意**：`log_rate` 需在 `[-10, 10]` 范围内，可在 per-neuron MLP head 输出后加 `clamp`：

```python
log_rate = self.mlp(x).clamp(-10, 10)
```

### 4.2 spike_counts 模态注册

```python
# 在 poyo/data/registry.py（或新建 neurohorizon/registry.py）中添加

from poyo.data.registry import DataType, ModalitySpec

# 注册 spike_counts 模态
SPIKE_COUNTS = ModalitySpec(
    name="spike_counts",
    data_type=DataType.DISCRETE,
    unit="count",
    description="Binned spike counts per neuron per time bin",
)
```

同时修改数据 pipeline，确保 spike 数据以 `spike_counts` 键写入样本字典，类型为 `torch.long` 或 `torch.float32`（Poisson NLL 接受 float）。

### 4.3 RotarySelfAttention causal mask 修改（重点）

**问题**：原始 `RotarySelfAttention` 的 reshape 逻辑不支持直接传入 causal mask，因为位置编码与 mask 应用顺序存在冲突。

**定位文件**：`poyo/model/rotary_attention.py`

**修改方案**：

```python
# 在 RotarySelfAttention.forward() 中添加 causal_mask 参数
def forward(self, x, mask=None, causal=False):
    B, T, D = x.shape
    q = self.to_q(x)
    k = self.to_k(x)
    v = self.to_v(x)

    # 应用 RoPE（旋转位置编码）
    q, k = apply_rotary_emb(q, k, ...)  # 保持原有逻辑

    # 重新 reshape 为多头格式
    q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))

    # 计算注意力分数
    dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

    # ★ 新增：causal mask（在 softmax 前应用）
    if causal:
        causal_mask = torch.ones(T, T, device=x.device).triu(1).bool()
        dots = dots.masked_fill(causal_mask[None, None], float('-inf'))

    # 原有 padding mask
    if mask is not None:
        dots = dots.masked_fill(~mask[:, None, None, :], float('-inf'))

    attn = dots.softmax(dim=-1)
    out = torch.einsum('bhij,bhjd->bhid', attn, v)
    out = rearrange(out, 'b h n d -> b n (h d)')
    return self.to_out(out)
```

**关键约束**：causal mask 只在**解码器** self-attn 中启用，encoder self-attn 和所有 cross-attn 保持双向。

### 4.4 自回归解码器设计（重点，含设计隐患分析）

**解码器结构**：

```
输入: bin_query [T_q, d_b] + unit_emb [N, d_u]
  ↓
bin_query + RoPE 位置编码
  ↓
[T_q × N 个 query] = concat(bin_repr[t], unit_emb[n]) → d_model
  ↓
RotaryCrossAttention(encoder_latents) × depth_dec_cross
  ↓
RotarySelfAttention(causal=True) × depth_dec_self
  ↓
FFN (GEGLU)
  ↓
Per-Neuron MLP Head → log_rate [T_q, N]
```

**信息瓶颈隐患分析**（重要）：

原始方案中，per-bin 单个 query（dim=d_b'）需要服务所有 N 个神经元，这意味着解码器必须把所有神经元的预测信息压缩进一个向量——当 N > 100 时，这个 bottleneck 会显著限制模型容量。

**四种解决方案对比**：

| 方案 | 描述 | 计算复杂度 | 推荐度 |
|------|------|-----------|--------|
| A | Per-neuron-per-bin queries：N×T 个 query | O(N×T) | 低（显存爆炸） |
| B | Cross-attn over units after bin-level decode | O(T×N) | 中 |
| C | POYO-style per-query 方式 | 参考 POYO 实现 | 中 |
| **D（推荐）** | concat(bin_repr, unit_emb) + 轻量神经元交互层 | O(T×N²) 可控 | **高** |

**推荐方案 D 实现**（参考 CaPOYO 拼接模式）：

```python
class AutoregressiveDecoder(nn.Module):
    def __init__(self, dim, depth_cross, depth_self, heads, cross_heads):
        super().__init__()
        # bin query projection: d_b → dim//2
        self.bin_proj = nn.Linear(d_bin, dim // 2)
        # unit_emb 已为 dim//2（由 IDEncoder 或 InfiniteVocabEmbedding 输出）
        # concat → dim

        self.cross_attn_layers = nn.ModuleList([
            RotaryCrossAttention(dim, heads=heads, dim_head=dim//heads,
                                 cross_heads=cross_heads, rotate_value=False)
            for _ in range(depth_cross)
        ])
        self.self_attn_layers = nn.ModuleList([
            RotarySelfAttention(dim, heads=heads, dim_head=dim//heads)
            for _ in range(depth_self)
        ])
        # 轻量神经元交互：在时间维度上聚合神经元间相关性
        self.neuron_interact = nn.Linear(dim, dim)

    def forward(self, bin_queries, unit_embs, encoder_latents, causal=True):
        # bin_queries: [B, T, dim//2]
        # unit_embs: [B, N, dim//2]
        T, N = bin_queries.shape[1], unit_embs.shape[1]

        # 拼接：[B, T, N, dim]
        b = bin_queries.unsqueeze(2).expand(-1, -1, N, -1)
        u = unit_embs.unsqueeze(1).expand(-1, T, -1, -1)
        x = torch.cat([b, u], dim=-1)  # [B, T, N, dim]

        # 展平时间和神经元维度进行注意力计算
        x = x.view(B, T * N, dim)

        for cross_attn in self.cross_attn_layers:
            x = cross_attn(x, encoder_latents) + x
        for self_attn in self.self_attn_layers:
            x = self_attn(x, causal=causal) + x  # causal mask 在时间维度

        x = x.view(B, T, N, dim)
        return x
```

**注意**：causal mask 需在 T 维度上应用（时间自回归），N 维度上保持全连接，需要自定义 mask 生成逻辑以区分时间和神经元维度。

### 4.5 Per-Neuron MLP Head

```python
class PerNeuronMLPHead(nn.Module):
    """
    输入 decoder 输出 [B, T, N, dim]，输出每个神经元的 log_rate [B, T, N]
    """
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or dim // 2
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        # x: [B, T, N, dim]
        log_rate = self.mlp(x).squeeze(-1)  # [B, T, N]
        return log_rate.clamp(-10, 10)      # 数值稳定
```

### 4.6 NeuroHorizon 模型组装（分步骤）

**步骤 2.5a：模型骨架 + Encoder 复用**

```python
# neurohorizon/model/neurohorizon.py
class NeuroHorizon(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # 复用 POYO Encoder（不修改）
        self.encoder = PerceiverIO(
            depth=cfg.enc_depth,
            dim=cfg.dim,
            queries_dim=cfg.dim,
            cross_heads=cfg.cross_heads,     # 注意：使用 2，非 4
            self_heads=cfg.heads,
            rotate_value=True,               # Encoder 使用 rotate_value=True
        )
        self.latent = nn.Parameter(torch.randn(cfg.num_latents, cfg.dim))
```

**步骤 2.5b：自回归解码器实例化**

```python
        self.decoder = AutoregressiveDecoder(
            dim=cfg.dim,
            depth_cross=cfg.dec_depth,
            depth_self=cfg.dec_depth // 2,
            heads=cfg.heads,
            cross_heads=cfg.cross_heads,     # cross_heads=2
        )
        self.head = PerNeuronMLPHead(cfg.dim)
```

**步骤 2.5c：tokenize() 方法实现**

```python
    def tokenize(self, batch):
        """
        将 batch 转换为 encoder/decoder 输入 token 序列
        返回: input_tokens, input_mask, query_tokens, query_mask
        """
        # 1. 从 spike_counts 生成 spike token（历史窗口）
        spike_tokens = self.spike_tokenizer(batch['spike_counts'])  # [B, T_in, dim]

        # 2. 上下文信息（trial metadata, stimulus info 等）
        context_tokens = self.context_encoder(batch)                # [B, T_ctx, dim]

        # 3. 拼接 encoder 输入
        input_tokens = torch.cat([spike_tokens, context_tokens], dim=1)

        # 4. 生成 decoder query（预测时间步）
        query_tokens = self.query_generator(batch['query_timestamps'])  # [B, T_q, dim//2]

        return input_tokens, query_tokens
```

### 4.7 验收标准

| 测试项 | 验收条件 | 测试方法 |
|--------|----------|----------|
| Causal mask 正确性 | t 时刻输出只依赖 ≤t 时刻输入 | 修改 t+1 输入，验证 t 输出不变 |
| Poisson NLL 收敛 | 训练 loss 持续下降，无 NaN/Inf | 监控 wandb loss 曲线 |
| 自回归生成 | 50 步生成无误差爆炸（spike rate < 200Hz） | 合成数据上跑 100 trial |
| 基线对比 | R² > 0.3（非 "> 0"） | 在 Brainsets held-out session 上评估 |
| 非自回归基线 | 并行预测作为 ablation，对比 R² 差异 | 同数据同指标 |

---

## 五、Phase 2 执行参考：IDEncoder 与跨 Session

### 5.1 IDEncoder 输入设计

> **设计更新**：IDEncoder 的输入为**原始神经活动数据**（binned spike counts 或 spike events），而非手工统计特征（如 firing rate、ISI histogram 等）。参考 SPINT (Le et al., NeurIPS 2025) 的方案，先实现 Binned Timesteps 方案（方案 A），再实现 Spike Event Tokenization 方案（方案 B）。

**两种 Tokenization 方案**：

**方案 A：Binned Timesteps（SPINT 风格）—— 基础实现**

参考窗口内每个 unit 的 spike events -> binning (20ms bin) -> spike count 序列 -> 插值到固定长度 T_ref，输入 IDEncoder 的 MLP1。

```
X_i^ref ∈ ℝ^(M x T_ref)    （unit i 的 M 个参考窗口，每个长度 T_ref）
E_i = MLP2( mean_pool_M( MLP1(X_i^ref) ) )
```

- `T_ref`：参考 SPINT 设置（M2: T=100, M1: T=1024），初始建议 T_ref=100
- Bin size：20ms（与 SPINT 一致，与 NeuroHorizon 预测 bin size 匹配）
- 作为基础实现，验证 IDEncoder 在 NeuroHorizon 框架下的可行性

**方案 B：Spike Event Tokenization（POYO 风格）—— NeuroHorizon 创新方案**

> **这是 NeuroHorizon 提出的创新点之一**：将 POYO 的 spike event tokenization 思想引入 IDEncoder 输入表示，与方案 A 进行对比实验，验证保留精确 spike timing 信息对 identity 推断的贡献。

参考窗口内每个 unit 的 spike events，每个 spike event 仅需注入时间位置编码（rotary time embedding），通过 attention pooling 聚合为固定维度向量。

```
参考窗口 spike events: {(t_1), (t_2), ..., (t_K)}   （unit i 的 K 个 spike）
→ 每个 spike 注入 rotary time embedding: emb_k = rotary_emb(t_k)
→ attention pooling / mean pooling: h_i = pool({emb_1, ..., emb_K})  ∈ ℝ^H
→ MLP(h_i) → E_i ∈ ℝ^d_model
```

| 对比维度 | 方案 A (Binned, base) | 方案 B (Spike Event, 创新) |
|---------|----------------------|--------------------------|
| 输入表示 | binned spike counts (固定长度 T_ref) | raw spike event timestamps (变长) |
| 时间分辨率 | 20ms bin (离散化) | spike-level (连续, ~0.1ms) |
| 网络结构 | 纯 MLP (SPINT 风格) | Rotary time emb + attention pooling + MLP |
| 与主模型一致性 | 不一致 (主模型用 spike events) | **一致** (主模型也用 spike events + rotary emb) |
| 信息损失 | binning + 插值丢失精确 timing | 无信息损失 |
| 实现复杂度 | 低 (直接参考 SPINT) | 中 (需 attention pooling 层) |
| 论文创新性 | 低 (复现 SPINT) | **高** (NeuroHorizon 原创) |

**实验计划**：先实现方案 A 验证 IDEncoder 基本功能和跨 session 泛化能力，再实现方案 B 进行对比实验。方案 A vs B 的对比结果将作为论文的消融实验之一。

### 5.2 IDEncoder 网络架构（参考 SPINT）

架构参考 SPINT 的 feedforward 设计：MLP1 -> mean pooling -> MLP2。后续视效果决定是否调整。

**方案 A 对应的网络**：

```python
class IDEncoder(nn.Module):
    # 从参考窗口的原始神经活动推断 unit embedding
    # 输出用于替换 POYO 的 InfiniteVocabEmbedding（见 5.3）

    def __init__(self, input_dim, hidden_dim, output_dim):
        # input_dim: T_ref (方案 A) 或 pooled_dim (方案 B)
        # hidden_dim: SPINT 用 512~1024
        # output_dim: d_model（模型隐层维度）
        self.mlp1 = ThreeLayerFC(input_dim, hidden_dim)   # per-window
        self.mlp2 = ThreeLayerFC(hidden_dim, output_dim)   # -> unit embedding

    def forward(self, ref_data):
        # ref_data: [N_units, M_windows, input_dim]
        h = self.mlp1(ref_data)           # [N_units, M_windows, hidden_dim]
        h = h.mean(dim=1)                 # [N_units, hidden_dim]  (mean pool)
        return self.mlp2(h)               # [N_units, output_dim]
```

**超参建议**（初始值参考 SPINT）：
- `input_dim`：100 (T_ref, 约 2s 参考窗口 @ 20ms bin)
- `hidden_dim`：512 (SPINT M2) 或 1024 (SPINT M1)
- `output_dim`：d_model (与模型隐层维度一致)

### 5.3 Identity 注入方式：替换 unit_emb（非加法注入）

> **与 SPINT 的关键差异**：SPINT 将 IDEncoder 输出 E_i 以加法方式注入 activity window（`Z = X + E`，作为位置编码）。NeuroHorizon 则将 E_i 直接作为 unit embedding，替换 POYO 中的 `InfiniteVocabEmbedding` 输出。

**在 POYO forward() 中的改造**：

```python
# POYO 原代码：
inputs = self.unit_emb(input_unit_index) + self.token_type_emb(input_token_type)

# NeuroHorizon（IDEncoder 启用时）：
unit_embs = self.id_encoder(ref_data)        # [N_units, d_model]
inputs = unit_embs[input_unit_index] + self.token_type_emb(input_token_type)
```

**设计动机**：
- POYO 的 Perceiver 架构中，unit_emb 是每个 spike event token 的"身份标签"
- IDEncoder 的输出自然地填充 unit_emb 的角色：从"按 ID 查表"变为"从神经活动推断"
- 维度匹配：IDEncoder output_dim = d_model，与原 InfiniteVocabEmbedding(dim) 一致

**SPINT 加法注入 vs NeuroHorizon unit_emb 替换**：

| 方面 | SPINT | NeuroHorizon |
|------|-------|-------------|
| 注入方式 | Z = X + E (加到 activity window) | inputs = E[idx] + token_type_emb |
| E 维度 | W (activity window size) | d_model (模型隐层维度) |
| 语义 | window-level 位置编码 | token-level 身份标签 |
| 适配架构 | SPINT 的 cross-attn 直接解码 | POYO 的 Perceiver encoder |

### 5.4 InfiniteVocabEmbedding 替换注意事项

`InfiniteVocabEmbedding` 不仅是 `nn.Embedding`，还包含：
- `tokenizer`：将 unit UUID/ID 映射到 vocab index
- `detokenizer`：反向映射
- `expand_vocab()`：动态扩展词表
- 权重初始化策略

**替换时的约束**：
1. **必须保留** `tokenizer()`/`detokenizer()` 接口（data pipeline 依赖）
2. 通过 `use_id_encoder` flag 切换两种路径，保留原 InfiniteVocabEmbedding 供 Phase 1 使用
3. collate 函数中的 padding 策略不变（通过 `input_unit_index` 索引 IDEncoder 输出）

**优化器参数分组**：

```python
# IDEncoder 参数用 AdamW
optimizer = torch.optim.AdamW([
    {'params': model.encoder.parameters(), 'lr': 1e-4},
    {'params': model.id_encoder.parameters(), 'lr': 3e-4},
    {'params': model.decoder.parameters(), 'lr': 1e-4},
])
# 若保留 session_emb，单独使用 SparseLamb
```

### 5.5 跨 Session 实验设计

**实验矩阵**：

| 实验 | 训练集 | 测试集 | 指标 |
|------|--------|--------|------|
| 同 Session 基线 | Session A (90%) | Session A (10%) | R^2, Poisson NLL |
| 跨 Session (同动物) | Sessions A,B,C | Session D (同动物) | R^2 下降幅度 |
| 跨 Session (不同动物) | 所有动物训练集 | 新动物 held-out | R^2 泛化能力 |
| IDEncoder vs 查表 | 同上 | 同上 | 两者 R^2 对比 |
| 方案 A vs 方案 B | 同上 | 同上 | Tokenization 方案对比 |

### 5.6 验收标准

| 测试项 | 验收条件 |
|--------|----------|
| IDEncoder embedding 质量 | 不同 session 的功能相似 unit 在 t-SNE 上可聚类 |
| 零样本新 Session | R^2 > 0.2 (vs. InfiniteVocabEmbedding 零样本约 0) |
| 跨动物泛化 | R^2 下降 < 30% (vs. 同动物内训练) |
| 收敛稳定性 | 3 个随机种子结果方差 < 0.05 |
| 方案 A vs B | 方案 B 不低于方案 A (验证 spike event tokenization 可行性) |


## 六、Phase 3 执行参考：Data Scaling

**核心问题**：联合训练时的数据异质性处理。

**数据格式统一**：

```python
# 所有数据集统一到以下格式
{
    'spike_counts': Tensor[T, N],      # T=500(5s@10ms), N 可变
    'ref_spike_data': Tensor[N, M, T_ref],  # IDEncoder 输入（M 个参考窗口，每个 T_ref bins）
    'trial_start': float,              # 对齐基准
    'dataset_id': str,                 # 来源标识（用于分层采样）
    'brain_region': List[str],         # 每个 unit 的脑区标签
}
```

**分层采样策略**：避免某一数据集主导训练

```python
sampler = WeightedDatasetSampler(
    datasets=[brainsets_ds, ibl_ds, allen_ds],
    weights=[0.4, 0.4, 0.2],           # 按数据量逆比例加权
    strategy='balanced_trial'
)
```

**Scaling 实验设计**：用 1%、10%、100% 数据量分别训练，绘制 scaling curve，预期 log-linear 关系。若发现饱和，分析是模型容量瓶颈还是数据异质性导致。

---

## 七、Phase 4 执行参考：多模态引入

### 7.1 DINOv2 集成注意事项

**必须离线预计算**（不可在训练循环中在线推理）：

```bash
# 离线提取 Allen Natural Movies 的 DINOv2 特征
python scripts/extract_dino_features.py \
    --video_dir /data/allen/natural_movies \
    --model facebook/dinov2-base \
    --output_dir /data/allen/dino_features \
    --batch_size 64
```

**灰度图像处理**（Allen 数据为灰度，DINOv2 期望 RGB）：

```python
# 将灰度图复制为 3 通道
frame_rgb = frame_gray.unsqueeze(0).repeat(3, 1, 1)  # [1,H,W] → [3,H,W]
```

**分辨率不匹配**（Allen: 918×1174 → DINOv2: 224×224）：

```python
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

特征维度：DINOv2-Base 输出 768d CLS token，或 14×14=196 个 patch tokens（用于空间感知任务）。

### 7.2 多模态注入位置分析

**方案对比**：

| 注入位置 | 实现方式 | 优点 | 缺点 |
|----------|----------|------|------|
| Encoder 输入端 | 与 spike tokens 拼接 | 简单 | 视觉特征与 spike token 维度需对齐 |
| Encoder latent 初始化 | 用 DINOv2 特征初始化 latent | 信息密度高 | 实现复杂 |
| **Cross-attn context**（推荐） | 作为额外 KV 源加入 cross-attn | 模块化，可消融 | 需修改 cross-attn 接口 |

**推荐方案**：将 DINOv2 CLS token 映射到 `dim` 维度，作为额外的 context token 拼接到 encoder 的输入序列中：

```python
visual_token = self.visual_proj(dino_feat)  # [B, 1, dim] 或 [B, T_vis, dim]
input_tokens = torch.cat([spike_tokens, visual_token, context_tokens], dim=1)
```

### 7.3 Allen 数据适配

- **Stimulus 对齐**：将视频帧时间戳与 spike bin 时间对齐（通常精度 < 1ms）
- **Trial 定义**：Allen Natural Movies 为连续刺激，需定义 pseudo-trial（例如每 5s 为一个 trial）
- **视觉特征时序**：DINOv2 特征按帧提取（30fps → 每 33ms 一帧），需下采样到 10ms bin 分辨率（线性插值）

---

## 八、关键文件清单

| 文件路径 | 改动类型 | 改动内容 |
|----------|----------|----------|
| `poyo/model/rotary_attention.py` | 修改 | `RotarySelfAttention.forward()` 添加 `causal` 参数 |
| `poyo/data/collate.py` | 修改 | 添加变长 N（神经元数）的 padding 函数 |
| `neurohorizon/model/neurohorizon.py` | 新建 | 主模型类，组装 encoder + decoder + head |
| `neurohorizon/model/id_encoder.py` | 新建 | IDEncoder：参考窗口神经活动 → unit embedding（参考 SPINT 架构） |
| `neurohorizon/model/ar_decoder.py` | 新建 | AutoregressiveDecoder（含 causal self-attn） |
| `neurohorizon/model/heads.py` | 新建 | PerNeuronMLPHead |
| `neurohorizon/losses.py` | 新建 | PoissonNLLLoss |
| `neurohorizon/data/registry.py` | 新建 | spike_counts 模态注册 |
| `neurohorizon/data/ibl_dataset.py` | 新建 | IBL 数据集适配器 |
| `neurohorizon/data/allen_dataset.py` | 新建 | Allen Natural Movies 适配器 |
| `neurohorizon/data/feature_extractor.py` | 新建 | IDEncoder 参考窗口数据准备（binning + 插值） |
| `scripts/extract_dino_features.py` | 新建 | DINOv2 特征离线预计算 |
| `configs/model/small.yaml` | 新建 | Small 配置（dim=256, enc=4, dec=2, heads=4） |
| `configs/model/base.yaml` | 新建 | Base 配置（dim=512, enc=8, dec=4, heads=8） |
| `.claude/settings.local.json` | 修改 | SSH 权限追加（按 CLAUDE.md 规范） |

---

## 九、风险与应对汇总

| 风险 | 来源 Phase | 可能性 | 影响 | 应对措施 |
|------|-----------|--------|------|----------|
| 自回归 50 步误差累积 | Phase 1 | 高 | 高 | Scheduled sampling（从并行逐步切换到自回归）；并行预测作为 ablation |
| causal mask 维度错误 | Phase 1 | 中 | 高 | 单元测试：修改 t+1 输入，验证 t 输出不变 |
| Poisson NLL 数值不稳定 | Phase 1 | 中 | 中 | log_rate clamp(-10, 10)；监控梯度范数 |
| IBL 数据适配工作量低估 | Phase 0 | 高 | 中 | 预留额外 1 周；优先 Brainsets 验证 pipeline |
| IDEncoder 输入表示能力不足 | Phase 2 | 中 | 中 | 方案 A/B 对比实验；备选：增加参考窗口长度或混合方案 |
| 跨数据集格式不统一 | Phase 3 | 高 | 高 | 统一数据格式规范（见 6 节）；早期集成测试 |
| 4090 显存不足（T×N×dim） | Phase 1-4 | 中 | 高 | gradient checkpointing；BF16；Small 配置先行 |
| DINOv2 在线推理 OOM | Phase 4 | 高 | 高 | 强制离线预计算，不允许在训练循环中调用 DINOv2 |
| Allen 灰度图颜色统计偏移 | Phase 4 | 低 | 低 | 使用灰度专用 normalization，或三通道复制后微调 |
| 基线 R² 目标过低 | Phase 1 | 低 | 中 | 目标 R² > 0.3（参考论文数字的 80%），非 "> 0" |

---

## 十、合理性评估与已识别问题

### 10.1 架构合理性

**已确认合理的设计**：
- Perceiver IO 作为 encoder 是核心，非可选（POYO 架构强依赖，不可去除）
- RoPE 位置编码适合神经时序数据（相对位置泛化性好）
- Per-Neuron MLP Head 是解决变长输出维度的正确方案
- UnitDropout 复用（已有 TriangleDistribution，无需重新实现）
- DINOv2 离线预计算（显存约束下唯一可行方案）

**需要重新审视的设计**：

1. **SwiGLU vs GEGLU**：proposal 中提到 SwiGLU，但 POYO 实际使用 GEGLU。建议保持 GEGLU 一致性，避免引入不必要变量。

2. **信息瓶颈**：per-bin 单 query 服务多神经元的方案需要验证。推荐加入轻量神经元交互层（见 4.4 方案 D），并在 Phase 1 就进行消融对比。

3. **非自回归基线缺失**：自回归解码器的必要性需要 ablation 验证，否则无法在论文中为复杂性辩护。

### 10.2 关键已识别问题

**问题 1：可获取性** — Jia Lab 数据不可用
- 解决方案：Brainsets（主要）+ IBL（跨区域）+ Allen Natural Movies（视觉皮层）

**问题 2：时间不均匀性** — Allen Natural Scenes 试次间隔不规则
- 解决方案：改用 Allen Natural Movies（连续刺激）

**问题 3：变长输出** — 不同 Session 神经元数量不同
- 解决方案：Per-Neuron MLP Head，输入 concat(bin_repr, unit_emb)

**问题 4：自回归误差累积**
- 解决方案：Scheduled sampling（从 teacher forcing 渐进切换）+ 并行 baseline

**问题 5：4090 计算限制**
- 解决方案：Small 配置（dim=256）起步，BF16，gradient checkpointing，按需扩展

**问题 6：causal mask 兼容性**
- 解决方案：修改 `RotarySelfAttention.forward()` 添加 `causal` 参数（见 4.3）

### 10.3 模型规模配置参考

| 配置 | enc_depth | dec_depth | dim | heads | cross_heads | 参数量 |
|------|-----------|-----------|-----|-------|-------------|--------|
| Small | 4 | 2 | 256 | 4 | 1 | ~5M |
| Base | 8 | 4 | 512 | 8 | 2 | ~30M |
| Large | 12 | 6 | 768 | 12 | 2 | ~100M |

建议路径：Small 验证正确性 → Base 正式实验 → Large 仅在资源允许时使用。
