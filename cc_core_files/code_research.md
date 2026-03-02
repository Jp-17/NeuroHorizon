# POYO 代码架构分析

**日期**：2026-02-21（2026-03-01 修订）
**项目**：NeuroHorizon（基于 POYO/POYO+ fork）
**代码库**：/root/autodl-tmp/NeuroHorizon

---

## 1. 项目概述

**POYO**（Population-level Your Own, NeurIPS 2023）是一个基于 Transformer 的神经群体解码框架，用于从电生理记录（neural spike data）解码行为输出（如光标速度、手部运动）。

**核心特点**：
- 支持任意神经和行为模态
- 多 session、多 recording 训练能力
- 按需数据加载（lazy loading HDF5）
- 三个模型变体：POYO、POYOPlus（多任务）、CaPOYO（钙成像）

> **注**：`poyo_mp` 是一个工厂函数（factory function），返回使用预设 1.3M 参数配置的 POYO 实例，而非独立的模型类。

---

## 2. 项目结构

```
NeuroHorizon/
├── torch_brain/                    # 主包
│   ├── models/                     # 模型定义
│   │   ├── poyo.py                 # POYO 基础模型
│   │   ├── poyo_plus.py            # POYO+ 多任务扩展
│   │   └── capoyo.py              # CaPOYO 钙成像变体
│   ├── nn/                         # 神经网络模块
│   │   ├── rotary_attention.py     # 旋转位置编码注意力（RotaryCrossAttention, RotarySelfAttention）
│   │   ├── position_embeddings.py  # 时间位置编码（SinusoidalTimeEmbedding, RotaryTimeEmbedding）
│   │   ├── infinite_vocab_embedding.py  # 动态词汇嵌入
│   │   ├── multitask_readout.py    # 多任务读出层 + prepare_for_multitask_readout()
│   │   ├── embedding.py            # 标准嵌入
│   │   ├── feedforward.py          # GEGLU FFN 层
│   │   └── loss.py                 # 损失函数（MSE, CE, Mallow）
│   ├── data/                       # 数据加载与采样
│   │   ├── collate.py              # 批处理整理（pad8, chain, track_mask8）
│   │   └── sampler.py              # 采样器（5种，见§4.2）
│   ├── dataset/                    # 数据集 API
│   │   ├── dataset.py              # Dataset 类（HDF5加载）
│   │   ├── mixins.py               # Dataset mixins
│   │   └── nested.py               # NestedSpikingDataset（多数据集组合，命名空间机制）
│   ├── transforms/                 # 数据增强
│   │   ├── container.py            # Compose, ConditionalChoice, RandomChoice 组合器
│   │   ├── unit_dropout.py         # 随机丢弃神经元
│   │   ├── unit_filter.py          # 按条件/正则过滤神经元（UnitFilter, UnitFilterById）
│   │   ├── output_sampler.py       # 随机采样输出 token（RandomOutputSampler）
│   │   ├── random_crop.py          # 随机时间裁剪
│   │   └── random_time_scaling.py  # 时间拉伸增强
│   ├── utils/                      # 工具函数
│   │   ├── tokenizers.py           # token 创建（start/end/latent tokens）
│   │   ├── readout.py              # 输出准备（归一化、加权）
│   │   ├── stitcher.py             # 预测拼接（重叠窗口合并）
│   │   ├── binning.py              # 时间 binning
│   │   └── weights.py              # 区间加权
│   ├── registry.py                 # 模态注册系统
│   └── optim.py                    # SparseLamb 优化器
├── examples/
│   ├── poyo/                       # POYO 训练示例
│   │   ├── train.py                # 训练脚本
│   │   ├── configs/                # Hydra 配置文件
│   │   └── datasets/               # 数据集实现
│   └── poyo_plus/                  # POYO+ 训练示例
│       ├── train.py                # 训练脚本
│       └── configs/                # 配置文件
├── tests/                          # 单元测试
├── docs/                           # 文档
└── pyproject.toml                  # 包配置与依赖
```

**关键补充说明**：
- `RotaryTimeEmbedding` 定义在 `nn/position_embeddings.py` 中，被 `rotary_attention.py` 引用
- `NestedSpikingDataset` 通过命名空间将多个 Dataset 组合在一起，recording_id 变为 `"<dataset_name>/<recording_id>"` 形式，是 POYO+ 多数据集训练的关键

---

## 3. 模型架构

### 3.1 Encoder-Processor-Decoder 架构

POYO+ 采用经典的 **编码器-处理器-解码器** Transformer 架构：

```
输入 Spike 序列
    ↓
[嵌入层] Unit Embedding + Token Type Embedding + Rotary Time Embedding
    ↓
[编码器] Cross-Attention (spikes → latents) + FFN
    ↓
[处理层] (6-24 层) Self-Attention + FFN
    ↓
[解码器] Cross-Attention (latents → outputs) + FFN
    ↓
[读出层] 任务特定的线性投影
    ↓
输出预测 (如 2D 光标速度)
```

### 3.2 核心组件

| 组件 | 类名 | 文件 | 功能 |
|------|------|------|------|
| Unit Embedding | `InfiniteVocabEmbedding` | `nn/infinite_vocab_embedding.py` | 动态词汇量的 unit 嵌入，支持 lazy 初始化 |
| Session Embedding | `InfiniteVocabEmbedding` | 同上 | session 级别嵌入 |
| Token Type Embedding | `Embedding` | `nn/embedding.py` | 区分 3 种 token 类型（DEFAULT=0, START_OF_SEQUENCE=1, END_OF_SEQUENCE=2；嵌入表容量 4，index=3 预留） |
| Latent Embedding | `Embedding` | 同上 | 可学习的 latent tokens |
| 时间编码 | `RotaryTimeEmbedding` | `nn/position_embeddings.py` | RoFormer 风格旋转位置编码 |
| Cross-Attention | `RotaryCrossAttention` | `nn/rotary_attention.py` | 带 RoPE 的交叉注意力 |
| Self-Attention | `RotarySelfAttention` | `nn/rotary_attention.py` | 带 RoPE 的自注意力 |
| FFN | `FeedForward` | `nn/feedforward.py` | **GEGLU 激活**的前馈网络（见下方说明） |
| 多任务读出 | `MultitaskReadout` | `nn/multitask_readout.py` | 按任务分发的线性读出层 |

**GEGLU 激活函数**：FeedForward 使用 GEGLU（Gated Gaussian Error Linear Unit），而非标准 GELU/ReLU。输入先通过线性层扩展到 `dim * mult * 2`（默认 mult=4），然后 chunk 为两半——一半经 GELU 激活作为门控，另一半作为值，两者相乘得到 `dim * mult` 维输出，再经线性层回到 `dim`。

**rotate_value 参数差异**：
- 编码器 cross-attention (`enc_atn`)：`rotate_value=True` — value 上也应用旋转编码
- 处理层 self-attention (`proc_layers`)：`rotate_value=True`
- 解码器 cross-attention (`dec_atn`)：`rotate_value=False` — 不对 value 应用旋转

### 3.3 模型规模

| 配置 | 参数量 | dim | depth | latent_step | num_latents_per_step | cross_heads | self_heads | atn_dropout |
|------|--------|-----|-------|-------------|---------------------|-------------|------------|-------------|
| POYO-MP | 约1.3M | 64 | 6 | 0.125 | 16 | 2 | 8 | 0.2 |
| POYO-1 | 约11.8M | 128 | 24 | 0.125 | 32 | 4 | 8 | 0.0 |

### 3.4 Forward Pass 详解 (POYOPlus)

基于 `torch_brain/models/poyo_plus.py` 的 `forward()` 方法：

```python
def forward(self, *, input_unit_index, input_timestamps, input_token_type, input_mask,
            latent_index, latent_timestamps,
            output_session_index, output_timestamps, output_decoder_index, ...):

    # 1. 输入嵌入
    inputs = self.unit_emb(input_unit_index) + self.token_type_emb(input_token_type)
    input_timestamp_emb = self.rotary_emb(input_timestamps)  # RoPE

    # 2. Latent tokens
    latents = self.latent_emb(latent_index)
    latent_timestamp_emb = self.rotary_emb(latent_timestamps)

    # 3. 输出查询
    output_queries = self.session_emb(output_session_index) + self.task_emb(output_decoder_index)
    output_timestamp_emb = self.rotary_emb(output_timestamps)

    # 4. 编码：spikes → latents (Perceiver cross-attention)
    latents = latents + self.enc_atn(latents, inputs, latent_timestamp_emb, input_timestamp_emb, input_mask)
    latents = latents + self.enc_ffn(latents)

    # 5. 处理：多层自注意力
    for self_attn, self_ff in self.proc_layers:
        latents = latents + self.dropout(self_attn(latents, latent_timestamp_emb))
        latents = latents + self.dropout(self_ff(latents))

    # 6. 解码：latents → outputs (cross-attention)
    output_queries = output_queries + self.dec_atn(output_queries, latents, output_timestamp_emb, latent_timestamp_emb)
    output_latents = output_queries + self.dec_ffn(output_queries)

    # 7. 多任务读出
    output = self.readout(output_embs=output_latents, output_readout_index=output_decoder_index, ...)
    return output
```

**POYO 与 POYOPlus 的关键差异**：
- **输出查询构建**：POYO 版本中无 `task_emb`，output_queries 仅由 `session_emb` 构成
- **输出层**：POYO 使用 `nn.Linear` 直接投影；POYOPlus 使用 `MultitaskReadout` 按 `readout_index` 分发到不同 Linear 层
- **返回值**：POYO 返回 `Tensor` 或 `List[Tensor]`（取决于 `unpack_output`）；POYOPlus 返回 `Tuple[List[Dict[str, Tensor]]]`（每个样本的每个任务的预测字典）

**Variable-Length Forward**：所有 attention 模块均实现了 `forward_varlen()` 方法，支持将变长序列 chain 后通过 xformers 的 `BlockDiagonalMask` 高效处理，减少 padding 计算浪费。

### 3.5 CaPOYO 模型分析

CaPOYO 展示了 POYO 框架如何处理**非 spike 的连续值输入**（calcium imaging traces），其设计模式对 NeuroHorizon 有参考价值：

**关键设计差异**（相比 POYOPlus）：
1. **input_value_map**：`nn.Linear(1, dim // 2)` 将标量钙信号值映射到半维度空间
2. **unit_emb 维度减半**：`InfiniteVocabEmbedding(dim // 2)`（而非 `dim`）
3. **拼接而非相加**：`cat((input_value_map(values), unit_emb(index)), dim=-1)` — 值嵌入和 unit 嵌入拼接为完整维度

**对 NeuroHorizon 的启示**：若 decoder 中需同时输入 bin 信息和 unit 信息（如 `concat(bin_repr, unit_emb)`），CaPOYO 的拼接模式是可参考的实现方式。

---

## 4. 数据管线

### 4.1 数据加载

**Dataset 类** (`torch_brain/dataset/dataset.py`)：
- 基于 HDF5 文件的 lazy loading
- 每个 HDF5 文件对应一个 recording session
- 通过 `temporaldata.Data` 对象提供结构化访问
- 支持时间域切片：`data.slice(start_time, end_time)`
- 提供 `get_recording_hook` 方法，可被子类覆盖用于自定义后处理（如 `SpikingDatasetMixin.get_recording_hook` 给 unit_id 加前缀）

**数据索引** (`DatasetIndex`)：
- 三元组：`(recording_id, start_time, end_time)`
- 由 Sampler 生成

### 4.2 采样策略

| 采样器 | 文件位置 | 用途 |
|--------|---------|------|
| `RandomFixedWindowSampler` | `data/sampler.py` | 训练：从训练区间随机采样固定长度窗口，支持时间抖动增强 |
| `SequentialFixedWindowSampler` | 同上 | 确定性顺序滑动窗口 |
| `TrialSampler` | 同上 | 按 trial 区间采样（对 trial-aligned 预测有参考价值） |
| `DistributedEvaluationSamplerWrapper` | 同上 | 通用分布式评估包装器 |
| `DistributedStitchingFixedWindowSampler` | 同上 | 分布式推理 + 拼接：步长 window_length/2 滑动窗口，配合 `DecodingStitchEvaluator` 使用 |

### 4.3 批处理整理 (Collation)

`torch_brain/data/collate.py` 提供的核心函数：
- `pad8(seq)`: 将序列填充到 8 的倍数（GPU 效率优化）
- `track_mask8(seq)`: 生成填充位置的布尔 mask
- `chain(sequences)`: 将变长序列首尾相连
- `track_batch(sequences)`: 追踪每个元素所属的 batch index

另有 `pad`, `track_mask`, `pad2d`, `track_mask2d` 等变体。

### 4.4 Tokenization

POYOPlus 的 `tokenize()` 方法：

1. **Spike tokens**: 每个 spike (unit_id, timestamp) 成为一个 token
2. **Start/End tokens**: 每个 unit 的时间窗口起止标记
3. **Latent tokens**: 等间距可学习 latent tokens（由 `create_linspace_latent_tokens` 生成）
4. **Output queries**: session + task 嵌入，在指定时间戳处查询行为预测

**输出查询构建**：`prepare_for_multitask_readout()` 负责从 Data 对象中提取各任务时间戳和值、执行 z-score 归一化、根据 eval_interval 生成评估 mask、分配 readout_index。

**关键数据流**：
```
Raw HDF5 → Dataset.__getitem__() → data.slice(start, end) → Transforms → model.tokenize() → Batch
```

**Collation 规范**：tokenize 返回的字典需使用 `pad8()`, `chain()` 等包装函数标记每个字段的 collation 策略，NeuroHorizon 的 tokenize 必须遵循此规范。

---

## 5. 训练流程

### 5.1 框架

基于 **PyTorch Lightning** + **Hydra** 配置管理：
- `TrainWrapper(LightningModule)`: 封装模型的训练/验证/测试步骤
- `DataModule(LightningDataModule)`: 管理数据加载和预处理
- Hydra YAML 配置文件控制所有超参数

### 5.2 优化器

**SparseLamb** (`torch_brain/optim.py`):
- LAMB 优化器的变体，只更新梯度非零的参数
- 特别适用于 InfiniteVocabEmbedding（不是每次都激活所有词汇）
- 参数组（`examples/poyo_plus/train.py`）：
  - `sparse=True`：unit_emb + session_emb + **readout** 参数合并为一组
  - 标准更新：其余所有参数

### 5.3 学习率调度

- **Base LR**: 3.125e-5（按 batch size 线性缩放）
- **Weight Decay**: 1e-4
- **Scheduler**: OneCycleLR, cosine annealing
  - `div_factor=1`：初始 lr = max_lr（无从低到高的 warmup 阶段）
  - `pct_start=0.5`（通过 `cfg.optim.lr_decay_start` 配置）：前 50% 步数保持高 lr，后 50% cosine 衰减

### 5.4 验证与评估

- **DecodingStitchEvaluator**（自定义 Lightning Callback）：拼接重叠窗口预测、计算任务特定指标（R²Score）、支持加权 loss、在指定区间上评估（如 reach_period）
- **MultiTaskDecodingStitchEvaluator**：用于 POYOPlus 的多任务评估

---

## 6. 模态注册系统

`torch_brain/registry.py` 提供全局模态注册：

```python
@dataclass
class ModalitySpec:
    id: int          # 唯一数字 ID
    dim: int         # 输出维度
    type: DataType   # CONTINUOUS/BINARY/MULTINOMIAL/MULTILABEL
    timestamp_key: str  # 数据中时间戳的访问路径
    value_key: str      # 数据中值的访问路径
    loss_fn: Callable   # 损失函数
```

**已注册模态**（共 19 个）：
| 模态名 | dim | 类型 | Loss |
|--------|-----|------|------|
| cursor_velocity_2d | 2 | CONTINUOUS | MSE |
| cursor_position_2d | 2 | CONTINUOUS | MSE |
| arm_velocity_2d | 2 | CONTINUOUS | MSE |
| running_speed | 1 | CONTINUOUS | MSE |
| drifting_gratings_orientation | 8 | MULTINOMIAL | CE |
| natural_scenes | 119 | MULTINOMIAL | CE |
| natural_movie_one_frame | 900 | MULTINOMIAL | CE |
| 等... | | | |

---

## 7. 依赖环境

**核心依赖** (pyproject.toml)：
- `torch~=2.0`: 深度学习框架
- `temporaldata>=0.1.3`: 时间序列数据结构
- `einops~=0.6.0`: 张量操作
- `hydra-core~=1.3.2`: 配置管理
- `torchmetrics>=1.6.0`: 指标计算
- `pydantic~=2.0`: 数据验证

**运行时依赖**：
- `lightning`: PyTorch Lightning 训练框架
- `torch-optimizer==0.3.0`: SparseLamb 优化器
- `wandb`: 实验记录
- `brainsets`: 神经数据集工具

**当前已验证环境** (conda env `poyo`):
- Python 3.10
- PyTorch 2.10.0+cu128
- Lightning 2.6.1
- brainsets 0.2.1.dev4 (GitHub)
