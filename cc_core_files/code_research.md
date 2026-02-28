# POYO 代码架构分析

**日期**：2026-02-21
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

---

## 2. 项目结构

```
NeuroHorizon/
├── torch_brain/                    # 主包
│   ├── models/                     # 模型定义
│   │   ├── poyo.py                 # POYO 基础模型
│   │   ├── poyo_plus.py            # POYO+ 多任务扩展（377行核心逻辑）
│   │   └── capoyo.py              # CaPOYO 钙成像变体
│   ├── nn/                         # 神经网络模块
│   │   ├── rotary_attention.py     # 旋转位置编码注意力
│   │   ├── infinite_vocab_embedding.py  # 动态词汇嵌入
│   │   ├── multitask_readout.py    # 多任务读出层
│   │   ├── embedding.py            # 标准嵌入
│   │   ├── feed_forward.py         # FFN层
│   │   └── loss.py                 # 损失函数（MSE, CE, Mallow）
│   ├── data/                       # 数据加载与采样
│   │   ├── dataset.py              # 废弃的数据加载器
│   │   ├── collate.py              # 批处理整理（pad8, chain, track_mask8）
│   │   └── sampler.py              # 采样器（RandomFixedWindow, StitchingFixedWindow）
│   ├── dataset/                    # 新数据集 API
│   │   ├── dataset.py              # 活跃的 Dataset 类（HDF5加载）
│   │   ├── mixins.py               # Dataset mixins
│   │   └── nested.py               # NestedSpikingDataset（多数据集组合）
│   ├── transforms/                 # 数据增强
│   │   ├── unit_dropout.py         # 随机丢弃神经元
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
│       ├── train.py                # 训练脚本（418行）
│       └── configs/                # 配置文件
├── tests/                          # 单元测试
├── docs/                           # 文档
└── pyproject.toml                  # 包配置与依赖
```

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
| Token Type Embedding | `Embedding` | `nn/embedding.py` | 区分 spike/start/end tokens (4种) |
| Latent Embedding | `Embedding` | 同上 | 可学习的 latent tokens |
| 时间编码 | `RotaryTimeEmbedding` | `nn/rotary_attention.py` | RoFormer 风格旋转位置编码 |
| Cross-Attention | `RotaryCrossAttention` | `nn/rotary_attention.py` | 带 RoPE 的交叉注意力 |
| Self-Attention | `RotarySelfAttention` | `nn/rotary_attention.py` | 带 RoPE 的自注意力 |
| FFN | `FeedForward` | `nn/feed_forward.py` | 带 dropout 的前馈网络 |
| 多任务读出 | `MultitaskReadout` | `nn/multitask_readout.py` | 按任务分发的线性读出层 |

### 3.3 模型规模

| 配置 | 参数量 | dim | depth | latent_step | num_latents_per_step | heads |
|------|--------|-----|-------|-------------|---------------------|-------|
| POYO-MP | ~1.3M | 64 | 6 | 0.125 | 16 | 4/4 |
| POYO-1 | ~11.8M | 128 | 24 | 0.125 | 32 | 4/8 |

### 3.4 Forward Pass 详解 (POYOPlus)

基于 `torch_brain/models/poyo_plus.py` 第 200-267 行：

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

---

## 4. 数据管线

### 4.1 数据加载

**Dataset 类** (`torch_brain/dataset/dataset.py`)：
- 基于 HDF5 文件的 lazy loading
- 每个 HDF5 文件对应一个 recording session
- 通过 `temporaldata.Data` 对象提供结构化访问
- 支持时间域切片：`data.slice(start_time, end_time)`

**数据索引** (`DatasetIndex`)：
- 三元组：`(recording_id, start_time, end_time)`
- 由 Sampler 生成

### 4.2 采样策略

**训练采样** — `RandomFixedWindowSampler`：
- 从训练区间中随机采样固定长度窗口
- 支持时间抖动（temporal jitter）数据增强
- 产出：`DatasetIndex(recording_id, random_start, random_start + window_length)`

**验证/测试采样** — `DistributedStitchingFixedWindowSampler`：
- 按步长为 window_length/2 的滑动窗口覆盖序列
- 支持分布式评估
- 配合 `DecodingStitchEvaluator` 将重叠预测拼接为连续输出

### 4.3 批处理整理 (Collation)

`torch_brain/data/collate.py` 提供的核心函数：
- `pad8(seq)`: 将序列填充到 8 的倍数（GPU 效率优化）
- `track_mask8(seq)`: 生成填充位置的布尔 mask
- `chain(sequences)`: 将变长序列首尾相连
- `track_batch(sequences)`: 追踪每个元素所属的 batch index

### 4.4 Tokenization

POYOPlus 的 `tokenize()` 方法（第 269-353 行）：

1. **Spike tokens**: 每个 spike (unit_id, timestamp) 成为一个 token
2. **Start/End tokens**: 每个 unit 的时间窗口起止标记
3. **Latent tokens**: 等间距可学习 latent tokens（由 `create_linspace_latent_tokens` 生成）
4. **Output queries**: session + task 嵌入，在指定时间戳处查询行为预测

**关键数据流**：
```
Raw HDF5 → Dataset.__getitem__() → data.slice(start, end) → Transforms → model.tokenize() → Batch
```

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
- 参数组：
  - Unit/Session embeddings: `sparse=True`
  - 其余参数: 标准更新

### 5.3 学习率调度

- **Base LR**: 3.125e-5（按 batch size 线性缩放）
- **Weight Decay**: 1e-4
- **Scheduler**: OneCycleLR, cosine annealing
  - 50% warm-up, 然后 cosine 衰减

### 5.4 验证与评估

**DecodingStitchEvaluator**（自定义 Lightning Callback）：
- 拼接重叠窗口的预测结果
- 计算任务特定指标（R²Score）
- 支持加权 loss
- 在指定区间上评估（如 reach_period）

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

**已注册模态**：
| 模态名 | dim | 类型 | Loss |
|--------|-----|------|------|
| cursor_velocity_2d | 2 | CONTINUOUS | MSE |
| cursor_position_2d | 2 | CONTINUOUS | MSE |
| arm_velocity_2d | 2 | CONTINUOUS | MSE |
| running_speed | 1 | CONTINUOUS | MSE |
| drifting_gratings_orientation | 8 | MULTINOMIAL | CE |
| natural_scenes | 119 | MULTINOMIAL | CE |
| natural_movie_one_frame | 900 | MULTINOMIAL | CE |
| 等（共 16 个模态）| | | |

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

---

## 8. 与 NeuroHorizon 改造的关键接口

以下是 NeuroHorizon 需要修改/替换的关键接口点：

### 8.1 需要替换的组件
- `InfiniteVocabEmbedding` → **IDEncoder** (从参考窗口原始神经活动生成 unit embedding（参考 SPINT 架构）)
- `MultitaskReadout` → **Shared per-neuron MLP** (适应可变输出维度)
- `MSELoss/CrossEntropyLoss` → **PoissonNLLLoss** (spike count 预测)

### 8.2 需要扩展的组件
- `RotarySelfAttention` → 支持 **causal masking** (自回归解码器)
- `registry.py` → 注册新的 **spike_counts** 模态
- Decoder → 从单层 cross-attention 扩展为 **多层 autoregressive decoder**

### 8.3 需要保持的接口
- `model.tokenize(data: Data) -> Dict`: tokenizer 接口
- `model.forward(**batch["model_inputs"]) -> output`: forward 接口
- `Dataset.__getitem__(DatasetIndex) -> Data`: 数据加载接口
- `collate(batch_list)`: 批处理接口

---

## 9. 已验证的基线

根据 `cc_todo/poyo_setup_log.md`：
- POYO (1.3M params) 在 MC Maze Small 数据集上成功运行
- 2 epoch 后 R² ≈ -0.03（正常初始值，完整训练需 1000 epoch）
- GPU (CUDA) 正常调用，checkpoint 已保存
