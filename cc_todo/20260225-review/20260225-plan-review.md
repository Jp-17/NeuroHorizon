# plan.md 审查分析报告

**审查日期**：2026-02-25
**审查对象**：cc_core_files/plan.md
**审查依据**：code_research.md 审查 + proposal.md 审查 + dataset.md + POYO 代码库实际实现 + 工程可行性评估

---

## 总体评价

plan.md 是一份结构清晰的执行计划，包含合理的阶段划分、增量式开发策略、风险评估和里程碑定义。但在**技术细节**、**依赖关系**、**时间估算**和**与 proposal/code_research 的一致性**方面存在若干问题。以下按文档结构逐段分析。

---

## 第1-2节：项目目标与 POYO 差异 — 准确清晰

差异对比表准确且有价值。

**建议补充一行**：

| 维度 | POYO/POYO+ | NeuroHorizon |
|------|-----------|-------------|
| **激活函数** | GEGLU | 待定（proposal 说 SwiGLU，POYO 用 GEGLU） |

这个不一致应在正式开发前解决。

---

## 第3节：合理性评估 — 问题识别准确，应对方案需要深化

### 3.2 问题 1：Jia Lab 数据不可用

应对方案"IBL 连续记录 + Allen Natural Movies"基本合理，但有一个遗漏：

**缺少对"大量重复 trial"替代方案的深入讨论**

Jia Lab 数据的核心优势是大量重复 trial（同一刺激重复数十到上百次），这使得可以计算可靠的 trial-averaged PSTH 作为 ground truth。IBL 和 Allen 的替代方案如何解决这个问题？

- **IBL**：每个特定对比度条件通常重复 20-50 次，可以计算 PSTH 但统计可靠性不如 Jia Lab
- **Allen Natural Scenes**：每张图像重复约 50 次（across trials），可以用于 PSTH，但每次呈现只有 250ms
- **Allen Natural Movies**：约 10 次重复的 30s 连续视频，PSTH 的统计可靠性较低

**建议**：在计划中明确使用 Allen Natural Scenes（50 次重复）作为 PSTH 评估的主要来源，尽管每次呈现只有 250ms。对于长时程预测（>250ms），使用 Allen Natural Movies 的连续响应进行评估，但需要接受 trial 重复次数较少带来的更大统计误差。

### 3.2 问题 3：可变输出维度

设计方案"shared per-neuron MLP head: concat(bin_repr, unit_emb) → log_rate"是合理的。但需要补充以下技术细节：

1. **批处理效率**：同一 batch 内不同 session 的 neuron 数量不同。如果每个 sample 有不同数量的 neuron 输出，batch 化需要 padding 或者使用不规则张量。
2. **MLP 输入维度**：如果 bin_repr 的维度是 d_model（如 512），unit_emb 的维度也是 d_model，拼接后输入维度为 2*d_model。这个 MLP 的参数量和计算量需要评估。
3. **与 CaPOYO 的参考**：CaPOYO 使用 `dim // 2` 的 unit embedding + `dim // 2` 的 value mapping 拼接为 `dim`。可以参考类似的维度分配策略，而非简单拼接两个 `dim` 维向量。

### 3.2 问题 4：自回归误差累积

应对方案中提到的 "Scheduled sampling" 和 "Non-autoregressive parallel prediction baseline" 是正确的。

**需要补充的重要策略**：

1. **Teacher forcing ratio 的 schedule**：需要明确从什么比例开始（如 100% teacher forcing），以什么速度衰减（如线性或指数），到什么最终比例（如 10% teacher forcing）。参考经验：通常从 100% 开始，在 20-50 个 epoch 内线性衰减到 0%。
2. **Coarse-to-fine 策略的具体设计**：先用 100ms bin 预测（10 步），再用 20ms bin 细化。这需要两级 decoder，增加了模型复杂度。需要评估是否值得。
3. **预测质量的时间衰减分析**：应该在早期实验中就绘制"预测准确性 vs 预测时间步"的曲线，了解误差累积的速度，据此决定是否需要缩短预测窗口。

### 3.2 问题 6：rotary_attention 不支持 causal mask

描述准确。

**更具体的修改方案**：

当前 `rotary_attn_pytorch_func` 中 mask 的处理是：
```python
# 当前：mask shape (B, N) → reshape to (B, 1, 1, N)
attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)
```

需要改为支持多种 mask 形状：
```python
# 方案1：直接支持 2D causal mask
if attn_mask.dim() == 2:  # (B, N_kv)
    attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, N_kv)
elif attn_mask.dim() == 3:  # (B, N_q, N_kv)
    attn_mask = attn_mask.unsqueeze(1)  # (B, 1, N_q, N_kv)
elif attn_mask.dim() == 4:  # (B, H, N_q, N_kv)
    pass  # already correct shape

# 方案2：使用 PyTorch 的 is_causal 参数
# F.scaled_dot_product_attention(..., is_causal=True)
```

方案 2 更简洁，但要求 query 和 key 长度相同。对于 cross-attention（decoder attend to encoder），只有方案 1 可行。

**建议**：在 plan 中明确这是一个需要仔细实现和测试的技术点，不应被低估。

---

## 第4节：架构设计 — 与 proposal 的一致性检查

### 4.1 整体架构图

架构图与 proposal 的描述基本一致，但比 proposal 更具体化了一些决策（如明确了 20ms bins、IDEncoder 输入特征 ~33 维等）。

**发现的一个问题**：架构图中写"[Token Embedding] = IDEncoder(unit_idx) + TokenType + RoPE"，但 RoPE 不是加到 token embedding 上的，而是在 attention 计算时直接应用于 query 和 key。这个描述与 POYO 实际实现不一致：
```python
# POYO 实际实现：RoPE 不加到 embedding 上
inputs = self.unit_emb(input_unit_index) + self.token_type_emb(input_token_type)
input_timestamp_emb = self.rotary_emb(input_timestamps)  # 单独计算，在 attention 中使用
```

### 4.2 IDEncoder

IDEncoder 输入特征设计（~33 维）：
- 平均发放率 (1d)
- ISI 变异系数 (1d)
- ISI log-histogram (20d)
- 自相关特征 (10d)
- Fano factor (1d)

**技术可行性评估**：

1. **ISI log-histogram (20d)**：需要将 ISI 分布离散化到 20 个 log-spaced bins。对于低发放率的 neuron（<1 Hz），参考窗口内可能只有几十个 ISI，直方图会非常嘈杂。建议考虑使用 kernel density estimation 代替。
2. **自相关特征 (10d)**：需要定义自相关的时间范围和分辨率。建议使用 0-100ms 范围、10ms 分辨率（10 个值），或者使用 log-spaced lags。
3. **发放率和 Fano factor**：简单可靠，没有问题。
4. **缺少波形特征**：proposal.md 提到了波形特征，但 plan.md 的 33 维特征中没有包含。IBL 数据确实有波形模板（waveforms），但下载量较大。建议先不包含，作为后续扩展。

**网络结构**："3 层 MLP (Linear + GELU + LayerNorm)"——与 proposal 的 3 层 FFN 一致，但 proposal 使用 GELU 而 POYO 的 FFN 使用 GEGLU。建议明确 IDEncoder 使用标准 GELU（因为输入不是高维嵌入，GEGLU 的门控机制在低维输入上可能不必要）。

### 4.4 模型规模

与 proposal 一致。需要验证参数量估算（见 proposal review）。

---

## 第5节：数据集 — 与 dataset.md 的一致性

### 5.1 IBL 数据集

"459 sessions, 139 mice, 12 labs, 621,733 total units, 75,708 good quality units, 241 brain regions"——这些数字与 dataset.md 一致。

**需要注意**：IBL 2025 版本的确切数据量需要在实际下载后确认。ONE API 搜索可能返回不同数量的 session（取决于过滤条件）。

### 5.2 Allen 数据集

"58 sessions"——正确。

**存储预估需要更新**：
- "IBL (spike + behavior): ~100-200 GB"——如果下载全量 459 sessions 的 spike + behavior 数据，这个估算合理
- "Allen (NWB): ~150 GB"——如果下载全部 58 个 NWB session 文件（约 146.5 GB），这个估算准确
- 但如果预处理后转为 HDF5（仅保留 spike times + behavior），Allen 的存储可能降至 20-30 GB

---

## 第6节：执行计划 — 核心问题分析

### Phase 0：环境与数据准备

**问题1：缺少 NLB 数据集的下载脚本**

proposal.md 将 NLB 列为"必要基准"（★★☆），但 Phase 0 的脚本清单中没有 NLB 数据的下载和预处理脚本。需要补充。

**问题2：数据预处理的目标格式未明确**

Phase 0.4 说"转 HDF5"，但没有明确 HDF5 的字段结构。dataset.md 中有一个示例格式（Section 4.2），但 plan.md 没有引用。建议在 Phase 0 中明确数据格式规范，确保与 POYO 的 Dataset 类兼容。

POYO 的 Dataset 类期望 HDF5 文件中包含 `temporaldata.Data` 可以序列化/反序列化的结构。需要参考 POYO 现有数据集的 HDF5 格式来设计 NeuroHorizon 的数据格式。

**问题3：参考特征提取（0.5）依赖于数据预处理（0.4），但计划中两者看起来是并列的**

实际依赖链：
```
0.1 (环境) → 0.2/0.3 (下载) → 0.4 (预处理) → 0.5 (特征提取) → 0.6 (验证)
```
建议在计划中明确标注这个依赖顺序。

### Phase 1：POYO 基线验证

**问题1："IBL 适配"的复杂度被低估**

Phase 1.1 说"在 IBL 数据上运行现有 POYO+ 模型"。但 POYO+ 的现有数据集适配器可能不支持 IBL 数据格式。需要：
1. 实现 IBL 数据的 Dataset 子类
2. 确保 IBL 数据中的行为变量（wheel velocity）对应到 POYO 注册的模态
3. 处理 IBL 数据中 "good quality" unit 的过滤逻辑
4. 处理 IBL 数据中 trial 结构和时间戳的对齐

这些工作量可能占 1-2 周，不应被简单概括为"IBL 适配"。

**问题2：基线指标 "R² > 0" 标准过低**

Phase 1.2 的标准"wheel velocity 解码 R² > 0"意味着只要模型的预测比全零预测稍好就算通过。这不足以验证 POYO 在 IBL 上的正确运行。建议参考 POYO 论文中报告的 IBL 数据 R² 值，设定更有意义的验证标准（如 R² > 0.3 或与论文报告值的差距 < 20%）。

### Phase 2：核心模型实现

**问题1：2.1-2.6 的依赖关系不清晰**

这些步骤有明确的依赖关系：
```
2.1 (PoissonNLLLoss) ← 独立
2.2 (spike_counts 模态) ← 独立
2.3 (IDEncoder) ← 独立，但需要 0.5 的参考特征
2.4 (NeuroHorizon 模型) ← 依赖 2.1, 2.2, 2.3，以及 rotary_attention 的 causal mask 修改
2.5 (训练流程) ← 依赖 2.4
2.6 (评估指标) ← 独立
```

特别注意：**2.4 还隐含依赖于 rotary_attention.py 的 causal mask 修改**，但这个修改在 Phase 2 的任务清单中没有明确列出。它被放在了 Section 8 "关键文件清单" 中，但没有对应的 Phase 2 步骤。

**建议**：在 Phase 2 中增加一个显式步骤 "2.X：修改 rotary_attention.py 支持 causal mask"，放在 2.4 之前。

**问题2：NeuroHorizon 模型的实现复杂度被低估**

Phase 2.4 说"新建 `torch_brain/models/neurohorizon.py`"。但这个文件需要实现：
- 继承/组合 POYO 的 Perceiver encoder
- 实现多层 autoregressive decoder（cross-attention + causal self-attention + FFN）
- 实现 per-neuron MLP head
- 实现 tokenize() 方法（需要构建 spike count targets）
- 处理可变数量的输出神经元
- 多模态 cross-attention 的预留接口

这是一个相当大的工程量，建议拆分为多个子步骤，分别实现和测试。

### Phase 3：多模态扩展

Phase 3.1-3.3 的设计合理，但：

**问题：DINOv2 模型的 GPU 显存占用**

如果在训练时加载冻结的 DINOv2 模型（如 ViT-B/14，~86M 参数），会占用额外的 GPU 显存。在 4090 (24GB) 上同时运行 NeuroHorizon (30M) + DINOv2 (86M) + 数据 + 梯度，显存可能吃紧。

**建议**：Phase 3.1 中应明确"预计算 DINOv2 embedding 并存储"的策略（dataset.md 中已提到），不要在训练时实时计算 DINOv2 embedding。这样可以避免显存问题，也加速训练。

### Phase 4：实验

实验列表全面，但：

**问题1：实验优先级排序需要更明确**

当前实验 1-6 按编号排列，但没有明确的优先级。结合 MVP 路径，建议排序为：
1. **长时程预测**（核心创新验证）
2. **跨 Session 泛化**（核心创新验证）
3. **消融实验**（A1 IDEncoder、A4 窗口长度、A10 Loss 函数）
4. **多模态贡献分析**
5. **Data Scaling Laws**
6. **下游任务泛化**

**问题2：缺少 baseline 方法的实现计划**

实验需要与 Neuroformer、NDT3、LFADS 等方法对比，但 plan 中没有这些 baseline 的实现/复现步骤。需要明确：
- 使用原始代码库复现？
- 使用论文报告的数字直接对比？
- 在相同数据集上重新训练 baseline？

---

## 第7节：MVP 路径 — 方向正确但需要补充

MVP 定义为"IBL 数据 + 核心 NeuroHorizon 模型 + 跨 session 泛化实验 + 长时间预测实验 + 关键消融"。这是合理的最小可发表单元。

**需要补充的关键 MVP 细节**：

1. **MVP 的模型规模**：应使用 Small (5M) 还是 Base (30M)？建议 MVP 使用 Small 做全流程验证，再用 Base 做最终实验。
2. **MVP 需要的最少 session 数**：跨 session 泛化至少需要 30-50 个 training sessions 和 10+ 个 test sessions。建议 MVP 使用 50 个 IBL sessions。
3. **MVP 的时间估算**：Phase 0 + Phase 1 + Phase 2（不含多模态）+ Phase 4 的核心实验 ≈ 10-12 周。这与总体 16-20 周的时间线一致。

---

## 第8节：关键文件清单 — 需要扩展

### 需要修改的文件

| 文件 | 文档描述 | 实际需要的修改 | 遗漏 |
|------|---------|-------------|------|
| `torch_brain/nn/loss.py` | 添加 PoissonNLLLoss | 正确 | — |
| `torch_brain/nn/__init__.py` | 导出 IDEncoder | 正确 | 还需导出 PoissonNLLLoss |
| `torch_brain/nn/rotary_attention.py` | 支持 causal mask | 正确，但比列出的更复杂 | 需要同时修改 pytorch 和 xformers 两个后端 |
| `torch_brain/registry.py` | 注册 spike_counts 模态 | 正确 | — |
| `torch_brain/models/__init__.py` | 导出 NeuroHorizon | 正确 | — |

**遗漏的修改文件**：
- `torch_brain/data/collate.py`：可能需要新的 collation 函数来处理可变数量的神经元输出
- `torch_brain/utils/tokenizers.py`：可能需要新增 token 类型（如 decoder bin query token）
- `torch_brain/dataset/dataset.py`：可能需要扩展 `get_recording_hook` 来处理参考窗口特征提取

### 需要新建的文件

文件清单基本完整。

**建议补充**：
- `torch_brain/nn/causal_attention.py`（或修改 rotary_attention.py）：如果 causal mask 的修改比较大，可以考虑新建文件而非修改现有文件
- `scripts/compute_reference_features.py`：计算 IDEncoder 参考特征的脚本（与 `extract_reference_features.py` 重复？需要明确是同一个还是不同的）
- `tests/test_neurohorizon.py`：核心模块的单元测试

---

## 第9节：风险与应对 — 补充建议

### 现有风险评估基本准确

Jia Lab 数据、自回归误差累积、显存限制、Natural Scenes 时间间隔、依赖冲突、下载速度——这些风险的识别和应对都是合理的。

### 缺少的风险项

| 风险 | 等级 | 说明 |
|------|------|------|
| **IDEncoder 特征维度不足** | 中 | 33 维手工特征可能不足以区分功能特性相似但空间位置不同的神经元 |
| **Poisson NLL Loss 训练不稳定** | 中 | 对于低发放率神经元，预测的 λ 接近 0 时 log(λ) 趋向负无穷，需要数值稳定化处理 |
| **跨数据集格式不统一** | 中 | IBL 和 Allen 的原始数据格式差异大，统一为 POYO 兼容的 HDF5 格式的工作量可能超出预期 |
| **POYO 代码 API 变化** | 低 | POYO 是 NeurIPS 2023 的工作，代码库可能不再积极维护，依赖的库（如 temporaldata）版本可能出现兼容性问题 |
| **评估指标选择不当** | 中 | Spike count 预测的评估指标（Poisson log-likelihood, R², PSTH correlation）各有侧重，需要在实验设计中明确主要指标 |

---

## 与 proposal.md 的一致性对比

| 方面 | proposal.md | plan.md | 一致性 |
|------|------------|---------|--------|
| 模型规模 | Small/Base/Large | 相同 | ✅ |
| 优化器 | AdamW | 未明确指定 | ⚠️ 需统一 |
| 数据集 | Jia Lab ★★★, IBL ★★★, Allen ★★☆, NLB ★★☆ | IBL ★, Allen ★★ (Jia Lab 已标为不可用) | ⚠️ 需在 proposal 中同步更新 |
| 多模态注入位置 | "每隔2层" | "可选：每隔2层" | ✅ |
| Perceiver | "可选" | 架构图中包含但未标注为必需 | ⚠️ 应标为必需 |
| 课程学习 | 有 | 未提及 | ⚠️ plan.md 应补充 |
| 预测窗口 | 100/200/500/1000ms | 100/200/500/1000ms | ✅ |
| 消融实验 | 10 个 | 10 个 | ✅ |
| 自回归误差累积 | 未详细讨论 | Scheduled sampling + parallel baseline | plan.md 更详细 ✅ |

---

## 总结：plan.md 的关键修正建议

### 高优先级

1. **增加显式步骤：修改 rotary_attention.py 支持 causal mask**——这是 decoder 实现的前置依赖，不应被隐藏在"文件清单"中
2. **Phase 2.4 拆分为多个子步骤**——NeuroHorizon 模型实现的复杂度太高，需要拆分为 encoder 复用、decoder 实现、tokenize 实现、forward 集成等子任务
3. **Phase 0 增加 NLB 数据下载和预处理**——NLB 是必要的 benchmark 基准
4. **明确 Phase 0 的依赖关系**——下载→预处理→特征提取→验证的顺序需要清晰标注
5. **提高 Phase 1.2 基线标准**——"R² > 0" 过低，应设定更有意义的验证阈值

### 中优先级

6. **补充优化器选择的说明**——SparseLamb 还是 AdamW，对哪些参数组使用哪种优化策略
7. **补充课程学习策略的具体实现**——proposal 提到但 plan 未包含
8. **明确 DINOv2 embedding 的预计算策略**——避免训练时的显存问题
9. **补充 baseline 方法的实现/对比计划**——如何获得 NDT3、Neuroformer 等的对比数据
10. **增加缺失的风险项**——特别是 IDEncoder 特征维度不足和 Poisson NLL 数值稳定性

### 低优先级

11. **统一术语**——proposal 用"encoder"、plan 用"processor"，建议统一
12. **补充单元测试计划**——关键模块（IDEncoder、causal attention、per-neuron MLP）应有测试
13. **明确 MVP 使用的模型规模和 session 数量**

---

## 附录：推荐的修正后 Phase 2 任务清单

```
Phase 2: 核心模型实现

2.1 [独立] PoissonNLLLoss
    - 修改 torch_brain/nn/loss.py
    - 注意数值稳定性（clamp log_rate）
    - 实现 forward(input, target, weights) 接口

2.2 [独立] spike_counts 模态注册
    - 修改 torch_brain/registry.py
    - 定义 timestamp_key, value_key

2.3 [依赖 Phase 0.5] IDEncoder 模块
    - 新建 torch_brain/nn/id_encoder.py
    - 输入：33 维参考特征
    - 输出：d_model 维 unit embedding
    - 3 层 MLP (Linear + GELU + LayerNorm)
    - 单元测试

2.4 [独立] Causal Mask 支持
    - 修改 torch_brain/nn/rotary_attention.py
    - 修改 rotary_attn_pytorch_func 的 mask 处理逻辑
    - 修改 rotary_attn_xformers_func 的 mask 处理逻辑
    - 新增 create_causal_mask 工具函数
    - 单元测试：验证 causal mask 正确阻止未来信息

2.5 [依赖 2.1-2.4] NeuroHorizon 模型（分三步）
    2.5a 模型骨架
        - 新建 torch_brain/models/neurohorizon.py
        - 复用 POYO 的 Perceiver encoder + processing layers
        - 添加 IDEncoder 作为 unit embedding 生成器
        - 实现 forward() 的 encoder 部分，验证维度匹配
    2.5b Autoregressive Decoder
        - 实现多层 decoder block (cross-attn + causal self-attn + FFN)
        - 实现 per-neuron MLP head
        - 实现 forward() 的 decoder 部分
        - 验证 teacher forcing 和自回归推理两种模式
    2.5c Tokenize 方法
        - 实现 tokenize() 方法
        - 构建 spike count targets（binning spike events）
        - 处理参考窗口特征提取
        - 验证与 collate 函数的兼容性

2.6 [依赖 2.5] 训练流程
    - 新建 examples/neurohorizon/train.py + configs
    - 实现 TrainWrapper（适配 Poisson loss）
    - 实现 DataModule（适配 IBL/Allen 数据）

2.7 [独立] 评估指标
    - 新建 torch_brain/utils/neurohorizon_metrics.py
    - Poisson log-likelihood
    - PSTH correlation
    - R² of firing rates
    - 预测准确性随时间步的衰减曲线
```
