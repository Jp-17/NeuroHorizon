# NeuroHorizon 执行计划

> **本文档是项目的可执行计划主体，聚焦任务列表与完成状态。**
>
> - 项目背景与架构分析：`cc_core_files/proposal_review.md`（**执行前请先阅读对应 Phase 的内容**）
> - 数据集详细规划：`cc_core_files/dataset.md`
> - 完整研究提案：`cc_core_files/proposal.md`
> - 任务执行记录：`cc_todo/{phase-folder}/{YYYYMMDD}-{phase}-{task_num}-{task}.md`
>
> **关键文件清单、风险与应对、架构设计细节均在 `cc_core_files/proposal_review.md` 中。**

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

## 执行通则

> 以下规则适用于所有 Phase，优先级高于各 Phase 内的具体指引。

1. **显存不足处理**：遇到 OOM 或显存相关错误时，优先自行排查（batch size、梯度累积、混合精度、模型规模等）。若确认当前资源确实不够，直接告知用户，用户可提供更多资源。
2. **效果不达标处理**：实验效果未达预期时，优先在现有计划范围内排查和调整（超参数、训练策略、数据预处理等）。若穷尽计划内手段仍无法解决，可以质疑原始 proposal 方案，甚至提出替代方案——但 **必须写入文档并提前获得用户同意** 后再执行。

---

## Phase 0：环境准备与基线复现

> **目标**：验证开发环境完整性，深度理解 POYO 代码架构，在 Brainsets 数据上建立行为解码 baseline。
> **数据集**：Brainsets 原生（Perich-Miller 2018 为主）
> **执行参考**：`cc_core_files/proposal_review.md` 第一节（§一）
> **cc_todo**：`cc_todo/phase0-env-baseline/`

### 0.1 环境验证与代码理解

- [x] **0.1.1** 确认并验证 POYO conda 环境可用性
  - 📄 `cc_todo/phase0-env-baseline/20260228-phase0-env-code-understanding.md`
  - 服务器上已有 POYO 相关 conda 环境，先 `conda env list` 查看现有环境，尝试直接激活使用
  - 验证核心依赖完整性：PyTorch, wandb, hydra, brainsets；缺失项按需补装而非重建环境
  - 梳理代码模块依赖关系图：spike tokenization → unit embedding → Perceiver encoder → readout → 训练循环

- [x] **0.1.2** 精读 SPINT（IDEncoder 机制）和 Neuroformer（自回归生成 + 多模态）两篇关键论文
  - 📄 `cc_todo/phase0-env-baseline/20260228-phase0-env-code-understanding.md`

- [x] **0.1.3** 基于0.1.1和0.1.2的执行结果给予对于后续阶段要修改代码的建议
  - 📄 `cc_todo/phase0-env-baseline/20260228-phase0-env-code-understanding.md`

### 0.2 数据准备与探索

- [x] **0.2.1** 检查 `NeuroHorizon/data/` 中已有的 Brainsets 数据
  - 📄 `cc_todo/phase0-env-baseline/20260228-phase0-data-explore.md`
  - 列出 `data/raw/` 和 `data/processed/` 下的内容，判断是否已下载 Perich-Miller 或其他 Brainsets 数据集
  - 若已有数据：确认格式是否符合 brainsets pipeline 要求，可直接复用
  - 若无或不完整：通过 brainsets API 补充下载 `perich_miller_population_2018`（先 5-10 sessions）
  - 记录数据存放位置到 `cc_core_files/data.md`

- [x] **0.2.2** 数据加载验证
  - 📄 `cc_todo/phase0-env-baseline/20260228-phase0-data-explore.md`
  - 通过 POYO 数据 pipeline 的 sanity check，确认数据可正常流入训练框架

- [x] **0.2.3** 数据深度探索与可视化分析
  - 📄 `cc_todo/phase0-env-baseline/20260228-phase0-data-explore.md`

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
    - 辅助信息：trial 标签、目标方向、成功/失败标注等

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

- [x] **0.2.4** （可选）下载 NLB MC_Maze 数据（brainsets 内）作为改造后的 sanity check 基准
  - 📄 数据已存在于 `data/nlb/processed/pei_pandarinath_nlb_2021/`（jenkins_maze_train.h5 + jenkins_maze_test.h5）

### 0.3 POYO 基线复现

- [x] **0.3.1** 在 Perich-Miller 数据上运行现有 POYO+ 行为解码，验证 R² > 0.3（或接近论文数值 ±20%）
  - 📄 `cc_todo/phase0-env-baseline/20260228-phase0-poyo-baseline.md`
- [x] **0.3.2** 分析 POYO encoder 输出的 latent representation 质量（PCA / 解码探针）
  - 📄 `cc_todo/phase0-env-baseline/20260228-phase0-poyo-baseline.md`
- [x] **0.3.3** 记录基线性能报告，作为后续改造前后的对比锚点
  - 📄 `cc_todo/phase0-env-baseline/20260228-phase0-poyo-baseline.md`
- [x] **0.3.4** 分析 Perich-Miller 数据在 POYO+ 训练/评估中的完整数据流（时间尺度关系 + 采样/损失/评估机制可视化）
  - 📄 `cc_todo/phase0-env-baseline/20260309-phase0-0.3.4-data-flow-analysis.md`

### 0.4 Benchmark 探究

- [x] **0.4.1** NLB Benchmark 数据分析与适配性调查
  - 📄 `cc_todo/phase0-env-baseline/20260309-phase0-0.4-benchmark-analysis.md`
  - **Part A**: 分析 brainsets 中 NLB MC_Maze 数据的结构（类似 0.3.4 对 Perich-Miller 的分析）
    - 数据属性：domain, trials, train/valid/test_domain, nlb_eval_intervals, hand/eye 行为数据
    - 可视化时间轴 + 统计摘要
    - 与 Perich-Miller 数据结构的对比
  - **Part B**: NLB 数据适配性调查
    - brainsets 的 train/valid/test_domain 与 NLB 原始 split（train_mask_nwb/test_mask_nwb）是否一致
    - NLB 其他子数据集（MC_RTT, Area2_Bump, DMFC_RSG, MC_Cycle）能否方便地适配到当前 POYO 框架
    - test 文件 units 数量 (107) 与 train (142) 的差异（held-in/held-out 机制）对评估的影响
    - 结论：基于 brainsets 数据的评估结果是否与 NLB 平台的数值具有可比性
  - **Part C**: NLB 指标对齐
    - 调查 NLB 的核心指标：co-bps, fp-bps, PSTH R² 的计算方式
    - 评估 NeuroHorizon 中实现这些指标的可行性和所需改动
    - 检查 nlb_tools 包的评估 API

- [x] **0.4.2** 对比模型可行性分析（ndt2, neuroformer, ndt-mtm 等）
  - 📄 `cc_todo/phase0-env-baseline/20260309-phase0-0.4-benchmark-analysis.md`
  - 评估在 Perich-Miller 和 NLB MC_Maze 数据上与 ndt2, neuroformer, ndt-mtm 对比的可行性
  - 分析各模型的数据输入要求、是否支持当前数据组织方式
  - 评估适配难度和所需工作量

---

## Phase 1：自回归改造验证 + 长时程生成验证

> **目标**：实现真正的自回归解码器（含预测反馈），用新指标（fp-bps、PSTH-R²）验证，在多种预测/观察窗口和 session 数目下测试。
> **数据集**：Perich-Miller 2018（Brainsets 原生，1-10 sessions）
> **执行参考**：`cc_core_files/proposal_review.md` 第二节
> **cc_todo**：`cc_todo/phase1-autoregressive/`

### 1.1 核心模块实现

#### 1.1.1-1.1.6 [x] v1 基础模块（已完成）
> 依赖：`cc_core_files/proposal_review.md` §2.1–§2.9, `cc_core_files/code_research.md`
> 产出：`torch_brain/nn/loss.py`, `torch_brain/nn/autoregressive_decoder.py`, `torch_brain/models/neurohorizon.py`, `examples/neurohorizon/train.py` + configs
> 记录：`cc_todo/phase1-autoregressive/20260302-phase1-1.1-core-modules.md`

- [x] PoissonNLLLoss
- [x] spike_counts 模态注册
- [x] Causal mask 支持
- [x] AutoregressiveDecoder + PerNeuronMLPHead
- [x] NeuroHorizon 模型
- [x] 训练脚本 + configs

#### 1.1.7 [x] 评估指标补充 <!-- 记录：cc_todo/phase1-autoregressive/20260311-phase1-1.1.7-1.1.9-implementation.md -->
> 依赖：`cc_core_files/proposal_review.md` §2.10（指标体系）
> 产出：`torch_brain/utils/neurohorizon_metrics.py`, `scripts/analysis/neurohorizon/eval_psth.py`
> 记录：`cc_todo/phase1-autoregressive/20260311-phase1-1.1.7-1.1.9-implementation.md`

- [x] 重建 `torch_brain/utils/neurohorizon_metrics.py`
  - fp-bps（forward prediction bits per spike）
  - PSTH-R²（peri-stimulus time histogram R²）
  - 保留：r2_score、firing_rate_correlation、poisson_log_likelihood
- [x] fp-bps null model 计算：训练前遍历数据，计算 per-neuron 平均发放率
- [x] 在 train.py validation_step 中集成 fp-bps
- [x] Per-bin fp-bps（val/fp_bps_bin{t}）用于衰减分析
- [x] PSTH-R² 独立评估脚本（scripts/analysis/neurohorizon/eval_psth.py）

#### 1.1.8 [ ] AR 修复：预测反馈实现【方案待决策】
> 依赖：`cc_core_files/proposal_review.md` §2.5b（AR 反馈机制）, `cc_todo/phase1-autoregressive/20260302-phase1-1.2-foundation-verify.md`（TF≡AR 分析）
> 产出：`torch_brain/nn/prediction_feedback.py`, `torch_brain/nn/autoregressive_decoder.py`（feedback 参数）, `torch_brain/models/neurohorizon.py`（generate 反馈）
> 记录：`cc_todo/phase1-autoregressive/20260311-phase1-1.1.7-1.1.9-implementation.md`

- [x] 分析 TF≡AR 问题根因（bin_query 无状态，causal mask 仅限隐状态可见性）
- [x] 实现 prediction_feedback.py：4 种反馈编码方式（mlp_pool / rate_weighted / cross_attn / none）
- [x] 修改 autoregressive_decoder.py：forward() 接受 feedback 参数（方案 A: Query Augmentation）
- [x] 修改 neurohorizon.py：forward() 传递 GT counts（TF），generate() 使用预测 feedback（AR）
- [x] 修改 train.py：传递 target counts + feedback_method 配置
- [ ] **方案决策**：对比方案 A/B/C，确定最终集成方式（当前仅实现方案 A）
- [ ] **编码方式决策**：对比 4 种编码方式的效果（需 1.2.3 验证结果支持）

#### 1.1.9 [x] Trial-Aligned 数据加载 <!-- 记录：cc_todo/phase1-autoregressive/20260311-phase1-1.1.7-1.1.9-implementation.md -->
> 依赖：HDF5 数据结构（`data/processed/perich_miller_population_2018/*.h5` 中 trials 字段），`cc_core_files/proposal_review.md` §2.1b
> 产出：`torch_brain/data/trial_sampler.py`, `torch_brain/data/dataset.py`（get_trial_intervals）
> 记录：`cc_todo/phase1-autoregressive/20260311-phase1-1.1.7-1.1.9-implementation.md`

- [x] 新建 torch_brain/data/trial_sampler.py（TrialAlignedSampler）
  - 每个 sample = 一个 trial，以 go_cue_time 为对齐点
  - window = [go_cue_time - obs_window, go_cue_time + pred_window]
  - 支持 shuffle（训练）/ sequential（评估）
- [x] 在 dataset.py 中添加 get_trial_intervals() 方法
- [x] 在 train.py 中添加 trial_aligned config 参数，控制 sampler 选择
- [x] 传递 trial metadata（target_id、go_cue_time）到 batch

### 1.2 基础功能验证

#### 1.2.1-1.2.2 [x] v1 验证（已完成）
> 依赖：1.1 产出代码
> 产出：`results/logs/phase1_small_250ms/ar_verify_results.json`, `scripts/analysis/neurohorizon/ar_verify.py`
> 记录：`cc_todo/phase1-autoregressive/20260302-phase1-1.2-foundation-verify.md`

- [x] Teacher forcing 训练收敛（R²=0.2658）
- [x] AR vs TF 一致性验证（max_diff=3e-6）

#### 1.2.3 [ ] v2 AR 修复验证
> 依赖：`cc_core_files/proposal_review.md` §2.5b + §2.11, 1.1.8 产出代码
> 注：本条主要适用于引入显式 feedback / prediction-memory 的 1.9+ 变体。对 v2 baseline（`decoder_variant=query_aug`, `feedback_method=none`），forward 与 rollout 数学等价，因此“TF vs AR 输出差异”不作为该 baseline 的有效验收项。
- [ ] 验证修复后 TF 和 AR 推理输出不再相同（diff >> 1e-6）
- [ ] 验证修改 bin t 的预测确实影响 bin t+1 及之后的输出
- [ ] TF 模式下训练收敛验证（loss 下降，无 NaN/Inf）
- [ ] AR 推理的 fp-bps 评估

#### 1.2.4 [x] v2 指标验证
> 依赖：`cc_core_files/proposal_review.md` §2.10（指标） + §2.1b（sampler），1.1.7 + 1.1.9 产出代码
> 产出：`scripts/tests/test_1_2_4_metrics_verification.py`，验证结果 JSON
> 记录：`cc_todo/phase1-autoregressive/20260311-phase1-1.1.7-1.1.9-implementation.md`

**实验目的**：验证新实现的 fp-bps 指标和 trial-aligned sampler 的正确性，确保后续实验的评估基础可靠。

**实验方法**：
- 合成数据单元测试（9 项）：global per-neuron mean null 的 fp-bps=0、随机预测 fp-bps<0、oracle fp-bps>0、NLB 交叉验证 diff<1e-5
- Trial-aligned sampler 功能验证：go_cue_time 对齐、target_id 正确传递、obs/pred 窗口边界正确

**实验结果**（合成数据，非真实模型评估）：
- fp-bps: null=0.0, random=-0.37, oracle=0.068（NLB 量级）
- NLB 交叉验证 diff=2.8e-7
- Trial-aligned sampler: 9/9 项通过

**局限性**：
- 本验证使用合成数据，仅确认指标计算逻辑正确性
- 当前仅完成 trial-aligned sampler 的功能正确性验证，尚未完成 domain / trial boundary audit；`go_cue - obs_window` 与 `go_cue + pred_window` 的越界审计需后续单列执行
- 实际模型训练中的 fp-bps 数值需在 1.3.4 中获得

- [x] fp-bps 正确性：global per-neuron mean null 的 fp-bps = 0.0，随机预测 < 0，训练好模型 > 0
- [x] Trial-aligned sampler 正确性：每个 sample 以 go_cue_time 对齐，target_id 正确

#### 1.2.5 [x] baseline_v2 评估协议复核与完成项补记
> 依赖：`cc_todo/20260316-review/20260316-plan-md-v2-code-review_codex.md`，`cc_todo/20260316-review/QA_codex.md` §1
> 产出：`cc_core_files/plan.md`（1.2 / 1.3 口径收口）
> 记录：`cc_todo/phase1-autoregressive/20260318-phase1-1.2.5-1.3.7-plan-standard.md`

**补记目的**：将 baseline_v2 审查中已经确认、且已在代码与 `1.3.4 evalfix` 中落地的评估协议修正补记到计划正文，避免后续继续按旧口径引用 v2 结果。

**已完成事项**：
- continuous validation / test 已统一改用 `SequentialFixedWindowSampler`，不再沿用训练式随机 fixed-window eval
- continuous 主 `fp-bps` / `R-squared` 已统一改为全局累计版，正式结果不再使用 batch mean 口径
- 主 `fp-bps` 明确固定为 `global spike-weighted + train-split null`，不再与 per-neuron mean / eval-split null 混写
- v2 baseline 当前主表解释明确限定为“当前实现语义下的 forward prediction baseline”；`TF ≡ rollout` 只对 `query_aug + feedback=none` 成立

**当前未纳入完成项**：
- trial boundary / split 边界审计仍未完成，不在本条打勾范围内
- 训练期 logger 仍只记录前 12 个 bin；完整 horizon 正式结果仍以离线 `eval_phase1_v2.py` 为准

- [x] continuous eval sampler 确定性修正已落地
- [x] continuous 主指标全局累计修正已落地
- [x] baseline_v2 主结果解释边界已写回计划正文

### 1.3 预测窗口实验

#### 1.3.1-1.3.3 [x] v1 实验（已完成，R² 为主指标）
> 依赖：1.1 产出 + configs
> 产出：`results/logs/phase1_small_{250,500,1000}ms/`
> 记录：`cc_todo/phase1-autoregressive/20260302-phase1-1.3-prediction-window.md`
- [x] 250ms：R²=0.2658
- [x] 500ms：R²=0.2417（-9.1%）
- [x] 1000ms：R²=0.2343（AR）/ R²=0.2354（non-AR，差异<0.002）

#### 1.3.4 [x] v2 实验（fp-bps 为主指标 + trial-aligned 模式）
> 依赖：`cc_core_files/proposal_review.md` §2.11（窗口实验） + §2.10（指标） + §2.1b（trial-aligned）
> 产出：`results/logs/phase1_v2_{250ms,500ms,1000ms}_{cont,trial}/`，`results/logs/phase1_v2_evalfix_{250ms,500ms,1000ms}_{cont,trial}/`，`results/figures/phase1_v2/`，`results/logs/phase1_v2_*/eval_v2_{valid,test}_results.json`
> 记录：`cc_todo/phase1-autoregressive/20260311-phase1-1.3.4-v2-experiment.md`
> 评估协议修正记录：`cc_todo/phase1-autoregressive/20260317-phase1-1.3.4-evalfix-rerun.md`

**实验目的**：
1. 用 fp-bps 替代 R-squared 作为主要评估指标，重新进行预测窗口实验
2. 对比连续训练 vs trial-aligned 训练两种数据组织方式的效果
3. 计算 `per_neuron_psth_r2` 评估神经元粒度的 trial-averaged 预测质量
4. 分析 per-bin fp-bps 衰减趋势，量化自回归预测的有效时程

**实验配置（6 个训练运行）**：

| 条件 | 训练模式 | pred_window | obs_window | bins | batch_size | log_dir | config |
|------|---------|-------------|------------|------|------------|---------|--------|
| 250ms-cont | 连续 | 250ms | 500ms | 12 | 64 | `phase1_v2_250ms_cont` | train_v2_250ms.yaml |
| 250ms-trial | trial-aligned | 250ms | 500ms | 12 | 64 | `phase1_v2_250ms_trial` | train_v2_250ms_trial.yaml |
| 500ms-cont | 连续 | 500ms | 500ms | 25 | 64 | `phase1_v2_500ms_cont` | train_v2_500ms.yaml |
| 500ms-trial | trial-aligned | 500ms | 500ms | 25 | 64 | `phase1_v2_500ms_trial` | train_v2_500ms_trial.yaml |
| 1000ms-cont | 连续 | 1000ms | 500ms | 50 | 32 | `phase1_v2_1000ms_cont` | train_v2_1000ms.yaml |
| 1000ms-trial | trial-aligned | 1000ms | 500ms | 50 | 32 | `phase1_v2_1000ms_trial` | train_v2_1000ms_trial.yaml |

**共用模型配置**：neurohorizon_small（dim=128, enc_depth=6, dec_depth=2），300 epochs，bf16-mixed，seed=42

**评估指标**：fp-bps（整体 + per-bin）、R-squared、`per_neuron_psth_r2`（trial-aligned eval，8 方向）

**评估协议修正（2026-03-17 起）**：
- continuous validation / test 统一改用 `SequentialFixedWindowSampler`，不再使用随机 fixed-window valid sampler
- continuous `fp-bps` / `R-squared` 统一使用全局累计版（累计 `nll_null` / `nll_model` / `spikes` 和 `ss_res` / `ss_tot`）
- trial-aligned 主 PSTH 指标统一为 `per_neuron_psth_r2`；旧 `PSTH-R-squared` 结果为 legacy population-mean 口径，仅用于历史对照
- 训练期 logger 仍仅记录前 12 个 bin 的 per-bin 指标；完整 horizon 以离线 `eval_v2_{valid,test}_results.json` 为准
- v2 baseline（`query_aug + feedback=none`）满足 `TF ≡ rollout`；该说明不自动外推到 1.9+ 显式 feedback 变体

**v1 参考值（仅 R-squared，无 fp-bps）**：250ms R-squared=0.2658, 500ms R-squared=0.2417, 1000ms R-squared=0.2343

**注意：不复用 v1 checkpoint，全部重新训练**（v1 无 fp-bps 集成，代码已变更 prediction feedback 等）

**可视化（5 张图）**：
1. fp-bps vs pred_window：连续 vs trial-aligned 两条线
2. per-bin fp-bps 衰减曲线：6 条线（3 窗口 x 2 模式）
3. `per_neuron_psth_r2` 热力图：target_id(0-7) x 条件(6 个)
4. 连续 vs trial-aligned 对比柱状图（fp-bps + R-squared 双指标）
5. 训练曲线：6 个模型的 val_loss / val_fp_bps vs epoch

- [x] 250ms：连续 + trial-aligned，fp-bps / R-squared / `per_neuron_psth_r2`
- [x] 500ms：连续 + trial-aligned，fp-bps / R-squared / `per_neuron_psth_r2`
- [x] 1000ms：连续 + trial-aligned，fp-bps / R-squared / `per_neuron_psth_r2`
- [x] 分析：fp-bps 随预测窗口的衰减曲线（per-bin fp-bps）
- [x] 分析：连续 vs trial-aligned 训练效果差异
- [x] 分析：`per_neuron_psth_r2` 在不同预测窗口下的表现

**Legacy internal reference（protocol-fix legacy simplified baselines，非正式 benchmark）**：

> 说明：
> - 下表仅保留为项目内 legacy simplified baseline 的 protocol-fix held-out internal reference
> - 它们不是对原始 NDT2 / IBL-MtM / Neuroformer 的正式 faithful benchmark
> - 当前正式 benchmark 状态以 1.8.3 的 faithful 250ms gate 为准

| 模型 | 250ms test fp-bps | 500ms test fp-bps | 1000ms test fp-bps | 备注 |
|------|-------------------|-------------------|--------------------|------|
| **NeuroHorizon evalfix** | **0.2223** | **0.1740** | **0.1348** | phase1_v2_evalfix continuous held-out test |
| Legacy Neuroformer-like | 0.1968 | 0.1579 | 0.1004 | protocol-fix held-out test |
| Legacy IBL-MtM-like | 0.1859 | 0.1505 | 0.0869 | protocol-fix held-out test |
| Legacy NDT2-like | 0.1791 | 0.1397 | 0.0989 | protocol-fix held-out test |

**当前可保留的内部结论**：
1. 在 protocol-fix legacy internal reference 上，NeuroHorizon 的 held-out continuous fp-bps 仍然更高。
2. 这只能说明 NeuroHorizon 优于项目内简化 baseline，不构成对原始 benchmark 模型的正式公平比较。
3. 当前 1.8 正式 benchmark 叙事必须回到 faithful 250ms gate：NDT2 只保留现状，IBL-MtM 继续 short formal run，Neuroformer 先解 runtime blocker。

- [x] Legacy internal reference 整理（非正式 benchmark）

#### 1.3.5 [x] IBL-MtM 风格 bps 对照指标（`ibl_mtm_bps`）
> 依赖：`cc_todo/20260316-review/QA_codex.md` §1.8（尤其 §3.3）
> 产出：更新后的 `eval_v2_{valid,test}_results.json`（新增 `continuous.ibl_mtm_bps` 字段），`results/logs/phase1_v2_metric_extension_comparison/`
> 记录：`cc_todo/phase1-autoregressive/20260318-phase1-1.3.5-ibl-metric.md`

**实验目的**：
1. 在现有主 `fp-bps` 不变的前提下，补一个更接近 IBL-MtM paper 口径的对照指标
2. 明确区分：
   - 主指标：global spike-weighted `fp-bps` + train-split null
   - 对照指标：per-neuron mean `ibl_mtm_bps` + eval-split null
3. 为 1.8 benchmark 对比提供第二条更接近原始 IBL-MtM 表格的解释坐标轴

**指标定义**：
- `ibl_mtm_bps`
  - 先在当前 evaluation split 上对每个 neuron 单独累计 model NLL
  - null rate 使用当前 evaluation split 的该 neuron 平均 spike count / bin
  - 对有效 neuron 的 bps 做简单平均
- **注意**：该指标仅作为 comparison metric，不替代 1.3.4/1.8 中的主 `fp-bps`

**实验范围**：
- 仅补评估 1.3.4 evalfix 的 continuous 三个窗口：
  - `phase1_v2_evalfix_250ms_cont`
  - `phase1_v2_evalfix_500ms_cont`
  - `phase1_v2_evalfix_1000ms_cont`
- 不重新训练 causal baseline

**评估指标**：
- `fp-bps`
- `ibl_mtm_bps`
- `R-squared`

**结果概览**：

| 条件 | valid fp-bps | valid ibl_mtm_bps | test fp-bps | test ibl_mtm_bps |
|------|--------------|-------------------|-------------|------------------|
| 250ms-cont | 0.2164 | 0.2234 | 0.2223 | 0.2321 |
| 500ms-cont | 0.1823 | 0.1609 | 0.1740 | 0.1555 |
| 1000ms-cont | 0.1374 | 0.0381 | 0.1348 | 0.0532 |

**结论**：
1. 两种指标对 `250ms > 500ms > 1000ms` 的窗口排序保持一致
2. `ibl_mtm_bps` 在 `1000ms` 上下降更剧烈，说明 per-neuron mean + eval-split null 口径对低发放率 / 难预测 neuron 更敏感
3. 因此 `ibl_mtm_bps` 可作为与 IBL-MtM paper 更接近的 comparison metric，但不应替代主 `fp-bps`

- [x] 250ms-cont：valid/test `fp-bps` + `ibl_mtm_bps`
- [x] 500ms-cont：valid/test `fp-bps` + `ibl_mtm_bps`
- [x] 1000ms-cont：valid/test `fp-bps` + `ibl_mtm_bps`
- [x] 汇总：`results/logs/phase1_v2_metric_extension_comparison/comparison.{json,md}`
- [x] 分析：两种 bps 口径是否对窗口排序和相对衰减给出一致结论

#### 1.3.6 [x] baseline_v2 non-causal ablation（双向 decoder）
> 依赖：`cc_todo/20260316-review/neurips_innovation_claude.md` §2.2
> 产出：`results/logs/phase1_v2_nocausal_{250ms,500ms,1000ms}_cont/`，`results/logs/phase1_v2_nocausal_comparison/`
> 记录：`cc_todo/phase1-autoregressive/20260318-phase1-1.3.6-nocausal-ablation.md`

**实验目的**：
1. 在 baseline_v2 训练方式下，仅替换 decoder self-attention 的因果掩码
2. 测试 causal constraint 是否真的为 future-bin prediction 带来有效的时序归纳偏置
3. 用两套 bps 口径同时比较 causal vs non-causal：
   - 主 `fp-bps`
   - 对照 `ibl_mtm_bps`

**实验配置（3 个训练运行）**：

| 条件 | 训练模式 | pred_window | obs_window | decoder attention | batch_size | log_dir | config |
|------|---------|-------------|------------|-------------------|------------|---------|--------|
| 250ms-cont | 连续 | 250ms | 500ms | **双向** | 64 | `phase1_v2_nocausal_250ms_cont` | `train_v2_nocausal_250ms.yaml` |
| 500ms-cont | 连续 | 500ms | 500ms | **双向** | 64 | `phase1_v2_nocausal_500ms_cont` | `train_v2_nocausal_500ms.yaml` |
| 1000ms-cont | 连续 | 1000ms | 500ms | **双向** | 32 | `phase1_v2_nocausal_1000ms_cont` | `train_v2_nocausal_1000ms.yaml` |

**共用配置**：其余超参全部与 1.3.4 evalfix continuous baseline 对齐；唯一核心变量为 `causal_decoder: false`

**评估指标**：
- `fp-bps`
- `ibl_mtm_bps`
- `R-squared`
- `per_neuron_psth_r2`（补充）

**结果概览**：

| 窗口 | causal valid fp-bps | non-causal valid fp-bps | causal valid ibl_mtm_bps | non-causal valid ibl_mtm_bps |
|------|---------------------|-------------------------|--------------------------|------------------------------|
| 250ms | 0.2164 | 0.2124 | 0.2234 | 0.1958 |
| 500ms | 0.1823 | 0.1817 | 0.1609 | 0.1577 |
| 1000ms | 0.1374 | 0.1459 | 0.0381 | 0.1040 |

| 窗口 | causal test fp-bps | non-causal test fp-bps | causal test ibl_mtm_bps | non-causal test ibl_mtm_bps |
|------|--------------------|------------------------|-------------------------|-----------------------------|
| 250ms | 0.2223 | 0.2178 | 0.2321 | 0.1390 |
| 500ms | 0.1740 | 0.1778 | 0.1555 | 0.1585 |
| 1000ms | 0.1348 | 0.1375 | 0.0532 | 0.0660 |

**结论**：
1. `250ms` 下 causal 略优，尤其在 `ibl_mtm_bps` 上优势明显
2. `500ms` 下两者几乎持平，non-causal 在 test 上有非常小的正差值
3. `1000ms` 下 non-causal 在两套 bps 口径上都超过 causal，说明 causal constraint 不是 baseline_v2 长窗口收益的充分必要条件

- [x] 250ms-cont：non-causal 训练 + valid/test 评估
- [x] 500ms-cont：non-causal 训练 + valid/test 评估
- [x] 1000ms-cont：non-causal 训练 + valid/test 评估
- [x] 汇总：`results/logs/phase1_v2_nocausal_comparison/comparison.{json,md}`
- [x] 分析：causal vs non-causal 在两个 bps 口径上的差值

#### 1.3.7 [x] NeuroHorizon 实验默认数据与指标标准
> 依赖：`cc_todo/20260316-review/20260316-plan-md-v2-code-review_codex.md`，`cc_todo/20260316-review/QA_codex.md` §1，1.3.5，1.3.6
> 产出：`cc_core_files/plan.md`（NeuroHorizon 实验默认协议）
> 记录：`cc_todo/phase1-autoregressive/20260318-phase1-1.2.5-1.3.7-plan-standard.md`

**适用范围**：
- 作为后续 NeuroHorizon 主线实验的默认数据处理、sampler 与指标协议
- `1.8.x` benchmark 对比在引用 NeuroHorizon 结果时也默认使用本节口径；外部 faithful baseline 若保留原生协议，必须在对应文档中显式标注差异

**默认数据与 dataloader 协议**：
- 数据划分默认沿用 dataset config 的 `train / valid / test`
- continuous 训练默认使用 `RandomFixedWindowSampler`
- continuous valid/test 默认使用 `SequentialFixedWindowSampler`
- 默认实验协议使用 continuous；trial-aligned 仅在需要做 hold/reach 对齐分析或 trial-averaged 对照时按需启用，不作为默认设置
- 代码位置：`examples/neurohorizon/train.py`；离线正式评估入口：`scripts/analysis/neurohorizon/eval_phase1_v2.py`

**默认指标与聚合协议**：
- 主指标：`global spike-weighted fp-bps + train-split null`
- 默认正式结果只使用主 `fp-bps`；`R-squared`、`ibl_mtm_bps`、`per_neuron_psth_r2` 均为按需启用的可选补充指标，默认不写入主结论
- `ibl_mtm_bps` 仅在需要做 IBL-MtM 风格 comparison 时启用；`per_neuron_psth_r2` 仅在 trial-aligned 分析中启用
- `per-bin fp-bps` 可在比较 long-horizon prediction decay 时启用：
  - 通过 `fp_bps_per_bin_stats()` 先按 bin 分别累计 `nll_model / nll_null / spikes`
  - 再用 `finalize_fp_bps_per_bin_from_stats()` 得到严格累计版 per-bin 曲线
  - 训练期 dashboard 的 `val/fp_bps_bin{t}` 是 batch 级日志；正式衰减分析默认引用 `eval_phase1_v2.py` 的离线累计结果
  - 简要实现说明参考 `cc_todo/20260316-review/QA_codex.md` Q1 的 `1.3 time` 与 `1.4 batch`
- 代码位置：`torch_brain/utils/neurohorizon_metrics.py` 与 `scripts/analysis/neurohorizon/eval_phase1_v2.py`

**报告与解释规则**：
- continuous 正式表格与曲线默认引用离线 `eval_v2_{valid,test}_results.json`
- 非主指标若启用，必须在对应结果或文档中显式注明其用途是 comparison / diagnosis，而不是替代主 `fp-bps`
- 对 baseline_v2（`query_aug + feedback=none`）可保留 `TF ≡ rollout` 说明；该说明不外推到 `1.9+` 显式 feedback 变体，也不自动外推到 `1.8.x` faithful runner

**与现有任务的一致性**：
- `1.3.5` 已按本标准执行：continuous evalfix 继续使用主 `fp-bps`，仅额外补充 `ibl_mtm_bps` comparison metric
- `1.3.6` 已按本标准执行：训练/评估协议与 `1.3.4 evalfix` / `1.3.5` 保持一致，唯一核心变量为 `causal_decoder: false`
- 因此后续 NeuroHorizon 结果若未特别声明，默认引用本节标准；与 external benchmark 协议不一致处必须单列说明

- [x] 数据划分、dataloader 与 sampler 默认协议已固定
- [x] 主指标与可选补充指标的默认使用边界已固定
- [x] `1.3.5` 与 `1.3.6` 是否符合该标准已明确写清

### 1.4 [x] 观察窗口长度实验
> 依赖：1.3.4 完成，确定最优 pred_window（预期 250ms）
> 产出：`results/logs/phase1_v2_obs{250,500,750,1000}ms/`，`results/figures/phase1_v2/`
> 记录：`cc_todo/phase1-autoregressive/{date}-phase1-1.4-obs-window.md`

**实验目的**：固定 pred_window=250ms，调节 obs_window（历史观察窗口），研究历史信息量对预测质量的影响，确定最优 obs_window 及饱和点。

**实验配置**：

| 条件 | obs_window | pred_window | sequence_length | bins | 备注 |
|------|-----------|-------------|-----------------|------|------|
| obs250 | 250ms | 250ms | 500ms | 12 | 仅覆盖 hold 末段 |
| obs500 | 500ms | 250ms | 750ms | 12 | = 1.3.4 baseline（复用） |
| obs750 | 750ms | 250ms | 1000ms | 12 | 延伸到 hold 前段 |
| obs1000 | 1000ms | 250ms | 1250ms | 12 | 可能跨越前一 trial |

**复用说明**：obs500 直接复用 1.3.4 的 250ms-cont 模型，无需重新训练（仅需 3 个新训练）。

**评估指标**：fp-bps（整体 + per-bin）、R-squared、PSTH-R-squared

**分析重点**：
- fp-bps vs obs_window 曲线：是否存在饱和点（边际增益递减）
- trial-aligned 下 250ms obs 仅覆盖 hold period 末段 vs 500ms 包含更多 hold 信息
- obs_window 过长（1000ms）是否引入跨 trial 边界的噪声

- [x] 250ms obs + 250ms pred（连续 + trial-aligned）
- [x] 500ms obs + 250ms pred（= 1.3.4 baseline，复用）
- [x] 750ms obs + 250ms pred
- [x] 1000ms obs + 250ms pred
- [x] 分析：fp-bps vs obs_window 曲线，是否存在饱和点
- [x] 分析：trial-aligned 下 250ms obs 仅覆盖 hold period vs 500ms 延伸到前一 trial
- [x] 配置说明：sequence_length = obs_window + pred_window，pred_window 固定

**Benchmark 对比说明**：
- 各 obs_window 条件下，同时训练 NDT2 / Neuroformer / IBL-MtM（与 NeuroHorizon 相同 obs_window + pred_window=250ms）
- obs500 复用 1.8.3 已有 benchmark 结果（条件一致）
- 比较维度：fp-bps / R² vs obs_window，4 个模型在同一图中

- [x] Benchmark 训练：obs250 × 3 模型
- [x] Benchmark 训练：obs750 × 3 模型
- [x] Benchmark 训练：obs1000 × 3 模型
- [x] Benchmark 对比可视化：fp-bps vs obs_window（4 模型曲线）

### 1.5 [x] Session 数目实验
> 依赖：1.3.4 完成（10-session 结果作为 baseline），dataset configs 已存在
> 产出：`results/logs/phase1_v2_{1,4,7}sessions/`，`examples/neurohorizon/configs/dataset/perich_miller_{1,4,7}sessions.yaml`（已创建）
> 记录：`cc_todo/phase1-autoregressive/{date}-phase1-1.5-session-scaling.md`

**实验目的**：研究多 session 联合训练对 spike 预测质量的影响，分析跨受试体泛化增益或干扰，为 Phase 2 跨 session 实验提供基线参考。

**实验配置**：

| 条件 | sessions 数 | 动物 | sessions 列表 | 备注 |
|------|------------|------|-------------|------|
| 1-session | 1 | C | c_20131003 | 单 session baseline |
| 4-sessions | 4 | C | c_20131003/1022/1101/1204 | 同动物多 session |
| 7-sessions | 7 | C+J | C 全部 + J 全部 | 跨动物 |
| 10-sessions | 10 | C+J+M | 全部 | = 1.3.4 baseline（复用） |

**共用配置**：250ms pred_window + 500ms obs_window（train_small 配置），连续训练模式

**复用说明**：10-session 直接复用 1.3.4 的 250ms-cont 模型。

**评估指标**：fp-bps（整体 + per-session）、R-squared、PSTH-R-squared

**分析重点**：
- 多 session 联合训练是否提升单 session 预测质量（正迁移 vs 干扰）
- 跨受试体（C->J->M）泛化：用 C 训练的模型在 J/M sessions 上的 fp-bps
- per-session fp-bps 分布变化：联合训练后各 session 性能是否均匀提升

- [x] 1 session（仅 c_20131003）
- [x] 4 sessions（C 动物：c_20131003/1022/1101/1204）
- [x] 7 sessions（C + J 动物）
- [x] 10 sessions（全部，= 1.3.4 baseline，复用）
- [x] 分析：多 session 联合训练对单 session 预测质量的影响
- [x] 分析：跨受试体（C->J->M）的泛化增益或干扰
- [x] 报告：per-session fp-bps
- [x] 新增配置：perich_miller_{1,4,7}sessions.yaml

**Benchmark 对比说明**：
- 各 session 数目条件下，同时训练 NDT2 / Neuroformer / IBL-MtM（与 NeuroHorizon 相同 session 配置 + pred_window=250ms）
- 10-session 复用 1.8.3 已有 benchmark 结果（条件一致）
- 比较维度：fp-bps / R² vs session_count，4 个模型在同一图中

- [x] Benchmark 训练：1-session × 3 模型
- [x] Benchmark 训练：4-sessions × 3 模型
- [x] Benchmark 训练：7-sessions × 3 模型
- [x] Benchmark 对比可视化：fp-bps vs session_count（4 模型曲线）

### 1.6 [待定] Forward Prediction → 行为解码假设

> 状态：待 Phase 1 核心实验完成后探索
>
> **假设**：fp-bps 越高的模型，冻结 encoder 后接行为解码 head 的 R² 也越高。
>
> **实验设计**（Phase 3.2 实施）：
> 1. 取 1.3/1.4/1.5 中不同配置的 NeuroHorizon 模型
> 2. 冻结 encoder，接 linear probe 做行为解码（cursor velocity R²）
> 3. 绘制 behavior R² vs fp-bps 散点图
> 4. 正相关 = 支持假设的有力证据
>
> **关联**：NDT-MtM 的 multi-task masking 提升 fp-bps（0.50→0.54）已间接支持此假设

### 1.7 [可选] Scheduled Sampling

> 状态：可选优化，AR 修复验证有效后再考虑
>
> 原理：训练中以概率 p（退火 0→0.5）用模型预测替代 GT feedback
> 注意：引入后失去训练并行性

### 1.8 Benchmark 对比实验：多模型长时程预测 Baseline
> 依赖：1.3.4 完成（NeuroHorizon baseline 结果），`cc_todo/phase0-env-baseline/20260309-phase0-0.4-benchmark-analysis.md` §0.4.2–§0.4.3
> 产出：`neural_benchmark/`（目录）、`neural_benchmark/benchmark_model.md`、各模型适配代码与实验结果
> 记录：`cc_todo/1.8-benchmark_model/benchmark_index.md`

**实验目的**：
构建 1.8 的 benchmark 工作流，但要明确区分两件事：

1. legacy simplified baselines 的内部参考价值
2. original NDT2 / IBL-MtM / Neuroformer 的 faithful reproduction 是否真的成立

当前正式目标不再是直接宣称公平 benchmark 已完成，而是：

- 统一数据源、split、continuous held-out eval 和主指标
- 尽量保留上游模型的输入契约、训练目标和推理程序
- 用 250ms gate 先判断 faithful runner 在当前 Perich-Miller forward prediction setting 下到底是 objective mismatch、metadata mismatch、runtime mismatch，还是模型本身确实不适合

**核心标准**：
1. **统一 split 与连续评估语义**：所有模型共用 torch_brain 的 train / valid / test 来源；正式 continuous valid/test 必须对齐 deterministic canonical coverage。faithful runner 若不直接调用 SequentialFixedWindowSampler，也必须在语义上等价，并显式写明差异
2. **统一主指标**：benchmark 主结果默认只认 global spike-weighted fp-bps + train-split raw-event null 的 held-out test；ibl_mtm_bps 仅作 comparison metric，per_neuron_psth_r2 仅作 trial-aligned 补充分析
3. **训练语义尽量保留上游**：各模型的 architecture / training loop / inference code 应尽量保持原样；允许只在数据桥接、canonical windows 与统一评估接口上做兼容层，不再把项目内重写 wrapper 称作正式 benchmark

**优先实现的对比模型**：

| 模型 | 仓库 | 论文 | 架构类型 | FP 方式 | 适配难度 |
|------|------|------|---------|---------|---------|
| **NDT2** | [joel99/context_general_bci](https://github.com/joel99/context_general_bci) | NeurIPS 2023 | Encoder-Decoder, MAE | 并行填充（mask future bins → 重建） | 低 |
| **Neuroformer** | [a-antoniades/Neuroformer](https://github.com/a-antoniades/Neuroformer) | ICLR 2024 | Decoder-only, GPT-style | 逐 spike 自回归生成 | 中 |
| **IBL-MtM** | [colehurwitz/IBL_MtM_model](https://github.com/colehurwitz/IBL_MtM_model) | NeurIPS 2024 | Encoder-only, Multi-task Masking | temporal causal masking（并行预测） | 中 |

**备选模型**（后续视需要扩展）：NEDS、STNDT、NDT3

**数据集**：Perich-Miller 2018（与 1.3.4 完全相同的 10 sessions，相同 train/valid/test 划分）

**参照实验**：1.3.4 v2 实验（6 个条件：3 窗口 × 2 模式，fp-bps/R²/PSTH-R² 指标）

#### 1.8.1 [x] 模型调研与文档整理 <!-- 记录：cc_todo/1.8-benchmark_model/20260312_benchmark_main_task_log.md -->

> 依赖：`cc_todo/phase0-env-baseline/20260309-phase0-0.4-benchmark-analysis.md` §0.4.2–§0.4.3
> 产出：`neural_benchmark/benchmark_model.md`
> 记录：同 1.8 主记录

**任务**：
1. 创建 `neural_benchmark/` 目录
2. 深入调研 NDT2、Neuroformer、IBL-MtM 三个模型，整理以下信息到 `neural_benchmark/benchmark_model.md`：

**每个模型必须包含的信息**：
- GitHub 链接
- 模型大小（参数量）
- 资源需求（GPU 显存、训练时间）
- 是否提供 inference code 和预训练 weight（可直接测试）
- 是否支持 forward prediction（怎么实现的）
- 一般支持的 forward prediction 时长
- 观察窗口（observation window）和预测窗口（prediction window）时长
- Paper 或代码中是否提到过 fp-bps 的数值参考
- **如何适配到 torch_brain 框架**：数据适配方案（spike events → 模型输入格式的转换逻辑）和 metric 评估方案（模型输出 → 统一评估接口的对接方式）

**备选模型备注**：在 benchmark_model.md 末尾添加备注节，列出 NEDS、STNDT、NDT3 等备选模型的基本信息和 GitHub 链接，标注为"待评估"

**可参考内容**：`cc_todo/phase0-env-baseline/20260309-phase0-0.4-benchmark-analysis.md` §0.4.2（模型分析）+ §0.4.3（适配方案）

- [ ] 创建 neural_benchmark/ 目录结构
- [ ] 调研 NDT2 并整理到 benchmark_model.md
- [ ] 调研 Neuroformer 并整理到 benchmark_model.md
- [ ] 调研 IBL-MtM 并整理到 benchmark_model.md
- [ ] 添加备选模型备注（NEDS, STNDT, NDT3）

#### 1.8.2 [x] 环境配置与模型部署 <!-- 记录：cc_todo/1.8-benchmark_model/20260312_benchmark_main_task_log.md -->

> 依赖：1.8.1 完成
> 产出：`neural_benchmark/benchmark_models/` 下各模型代码、`neural_benchmark/envs/` conda 环境
> 记录：同 1.8 主记录

**任务**：

1. **存储映射**：将 conda 环境存储目录从默认的 `miniconda3/envs` 映射到 `NeuroHorizon/neural_benchmark/envs/`
   ````bash
   mkdir -p /root/autodl-tmp/NeuroHorizon/neural_benchmark/envs
   conda config --prepend envs_dirs /root/autodl-tmp/NeuroHorizon/neural_benchmark/envs
   ````
   使实际存储占用在项目数据盘，但不影响 conda 正常使用（`conda activate benchmark-env` 等）

2. **Clone 模型仓库**到 `neural_benchmark/benchmark_models/`：
   ````bash
   mkdir -p neural_benchmark/benchmark_models
   git clone https://github.com/joel99/context_general_bci.git neural_benchmark/benchmark_models/ndt2
   git clone https://github.com/a-antoniades/Neuroformer.git neural_benchmark/benchmark_models/neuroformer
   git clone https://github.com/colehurwitz/IBL_MtM_model.git neural_benchmark/benchmark_models/ibl-mtm
   ````

3. **创建共用 conda 环境** `benchmark-env`：
   - 优先在同一个 conda 环境中安装所有模型依赖
   - 核心共用依赖：PyTorch (CUDA)、numpy、scipy、h5py、wandb、hydra 等
   - 各模型特有依赖按其 README / requirements.txt / env.yaml 安装
   - **仅当**存在严重不兼容（如 PyTorch 版本冲突、Python 版本不兼容）时，才为该模型单独创建 conda 环境（如 `benchmark-ndt2-env`）

4. **验证安装**：每个模型完成安装后，运行其自带的最小测试或 import 检查，确保环境正常

- [ ] 配置 conda envs_dirs 映射到 neural_benchmark/envs/
- [ ] Clone NDT2 + 安装依赖 + 验证
- [ ] Clone Neuroformer + 安装依赖 + 验证
- [ ] Clone IBL-MtM + 安装依赖 + 验证
- [ ] 记录环境配置结果（成功/失败/兼容性问题）

#### 1.8.3 [ ] 模型适配与对比实验（faithful benchmark 持续维护）

> 依赖：1.8.2 完成（环境就绪），1.3.4 完成（NeuroHorizon baseline 结果），1.3.7（NeuroHorizon 默认数据与指标标准）
> 总入口：`cc_todo/1.8-benchmark_model/benchmark_index.md`
> 主任务记录：`cc_todo/1.8-benchmark_model/20260312_benchmark_main_task_log.md`
> legacy 审计：`cc_todo/1.8-benchmark_model/20260316_benchmark_legacy_audit_codex.md`
> faithful 审计：`cc_todo/1.8-benchmark_model/20260318_benchmark_faithful_audit_detail_codex.md`
> faithful 审计执行记录：`cc_todo/1.8-benchmark_model/20260318_benchmark_faithful_audit_task_log.md`

**定位**：
- `1.8.3` 不再把 benchmark 结果写成一段段状态说明，而是作为 benchmark 持续维护规范的入口
- 所有新的 benchmark 模型实现优化、协议调整、复现实验与结果回填，统一记录到 `cc_todo/1.8-benchmark_model/`
- benchmark 主协议继续遵守 `1.3.7`：统一 `train/valid/test` 来源、continuous held-out eval 语义、主指标 `fp-bps`、补充 `per-bin fp-bps`；trial-aligned 仅作为补充分析

**任务记录规范**：
- 每次新的 benchmark 实现优化任务都固定写入：`cc_todo/1.8-benchmark_model/{date}_{content}.md`
- 每份任务记录至少必须包含：
  - 想法描述、动机与目的、相比现有方案的改动点、模型改进实现方案、涉及改动模块、想法摘要
  - 详细实验配置（数据集、sessions 数、采样方式、obs/pred 窗口）
  - 关键超参数（至少 `epoch / batch_size / lr / weight_decay`）
  - 训练 loss 结果、train 期间最佳 val `fp-bps`、test `fp-bps`、test checkpoint 标识/时间
  - 各条件指标结果（至少 `fp-bps / per-bin fp-bps`）
  - 与 baseline 的对比
  - 若模型支持多 inference 模式，必须同时记录 `rollout` 与 `teacher-forced / true_past`
  - 每次训练和评估的脚本命令
  - 可视化索引（训练曲线、配置时间轴图、benchmark 对比图）
  - 当前结论、后续安排

**当前固定脚本入口**：
- IBL-MtM faithful 训练入口：`neural-benchmark/faithful_ibl_mtm.py`
- Neuroformer faithful 训练 / 正式评估入口：`neural-benchmark/faithful_neuroformer.py`
- benchmark history 可视化入口：`neural-benchmark/plot_benchmark_history.py`
- 若一轮任务使用批量编排脚本（例如 `neural-benchmark/run_faithful_1p8_aligned.sh`），必须在任务记录中同时写明批量脚本和底层 train/eval 命令；若脚本入口后续变更，需在对应任务记录中单独说明

**实验执行规范**：
- 开始实验前必须明确确认本轮 benchmark 是否遵循 `1.3.7 NeuroHorizon 实验默认数据与指标标准`
- 如果与 `1.3.7` 不一致，必须在任务记录中书面记录差异项、原因和影响范围
- benchmark 默认使用 `train / valid / test` split；`valid` 用于 checkpoint selection 和中期诊断，`test` 只用于训练结束后的正式 held-out 报告
- continuous benchmark 默认采样语义参照 `1.3.7`：
  - continuous 训练默认使用 `RandomFixedWindowSampler`
  - continuous valid/test 默认使用 `SequentialFixedWindowSampler`
  - faithful runner 若不直接调用上述 sampler 类，也必须在任务记录中说明其 train / valid / test 窗口构造在语义上与之等价
- 默认优先跑最小必要验证，再做正式 benchmark 条件；正式条件可按任务内容裁剪，例如 Neuroformer 除 canonical `500/250` 外，可增加 `150/50` reference sanity run
- 训练过程中必须按 eval epoch 持续记录 `train loss` 和 `valid fp-bps` 曲线，并显式保存 `last` checkpoint
- `best` checkpoint 默认以 `max(valid fp-bps)` 为主进行选择；`loss curve` 用作稳定性审计和异常剔除依据。若 `best fp-bps` 对应 checkpoint 出现明显训练崩坏、数值异常或不合理抖动，必须在任务记录中说明最终采用的 checkpoint 及其理由
- 正式结果必须在训练结束后使用 `best` checkpoint 重新计算 continuous `valid` 和 `test` 的 `fp-bps` 等指标，不允许直接用训练过程中某个 epoch 的即时验证值代替正式结果
- benchmark 主指标为 continuous `fp-bps`，`per-bin fp-bps` 用于窗口衰减分析；其他 comparison metric 按需要补充
- IBL-MtM 和 Neuroformer 的 benchmark 主流程默认不要求 `test trial-aligned`；如后续确需保留，只能作为附加分析，不能替代 continuous held-out 主结果
- 若模型支持多 inference 模式，正式 eval 必须在同一任务记录中并排报告 `rollout` 与 `teacher-forced / true_past`
- Neuroformer 默认按 `valid rollout fp-bps` 选择 `best_model.pt`
  - 在当前 benchmark 目标是 held-out forward prediction 的前提下，`rollout fp-bps` 更接近正式测试语义，因此默认用作 model selection；`teacher-forced / true_past` 仅作为 oracle-history 诊断指标

**实验记录规范**：
- 任务记录：`cc_todo/1.8-benchmark_model/{date}_{content}.md`
- 脚本：`scripts/phase1-autoregressive-1.8-benchmark_model/{date}_{content}/`
- 日志：`results/logs/phase1-autoregressive-1.8-benchmark_model/{date}_{content}/`
- 可视化：`results/figures/phase1-autoregressive-1.8-benchmark_model/{date}_{content}/`
  - 除 training curves 外，还必须补充：
    - 随预测窗口长度变化的 `fp-bps` 趋势图
    - 每个预测窗口的 `per-bin fp-bps` 衰减曲线
    - 配置时间轴图（至少 `lr / weight_decay / effective_batch_size / warmup_progress`）
    - 一个表格型 PNG，用于汇总每次模型改进在不同预测窗口下的最佳 `val fp-bps`、test `fp-bps`、test checkpoint 标识/时间等核心结果
- legacy / protocol-fix / faithful 的历史结果目录保持原名，仅在新任务记录中引用，不做目录重命名

**当前维护中的 benchmark 任务**：
- `IBL-MtM combined aligned`：见 `cc_todo/1.8-benchmark_model/20260319_benchmark_aligned_runs.md`
- `Neuroformer canonical 500/250`：见 `cc_todo/1.8-benchmark_model/20260319_benchmark_aligned_runs.md`
- `Neuroformer 150/50 reference`：见 `cc_todo/1.8-benchmark_model/20260319_benchmark_aligned_runs.md`
- `NDT2` 当前仅保留现状记录，不进入新的 benchmark 扩展

- [x] benchmark 文档入口统一到 `cc_todo/1.8-benchmark_model/`
- [x] benchmark 任务记录模板固定
- [x] benchmark 实验执行/记录规范与 `1.9.0 Step 3 / Step 4` 对齐
- [ ] 当前 aligned faithful benchmark 任务持续回填
- [ ] 仅在模型通过当前 gate 后，再决定是否扩展新的 benchmark 条件

### 1.9 增量模型优化管理
> 定位：支持不断的模型迭代优化记录和实验的增量验证工作
> 产出：`cc_core_files/model.md`（模型演进文档）、各次优化的代码/实验/记录
> 记录：`cc_todo/phase1-autoregressive/1.9-module-optimization/{date}_{module_name}`
> 效果追踪：`cc_todo/phase1-autoregressive/1.9-module-optimization/results.tsv`

**目标**：
建立标准化的模型改进流程，确保每次优化想法都经过充分讨论分析、规范实施、统一实验验证，并可追踪指标变化趋势。

#### 1.9.0 执行规范

**Step 1 -- 想法记录与讨论**（在提出改进想法时）：
- 在 `cc_core_files/model.md` 中新增一节，标注日期和改进名称
- 记录：想法描述、动机与目的、相比现有方案的改动点、模型改进的实现方案、涉及改动模块
- 进行充分的批判性分析：优缺点、风险、替代方案
- 基于当前仓库代码实现，给出可落地的修改方案和基本功能验证方案
- 标记状态为"提出"

**Step 2 -- 分支创建与代码实施**（在用户确认可实施时）：
- 如果用户未明确指定实施分支，则默认在当前 git 状态基础上创建并切换到分支 `dev/{date}_{module_name}`
- 按 `model.md` 中该节的修改方案进行代码改动
- 完成基本功能验证（代码可跑、无错误）
- 在 `model.md` 中更新状态为"实施中"
- **Step 2 完成后立即执行一次 `git commit` + `git push`**：提交当前轮的代码、配置、文档和最小验证结果，作为实现阶段 checkpoint；即使后续正式实验效果不佳，也必须保留这一版实现记录

**Step 3 -- 实验验证**（按优先级依次进行）：
1. **必做 -- 协议确认 + 预测窗口实验**（实验骨架参照 1.3.4，默认协议参照 1.3.7）：
   - 开始实验前必须明确确认本轮实验是否遵循 `1.3.7 NeuroHorizon 实验默认数据与指标标准`
   - 如果与 `1.3.7` 不一致，必须在任务记录中书面记录差异项、原因和影响范围
   - 数据: `examples/neurohorizon/configs/dataset/perich_miller_10sessions.yaml`
   - 采样方式: 连续滑动窗口（非 trial-aligned）
   - `1.3.7` continuous 默认实现位置:
     - 训练入口：`examples/neurohorizon/train.py`
     - 离线正式评估入口：`scripts/analysis/neurohorizon/eval_phase1_v2.py`
     - continuous train sampler：`RandomFixedWindowSampler`
     - continuous valid/test sampler：`SequentialFixedWindowSampler`
   - 观察窗口: 500ms
   - 预测窗口: 250ms / 500ms / 1000ms（3 个条件）
   - 默认至少记录的指标: `fp-bps` / `per-bin fp-bps`
   - `R-squared` / `PSTH-R-squared` / `Poisson NLL` 作为补充指标按需要记录
   - 当前正式实现要求：每个 eval epoch 保存 checkpoint，train end 按 `max(val/fp_bps)` + `min(val_loss)` 选 best ckpt，显式保存真正 final `last.ckpt`，并对 best ckpt 输出 `valid/test` 指标
2. **可选 -- 观察窗口实验**（参照 1.4，用户确认后执行）
3. **可选 -- Session 数目实验**（参照 1.5，用户确认后执行）

**Step 4 -- 实验记录**（遵循 CLAUDE.md 中"任务执行中"的记录规范）：
- **任务记录**: `cc_todo/phase1-autoregressive/1.9-module-optimization/{date}_{module_name}.md`
  - 必须包含：改进想法摘要、实现方案、涉及改动模块、详细实验配置（数据集、sessions 数、采样方式、obs/pred 窗口）、每次实验的关键超参数（至少包括 `epoch`、`batch_size`、`lr`、`weight_decay`）、训练 loss 结果、train 期间最佳 `val fp-bps`、test `fp-bps`、test 使用的 checkpoint 标识/时间、各条件指标结果（至少 `fp-bps` / `per-bin fp-bps`）、与 baseline 的对比、记录每次训练和评估的脚本命令
- **脚本**: `scripts/phase1-autoregressive-1.9-module-optimization/{date}_{module_name}/`
- **实验日志**: `results/logs/phase1-autoregressive-1.9-module-optimization/{date}_{module_name}/`
- **可视化**: `results/figures/phase1-autoregressive-1.9-module-optimization/{date}_{module_name}/`
  - 除 training curves 外，还必须补充：
    - 随预测窗口长度变化的 `fp-bps` 趋势图
    - 每个预测窗口的 `per-bin fp-bps` 衰减曲线
    - 配置时间轴图（至少 `lr / weight_decay / effective_batch_size / warmup_progress`）
    - 一个表格型 PNG，用于汇总每次模型改进在不同预测窗口下的最佳 `val fp-bps`、test `fp-bps`、test checkpoint 标识/时间等核心结果
- **汇总更新**: `cc_core_files/scripts.md` 和 `cc_core_files/results.md` 按 CLAUDE.md 规范更新
- **TSV 更新**: 将预测窗口实验结果追加到 `cc_todo/phase1-autoregressive/1.9-module-optimization/results.tsv`，至少记录每个窗口的最佳 `val fp-bps`、test `fp-bps`、test checkpoint 标识/时间以及必要备注
- **趋势图更新**: 运行 `plot_optimization_progress.py` 更新优化进度折线图
- **Step 4 完成后再执行一次 `git commit` + `git push`**：提交正式实验结果、汇总图表、结论更新与状态变更，保证每轮优化至少留下“实现 checkpoint”和“结果 checkpoint”两次提交

**Step 5 -- 分支合并**（由用户决定）：
- 效果好 -> merge 到 main，在 `model.md` 中标记状态为"已合并"
- 效果不佳 -> 保留分支供参考，在 `model.md` 中标记状态为"已放弃"并记录原因

#### 1.9.1 [x] 模型版本基线

> 产出：`cc_core_files/model.md`（v1/v2 总结）、`results.tsv`（baseline 数据）

- [x] 整理 v1 架构总结（plan.md 1.1.1-1.1.6, commit `bb9439d`）
- [x] 整理 v2 架构总结（plan.md 1.1.7-1.1.9, commit `e5dea0a`）
- [x] 记录 v2 baseline fp-bps 到 results.tsv
- [x] 记录 benchmark 模型结果到 results.tsv
- [x] 创建 plot_optimization_progress.py 并生成初始趋势图

#### 1.9.2 [ ] 模型优化迭代记录

> 以下按时间顺序记录每次模型优化想法及其路径信息

<!-- 模板（每次新优化时复制并填写）:
##### {date}_{module_name} -- {改进名称}
> 状态: 提出 / 实施中 / 验证中 / 已合并 / 已放弃
> 分支: `dev/{date}_{module_name}`
> 文档: `cc_core_files/model.md` 对应小节
> 任务记录: `cc_todo/phase1-autoregressive/1.9-module-optimization/{date}_{module_name}.md`
> 脚本: `scripts/phase1-autoregressive-1.9-module-optimization/{date}_{module_name}/`
> 日志: `results/logs/phase1-autoregressive-1.9-module-optimization/{date}_{module_name}/`
> 可视化: `results/figures/phase1-autoregressive-1.9-module-optimization/{date}_{module_name}/`
> commit: （实施后填写）
> 结果: 250ms fp-bps= / 500ms fp-bps= / 1000ms fp-bps=
-->

##### 20260312_prediction_memory_decoder -- Structured Prediction Memory Decoder
> 状态: 已放弃
> 分支: `dev/20260312_prediction_memory_decoder`
> 文档: `cc_core_files/model.md` 中“2026-03-12 — Structured Prediction Memory Decoder”
> 任务记录: `cc_todo/phase1-autoregressive/1.9-module-optimization/20260312_prediction_memory_decoder.md`
> 脚本: `scripts/phase1-autoregressive-1.9-module-optimization/20260312_prediction_memory_decoder/`
> 日志: `results/logs/phase1-autoregressive-1.9-module-optimization/20260312_prediction_memory_decoder/`
> 可视化: `results/figures/phase1-autoregressive-1.9-module-optimization/20260312_prediction_memory_decoder/`
> commit: `ebb59fa`
> 结果: 250ms fp-bps=0.1510 / 500ms fp-bps=0.0200 / 1000ms fp-bps=-0.2192

- 核心设计：`event-based POYO encoder + time-bin autoregressive decoder + structured prediction memory`
- 旧 `feedback_method` / Query Augmentation 保留为 baseline / ablation，不再作为主线最终架构
- 本次实现固定 `prediction_memory_k=4`，输出仍为 `future T bins x N units` 的 spike counts
- 训练使用 `shift-right` GT counts 构造 prediction memory；推理使用 `exp(log_rate)` 得到 expected counts 再编码
- 必做实验保持 1.9 统一规范：10 sessions、连续滑动窗口、obs=500ms、pred=250/500/1000ms
- training curves：`results/figures/phase1-autoregressive-1.9-module-optimization/20260312_prediction_memory_decoder/training_curves.png`
- 2026-03-20 best-ckpt teacher-forced valid/test：`250ms 0.2974 / 0.2943`，`500ms 0.2840 / 0.2791`，`1000ms 0.2752 / 0.2753`
- 结论：teacher-forced 指标很高，但 rollout 显著差于 `baseline_v2`，尤其长窗口出现严重误差积累；该方案不作为主线继续推进

##### 20260313_local_prediction_memory -- Local Prediction Memory Decoder
> 状态: 已放弃
> 分支: `dev/20260313_local_prediction_memory`
> 文档: `cc_core_files/model.md` 中“2026-03-13 — Local Prediction Memory Decoder”
> 任务记录: `cc_todo/phase1-autoregressive/1.9-module-optimization/20260313_local_prediction_memory.md`
> 脚本: `scripts/phase1-autoregressive-1.9-module-optimization/20260313_local_prediction_memory/`
> 日志: `results/logs/phase1-autoregressive-1.9-module-optimization/20260313_local_prediction_memory/`
> 可视化: `results/figures/phase1-autoregressive-1.9-module-optimization/20260313_local_prediction_memory/`
> commit: `22faac6`
> 结果: 250ms fp-bps=0.1679 / 500ms fp-bps=0.0316 / 1000ms fp-bps=-0.1749

- 核心设计：保留 structured memory，但 query 只访问紧邻上一步的 local memory block
- 设计动机：针对 20260312 版本全历史 memory 检索带来的 rollout 崩塌，优先收缩 feedback 通路容量
- training curves：`results/figures/phase1-autoregressive-1.9-module-optimization/20260313_local_prediction_memory/training_curves.png`
- 2026-03-20 best-ckpt teacher-forced valid/test：`250ms 0.2833 / 0.2857`，`500ms 0.2838 / 0.2750`，`1000ms 0.2736 / 0.2729`
- 结果结论：相对上一轮有小幅改善，但 rollout 仍显著差于 `baseline_v2`；`500ms/1000ms` 仍在中后段转负，因此本轮也不作为主线继续推进

##### 20260313_prediction_memory_alignment -- Prediction Memory Alignment Training
> 状态: 验证中
> 分支: `dev/20260313_prediction_memory_alignment`
> 文档: `cc_core_files/model.md` 中“2026-03-13 — Prediction Memory Alignment Training”
> 任务记录: `cc_todo/phase1-autoregressive/1.9-module-optimization/20260313_prediction_memory_alignment.md`
> 脚本: `scripts/phase1-autoregressive-1.9-module-optimization/20260313_prediction_memory_alignment/`
> 日志: `results/logs/phase1-autoregressive-1.9-module-optimization/20260313_prediction_memory_alignment/`
> 可视化: `results/figures/phase1-autoregressive-1.9-module-optimization/20260313_prediction_memory_alignment/`
> commit: `64415f4`
> 结果: 250ms fp-bps=0.1904 / 500ms fp-bps=0.1623 / 1000ms fp-bps=0.1120

- 核心设计：在 `local_prediction_memory` 上增加训练期 memory 输入对齐，不再继续改 decoder 结构
- 训练策略：将 `shift-right GT counts` 与 `shift-right predicted expected counts` 做时间步级混合，并对 memory encoder 输入施加 noise / dropout
- 设计动机：针对 `20260313_local_prediction_memory` 暴露出的核心 train / inference mismatch，而不是继续只改 memory mask
- 必做实验保持 1.9 统一规范：10 sessions、连续滑动窗口、obs=500ms、pred=250/500/1000ms
- training curves：`results/figures/phase1-autoregressive-1.9-module-optimization/20260313_prediction_memory_alignment/training_curves.png`
- 2026-03-20 best-ckpt teacher-forced valid/test：`250ms 0.2690 / 0.2747`，`500ms 0.2799 / 0.2729`，`1000ms 0.2762 / 0.2773`
- 当前结果：相对 `20260313_local_prediction_memory` 大幅改善，并在三个窗口上都逼近 `baseline_v2`；是否继续迭代或转主线由用户决定

##### 20260313_prediction_memory_alignment_tuning -- Prediction Memory Alignment Tuning
> 状态: 验证中
> 分支: `dev/20260313_prediction_memory_alignment_tuning`
> 文档: `cc_core_files/model.md` 中“2026-03-13 — Prediction Memory Alignment Tuning”
> 任务记录: `cc_todo/phase1-autoregressive/1.9-module-optimization/20260313_prediction_memory_alignment_tuning.md`
> 脚本: `scripts/phase1-autoregressive-1.9-module-optimization/20260313_prediction_memory_alignment_tuning/`
> 日志: `results/logs/phase1-autoregressive-1.9-module-optimization/20260313_prediction_memory_alignment_tuning/`
> 可视化: `results/figures/phase1-autoregressive-1.9-module-optimization/20260313_prediction_memory_alignment_tuning/`
> commit: `dd424ce`
> 结果: 250ms fp-bps=0.1949 / 500ms fp-bps=0.1635 / 1000ms fp-bps=0.1264

- 核心设计：保持上一轮 alignment 训练逻辑不变，只做小范围超参调优
- 调参内容：`mix_prob 0.25 -> 0.35`，`dropout 0.10 -> 0.05`，`noise_std 0.05 -> 0.03`
- 设计动机：上一轮已逼近 `baseline_v2`，当前更值得测试的是 alignment 强度与 regularization 强度的平衡，而不是继续改结构
- 必做实验保持 1.9 统一规范：10 sessions、连续滑动窗口、obs=500ms、pred=250/500/1000ms
- training curves：`results/figures/phase1-autoregressive-1.9-module-optimization/20260313_prediction_memory_alignment_tuning/training_curves.png`
- 2026-03-20 best-ckpt teacher-forced valid/test：`250ms 0.2636 / 0.2667`，`500ms 0.2718 / 0.2655`，`1000ms 0.2815 / 0.2820`
- 当前结果：较 `20260313_prediction_memory_alignment` 再次小幅提升，`1000ms` 距 `baseline_v2` 仅差 `0.0099 fp-bps`；是否继续细调由用户决定
### 1.11 Diffusion Decoder 增量模型管理
> 定位：支持 diffusion decoder 主线的持续模型构建、实验记录与结果对比
> 产出：`cc_todo/1.11-diffusion-decoder/model.md`（diffusion 路线演进文档）、各次扩散解码实验的代码/结果/记录
> 记录：`cc_todo/1.11-diffusion-decoder/{date}_{module_name}.md`
> 效果追踪：`cc_todo/1.11-diffusion-decoder/results.tsv`

**目标**：
建立独立于 1.9 AR 优化线的 diffusion decoder 主线管理机制，围绕 `Option 2B + flow matching + DiT` 持续推进模型实现、验证与结果汇总，同时保留 `Option 2A` 作为后续备选。

#### 1.11.0 执行规范

**Step 1 -- 想法记录与讨论**（在提出 diffusion 改进想法时）：
- 在 `cc_todo/1.11-diffusion-decoder/model.md` 中新增一节，标注日期和改进名称
- 记录：前因后果、想法描述、动机与目的、相比现有方案的改动点、涉及改动模块、与当前仓库实现的结合方式
- 必须显式说明与 `1.9` / `cc_core_files/model.md` 中 AR 路线的关系，包括：哪些失败结论促使转向 diffusion、哪些经验仍可复用
- 进行充分的批判性分析：优缺点、风险、替代方案；若保留 `Option 2A latent diffusion` 作为备选，也需说明与当前主线的取舍
- 基于当前仓库代码实现，给出可落地的修改方案和基本功能验证方案
- 标记状态为“提出”

**Step 2 -- 分支与代码实施**（在用户确认可实施时）：
- 如果用户未明确指定实施分支，则默认直接在 `dev/diffusion` 分支实施
- 按 `cc_todo/1.11-diffusion-decoder/model.md` 中对应小节的修改方案进行代码改动
- 优先清理不再服务于 baseline 或 diffusion 主线的 1.9 运行时代码，保持当前实现简洁
- 完成基本功能验证（代码可跑、核心接口无错误）
- 在 `cc_todo/1.11-diffusion-decoder/model.md` 中更新状态为“实施中”
- **Step 2 完成后立即执行一次 `git commit` + `git push`**：提交当前轮的代码、配置、文档和最小验证结果，作为实现阶段 checkpoint

**Step 3 -- 实验验证**（按优先级依次进行）：
1. **必做 -- 协议确认 + 预测窗口实验**（实验骨架参照 1.3.4，默认协议参照 1.3.7）：
   - 开始实验前必须明确确认本轮实验是否遵循 `1.3.7 NeuroHorizon 实验默认数据与指标标准`
   - 如果与 `1.3.7` 不一致，必须在任务记录中书面记录差异项、原因和影响范围
   - 数据：`examples/neurohorizon/configs/dataset/perich_miller_10sessions.yaml`
   - 采样方式：连续滑动窗口（非 trial-aligned）
   - 默认实现位置（本轮默认保持不变，若改动必须在记录中注明）：
     - 训练入口：`examples/neurohorizon/train.py`
     - 离线正式评估入口：`scripts/analysis/neurohorizon/eval_phase1_v2.py`
     - continuous train sampler：`RandomFixedWindowSampler`
     - continuous valid/test sampler：`SequentialFixedWindowSampler`
   - 观察窗口：500ms
   - 预测窗口：250ms / 500ms / 1000ms（3 个条件）
   - 默认至少记录的指标：`fp-bps` / `per-bin fp-bps`
   - `R-squared` / `PSTH-R-squared` / diffusion validation loss 作为补充指标按需要记录
   - 当前正式实现要求：每个 eval epoch 保存 checkpoint，train end 按 `max(val/fp_bps)` + `min(val_loss)` 选 best ckpt，显式保存真正 final `last.ckpt`，并对 best ckpt 输出 `valid/test` 指标
2. **可选 -- 观察窗口实验**（参照 1.4，用户确认后执行）
3. **可选 -- Session 数目实验**（参照 1.5，用户确认后执行）

**Step 4 -- 实验记录**（遵循 AGENTS/CLAUDE 记录规范）：
- **任务记录**：`cc_todo/1.11-diffusion-decoder/{date}_{module_name}.md`
  - 必须包含：改进想法摘要、实现方案、涉及改动模块、详细实验配置（数据集、sessions 数、采样方式、obs/pred 窗口）、每次实验的关键超参数（至少包括 `epoch`、`batch_size`、`lr`、`weight_decay`、`flow_steps_eval`）、训练 loss 结果、最佳 `val fp-bps`、test `fp-bps`、test 使用的 checkpoint 标识/时间、各条件指标结果（至少 `fp-bps` / `per-bin fp-bps`）、与 baseline 的对比、记录每次训练和评估的脚本命令
- **脚本**：`scripts/1.11-diffusion-decoder/{date}_{module_name}/`
- **实验日志**：`results/logs/1.11-diffusion-decoder/{date}_{module_name}/`
- **可视化**：`results/figures/1.11-diffusion-decoder/{date}_{module_name}/`
  - 除 training curves 外，还必须补充：
    - 随预测窗口长度变化的 `fp-bps` 趋势图
    - 每个预测窗口的 `per-bin fp-bps` 曲线
    - 一个表格型 PNG，用于汇总各窗口的最佳 `val fp-bps`、test `fp-bps`、test checkpoint 标识/时间等核心结果
- **汇总更新**：`cc_core_files/scripts.md` 和 `cc_core_files/results.md` 按规范更新
- **TSV 更新**：将预测窗口实验结果追加到 `cc_todo/1.11-diffusion-decoder/results.tsv`
- **Step 4 完成后再执行一次 `git commit` + `git push`**：提交正式实验结果、图表、结论更新与状态变更，保证每轮 diffusion 迭代至少留下“实现 checkpoint”和“结果 checkpoint”两次提交

**Step 5 -- 分支后续策略**（由用户决定）：
- 效果好 -> merge 到 main，在 `cc_todo/1.11-diffusion-decoder/model.md` 中标记状态为“已合并”
- 效果不佳 -> 保留在 `dev/diffusion` 主线下继续修正，或在文档中标记“已放弃”并记录原因

#### 1.11.1 [x] Diffusion Decoder 路线基线 <!-- 记录：cc_todo/1.11-diffusion-decoder/20260320_direct_count_flow_dit.md -->

> 产出：`cc_todo/1.11-diffusion-decoder/model.md`、`cc_todo/1.11-diffusion-decoder/results.tsv`

- [x] 整理 1.9 AR 路线的失败结论与迁移动机
- [x] 明确 `Option 2B + flow matching + DiT` 为当前主线
- [x] 将 `Option 2A latent diffusion` 记录为后续备选
- [x] 创建 diffusion 路线的首轮配置、脚本和结果目录

#### 1.11.2 [ ] Diffusion Decoder 迭代记录

> 以下按时间顺序记录每次 diffusion decoder 想法及其路径信息

<!-- 模板（每次新优化时复制并填写）:
##### {date}_{module_name} -- {改进名称}
> 状态: 提出 / 实施中 / 验证中 / 已合并 / 已放弃
> 分支: `dev/diffusion`
> 文档: `cc_todo/1.11-diffusion-decoder/model.md` 对应小节
> 任务记录: `cc_todo/1.11-diffusion-decoder/{date}_{module_name}.md`
> 脚本: `scripts/1.11-diffusion-decoder/{date}_{module_name}/`
> 日志: `results/logs/1.11-diffusion-decoder/{date}_{module_name}/`
> 可视化: `results/figures/1.11-diffusion-decoder/{date}_{module_name}/`
> commit: （实施后填写）
> 结果: 250ms fp-bps= / 500ms fp-bps= / 1000ms fp-bps=
-->

##### 20260320_direct_count_flow_dit -- Direct Count-Space Flow Matching with DiT
> 状态: 已放弃
> 分支: `dev/diffusion`
> 文档: `cc_todo/1.11-diffusion-decoder/model.md` 中“2026-03-20 — Direct Count-Space Flow Matching with DiT”
> 任务记录: `cc_todo/1.11-diffusion-decoder/20260320_direct_count_flow_dit.md`
> 脚本: `scripts/1.11-diffusion-decoder/20260320_direct_count_flow_dit/`
> 日志: `results/logs/1.11-diffusion-decoder/20260320_direct_count_flow_dit/`
> 可视化: `results/figures/1.11-diffusion-decoder/20260320_direct_count_flow_dit/`
> commit: `729b4b0`（实现 checkpoint）
> 结果: 250ms fp-bps=`-7.4950` / 500ms fp-bps=`-7.8601` / 1000ms fp-bps=`-8.2277`

- 核心设计：保持 POYO+ encoder 不变，使用 direct count-space flow matching + DiT 风格时间主干生成未来 spike count 场
- 当前主线不再继续 1.9 的 prediction-memory / alignment decoder 分支，相关历史结果仅保留作参考
- 本轮必做实验遵循 1.11 统一规范：10 sessions、连续滑动窗口、obs=500ms、pred=250/500/1000ms
- `Option 2A latent diffusion` 暂不实施，仅作为后续备选保留在文档中
- 三窗口 formal 都跑满 `300 epochs`，但 `continuous / trial-aligned` 指标全部显著为负，说明该变体不是可调优基线，而是结构性失败
- 失败特征不是“只在 tail 崩塌”，而是所有预测 bin 全面为负；后续 diffusion 若继续推进，优先重做 unit-level tokenization 或 factorized time-unit attention，而不是继续沿当前 `per-bin summary` 微调

##### 20260320_factorized_unit_time_flow -- Factorized Unit-Time Flow Tokens
> 状态: 实施中
> 分支: `dev/diffusion`
> 文档: `cc_todo/1.11-diffusion-decoder/model.md` 中“2026-03-20 — Factorized Unit-Time Flow Tokens”
> 任务记录: `cc_todo/1.11-diffusion-decoder/20260320_factorized_unit_time_flow.md`
> 脚本: `scripts/1.11-diffusion-decoder/20260320_factorized_unit_time_flow/`
> 日志: `results/logs/1.11-diffusion-decoder/20260320_factorized_unit_time_flow/`
> 可视化: `results/figures/1.11-diffusion-decoder/20260320_factorized_unit_time_flow/`
> commit:
> 结果: 250ms smoke val/fp-bps=`-15.2800`（offline valid=`-15.2633`） / 500ms fp-bps= / 1000ms fp-bps=

- 设计动机：上一轮确认失败主因更可能来自 `per-bin summary` 对 unit-level 信息的压缩，而不是单纯的调参或 integration 步数
- 本轮继续保留 `Option 2B` 和现有训练/评估入口，只替换 decoder 内部结构，先验证 factorized unit-time token 表达本身
- 最小验收：完成 synthetic 前向与 `250ms` 真实数据 smoke，确认训练、checkpoint 与离线评估链路保持可用


---

## Phase 2：跨 Session 测试

> **目标**：实现 IDEncoder，在 Brainsets 数据上验证跨 session 零样本泛化；可选扩展至 IBL 大规模数据。
> **数据集**：Perich-Miller 2018（必做）；IBL（可选扩展，详见 `cc_core_files/dataset.md` 第 3.3 节）
> **前提**：Phase 1 的自回归改造已验证 causal mask 正确、loss 收敛
> **执行参考**：`cc_core_files/proposal_review.md` 第三节（§三）
> **cc_todo**：`cc_todo/phase2-cross-session/`

### 2.1 IDEncoder 实现

- [ ] **2.1.1** 实现参考窗口数据准备（方案 A: Binned Timesteps）
  - 新建 `scripts/extract_reference_data.py`
  - 从每个 session 的参考时段提取各 unit 的 spike events，binning (20ms) 后插值到固定长度 T_ref
  - 参数：T_ref=100（约 2s 参考窗口），M 个参考窗口（从不同 trial 或时段采样）
  - 输出格式：`ref_data[unit_i] = Tensor[M, T_ref]`（每个 unit 的 M 个参考窗口 binned spike counts）

- [ ] **2.1.2** 实现 IDEncoder 网络（方案 A，参考 SPINT 架构）
  - 新建 `torch_brain/nn/id_encoder.py`
  - 网络结构：MLP1(T_ref -> H) -> mean pool -> MLP2(H -> d_model)，每个 MLP 为 3 层 FC
  - 参数参考 SPINT：hidden_dim=512–1024，output_dim=d_model
  - 输入为原始 binned spike counts（非手工统计特征），端到端学习

- [ ] **2.1.2b**（Phase 2 后期）实现方案 B: Spike Event Tokenization IDEncoder
  - 输入：参考窗口 raw spike event timestamps，注入 rotary time embedding
  - 聚合：attention pooling / mean pooling -> MLP -> d_model
  - 与方案 A 对比实验，验证保留精确 spike timing 对 identity 推断的贡献
  - **作为 NeuroHorizon 的创新点之一**：Spike Event Tokenization for IDEncoder

- [ ] **2.1.3** 集成到 NeuroHorizon
  - IDEncoder 输出替换 `InfiniteVocabEmbedding` 的 unit_emb（非加法注入），**注意保留其 tokenizer/detokenizer 逻辑**（参见 `proposal_review.md` 第三节（§三））
  - 更新 `torch_brain/nn/__init__.py`；优化器参数组：IDEncoder 用 AdamW，session_emb 保留 SparseLamb
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
  - IDEncoder 结果是否足够支持 paper 贡献？是否需要 IBL 扩展（详见 2.4）？

### 2.4 IBL 跨 Session 扩展（可选）

> **前提**：Phase 2.3 结果令人满意，需更大规模验证跨 session 泛化

- [ ] **2.4.1** 安装 ONE API + ibllib，验证数据管线（下载 10-20 sessions 调试，约5-10GB）

- [ ] **2.4.2** 编写 IBL Dataset 类
  - 新建 `examples/neurohorizon/datasets/ibl.py`
  - 实现滑动窗口策略（不对齐 trial）；质量过滤：仅使用 `clusters.label == 1`

- [ ] **2.4.3** IBL 大规模跨 session 实验
  - 逐步扩展：20 → 50 → 100 sessions（视结果动态调整）
  - 按实验室划分 train/test（12 个 labs 跨实验室泛化）

### 2.5 FALCON Benchmark（可选补充）

- [ ] **2.5.1** 注册并下载 FALCON M1/M2 数据
- [ ] **2.5.2** 在 FALCON 上量化跨 session 泛化改进（与 IDEncoder baseline 对比）

---

## Phase 3：Data Scaling + 下游任务泛化

> **目标**：揭示性能随训练数据量（session 数）的 scaling 规律；验证自回归预训练对行为解码下游任务的迁移增益。
> **数据集**：Perich-Miller 2018（必做）；IBL（可选扩展，需 Phase 2.4 管线就绪）
> **前提**：Phase 2 跨 session 泛化已有基本结论
> **执行参考**：`cc_core_files/proposal_review.md` 第四节（§四）
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
> **执行参考**：`cc_core_files/proposal_review.md` 第五节（§五）
> **cc_todo**：`cc_todo/phase4-multimodal/`

### 4.1 Allen 数据准备

- [ ] **4.1.1** 确认存储空间（需 > 150GB），规划数据下载路径
- [ ] **4.1.2** 安装 AllenSDK（独立 conda 环境，避免依赖冲突）
- [ ] **4.1.3** 下载 Allen Visual Coding Neuropixels NWB 数据（58 sessions）；记录到 `cc_core_files/data.md`
- [ ] **4.1.4** 预处理转 HDF5，主环境直接加载
- [ ] **4.1.5** 离线预提取 DINOv2 embeddings（**必须离线预计算，不可训练时实时计算**）
  - 新建 `scripts/extract_dino_embeddings.py`
  - 灰度图（918×1174）→ 复制三通道转 RGB → resize 224×224 → DINOv2 ViT-B → 缓存为 `.pt`

### 4.2 多模态数据集与模型实现

- [ ] **4.2.1** 编写 Allen Dataset 类（`examples/neurohorizon/datasets/allen_multimodal.py`）
- [ ] **4.2.2** 实现行为条件注入（linear projection + rotary time embedding → 输入端拼接，方案详见 `proposal_review.md` §五）
- [ ] **4.2.3** 实现 DINOv2 图像条件注入（linear projection → 输入端拼接，DINOv2 权重冻结，仅训练投影层，方案详见 `proposal_review.md` §五）

### 4.3 多模态实验

- [ ] **4.3.1** Allen Natural Movies 长时程连续预测（250ms / 500ms / 1s 梯度测试）
- [ ] **4.3.2** 图像-神经对齐实验（Natural Scenes + DINOv2，量化贡献）
- [ ] **4.3.3** 多模态消融（neural only / +behavior / +image / +behavior+image）

---

## Phase 5：完整实验、消融与论文

> **cc_todo**：`cc_todo/phase5-experiments-paper/`

### 5.1 完整实验矩阵

- [ ] **5.1.1** 长时程预测实验（100ms / 200ms / 500ms / 1000ms 预测窗口系统性对比）
- [ ] **5.1.2** 跨 session 泛化核心实验（含 baseline 全面对比）
- [ ] **5.1.3** Data scaling law 实验（Brainsets + 可选 IBL）
- [ ] **5.1.4** 多模态贡献分析（可选，若 Phase 4 完成）

### 5.2 消融实验

- [ ] **5.2.1** IDEncoder 消融（A1: IDEncoder vs 可学习嵌入；A2: IDEncoder vs random）
- [ ] **5.2.2** Decoder 深度消融（N_dec = 1 / 2 / 4）
- [ ] **5.2.3** 预测窗口长度消融（100ms / 250ms / 500ms / 1000ms）
- [ ] **5.2.4** Scheduled sampling 消融（无 / 固定比例 / 逐步衰减）
- [ ] **5.2.5** Causal decoder vs parallel prediction（非自回归并行预测）对比

### 5.3 Baseline 对比

- [ ] **5.3.1** PSTH-based baseline、线性预测 baseline
- [ ] **5.3.2** Neuroformer（自回归生成 + 多模态，最接近 NeuroHorizon）
- [ ] **5.3.3** NDT1/NDT2（masked spike prediction，binned counts）
- [ ] **5.3.4** NDT3（IBL 上有公开结果，IBL 实验时引入）
- [ ] **5.3.5** NEDS（同时支持 spike 预测 + 行为解码，与场景最接近）

### 5.4 结果可视化

- [ ] **5.4.1** Scaling curves（session 数 vs 性能折线图）
- [ ] **5.4.2** 预测性能随时间窗口衰减曲线
- [ ] **5.4.3** 模态贡献归因图
- [ ] **5.4.4** IDEncoder embedding 空间可视化（PCA / t-SNE）
- 所有图表记录到 `cc_core_files/results.md`

### 5.5 论文撰写

- [ ] **5.5.1** 方法章节初稿（Architecture + IDEncoder + Autoregressive Decoder）
- [ ] **5.5.2** 实验章节初稿（含所有核心图表）
- [ ] **5.5.3** Introduction + Related Work + Discussion 完善
- [ ] **5.5.4** 论文定稿，准备目标会议（NeurIPS / ICLR / Nature Methods）投稿

---

*计划创建：2026-02-28；附录内容已迁移至 `cc_core_files/proposal_review.md`*
