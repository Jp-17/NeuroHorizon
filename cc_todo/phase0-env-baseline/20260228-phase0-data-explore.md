# 2026-02-28 Phase 0.2 数据准备与探索

**日期**：2026-02-28  
**对应 plan.md 任务**：0.2.1、0.2.2、0.2.3  
**任务目标**：检查/下载 Perich-Miller 数据，验证数据加载，完成数据深度探索分析

---

## 做了什么

### 0.2.1 检查并下载 Brainsets 数据

**检查结果**：
- `data/` 下已有 IBL、Allen、NLB 数据（来自已废弃任务），但无 Brainsets/Perich-Miller 数据
- 需要新建 `data/raw/` 和 `data/processed/` 目录并下载

**操作步骤**：
1. 通过 DANDI REST API 查询 DANDI:000688 总计 111 sessions，总大小 约12.3GB
   - sub-C：68 sessions，9.3GB；sub-J：3 sessions，84MB；sub-M：28 sessions，2.8GB；sub-T：12 sessions，418MB
2. 更新 `~/.brainsets.yaml` 指向 NeuroHorizon 数据目录
3. 在 poyo 环境安装 `dandi==0.61.2`（pipeline 依赖）
4. 编写 `scripts/data/perich_miller_pipeline.py`（brainsets pipeline 修改版）
   - 通过 `SELECTED_PATHS` 过滤，只下载 10 sessions（4 C-CO + 3 J-CO + 3 M-CO，共 320MB）
   - 选择标准：小文件（26-40MB），覆盖三个动物（C/J/M），均为 center_out_reaching
5. 运行 `brainsets.runner` 下载并处理数据

**输出**：
- raw NWB：`data/raw/perich_miller_population_2018/sub-{C,J,M}/`（10 files，约320MB）
- processed HDF5：`data/processed/perich_miller_population_2018/`（10 files，约100MB）

**完整 session 列表**：
- sub-C: c_20131003, c_20131022, c_20131101, c_20131204（center_out_reaching）
- sub-J: j_20160405, j_20160406, j_20160407（center_out_reaching，J 的全部 3 sessions）
- sub-M: m_20150610, m_20150612, m_20150615（center_out_reaching）

### 0.2.2 数据加载验证

**使用 temporaldata.Data.from_hdf5() 验证单文件加载**：
- `c_20131003_center_out_reaching.h5`：domain 0-663s，319409 spikes，71 units，157 valid trials
- hold_period 均值 247ms；reach_period 均值 1022ms
- train/valid/test split 已预计算（38 intervals / 16 trials / 32 trials）

**使用 torch_brain.data.Dataset 验证 pipeline 加载**：
- 4 sessions 测试：Dataset 创建成功，4 sessions × 约580 windows = 2324 train windows
- 全 10 sessions：6372 train windows（1s 窗口）
- RandomFixedWindowSampler 正常，单样本访问正常（约350 spikes/window）
- 发现：`torch_brain.data.Dataset` 已废弃，新 API 为 `torch_brain.dataset.Dataset`

### 0.2.3 数据深度探索分析

**脚本**：`scripts/analysis/explore_brainsets.py`  
**结果**：`results/figures/data_exploration/`

**主要发现**：

1. **数据集概览**：
   - Subject C：41-71 units/session（中位 46）；J：18-38（中位 19）；M：37-49（中位 41）
   - Valid trials/session：150-243（C），195-203（J），164-184（M）
   - 记录时长：10-21 分钟

2. **任务结构分析**：
   - Hold period（输入窗口候选）：均值 676ms，87% > 250ms，61% > 500ms
   - Reach period（预测窗口候选）：均值 1090ms，**100% > 250ms，100% > 500ms，75% > 1000ms**
   - → **"hold period 作为输入，reach period 作为预测目标"的设计完全可行**

3. **神经元统计**：
   - per-unit 平均 firing rate：均值 6.8Hz，中位 3.5Hz，14.5% < 1Hz
   - per-unit spike 稀疏度（20ms bin）：mean=0.133，zero_frac=87.6%
   - Mean-Variance 关系接近 Poisson 特性

4. **自回归可行性**：
   - Population-level 20ms bin：mean=5.17 spikes（4.1% zero）→ Poisson NLL 适用
   - 250ms → 12 autoregressive steps（Phase 1 起点）
   - 500ms → 25 steps；1000ms → 50 steps（需 scheduled sampling）

---

## 遇到的问题及解决方法

### 1. `.gitignore` 中 `data/` 规则导致 `scripts/data/` 被忽略
- **问题**：git 的 `data/` 模式匹配任意目录层级的 `data` 文件夹，包括 `scripts/data/`
- **解决**：在 `.gitignore` 末尾添加 `!scripts/data/` 和 `!scripts/data/**` 例外规则，并用 `git add -f` 强制追踪

### 2. temporaldata lazy loading 需在 h5py 上下文内访问
- **问题**：`Data.from_hdf5` 默认 lazy loading，关闭 h5 文件后访问属性报 RuntimeError
- **解决**：所有数据访问操作放在 `with h5py.File() as f:` 上下文内完成

### 3. torch_brain.data.Dataset API 变动
- **问题**：`recording_ids` 属性不存在，API 签名和参数名变化（`interval_dict` → `sampling_intervals`）
- **解决**：通过 inspect 检查当前 API 签名，使用 `recording_dict.keys()` 和正确参数名

### 4. DANDI asset 列表需要 dandi 包
- **问题**：brainsets pipeline 运行需要 dandi 包，但 poyo 环境未安装
- **解决**：`pip install dandi==0.61.2 scikit-learn` 安装到 poyo 环境

---

## 结果如何

- data pipeline sanity check **PASSED**：10 sessions 全部可通过 torch_brain 加载
- 探索分析完成，关键决策依据已建立：
  - Phase 1 推荐起点：c_20131003（71 units，最大 C session），250ms 预测窗口
  - Poisson NLL 适用性：confirmed（mean-variance ~ Poisson）

---

## 各种文件在哪里

| 文件 | 路径 |
|------|------|
| 下载 pipeline | `scripts/data/perich_miller_pipeline.py` |
| 探索分析脚本 | `scripts/analysis/explore_brainsets.py` |
| raw NWB 数据 | `data/raw/perich_miller_population_2018/` |
| processed HDF5 | `data/processed/perich_miller_population_2018/` |
| 概览图 | `results/figures/data_exploration/01_dataset_overview.png` |
| 神经统计图 | `results/figures/data_exploration/02_neural_statistics.png` |
| 统计摘要 | `results/figures/data_exploration/exploration_summary.json` |

---

## 还有什么没有做

- 0.2.4（可选）：NLB MC_Maze 数据已存在（`data/nlb/`），可直接用，无需重新下载
- 完整数据集（111 sessions）尚未下载，后续按 Phase 2/3 需求扩展
- Session 间神经元重叠度未深入分析（brainsets 中同一动物跨 session 的 unit 对应关系）
