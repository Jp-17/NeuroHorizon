# 数据集管理记录

> 记录项目中所有数据集的处理信息、存储位置、字段含义等。
> 每次下载/处理数据，必须在此处更新记录。

---

## 目录结构规范

```
data/
├── raw/          # 原始下载数据（未做任何处理）
├── processed/    # 预处理后数据（格式转换、筛选等，通常为 HDF5）
└── generated/    # 模型生成/推理产生的数据
```

---

## 记录格式

```
### 数据集名称

- **存储路径**：data/xxx/
- **数据来源**：下载链接或来源说明
- **下载时间**：YYYY-MM-DD
- **数据规模**：文件大小、session 数、unit 数等
- **数据内容**：包含哪些字段、模态
- **处理状态**：raw / processed / ready
- **处理脚本**：使用哪个脚本处理（参考 scripts.md）
- **用途**：在项目中用于什么实验
- **备注**：其他注意事项
```

---

## 已有数据

### IBL Brain-wide Map 数据（现有缓存）

- **存储路径**：data/ONE/、data/ibl_cache/、data/ibl_processed/
- **数据来源**：International Brain Laboratory，ONE API (AWS)
- **数据规模**：IBL 总计 459 sessions，75,708 good quality units，241 个脑区
- **数据内容**：spike times, unit 信息, 行为数据（wheel velocity, choice 等）
- **处理状态**：部分预处理（来自已废弃任务，需核实状态）
- **用途**：跨 session 泛化、scaling law、长时间预测
- **备注**：⚠️ data/ibl_cache/ 和 data/ibl_processed/ 中的数据来自已废弃的 cc_todo/20260221-cc-1st 任务，使用前需核实数据格式是否符合当前项目规范

### Allen Brain Observatory 数据（现有缓存）

- **存储路径**：data/allen_cache/、data/allen_processed/、data/allen_embeddings/、data/allen_stimuli/
- **数据来源**：Allen Institute for Brain Science，AllenSDK (AWS NWB)
- **数据规模**：58 sessions，多种刺激类型（Natural Scenes, Natural Movies, Drifting Gratings 等）
- **数据内容**：spike times, 刺激图像, 单元信息
- **处理状态**：部分预处理（来自已废弃任务，需核实状态）
- **用途**：多模态（图像）融合实验
- **备注**：⚠️ 同上，来自已废弃任务，使用前需核实

---

### Perich-Miller Population 2018（Brainsets 下载）

- **存储路径**：
  - raw: `data/raw/perich_miller_population_2018/`（原始 NWB 文件）
  - processed: `data/processed/perich_miller_population_2018/`（brainsets 标准 HDF5 格式）
- **数据来源**：DANDI Archive（DANDI:000688/draft），通过 brainsets CLI 下载
- **下载时间**：2026-02-28
- **数据规模**：10 sessions（初始子集），总计约 320MB raw / 100MB processed
  - sub-C（猴 C）：4 sessions（center_out_reaching，2013年）
  - sub-J（猴 J）：3 sessions（center_out_reaching，2016年，全部）
  - sub-M（猴 M）：3 sessions（center_out_reaching，2015年）
  - 完整数据集共 111 sessions（C:68 / J:3 / M:28 / T:12），总 ~12.3GB，按需扩展
- **数据内容**：
  - 神经数据：spike times（Utah Array，M1/PMd 皮层，手动 spike sorting）
  - 行为数据：cursor position / velocity / acceleration（2D，连续时间序列）
  - 任务结构：center_out_reaching（center-out task），含 hold_period / reach_period / return_period
  - 元数据：trial 标签（valid/invalid）、go_cue_time、target_id、result 等
- **处理状态**：processed（brainsets 标准 HDF5，含 train/valid/test split）
- **处理脚本**：`scripts/data/perich_miller_pipeline.py`（本地修改版 pipeline，限 10 sessions）
- **用途**：
  - Phase 0：POYO+ baseline 复现（task 0.2/0.3）
  - Phase 1：自回归改造验证（causal mask / Poisson NLL / 预测窗口梯度测试）
  - Phase 2：跨 session 泛化（按动物划分 train/test：C+J 训练，M held-out）
  - Phase 3：data scaling law（需按需扩展至 70+ sessions）
- **字段说明（HDF5 格式）**：
  - `spikes`：spike 时间戳（IrregularTimeSeries），字段含 `timestamps`、`unit_index`
  - `units`：神经元信息
  - `cursor`：行为数据（pos/vel/acc），IrregularTimeSeries
  - `trials`：trial 区间，含 `hold_period`、`reach_period`、`return_period`
  - `movement_phases`：各阶段 Interval（hold/reach/return/random/invalid）
  - train/valid/test domain 已划分（valid=0.1, test=0.2，random_state=42）
- **备注**：
  - 数据通过 `brainsets.runner` + `--use-active-env` 在 poyo conda 环境下运行处理
  - 使用 `perich_miller_pipeline.py` 中的 `SELECTED_PATHS` 过滤指定 sessions
  - 后续扩展：修改 `SELECTED_PATHS` 或直接用 `brainsets prepare perich_miller_population_2018` 下载全量

---

## 待下载/处理数据

*（随项目正式启动后持续补充）*
