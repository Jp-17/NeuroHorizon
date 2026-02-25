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

## 待下载/处理数据

*（随项目正式启动后持续补充）*
