# 脚本管理记录

> 记录项目中所有脚本（数据处理 / 分析 / 项目运行 / 测试等）的信息。
> 每新建一个脚本，必须在此处更新记录。

---

## 记录格式

```
### 脚本名称

- **路径**：scripts/xxx/xxx.py
- **功能用途**：描述该脚本做什么
- **创建时间**：YYYY-MM-DD
- **使用方式**：
  ```bash
  python scripts/xxx/xxx.py --args
  ```
- **输入**：说明输入文件/目录
- **输出**：说明输出文件/目录
- **依赖**：特殊依赖（环境、包等）
- **备注**：其他注意事项
```

---

## 脚本列表


### perich_miller_pipeline.py

- **路径**：`scripts/data/perich_miller_pipeline.py`
- **功能用途**：下载并处理 Perich-Miller Population 2018 数据集的子集（10 sessions）
  - 从 DANDI Archive（DANDI:000688/draft）下载指定 sessions 的 NWB 文件
  - 提取 spike times、cursor 行为数据、trial 结构，转换为 brainsets 标准 HDF5 格式
  - 划分 train/valid/test splits（valid=0.1, test=0.2）
- **创建时间**：2026-02-28
- **使用方式**：
  ```bash
  # 激活 poyo 环境后运行
  conda activate poyo
  python -m brainsets.runner scripts/data/perich_miller_pipeline.py \
      --raw-dir=data/raw --processed-dir=data/processed -c4
  ```
- **输入**：DANDI Archive 在线下载（需网络）
- **输出**：
  - raw NWB：`data/raw/perich_miller_population_2018/sub-{C,J,M}/`
  - processed HDF5：`data/processed/perich_miller_population_2018/*.h5`
- **依赖**：poyo conda 环境（dandi>=0.61.2, scikit-learn, temporaldata, brainsets）
- **备注**：
  - 修改自 brainsets 官方 pipeline，通过 `SELECTED_PATHS` 限制为 10 sessions
  - 如需下载更多 sessions，修改 `SELECTED_PATHS` 集合或直接用 `brainsets prepare`
  - 运行需要 `brainsets` 配置（`~/.brainsets.yaml`）或通过 `--raw-dir/--processed-dir` 覆盖

---



---

## 现有脚本（POYO 框架原有）

### calculate_normalization_scales.py

- **路径**：scripts/calculate_normalization_scales.py
- **功能用途**：计算数据集归一化参数（POYO 框架原有脚本）
- **创建时间**：（POYO 框架原有）
- **备注**：POYO 框架自带，非 NeuroHorizon 新增
