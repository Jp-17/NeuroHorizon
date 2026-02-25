# NeuroHorizon 数据集下载与使用指南

> 本文档针对 `research_proposal_NeuroHorizon.md` 中涉及的核心数据集，重点介绍 **IBL Brain-wide Map** 和 **Allen Brain Observatory Neuropixels Visual Coding** 的下载方式、所需数据部分及使用方法，并附数据集与项目目标的匹配度分析。

---

## 目录

1. [数据集总览与优先级](#1-数据集总览与优先级)
2. [IBL 数据集（跨session泛化核心）](#2-ibl-数据集)
3. [Allen Brain Observatory 数据集（多模态融合核心）](#3-allen-brain-observatory-数据集)
4. [NeuroHorizon 项目推荐工作流](#4-neurohorizon-项目推荐工作流)
5. [数据集匹配度深度分析与执行注意事项](#5-数据集匹配度深度分析与执行注意事项)

---

## 1. 数据集总览与优先级

| 数据集 | NeuroHorizon 用途 | 规模 | 下载方式 | 磁盘空间 |
|--------|------------------|------|---------|---------|
| **IBL Brain-wide Map** | 跨session泛化、data scaling | 459 sessions，75,708 高质量units | ONE API (Python) | 按需下载，每session约数百MB |
| **Allen Neuropixels Visual Coding** | 多模态（neural + image）融合 | 58 sessions，~100,000 units | AllenSDK (Python) | 完整约855GB，NWB文件146.5GB |

---

## 2. IBL 数据集

### 2.1 数据集概况（2025年版本）

IBL（国际脑实验室）Brain-wide Map 是目前规模最大的标准化全脑 Neuropixels 记录数据集，2025年最新版本包含：

- **459个实验session**，699次探针插入，来自139只小鼠、12个实验室
- **621,733个总units**，其中 **75,708个通过质量控制**（"good quality"）
- 覆盖 **241个脑区**（视觉皮层、前额叶、纹状体、丘脑、海马、小脑等）
- 统一行为任务：视觉决策任务（小鼠根据光栅对比度转动滚轮做出左/右判断）
- 数据通过 [AWS 开放数据注册表](https://registry.opendata.aws/ibl-brain-wide-map/) 公开获取

### 2.2 安装依赖

```bash
# 安装 ONE API 和 IBL 工具链
pip install ONE-api ibllib brainbox

# 可选：安装可视化工具
pip install iblviewer
```

### 2.3 连接设置（无需账号，公开访问）

```python
from one.api import ONE

# 使用公开访问地址（无需注册账号）
one = ONE(base_url='https://openalyx.internationalbrainlab.org',
          password='international')

# 验证连接
print(one.alyx.base_url)
```

> **注意**：也可以通过 AWS 访问数据（无需单独下载），适合在云端进行大规模实验。

### 2.4 需要下载的数据部分

对于 NeuroHorizon 项目，建议按以下优先级下载：

#### 核心数据（必须）

| 数据对象 | ALF 文件名 | 说明 |
|---------|-----------|------|
| `spikes.times` | `spikes.times.npy` | spike 时间戳（秒），IDEncoder + tokenization 核心输入 |
| `spikes.clusters` | `spikes.clusters.npy` | 每个 spike 对应的 cluster ID |
| `spikes.amps` | `spikes.amps.npy` | spike 幅值（用于质量评估） |
| `spikes.depths` | `spikes.depths.npy` | spike 在探针上的深度 |
| `clusters.metrics` | `clusters.metrics.pqt` | 质量控制指标（必须用于过滤 good units） |
| `clusters.channels` | `clusters.channels.npy` | cluster 对应的通道 |
| `clusters.depths` | `clusters.depths.npy` | cluster 深度 |
| `clusters.brainLocations` | 需要组织学数据 | 脑区归属（需组织学追踪数据才可用） |

#### 行为数据（多模态实验必须）

| 数据对象 | ALF 文件名 | 说明 |
|---------|-----------|------|
| `trials.stimOn_times` | `_ibl_trials.stimOn_times.npy` | 刺激呈现时间 |
| `trials.response_times` | `_ibl_trials.response_times.npy` | 小鼠响应时间 |
| `trials.choice` | `_ibl_trials.choice.npy` | 小鼠选择（-1=左，1=右） |
| `trials.contrastLeft` | `_ibl_trials.contrastLeft.npy` | 左侧光栅对比度 |
| `trials.contrastRight` | `_ibl_trials.contrastRight.npy` | 右侧光栅对比度 |
| `trials.feedbackType` | `_ibl_trials.feedbackType.npy` | 奖励/惩罚 |
| `trials.reactionTime` | `_ibl_trials.reactionTime.npy` | 反应时间 |
| `wheel.position` | `_ibl_wheel.position.npy` | 滚轮位置（连续信号） |
| `wheel.timestamps` | `_ibl_wheel.timestamps.npy` | 滚轮时间戳 |
| `wheelMoves.intervals` | `_ibl_wheelMoves.intervals.npy` | 运动片段的时间区间 |

#### 可选数据（建议下载）

| 数据对象 | 说明 |
|---------|------|
| `clusters.waveforms` | 波形模板（用于 IDEncoder 特征提取，但waveforms文件较大） |
| `passiveGabor / passiveRFM` | 被动视觉刺激数据（如有） |

### 2.5 批量搜索并下载 Session 列表

```python
from one.api import ONE
import pandas as pd

one = ONE(base_url='https://openalyx.internationalbrainlab.org',
          password='international')

# ① 搜索所有 Brain-wide Map session（带 spike 数据）
eids = one.search(
    project='brainwide',
    task_protocol='_iblrig_tasks_ephysChoiceWorld',
    datasets=['spikes.times.npy', '_ibl_trials.table.pqt']
)
print(f"找到 {len(eids)} 个满足条件的 session")

# ② 获取所有 session 的 metadata 表格（便于按条件筛选）
sessions = one.alyx.rest('sessions', 'list', project='brainwide',
                          task_protocol='_iblrig_tasks_ephysChoiceWorld',
                          performance_gte=70)  # 可选：按任务表现筛选

# 保存 session 列表
df = pd.DataFrame(sessions)
df.to_csv('ibl_sessions.csv', index=False)
```

### 2.6 加载单个 Session 的 Spike 数据（推荐方式）

IBL 官方推荐使用 `SpikeSortingLoader` 来加载 spike 数据：

```python
from one.api import ONE
from brainbox.io.one import SpikeSortingLoader

one = ONE(base_url='https://openalyx.internationalbrainlab.org',
          password='international')

# 方式一：使用 probe insertion ID (pid)
pid = 'da8dfec1-d265-44e8-84ce-6ae9c109b8bd'  # 替换为实际的 pid
ssl = SpikeSortingLoader(pid=pid, one=one)
spikes, clusters, channels = ssl.load_spike_sorting()

# 方式二：使用 session ID (eid) + probe 名称
eid = 'caa5dddc-9290-4e27-9f5e-575ba3598614'
pname = 'probe00'
ssl = SpikeSortingLoader(eid=eid, pname=pname, one=one)
spikes, clusters, channels = ssl.load_spike_sorting()

# 合并 cluster 信息
clusters = ssl.merge_clusters(spikes, clusters, channels)

print(f"总 spikes: {len(spikes['times'])}")
print(f"总 clusters: {len(clusters['cluster_id'])}")
print(f"可用字段 - spikes: {list(spikes.keys())}")
print(f"可用字段 - clusters: {list(clusters.keys())}")
```

### 2.7 质量过滤（获取 Good Units）

```python
import numpy as np

# ① 基于 label 过滤（推荐，最简单）
# label == 1 表示 good quality
good_mask = clusters['label'] == 1
good_cluster_ids = clusters['cluster_id'][good_mask]
print(f"Good quality units: {good_mask.sum()} / {len(good_mask)}")

# ② 仅保留 good units 的 spikes
spike_mask = np.isin(spikes['clusters'], good_cluster_ids)
good_spikes = {k: v[spike_mask] for k, v in spikes.items()}

# ③ 精细化过滤（可选，使用 clusters.metrics 中的具体指标）
# firing_rate > 0.1 Hz，amplitude > 50 uV，presence_ratio > 0.9
if 'firing_rate' in clusters and 'amp_median' in clusters:
    quality_mask = (
        (clusters['label'] == 1) &
        (clusters['firing_rate'] > 0.1) &
        (clusters['amp_median'] > 50) &
        (clusters['presence_ratio'] > 0.9)
    )
    filtered_cluster_ids = clusters['cluster_id'][quality_mask]
    print(f"精细过滤后 units: {quality_mask.sum()}")
```

### 2.8 加载行为数据（Trial 和 Wheel）

```python
# 加载 trials 对象（包含所有行为变量）
trials = one.load_object(eid, 'trials', collection='alf')

# 核心字段说明：
# trials.stimOn_times     - 刺激呈现时刻（对齐 spike 时间戳的参考点）
# trials.response_times   - 小鼠做出选择的时刻
# trials.goCueTrigger_times - Go 信号触发时刻
# trials.intervals        - 每个 trial 的开始和结束时间 [N, 2]
# trials.choice           - -1 (左)，1 (右)，0 (未响应)
# trials.contrastLeft/Right - 左右光栅对比度 (0, 0.0625, 0.125, 0.25, 1.0)
# trials.feedbackType     - 1 (正确)，-1 (错误)
# trials.reactionTime     - 反应时间（秒）

print(f"总 trial 数: {len(trials.stimOn_times)}")
print(f"正确率: {(trials.feedbackType == 1).mean():.2%}")

# 加载连续行为数据（wheel）
wheel_times = one.load_dataset(eid, '_ibl_wheel.timestamps.npy')
wheel_pos = one.load_dataset(eid, '_ibl_wheel.position.npy')
```

### 2.9 针对 NeuroHorizon 的批量下载脚本

```python
"""
批量下载 IBL Brain-wide Map 数据用于 NeuroHorizon 训练
建议首先下载10-20个session进行代码调试，再扩展到全量
"""
from one.api import ONE
from brainbox.io.one import SpikeSortingLoader
import numpy as np
import os

one = ONE(base_url='https://openalyx.internationalbrainlab.org',
          password='international')

# 搜索满足条件的 sessions
eids = one.search(
    project='brainwide',
    datasets=['spikes.times.npy', '_ibl_trials.table.pqt'],
    task_protocol='_iblrig_tasks_ephysChoiceWorld'
)
print(f"共找到 {len(eids)} 个 session")

# 获取每个 session 的 probe 列表
for eid in eids[:5]:  # 先下载前5个 session 进行测试
    try:
        insertions = one.alyx.rest('insertions', 'list', session=eid)
        for ins in insertions:
            pid = ins['id']
            pname = ins['name']

            # 下载 spike sorting 数据
            ssl = SpikeSortingLoader(pid=pid, one=one)
            spikes, clusters, channels = ssl.load_spike_sorting()
            clusters = ssl.merge_clusters(spikes, clusters, channels)

            # 下载 trials 数据
            trials = one.load_object(eid, 'trials', collection='alf')

            # 可在此处将数据转存为 .npz 或 HDF5 格式
            save_path = f"./ibl_data/{eid}/{pname}"
            os.makedirs(save_path, exist_ok=True)
            np.savez(f"{save_path}/spikes.npz",
                     times=spikes['times'],
                     clusters=spikes['clusters'],
                     amps=spikes['amps'])
            np.savez(f"{save_path}/clusters.npz",
                     cluster_id=clusters['cluster_id'],
                     label=clusters['label'],
                     firing_rate=clusters.get('firing_rate', np.array([])))
            print(f"  ✓ {eid}/{pname}: {(clusters['label']==1).sum()} good units")
    except Exception as e:
        print(f"  ✗ {eid}: {e}")
```

### 2.10 IBL 数据的 NeuroHorizon 适配注意事项

- **IDEncoder 参考窗口**：建议取每个 session 开头 10-30 秒（task 开始前的静息期或第一个 trial block）作为参考窗口，提取 firing rate、ISI 统计等特征。
- **时间对齐**：所有 spike 时间戳均以 session 开始为零点（秒），与 `trials.stimOn_times` 直接对齐。
- **跨session划分**：按 session 划分 train/val/test（而非按 trial），建议 80/10/10 比例。
- **脑区过滤**：如需聚焦特定脑区，通过 `channels.brainLocations.acronym` 过滤（需要组织学数据）。

---

## 3. Allen Brain Observatory 数据集

### 3.1 数据集概况

Allen Brain Observatory Neuropixels Visual Coding 由艾伦脑科学研究所发布：

- **58个实验 session**（来自不同小鼠，包括野生型和3种转基因品系）
- 同时记录最多 **8个视觉相关脑区**：V1、LM、AL、PM、AM、RL、LGN、LP
- 约 **100,000个总 units**
- 两套刺激集：`brain_observatory_1.1`（与2P数据共享刺激，含自然图像118张）和 `functional_connectivity`（高重复次数刺激）
- 提供完善的 Python API（AllenSDK），数据以 NWB 格式存储
- **存储需求**：NWB session 文件约 146.5 GB（58个session，每个 1.7-3.3 GB）；完整数据集（含原始数据）约 80 TB

### 3.2 安装依赖

```bash
# 安装 AllenSDK（注意版本兼容性）
pip install allensdk

# 验证安装
python -c "from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache; print('OK')"
```

> **版本注意**：AllenSDK 2.0.0 之前版本的 NWB 文件（2020年6月11日前发布）与新版 pynwb 不兼容，建议使用 AllenSDK >= 2.0.0 并下载最新版本的 NWB 文件。

### 3.3 初始化缓存

```python
import os
import numpy as np
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

# 设置本地缓存目录（确保有足够空间）
cache_dir = '/your/storage/path/allen_cache'  # 替换为实际路径
manifest_path = os.path.join(cache_dir, 'manifest.json')

# 初始化缓存（首次运行会下载 manifest，几MB）
cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

# 查看可用的 session 类型
print(cache.get_all_session_types())
# 输出: ['brain_observatory_1.1', 'functional_connectivity']
```

### 3.4 两套刺激集说明

| 刺激集 | 主要刺激类型 | 特点 | NeuroHorizon 用途 |
|--------|------------|------|------------------|
| `brain_observatory_1.1` | 自然图像（118张）、drifting gratings、static gratings、natural movies | 与 2P 数据共享，视觉刺激多样 | 多模态 neural + image 实验（DINOv2 图像编码） |
| `functional_connectivity` | Gabor patches（高重复次数） | 重复次数多，适合统计分析 | 功能连接研究、高重复trial验证 |

**NeuroHorizon 建议优先使用 `brain_observatory_1.1`**，因为它包含自然图像，可用于验证 DINOv2 多模态融合效果。

### 3.5 需要下载的数据部分

#### 必须下载

通过 `cache.get_session_data(session_id)` 下载包含以下所有内容的 NWB 文件（每个文件 1.7-3.3 GB）：

| 数据类型 | 访问方式 | 说明 |
|---------|---------|------|
| **Spike times** | `session.spike_times` | dict: unit_id → array of spike times（秒）|
| **Units 表格** | `session.units` | 包含 unit 质量指标的 DataFrame |
| **Stimulus presentations** | `session.stimulus_presentations` | 刺激呈现表（start_time, end_time, stimulus_name 等）|
| **Running speed** | `session.running_speed` | 小鼠跑动速度时间序列 |
| **Pupil tracking** | `session.eye_tracking` | 瞳孔直径和位置 |

#### 可选下载（LFP 文件，更大）

LFP 数据单独存储在 LFP NWB 文件中，若只做 spike 分析可跳过。

### 3.6 浏览可用 Session

```python
# 获取所有 session 的元数据表
sessions_table = cache.get_session_table()
print(sessions_table.columns.tolist())
# 包含：ecephys_session_id, specimen_id, session_type, sex, age_in_days,
#        full_genotype, unit_count, channel_count, probe_count, ...

# 按 session 类型筛选（推荐多模态实验使用 brain_observatory_1.1）
bo_sessions = sessions_table[
    sessions_table.session_type == 'brain_observatory_1.1'
]
print(f"Brain Observatory 1.1 sessions: {len(bo_sessions)}")

# 获取高质量 unit 数量多的 session
top_sessions = bo_sessions.nlargest(10, 'unit_count')
print(top_sessions[['ecephys_session_id', 'unit_count', 'probe_count']])
```

### 3.7 下载并加载单个 Session

```python
# 选择一个 session ID（从 sessions_table 中获取）
session_id = 756029989  # 替换为实际 ID

# 下载（首次需要下载 NWB 文件，约 2-3 GB，之后从缓存读取）
session = cache.get_session_data(session_id)

# 基本信息
print(f"Session ID: {session_id}")
print(f"总 units: {len(session.units)}")
print(f"Spike times 字典中的 unit 数: {len(session.spike_times)}")
print(f"刺激类型: {session.stimulus_presentations.stimulus_name.unique()}")
```

### 3.8 质量过滤（获取高质量 Units）

```python
# 默认过滤标准（AllenSDK 自动应用）：
# - presence_ratio > 0.95（unit 在整个 session 中稳定存在）
# - isi_violations < 0.5（不应期侵犯率低）
# - amplitude_cutoff < 0.1（spike 幅值完整）

# 默认加载时已过滤（filter_by_validity=True 是默认值）
units = session.units
print(f"默认过滤后 units: {len(units)}")

# 查看质量指标列
print(units[['presence_ratio', 'isi_violations', 'amplitude_cutoff',
             'ecephys_structure_acronym']].describe())

# 进一步筛选 V1 (VISp) 的 units（NeuroHorizon 多模态实验重点脑区）
visp_units = units[units['ecephys_structure_acronym'] == 'VISp']
print(f"VISp units: {len(visp_units)}")

# 若要加载所有 units（不过滤），可传入自定义参数
# session = cache.get_session_data(session_id,
#     unit_filter_kwargs={
#         'amplitude_cutoff_maximum': np.inf,
#         'presence_ratio_minimum': -np.inf,
#         'isi_violations_maximum': np.inf
#     })
```

### 3.9 提取刺激对齐的 Spike Counts（NeuroHorizon 核心）

```python
# ① 获取自然图像刺激呈现表
natural_scenes = session.get_stimulus_table('natural_scenes')
print(f"自然图像 presentations: {len(natural_scenes)}")
print(natural_scenes[['start_time', 'stop_time', 'frame']].head())
# 'frame' 字段是图像 ID（0-117），可用于加载对应的图像文件

# ② 获取对齐的 spike times（presentationwise）
spikes_df = session.presentationwise_spike_times(
    stimulus_presentation_ids=natural_scenes.index.values,
    unit_ids=visp_units.index.values
)
# spikes_df 包含：time_since_stimulus_onset, stimulus_presentation_id, unit_id

# ③ 获取 binned spike counts（直接用于 NeuroHorizon 的预测目标 Y）
time_step = 0.010  # 10ms bin
time_bins = np.arange(-0.05, 0.5 + time_step, time_step)  # -50ms 到 500ms

spike_counts = session.presentationwise_spike_counts(
    stimulus_presentation_ids=natural_scenes.index.values,
    bin_edges=time_bins,
    unit_ids=visp_units.index.values
)
# spike_counts 是 xarray DataArray，维度为 [时间, 刺激呈现, 单元]
print(f"Spike counts shape: {spike_counts.shape}")
# 例如: (56, 2311, 47) = (time_bins, presentations, units)

# ④ 将 spike counts 转为 numpy array（用于 NeuroHorizon 数据集）
counts_np = spike_counts.data  # shape: [n_timebins, n_presentations, n_units]
```

### 3.10 获取自然图像并与 DINOv2 对齐

对于 NeuroHorizon 的多模态实验，需要将视觉刺激图像与神经响应对齐：

```python
# AllenSDK 提供了访问刺激图像的接口
# 获取自然图像数据（需要单独下载 stimulus template）
natural_scenes_table = session.get_stimulus_table('natural_scenes')

# 访问刺激模板（图像像素数据）
template = session.get_stimulus_template('natural_scenes')
# template 是 numpy array，shape: [n_images, height, width]
print(f"图像模板 shape: {template.shape}")
# 例如: (118, 918, 1174) - 118张图像，灰度图

# 将 frame ID 映射到图像，并与 DINOv2 处理对齐
from torchvision import transforms
from PIL import Image
import torch

def preprocess_for_dinov2(gray_image):
    """将 Allen 灰度图转换为 DINOv2 可处理的 RGB 张量"""
    # 转为 RGB（重复灰度通道）
    rgb = np.stack([gray_image] * 3, axis=-1).astype(np.uint8)
    pil_img = Image.fromarray(rgb)

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(pil_img)

# 预提取所有 118 张图像的 DINOv2 embeddings（离线处理，节省训练时间）
# 建议在数据预处理阶段完成，存储为 .pt 文件
```

### 3.11 提取行为数据

```python
# 跑动速度（连续信号，对应 IBL 的 wheel velocity）
running_speed = session.running_speed
print(f"Running speed 时间点数: {len(running_speed)}")
# DataFrame with columns: timestamps, velocity

# 瞳孔跟踪数据
eye_tracking = session.eye_tracking
print(eye_tracking.columns.tolist())
# 包含：timestamps, pupil_area, pupil_width, pupil_height, corneal_reflection_*

# 获取刺激期间的平均跑动速度（逐 trial）
def get_running_speed_per_trial(session, stimulus_table, pre=0.0, post=0.5):
    """提取每个 trial 的平均跑动速度"""
    running = session.running_speed
    speeds = []
    for _, row in stimulus_table.iterrows():
        mask = ((running.timestamps >= row.start_time + pre) &
                (running.timestamps < row.start_time + post))
        speeds.append(running.velocity[mask].mean() if mask.any() else np.nan)
    return np.array(speeds)

trial_speeds = get_running_speed_per_trial(session, natural_scenes)
```

### 3.12 批量下载多个 Session

```python
# 分批下载 brain_observatory_1.1 的 session
bo_session_ids = sessions_table[
    sessions_table.session_type == 'brain_observatory_1.1'
].index.tolist()

print(f"准备下载 {len(bo_session_ids)} 个 session...")

for sid in bo_session_ids:
    print(f"下载 session {sid}...")
    try:
        session = cache.get_session_data(sid)
        # NWB 文件已缓存到本地 cache_dir，后续访问无需重新下载
        print(f"  ✓ units: {len(session.units)}, "
              f"stimulus types: {session.stimulus_presentations.stimulus_name.nunique()}")
    except Exception as e:
        print(f"  ✗ 下载失败: {e}")
```

### 3.13 Allen 数据集在 NeuroHorizon 中的适配注意事项

- **时间分辨率**：Allen 数据默认以秒为单位的连续时间戳，适合直接用于 spike-level tokenization（与 IBL 格式一致）。
- **刺激对齐**：`stimulus_presentations.start_time` 是 DINOv2 图像 embedding 注入时间点的参考坐标，建议在该时间点前后各取 50ms 作为缓冲。
- **IDEncoder 参考窗口**：建议使用 session 开始的前 5 分钟静息期（若有），或者使用 drifting gratings block 的平均响应作为参考。
- **多脑区分析**：Allen 数据同时记录多个脑区，可以按 `ecephys_structure_acronym` 分组，分别分析不同脑区对视觉刺激的响应。

---

## 4. NeuroHorizon 项目推荐工作流

### 4.1 开发阶段（Phase 1-2，Week 1-11）

```
Step 1: 从 IBL 下载 10-20 个 session（用于代码调试）
        → 覆盖 3-5 个不同脑区，确保探针位置多样性

Step 2: 从 Allen 下载 5-10 个 brain_observatory_1.1 session
        → 优先选择 unit 数量多（>300）的 session

Step 3: 为 IBL 和 Allen 分别实现数据适配器
        （继承 POYO 的数据加载基类）

Step 4: 完整 IBL 数据集（全部 459 sessions）→ 跨 session 泛化实验
Step 5: 完整 Allen 数据集（全部 58 sessions）→ 多模态融合实验
```

### 4.2 数据格式统一化建议

为了在 NeuroHorizon 中统一处理 IBL 和 Allen 数据，建议转换为以下通用格式：

```python
# 建议的统一数据格式（存储为 HDF5）
import h5py

def save_session_to_hdf5(filepath, spikes_times, spike_unit_ids,
                          unit_ids, unit_labels,
                          behavior_timestamps, behavior_values,
                          trial_start_times=None, stimulus_ids=None):
    """将单个 session 保存为统一的 HDF5 格式"""
    with h5py.File(filepath, 'w') as f:
        # Spike 数据（按时间排序）
        f.create_dataset('spikes/times', data=spikes_times)
        f.create_dataset('spikes/unit_ids', data=spike_unit_ids)

        # Unit 元数据
        f.create_dataset('units/unit_ids', data=unit_ids)
        f.create_dataset('units/quality_labels', data=unit_labels)

        # 行为数据
        f.create_dataset('behavior/timestamps', data=behavior_timestamps)
        f.create_dataset('behavior/values', data=behavior_values)

        # 可选：trial/刺激对齐信息
        if trial_start_times is not None:
            f.create_dataset('trials/start_times', data=trial_start_times)
        if stimulus_ids is not None:
            f.create_dataset('trials/stimulus_ids', data=stimulus_ids)
```

### 4.3 存储空间规划

| 数据集 | 建议下载量 | 预估空间 |
|--------|----------|---------|
| IBL（全量 spike + behavior，不含 LFP） | 459 sessions | ~100-200 GB |
| IBL（预处理后，转存 HDF5） | 全部 | ~50-100 GB |
| Allen NWB 文件（spike + behavior，不含 LFP） | 58 sessions | ~146.5 GB |
| Allen 图像 DINOv2 embeddings（预提取） | 118 张图像 × ViT-B | <1 GB |
| **合计** | | **~300-350 GB** |

### 4.4 关键参考资源

- **IBL ONE API 文档**：https://docs.internationalbrainlab.org/notebooks_external/data_download.html
- **IBL 加载 Spike Sorting 数据**：https://docs.internationalbrainlab.org/notebooks_external/loading_spikesorting_data.html
- **IBL 2025 数据发布说明**：https://docs.internationalbrainlab.org/notebooks_external/2025_data_release_brainwidemap.html
- **IBL AWS 开放数据**：https://registry.opendata.aws/ibl-brain-wide-map/
- **Allen SDK Neuropixels 文档**：https://allensdk.readthedocs.io/en/latest/visual_coding_neuropixels.html
- **Allen 数据访问教程**：https://allensdk.readthedocs.io/en/latest/_static/examples/nb/ecephys_data_access.html
- **Allen Session 分析教程**：https://allensdk.readthedocs.io/en/latest/_static/examples/nb/ecephys_session.html
- **Allen AWS 开放数据**：https://registry.opendata.aws/allen-brain-observatory/
- **POYO 代码库**（数据加载参考）：https://github.com/mehdiazabou/poyo-1

---

*本文档基于 2025年2月 的最新数据集版本整理，IBL 2025-Q3 新版本已包含更多 session。如有更新请参考官方文档。*

---

## 5. 数据集匹配度深度分析与执行注意事项

本节记录对 IBL 和 Allen 两个数据集与 NeuroHorizon 三大核心目标（长时程预测★★★、跨session泛化★★★、多模态融合★★☆）匹配情况的深度评估，以及在实际执行中需要特别处理的关键细节。

### 5.1 IBL 数据集匹配度

**总体评价：跨session泛化实验高度匹配；长时程预测存在一个需提前处理的细节问题。**

IBL 对跨session泛化实验是无可替代的选择——459个session、12个实验室、标准化行为变量，是测试IDEncoder泛化能力的理想环境，POYO/POYO+也在同一数据集上有过验证，可直接对比。对于data scaling law实验（10/50/100/200+ sessions），IBL是唯一真正满足规模需求的公开数据集，Allen的58个session根本不够。

**关键执行问题：IBL trial结构与1秒预测窗口的对齐策略**

IBL是行为驱动的变长trial，刺激呈现时间随对比度不同而变化（低对比度trial更长），trial之间有随机ITI。要从IBL数据构造1秒的"输入窗口→预测窗口"对，有两种可行策略：

- **策略A（以stimOn为锚点）**：取 `stimOn_times` 前后各500ms，输入窗口=`[stimOn - 500ms, stimOn]`，预测窗口=`[stimOn, stimOn + 500ms]`。优点：神经活动有明确的任务对齐；缺点：反应时间变长时预测窗口会延伸到decision/reward期，语义较杂。
- **策略B（连续截取，不依赖trial结构）**：将每个session的连续spike记录直接截取为1秒窗口（不对齐trial），在时间上滑动。优点：最大化数据利用率，不受trial结构限制；缺点：单个窗口的神经活动没有刺激标注，不能做多模态实验。

**建议**：对于长时程预测实验，优先使用策略B（连续截取），充分利用IBL的连续记录；跨session实验两种策略均可；多模态实验不使用IBL（无语义图像刺激）。

**IBL不适用的实验**：
- neural + image多模态实验（Gabor patch无语义信息，无法对齐DINOv2 embedding）

### 5.2 Allen Brain Observatory 匹配度

**总体评价：多模态实验高度匹配；对长时程预测的支持需要特别处理刺激呈现结构。**

Allen数据集对neural + image多模态实验是绝对首选，118张自然图像配合DINOv2是项目最干净的实验设计，多脑区同步记录适合跨脑区分析。

**关键执行问题1：Natural Scenes呈现时长与1秒预测窗口的错配**

Natural Scenes每张图像仅呈现250ms，之后有约500ms灰屏间隔。如果预测窗口是500ms-1000ms，会有相当比例落在灰屏期而非视觉响应期。这在实验中必须明确说明：**NeuroHorizon预测的是图像结束后的神经自发活动延续，而非另一张图像的视觉响应**。在论文中需要单独分析刺激结束后神经活动的衰减特性。

**建议**：Allen数据中更适合长时程预测实验的是 **Natural Movies** 刺激（约30秒连续视频，而非单张图像）。Natural Movies提供了真正连续的视觉刺激驱动的神经活动，不存在刺激间隔问题，可以直接截取1秒窗口进行预测。建议在执行时优先用Natural Movies验证长时程预测能力，再用Natural Scenes做多模态对齐实验。

**关键执行问题2：session数量与scaling实验的限制**

Allen只有58个session，提案中data scaling实验要测试"10/50/100/200+"个session的曲线，Allen连200+的基准都达不到。因此：
- scaling law实验必须在IBL数据上做
- Allen仅作为多模态和部分跨session实验的数据集

### 5.3 两个数据集组合的整体评估

| 目标 | IBL适配性 | Allen适配性 | 执行关键点 |
|------|---------|-----------|----------|
| 长时程预测（★★★） | ⚠️ 可行，但需处理trial结构 | ⚠️ Natural Scenes有250ms限制，推荐用Natural Movies | 建议用IBL连续截取 + Allen Natural Movies双轨验证 |
| 跨session泛化（★★★） | ✅ 最佳选择，459个session | ⚠️ 仅58个session，适合验证不适合scaling | scaling用IBL，细粒度泛化分析用Allen |
| 多模态融合（★★☆） | ❌ 仅Gabor，无语义图像 | ✅ 最佳选择，118张自然图像 | 完全依赖Allen，需提前确认DINOv2 embedding预处理 |
| Data Scaling Law | ✅ 唯一满足需求（459 sessions） | ❌ 58个session不够 | 必须用IBL |

### 5.4 对Jia Lab内部数据的建议

提案将Jia Lab数据列为长时程预测的★★★核心数据集，这一判断是正确的，因为：大量重复trials提供了统计可靠的ground truth PSTH，这是评估1秒预测准确性统计显著性的关键。IBL和Allen在这点上都无法替代——IBL每个刺激重复次数较少，Allen虽然每张图像重复50次但每次只有250ms。如果Jia Lab数据在短期内难以获取，建议优先在Allen的Natural Movies（连续刺激）上验证长时程预测的基本可行性，以避免整体开发时间线受阻。
