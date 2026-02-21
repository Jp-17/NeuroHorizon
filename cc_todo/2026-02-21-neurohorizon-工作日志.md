# NeuroHorizon 工作日志

---

## 2026-02-21

### 完成的工作

#### 1. POYO 代码库分析 ✅
- 完成了对 POYO/POYO+ 代码架构的全面分析
- 理清了 Encoder-Processor-Decoder 架构、数据管线、训练流程、模态注册系统
- 识别了 NeuroHorizon 需要修改/替换的关键接口点
- 文档保存：`cc_todo/2026-02-21-poyo-代码分析.md`

#### 2. NeuroHorizon 项目评估与计划 ✅
- 完成了项目合理性评估，识别了 6 个主要问题及应对方案
- 制定了 6 阶段执行计划（Phase 0-5）
- 确认了最小可行发表路径 (MVP)
- 文档保存：`cc_todo/2026-02-21-neurohorizon-项目分析与执行计划.md`

#### 3. 关键决策确认 ✅
- GPU: 4090 单卡，后续可扩展
- Jia Lab 数据：确认不可用，使用 IBL + Allen Natural Movies 替代
- 执行顺序：数据管线优先 (Phase 0)

### 当前进行中

#### Phase 0.1: 环境扩展
- 目标：在 conda `poyo` 环境中安装 IBL/Allen 数据下载依赖
- 状态：开始执行

### 待完成
- Phase 0.2: IBL 数据下载脚本与预处理
- Phase 0.3: Allen 数据下载脚本与预处理
- Phase 0.4: 参考特征提取
- Phase 0.5: 数据验证
- Phase 1: POYO 基线验证
- Phase 2: NeuroHorizon 核心模型实现
- Phase 3: 多模态扩展
- Phase 4: 实验
- Phase 5: 分析与论文

### 遇到的问题
（暂无）

### 版本记录
| 日期 | 版本 | 描述 |
|------|------|------|
| 2026-02-21 | v0.1 | 项目分析完成，计划制定，开始执行 Phase 0 |

---
