# 工程整理清单（2026-08-17）

## 目标与边界

本次整理面向导师整体审阅：活动源码、当前任务定义、关键实验演进与原始证据必须可追溯，同时减少根目录杂项和“历史模型仍像当前候选”的歧义。`D:\py\DRL2`是唯一整理对象；`D:\py\DRL`未读取后写入，也未修改。未生成新的审查ZIP。

## 已永久删除

- 根目录旧审查包`PHASE2_S1_TAU001_SINGLE_FACTOR_UPPER_REVIEW_20260815.zip`：内容可由保留的源码、文档和日志重新整理；
- `.idea/`：本机IDE设置；
- `legacy/old_source/`：与活动源码重复的旧代码快照；
- 工程源码树中的8个`__pycache__`/pytest缓存目录。

这些内容不在回收站中；研究日志、模型、当前源码、历史说明和`archive/`均未永久删除。

## 已移动但未删除

- 11组旧S1单因素模型由`models/`迁至`legacy/model_history/`；
- 两组S1-v2主线中除代表checkpoint以外的18个周期checkpoint迁至同一历史目录；
- 8组更早的Phase-I完整日志由`logs/`迁至`legacy/log_history/`；
- 详细阶段分析文档集中到`docs/history/`，内容保持原始实验口径。

移动后的历史资产仍可审计，但不应被解释为当前候选模型。历史报告中的旧路径可能保留当时写法，实际位置以`docs/EXPERIMENT_INDEX.md`为准。

## 当前保留

- `.venv/`：按用户要求保留；Python 3.12.6，本机验证有效，约4.86 GiB，但绑定本机基础解释器路径且不可移植；
- `models/`：两组S1-v2运行的4个代表模型，用于空间接近、速度控制与后期遗忘对照；
- `logs/`：Phase-2消融、正式S1单因素和S1-v2全部审计日志；
- `docs/history/`、`legacy/documents/`、`legacy/reference_baseline_evidence/`、`legacy/reference_model/`、`legacy/model_history/`、`legacy/log_history/`：历史追溯材料；
- `archive/`：安全审查认为仍可能包含可追溯资料，本轮保留。

## 验证状态

整理完成后使用保留的`.venv`执行`python -m pytest -q`，结果为`96 passed in 18.25s`。整理操作未修改活动Python源码。当前仍无通过独立20-seed Gate acquisition的合格S1模型，不能进入S2；4个活动模型仅是诊断证据。
