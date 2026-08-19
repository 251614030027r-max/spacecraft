# DRL2：六自由度翻滚目标预捕获研究工程

本工程研究高保真六自由度 SE(3) 翻滚目标交会中的两阶段预捕获控制。truth propagation包含J2、刚体耦合动力学和RK45积分；当前学习主线是Pure SAC，MPC保留为独立约束控制基线。工程尚未得到满足独立种子泛化要求的合格S1策略，现有模型是诊断证据，不是成功结论。

审阅建议依次阅读：

1. `docs/PROJECT_OVERVIEW.md`：问题、系统结构和证据边界；
2. `docs/DEVELOPMENT_HISTORY.md`：从初始复现到当前瓶颈的演进；
3. `docs/CURRENT_PHASE2_BASELINE.md`：当前冻结任务、算法和结论；
4. `docs/EXPERIMENT_INDEX.md`：日志、代表模型和历史材料索引；
5. `docs/REPRODUCIBILITY.md`：环境重建、测试、训练与评估入口。
6. `docs/CLEANUP_MANIFEST_20260817.md`：导师移交整理中的删除、迁移和保留边界。

活动源码仅位于`dynamics/`、`env/`、`controllers/`、`train/`、`eval/`和`experiments/`；`legacy/`只保存早期复现与模型历史，不参与当前import。本次整理回归基线为96项通过。`.venv`按用户要求保留用于本机复核，但其解释器路径不可移植，其他机器应按依赖清单重建。

`D:\py\DRL2`是当前工程；`D:\py\DRL`不是本工程的一部分，也未在本次整理中修改。
