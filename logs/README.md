# 实验日志说明

日志是本工程的主要审计证据，因此失败实验也保留。正式run通常包含：

```text
manifest.json                 完整任务、算法、版本和执行状态
train.monitor.csv             episode级训练结果
phase2_diagnostics.json       Q、alpha、动作、Gate和失败类型
tensorboard/                  SB3训练标量
evaluations/                  固定seed周期deterministic评估
```

`phase2_sac_v2_*`属于早期受约束Phase-2探索；`phase2_mission_s1_*`记录两阶段任务、语义修正和单因素演进；两组`phase2_mission_s1v2_*`是当前最关键证据。具体阅读顺序见`docs/EXPERIMENT_INDEX.md`。8组更早的Phase-I完整日志已迁至`legacy/log_history/`，选定摘要另见`legacy/reference_baseline_evidence/`，不再混放于当前日志根目录。
