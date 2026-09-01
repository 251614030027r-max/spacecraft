# 实验日志说明

日志是本工程的主要审计证据，失败实验也保留。正式训练目录通常包含：

```text
manifest.json
train.monitor.csv
phase2_diagnostics.json
tensorboard/
evaluations/
```

当前证据阅读顺序和模型对应关系见 `docs/EVIDENCE_INDEX.md`。根目录训练日志按运行名称保留，不因结论为负而删除。

`invalidated/observation_velocity_rotation_bug/` 保存观测偏置错误旋转惯性速度时生成的网格输出。它们只用于复查错误机制，不得支持当前性能主张。修正后的观测误差结论见 `docs/PROBE_RESULTS.md`。
