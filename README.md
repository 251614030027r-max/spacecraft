# DRL2：六自由度翻滚非合作目标预捕获控制

本仓库实现高保真 SE(3) 六自由度预捕获控制：目标自由翻滚，真值传播包含中心引力、J2、引力梯度、二阶矩与刚体耦合，采用 RK45 积分，控制周期 0.1 s。

## 当前状态

截至 2026-09-16，T12 正式评估仍不支持“学习控制优于固定设定点”。转向后的非合作探针 2 也命中预签的“基本持平”停止分支：同一感知环境和 48 种子下，EKF 估计控制 Pure MPC 为 31/48，真值控制为 32/48（`ΔC=1`），不足以支撑探针 3 或训练。当前等待上层重新定义真正需要长期决策的观测/相位耦合问题，不得通过加噪声、缩视场或改任务人为制造 gap。正式结果见 `docs/PROBE2_NONCOOP_RESULT_20260916.md`；T12 结论见 `docs/T12_S10_FORMAL_EVALUATION_REPORT_20260915.md`。

## 阅读顺序

1. `CLAUDE.md`：研究纪律、系统边界和长期主线。
2. `docs/PROBE2_NONCOOP_RESULT_20260916.md`：非合作估计控制与真值控制的正式配对结果及停止裁决。
3. `docs/PRECAPTURE_NONCOOP_S1_AND_PROBES.md`：非合作 S1 接口、观测窗口与探针顺序。
4. `docs/T12_S10_FORMAL_EVALUATION_REPORT_20260915.md`：T12/S10 正式结果与证据边界。
5. `docs/T12_S10_EVAL_EXECUTION_ORDER.md`：T12 正式评估的预登记执行单与三分叉判据。
6. `docs/PERCEPTION_PRECAPTURE_INTEGRATION.md`：感知接入的阶段设计提案。
7. `docs/EVIDENCE_INDEX.md`、`docs/REPRODUCIBILITY.md`：证据和复现索引。
8. `docs/HISTORY.md`、`docs/handoffs/README.md`：被后续结果取代的演进与历史交接。

旧的多份交接、探针操作说明和逐轮阶段报告已整合进上述文件；原文仍可从 Git 历史恢复，不再作为当前入口。

## 只读验证

T12 启动前完整回归为：

```text
228 passed in 182.13s
```

T12 训练与两轮评估原始产物分别保留在本机 `logs/t12_train/`、`logs/t12_eval/` 和 `logs/t12_eval_v2/`；不得因本次精简入库而删除或覆盖。

## 目录

- `dynamics/`、`env/`：SE(3) 动力学、任务和环境；
- `controllers/`：MPC 与终端值实验实现；
- `train/`、`eval/`、`experiments/`：训练、统一评价和诊断工具；
- `tests/`：回归测试；
- `logs/`：主要审计证据，失败实验也保留；
- `models/`：本机未跟踪的历史最终模型；中间 checkpoint 已清理；
- `References/`：对标论文；
- `docs/`：当前状态、证据索引、复现方法、清理记录和历史 handoff 归档；
- `local_artifacts/patches/`：本机未跟踪的旧阶段补丁归档，不是当前执行入口。
