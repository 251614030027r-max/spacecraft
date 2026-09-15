# DRL2：六自由度翻滚非合作目标预捕获控制

本仓库实现高保真 SE(3) 六自由度预捕获控制：目标自由翻滚，真值传播包含中心引力、J2、引力梯度、二阶矩与刚体耦合，采用 RK45 积分，控制周期 0.1 s。

## 当前状态

截至 2026-09-15，T12 六个 SAC-MPC 模型已完成训练和固定 48 种子正式评估。预登记判据落入分叉 2：`arrival_condition` 相对 `radial_local` 显著改善可学性，但未稳定超过 Pure MPC，且成功时间和燃料更高；当前不支持“学习控制优于固定设定点”。训练与评估均使用完整真值状态 `perception=None`，不能写成局部视觉/EKF 结果。正式结论与审查边界见 `docs/T12_S10_FORMAL_EVALUATION_REPORT_20260915.md`，精简机器证据见 `eval/results/t12_s10_20260915/`。

## 阅读顺序

1. `CLAUDE.md`：研究纪律、系统边界和长期主线。
2. `docs/T12_S10_FORMAL_EVALUATION_REPORT_20260915.md`：T12/S10 正式结果、证据边界与上层待裁决项。
3. `docs/T12_S10_EVAL_EXECUTION_ORDER.md`：本轮正式评估的预登记执行单与三分叉判据。
4. `docs/PERCEPTION_PRECAPTURE_INTEGRATION.md`：未来感知接入提案，尚未执行。
5. `docs/T12_LOCAL_RUNNING_HANDOFF_20260913.md`、`docs/T12_EXECUTION_ORDER.md`：训练期事实与历史执行单。
6. `docs/T11_GATE_B_RULING.md`、`docs/T11_INTERFACE_CALIBRATION.md`：闸门 B 裁决与接口校准。
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
