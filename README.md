# DRL2：六自由度翻滚非合作目标预捕获控制

本仓库实现高保真 SE(3) 六自由度预捕获控制：目标自由翻滚，真值传播包含中心引力、J2、引力梯度、二阶矩与刚体耦合，采用 RK45 积分，控制周期 0.1 s。

## 当前状态

当前执行的是 `radial_local`、h20、2 s 高层决策、宏观势函数 shaping 的三种子 v3 训练，训练基线提交为 `1655b0b`。D0 已证明 QP 不可行和零 wrench 级联是真实失败通道，但不能解释全部失败；求解器回退与 slack-limit 两条补救均已由上层测为负结果，不再调整。当前仍是 `precapture_planning_full_state_v1_24d`、`perception=null`，只能形成 full-state 调度机制证据，不能写成局部视觉/EKF 结果。

## 阅读顺序

1. `CLAUDE.md`：研究主线、正式结果、纪律和陷阱。
2. `HANDOFF_UPPER_RULING_20260908.md`：当前上层裁决与禁止事项。
3. `docs/D0_HYBRID_INFEASIBILITY_DIAGNOSIS_20260909.md`：80 局无训练失败归因。
4. `docs/SAC_MPC_COUPLING_DESIGN.md`：耦合设计、动作空间与 D1 宏观势函数。
5. `docs/PRECAPTURE_ENTRY_WINDOW_GAP.md`：入口窗口与参考选择缺口。
6. `docs/EVIDENCE_INDEX.md`：日志、模型和论文证据索引。
7. `docs/HISTORY.md`：已被后续结果取代的演进摘要。
8. `docs/handoffs/README.md`：历史 handoff 归档说明。

旧的多份交接、探针操作说明和逐轮阶段报告已整合进上述文件；原文仍可从 Git 历史恢复，不再作为当前入口。

## 只读验证

仓库虚拟环境当前可用。最近一次完整回归为：

```text
199 passed in 103.05s
```

当前活动训练目录为 `logs/hybrid/sac_mpc_hybrid_v3_macro_262100`、`262101`、`262102`，整理或清理时不得触碰。

## 目录

- `dynamics/`、`env/`：SE(3) 动力学、任务和环境；
- `controllers/`：MPC 与终端值实验实现；
- `train/`、`eval/`、`experiments/`：训练、统一评价和诊断工具；
- `tests/`：回归测试；
- `logs/`：主要审计证据，失败实验也保留；
- `models/`：本机未跟踪的历史最终模型；中间 checkpoint 已清理；
- `References/`：对标论文；
- `docs/`：当前状态、证据索引、复现方法、清理记录和历史 handoff 归档。
