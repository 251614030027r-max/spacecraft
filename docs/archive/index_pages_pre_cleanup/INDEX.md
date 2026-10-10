# 文档权威地图

> 2026-09-22 静态整理。范围为 `docs/` 当前全部 95 份 Markdown（81 份既有顶层文件、本文及 13 份 `handoffs/` 文件）。文件保持原位；本页只标状态，不改变证据路径。

状态只有三类：**LIVE** 当前有效且新窗口应优先读；**EVIDENCE** 已结案但仍支撑当前事实或审计链；**CLOSED** 历史研究线，保留但勿据此重开方向。发生冲突时，以 `../CLAUDE.md` 和最新 LIVE 文档为准。

## LIVE（当前有效，必读）

| 文件 | 用途 |
|---|---|
| `../CLAUDE.md` | 仓库入口、当前状态、科学边界与运行纪律。 |
| `README.md` | 最短阅读入口与最近证据清单。 |
| `INDEX.md` | 本权威地图。 |
| `ADAPTIVE_ROUND1_CLOSEOUT_20260921.md` | 第一轮收尾、R5 根因终判与证据边界。 |
| `ROUND1_EXTERNAL_REVIEW_20260922.md` | 第一轮外部审查件。 |
| `V3_LITERATURE_AND_COUPLING_DIRECTION_20260922.md` | 当前耦合设计方向与三篇核心文献。 |
| `V2_INTERFACE_IMPLEMENTATION_20260922.md` | V2 接口实现、架构地板与验证。 |
| `V2_RATE_LIMIT_DECISION_20260922.md` | 米单位参考步长上界 `0.40 m` 的最终裁定。 |
| `V2_TRAINING_ORDER_20260923.md` | V2 第一轮训练单；训练前步骤已完成，启动仍需用户明确同意。 |
| `ADAPTIVE_MAINLINE_RUNSHEET.md` | 当前主线运行框架；训练仍需用户新指令。 |
| `PURE_MPC_ROW_VERIFIED.md` | 冻结 Pure MPC 基线行。 |
| `REPRODUCIBILITY.md` | 当前环境与复现入口。 |
| `EVIDENCE_INDEX.md` | 日志、模型、评估与文献证据索引。 |

## EVIDENCE（已结案，保留审计链）

| 文件 | 证据角色 |
|---|---|
| `BIDIRECTIONAL_SACMPC_EXECUTION.md` | 基线锚定双向耦合的设计背景。 |
| `COUPLING_DEFECT_DIAGNOSIS_LOG_20260921.md` | D0–D4 诊断记录与可复核产物。 |
| `COUPLING_DEFECT_DIAGNOSIS_ORDER_20260921.md` | 第一轮缺陷诊断的预登记执行单。 |
| `DECK_CLEARING_ORDER_20260922.md` | V2 设计讨论前的零机时收尾边界。 |
| `FAST_TUMBLING_CAPTURE_MOTIVATION.md` | 快速翻滚任务动机与文献支撑。 |
| `REPO_TIDY_ORDER_20260922.md` | 本次“地图而非搬家”的整理授权与禁止项。 |
| `REWARD_UNITS_FIX_20260919.md` | 奖励量纲缺陷、修复与旧运行失效边界。 |
| `SAC_MPC_COUPLING_DESIGN.md` | SAC–MPC 耦合概念与不得越界的设计背景。 |
| `TRAINING_PREFLIGHT_AUDIT_20260919.md` | 第一轮训练前静态与闭环审计。 |
| `V2_EXECUTION_ORDER_20260922.md` | V2 接口实现预登记执行单。 |
| `V2_EXECUTION_SIGNAL_AUDIT_20260922.md` | V2 可用反馈信号的离线证据。 |
| `V2_FIX_ORDER_RATE_LIMIT_20260922.md` | 米单位限速缺陷修复单。 |
| `V2_RATE_LIMIT_SWEEP_STOP_20260922.md` | 四档扫描、探针纠偏、停止与最终裁决记录。 |
| `V2_REVIEW_RATE_LIMIT_20260922.md` | V2 米位移无界缺陷的复核记录。 |

## CLOSED（历史线，勿据此重开方向）

### 感知与早期分层线

| 文件 | 历史用途 |
|---|---|
| `A1_PERCEPTION_FOUNDATION_MANIFEST.md` | A1 感知地基；已关闭。 |
| `A2_GUIDANCE_FREE_MANIFEST.md` | A2 预登记；已关闭。 |
| `A2_GUIDANCE_FREE_RESULTS.md` | A2 结果；已关闭。 |
| `A3_DEPLOYABLE_PLANNING_P1_MANIFEST.md` | A3 P1 预登记；已关闭。 |
| `A3_DEPLOYABLE_PLANNING_P1_RESULTS.md` | A3 P1 结果；已关闭。 |
| `G0_PERCEPTION_CLOSED_LOOP_MANIFEST.md` | G0 感知闭环；已关闭。 |
| `PERCEPTION_PRECAPTURE_INTEGRATION.md` | 感知接入方案；不是当前 V2 执行入口。 |
| `PRECAPTURE_NONCOOP_S1_AND_PROBES.md` | 非合作观测 S1 与探针；已关闭。 |
| `PROBE2_NONCOOP_RESULT_20260916.md` | 非合作探针 2 证据；不得外推为当前 V2 性能。 |

### 早期 precapture / waypoint / 探针线

| 文件 | 历史用途 |
|---|---|
| `ATTITUDE_REFERENCE_PROBE.md` | 姿态参考假设证伪。 |
| `CLEANUP_20260909.md` | 2026-09-09 仓库清理记录。 |
| `D0_HYBRID_INFEASIBILITY_DIAGNOSIS_20260909.md` | 早期 hybrid MPC 不可行诊断。 |
| `INTERFACE_SINGLE_FACTOR.md` | 旧接口单因子结论；解释历史，不恢复旧接口。 |
| `LOWER_EXECUTION_PLAN_20260910.md` | 旧下层执行计划。 |
| `MAINLINE_TRAIN_EVAL_RUNSHEET.md` | 旧双向主线运行单，已被 adaptive/V2 主线取代。 |
| `OPPORTUNITY_MAINLINE_RUNSHEET.md` | 外层硬走廊机会任务，已明确废止。 |
| `P0_HORIZON_PERSISTENT_FAILURES.md` | P0 horizon 历史证据。 |
| `P1_A_COMMIT_TIME_AND_HOLD_RADIUS.md` | P1-A 扫描历史。 |
| `P1_B_APPROACH_RATE_SCAN.md` | P1-B 扫描历史。 |
| `P1_C_LATERAL_SCAN.md` | P1-C 扫描历史。 |
| `P2_FEEDBACK_SEPARABILITY.md` | P2 反馈可分性历史。 |
| `PRECAPTURE_ENTRY_WINDOW_GAP.md` | 旧 entry-window 缺口叙事。 |
| `PRECAPTURE_ENTRY_WINDOW_GAP_WITH_F1.md` | F1 后旧缺口复查。 |
| `PRECAPTURE_REFERENCE_TRACKING_DIAGNOSIS.md` | 3D waypoint 跟踪诊断。 |
| `PRECAPTURE_TASK_SPEC.md` | 历史任务规范；当前配置以代码与 LIVE 文档为准。 |
| `PRECAPTURE_V2_FASTTRACK_RESULTS.md` | 旧 precapture v2 快速线结果。 |
| `PRECAPTURE_V2_Q_AND_CORRECTIONS.md` | 旧 v2 结案与纠偏。 |
| `PRECAPTURE_V2_S3_RESULTS.md` | 旧 S3 开发结果。 |
| `PRECAPTURE_V2_S4_RESULTS.md` | 旧 S4 手工引导结果。 |
| `PROBE_RESULTS.md` | 四个历史 hybrid 探针。 |
| `TERMINAL_GATE_DEFECT.md` | 终端门缺陷历史。 |
| `TIMING_VALUE_FOUNDATION_HANDOFF_20260916.md` | timing-value foundation 交接，已关闭。 |
| `TIMING_VALUE_PROBE_SETUP.md` | timing-value A/B 探针设置，已关闭。 |
| `V4_ENTRY_CHANNEL_AND_ACCELERATION_20260910.md` | V4 entry-channel 诊断，已关闭。 |

### T0–T12 训练、闸门与评估线

| 文件 | 历史用途 |
|---|---|
| `PREFLIGHT_DELIVERY_20260911.md` | T0–T5 工程交付。 |
| `T0_AMENDMENT_20260910.md` | T0 事实基线修订。 |
| `T6_COUPLING_INTERFACE.md` | T6 旧 `arrival_condition` 接口。 |
| `T7_RESULTS_AND_VERDICT.md` | T7 训练结果。 |
| `T7_T9_REVIEW_20260912.md` | T7–T9 第三方审查说明。 |
| `T7_TRAINING_ORDER.md` | T7 训练单。 |
| `T8_EXECUTION_ORDER.md` | T8 评估与诊断单。 |
| `T8_GATE1_REPORT_20260912.md` | T8 闸门报告。 |
| `T10_EXECUTION_ORDER.md` | T10 执行单。 |
| `T10_GATE_A_REPORT_20260913.md` | T10 闸门 A 交付。 |
| `T10_GATE_A_RULING.md` | T10 闸门 A 裁决。 |
| `T10_HEADROOM_RESULT.md` | T10 headroom 结果。 |
| `T11_EXECUTION_ORDER.md` | T11 执行单。 |
| `T11_GATE_B_REPORT_20260913.md` | T11 闸门 B 交付。 |
| `T11_GATE_B_RULING.md` | T11 闸门 B 裁决。 |
| `T11_INTERFACE_CALIBRATION.md` | T11/S8 旧接口标定。 |
| `T12_EXECUTION_ORDER.md` | T12 训练单。 |
| `T12_LOCAL_RUNNING_HANDOFF_20260913.md` | T12 本地运行交接。 |
| `T12_S10_EVAL_EXECUTION_ORDER.md` | T12/S10 正式评估预登记。 |
| `T12_S10_FORMAL_EVALUATION_REPORT_20260915.md` | T12/S10 正式结果证据。 |
| `T12_S10_REVIEW_DOSSIER_20260915.md` | T12/S10 第三方评审卷宗。 |

### 仓库说明与历史交接

| 文件 | 历史用途 |
|---|---|
| `HISTORY.md` | 项目历史演进摘要。 |
| `handoffs/README.md` | 交接目录规则。 |
| `handoffs/archive_202609/HANDOFF.md` | 旧总交接。 |
| `handoffs/archive_202609/HANDOFF_ACTION_SPACE_FIX_20260908.md` | 旧动作空间修复交接。 |
| `handoffs/archive_202609/HANDOFF_COMMIT5B_RESOLVED_20260904.md` | 旧 commit 5B 交接。 |
| `handoffs/archive_202609/HANDOFF_LOWER_EXECUTION_20260905.md` | 旧下层执行交接。 |
| `handoffs/archive_202609/HANDOFF_LOWER_FASTTRACK_20260905.md` | 旧 fast-track 交接。 |
| `handoffs/archive_202609/HANDOFF_LOWER_S4_REWORK_20260905.md` | 旧 S4 重做交接。 |
| `handoffs/archive_202609/HANDOFF_TRAIN_NOW_20260907.md` | 旧训练启动交接，不再授权训练。 |
| `handoffs/archive_202609/HANDOFF_UPPER_RULING_20260908.md` | 旧上层裁决。 |
| `handoffs/archive_202609/HANDOFF_upper_v2_pivot.md` | 旧 v2 pivot 交接。 |
| `handoffs/archive_202609/NEXT_WINDOW_PROMPT_20260919.md` | 旧窗口提示。 |
| `handoffs/archive_202609/WINDOW_HANDOFF_20260913.md` | 旧 09-13 窗口交接。 |
| `handoffs/archive_202609/WINDOW_HANDOFF_20260919.md` | 旧 09-19 窗口交接。 |

## 使用规则

1. 新窗口先读 LIVE，不从 CLOSED 执行单恢复工作。
2. EVIDENCE/CLOSED 的路径与文件名是审计链的一部分，不移动、不改名、不因结论失效而删除。
3. 新事实优先更新现有 LIVE 入口；确需新增文档时，同一提交更新本索引。
