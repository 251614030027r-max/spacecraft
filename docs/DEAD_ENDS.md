# 已验证的负结果台账（死胡同）

**用途：** 以后任何方法、接口、奖励或任务改动，提出之前先在这里查。

**判断规则：**
- 如果提案落在某一条的"试了什么"范围内，必须说明"重开条件"里的哪一条已经变化。
- 说不出来，就不做。

**字段说明：**
- **证据**：完整相对路径，从仓库根目录算起。标 `@tag` 的，表示只能在冻结标签 `archive/pre-cleanup-20261010` 下查到，例如整理前的 CLAUDE.md。
- **根因**：分为"实测"（有专门实验证实）和"推断"（由结果反推，未单独验证）。

下层盘点的本地记录在协作分支 `lower/handoffs/cleanup_inventory_20261010/tried_variants.csv`，共 288 条：其中 120 条是有运行资产的实际尝试，168 条是历史证据族索引（指向报告里的段落，不等于独立实验）。文字里"提议"和"实际执行"的歧义，保留在该 CSV 的出处列中。能确定结论的条目已经并入下面各条。

---

## 一、学习方式与训练设置

| ID | 试了什么 | 结果 | 根因 | 证据 | 什么条件变了才值得重开 |
|---|---|---|---|---|---|
| D01 | 纯 SAC 端到端控制（单阶段任务，不带 MPC） | 三个种子完成 3/1/0，零违规回合 20/10/0（两者是不同统计） | 实测：学习策略无法同时满足约束和完成任务 | CLAUDE.md "Historical research line" @tag | 不重开。它只用来说明为什么需要约束 MPC 层 |
| D02 | SAC 自动熵温度 | α 在每个运行中单调上升；校准误差随 α 单调变大，两种观测方案下一致 | 实测：温度推高评论家的高估 | 同上 @tag | 只有在同时监控评论家校准时才考虑。此后熵系数固定为 0.005 |
| D03 | 悬停吸引子：奖励下"同步旋转但不收尾"可拿到大部分回报 | 旧任务上悬停可拿到脚本回报的 73%；T7 有两个种子（262201、262301）超时率从约 31% 升到约 69% | 实测：折扣下的完成奖励在开局时价值很小 | `docs/archive/02_coupling_interface_T6_T12/T7_RESULTS_AND_VERDICT.md` | 一直有效的风险：凡是改奖励、改时域、改折扣，都要看"超时增多而回报上升"这个信号 |

## 二、接口与动作表达

| ID | 试了什么 | 结果 | 根因 | 证据 | 什么条件变了才值得重开 |
|---|---|---|---|---|---|
| D04 | 4 维 `radial_local` 参数化 | 共同预算 25,697 个决策时，三个种子 0/1308 回合完成；2 维接口最好的种子 55/285 | 实测，单因素：表达能力是瓶颈，信息不是 | `docs/archive/02_coupling_interface_T6_T12/INTERFACE_SINGLE_FACTOR.md` | 不重开 |
| D05 | 把执行反馈（回退率、松弛、执行器占用）放进观测（T7 V3 对 V2） | 不支持，点估计反而不利（每组只有 1 个学会的种子） | 推断：样本不足以下结论 | `docs/archive/02_coupling_interface_T6_T12/T7_RESULTS_AND_VERDICT.md` | 事实：V3e 主线仍开着 `execution_feedback`，它的作用从未被正式验证（见 O4） |
| D06 | 第一轮残差 `a_commit = clip(1 + 2r)` | 有效介入仅 0.114% 和 0.484%，训练到 60k 降为 0；耦合从未真正起作用 | 实测（R5）：r ≥ 0 全部映射到同一个参考，梯度恒为 0；承诺棘轮把决策压缩到第 0 步 | `docs/archive/03_adaptive_round1_v2/ADAPTIVE_ROUND1_CLOSEOUT_20260921.md`、`COUPLING_DEFECT_DIAGNOSIS_*` | 不重开。教训：任何新接口都要先测"动作通道是否有效"和梯度是否为零 |
| D07 | V3 接口（每步参考 ≤ 0.40 m 加近场下限）下的学习任务策略 | 学习策略 ≈ 脚本 nominal（262000 块上均值 37.7 对 40）；最快的选项就是平滑的 nominal，约为 Pure 用时的 2 倍 | 实测：接口本身封顶了学习层的速度 | `docs/archive/04_v3_value_arbitration/V3E_STATE_REVIEW_20260930.md` | 事实：每步参考上限决定了学习层速度的上界；不改这个上限，学习层不可能比 nominal 快 |

## 三、任务设计

| ID | 试了什么 | 结果 | 根因 | 证据 | 什么条件变了才值得重开 |
|---|---|---|---|---|---|
| D08 | 外层惯性接近走廊（opportunity 任务） | Pure MPC 违反新的硬约束，2 回合烟测 0/2 合法 | 设计层面：属于人为制造差距 | CLAUDE.md "superseded" @tag、`docs/archive/02_coupling_interface_T6_T12/OPPORTUNITY_MAINLINE_RUNSHEET.md` | 不重开 |
| D09 | 相位门、有利度合法性门等阈值类任务规则 | 2026-09-18 按设计否决 | 设计层面：用阈值代替连续代价 | CLAUDE.md "Direction discipline" @tag | 不重开 |
| D10 | 用"翻滚越快、分段进入越占优"的工况层面叙事 | 实测不成立：一律分段进入退化得更快，在 2.36 °/s 和 3.54 °/s 时最差真值裕度为负 | 实测 | `docs/archive/03_adaptive_round1_v2/ROUND1_EXTERNAL_REVIEW_20260922.md` | 不重开 |
| D11 | 更快的翻滚工况（3.0、3.5 °/s） | Pure 在这些格子里因距离超限失败，而不是超时，不满足筛选规则 | 实测 | `docs/archive/05_handoff_stopping_final/REGIME_SCREEN_20261006.md`、协作分支的筛选读数 | 只有当失败模式变成超时（例如初速度或距离设置改变）时才重新考虑 |
| D12 | 非合作估计（EKF 对真值）作为差距来源 | Pure 用估计值 31/48，用真值 32/48，基本持平 | 实测 | `docs/archive/02_coupling_interface_T6_T12/PROBE2_NONCOOP_RESULT_20260916.md` | 只有改用更差的传感器模型时才重开 |

## 四、Pure MPC 侧的改造（全部关闭）

| ID | 试了什么 | 结果 | 根因 | 证据 | 什么条件变了才值得重开 |
|---|---|---|---|---|---|
| D13 | 修改姿态参考模式（aimed、frozen、swept）以救回 262006 | 三种全部失败，aimed 失败得更早 | 实测：指向能力要从平移中借，而同步旋转已经用满三轴能力 | `docs/archive/01_precapture_task_pure_mpc/ATTITUDE_REFERENCE_PROBE.md` | 不重开 |
| D14 | 4 类位置决策（承诺时间、保持半径、接近速率、横向偏移）以救回 262006 | 约 90 个扫描格全部失败，死于视场约束 | 实测：属于单个开局的局限 | `docs/archive/01_precapture_task_pure_mpc/P1_*` | 不重开 |
| D15 | 学到的终端代价（AC4MPC 式），以及并行双解 | 终端代价实测有害；双解需要 1.5 倍计算预算 | 实测 / 预算 | CLAUDE.md 参考文献表 @tag、`docs/archive/02_coupling_interface_T6_T12/T7_T9_REVIEW_20260912.md` | 不重开 |
| D16 | 在线调整 MPC 代价权重（上海交大的方法） | 按设计排除 | 设计层面 | CLAUDE.md 参考文献表 @tag | 不重开 |

## 五、上层调度与协调

| ID | 试了什么 | 结果 | 根因 | 证据 | 什么条件变了才值得重开 |
|---|---|---|---|---|---|
| D17 | 脚本化"先保持、再承诺"的择时启发式 | B 17/35 对 A 26/35，入口相位的有利度也没有提高 | 实测（固定启发式） | `docs/archive/02_coupling_interface_T6_T12/MAINLINE_TRAIN_EVAL_RUNSHEET.md`、`TIMING_VALUE_*` | 固定启发式不重开 |
| D18 | 学习层无条件改写参考（T12） | 三个 arrival 种子平均 26/48，Pure 32/48；更慢，也更费燃料 | 实测 | `docs/archive/02_coupling_interface_T6_T12/T12_S10_FORMAL_EVALUATION_REPORT_20260915.md` | 不重开。由此得出：必须保留回到 Pure MPC 的路径 |
| D19 | V3 策略级价值仲裁（拟合 V_L、V_B，初始选择加单向交回） | 第二轮不成立：只有 262420 通过 M6，其仲裁行 40/48 且有 1 次违规 | 实测：价值拟合的区分力不足 | `docs/archive/04_v3_value_arbitration/V3E_VALUE_R2_ANALYSIS_20260930.md`、`V3E_STATE_REVIEW_20260930.md` | 不重开"两个连续价值择优"的形式 |
| D20 | 状态分类器 p_M(s) 加精度阈值决定交接（阶段 C） | 折外加权精度最高 0.89，低于 0.95，C_STOP；k = 0 时 AUC 0.45，k ≥ 1 时 0.79 | 实测：开局时刻无法从状态区分 Pure 能否成功 | `docs/archive/05_handoff_stopping_final/STAGE_C_RESULT_20261002.md` | 不重开分类器加阈值的形式 |
| D21 | 固定时刻交接 | 无增益 | 实测（探索性） | `docs/archive/05_handoff_stopping_final/STAGE_C_RESULT_20261002.md` | 不重开 |
| D22 | 独立的伯努利停止头（离散软 AC），在原工况上 | 正式轮 267000：Pure 41；stopping 37/41/36；交接增益 +25/+10/+19；0 违规。停止头与自身价值不一致：127 次交接中 59 次在 Q_H < Q_C 时触发 | 实测 | `docs/archive/05_handoff_stopping_final/STOPPING_FORMAL_REVIEW_20261006.md` | 不重开独立停止头，已由价值规则取代 |
| D23 | 价值规则作用于停止头训练出的冻结模型（原工况） | 268000：Pure 39，价值规则 36/38/38；毁掉 5/2/2，救回 2/1/1 | 结果为实测；"原工况 Pure 接近饱和、交接时刻难以由状态辨识"是推断，依据是阶段 C 与正式轮的关联证据，没有专门诊断 | `docs/archive/05_handoff_stopping_final/STOPPING_VALUE_RULE_RESULT_20261006.md` | 已转为在新工况上重训（F01） |
| D24 | 停止目标中的确定性 V_C 与软继续项混用（`STOPPING_VALUE_20261006`） | 不是失败，是一处目标不一致，已修正为 `bellman_stop_value="soft"` | 设计修正 | `train/stopping.py` | — |

---

## 未结问题（尚无结论，只列已知事实）

| ID | 问题 | 已知事实 | 证据 |
|---|---|---|---|
| O1 | 和交接选项一起训练时，学习策略单飞能力不提升 | V3e 单独训练时，完成率从约 25% 升到约 75%。和交接一起训练时（final2），未交接回合的完成率在 262460 上约 30% 持平，在 262462 上约 50% 持平，在 262461 上为 0。上一轮正式评估单飞为 12/31/17 | `evidence/E01_v3e_60k/final_training_audit_60k.json`、协作分支 `ppt_temporary_20261009/plot_data/` |
| O2 | 交接时刻能否从状态辨识 | 事后最优交接 48/48 对 Pure 37（262000）；但分类器（D20）和价值估计（D22、D23）都不足以辨识 | `evidence/E02_E03_stage_b/readout_b1b2.json` |
| O3 | 原工况下 Pure 是否饱和 | Pure 在 267000 上 41/48、268000 上 39/48；新工况 Pure 34/48 | 筛选读数 |
| O4 | 执行反馈观测是否必要 | 见 D05 | — |
| O5 | 训练中会产生非法穿越 | 是诊断计数，不是违规；但训练期占比高（最近 50 回合中 31–33 回合出现） | 训练快照 |

F01 正式评估交付后，用 ε_H、ε_C 的实测值更新 O1 和 O2 的事实栏。
