# 执行单存放位置

上层新增 `<主题>_<日期>.md`，使用 `../TASK_TEMPLATE.md`。已执行的文件保持不变；修订时另建新文件，并在文件中写明替代了哪一份。

| 文件 | 状态 |
|---|---|
| `T1_O_V2_GATES_LOCAL_20261017.md` | **第 1 版，唯一有效**，已授权（用户 2026-10-17）：下层本机复现 Gate A，再依次跑 Gate B / C（含零步）/ E，遇 FAIL 即停，交上层核实；不含 Gate D 试学与任何正式训练；科学 `d3902a9`（分支 `experiment/t1_o_tcpf_20261017`），工况 P2，λ_f = 20 |
| `T1_L_S_TCPF_20261016.md` | **取消，未启动**（研究层 2026-10-17，由 T1-O V2 取代）。原：T1-L-S 任务一致性预测反馈 = 审议区间价值目标 + 原 MPC 相对项；三臂 sb2 / sn0 / sndelta 各 40k，种子 0；训练器 A/B 由预注册规则（`experiments/t1_l_s_trainer_choice.py`，读旧 N0）二选一；n_envs 与调度由下层实测决定、三臂统一；16 开局全曲线 + 64 开局 0/20k/40k + T1-R 64 开局 0/40k；独立克隆 `D:\py\DRL2_T1_L_S`；科学 `672ba79` |
| `T1_L_OPTION_THREE_ARM_20261015.md` | **第 2 版，唯一有效**，已授权（用户 2026-10-15），与在训 v4 并行的支线：T1-L 三臂 B2 / N0 / NΔ 各 40k，种子 0；NΔ 相对项权重按梯度一次校准（5%，上限 10%）；规则评估结束后才训练；独立克隆 `D:\py\DRL2_T1_L`，Idle 优先级；科学 `166244b`。第 1 版（`da9be02`）未启动，作废 |
| `T1_FIX_FIRST_TRAIN_20261015.md` | **第 2.1 版，唯一有效**（2.1：准入 C 改为同机旧版逐值比较，回应 872bc96），已授权，下层自主一次性执行到交付：接口 v4（SAC 选新计划入口速度）+ G★ v0.6 + 共用 critic 修复；λ 校准第 2 版；M0/M2 种子 0 各 40k，无成绩止损；科学 1d432c8。第 1 版（7456db8，科学 1282464，未启动）作废 |
| `T1_BETA_B012_20261014.md` | 已结束：B0 通过；B1 NO-GO（科学 32f5419）；B2 不运行，β 路线停止 |
| `T1_ROOTCAUSE_DIAG_20261013.md` | 第一阶段已执行（collab a5395ef），报告见 reviews/T1_ROOTCAUSE_DIAG_REPORT_20261013.md；原：根因诊断第一阶段（P1 干预、P2/P3 Critic 排序与 M1 目标；不训练；科学 ca82c63；约 5–6 CPU·h，试跑超 6.5 先上报） |
| `T1_V2_SEED0_VALIDATION_20261013.md` | 已执行（32088 步平台期停止），已授权待执行：M1 种子 0 单条开发验证（提交 78eb63c，20k–60k，平台期自动停止） |
| `T1_ROUND1_M0M1_20261012.md` | 准入与种子 0 已执行；种子 0 中断（collab 709a3ec）；其余部分由 T1_V2_SEED0_VALIDATION 取代 |
| `FINAL_RERUN_20261007.md` | F01 已结束（用户判失败），只作历史（提交 f2f8169，种子 262460–262462） |
| `ASSET_INVENTORY_20261007.md` | 已授权，FINAL_RERUN 交付后执行 |
| `FINAL_EVIDENCE_20261007.md` | 已并入 FINAL_RERUN，不再单独执行；原（补充 FINAL_MAINLINE；A/B 用冻结提交，C 用分析提交 ed8db6a） |
| `FINAL_MAINLINE_20261006.md` | 筛选已完成；训练被误中断；其余部分由 FINAL_RERUN 取代 |
| `STOPPING_VALUE_RULE_CHECK_20261006.md` | 已执行，结论 VALUE_RULE_DOES_NOT_HOLD |
