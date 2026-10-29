# 下层执行与交付入口（2026-10-15）

**2026-10-17最新：** [T1-O V2本机门控](handoffs/t1o_v2_gates_20261017/RESULT.md)，停止阶段C；只交事实，待上层核实。未执行Gate D或正式训练。


**2026-10-17最新交付：** [T1-L三臂种子0完整40k结果](handoffs/t1_l_option_20261015/RESULT.md)，480开发回合、112基线回合齐全；33模型仅SHA，完整独立证据ZIP约12MB随交付。N0/NΔ40k均16/16，回报81.3686/81.8640；B2为14/16、49.4190。仅事实交付，待上层分析。T1-L-S新支线按用户决定暂停，未启动。下文旧现场为历史。


**当前T1主线：**[T1 v4第2.1版执行单](../upper/run_orders/T1_FIX_FIRST_TRAIN_20261015.md)，科学`1d432c8aa0e26e31354b8d43771e2a45c8d1caf5`。在本机按单连续完成必要准入、六套强固定规则、2k校准、M0/M2各40k训练、确定性开发评估、20k只读快照和最终交付；**不因阶段性成绩下降停训或重复请示**。技术停机条件仍以执行单为准。

**最新核对现场：**[CURRENT.json](../CURRENT.json)截至2026-10-15 05:03 UTC为`t1_v4_admission_r21_continuing`、`training_active=false`，不能推断40k已开跑。后续状态以CURRENT和机器进程为准。待完成后交付目录`handoffs/t1_v4_s0_20261015/`（尚非已发布结果）。

**下层职责：**执行本机版本确认、训练/评估/监控/结果读出、检查点与失败样本保存、文件SHA及真实异常记录；常规进程、日志、路径问题按执行单自行解决。上层负责结果审查，**不把下层执行转回上层反复预筛选**；不擅自修改科学方法、场景或预算。参见[WORKFLOW](../WORKFLOW.md)。

## 历史证据入口

- [T1根因诊断](handoffs/t1_rootcause_20261013/RECEIPT.md)：旧M1失败机制。
- [T1 v2种子0结果](handoffs/t1_v2_seed0_20261013/REPORT.md)：32,088决策、开发块负结果。
- [T1 v2准入](handoffs/t1_v2_seed0_20261013/ADMISSION.md)：旧接口技术记录。
- [旧中断删除记录](handoffs/t1_round1_20261012/DELETION.md)：不复用。
- [F01已收口](handoffs/F01_closed_records_20261010/README.md)：METHOD_DOES_NOT_HOLD，历史。
- [历史窗口交接](handoffs/window_handoff_20261009/WINDOW_HANDOFF_20261009.md)及[DRL2整理回执](handoffs/DRL2_return_20261010/README.md)：旧阶段资料。

新交付放`lower/handoffs/<主题>_<日期>/`并更新本索引、CURRENT；大ZIP和模型可以留本机，登记SHA、路径和结构化审查证据。不得只提交最优检查点。

T1 v4第2.1版准入和校准通过：λ_mc=0.03、λ_eq=0.1，正式训练由用户PyCharm手动启动；[准入回执](handoffs/t1_v4_s0_20261015/admission/ADMISSION.md)、[手动命令](handoffs/t1_v4_s0_20261015/admission/PyCharm手动启动.md)。


## 2026-10-29 T2交付路径统一（已执行书不回写）

仅移动目录，所有交付文件Git blob逐项相同。旧路径可由保全标签恢复；现行入口使用新路径。

|原仓库根路径|现仓库根路径|
|---|---|
|`lower/handoffs/t2_2_test_20261022/`|`docs/collaboration/lower/handoffs/t2_2_test_20261022/`|
|`lower/handoffs/t2_3_test_20261023/`|`docs/collaboration/lower/handoffs/t2_3_test_20261023/`|
|`lower/handoffs/t2_4_test_20261025/`|`docs/collaboration/lower/handoffs/t2_4_test_20261025/`|
|`lower/handoffs/t2_5_1_validate_20261026/`|`docs/collaboration/lower/handoffs/t2_5_1_validate_20261026/`|
|`lower/handoffs/t2_5_validate_20261026/`|`docs/collaboration/lower/handoffs/t2_5_validate_20261026/`|
|`lower/handoffs/t2_6_pressure_20261027/`|`docs/collaboration/lower/handoffs/t2_6_pressure_20261027/`|
|`lower/handoffs/t2_6w_wide_20261027/`|`docs/collaboration/lower/handoffs/t2_6w_wide_20261027/`|

逐文件对照：[MOVED.csv](handoffs/git_cleanup_20261029/MOVED.csv)。已执行的upper/run_orders文件、交付包内SHA和正文历史路径均保留原样；旧报告中的绝对本机路径不作为现行入口。
