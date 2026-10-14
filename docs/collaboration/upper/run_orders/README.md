# 执行单存放位置

上层新增 `<主题>_<日期>.md`，使用 `../TASK_TEMPLATE.md`。已执行的文件保持不变；修订时另建新文件，并在文件中写明替代了哪一份。

| 文件 | 状态 |
|---|---|
| `T1_BETA_B012_20261014.md` | 已结束：B0 通过；B1 NO-GO（科学 32f5419）；B2 不运行，β 路线停止 |
| `T1_ROOTCAUSE_DIAG_20261013.md` | 第一阶段已执行（collab a5395ef），报告见 reviews/T1_ROOTCAUSE_DIAG_REPORT_20261013.md；原：根因诊断第一阶段（P1 干预、P2/P3 Critic 排序与 M1 目标；不训练；科学 ca82c63；约 5–6 CPU·h，试跑超 6.5 先上报） |
| `T1_V2_SEED0_VALIDATION_20261013.md` | 已执行（32088 步平台期停止），已授权待执行：M1 种子 0 单条开发验证（提交 78eb63c，20k–60k，平台期自动停止） |
| `T1_ROUND1_M0M1_20261012.md` | 准入与种子 0 已执行；种子 0 中断（collab 709a3ec）；其余部分由 T1_V2_SEED0_VALIDATION 取代 |
| `FINAL_RERUN_20261007.md` | F01 已结束（用户判失败），只作历史（提交 f2f8169，种子 262460–262462） |
| `ASSET_INVENTORY_20261007.md` | 已授权，FINAL_RERUN 交付后执行 |
| `FINAL_EVIDENCE_20261007.md` | 已并入 FINAL_RERUN，不再单独执行；原（补充 FINAL_MAINLINE；A/B 用冻结提交，C 用分析提交 ed8db6a） |
| `FINAL_MAINLINE_20261006.md` | 筛选已完成；训练被误中断；其余部分由 FINAL_RERUN 取代 |
| `STOPPING_VALUE_RULE_CHECK_20261006.md` | 已执行，结论 VALUE_RULE_DOES_NOT_HOLD |
