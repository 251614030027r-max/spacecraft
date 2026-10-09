# 上层执行单索引

## 进行中

| 执行单 | 状态 |
|---|---|
| [FINAL_RERUN_20261007](run_orders/FINAL_RERUN_20261007.md) | F01 训练、F03–F05 评估进行中（代码 `f2f8169`）。F02 结构检查已通过 |
| [CLEANUP_B_LOWER_20261010](run_orders/CLEANUP_B_LOWER_20261010.md) | 第 1–3 步已完成：7 个标签、删除 6 个分支、流水线脚本入库。第 5 步验证等评估结束 |
| [PROJECT_CLEANUP_20261010](run_orders/PROJECT_CLEANUP_20261010.md) | A 段、B 段已完成；C 段本地归档等验证通过并经用户确认 |

## 已完成

| 执行单 | 结果 |
|---|---|
| [FIGURES_INTERIM_20261009](run_orders/FIGURES_INTERIM_20261009.md) | 阶段汇报图，交付在 `lower/handoffs/figures_interim_20261009/` |
| [STOPPING_VALUE_RULE_CHECK_20261006](run_orders/STOPPING_VALUE_RULE_CHECK_20261006.md) | 268000，VALUE_RULE_DOES_NOT_HOLD |

## 已被取代（不得照单执行）

- `FINAL_MAINLINE_20261006`：269000 筛选结果与正式判据被沿用，其余由 FINAL_RERUN 取代。
- `FINAL_EVIDENCE_20261007`：已并入 FINAL_RERUN。
- `ASSET_INVENTORY_20261007`：从未执行，由 PROJECT_CLEANUP 取代。

更早的执行单在科学分支的 `docs/archive/<阶段>/` 下，以及标签 `archive/pre-cleanup-20261010` 中。

新执行单放在 `run_orders/<主题>_<日期>.md`，按 [TASK_TEMPLATE.md](TASK_TEMPLATE.md) 填写，并更新本索引。只写已授权、要执行的内容；设想和讨论不写进来。
