# 上层输入

**当前执行单（已授权待执行，2026-10-07）**：[最终主线干净重训与正式评估（合并版）](run_orders/FINAL_RERUN_20261007.md)。训练误中断后从零重训，种子 262460–262462，提交 `f2f8169`；合并了 FINAL_EVIDENCE；取代 FINAL_MAINLINE 尚未执行的部分。

**整理单（2026-10-10）**：[项目整体整理：现状分析、目标结构与分阶段执行单](run_orders/PROJECT_CLEANUP_20261010.md)。A 段只读盘点现在执行，与评估并行；B 段 Git 整理和 C 段本地整理，在评估交付、实验停止、用户确认后执行。取代 `ASSET_INVENTORY_20261007.md`。附件在 `cleanup/`。

**并行绘图单（已授权待执行，2026-10-09）**：[阶段汇报基准图，只用已有资产](run_orders/FIGURES_INTERIM_20261009.md)。不做新仿真，不碰训练；图 1 为 V3e 训练曲线，图 2 为 267000 基准对比，图 3 为新工况基线能耗与时间，图 4 为交接时机上界，图 5 为 266019 案例改标注。

**已被 PROJECT_CLEANUP_20261010 取代**：`ASSET_INVENTORY_20261007.md`（从未执行）。

**已被取代、不得再照单执行**：`FINAL_MAINLINE_20261006.md`（其 269000 筛选结果和正式判据被沿用）、`FINAL_EVIDENCE_20261007.md`（已并入 FINAL_RERUN）。

---

以下为历史索引。


**最新执行单（已授权待执行，2026-10-07）**：[只读资产盘点与证据链登记](run_orders/ASSET_INVENTORY_20261007.md)。不干扰训练，不删不移。

**补充执行单（暂缓下发，等盘点完成后由用户下发，内容不变）**：[论文闭环最小证据：nominal 正式行 + 第二正式块 272000 + 确定性重放](run_orders/FINAL_EVIDENCE_20261007.md)。A 部分（nominal 两块、Pure 272000）现在就可与训练并行；B 部分随正式评估；C 部分在全部评估完成后拉取分析工具提交 `ed8db6a`。只增加描述性证据，不新增门槛。

**主执行单（执行中）**：[最终主线：工况筛选 → 价值停止 → 3×60k → 正式评估](run_orders/FINAL_MAINLINE_20261006.md)。
- 科学提交 `e855bb7`；
- 筛选块 269000，结构检查块 266000（30k 检查点），正式块 271000；
- 训练种子 262450 / 262451 / 262452。

授权来源：用户 2026-10-06 原话写在执行单头部；执行范围以用户转达下层时的最新指令为准。

## 已结束的执行单（不得再次照单启动）

- [价值规则开发检验](run_orders/STOPPING_VALUE_RULE_CHECK_20261006.md)：758c9a8，268000 块，结论 `VALUE_RULE_DOES_NOT_HOLD`。上层结果记录见 [`docs/STOPPING_VALUE_RULE_RESULT_20261006.md`](https://github.com/251614030027r-max/spacecraft/blob/2336a3e/docs/STOPPING_VALUE_RULE_RESULT_20261006.md)。
- 停止头冻结单：[原文](https://github.com/251614030027r-max/spacecraft/blob/2e5c236f7412caea3b203519619ef7a48bb16df9/docs/STOPPING_RUN_ORDER_LOWER_20261002.md)，[预注册](https://github.com/251614030027r-max/spacecraft/blob/2e5c236f7412caea3b203519619ef7a48bb16df9/docs/STOPPING_METHOD_PREREGISTRATION_20261002.md)。结论 `METHOD_DOES_NOT_HOLD`。
- 上层审查：[STOPPING_FORMAL_REVIEW_20261006.md](https://github.com/251614030027r-max/spacecraft/blob/5e793f369fd65c9fff7a731bfdad190181dd1837/docs/STOPPING_FORMAL_REVIEW_20261006.md)；讨论材料：[DISCUSSION_BRIEF_20261006.md](https://github.com/251614030027r-max/spacecraft/blob/b42385f/docs/DISCUSSION_BRIEF_20261006.md)。

后续执行单直接新增到本分支 `upper/run_orders/`，更新本索引，再告知下层读固定协作入口。模板见 [TASK_TEMPLATE.md](TASK_TEMPLATE.md)。
