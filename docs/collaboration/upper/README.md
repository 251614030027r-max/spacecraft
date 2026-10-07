# 上层输入

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
