# 上层执行与审查入口（2026-10-18）

**上层职责：**方法代码实现、必要的单元/回归和短冒烟、真实动作生效检查、冻结SHA和下层执行单；下层交付后集中审查。**不默认由上层承担大量固定规则筛选、40k长训、开发评估或反复证明方法一定优于基线。**细则见[WORKFLOW](../WORKFLOW.md)。

## 当前执行单

**无有效执行单。** T1-P 候选 `8296c59` / `08f52b6` 未通过完整任务链（含追加窗口，用户已终止本轮）（[测试记录](T1_P_UPPER_TEST_RECORD_20261018.md)），待研究层决定。[T1_N_R2_ROUND1_20261018](run_orders/T1_N_R2_ROUND1_20261018.md) 已被 T1-P 取代，不得执行。

## 历史，不能替代当前执行单

- [T1_O_V2_GATES_LOCAL](run_orders/T1_O_V2_GATES_LOCAL_20261017.md)：第 2 版已执行，停于 Gate C；第 1 版未执行。
- [T1_L_S_TCPF](run_orders/T1_L_S_TCPF_20261016.md)：取消，未启动。
- [T1_L_OPTION_THREE_ARM](run_orders/T1_L_OPTION_THREE_ARM_20261015.md)：已完成（交付 `68680eb`，审查 `71d6b82`）。
- [T1_FIX_FIRST_TRAIN](run_orders/T1_FIX_FIRST_TRAIN_20261015.md)：第 2.1 版已完成（交付 `2242f1b`，审查 `110ce61`）。

- [T1_BETA_B012](run_orders/T1_BETA_B012_20261014.md)：B1 NO-GO，B2停止。
- [T1_ROOTCAUSE_DIAG](run_orders/T1_ROOTCAUSE_DIAG_20261013.md)：根因第一阶段已完成。
- [T1_V2_SEED0_VALIDATION](run_orders/T1_V2_SEED0_VALIDATION_20261013.md)：M1 32,088决策的历史单种子负结果。
- [T1_ROUND1_M0M1](run_orders/T1_ROUND1_M0M1_20261012.md)：旧接口中断，其余被取代。
- [FINAL_RERUN](run_orders/FINAL_RERUN_20261007.md)：旧F01已结束，METHOD_DOES_NOT_HOLD。
- [run_orders索引](run_orders/README.md)：其余旧单及历史记录。

新执行单使用[精简模板](TASK_TEMPLATE.md)，明确授权范围、科学SHA、必要测试、训练评估和交付；不增加无依据的前置性能门槛。
