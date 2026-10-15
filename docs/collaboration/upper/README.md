# 上层执行与审查入口（2026-10-15）

**上层职责：**方法代码实现、必要的单元/回归和短冒烟、真实动作生效检查、冻结SHA和下层执行单；下层交付后集中审查。**不默认由上层承担大量固定规则筛选、40k长训、开发评估或反复证明方法一定优于基线。**细则见[WORKFLOW](../WORKFLOW.md)。

## 当前唯一有效T1执行单

[T1_FIX_FIRST_TRAIN_20261015.md](run_orders/T1_FIX_FIRST_TRAIN_20261015.md) **第2.1版**：科学`1d432c8`，G★ v0.6、接口v4（3维动作/25维观测），M0/M2各40k及2k校准，无成绩止损，20k只读快照。2.1版仅将零动作兼容判据从跨机证据改为本机旧版逐值比较；通过的检查可复用，不得重复启动。最新运行状态见[CURRENT](../CURRENT.json)，本入口不宣称训练已完成。旧第1版科学`1282464`未启动、已作废。

## 历史，不能替代当前执行单

- [T1_BETA_B012](run_orders/T1_BETA_B012_20261014.md)：B1 NO-GO，B2停止。
- [T1_ROOTCAUSE_DIAG](run_orders/T1_ROOTCAUSE_DIAG_20261013.md)：根因第一阶段已完成。
- [T1_V2_SEED0_VALIDATION](run_orders/T1_V2_SEED0_VALIDATION_20261013.md)：M1 32,088决策的历史单种子负结果。
- [T1_ROUND1_M0M1](run_orders/T1_ROUND1_M0M1_20261012.md)：旧接口中断，其余被取代。
- [FINAL_RERUN](run_orders/FINAL_RERUN_20261007.md)：旧F01已结束，METHOD_DOES_NOT_HOLD。
- [run_orders索引](run_orders/README.md)：其余旧单及历史记录。

新执行单使用[精简模板](TASK_TEMPLATE.md)，明确授权范围、科学SHA、必要测试、训练评估和交付；不增加无依据的前置性能门槛。
