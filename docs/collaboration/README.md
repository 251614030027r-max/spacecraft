# Spacecraft 上下层协作入口

新窗口必读：[HANDOFF.md](HANDOFF.md)（当前研究及版本）与[研究演进](RESEARCH_EVOLUTION_AND_COUPLING_DESIGN_20261012.md)（长期研究逻辑）；工程事实按需核验[PROJECT_STATE](PROJECT_STATE.md)与[CURRENT](CURRENT.json)。**运行实验只用有效执行单指定的冻结科学提交，不能用`collab/spacecraft`上的旧代码。**

| 用途 | 入口 |
|---|---|
| 两份研究交接 | [HANDOFF](HANDOFF.md) · [研究演进](RESEARCH_EVOLUTION_AND_COUPLING_DESIGN_20261012.md) |
| 上层科学及工程状态 | [PROJECT_STATE](PROJECT_STATE.md) · [upper](upper/README.md) |
| 下层真实现场与交付 | [CURRENT](CURRENT.json) · [lower](lower/README.md) |
| 协作纪律 | [HANDOFF](HANDOFF.md) 第 7 节（摘要见 [WORKFLOW](WORKFLOW.md)） |
| 历史实验与审查 | [registry](registry/README.md) · [reviews](reviews/README.md) |

**当前（2026-10-31）：** T2.6W1 已交付（`no_gain`）。执行书 [T2_7_GUIDE_20261031](upper/run_orders/T2_7_GUIDE_20261031.md)（科学冻结 `2208a51`，执行器引导）已批准并下发给下层。用户对方向与方法的理解见汇报稿 [wang23.pdf](wang23.pdf)。`CURRENT.json` 由下层维护，最近一次同步为 10-29（T2.6W 关闭、资产清理）；下层开跑 T2_6W1 时更新。

**推进方式：**研究层定方向与授权；T2.3 起上层主导 T2.x 迭代（诊断、设计、实现、测试、冻结、执行书）；下层完整训练、评估、诊断与交付；上层集中审查。技术正确性要检查，研究效果不要求训练前证明；旧方案筛选不能反复阻断已授权主线。细则见 HANDOFF 第 7 节。
