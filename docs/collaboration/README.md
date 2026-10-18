# Spacecraft 上下层协作入口

新窗口必读：[HANDOFF.md](HANDOFF.md)（当前研究及版本）与[研究演进](RESEARCH_EVOLUTION_AND_COUPLING_DESIGN_20261012.md)（长期研究逻辑）；工程事实按需核验[PROJECT_STATE](PROJECT_STATE.md)与[CURRENT](CURRENT.json)。**运行实验只用有效执行单指定的冻结科学提交，不能用`collab/spacecraft`上的旧代码。**

| 用途 | 入口 |
|---|---|
| 两份研究交接 | [HANDOFF](HANDOFF.md) · [研究演进](RESEARCH_EVOLUTION_AND_COUPLING_DESIGN_20261012.md) |
| 上层科学及工程状态 | [PROJECT_STATE](PROJECT_STATE.md) · [upper](upper/README.md) |
| 下层真实现场与交付 | [CURRENT](CURRENT.json) · [lower](lower/README.md) |
| 协作纪律 | [HANDOFF](HANDOFF.md) 第 7 节（摘要见 [WORKFLOW](WORKFLOW.md)） |
| 历史实验与审查 | [registry](registry/README.md) · [reviews](reviews/README.md) |

**当前（2026-10-18）：** 没有有效执行单在运行。T1-O V2 本机门控已停于 Gate C（审查 [reviews/T1O_V2_GATES_UPPER_REVIEW_20261017.md](reviews/T1O_V2_GATES_UPPER_REVIEW_20261017.md)），下一任务版本由研究层设计。科学代码只在 `claude/sac-mpc-coupling-design-ns7g6i`；全部分支与资产见科学分支 `docs/archive/GIT_ASSETS_20261018.md`。

**推进方式：**研究层定方向与授权；上层做必要实现、测试和冻结；下层完整训练、评估与交付；上层集中审查。技术正确性要检查，研究效果不要求训练前证明；旧方案筛选不能反复阻断已授权主线。细则见 HANDOFF 第 7 节。
