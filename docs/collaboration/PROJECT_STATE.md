# 项目状态（上层维护；2026-11-18）

开发期（至 11-11）的完整状态表与执行书记录：[archive/PROJECT_STATE_20261111.md](archive/PROJECT_STATE_20261111.md)。

## 1. 当前状态（2026-11-18）

| 项 | 状态 |
|---|---|
| 第一轮正式实验 | 已结束，只运行了四个训练组（ECC、SAC–MPC、ECC w/o 认证、ECC w/o 执行反馈；科学 `561a89d3`、`fade060b`、`aa9efdb4`、`1c191a89`）。紧急任务：ECC 236/288，SAC–MPC 204/288；评价器反馈的作用不显著；经济任务无收益。剩余的 TD3–MPC、ECC w/o 旋转引导与固定策略评估不再运行（用户 2026-11-17）。结果此后作为开发证据，论文不再把认证开关、训练–执行一致性作为主张（科学 `e721c68c`） |
| 方法 | 重构中：ECC 第二版"执行后果门控的选项价值学习"，方向经用户认可；规格与判据在科学 `docs/ecc2/GATE0.md`。RWG、规划器、MPC、模式预测不动 |
| 运行中 / 待执行 | [RUN_ECC2_GATE0](upper/run_orders/RUN_ECC2_GATE0.md)：离线可行性审查（日志分析 + 开发初始条件上的分支重放），不训练 |
| 下一步 | 第 0 关通过后另立第 1 关（开发期小规模训练）规格；不通过则按判据停止或调整 |
| 论文写作 | 写作窗口（用户组织）；方法事实稿 `docs/paper/facts/`。ECC 一节待第二版方法定稿后改写 |

## 2. 执行书（方法线，按时间倒序）

| 状态 | 执行书 |
|---|---|
| 已交付（collab `fa7506d4`，科学 `6d3d6baa`；上层复核通过） | [RUN_CLEANUP](upper/run_orders/RUN_CLEANUP.md)：归档标签、Windows 逐位核验、本机资产整理 |
| 待执行 | [RUN_ECC2_GATE0](upper/run_orders/RUN_ECC2_GATE0.md)：ECC 第二版离线可行性审查（不训练） |
| 已结束（四个训练组；其余不再运行，用户 11-17） | [RUN_FORMAL](upper/run_orders/RUN_FORMAL.md)：冻结科学 `e8474e21`（下层检出、全量 pytest 160 passed、打标签 `archive/formal-code-20261114`）；队列排程 + 训练完即评估，按场次运行（每场一个实验组 × 三个种子，由用户下达开场指令，场次之间挂起），约 9–12 场；前提已满足——RUN_CLEANUP 已交付并复核通过（collab `fa7506d4`，科学证据 `6d3d6baa`）；Windows pytest 158 passed、逐位等价；本机只保留最近四轮，正式目录不存在，磁盘充足。由用户单独授权 |
| 已交付（科学 `e47a1165`，collab `5496cb51`；上层复核通过） | [RUN_FORMAL_PREP](upper/run_orders/RUN_FORMAL_PREP.md)：正式实验准备 WP1–WP8 |
| 已交付（科学 `d738839e`） | [RUN_GOV_B](upper/run_orders/RUN_GOV_B.md)：在环 ECC 选型（B 步） |
| 已交付 | [RUN_GOV_A](upper/run_orders/RUN_GOV_A.md)：执行机会调节器（否决，D45） |
| 已交付（科学 `cc1a0954`） | [RUN_ECC](upper/run_orders/RUN_ECC.md)：ECC 开发轮 |
| 已交付 | [RUN_U2C](upper/run_orders/RUN_U2C.md)、[RUN_U2](upper/run_orders/RUN_U2.md)：决策时认证、探询作 critic 输入 |
| 已交付 | [RUN_PRETEST](upper/run_orders/RUN_PRETEST.md)：EACH 预实验 |

更早的执行书（T1、F01、T2.2–T2.19 等 58 份）在 [archive/run_orders/](archive/run_orders/)。

## 3. 未完成

| 项 | 前提 |
|---|---|
| ~~RUN_CLEANUP 交付~~ | 已完成（2026-11-13） |
| 正式实验 | 用户单独授权 RUN_FORMAL |
| 作图补项（F4 逐对分布、RWG 图离散点与箭头） | 写作阶段，只读已存数据 |
| `main` 合并 | 正式实验结束后 |
| 论文写作 | 写作窗口（用户组织）：只读 Git 资产与参考文献，输出给用户；取材指南见科学分支 `docs/paper/README.md`；正式数字待 `evidence/FORMAL/` |
