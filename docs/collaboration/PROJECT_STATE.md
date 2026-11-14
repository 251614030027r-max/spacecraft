# 项目状态（上层维护；2026-11-12）

开发期（至 11-11）的完整状态表与执行书记录：[archive/PROJECT_STATE_20261111.md](archive/PROJECT_STATE_20261111.md)。

## 1. 当前状态

| 项 | 状态 |
|---|---|
| 方法 | 冻结：`ECC_CERT_LOOP`（ECC + RWG），行为基准科学 `e0a61753`；论文方法 `docs/paper/METHOD.md` |
| 代码 | 正式代码科学 `e8474e21`（11-13：不改结果的提速，环境墙钟约 −14%，沙箱逐位核验 `evidence/SPEEDUP_20261113/`）；工程收尾后的科学 `21730075`：只保留正式实验用到的代码；冻结路径对 `fcd8d2ee` 逐位等价（上层沙箱：环境 7 种情形、6 个配置单进程短训；`docs/paper/AUDIT.md` 第 4 节）；执行器与物理模块与 `e0a61753` 逐字节相同 |
| 正式实验设计 | 事先登记：`docs/paper/EXPERIMENTS.md`；条件 U / E / N；经济任务预算 2.0 m/s（训练与评估，来自预筛），紧急 3.0 m/s，取消 R（用户 11-13）；不加第七组 |
| 运行中 | 无。下层预检已完成（collab `6403b21b`：检出 `e8474e21`，pytest 160 passed，标签 `archive/formal-code-20261114`；Defender 排除与 Windows Update 策略待管理员落实） |
| 待授权 | [RUN_FORMAL](upper/run_orders/RUN_FORMAL.md)：冻结科学 `e8474e21`（下层检出、全量 pytest 160 passed、打标签 `archive/formal-code-20261114`）；队列排程 + 训练完即评估，按场次运行（每场一个实验组 × 三个种子，由用户下达开场指令，场次之间挂起），约 9–12 场；前提已满足——RUN_CLEANUP 已交付并复核通过（collab `fa7506d4`，科学证据 `6d3d6baa`）；Windows pytest 158 passed、逐位等价；本机只保留最近四轮，正式目录不存在，磁盘充足。由用户单独授权 |

## 2. 执行书（方法线，按时间倒序）

| 状态 | 执行书 |
|---|---|
| 已交付（collab `fa7506d4`，科学 `6d3d6baa`；上层复核通过） | [RUN_CLEANUP](upper/run_orders/RUN_CLEANUP.md)：归档标签、Windows 逐位核验、本机资产整理 |
| 待授权 | [RUN_FORMAL](upper/run_orders/RUN_FORMAL.md)：冻结科学 `e8474e21`（下层检出、全量 pytest 160 passed、打标签 `archive/formal-code-20261114`）；队列排程 + 训练完即评估，按场次运行（每场一个实验组 × 三个种子，由用户下达开场指令，场次之间挂起），约 9–12 场；前提已满足——RUN_CLEANUP 已交付并复核通过（collab `fa7506d4`，科学证据 `6d3d6baa`）；Windows pytest 158 passed、逐位等价；本机只保留最近四轮，正式目录不存在，磁盘充足。由用户单独授权 |
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
