# 项目状态（上层维护；2026-11-19）

开发期（至 11-11）的完整状态表与执行书记录：[archive/PROJECT_STATE_20261111.md](archive/PROJECT_STATE_20261111.md)。

## 1. 当前状态（2026-11-19）

| 项 | 状态 |
|---|---|
| 方法 | 最终 ECC = `ECC_LE`：执行机会集合 + 机会协调（保持 / 修正 / 启动远场入口）+ SAC。采用依据：对第一轮 ECC 捕获 251 对 236/288，逐种子 0 / +4 / +11，回报 47.8 对 36.4，0 次不安全 |
| 实验 | 全部交付：第一轮四组、ECC_LE、SAC_LE、TD3_C（云端）。论文数据 `evidence/PAPER_READOUT/`（科学 `8acc41dd`），事实稿 `docs/paper/facts/08_实验结果.md` |
| 论文 | 第一稿：方法章与结果章已完成（科学 `9c2c711d`，`docs/paper/latex/`），其余由写作窗口完成 |
| 下一步 | 新窗口：一轮干净的新正式实验（去掉评价器输入 z、奖励设计、对比组从简、只用紧急任务），见 [HANDOFF](HANDOFF.md) 第 2 节 |
| 运行中 | 无 |

## 2. 执行书（方法线，按时间倒序）

| 状态 | 执行书 |
|---|---|
| 已交付（collab `fa7506d4`，科学 `6d3d6baa`；上层复核通过） | [RUN_CLEANUP](upper/run_orders/RUN_CLEANUP.md)：归档标签、Windows 逐位核验、本机资产整理 |
| 已交付（科学 `af61a48b`） | [RUN_PAPER_FIGS](upper/run_orders/RUN_PAPER_FIGS.md)：论文两张案例图的重放导出 |
| 已交付（ECC_LE `b803b0eb`、SAC_LE `e8b9af0a`；采用 ECC_LE） | [RUN_ECC_LE](upper/run_orders/RUN_ECC_LE.md)：远场入口（检验 + 训练） |
| 已结束（四个训练组；TD3_C 后由云端补跑） | [RUN_FORMAL](upper/run_orders/RUN_FORMAL.md)：冻结科学 `e8474e21`（下层检出、全量 pytest 160 passed、打标签 `archive/formal-code-20261114`）；队列排程 + 训练完即评估，按场次运行（每场一个实验组 × 三个种子，由用户下达开场指令，场次之间挂起），约 9–12 场；前提已满足——RUN_CLEANUP 已交付并复核通过（collab `fa7506d4`，科学证据 `6d3d6baa`）；Windows pytest 158 passed、逐位等价；本机只保留最近四轮，正式目录不存在，磁盘充足。由用户单独授权 |
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
