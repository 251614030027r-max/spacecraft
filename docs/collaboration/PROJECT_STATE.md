# 项目状态（上层维护；2026-11-19）

开发期（至 11-11）的完整状态表与执行书记录：[archive/PROJECT_STATE_20261111.md](archive/PROJECT_STATE_20261111.md)。

## 1. 当前状态（2026-11-19，第二轮）

| 项 | 状态 |
|---|---|
| 第一稿 | 完成：ECC_LE 251/288，论文数据 `evidence/PAPER_READOUT/`，代码 = 科学 `72a8d41` |
| 第二轮方向（用户 2026-11-19） | 正式组：ECC、无反馈（SAC–MPC）、无核心（关修正，附加归因）、TD3–MPC、MPC；副机制指 RWG（重定时评估开关、固定意图配对，学习闭环内消融可选；远场入口属于协调，不单独消融）；6000 步、3 种子、只用限时任务；最终模型只评估 6000 步；学习曲线用 TensorBoard；泛化蒙特卡洛最后再定 |
| 规格 | 科学 `docs/final_method/ROUND2.md`（清理、方法收敛、机会可达性势函数、开发检查判据），先于实现入库；实现与证据在科学 `3a8fc970` |
| 协作流程 | 快速通道（[WORKFLOW.md](WORKFLOW.md)）：开跑 `preflight`、交付 `pack`（本地与云端合并），不自制核验文件，执行书开跑前原地修改 |
| 清理 | 已完成：删去评价器输入 z（DEAD_ENDS D46）与经济任务；17 项观测；第一轮专用脚本删除；27 个开发期证据目录移出工作树（`c15ece12` 可取回） |
| 方法收敛 | 已完成：协调写成一个机会集加一条规则，与第一稿 ECC_LE 执行逐位一致（`evidence/R2_EQUIV/`，12/12 回合） |
| 奖励 | 已实现并直接采用：机会可达性势函数（引理 1 窗口 + 命题 1 经过时机 + 制动包络 + 期限）；不单独做开发验证（用户 2026-11-20） |
| 反馈机制 | 开发轮已交付并判定（2026-11-21，科学 `32128666`）：`ECC_FB` 472/576 对 `ECC` 472/576；同一模型改回最近规则 469，翻转 8 : 5；P、M 均未通过 → 正式 ECC 保持 `ECC`（DEAD_ENDS D47） |
| 正式实验 | 已交付并核查（科学 `36d443fb` 交付，上层核查 `3ad2dbd6`，`FORMAL.md` 第 9 节）：ECC 492/576 对 SAC–MPC 425/576，+11.6 [8.3, 15.1]；无核心 505/576（ECC 对无核心 −2.3 [−4.2, −0.3]）；RWG 重定时 +2.1、旋转引导 +20.3 |
| 下一步 | 论文初稿（写作窗口按审查 R1 上层核实版改稿）；补充资产 [RUN_R2_ASSETS](upper/run_orders/RUN_R2_ASSETS.md)（只评估：转速 / 期限 / 章动泛化、案例逐 0.1 s 重放、终端误差；规格 `FORMAL.md` 第 10 节，科学 `9cde45f1`，沙箱 178 passed）；初稿整理完后针对无核心做一轮修复实验（另写规格） |
| 运行中 | 无（RUN_R2_ASSETS 等用户开场指令） |

## 2. 执行书（方法线，按时间倒序）

| 状态 | 执行书 |
|---|---|
| 等开场指令 | [RUN_R2_ASSETS](upper/run_orders/RUN_R2_ASSETS.md)：论文补充资产（只评估） |
| 已交付并核查（`36d443fb` / `3ad2dbd6`） | [RUN_R2_FORMAL](upper/run_orders/RUN_R2_FORMAL.md)：第二轮正式实验 |
| 已交付（本机 `f211789a`、云端 `1568df1e`；上层判定 `32128666`） | [RUN_R2_FB2](upper/run_orders/RUN_R2_FB2.md)：反馈机制开发轮，P、M 未通过，正式 ECC 保持 `ECC` |
| 撤回 | [RUN_R2_FB3](upper/run_orders/RUN_R2_FB3.md)（并入 FB2 追加） |
| 作废，未执行 | [RUN_R2_FB](upper/run_orders/RUN_R2_FB.md) |
| 撤销，未执行（用户 2026-11-20） | [RUN_R2_DEV](upper/run_orders/RUN_R2_DEV.md)：势函数开发检查 |
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
