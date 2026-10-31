# 快速翻滚非合作目标的六自由度预捕获：SAC–G★–MPC 训练内耦合

高保真 SE(3) 六自由度预捕获控制。目标自由翻滚，真值用 RK45 传播（中心引力 + J2）；控制周期 0.1 s，推力每轴 2 N。

- **学习层：** SAC 在审议点（每 6 s，以及计划事件之后）选择任务意图：何时交会、下降节奏、入口速度。
- **G★：** 把意图变成可执行的最小 Δv 轨迹，负责计划保持、恢复和期限保护。
- **MPC：** 共享约束 MPC 闭环执行，并提供 2 s 影子预测，作为训练期反馈。

**当前状态（2026-10-18）：** T1-L 与 T1-O V2 两次检验都没有找到可由观测兑现的状态相关决策余量（T1-O V2 停于 Gate C）。下一任务版本由研究层设计，尚未冻结。

## 从这里开始

| 文件 | 内容 |
|---|---|
| `docs/current/TASK.md` | T1 任务族：物理、安全与捕获、奖励、初值档案、G★、审议接口、开局号段 |
| `docs/current/METHOD.md` | 方法构件：阶段感知 SAC、MPC 相对项、区间 Bellman、2×2 样本资格。尚无冻结的正式方法 |
| `docs/current/EVALUATION.md` | 学习空间准入协议（物理筛选、Gate A–E）与已有结果 |
| `docs/current/REPRODUCE.md` | 环境、测试与命令 |
| `docs/HISTORY.md` | 一页研究时间线 |
| `docs/DEAD_ENDS.md` | 已验证的负结果台账。**改方法前必读** |
| `docs/archive/GIT_ASSETS_20261018.md` | 分支、标签、证据目录与本机产物总表 |
| `CLAUDE.md` | 规则与陷阱（给 AI 协作窗口） |

协作入口（执行单、交接、实验总账、项目状态）在 `collab/spacecraft` 分支的 `docs/collaboration/`。

## 目录

| 目录 | 内容 |
|---|---|
| `dynamics/` `env/` `controllers/` `estimation/` | 动力学、任务与环境（T1 在 `env/t1_*.py`）、约束 MPC、相对 EKF |
| `train/` | 训练入口与学习器（T1 在 `train/t1_*.py`、`train/train_t1*.py`） |
| `experiments/` | 评估、筛选、Gate 与读出工具（T1-O 在 `experiments/t1o_*.py`） |
| `eval/` | 指标代码 |
| `evidence/<实验ID>/` | 已发布的证据，附 SHA256SUMS |
| `tests/` | 回归与契约测试 |
| `docs/` | 当前文档；`docs/archive/` 为按阶段归档的历史文档 |
| `References/` | 文献及用途说明 |

## 分支与历史

| 名称 | 说明 |
|---|---|
| `claude/sac-mpc-coupling-design-ns7g6i` | 科学代码与证据，唯一的科学分支 |
| `collab/spacecraft` | 协作分支：只改 `docs/collaboration/` |
| `main` | 实验全部结束后统一并入 |
| `archive/*` 标签 | 已删除分支的冻结标签，以及整理前的完整仓库 `archive/pre-cleanup-20261010` |

历史从未重写，所有引用过的 SHA 都可达。
