# 快速翻滚非合作目标的六自由度预捕获：学习层与 MPC 的单向交接

高保真 SE(3) 六自由度预捕获控制。目标自由翻滚，真值用 RK45 传播，包含中心引力、J2、重力梯度和二阶矩；控制周期 0.1 s。

- **学习层**：SAC，每 2 s 给出任务级参考。
- **下层**：约束 MPC，负责执行与约束满足。
- **协调**：每个决策点判断"继续学习策略，还是现在交给 Pure MPC"，即基于价值的单向交接。

## 从这里开始

| 文件 | 内容 |
|---|---|
| `docs/current/TASK.md` | 任务、工况、约束、完成条件 |
| `docs/current/METHOD.md` | 当前方法：价值停止 |
| `docs/current/EVALUATION.md` | 样本块、判据、描述性证据 |
| `docs/current/REPRODUCE.md` | 环境与命令 |
| `docs/HISTORY.md` | 一页研究时间线 |
| `docs/DEAD_ENDS.md` | 已验证的负结果台账。**改方法前必读** |
| `CLAUDE.md` | 规则与陷阱（给 AI 协作窗口） |

协作入口（执行单、交接、实验总账、项目状态）在 `collab/spacecraft` 分支的 `docs/collaboration/`。

## 目录

| 目录 | 内容 |
|---|---|
| `dynamics/` `env/` `controllers/` `estimation/` | 动力学、任务与环境、约束 MPC、相对 EKF |
| `train/` | 主线配置、工况、停止方法、训练入口 |
| `experiments/` | 主线评估、读数、重放与基线工具（19 个） |
| `eval/` | 指标代码 |
| `evidence/<实验ID>/` | 已发布的证据 JSON |
| `tests/` | 回归与契约测试 |
| `docs/` | 当前文档；`docs/archive/` 为按阶段归档的历史文档 |
| `References/` | 文献及用途说明 |

## 分支与历史

| 名称 | 说明 |
|---|---|
| `claude/sac-mpc-coupling-design-ns7g6i` | 科学代码工作分支 |
| `collab/spacecraft` | 协作分支 |
| `main` | 实验全部结束后统一并入 |
| `archive/pre-cleanup-20261010` | 整理前的完整仓库 |
| `archive/*` | 已删除分支的冻结标签 |

历史没有重写，所有旧 SHA 都有效。
