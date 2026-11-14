# 快速翻滚非合作目标的六自由度预捕获：ECC 与旋转同步窗口引导

高保真 SE(3) 六自由度预捕获。追踪器 106 kg（每轴 ±2 N、±0.6 N·m），目标 225 kg 箱体，自旋 2.1–2.7°/s，惯量未知；500 km / 45° 轨道，真值 RK45（中心引力 + J2），控制周期 0.1 s。安全只由真值几何判定。

**两个问题，两个机制：**
- **理论问题——长期决策与短期受约束执行的耦合：ECC**（执行反馈闭环协同，Execution-feedback Closed-loop Coordination）。
  - 学习侧：评价器在训练时读执行器的 6 项执行反馈 z_E；策略只读 18 项有效观测。
  - 决策侧：执行器的机会评估在训练与执行中以同一认证映射修正意图。
- **场景问题——快速翻滚目标的动态捕获窗口：RWG**（旋转同步窗口引导，Rotation-synchronized Window Guidance）。
  - SAC 每 16 s 在目标自旋坐标系给出二维阶段意图（接近速度、沿扫掠速度），纬向由固定几何律调到端口扫掠锥上。
  - 入口由入口规划器（代码中的条件 G★）按端口经过时机规划，失败时在原到达时刻附近重定时。
  - 入口、下降与捕获由冻结 MPC 执行。

## 从这里开始

| 文件 | 内容 |
|---|---|
| `docs/paper/README.md` | 论文资料入口：写作取材指南 |
| `docs/paper/facts/` | **方法事实稿**（中文，按论文方法部分顺序）：术语符号、建模、总体框架、RWG、执行层、ECC、学习、对比方法 |
| `docs/paper/BACKGROUND.md` | 背景、两个问题、研究问题、贡献（引言素材） |
| `docs/paper/EXPERIMENTS.md` | 正式实验设计（事先登记）：任务条件、训练组、终点与统计、图表 |
| `docs/final_method/METHOD_REGISTRY.json` | 正式实验的唯一配置来源 |
| `docs/engineering/` | 工程记录：`REPRODUCE.md`（环境、测试、命令、开局号段）、`FACT_MAP.md`（论文对象 ↔ 源码）、`AUDIT.md`（一致性审计、收尾与提速的逐位核验） |
| `docs/HISTORY.md` | 研究时间线（问题 → 做法 → 结论 → 转向） |
| `docs/DEAD_ENDS.md` | 已验证的负结果台账，**改方法前必读** |
| `evidence/README.md` | 证据目录索引 |
| `CLAUDE.md` | 协作规则与陷阱 |

协作文档（执行书、交接、项目状态）在 `collab/spacecraft` 分支的 `docs/collaboration/`。

## 代码

| 路径 | 内容 |
|---|---|
| `env/t2_stage.py` | 阶段决策环境：RWG 几何与阶段目标、执行链（远场跟踪律、入口规则、窗口重定时、冻结 MPC）、观测视图、任务分布、ECC 机会评估与认证映射、只读审计 |
| `env/t2_probe.py` | ECC 模式预测：执行器自身的阶段层与入口规则，48 s |
| `env/t1_*.py`、`env/se3_rendezvous_env.py`、`env/task.py` 等 | 冻结执行器与任务：真值环境、约束与判据、阶段规划器、条件 G★、MPC 接口（文件名中的 `t1` / `t2` 是历史命名） |
| `controllers/mpc/`、`dynamics/`、`estimation/` | 冻结 MPC、六自由度动力学与引力模型、相对 EKF |
| `train/t2_mve_sac.py` | 学习器：SAC（`e_priv` 非对称评价器、`e_sac`）与 TD3（`e_td3`），异步采集 |
| `experiments/final.py` | 正式实验唯一入口；`experiments/t2.py` 回合运行器；`formal_readout.py` / `formal_figures.py` 读出与作图 |
| `tests/` | `python -B -m pytest -q`，以退出码判定 |

2026-11-12 工程收尾前的完整仓库（全部历史代码、测试、文档与证据）在科学提交 `fcd8d2ee`（标签 `archive/pre-cleanup-20261112`），对照表 `docs/archive/PATH_MAP_20261112.csv`。
