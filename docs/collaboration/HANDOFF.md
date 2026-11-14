# 交接（2026-11-12，三层共用）

**本文件不是执行单，不构成任何实验授权。** 现行工程状态见 [PROJECT_STATE](PROJECT_STATE.md)（上层维护）与 [CURRENT.json](CURRENT.json)（下层维护）。开发期的完整交接与研究演进在 [archive/](archive/README.md)。

## 1. 研究问题与方法

**对象：** 快速翻滚非合作目标的六自由度预捕获（追踪器 106 kg，每轴 ±2 N / ±0.6 N·m；目标 225 kg，自旋 2.1–2.7°/s，惯量未知；500 km / 45° 轨道，真值 RK45 含 J2，控制 0.1 s）。

**两个问题（与汇报稿 `wang24.pdf` 一致）：**
- **理论问题——长期决策与短期受约束执行的耦合 → ECC**（Execution-opportunity Certificate Coupling）：
  - 评价器在训练时读执行器的 6 项执行反馈 z_E，策略只读 18 项有效观测；
  - 执行器的机会评估（3 × 3 意图的 48 s 模式预测）在训练与执行中以同一认证映射修正意图；
  - 回放存策略的提议动作。
- **场景问题——快速翻滚目标的动态捕获窗口 → RWG**（旋转窗口引导）：
  - 自旋坐标系二维阶段意图，纬向由固定律调到端口扫掠锥；
  - 入口按端口经过时机规划，失败时在同一次经过内重定时；
  - 冻结 MPC 执行近场。

**唯一来源（科学分支）：**

| 内容 | 路径 |
|---|---|
| 方法（记号、命题、伪代码、不声称的内容） | `docs/paper/METHOD.md` |
| 论文对象 ↔ 源码 | `docs/paper/FACT_MAP.md` |
| 一致性审计（含收尾等价性） | `docs/paper/AUDIT.md` |
| 背景、命名、贡献 | `docs/paper/BASELINE.md` |
| 正式实验设计（事先登记） | `docs/paper/EXPERIMENTS.md` |
| 配置 | `docs/final_method/METHOD_REGISTRY.json` |
| 命令与开局号段 | `docs/REPRODUCE.md` |
| 时间线、负结果 | `docs/HISTORY.md`、`docs/DEAD_ENDS.md` |

## 2. 现状

- **方法冻结：** `ECC_CERT_LOOP`，行为基准科学 `e0a61753`。依据是开发期 B 步：紧急任务对 SAC_C +36.0，捕获 145 / 111，翻转 37 : 3，0 次不安全。
- **工程收尾（11-12）：** 科学分支只保留正式实验用到的代码，冻结路径与 `fcd8d2ee` / `e0a61753` 逐位等价；历史代码、测试、证据和文档移出工作树。原样保存在 `fcd8d2ee`，对照表 `docs/archive/PATH_MAP_20261112.csv`。
- **正式实验：**
  - 6 组（ECC、SAC–MPC、TD3–MPC、ECC w/o 认证、ECC w/o 执行反馈、ECC w/o 旋转引导）× 种子 20 / 21 / 22 × 5000 决策；
  - 条件 U / E / N 与 R（2.0 m/s）；
  - 执行书 [RUN_FORMAL](upper/run_orders/RUN_FORMAL.md)，**待用户单独授权**。
- **开放事项：**
  - 下降类认证的安全性按规则报告（DEAD_ENDS O7）；
  - 作图两处补项（F4 逐对分布、RWG 图的离散点与箭头）在写作阶段补。

## 3. 资产

- **Git**（仓库 `251614030027r-max/spacecraft`，三条分支）：
  - `claude/sac-mpc-coupling-design-ns7g6i`：科学代码与证据；
  - `collab/spacecraft`：只有 `docs/collaboration/`；
  - `main`：实验结束后合并。
- **归档标签**由下层打：`archive/pre-cleanup-20261010`、`archive/collab-pre-cleanup-20261029`、`archive/t2-11-*`，以及本次 `archive/pre-cleanup-20261112`（`fcd8d2ee`）等，见 RUN_CLEANUP。
- **证据：** 科学分支 `evidence/README.md`。正式实验证据将放在 `evidence/FORMAL/`。
- **本机**（下层）：
  - 科学检出 `D:\py\DRL2`；
  - 过程文件 `C:\Users\35884\Documents\Spacecraft\过程文件\`，只保留最近几轮（FORMAL_PREP、GOV_A / GOV_B、ECC 开发轮）；
  - 盘点见 `lower/handoffs/` 最新的资产交付。

## 3A. 写作窗口（2026-11-14 起，用户组织）

- 读科学分支的全部论文材料与证据，调研文献，起草论文；
- 只改科学分支 `docs/manuscript/`（规则见该目录 README）；方法、实验设计、代码、注册表、证据只读，口径问题提给上层；
- 正式数字只取 `evidence/FORMAL/`，交付前用占位符。

## 4. 三层职责与纪律

**工作链：** 研究层授权 → 上层实现、核验、冻结 → 下层本机执行、交付 → 上层集中审查 → 研究层 / 用户决定下一步。没有明确 GO，不自动启动训练。

| 角色 | 职责 |
|---|---|
| 研究层（用户 + 讨论窗口） | 定研究问题、成功标准、方向与授权；把关问题立意、创新归因与过度归因 |
| 上层 | 设计、实现、测试；先写规格与判据入库再实现；冻结 SHA；写可直接执行到交付的执行书；核验交付；维护科学分支文档与 PROJECT_STATE |
| 下层 | 在授权范围内于本机执行训练、评估、交付；维护 CURRENT.json 与本机资产；授权内常规问题自己修并写明，科学方案、阈值、开关、步数、开局一律不擅改 |

**共同红线：**
- 真值 RK45 几何是唯一的安全裁判；
- 执行器（MPC、规划器、入口规则、物理）冻结；纯 MPC 在每个正式块上干净重跑；
- 不制造差距，奖励不编码答案；
- 每次训练从零开始，≥ 3 个种子，报告分布，不换种子、不挑检查点、不丢失败种子；
- 正式块 300000–999999 与正式种子 20–22 只在授权的正式执行书下使用；
- 预注册判据事后不改，FAIL 按预注册处理；
- 任何改动先写规格与判据入库再实现，冻结路径逐位可复现，以 pytest 退出码判定；
- 进入决策或论文的数字必须指向产物；
- Git 只记事实，提案只以授权执行书入库；
- 提新方向前先说明 DEAD_ENDS 的哪一条条件已经变化。

**上层沙箱：**
- 4 核、约 15 GB，容器会被回收；
- 不能推标签、不能删远端分支；
- 有作业在跑时用 `git worktree`，不改它导入的工作树；
- 按 PID 结束进程；
- 提交附协作尾注，不写模型标识，不开 PR。

**下层：**
- 开始前读执行书，核对 Git、冻结代码、环境与进程；
- 后台隐藏，按 PID 结束；
- 交付须让上层能独立判定：冻结版本、配置、完整样本与覆盖、原生判定、异常 / 缺项、原始证据位置与 SHA；
- 先 fetch，精确提交，不强推。

## 5. 新窗口第一步

1. 读本文件、[PROJECT_STATE](PROJECT_STATE.md)、科学分支 `README.md` 与 `CLAUDE.md`。
2. 读 `docs/paper/METHOD.md`、`EXPERIMENTS.md`。
3. 需要开发期细节（为何选这个方法、为何否定其他路线）时，看 `docs/HISTORY.md`、`docs/DEAD_ENDS.md` 与 [archive/](archive/README.md) 里的研究演进和旧交接。
