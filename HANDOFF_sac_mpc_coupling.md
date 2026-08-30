# SAC–MPC 耦合方法：交接指导文件（给下一个窗口）

> 本文件是"方法探讨窗口"的产出，供**下一个窗口开始推进 SAC-MPC 耦合方法**时首先通读。
> 权威文档仍是仓库根 `CLAUDE.md`（研究主线、基线正式结果、纪律、陷阱）；本文件只补"方法层的已定决策 + 依据 + 第一步"。
> **不要重新推翻这里已定的决策**（除非有新的测量证据）；开放问题单列在第 8 节，需与用户确认后再动。

---

## 0. 一句话定性

在已完成的两个干净基线（Pure MPC 20/20 但算力超实时预算；Pure SAC 便宜但 3/1/0、不保证可行）之上，做一个**SAC-MPC 值函数耦合**的混合控制器：**下层短 horizon 约束凸 QP + 学习的凸终端代价，上层 SAC 提供终端值与热启动**。目标是在 **Λ>1 持续共旋约束捕获**这个别人不占据的 regime 里，同时占住"实时可部署 × 约束安全"两个角。定位：**扎实的应用创新**（组合+场景+compute 论证），不是新算法/新理论突破——这与用户"复用别人机制、不自造"的路线一致。

---

## 1. 已锁定的方法决策（不要相互 relitigate）

**脊柱 = 值函数耦合（value coupling）**：
- **下层**：保留现有**约束凸 QP**（CVXPY + CLARABEL），horizon 从 50 **缩短**（先试 5–15 步）。终端代价从"到 reference 的固定二次型 `terminal_weight·I`"换成**一个学习到的值函数在终端参考点的凸二次近似（Gauss-Newton / PSD）**。
- **上层**：SAC 的 **actor** 提供 QP 求解初值（warm-start）+ 一个候选动作；一个**专用终端值函数 V(x)**（输入是 MPC 的 12 维相对状态）提供终端代价来源。
- **耦合优化环（双向，非串联）**：终端值/actor 喂进 MPC；MPC-in-the-loop 的**闭环回报**回来训练终端值/actor。终端值学的是"混合策略"的 cost-to-go，不是裸 SAC 的。

**主要参考文献分工（关键——见第 3 节的三陷阱分析）**：
- **概念 + 理论**主参考：**AC4MPC**（Reiter & Diehl, IEEE TCST 2025，仓库 `References/AC4MPC.pdf`）。借它的机制概念（critic→终端值、actor→热启动）和闭环代价上界定理（hybrid ≥ actor + 误差项）。**在此基础上可适当优化改进**（用户已同意）。
- **实现**主参考：**凸终端代价 surrogate 线**——Bemporad《Learning Convex Terminal Costs for Complexity Reduction in MPC》(CDC 2021) / 《Value Function Approximation for NMPC: Learning a Terminal Cost with a Descent Property》(arXiv 2508.05804, 2025)。真正抄公式的地方：**监督回归拟合 cost-to-go，拟合成凸二次或 ICNN**，守住凸 QP。
- **佐证 + 可选增强**：RL-Guided MPC（arXiv 2506.13278, 2025）——同机制的干净单次求解版，多一个"actor 定义终端域约束"的可选增强（有利递归可行性；先不加复杂度）。

**明确排除（连同理由）**：
- **可微 MPC / NN 直接进 NMPC 目标**（Scaramuzza ACMPC 2306.09852 等）：训练期需穿 QP 反传、对含李群状态的网络求流形雅可比，算力爆炸、数值病态、笔记本训不动、难调参。
- **AC4MPC 原版的"并行双解 NMPC"**：加倍算力，与我们唯一差异化（compute）反向。**只做单次求解**。
- **在线调 MPC 代价权重**（上海交大 NCMPC、DQN-MPC）：CLAUDE.md 明确排除。
- **NN 终端代价直接塞进求解器**：见第 3 节陷阱 2/1。

---

## 2. 工程痛点与论文定位（动机链，已与真实资料核对）

**工程痛点（一个，我们独占）**：快速翻滚非合作目标的**同步受迫抵近（synchronous/forced-motion approach）**——body-fixed 抓捕点使 `Λ=ωr/v_max`，任务 12m 处 Λ=1.41→3m 处 0.35，穿过 1。Λ>1 意味着惯性悬停/零推力滑行不可行，必须**持续近饱和推力共旋**，同时被随目标体系旋转的走廊/FOV/速度/推力多硬约束夹住。
- 真实性已锚定：Envisat 实测翻滚 2.8–3.5°/s（我们 ω=0.0412 rad/s=2.36°/s，落在真实带内）；"synchronous approach / rotating LOS corridor / 相对速度上限"都是学界命名的真实概念；真实软对接速度限 0.03–0.3 m/s 与我们 0.35 同量级。
- **我们 claim 的是定量判据 Λ + "快翻滚+紧速度限逼出持续受迫"的观察**，不是发明场景。
- 两个诚实边界（论文必须主动交代，否则被审稿人反打）：① 速度限定义在**目标旋转系**（要补一句物理动机：近距接触能量/相对目标本体的传感器运动）；② 目标翻滚是**单一确定实现**（CLAUDE.md 陷阱 9，写成 limitation）。

**两个方法痛点（拉踩，守诚实边界）**：
- RL 派（哈工大 SE(3) 深度 RL）：**无约束/无可行性保证**。证据=我们 Pure SAC 3/1/0、260862 每 episode 违约。
- MPC 派（南航双层 / 上海交大 NCMPC）：**Λ>1 下计算复杂度爆炸，根子是模型陈旧**。南航**自己**在 Limitation 1 承认 LTI 假设 moderate 翻滚、把逐次重线性化列为 future work；我们量化成 258ms/次、沿 horizon 逐点重线性化 ≈13s/步=130×预算。**打南航用"它自己的话"打，交大用"正交杠杆+被排除"。** 北航不进这两痛点（它 ADP 处理了约束），作**工况证人**（慢翻滚、无速度限）。

**我们方法的相对优势（诚实版，不吹"完成他们不完成"）**：Pareto 双角——Pure MPC 完成+安全但**不实时**（p95=2.06×、max=7.7×，刷新步结构性超时）；Pure SAC 实时但**不安全**（种子彩票）；**Hybrid 目标同时占两角**。正是 Λ>1 把"实时"与"安全"逼成 tradeoff，easy regime 不需要 hybrid——这解释了为什么交大/南航不需要我们的方法。

**加固一招（强烈建议纳入实验，不改冻结的 S1-v2）**：给 Pure MPC 加 **100ms 硬截止**（超时即 hold 上一解），把抽象的"compute p95"翻译成"实时截止下的完成率下降"，而 hybrid 因短 horizon 压进预算仍 20/20——把"MPC 加速器"救成"实时约束下唯一可行"。

---

## 3. 三陷阱分析（AC4MPC 原版的坑，已核对原文；说明我们为何"借概念不照搬实现"）

另一路 AI 指出 AC4MPC 三处硬伤，**已对照 `References/AC4MPC.pdf` 原文核实属实**：
- **陷阱1 Critic Trap**（原文 line 192 "assume a well-trained actor-critic"；line 1690–1694 "the nonlinearity of the critic was occasionally preventing the solver from converging... diminished by multiplying... factor β"）：噪声大的 critic 当终端代价会带偏求解器。
- **陷阱2 Manifold Trap**（line 1674–1677 需 tanh 平滑、ReLU 不可微、要算网络一阶导）：NN 进 NMPC + 李群流形求导，数值病态。
- **陷阱3 Speed Trap**：并行双解 NMPC，算力翻倍。

**我们的方案如何对待**：
- **陷阱 2、3 已被我们的设计绕开**（这正是"裁剪 AC4MPC"的原因）：我们用**凸二次 surrogate 进凸 QP**（无 NN 进求解器、无流形求导、无 tanh 约束）+ **单次求解凸 QP**（非 NMPC、非双解）。AC4MPC 自己的补救（终端 Hessian 设小对角、只算一阶、乘 β）本就是"粗糙版凸二次近似"——我们做干净。
- **陷阱 1 是真·残留，用四点管**：① 我们要的是"能用的终端 cost-to-go 排序"，门槛远低于"好策略"；② **不复用噪声 Q 网，而是监督回归一个结构上凸的专用终端值**（凸 by construction→不可能造出带偏求解器的非凸包）；③ **下层硬约束凸 QP**→终端值再烂最坏只是"不够优"，**不会不安全/不可行**（软失败，非 NMPC 的 infeasible 崩溃）；④ **闭环训练**→终端值从 MPC 已兜底的好轨迹里学，随 hybrid 一起 bootstrap。
- **把三陷阱写进论文**：作为"为什么不直接用 AC4MPC / 为什么这样设计"的现成论证（有原文出处），把批判变成方法论贡献。

---

## 4. 第一步：最小验证实验（先打掉最大风险，再谈完整 hybrid）

**目的**：Critic Trap 是唯一残留的真风险，且核心实证问题就是"凸终端值 surrogate 的标定够不够用"（CLAUDE.md 一直点名的 critic 标定）。**先便宜地证伪/证实它，再决定要不要投入完整 hybrid。**

**做什么（单因子）**：短 horizon 约束凸 QP + **学习的凸终端值**，**先不接 actor 热启动、先不接调用门**。
- 终端值来源第一版：可先用一个**离线监督回归**的凸二次 V(x)（数据=脚本/MPC 闭环的 cost-to-go），或直接对现有 SAC critic 做终端点凸二次近似做 A/B。
- 对照：① 现有 Pure MPC（长 horizon，corridor 参考）；② 短 horizon **无**学习终端值（=终端代价仍固定二次）。
- **看什么**：短 horizon+学习终端值能否**打破 hover / 保持完成**，且 **worst-margin 不转负（安全不掉）**，同时**每步 compute 明显下降**（短 horizon 砍掉重线性化开销）。这一步能在**笔记本 CPU**上跑（前向 QP、无穿 QP 反传）。

**判据**：若"短 horizon+学习终端值"在不掉安全下把完成/质量拉到接近长 horizon MPC，且 compute 显著降→脊柱成立，进完整 hybrid（加 actor 热启动、调用门、闭环联合训练）。若终端值一接就掉安全/掉完成→回到陷阱 1，先修 surrogate（换 ICNN / 加 trust-region / 降终端权）再说。

---

## 5. 代码锚点（下一个窗口需要的接口，均已核对）

- **MPC 命令**：`controllers/mpc/controller.py::MPCController.command(relative_vector[12], *, target_state, time_seconds) -> (action[6], MPCStepDiagnostics)`。终端项、`_reference_trajectory`、exact-linearization 刷新逻辑都在此文件。
- **MPC 配置**：`controllers/mpc/config.py::MPCConfig`（`horizon_steps=50`, `terminal_weight=100`, `reference_source∈{fixed,corridor_guidance}`, `linearization_source=exact`, `exact_linearization_refresh_steps=10`, `solver=CLARABEL`）。`constrained_mpc_nominal_config()` / `corridor_tracking_mpc_config()`。**改终端代价来源就在这里加字段 + 在 command 里替换终端项。**
- **走廊制导**：`env/task.py::corridor_guidance_velocity(...)`（reward/观测/脚本/MPC 参考共用的唯一制导法，别再造第四份）。
- **SAC 配置**：`train/configs.py::PURE_SAC`（`gamma=0.997`, `ent_coef=0.005` 固定, `net_arch=(256,)*4`, 训练长度 200k）。动作空间 `Box(-1,1,(6,))`（=±5N/±0.6Nm）；观测 24d（`phase2_mission_v4_phase_guidance_error_24d`，被测试冻结——**终端值请建在 12 维 MPC 状态上，别碰 24d 观测**）。
- **共享测量路径**：`eval/evaluate_policy.py::evaluate_model` 只需一个有 `.predict(obs, deterministic)->(action,state)` 的对象；三方同一条 `main_table` 路径，口径唯一定义在 `eval/metrics.py`。
- **环境入口**：`env/phase2_env.py::phase2_environment_config("single_phase")`；任务 `env.task.phase2_s1v2_mission_config()`。
- **就位工具**：`experiments/evaluate_mpc.py --task single_phase`、`eval/main_table.py`、`experiments/profile_mpc_step.py`、`eval/digest_run.py`。
- **测试基线**：`python -B -m pytest -q` 应 **112 passed**（沙盒需先 `pip install --upgrade cffi` 再 `pip install -r requirements-dev.txt`）。

---

## 6. 纪律与约束（来自 CLAUDE.md，混合方法同样适用）

- **严格单因子实验**，每个因子独立 commit + manifest。**第一步的"终端代价来源"就是那一个可解释因子。**
- **≥3 训练 seed** 才能上结论（n=1 曾翻掉 8/9 个已发布数字）；**报 seed 分布，不报最好 seed**。
- **每次训练从零**（新 actor/critic/replay，不从 checkpoint 续、不阶段性 hand-off）。
- **S1-v2 任务参数冻结**（改它作废十一轮单因子可比性）。要制造更难 regime 只能**另开新副任务**，不动主基准。
- **compute 报 p95/max，不报均值**（均值随机器变）；注明实现（Python/CVXPY/CLARABEL/单核）+ 星载慢 1–2 数量级（可引 arXiv 2405.06771）。
- **长训练在用户本机跑**；本会话/窗口只做代码、pytest、≤5k 步 smoke、诊断。用户贴 `eval/digest_run.py` 输出，不传日志。
- **本会话无仓库写权限**：代码改动打包 **zip（保持仓库相对路径）**交用户本地 commit；**不要 git push、不要用 MCP 做 push/PR**。回答用**中文**。

---

## 7. 参考文献清单（仓库 `References/` + 新增网络源）

仓库内（已抽文到分析用，PDF 用 `pdf2txt.py -o out.txt` 抽取，勿整本读入）：
- `哈工大.pdf` SE(3) 建模祖本（不作贡献主张）｜`北航.pdf` 约束处理+慢翻滚工况证人｜`南航.pdf` 双层 MPC + **Limitation 1（核心动机）**｜`北航编队.pdf` RL 供 schedule/自适应 Δt 接口范式（冲量模型不可比）｜`上海交大.pdf` NCMPC 在线调权（**被排除**，借三方表格式）｜`引入死区迟滞...pdf` 用户前作（不引）
- **新增**：`AC4MPC.pdf`（概念+理论主参考）｜`RL-MPC综述.pdf`（Reiter/Diehl/Gros, ARC 2026，耦合机制分类总账，§6 架构 / §8.1.2 终端值 / §8.2.1 可微 MPC / §9 MPC 做 critic）｜`ecc26_rollout.pdf`（一步 Newton 精修，留作后期 compute 升级）

网络源（arxiv 被沙盒出口代理挡，用户本机可取全文核对公式）：
- 实现：Bemporad CDC2021 凸终端代价；arXiv **2508.05804**（descent property）
- 佐证：arXiv **2506.13278**（RL-Guided MPC）；arXiv **2102.11122**（RL 学 MPC horizon，自适应调用率参考）
- 排除对照：arXiv **2306.09852**（可微 ACMPC，Scaramuzza）
- compute 论证支撑：arXiv **2405.06771**（星载处理器 RL/RTA 计算时间）；arXiv **2505.05588**（ISS Astrobee 学习热启动，旋转动力学迭代 -60%）

---

## 8. 尚待与用户确认的开放决策（下一个窗口先问，别擅自定）

1. **上层动作空间的确切形式**：actor 直接出动作（做热启动初值 + 门开时直接控制）为首选；是否也把"调用门开关"并入（混合离散-连续动作）？
2. **决策周期 / MPC 调用率**：自适应门（借 2102.11122 + 北航编队 Δt）现在就做，还是第一步验证通过后再加？
3. **终端值参数化**：凸二次（最简、可调）vs ICNN（更强、稍复杂）；先凸二次。
4. **训练方案**：MPC-in-the-loop 闭环联合训练（贵、对齐）vs 两阶段（先训值再冻结进 MPC 微调，省算力）。笔记本约束下倾向两阶段起步。
5. **是否纳入"100ms 硬截止"评测口径**（第 2 节加固招）作为方法外第一个实验项。

---

## 9. 给下一个窗口的开场动作建议

1. 先通读本文件 + `CLAUDE.md`，`pytest -q` 确认 112 passed。
2. 就第 8 节的开放决策与用户对齐（尤其 1、3、4）。
3. 落 **第 4 节最小验证实验**：在 `MPCConfig` 加"终端代价来源"字段、在 `command` 里把固定终端二次替换为"学习凸终端值的终端点二次近似"，短 horizon，先离线/近似终端值，单因子 A/B。代码打 zip 交用户跑 ≥3 seed，回贴 `digest_run` / `main_table`。
4. 用结果决定进不进完整 hybrid。
