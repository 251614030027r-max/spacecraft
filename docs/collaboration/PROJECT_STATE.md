# 项目当前状态（2026-10-18）

*先读 [HANDOFF.md](HANDOFF.md) 与 [研究演进](RESEARCH_EVOLUTION_AND_COUPLING_DESIGN_20261012.md)（2026-10-18 按三窗口交接说明统一重写）。本文件只记事实：正在运行什么、东西在哪里、哪些已完成、哪些未完成。由上层维护；下层现场状态见 `CURRENT.json`。*

## 1. 当前状态

| 项 | 状态 |
|---|---|
| 运行中 | **无。** 没有任何训练或评估在跑，下层待命 |
| 研究阶段 | T1-N（[Research Plan R2](upper/RESEARCH_PLAN_R2_FOR_REVIEW_20261018.md)），研究层裁决采用"G★ 计划 + MPC 前 2 s 校正"的置信加权排序辅助项；已实现、核验并**冻结为 Round 1 版本 `2c504c1`**，唯一执行单 [T1_N_R2_ROUND1_20261018](upper/run_orders/T1_N_R2_ROUND1_20261018.md)。**正式训练待用户 GO**（GO 前下层只可做第 1–2 节同步、测试与吞吐复测） |
| 最新科学提交 | `claude/sac-mpc-coupling-design-ns7g6i` 上的 `2c504c1`（T1-N Round 1 冻结）。前序：实现 `74d1f7f`，审查证据 `6d73894`。T1-O V2 门控执行用的冻结提交是 `d3902a9` |
| 规格与结论 | 科学分支 `docs/current/`（T1 任务族与 T1-N 第 6 节、方法构件与 T1-N 学习器、准入协议与 T1-N 核验、命令），`docs/HISTORY.md` 阶段 7–8，`docs/DEAD_ENDS.md` D25–D29 |

## 2. 最近完成的工作（按时间倒序）

| 工作 | 结论 | 证据 |
|---|---|---|
| 研究层裁决后的排序项实现与最终诊断（34 对开发状态） | 实现正确（134 项测试通过；冒烟中排序项被冻结并施加）；排序信号与真实长期差方向一致 15/25，纯计划同为 15/25，相关约 0.1；MPC 校正使前 2 s 误差 0.166 → 0.006，但 0 次改变排序。确定性 0k ≈ fixed:0 | 科学 `evidence/T1N_R2_RANK/`（`2c504c1`）；审查第 8 节 |
| R2 上层实质可行性审查（T1-N，开发开局 292000–293999，沙盒） | 物理与接口无阻塞：t = 0 合法 2000/2000；中性请求真实改变计划与闭环；拒绝后保持 300 s 0 unsafe；12 开局闭环：条件最早 63.12、原完整 G★-earliest 62.43、固定请求 65.96、随机 19.62。MPC 配对预测准确（h = 2 误差中位 0.021），但 6 s 奖励差与长期差符号一致仅 2/11（G★ 计划级 9/11）→ 主创新风险，建议计划展开标签（B）。预算：方案 A 墙钟约 2.7 天，B 约 2.2 天 | 科学 `evidence/T1N_R2_PRECHECK/`（`6d73894`）；审查 `reviews/R2_UPPER_FEASIBILITY_REVIEW_20261018.md` |
| T1-O V2 本机门控（执行单 `T1_O_V2_GATES_LOCAL_20261017` 第 2 版，工况 P2，λ_f = 20） | Gate A PASS（上层沙盒）；B PASS；**C FAIL**：状态规则 − A15 = +1.21 [0.01, 2.42]；状态规则 − A0（总是最早）= −0.03。零步、E、D 与四臂训练均未运行 | 科学 `evidence/T1O_V2/`；collab `lower/handoffs/t1o_v2_gates_20261017/`（`be5727d`）；上层审查 `reviews/T1O_V2_GATES_UPPER_REVIEW_20261017.md` |
| T1-O 物理筛选 | P1（15–19 m，2.1–2.7°/s）FAIL：1 次远场外漂 unsafe。P2（14–18 m，2.1–2.7°/s）原 S3 FAIL：F1 32/32 捕获、0 unsafe。原判定保留 | 科学 `evidence/T1O_SCREEN/`（`e8c0f08`） |
| 原 T1-O 工况（15–19 m，2.4–3.0°/s） | 搁置：未登记的试探开局中 6/20 远场外漂 unsafe | `evidence/T1O_SCREEN/README.md` 机理节 |
| T1-L-S（区间 Bellman，`672ba79`） | 正式训练取消，代码保留 | `evidence/T1_L_S_PRECHECK` |
| T1-L 三臂 40k（`166244b`） | N0 / NΔ 相对 0k 学习成立，但低于最强规则；NΔ − N0 未证明；B2 退化；规则族余量 0.21 | 交付 `68680eb`；审查 `reviews/T1_L_S0_UPPER_REVIEW_20261017.md` |
| T1 v4 M0 / M2 40k（`1d432c8`） | 学习未确立，"later"漂移，critic 高估 | 交付 `2242f1b`；审查 `reviews/T1_V4_S0_UPPER_REVIEW_20261016.md` |

更早的 T1 工作（R0 / R1、v2、根因、β、S1）和 F01 线见科学分支 `docs/HISTORY.md` 与 `registry/EXPERIMENT_REGISTRY_20261007.csv`。

## 3. 东西在哪里

| 内容 | 位置 |
|---|---|
| 科学代码、测试、证据、当前文档 | 分支 `claude/sac-mpc-coupling-design-ns7g6i` |
| 分支、标签、证据目录与本机产物总表 | 科学分支 `docs/archive/GIT_ASSETS_20261018.md` |
| 执行单 | `upper/run_orders/`（索引 `README.md`） |
| 下层交付包 | `lower/handoffs/<主题>/` |
| 上层审查 | `reviews/`（索引 `README.md`） |
| 实验总账 | `registry/EXPERIMENT_REGISTRY_20261007.csv` |
| 本机工程 | 只有 `D:/py/DRL2`（检出 `4795910`，不是权威科学头）；旧 `DRL2_T1*` 克隆已保全后退役。保全库 `D:/py/DRL2/local_artifacts/资产归并_20261018/`；资产地图 `lower/handoffs/cleanup_inventory_20261018/LOCAL_ASSET_MAP.md`（`4f3d305`）。详见 HANDOFF 6.5 |
| 历史文档与索引 | 科学分支 `docs/archive/`：`PATH_MAP.csv`、`ARTIFACT_INDEX.csv`、`CODE_INDEX.csv`、`06_f01_value_stopping/` |

## 4. Git 结构（2026-10-18）

**只保留三条分支：**

| 分支 | 作用 |
|---|---|
| `claude/sac-mpc-coupling-design-ns7g6i` | 科学 |
| `collab/spacecraft` | 协作：只改 `docs/collaboration/`；**不得从该分支运行代码** |
| `main` | 实验结束后统一并入，现在不动 |

**已并入的实验分支：** `experiment/t1_l_option_relative_20261015`、`experiment/t1_l_s_tcpf_20261016`、`experiment/t1_o_tcpf_20261017` 是一条直线，已快进并入科学分支，并已由下层删除（远端核验只剩三条分支）。

**标签：** `archive/*` 全部保留，见同一文件第 3 节。历史从未重写。

## 5. 未完成

| 项 | 前提 |
|---|---|
| T1-N Round 1 正式训练与评估 | 用户训练 GO（执行单已就绪） |
| 正式块 300000–999999 的评估、多种子训练 | 未授权 |
