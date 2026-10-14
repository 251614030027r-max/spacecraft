# 项目当前状态（2026-10-13）

*先读 [HANDOFF.md](HANDOFF.md) 与 [研究演进与耦合设计专题分析](RESEARCH_EVOLUTION_AND_COUPLING_DESIGN_20261012.md)。只记事实：正在运行什么、东西在哪里、哪些事已完成、哪些事还没完成。由上层维护；下层的核验状态见 `CURRENT.json`。*

## 1. 当前状态

| 项 | 状态 |
|---|---|
| 主线 | T1 六自由度翻滚预捕获，比较 Pure MPC / M0（SAC + G★ + MPC）/ M1（M0 + 训练期门控 MPC 反事实 critic 目标）。F01 已结束（用户判失败），只作历史 |
| 当前科学提交 | `78eb63c22615c9672c43338be4cec54da713812c`（G★ 0.4、接口 v2、UTD 4、n_envs 6；配置由 `experiments.t1_frozen_config` 从代码输出）。第一轮冻结在 `f36ac77`（G★ 0.3.2），种子 0 中断后作废 |
| 已完成 | R0、R1（两遍，均 REVIEW_EXECUTOR，开发块）；M1 验收 A/B/C；训练前最终核查 P0/P1/P2；开发块规则基准 81.0895（16/16）。证据见下表 |
| 已完成（待研究层决定方向） | `T1_V2_SEED0_VALIDATION_20261013`：M1 种子 0 在 32088 步按平台期规则停止；无检查点超过未训练零点（16/16、81.02），25k/30k 因到达时刻反复推迟而超时。进展说明 `upper/T1_STATUS_BRIEF_20261013.md`，上层审查 `reviews/T1_V2_S0_UPPER_REVIEW_20261013.md`，证据 `evidence/T1_V2_S0`（科学 `818fcd8`）：到达择时相对规则的可学增益约 0.7 回报，远小于 critic TD 误差（约 4） |
| 已完成（待研究层复核） | 根因诊断第一阶段（collab `a5395ef`；证据 `evidence/T1_ROOTCAUSE_S1`，科学 `09d6406`）；完整报告 `reviews/T1_ROOTCAUSE_DIAG_REPORT_20261013.md`：30k 的失败通道是途中推迟（强制保持即 8/8）；30k critic 7/10 状态认为推迟优于保持（真值 3/10），对自身失败状态的 Q 为 91–100，而实际回报为 −105 到 −111；M1 目标继承同一高估。两项修正建议待决定；第二阶段未授权 |
| 已结束（β 路线停止） | T1_BETA_B012：B0 通过；B1 NO-GO（`evidence/T1_BETA_B1`，科学 `32f5419`）：8 开局 × β∈{0,0.3,1,3} 全部成功、无安全否决，但"作用"判据只有 β=3 达到 3/8（门槛 4/8）；平均回报 80.09→80.17，偏离触发的重规划 25→23。B2 不运行；登记 DEAD_ENDS D25。下一步转入既定 Critic 修复的下一阶段审查 |
| 已完成（待用户决定） | T1-S1 场景筛选（`evidence/T1_S1_SCREEN`，科学 `8c08288`）：初距 12–20 m、初速 ≤ 0.18 m/s，8 个新开发开局 × 5 种固定策略，共 40 回合。可执行（每种策略 7/8，296006 全部策略超时，偏离触发的重规划 14–17 次）；κ=1.0 在 7/7 成功开局上一致优于 0.7（统一效应）；τ 差异约 1 点；在前 4 个开局上冻结的状态规则，在后 4 个开局确认时低于统一策略（−1.06；240 s 期限离线精确重算为 −0.68）。结论：τ/κ 接口在 S1 上没有可观测的状态相关收益 |
| 已停止 | 第一轮 `T1_ROUND1_M0M1_20261012.md`：种子 0 成对训练在 13.1k/10.7k 被误中断（collab `709a3ec`），不续训；种子 1、2 未启动且不再按该单执行 |
| 开发块参照 | Pure MPC 11/16、平均折扣回报 0.91（`evidence/T1_dev_pure_295000`）；规则 16/16、81.0895（`evidence/T1_dev_rule_295000`） |
| 未授权 | 独立正式评估（正式块 300000 起）；M0 新训练；种子 1、2；任何奖励、任务、Pure MPC 或共享 MPC 的改动 |
| 已知局限（用户特别记录） | M1 开局择时阶段辅助门控通过率约 4%；计划管理的改善不得表述为初始择时的学习优势 |

**T1 证据（科学分支）：** `evidence/T1_R0`、`evidence/T1_R1_pass1`、`evidence/T1_R1_pass2`、`evidence/T1_M1_validationB`、`evidence/T1_dev_rule_295000`、`evidence/T1_dev_pure_295000`、`evidence/T1_pretrain_checks`。

## 2. 东西在哪里

| 内容 | 位置 |
|---|---|
| 科学代码 | 分支 `claude/sac-mpc-coupling-design-ns7g6i`。T1 本机检出 `D:/py/DRL2_T1`（独立工作树；第一轮在 `f36ac77`，本轮切到 `78eb63c`）；`D:/py/DRL2` 保持整理后的 `4795910` 不动（F01 结果已归档） |
| 当前规格 | 科学分支 `docs/current/`：TASK、METHOD、EVALUATION、REPRODUCE |
| 研究时间线、负结果台账 | 科学分支 `docs/HISTORY.md`、`docs/DEAD_ENDS.md` |
| 已发布证据 | 科学分支 `evidence/<实验ID>/` |
| 历史文档、产物和代码索引 | 科学分支 `docs/archive/`：`PATH_MAP.csv`、`ARTIFACT_INDEX.csv`、`CODE_INDEX.csv` |
| 实验总账 | `registry/EXPERIMENT_REGISTRY_20261007.csv` |
| 本地资产地图 | `lower/handoffs/cleanup_inventory_20261010/LOCAL_ASSET_MAP.md`、`local_assets.csv` |
| 本地尝试清单 | `lower/handoffs/cleanup_inventory_20261010/tried_variants.csv` |
| 流水线脚本（原本不在 git 中） | `lower/pipeline_scripts/<实验ID>/`，索引 `INDEX.csv`：81 个复制、22 个重复、3 个与 git 中已有文件逐字节相同 |
| 本地原始资产 | `D:/py/DRL2`、`D:/py/DRL2_v3e`、`Spacecraft/过程文件`，原位保留 |

## 3. Git 结构（2026-10-10 核实）

| 引用 | 作用 |
|---|---|
| `claude/sac-mpc-coupling-design-ns7g6i` | 科学代码 |
| `collab/spacecraft` | 协作，只改 `docs/collaboration/`。该分支上其余文件是旧副本，**不得从它运行代码** |
| `main` | 不动；实验全部结束后再统一并入 |
| `archive/pre-cleanup-20261010` → `813e27f` | 整理前的完整科学分支 |
| `archive/review/v3e-60k-value-r2-20260930` → `dc6682d` | E01 证据 |
| `archive/review/v3e-stage-c-20261002` → `b383cd2` | E04 证据 |
| `archive/review/stopping-20261006-audit` → `6c5cb97` | E06 证据 |
| `archive/review/stopping-value-check-20261006` → `2d8a581` | E09 证据 |
| `archive/v2-evaluation-20260924` → `736edbc`、`archive/v3-t0b-20260925` → `f50eb78` | 旧开发分支 |

上面这 6 个分支已在远端删除，内容保存在对应标签里。历史没有重写。

## 4. 未完成

| 项 | 前提 | 执行单 |
|---|---|---|
| F01–F05 交付 | 评估完成 | `FINAL_RERUN_20261007` |
| 整理验证：干净克隆、全部测试、正式开局逐字段核对 | 评估结束、进程退出 | `CLEANUP_B_LOWER_20261010` 第 5 步 |
| C 段本地归档 | 整理验证通过，并经用户确认 | `PROJECT_CLEANUP_20261010` 第 5 节 |
| F01 结果发布到 `evidence/F0x/`，并登记总账 | F01 交付 | — |
