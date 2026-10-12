# 项目当前状态（2026-10-12）

*先读 [HANDOFF.md](HANDOFF.md) 与 [研究演进与耦合设计专题分析](RESEARCH_EVOLUTION_AND_COUPLING_DESIGN_20261012.md)。只记事实：正在运行什么、东西在哪里、哪些事已完成、哪些事还没完成。由上层维护；下层的核验状态见 `CURRENT.json`。*

## 1. 当前状态

| 项 | 状态 |
|---|---|
| 主线 | T1 六自由度翻滚预捕获，比较 Pure MPC / M0（SAC + G★ + MPC）/ M1（M0 + 训练期门控 MPC 反事实 critic 目标）。F01 已结束（用户判失败），只作历史 |
| 冻结科学提交 | `f36ac77bc2dd328bcc6ec08bfa86e684f8e9ed8c`（G★ 0.3.2；冻结配置 `evidence/T1_pretrain_checks/frozen_config.json`） |
| 已完成 | R0、R1（两遍，均 REVIEW_EXECUTOR，开发块）；M1 验收 A/B/C；训练前最终核查 P0/P1/P2；开发块规则基准 81.0895（16/16）。证据见下表 |
| 已授权待执行 | 执行单 `upper/run_orders/T1_ROUND1_M0M1_20261012.md`：本机准入 → 种子 0 成对 20k → 5k 工程健康检查 → 种子 1、2 |
| 未授权 | 独立正式评估（正式块 300000 起）；任何奖励、网络、门控、预算、种子或任务的改动 |
| 已知局限（用户特别记录） | M1 开局择时阶段辅助门控通过率约 4%；计划管理的改善不得表述为初始择时的学习优势 |

**T1 证据（科学分支）：** `evidence/T1_R0`、`evidence/T1_R1_pass1`、`evidence/T1_R1_pass2`、`evidence/T1_M1_validationB`、`evidence/T1_dev_rule_295000`、`evidence/T1_pretrain_checks`。

## 2. 东西在哪里

| 内容 | 位置 |
|---|---|
| 科学代码 | 分支 `claude/sac-mpc-coupling-design-ns7g6i`。T1 第一轮冻结在 `f36ac77`（本机检出 `D:/py/DRL2_T1`，独立工作树）；F01 的本机检出 `D:/py/DRL2` 保持 `f2f8169` 不动 |
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
