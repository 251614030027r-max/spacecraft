# 项目当前状态（唯一入口，2026-10-10）

*只记事实：正在运行什么、东西在哪里、哪些事已完成、哪些事还没完成。方向判断、论文主张和方案讨论不写在这里，留在研究层（用户和讨论窗口）的对话中。由上层维护；下层的核验状态见 `CURRENT.json`。*

## 1. 正在运行

| 项 | 状态 | 依据 |
|---|---|---|
| F01 训练 | 262460、262462 已完成 60k；262461 训练中 | `CURRENT.json` |
| F02 30k 结构检查 | 三个模型 D1–D4 全部通过（开发块 266000，共 144 个开局） | `CURRENT.json`；`devcheck_*.json` 在本机，随 F03 一起发布 |
| F03–F05 正式评估 | 进行中，代码 `f2f8169`；271000 块优先 | `upper/run_orders/FINAL_RERUN_20261007.md` |

规则：评估期间不在 `D:/py/DRL2` 里做任何 git 操作；262461 必须完成训练并参与评估。

## 2. 东西在哪里

| 内容 | 位置 |
|---|---|
| 科学代码 | 分支 `claude/sac-mpc-coupling-design-ns7g6i`，整理后末端提交 `046878c`。正在运行的实验固定在 `f2f8169`；在用代码与 `f2f8169` 逐字节相同 |
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
