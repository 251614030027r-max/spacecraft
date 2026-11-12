# 项目整体整理：现状分析、目标结构与分阶段执行单_20261010

修订（用户 2026-10-10）：**不合并分支**。`main` 保持落后，实验全部结束后再统一并入；`claude/...` 和 `collab/spacecraft` 继续作为工作分支；只清理其余杂乱分支，并在 `claude/...` 上整理。新增要求：所有试过的方向都要留档，防止以后改进方法时重走死胡同（第 4 节 B2 第 8 步、A 段 3.8）。

状态：A 段已授权待执行（与正在进行的评估并行，只读）；B、C 段为计划，待评估交付、实验停止并经用户确认后执行。
人类授权来源：用户 2026-10-09/10 原话："可以提前让下层开始资产整理，git比较乱，本次评估完毕后停止实验我们会好好整理分析问题所在……旧指令无用数据整理归档，确保项目整体呈现一份干净且历史可查，历史线迭代清晰的工程"。

本单取代 `ASSET_INVENTORY_20261007.md`。那张单针对的是已作废的 262450–262452 运行，且从未执行。

---

## 1. 现状（上层核实，2026-10-10）

### 1.1 远端分支（9 个），逐个处理

| 分支 | 末次提交 | 与科学分支的关系 | 处理 |
|---|---|---|---|
| `main` | 09-04 `619e491` | 落后科学分支 232 个提交；多 3 个网页上传的 PDF | 不动，实验全部结束后统一并入 |
| `claude/sac-mpc-coupling-design-ns7g6i` | 10-08 `813e27f` | 科学代码主线，259 个提交；最新一个是网页上传的 PDF | 保留为工作分支，B 段在其上整理 |
| `collab/spacecraft` | 10-09 | 只改动 `docs/collaboration/`（已核实，其他路径零差异） | 保留为工作分支，不动 |
| `v2-evaluation-20260924` | 09-24 | 已完全包含在科学分支中 | 打归档标签后删除 |
| `v3-t0b-20260925` | 09-25 | 已完全包含在科学分支中 | 打归档标签后删除 |
| `review/v3e-60k-value-r2-20260930` | 09-30 | 多 1 个证据提交（17 个文件） | 打归档标签后删除 |
| `review/v3e-stage-c-20261002` | 10-02 | 多 4 个证据提交 | 打归档标签后删除 |
| `review/stopping-20261006-audit` | 10-06 | 多 1 个证据提交（403 个文件，逐开局 JSON） | 打归档标签后删除 |
| `review/stopping-value-check-20261006` | 10-06 | 多 1 个证据提交（363 个文件） | 打归档标签后删除 |

已有标签只有 `v3e-60k-value-r2-20260930` 一个。

### 1.2 科学分支的工作树：2109 个文件，共 1.8 GB

- **体积几乎全在已提交的训练产物上。** `logs/` 加 `local_artifacts/` 共 1630 个文件、1.7 GB，其中 172 个模型 ZIP 占 1.58 GB（`logs/t12_train` 716 MB、`logs/t7_retry1_20260911` 477 MB、`logs/hybrid` 303 MB）。
  - 这违反仓库自己的规则："只用 `git add -f` 单独发布证据文件，不提交整个产物目录"。
  - 清单见 `upper/cleanup/TIP_REMOVAL_CANDIDATES.csv`，含路径、blob SHA 和大小。
- **`docs/` 根目录平铺 127 个文件**，另有 4 个子目录。当前事实分散在 `CLAUDE.md`（约 400 行，层层叠加的 "Live status"）、`docs/INDEX.md`、`HISTORY.md`、`EVIDENCE_INDEX.md` 和协作分支的 `PROJECT_STATE.md` 等多处。
- **代码中当前主线实际用到的比例很低。** 从主线入口（`train_stopping`、`v3_stopping`、`regime_screen`、`final_tables`、`stopping_replay`、训练健康检查）做导入闭包：
  - `env`、`controllers`、`dynamics` 几乎全部在用（11/12、8/9、8/9）；
  - `experiments/` 只用到 9/79；
  - `eval/` 只用到 1/16；
  - `train/` 中 `train.py`、`configs.py`、`callbacks.py`（旧单阶段训练）不在用。
- 根目录有零散文件：`read_cal.py`（旧标定汇总脚本）、`local_artifacts/`（旧补丁和一次奖励单位缺陷的证据）。
- `eval/` 既放代码（`metrics.py`、`main_table.py`），又放证据 JSON（`eval/adp`、`eval/v3e`、`eval/stopping`），两类混在一起。

### 1.3 本地（下层机器）：上层看不到，由 A 段盘点

已知位置：
- `D:/py/DRL2`：训练和评估工作树，当前正在运行；
- `D:/py/DRL2_v3e`：V3e 训练；
- `C:/Users/35884/Documents/Spacecraft/过程文件`：记录、回执、协作工作树。

有三类风险需要先查清：
- 产生过数字、但不在 git 中的脚本；
- 无法对应到总账实验的孤立目录；
- 作废运行和有效运行混放。

### 1.4 两条整理原则（沿用 2026-09-22 整理单的结论，并修正一处）

1. **不重写 git 历史**（不用 filter-repo）。文档和总账中引用的上百个提交 SHA 和文件 SHA 是证据链本身。仓库体积是已付出的代价，不再回溯。
2. **以前"不能移动 docs/logs，否则断链"的结论，改为"先打冻结标签，再在新的主线上移动"。** 整理前在科学分支末端打不可变标签 `archive/pre-cleanup-<日期>`。旧文档里的每一个路径和 SHA，都能在这个标签下原样找到（`git show archive/pre-cleanup-<日期>:<路径>`）。新主线只保留当前需要的内容，加一张路径对照表。这样既干净，历史也查得到。

---

## 2. 目标结构（B 段完成后）

**分支：**
- `main`：不动，实验全部结束后统一并入；
- `claude/sac-mpc-coupling-design-ns7g6i`：科学代码工作分支，在其上整理；
- `collab/spacecraft`：协作工作分支，不动；
- 其余 6 个分支打归档标签后删除。

**历史：** 全部以 `archive/*` 标签保存。

```
README.md                    项目概述、环境、快速开始
CLAUDE.md                    ≤150 行：任务、当前方法、规则、陷阱、入口、指针（历史移出）
env/ controllers/ dynamics/ estimation/ train/   主线代码（本次只删整文件，不改行为）
experiments/                 只留主线工具（约 10 个）
eval/                        只留代码
evidence/<实验ID>/           已发布的证据 JSON（原 eval/v3e、eval/stopping 等），按总账 ID 分目录
tests/                       只留覆盖在用模块的测试；全部通过
references/                  文献，加 README（每篇用来做什么、不能做什么，取自 CLAUDE.md 的表）
docs/
  current/                   当前有效：任务规格、方法规格、评估协议、复现说明
  HISTORY.md                 一页研究时间线：每一阶段的问题、做了什么、结论、为什么转向
  archive/
    00_single_phase_perception/     A1–A3、G0、感知
    01_precapture_task_pure_mpc/    任务规格演变、Pure MPC 时域、终端门缺陷
    02_coupling_interface_T6_T12/   接口单因素、T7–T12
    03_adaptive_round1_v2/          自适应第一轮、V2（含评估器缺陷勘误）
    04_v3_value_arbitration/        V3b–V3e、价值仲裁
    05_handoff_stopping_final/      B/C 阶段、停止选项、价值规则、工况筛选、final2
    handoffs/                       历代窗口交接与开场提示
    README.md                       各组结论摘要与路径对照
    ARTIFACT_INDEX.csv              从工作树移除的文件 → 归档标签路径
```

文档归组草案见 `upper/cleanup/DOC_CLASSIFICATION_DRAFT.csv`：140 篇全部按文件名归入以上 8 组，无未分类项。B 段执行前由上层逐条复核"状态说明"列。

---

## 3. A 段：只读盘点（现在就做，与评估并行）

### 3.0 安全规则（违反任何一条就停止并报告）

**禁止：**
- 在 `D:/py/DRL2` 中执行任何改变工作树或索引的 git 命令（checkout、pull、fetch、reset、stash、clean、merge、switch、commit 等）；
- 删除、移动、重命名、压缩、覆盖任何文件；
- 读写正在写入的目录。正在写入的是 `eval/final2/` 下的评估输出和 `logs/final2_262461`，对它们只能列文件名和大小，不算哈希；
- 运行任何实验、训练或评估脚本；
- 把盘点输出写进 `D:/py/DRL2`。

**允许：** 列目录；读文件大小和修改时间；读 manifest；对已结束运行的文件算 SHA-256（`BelowNormal` 优先级）；只读的 git 命令（`rev-parse`、`status`、`log`、`ls-files`、`worktree list`、`stash list`、`branch -vv`）。

**输出位置：** `C:/Users/35884/Documents/Spacecraft/过程文件/整理_20261010/`。需要入库的部分复制到协作工作树。

### 3.1 Git 状态

对每一个本地克隆或工作树（`D:/py/DRL2`、`D:/py/DRL2_v3e`、协作工作树，以及其他凡是含 `.git` 的目录）记录：
- 路径、HEAD、当前分支、`status --porcelain`、stash 列表、worktree 列表、remote 地址；
- `git status --porcelain --untracked-files=all` 的完整列表。

### 3.2 不在 git 中的代码（最重要的一项）

在各工作树的未跟踪文件和被忽略文件中，列出所有 `*.py`、`*.ps1`、`*.bat`、`*.ipynb`、`*.md`、`*.json`（配置类）。逐个标注：
- 是否产生过进入决策或汇报的数字（依据交接记录判断）；
- 建议处理方式：`COMMIT_TO_GIT`（产生过数字，必须入库）、`ARCHIVE_ONLY` 或 `SCRATCH`。

### 3.3 本地资产清单

逐个列出 `D:/py/DRL2/logs`、`D:/py/DRL2/eval`、`D:/py/DRL2_v3e/logs`、`过程文件/` 下的一级和二级目录：
- 大小、文件数、最早和最晚修改时间；
- 如果有 `manifest.json`：读出 code_commit、code_dirty、seed、method、regime、status、actual_decision_steps。

### 3.4 对应到总账

每个目录归入总账的一个实验 ID（H01–H06、I01–I02、E01–E10、F00–F05），或标为 `ORPHAN`。

每个目录给一个状态：`ACTIVE`（正在写）、`OFFICIAL`、`EXPLORATORY`、`INVALID`、`CACHE`（如 `__pycache__`、tensorboard 临时文件、`*.tmp`、`*.lock`、副本）或 `UNKNOWN`。

注意：F00（262450–262452）、I01、I02 是 `INVALID`，必须保留，但要和有效结果分开放。

### 3.5 哈希

对已结束运行的关键文件算 SHA-256：`manifest.json`、`train.monitor.csv`、`final_model.zip`、逐开局评估 JSON、readout。

已有记录的直接核对，不重复计算：
- `eval/v3e/final_training_audit_60k.json` 里的哈希；
- review 分支里的 `SHA256.txt`；
- 总账中已记录的哈希。

### 3.6 过程文件夹

把 `过程文件/` 下的文件分为四类：执行回执、运行记录、交接、临时，并列出每个文件对应的执行单。

### 3.8 本地试过的方向清单（防重走死胡同）

把交接记录、回执和本地运行目录里出现过的每一种方法变体、探针和参数尝试都列出来，包括没有进入总账的探索性尝试。

每条写明：名称、日期、运行目录、对应的执行单、结果一句话（有数字就带上数字，数字要能指向文件）、当时为什么停止。

输出 `tried_variants.csv`。不要判断它是否"值得重开"，这一项由上层在 `DEAD_ENDS.md` 里统一写。

### 3.7 交付

推送到协作分支 `lower/handoffs/cleanup_inventory_20261010/`，只提交 `docs/collaboration/`：
- `LOCAL_ASSET_MAP.md`：每个位置是什么、对应哪个实验、状态，配一张总表；
- `local_assets.csv`，列为：路径、字节数、文件数、实验 ID、状态、manifest 摘要、哈希文件位置、建议处理；
  - "建议处理"取值：`KEEP_ARCHIVE` / `KEEP_ACTIVE` / `INVALID_KEEP` / `CACHE_DELETABLE` / `COMMIT_TO_GIT` / `ASK_USER`；
- `untracked_code.csv`（3.2 的结果）；
- `tried_variants.csv`（3.8 的结果）；
- 补全总账的本地列：`local_result_path`、`local_sha_verified`、`lower_note`；
- 更新 `CURRENT.json`。

**停止条件：** 3.1–3.8 都有结果，所有目录都归入了实验 ID 或标为 `ORPHAN`/`CACHE`。不做任何移动和删除，不对任何结果做解读。

---

## 4. B 段：Git 整理（评估交付且实验停止后，经用户确认再执行）

**执行方：** 上层在科学分支上做提交；下层在本机验证。

**B0 冻结。** 确认以下三项：
- FINAL_RERUN 的评估全部交付，结果文件已入协作分支；
- 没有任何训练或评估进程在运行；
- 已拉取下层最后的推送。

**B1 打标签（只增不删，可随时撤销）：**
- 科学分支末端：`archive/pre-cleanup-<日期>`；
- 每个待删分支各一个标签：`archive/review/...`、`archive/v2-evaluation-20260924`、`archive/v3-t0b-20260925`。
- 所有标签都是附注标签，说明里写明它保存的内容。

**B2 在科学分支上整理，每一步一个提交：**
1. 从工作树移除 `logs/` 和 `local_artifacts/`。生成 `docs/archive/ARTIFACT_INDEX.csv`，包含路径、blob SHA、大小，以及取回命令。
2. 按归组表把文档移入 `docs/archive/<组>/`，每组写一份 README：问题、做法、结论和判定、关键数字及其出处、被什么取代。
3. 写 `docs/HISTORY.md`：由 CLAUDE.md 中各段 "Live status" 和 PROJECT_STATE 浓缩成一页时间线。
4. 写 `docs/current/`：任务规格、方法规格（价值停止）、评估协议（块、判据、配对检验、真值违规）、复现说明。
5. 删除不在用的代码文件：`experiments/` 约 70 个、`eval/` 约 15 个、`train/train.py`、`train/configs.py`、`train/callbacks.py`、`read_cal.py`，以及只测试这些文件的测试。生成 `docs/archive/CODE_INDEX.csv`（路径、最后提交、归档标签）。
   - **不改任何在用模块的行为。** env 中的旧任务分支（single_phase 等）留到根部优化时再处理。
6. 把证据 JSON 从 `eval/` 移到 `evidence/<实验ID>/`，同步更新总账路径。
7. 重写 CLAUDE.md 和 README.md；建立 `references/README.md`。
8. **写 `docs/DEAD_ENDS.md`（已验证的负结果台账）。** 每个试过而没有成立的方向一条，固定写明：试了什么（一句话）；在什么任务和接口上试的；结果数字和判定；证据出处（实验 ID 加归档标签路径）；失败的根因（已测得的，还是推断的）；什么条件变了才值得重开。上层起草，来源包括 CLAUDE.md 历史段落、各组 README，以及 A 段 3.8 的本地清单。以后任何方法改动，提出前先对照这份台账。
9. `CODE_INDEX.csv` 里每个被移除的脚本，都标上它服务的实验 ID 和那个实验的判定。被删的只是工作树里的文件，所有试过的代码都能从标签取回，并且知道它当时得出了什么结论。

**B2 验证（下层在本机做，全部通过才能进入 B3）：**
- 全部测试通过：`python -B -m pytest -q`；
- 行为不变：
  - 用整理后的代码重跑 271000 块前 3 个开局的 Pure MPC，以及 262460 的 stopping 行前 3 个开局；
  - 结果必须与正式 JSON 逐字段一致（完成、时间、Δv、交接步）。
- 链接检查：`docs/current/`、`HISTORY.md`、各组 README、总账里的每个路径，在新主线或归档标签下都能找到。

**B3 分支收口（远端操作，用户逐项确认）：**
1. 确认 6 个归档标签都已推到远端，并且每个标签都指向对应分支的末端提交。
2. 删除远端分支：`v2-evaluation-20260924`、`v3-t0b-20260925` 和 4 个 `review/*`。
3. 不合并，不动 `main` 和 `collab/spacecraft`。

完成后远端只剩 `main`、`claude/...`、`collab/spacecraft` 三个分支，外加 `archive/*` 标签。

---

## 5. C 段：本地整理（B 段完成后，经用户确认）

1. **新建干净工作副本，不在旧目录里清理。**
   - 命令：`git clone --filter=blob:limit=5m -b claude/sac-mpc-coupling-design-ns7g6i <仓库> D:/py/spacecraft`。部分克隆不会下载历史里 1.5 GB 的模型 ZIP，需要时再按需取。
   - 旧 `D:/py/DRL2` 和 `DRL2_v3e` 设为只读，原样保留，直到用户决定删除。
2. **建立本地归档** `D:/spacecraft_archive/<实验ID>/`，按 A 段的映射复制（不移动），每个文件复制后核对 SHA。作废运行单独放进 `_invalid/<实验ID>/`。
3. **再做一份离线备份**（移动硬盘或网盘）。只有 `MANIFEST_SHA256.csv` 进入 git。
4. **删除只由用户执行**，并且只能删 A 段标为 `CACHE_DELETABLE` 的条目。

---

## 6. 时间与分工

| 段 | 执行方 | 前提 | 预计 |
|---|---|---|---|
| A | 下层 | 现在开始，低优先级，不碰评估 | 半天 |
| B1–B2 | 上层提交，下层验证 | 评估交付且实验停止，用户确认 | 1–2 天 |
| B3 | 用户 | B2 验证全部通过 | 半小时 |
| C | 下层，删除由用户执行 | B3 完成 | 半天 |

B 段完成后，才进入整体审查和根部优化。审查以 `docs/current/`、`HISTORY.md` 和总账为起点。
