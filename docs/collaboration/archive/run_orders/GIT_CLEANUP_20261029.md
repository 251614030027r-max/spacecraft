# 执行书：GIT_CLEANUP_20261029（下层；Git 资产整理，不涉及任何实验）

**状态：** 用户 2026-10-29 要求下层整理 Git。A、B 两部分可以直接执行；C 部分须用户逐项确认后再做。

**原则（所有步骤都适用）：**
- **不改写历史：** 不 rebase、不 filter-repo、不强推。已引用的 SHA 是证据链。
- **只在分支最新提交上删除或移动：** 删掉的文件仍留在历史里，按 SHA 可以取回。
  - 因此仓库的克隆体积不会变小，只有工作树（检出）变干净、变小。这是有意为之。
- **删除前先打标签：** 标签由下层推送，上层沙箱推不了标签。
- **逐步提交：** 每一步单独提交，提交信息写清删了 / 移了什么、为什么。推送前先 fetch。
- **不碰运行中的东西：** 不碰本机 `D:/py/DRL2` 中正在运行的检出；只用独立的文档检出操作。
- **整理完自检：**
  - 科学分支上 `python -B -m pytest -q tests/test_t2_stage.py` 退出码为 0（46 passed）；
  - collab 上 `docs/collaboration/` 内的链接不失效（至少 HANDOFF、PROJECT_STATE、README、研究演进）。

## 现状（上层 2026-10-29 只读检查）

| 分支 | 工作树 | 问题 |
|---|---|---|
| `collab/spacecraft`（`afed850`） | 3746 个文件，约 1.95 GB | 规定只放 `docs/collaboration/`，实际还带着 10 月清理前的整套旧仓库：`logs/` 约 1.71 GB（172 个模型 zip 共 1.58 GB，最后改动 09-22），`docs/` 下的旧目录，`eval/` 27 MB，`References/` 62 MB，`env/`、`train/`、`experiments/`、`tests/` 等旧科学代码（最后改动 10-06），`local_artifacts/`，根目录的 `read_cal.py`、`CLAUDE.md`、`requirements*`、`pytest.ini`。下层交付分两处：T2.1 及以前在 `docs/collaboration/lower/handoffs/`，T2.2 起在仓库根的 `lower/handoffs/` |
| `claude/sac-mpc-coupling-design-ns7g6i`（`8761ff7`） | 9531 个文件，约 665 MB | 总体有序。约 510 MB 是 T1-B 原始训练日志（`decisions.jsonl`、`model_prefix_releases.jsonl`，见 C1）；T1-B 各轮证据中有约 35 个模型 zip（约 30 MB），违反"模型只记 SHA"的规定 |
| `main`（`619e491`） | — | 不动 |
| 远端 | 只有上面 3 条分支，15 个标签 | 正常 |

## A. collab 分支：恢复"只放 `docs/collaboration/`"

**A0. 打保全标签：** 在当前 collab 头打 `archive/collab-pre-cleanup-20261029` 并推送，核验远端标签指向该 SHA。

**A1. 删除 `docs/collaboration/` 以外的旧内容（仅工作树）：**

```
logs/  eval/  References/  local_artifacts/  env/  train/  experiments/  tests/
controllers/  dynamics/  estimation/
docs/a2_guidance_free/  docs/a3_deployable_planning/  docs/g0_perception_closed_loop/  docs/handoffs/
read_cal.py  CLAUDE.md  requirements.txt  requirements-dev.txt  requirements-mpc.txt  pytest.ini
```

- **删除前核对：** 列出每个目录在科学分支上是否有对应副本，或已在旧标签 `archive/pre-cleanup-20261010` 中。结果写进清单 `docs/collaboration/lower/handoffs/git_cleanup_20261029/REMOVED.csv`，字段为：路径、blob SHA、大小、科学分支或标签中的对应位置。
- **处理 `docs/` 下的三个旧目录：** `a2_guidance_free`、`a3_deployable_planning`、`g0_perception_closed_loop` 若在科学分支的 `docs/archive/00_single_phase_perception/` 下已有同名内容，直接删；若没有，就挪到科学分支的同一位置后再删，并在清单中注明。
- **根目录 `README.md`：** 改成一句话，指向 `docs/collaboration/README.md`。
- **`.gitignore`：** 保留，内容精简为忽略 `docs/collaboration/` 以外的一切。

**A2. 统一下层交付位置：** 把仓库根的 `lower/handoffs/t2_*`（7 个目录）`git mv` 到 `docs/collaboration/lower/handoffs/`，然后删除空的根目录 `lower/`。
- 同步修改仍在使用的文档中的路径：HANDOFF 6.4、PROJECT_STATE 第 2–3 节、研究演进第七章、collab README。
- 已执行的执行书属于历史记录，路径不改。在 `docs/collaboration/lower/README.md` 加一张"旧路径 → 新路径"对照表即可。

**A3. `CURRENT.json`：** 更新到当前现场（T2.6W 已交付，T2_6W1 待批准，无运行中训练）。下层已计划做。

**A4. 总账：** 在 `registry/EXPERIMENT_REGISTRY_20261007.csv` 补 T2 段，下层列从 T2.2 填到 T2.6W。上层列由上层补，可以留空。

## B. 科学分支：轻量整理（不删证据）

**B1. 模型 zip 的例外登记：** 在 `evidence/` 中找出所有 `*.zip`（主要是 T1-B V1–V5.1 与 `T1_V2_S0` 下的 `models/`）。
- 每个 zip 核对同目录是否有 `MODEL_SHA256.json`，缺的补上。
- 不删除；在 `docs/archive/GIT_ASSETS_20261018.md` 中登记为已知例外。
- 从此以后，新交付不再提交模型 zip。

**B2. 下层的过程目录命名：** 各证据目录中的 `脚本/`、`记录/`、`输入/` 保留。以后统一用这三个名字，不再新增其它中文目录名。在 `docs/archive/GIT_ASSETS_20261018.md` 写一句约定。

**B3. 不动：** 代码目录、`docs/archive/`、`evidence/historical/`、各轮复用 SAC 时的评估副本。每个证据目录自成一体，副本有意保留。

## C. 需要用户逐项确认的可选项（未确认不做）

**用户裁决（2026-10-29）：C1、C2 都不做。** 本机磁盘吃紧时再议 C1。

**C1. 科学分支移除 T1-B 原始训练日志（工作树减少约 0.5 GB）。**
- **对象：** `evidence/T1B_V1/training/`、`T1B_V2_PILOT/training/`、`T1B_V3_MIDTERM_20261019/training/`、`T1B_V4_RESOURCE_RESTART1_20261020/training/`、`T1B_V51_ROUND/midterm_5000/training/`，以及 `T1B_V7_SIDE_BENCH` 中的大文件（> 1 MB）。
- **做法：** 先打 `archive/science-pre-cleanup-20261029`，再从最新提交中删除；在 `docs/archive/GIT_ASSETS_20261018.md` 记下每个文件的路径、blob SHA 与大小。各目录的汇总 JSON、README、SHA256SUMS 保留。
- **好处：** 本机 `D:/py/DRL2` 的检出明显变小。T1-B 已封存，没有任何现行代码读这些文件。
- **风险：** 想复查 T1-B 原始逐决策记录时，需要从标签取回。

**C2. collab 中 T1 时期的大交付 zip**（`docs/collaboration/lower/handoffs/` 下的 `T1_V4_S0_*.zip`、`T1_L_THREE_ARM_S0_FINAL.zip`、`T1_V2_S0_FINAL_*.zip`、`T1O_V2_GATES_LOCAL.zip`，约 80 MB）：处理方式同 C1。

## 上层自己的整理项（不交下层）

- 科学分支 `docs/DEAD_ENDS.md` 补 T1-N、T1-B、T2 期间的否决项（研究演进第六章已列）。
- `docs/HISTORY.md` 补 T2 段。
- `docs/archive/GIT_ASSETS_20261018.md` 补 T2 的提交与证据表。
- `docs/current/`（T1 时期规格）加一句"T2 以代码与 HANDOFF 为准"。

## 交付

- collab `docs/collaboration/lower/handoffs/git_cleanup_20261029/`：`REMOVED.csv`、`MOVED.csv`、`RESULT.md`。`RESULT.md` 写明：两个标签的 SHA、整理前后的文件数与工作树大小、自检结果（pytest 退出码、链接检查）。
- 只报告事实。
