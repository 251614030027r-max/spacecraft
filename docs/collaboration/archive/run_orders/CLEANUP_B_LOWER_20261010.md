# 执行单：整理 B 段的下层部分（打标签、删分支、收入脚本、验证）_20261010

状态：已授权待执行。
人类授权来源：用户 2026-10-10："我建议现在就开始整理不等实验结束，实验评估的资产后续整理到相应的模块即可，本地可以等实验跑完再整理"；以及"分支不并……主要是 claude 分支以及其他杂乱分析处理好"。

上层已在 `claude/sac-mpc-coupling-design-ns7g6i` 上完成整理提交（见第 4 节）。以下几步上层做不了，因为会话代理禁止推送标签和删除远端分支。

## 0. 安全规则

- **不在 `D:/py/DRL2` 里做任何 git 操作。** 评估仍在那里运行，固定在 `f2f8169`。
- 第 1–2 步在协作工作树或一个新的临时克隆中执行，只操作远端引用，不碰任何工作树。
- 第 3 步只写协作分支的 `docs/collaboration/lower/`。
- 第 5 步等评估全部交付、进程停止后再做。

## 1. 打归档标签（附注标签，必须先于第 2 步）

```bash
git fetch origin --prune
git tag -a archive/pre-cleanup-20261010 813e27f982ee24076d7ef9ae2cbb6061e0881c42 -m "Frozen science branch before the 2026-10-10 cleanup; every pre-cleanup path and SHA resolves here."
git tag -a archive/v2-evaluation-20260924                 736edbca691336efe8e5296bc2f10d8a39e35190 -m "Archived branch v2-evaluation-20260924"
git tag -a archive/v3-t0b-20260925                        f50eb78350604431fb042eca6b3f655cd5c26bab -m "Archived branch v3-t0b-20260925"
git tag -a archive/review/v3e-60k-value-r2-20260930       dc6682d4ec11e5858bedc9e619ac6c4142fc7e66 -m "Archived branch review/v3e-60k-value-r2-20260930 (E01 evidence)"
git tag -a archive/review/v3e-stage-c-20261002            b383cd285de78561f116b1dc6cdab10fe3694088 -m "Archived branch review/v3e-stage-c-20261002 (E04 evidence)"
git tag -a archive/review/stopping-20261006-audit         6c5cb97a319e425bd616d4fe323ed1a9c43902a4 -m "Archived branch review/stopping-20261006-audit (E06 evidence)"
git tag -a archive/review/stopping-value-check-20261006   2d8a5818037b6d753f4866969fae372da6cbef84 -m "Archived branch review/stopping-value-check-20261006 (E09 evidence)"
git push origin "refs/tags/archive/*"
```

**核对**：执行 `git ls-remote --tags origin "archive/*"`，应列出 7 个标签。其中 6 个分支标签，`^{}` 解引用后必须等于 `git ls-remote origin <分支>` 给出的分支末端 SHA，逐个比对。**任何一个不一致，就停止，不做第 2 步。**

## 2. 删除 6 个远端分支（只在第 1 步全部核对通过后）

```bash
git push origin --delete v2-evaluation-20260924 v3-t0b-20260925 review/v3e-60k-value-r2-20260930 review/v3e-stage-c-20261002 review/stopping-20261006-audit review/stopping-value-check-20261006
git ls-remote --heads origin   # 应只剩 main、claude/sac-mpc-coupling-design-ns7g6i、collab/spacecraft
```

不动 `main`、`claude/...` 和 `collab/spacecraft`。

## 3. 收入未入库的流水线脚本（A 段 3.2 中的 106 个 `COMMIT_TO_GIT`）

这些脚本是各阶段的启动、打包和核验脚本，属于证据链，但从未进入 git。

**去重规则：**
- 对每个文件算 SHA-256。
- 如果文件内容与 `archive/pre-cleanup-20261010` 中某个已跟踪文件逐字节相同（例如 `frozen/stopping.py`、`frozen_code/experiments/v3_stage_c.py`），**不复制**，只在清单里记录它对应的 git 路径和提交。
- 同一内容出现多份时，只保留一份。

**放置位置：** `docs/collaboration/lower/pipeline_scripts/<实验ID>/<原文件名>`。实验 ID 用 `untracked_code.csv` 里的值；`ORPHAN` 的放进 `_unassigned/`。

**同时提交** `docs/collaboration/lower/pipeline_scripts/INDEX.csv`，列为：原本地路径、SHA-256、实验 ID、处理方式（`copied` / `identical_to_git:<路径>@<提交>` / `duplicate_of:<路径>`）、对应的执行单或交接。

**不收入** `ARCHIVE_ONLY` 和 `SCRATCH` 两类。

只提交 `docs/collaboration/`，先 fetch 再提交，不 force-push。

## 4. 上层已完成的整理（供核对）

科学分支上的整理提交为 `b617e9a`、`9387360`、`3e8b48e`、`e62e50d`、`dccee28`，起点为 `813e27f`，末端为 `dccee28`：

1. 移除 `logs/`、`local_artifacts/`（1618 个文件）。索引见 `docs/archive/ARTIFACT_INDEX.csv`。
2. 文档按阶段归入 `docs/archive/<阶段>/`。对照见 `docs/archive/PATH_MAP.csv`。
3. 删除不在用的脚本和对应测试（82 个）。索引见 `docs/archive/CODE_INDEX.csv`。证据 JSON 从 `eval/` 移到 `evidence/`。
4. 新文档：`docs/current/`、`docs/HISTORY.md`、`docs/DEAD_ENDS.md`，以及重写的 CLAUDE.md 和 README。

**在用代码**（`env/`、`controllers/`、`dynamics/`、`estimation/`、`train/` 以及保留的 `experiments/` 和 `eval/metrics.py`）与 `f2f8169` 逐字节相同。上层已用 `git diff --name-status f2f8169 dccee28 -- env controllers dynamics estimation train experiments eval` 核对：只有删除，没有修改。测试结果：整理前 389 个通过、3 个预期失败；整理后 354 个通过、3 个预期失败，少掉的 35 个来自被移除的 6 个测试文件。

## 5. 整理后的验证（评估全部交付、进程停止后）

1. 新建干净克隆：

   ```bash
   git clone --filter=blob:limit=5m -b claude/sac-mpc-coupling-design-ns7g6i <仓库> D:/py/spacecraft
   ```

   **不要**在 `D:/py/DRL2` 里切换分支。

2. 在 `D:/py/spacecraft` 中运行 `python -B -m pytest -q`，必须全部通过。

3. 行为不变检查：用整理后的克隆重跑 271000 块前 3 个开局，两行：Pure，以及 262460 的 stopping（模型用 `D:/py/DRL2/logs/final2_262460/final_model.zip`）。完成、时间、Δv、交接步、违规必须与 F01 正式 JSON 逐字段一致。

4. 结果写入 `lower/handoffs/cleanup_b_verify_<日期>.md`，并更新 `CURRENT.json`。

## 6. 之后

C 段本地归档：按 `PROJECT_CLEANUP_20261010` 第 5 节执行，等用户确认后再开始。
