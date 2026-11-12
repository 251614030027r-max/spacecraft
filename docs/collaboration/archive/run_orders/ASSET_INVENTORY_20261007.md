# 执行单：只读资产盘点与证据链登记_20261007

状态：已授权待执行。
人类授权来源：用户 2026-10-07："先暂停下发我们整理下工作……给一份详细的整理清单我让下层执行，正好不干扰训练。"依据讨论窗口提案《项目资产与证据链整理提案》，上层审查后改写为本只读执行单。

| 模板项 | 内容 |
|---|---|
| 目的 | 在最终结果出来之前固定证据链：任何论文数字都能从证据矩阵查到实验，再从总账查到代码、模型、块、原始文件和审查结论 |
| 科学代码提交 | 不运行任何实验代码。训练工作树 `D:/py/DRL2` 只读 |
| 输入 | `PROJECT_STATE.md`、`registry/EXPERIMENT_REGISTRY_20261007.csv`（上层已按 Git 证据预填）、本地 `D:/py/DRL2` 与 `C:/Users/35884/Documents/Spacecraft` |
| 产出 | 补全的总账 CSV；`lower/LOCAL_ASSET_MAP_20261007.md`；`lower/handoffs/inventory_20261007.md`；更新 `CURRENT.json` |
| 预计 | 2–4 小时 |
| 停止条件 | 达到第 9 节五条就结束，不再扩张 |

## 上层对提案六个问题的决定

1. **同意：训练不停，只做只读盘点。**
2. **`FINAL_EVIDENCE_20261007` 暂缓下发。** 盘点交付后由用户下发，内容原样不变。它的 B、C 两段本来就要等训练结束，暂缓不影响进度。
3. **四件套压成三份。**
   - `PROJECT_STATE.md`：唯一当前入口，论文证据矩阵也放在里面；上层已写好。
   - 实验总账：CSV 加字段说明；上层已预填 23 行，下层补本地列。
   - `LOCAL_ASSET_MAP`：下层写。
4. **总账的收录范围。**
   - 每一个可能进入论文、或支撑过方向决策的实验，单独一行。
   - 旧路线（单阶段任务、T6/T7、T12、早期 Pure 时域对比、感知探针）按族各一行，标 `HISTORICAL`，指向原文档。
   - 逐开局证据留在原 review 分支，总账只链接，不复制。
5. **协作文档只在单独的协作工作树里编辑**，即 `Spacecraft/过程文件/协作/Git工作树`，**绝不在训练工作树切分支**。
6. **盘点完成后，`FINAL_EVIDENCE` 原样恢复。**

---

## 0. 安全规则（违反任何一条就停止并报告）

**禁止：**
- 在 `D:/py/DRL2` 中执行 `git checkout`、`pull`、`fetch`、`reset`、`stash`、`clean`、`rebase`、`merge`、`switch`、`commit`，以及任何改变工作树或索引的命令；
- 删除、移动、重命名、压缩、覆盖任何文件，包括 `*.lock`、`*.tmp`、旧逐开局 JSON 和作废结果；
- 读写 `logs/final_262450`、`final_262451`、`final_262452` 中除 `manifest.json`、`train.monitor.csv` 以外的文件，也不对它们算哈希；正在写入的检查点不碰；
- 运行任何实验、训练、评估或重放脚本；
- 把盘点脚本的输出写进 `D:/py/DRL2`。

**允许：**
- 列目录、计数、读文件大小和修改时间；
- 读 manifest；
- 对**已结束**运行的文件算 SHA-256；
- 在 `D:/py/DRL2` 中执行只读的 `git rev-parse`、`status`、`log`、`ls-files`、`worktree list`、`stash list`、`branch -vv`。

**输出位置：** 全部写到 `C:/Users/35884/Documents/Spacecraft/过程文件/盘点_20261007/`。需要入库的部分，复制到协作工作树。

**优先级：** 算哈希的进程用低优先级（`BelowNormal`），不能拖慢训练。

```powershell
$OUT = "C:/Users/35884/Documents/Spacecraft/过程文件/盘点_20261007"
New-Item -ItemType Directory -Force $OUT | Out-Null
```

## 1. 训练快照（只读）

```powershell
foreach ($s in 262450, 262451, 262452) {
  $m = Get-Content "D:/py/DRL2/logs/final_$s/manifest.json" | ConvertFrom-Json
  $rows = (Get-Content "D:/py/DRL2/logs/final_$s/train.monitor.csv").Count - 2
  $ck = Get-ChildItem "D:/py/DRL2/logs/final_$s/checkpoints" -Filter *.zip | Sort-Object LastWriteTime | Select-Object -Last 1
  "{0} status={1} commit={2} dirty={3} regime={4} rule={5} episodes={6} latest_ckpt={7}" -f $s,$m.status,$m.code_commit,$m.code_dirty,$m.regime.name,$m.stopping.stop_rule,$rows,$ck.Name
} | Tee-Object "$OUT/01_training_snapshot.txt"
Get-Process python -ErrorAction SilentlyContinue | Select-Object Id,StartTime,CPU,WorkingSet | Format-Table | Out-File "$OUT/01_processes.txt"
```

## 2. Git 状态（只读）

```powershell
Set-Location D:/py/DRL2
& { git rev-parse HEAD; git status --porcelain --untracked-files=no; git branch -vv; git worktree list; git stash list; git log --oneline -5 } *> "$OUT/02_git_experiment.txt"
git status --porcelain --untracked-files=all | Out-File "$OUT/02_untracked_files.txt"
```

在协作工作树中另记一份：`git log --oneline -3 origin/collab/spacecraft`、`git branch -r -vv`，写入 `$OUT/02_git_collab.txt`。在协作工作树执行 `fetch` 是允许的。

**核对：** 实验工程 HEAD 应为 `867a3a1…`，`git status --porcelain --untracked-files=no` 应为空。若不一致，**只记录、不修复**，写进交接。

## 3. 本地目录清单

对下列每个一级子目录，记录：路径、文件数、总大小（MB）、最新修改时间、其中被 Git 跟踪的文件数、是否仍在写入。

- `D:/py/DRL2/logs/*`
- `D:/py/DRL2/eval/*`，以及 `eval/` 下的第二级目录
- `C:/Users/35884/Documents/Spacecraft/` 下的 `上层交付`、`过程文件` 及其一级子目录

"是否仍在写入"的判断：最新修改时间在 30 分钟内，或属于 `final_2624*`。

```powershell
function Inventory($root) {
  Get-ChildItem $root -Directory | ForEach-Object {
    $files = Get-ChildItem $_.FullName -Recurse -File -ErrorAction SilentlyContinue
    $rel = $_.FullName.Replace("\","/").Replace("D:/py/DRL2/","")
    $tracked = if ($_.FullName -like "D:\py\DRL2*") { (git -C D:/py/DRL2 ls-files -- $rel | Measure-Object).Count } else { "n/a" }
    [pscustomobject]@{ path=$_.FullName; files=$files.Count; mb=[math]::Round(($files | Measure-Object Length -Sum).Sum/1MB,1);
      newest=($files | Sort-Object LastWriteTime | Select-Object -Last 1).LastWriteTime; git_tracked=$tracked;
      active=(($files | Sort-Object LastWriteTime | Select-Object -Last 1).LastWriteTime -gt (Get-Date).AddMinutes(-30)) }
  }
}
& { Inventory D:/py/DRL2/logs; Inventory D:/py/DRL2/eval; Get-ChildItem D:/py/DRL2/eval -Directory | ForEach-Object { Inventory $_.FullName };
    Inventory "C:/Users/35884/Documents/Spacecraft/上层交付"; Inventory "C:/Users/35884/Documents/Spacecraft/过程文件" } |
  Export-Csv "$OUT/03_directory_inventory.csv" -NoTypeInformation -Encoding UTF8
```

## 4. 关键文件 SHA-256（已结束的运行，低优先级）

逐个计算，并和总账或文档中记录的值对照，写入 `$OUT/04_hashes.csv`，列为：`path, sha256, expected, match`。

| 文件 | 期望值 |
|---|---|
| `logs/v3e_262420/final_model.zip`、`v3e_262421`、`v3e_262422` | 见 `eval/v3e/manifests/` 与 review `dc6682d` |
| `logs/stop_262430/final_model.zip` | `a3f36bde8589853efefc7abb24f13f3b068974badd3e9ad2b104742d272b0b3b` |
| `logs/stop_262431/final_model.zip` | `4b5b53ee4903e1af0d7534898360a3c0db003eb88b9582d55dbfefed33721b82` |
| `logs/stop_262432/final_model.zip` | `8b56adbf09d4a9753a040955324f3f75cda487bfcc77695644f16f6fdcccd74b` |
| `logs/stop_dev_262440/final_model.zip` | 见 review `6c5cb97` |
| `eval/v3e/stage_b/readout_b1.json` | `8bd00e8a5b4e514ac7173fe7d821e920593c6b2687eb2cd5f33c6af6b7e92aa0` |
| `eval/v3e/stage_b/readout_b1b2.json` | `2b84e14a19aec4d319862a781fbfa1f015fe69fcd698eb4471da5154cf5cbcd0` |
| `eval/v3e/stage_c/c2_report.json` | `7843be06702a0af8f07a3280fb40697225c45ed56f0990c08332a3a6cd610874` |
| 停止头正式轮 `readout.json`，本地原件 | 与 review `6c5cb97` 中的副本一致 |
| 价值规则检验 `readout_value.json`，本地原件 | `87495f9e14c2f75f759297fd395a47eb9681bf44d2fac9f9a6ffab86765cab7c` |
| `eval/regime_screen/readout_stage1.json`、`readout_final.json` | 与协作 `f3706a4` 中的 `SHA256.txt` 一致 |
| 停止头正式轮原始 ZIP | `94ab2368ea2d50d35aecc67230e0a6e37606dd5047d4ada202e316e33583f2ae` |

```powershell
$p = Start-Process powershell -ArgumentList "-NoProfile","-Command","Get-FileHash '<path>' -Algorithm SHA256 | Export-Csv -Append '$OUT/04_hashes_raw.csv'" -PassThru -WindowStyle Hidden
$p.PriorityClass = "BelowNormal"
```

上面是示例；也可以写一个循环逐个处理。**不对 `final_2624*` 算哈希**，它们的模型在训练完成后由正式流程登记。

## 5. 补全实验总账（在协作工作树中）

打开 `registry/EXPERIMENT_REGISTRY_20261007.csv`，每行补三列：

- `local_result_path`：本地原件位置；
- `local_sha_verified`：`yes` / `no` / `n/a`；
- `lower_note`：找不到的文件、与上层标注不同的意见、补充信息。

**不改上层填的状态和结论列。** 不同意的地方写在 `lower_note`。

## 6. 未登记资产

列出 `eval/`、`logs/` 以及 Spacecraft 目录中**不对应总账任何一行**的结果目录或读数文件。每项给出：
- 路径；
- 大致内容，例如训练运行、评估、诊断、临时文件；
- 建议归入哪一行或哪一类：`HISTORICAL` / `EXPLORATORY` / `INVALID` / 临时。

写进 `LOCAL_ASSET_MAP` 的"未登记"一节。若发现应进入论文、但总账漏掉的实验，单独标出，由上层补行。

## 7. 清理候选（只列不删）

列出以下几类，每项给路径、大小、理由：
- 重复副本，例如同一 ZIP 的多份；
- 非最终运行的中间检查点；
- 临时文件、残留锁文件。**残留锁只列出，不删除**。

每项的"可删除"一律写 `NO（待正式交付后再定）`。

## 8. 交付

在协作工作树中：
1. 新建 `lower/LOCAL_ASSET_MAP_20261007.md`，内容：
   - 第 3 节目录清单的摘要表；
   - 第 4 节哈希对照，不一致的单独列出；
   - 第 6 节未登记资产；
   - 第 7 节清理候选；
   - 第 1、2 节快照。
2. 提交补全后的 `registry/EXPERIMENT_REGISTRY_20261007.csv`。
3. 新建交接 `lower/handoffs/inventory_20261007.md`：
   - 做了什么；
   - 哈希不一致项；
   - 未找到的文件；
   - 对上层预填内容的异议摘要。
4. 更新 `CURRENT.json`：`phase` 保持 `formal_training_running`，增加 `"inventory_20261007": "delivered"`。
5. 先 `fetch`，再提交，**只提交 `docs/collaboration/`**；不强推。

原始的 `$OUT` 目录留在本地，交接里写明路径。

## 9. 结束条件（五条都满足就停，不再扩张）

1. 三个正式训练运行已登记，它们的状态快照在交接里；
2. 269000 筛选、B1 / B2 / C、停止头正式轮、价值规则检验的本地原件都能定位，哈希已对照；
3. 总账每一行都有本地路径或"未找到"的说明；
4. Git 分支角色与 `PROJECT_STATE.md` 第 4 节一致；如有出入，写在交接里；
5. 论文证据矩阵（`PROJECT_STATE.md` 第 3 节）的每一格，都能指到总账里的行，或标明"待最终数据"。

之后**不做**：重构仓库、统一历史文件名、删除分支或文件、为补格式重跑实验。
