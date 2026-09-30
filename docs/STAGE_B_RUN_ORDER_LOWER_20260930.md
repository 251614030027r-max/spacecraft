# 下层执行单：阶段 B1 交接时段扫描（2026-09-30）

*上层窗口写，用户批准后转下层。判据的唯一来源是 `docs/STAGE_B_HANDOFF_WINDOW_PREREGISTRATION_20260930.md`；本文件只把执行步骤写全。两者冲突时，以预注册为准，并停下报上层。*

---

## 0. 先读这一段：方向变了什么

第二轮价值仲裁的判定是"不成立"，**这条线已经收口**，不开第三轮，不再补 V_L 数据。

用户已经拍定新的推进顺序：

> A 工具与预注册（已完成）→ **B 机制测量（本次）** → C 新的协调价值 → D 唯一一次重训 → E 全冻结正式评估

协调问题改成了：**"继续让学习策略飞（以后仍可交接），还是现在就交给 Pure MPC？"**

本次 B1 只测一件事：

> 学习策略失败的回合里，是否存在"在这段时间交给 Pure MPC，就能干净完成"的时段，而且这个时段不只是一个决策宽的偶然点？

做法：用三个已有的 60k 模型，在两个开局块上，对学习策略每一条失败轨迹的**每一个决策点**都试一次"从这里交给 Pure MPC 跑到底"，记录结局。

**本次不做的事**：
- 不训练；
- 不改 SAC、奖励、任务、Pure MPC、接口（包括 0.40 m 上限）；
- 不跑仲裁，不拟合价值；
- 不自己设阈值，不解释判定。

有问题就停下，报上层。

---

## 1. 前提检查

在 `D:\py\DRL2` 下打开 PowerShell：

```powershell
Set-Location D:\py\DRL2
$py = "D:/py/DRL2/.venv/Scripts/python.exe"

# 1a 迁移已完成（REPOSITORY_PROMOTION_20260930.json 中 status=completed）
# 1b 三个模型和对照文件都在原处，以下 7 项都应为 True：
Test-Path logs/v3e_262420/final_model.zip, logs/v3e_262421/final_model.zip, logs/v3e_262422/final_model.zip
Test-Path eval/v3e/pure_mpc.json, eval/v3e/262420/learned_only.json, eval/v3e/262421/learned_only.json, eval/v3e/262422/learned_only.json
# 1c 每个模型的 M2 文件都在（每行都应为 True）：
foreach ($m in "262420","262421","262422") { foreach ($x in "a","b","c","d") { Test-Path "eval/v3e/$m/m2_$x.json" } }
```

任何一项为 False，就停下报上层，不要从别处拷贝替代文件。

## 2. 同步代码并核对（任一不过就停）

```powershell
git pull --ff-only origin claude/sac-mpc-coupling-design-ns7g6i
git rev-parse HEAD                               # 抄进 REPORT.md；B1 全程必须在这一个提交上
git status --porcelain --untracked-files=no      # 必须没有输出
git diff c84ed74 HEAD --stat -- env controllers dynamics train/train_hybrid.py train/v3_values.py experiments/v3_common.py
                                                 # 必须没有输出：物理系统与第二轮完全相同
$env:OMP_NUM_THREADS=1; $env:MKL_NUM_THREADS=1; $env:OPENBLAS_NUM_THREADS=1
& $py -B -m pytest -q tests/test_v3_handoff_scan.py tests/test_v3_handoff_readout.py tests/test_v3_mainline.py tests/test_controller_reset_determinism.py tests/test_evaluator_observation_parity.py
                                                 # 必须全部通过（约 5 分钟）
```

参考：上层沙箱在同一提交上的全套测试结果是 353 passed、3 xfailed。

## 3. 实机核对：一条最短的失败轨迹（约 1–2 小时）

262420 在开局 262006 上第 26 个决策就失败了（视场）。Pure MPC 在这个开局上也失败（超时）。输出直接写进正式目录，B1 会复用，不会重算。

```powershell
& $py -u -B -m experiments.v3_handoff_scan --run-dir logs/v3e_262420 --seeds 262006 --scan failures --output-dir eval/v3e/stage_b/262420
```

跑完后运行下面的自动核对，把输出原样抄进 REPORT.md：

```powershell
@'
import json
s = json.load(open("eval/v3e/stage_b/262420/seed_262006.json"))
pure = {r["seed"]: r for r in json.load(open("eval/v3e/pure_mpc.json"))["records"]}[262006]
formal = {r["seed"]: r for r in json.load(open("eval/v3e/262420/learned_only.json"))["records"]}[262006]
k0 = s["handoffs"][0]
checks = {
    "learned fails at decision 26 (as formal)": s["learned_full"]["decisions"] == 26 == formal["decisions"]
        and not s["learned_full"]["completed"] and not formal["completed"],
    "learned survival equals formal": s["learned_full"]["survival_s"] == formal["survival_s"],
    "prefix verification is [0, 13, 25]": s["verified_prefix_ks"] == [0, 13, 25],
    "k=0 completed equals Pure MPC": k0["completed"] == pure["completed"],
    "k=0 decisions equal Pure MPC": k0["decisions"] == pure["decisions"],
    "k=0 survival equals Pure MPC": k0["survival_s"] == pure["survival_s"],
    "26 handoffs scanned": [h["k"] for h in s["handoffs"]] == list(range(26)),
    "clean checkout": s["code_dirty"] is False,
}
for name, ok in checks.items():
    print(("PASS " if ok else "FAIL ") + name)
print("ALL_PASS" if all(checks.values()) else "STOP_AND_REPORT")
'@ | & $py -B -
```

- 输出 `ALL_PASS`：**直接进入第 4 步，不用等上层回复**，同时给用户报一句"262006 核对通过，B1 已启动"；
- 输出 `STOP_AND_REPORT`，或扫描报错：停下，把核对输出和报错交给上层。**不要**进入第 4 步。

## 4. B1 正式扫描（6 个进程，约 26–48 小时）

每个进程都用同一份种子列表。已经做完或已被其他进程认领（`seed_<开局>.lock`）的开局会自动跳过，所以同一个模型开几个进程，就能自动分担工作。按计算量分配：262420 开 1 个，262421 开 3 个，262422 开 2 个。

```powershell
Set-Location D:\py\DRL2
$py = "D:/py/DRL2/.venv/Scripts/python.exe"
$env:OMP_NUM_THREADS=1; $env:MKL_NUM_THREADS=1; $env:OPENBLAS_NUM_THREADS=1
$seeds = "262000-262047,270000-270047"
New-Item -ItemType Directory -Force eval/v3e/stage_b_logs | Out-Null
$plan = @(@{m="262420"; n=1}, @{m="262421"; n=3}, @{m="262422"; n=2})
foreach ($j in $plan) {
  for ($i = 1; $i -le $j.n; $i++) {
    Start-Process -FilePath $py -WorkingDirectory "D:\py\DRL2" -WindowStyle Hidden -PassThru `
      -ArgumentList "-u","-B","-m","experiments.v3_handoff_scan","--run-dir","logs/v3e_$($j.m)","--seeds",$seeds,"--scan","failures","--output-dir","eval/v3e/stage_b/$($j.m)" `
      -RedirectStandardOutput "eval/v3e/stage_b_logs/b1_$($j.m)_$i.out.log" `
      -RedirectStandardError  "eval/v3e/stage_b_logs/b1_$($j.m)_$i.err.log"
    Start-Sleep -Seconds 20
  }
}
```

**查看进度**（随时可以运行；每个模型做完是 96/96）：

```powershell
foreach ($m in "262420","262421","262422") { "{0}: {1}/96 done, {2} in progress" -f $m, (Get-ChildItem eval/v3e/stage_b/$m -Filter seed_*.json).Count, (Get-ChildItem eval/v3e/stage_b/$m -Filter seed_*.lock).Count }
Select-String -Path eval/v3e/stage_b_logs/*.err.log -Pattern "Traceback|RuntimeError|Error"
```

学习策略成功的开局几分钟就写出来；失败的开局要逐点扫描，一个 150 决策的超时回合单进程约需 4–7 小时。进度不均匀是正常的。

**停止条件**：
- **任一进程报 `RuntimeError`**，例如 "the two learned passes differ" 或 "snapshot handoff differs from the prefix-recompute path"：停下所有扫描进程，报上层，附上开局号、报错全文和日志。**不得**关掉核对（`--verify-prefix ""`）重跑，也不得删掉该开局重跑；
- **进程因别的原因退出**（断电、内存、手动中断）：先确认没有任何扫描进程在运行，再删掉 `eval/v3e/stage_b/*/` 下残留的 `*.lock`，然后用第 4 步同样的命令重新启动。已写出的开局不会重算。在 REPORT.md 里记下中断时间和原因；
- 不要在扫描期间 `git pull` 或改动工作树（每个结果文件都记录提交号，判读会检查全部一致、且工作树干净）。

## 5. B1 判读（三个模型都到 96/96 后）

```powershell
$F = "eval/v3e"
& $py -B -m experiments.v3_handoff_readout `
  --scan 262420=$F/stage_b/262420 --scan 262421=$F/stage_b/262421 --scan 262422=$F/stage_b/262422 `
  --formal-learned 262420=$F/262420/learned_only.json --formal-learned 262421=$F/262421/learned_only.json --formal-learned 262422=$F/262422/learned_only.json `
  --formal-pure $F/pure_mpc.json `
  --m2 "262420=$F/262420/m2_a.json,$F/262420/m2_b.json,$F/262420/m2_c.json,$F/262420/m2_d.json" `
  --m2 "262421=$F/262421/m2_a.json,$F/262421/m2_b.json,$F/262421/m2_c.json,$F/262421/m2_d.json" `
  --m2 "262422=$F/262422/m2_a.json,$F/262422/m2_b.json,$F/262422/m2_c.json,$F/262422/m2_d.json" `
  --seeds "262000-262047,270000-270047" --output $F/stage_b/readout_b1.json
```

屏幕会打印 `verdict`，只有三种。**下层不解释判定，照下表办：**

| verdict | 含义 | 下层做什么 |
|---|---|---|
| `FIDELITY_FAIL` | 工具或数据的精确性核对没过 | 交付 B1 包（第 7 节），附 `fidelity_problems` 全文，停下 |
| `STOP` | 不到 2 个模型有稳定、非退化的交接时段 | 交付 B1 包，停下 |
| `PROCEED` | 至少 2 个模型有 | 交付 B1 包，然后按第 6 步执行 B2-lite |

判读只看 `readout_b1.json`。不要人工挑案例，也不要用其他统计方式补充结论。

## 6. B2-lite（只有 B1 为 `PROCEED` 才执行；约 8–14 小时）

> 修订（2026-09-30，B1 结果出来之前提交）：原来的全量 B2（两个块、步长 5、约 30–54 小时）作废，改为 B2-lite。依据是预注册第 6 节。

B2-lite 只扫 262000–262047 块上学习策略**干净完成**的回合（三个模型共约 113 条），每 10 个决策试一次交接。它不决定是否进入阶段 C，只补另一边的数据："此时继续学习更好"的状态，以及"是否更早交给 MPC 反而更快更省"。

### 6.1 先交付 B1，再拉代码

B1 包交付之后（第 7 节），而且所有 B1 扫描进程都已结束，才执行：

```powershell
git pull --ff-only origin claude/sac-mpc-coupling-design-ns7g6i
git rev-parse HEAD                               # 抄进 B2 的 REPORT.md
git status --porcelain --untracked-files=no      # 必须没有输出
# 扫描工具与物理系统相对 B1 的提交 a714c61 没有任何改动（必须没有输出）：
git diff a714c61 HEAD --stat -- env controllers dynamics train experiments/v3_common.py experiments/v3_handoff_scan.py experiments/evaluate_hybrid_policy.py
# 应该只列出文档、判读脚本和它的测试：
git diff a714c61 HEAD --stat
& $py -B -m pytest -q tests/test_v3_handoff_readout.py tests/test_v3_handoff_scan.py
```

任何一项不符就停，报上层。

### 6.2 扫描（6 个进程，每个模型 2 个）

```powershell
$env:OMP_NUM_THREADS=1; $env:MKL_NUM_THREADS=1; $env:OPENBLAS_NUM_THREADS=1
foreach ($m in "262420","262421","262422") {
  for ($i = 1; $i -le 2; $i++) {
    Start-Process -FilePath $py -WorkingDirectory "D:\py\DRL2" -WindowStyle Hidden -PassThru `
      -ArgumentList "-u","-B","-m","experiments.v3_handoff_scan","--run-dir","logs/v3e_$m","--seeds","262000-262047","--scan","successes","--stride","10","--verify-prefix","middle","--output-dir","eval/v3e/stage_b2/$m" `
      -RedirectStandardOutput "eval/v3e/stage_b_logs/b2_${m}_$i.out.log" `
      -RedirectStandardError  "eval/v3e/stage_b_logs/b2_${m}_$i.err.log"
    Start-Sleep -Seconds 20
  }
}
```

每个模型做完是 **48/48**。查看进度和停止条件同第 4 步，把目录换成 `stage_b2`、分母换成 48。

### 6.3 判读

三个模型都到 48/48 后，重新运行第 5 步的判读命令，在末尾加上下面三项，并把输出改为 `$F/stage_b/readout_b1b2.json`（判读脚本拒绝覆盖已有文件）：

```
--b2 262420=$F/stage_b2/262420 --b2 262421=$F/stage_b2/262421 --b2 262422=$F/stage_b2/262422
```

屏幕会打印以下几项，逐项核对：
- `verdict` 必须与 `readout_b1.json` 相同（B2 不改变 B1 的判定）；
- `b2_valid` 必须为 `true`。B2 的核对包括：每个模型 48 个开局齐全、同一提交、工作树干净、步长 10、每回合有前缀核对、学习策略结果与 B1 同一开局逐位相同；
- `b2_valid` 为 `false` 时，交付时附上 `readout_b1b2.json` 中 `b2.fidelity_problems` 的全文，停下，不做任何补跑。

## 7. 交付

沿用第二轮的做法：
- **zip 包**，内含：
  - `REPORT.md`：提交号；第 2 步各项输出；第 3 步核对输出；各进程起止时间；中断和处理；判读打印的 verdict；
  - `readout_b1.json`（以及 B2 的 `readout_b1b2.json`）；
  - `eval/v3e/stage_b/*/seed_*.json`（以及 `stage_b2`）全部逐开局文件；
  - `eval/v3e/stage_b_logs/` 全部日志；
  - `JSON_SHA256.txt`、`ALL_FILES_SHA256.txt`；
- **审查分支** `review/v3e-stage-b1-<日期>`：只放 `REPORT.md`、`readout_b1.json` 和 SHA 清单；
- **GitHub release**：放完整 zip。

模型文件只报路径和 SHA-256，不打包。并行耗时不作为实时性结论。

B2-lite 完成后另交一个包（内容同上，加 `readout_b1b2.json` 与 `stage_b2` 逐开局文件），分支和 release 名里用 `b2`。
