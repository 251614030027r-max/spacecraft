# 执行书：RUN_PRETEST（EACH 预实验；下层本机；约 14 小时，硬上限 15 小时；自检约 40 分钟）

**状态：上层起草（研究层 11-05 定稿意见 `final_review`，用户 11-05 要求一次性落实）；用户转发本书即下发。** 本轮是最终方法冻结后**唯一**一次预实验，只检验实验有效性并记录研究观察，不调参、不改方法。通过后停止开发；正式实验（≥ 5000 决策 × 3 种子）由用户另行授权，见 `RUN_FORMAL.md`（草案，未授权）。

**可直接转发给下层的一句话：**

> 读 collab `docs/collaboration/upper/run_orders/RUN_PRETEST.md`，在 `D:\py\DRL2` detached 检出 `d1523cebb3682336c20fec70fa0e187d7b6e97c6`，按第 0 节自检（含全流程 dryrun），全部通过后用带 BOM 的 UTF-8 存第 1 节脚本为 `run.ps1` 并隐藏运行，跑完按"交付"一节提交证据与 RESULT.md，不改任何代码、配置与判据。

**科学提交（冻结）：** `d1523cebb3682336c20fec70fa0e187d7b6e97c6`（detached 检出）。
- 环境 t2-stage-0.13.0、学习器 t2-learner-7.0；
- 主方法路径与 T2.12 `e_priv`（`2e7d838`）逐位等价（`evidence/final_pretest/precheck/`）；
- 方法与配置：`docs/final_method/`（`METHOD_MINIMAL.md`、`METHOD_SPEC.md`、`C1_THEORY.md`、`CONFIG_LOCK.md`、`ABLATION_MATRIX.md`、`FIGURE_SPEC.md`、`METHOD_REGISTRY.json`）；
- 唯一入口：`experiments/final.py`、`experiments/final_figures.py`。它们只能启动注册表里的配置、本阶段的种子与开局，历史反事实分支不可能被误启动。

## 本轮做什么

| 部分 | 内容 | 规模 |
|---|---|---|
| 训练 | EACH、SAC、SAC + planner features，各 2 种子（10、11）× 2000 决策；训练开局从 11,000,000 与 12,000,000 起，从未用过 | 6 条，并行 |
| 评估 | 6 个 2000 步模型；EACH 两个模型另跑刚性承诺（窗口配对）；固定策略 pure、rally、c_10_0；rally、c_10_0 另跑刚性承诺 | 评估开局 294100–294147（48 个，从未用过） |
| 作图数据 | 几何扫描（图 A）、轨迹（图 B）；开局 294200–294211，不用于任何模型选择 | 小 |
| 其余三组训练消融 | DUP、NO_GEOM、RIGID 只在自检 dryrun 中走通配置，正式实验再训练 | — |
| 图与表 | `experiments.final_figures` 从原始日志一键生成 | — |

## 0. 准备与自检（任一项不满足就停下报告）

```powershell
cd D:\py\DRL2; git fetch origin claude/sac-mpc-coupling-design-ns7g6i; git checkout --detach d1523cebb3682336c20fec70fa0e187d7b6e97c6
$py  = "D:\py\DRL2\.venv\Scripts\python.exe"
$Out = "C:\Users\35884\Documents\Spacecraft\过程文件\final_pretest"
if ((git rev-parse HEAD) -ne "d1523cebb3682336c20fec70fa0e187d7b6e97c6") { throw "HEAD 不是冻结 SHA" }
if (git status --porcelain --untracked-files=no) { throw "科学工作树有改动" }
if ((Test-Path "$Out\B") -and @(Get-ChildItem "$Out\B" -Force).Count -gt 0) { throw "输出目录非空：$Out\B" }
New-Item -ItemType Directory -Force "$Out\B", "$Out\C", "$Out\S", "$Out\G", "$Out\T", "$Out\DRY\B", "$Out\DRY\C", "$Out\DRY\S" | Out-Null
$env:OMP_NUM_THREADS = "1"; $env:MKL_NUM_THREADS = "1"; $env:OPENBLAS_NUM_THREADS = "1"
& $py -B -m pytest -q tests/test_t2_stage.py tests/test_integrator.py *> "$Out\pytest.log"; $pt = $LASTEXITCODE   # 以退出码 0 判定
if ($pt -ne 0) { throw "pytest 退出码 $pt" }
& $py --version *> "$Out\ENV_LOCK.txt"; & $py -m pip freeze *>> "$Out\ENV_LOCK.txt"
$old = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'experiments\.' }
if ($old) { throw "有遗留的 python 进程：$($old.ProcessId -join ',')" }
# ---- 全流程 dryrun：6 组配置 × 2 种子 × 40 决策（smoke 学习器设置），评估、筛选、刚性配对、检查、汇总、出图，约 30–40 分钟 ----
$dryJobs = @()
foreach ($cfgName in "EACH","SAC","SAC_ENG","DUP","NO_GEOM","RIGID") { foreach ($seedNo in 10,11) {
  $dryJobs += Start-Process -FilePath $py -WorkingDirectory "D:\py\DRL2" -WindowStyle Hidden -PassThru `
    -ArgumentList "-B","-m","experiments.final","train","--stage","dryrun","--config",$cfgName,"--seed",$seedNo,"--output","$Out\DRY\B\TRAIN_${cfgName}_s$seedNo" `
    -RedirectStandardOutput "$Out\DRY\B\TRAIN_${cfgName}_s$seedNo.out.log" -RedirectStandardError "$Out\DRY\B\TRAIN_${cfgName}_s$seedNo.err.log" } }
$dryJobs | Wait-Process
foreach ($cfgName in "EACH","SAC","SAC_ENG","DUP","NO_GEOM","RIGID") { foreach ($seedNo in 10,11) {
  & $py -B -m experiments.final eval --stage dryrun --run "$Out\DRY\B\TRAIN_${cfgName}_s$seedNo" --label "${cfgName}_s$seedNo" --output "$Out\DRY\C\${cfgName}_s$seedNo" --workers 4 *>> "$Out\DRY\eval.log" } }
foreach ($seedNo in 10,11) { & $py -B -m experiments.final eval --stage dryrun --run "$Out\DRY\B\TRAIN_EACH_s$seedNo" --label "EACH_s${seedNo}_rigid" --output "$Out\DRY\C\EACH_s${seedNo}_rigid" --rigid --workers 4 *>> "$Out\DRY\eval.log" }
foreach ($polName in "pure","rally","c_10_0") { & $py -B -m experiments.final screen --stage dryrun --policy $polName --output "$Out\DRY\S\$polName" --workers 4 *>> "$Out\DRY\screen.log" }
& $py -B -m experiments.final screen --stage dryrun --policy rally --rigid --output "$Out\DRY\S\rally_rigid" --workers 4 *>> "$Out\DRY\screen.log"
& $py -B -m experiments.final check --stage dryrun --root "$Out\DRY" --output "$Out\DRY\CHECK.json" *> "$Out\DRY\check.log"
if (-not (Get-Content "$Out\DRY\CHECK.json" -Raw | ConvertFrom-Json).complete) { throw "dryrun 不完整，见 $Out\DRY\CHECK.json" }
& $py -B -m experiments.final_figures --stage dryrun --root "$Out\DRY" --output "$Out\DRY\figures" *> "$Out\DRY\figures.log"
if ($LASTEXITCODE -ne 0) { throw "dryrun 出图失败，见 $Out\DRY\figures.log" }
foreach ($cfgName in "EACH","SAC","SAC_ENG","DUP","NO_GEOM","RIGID") {
  $mf = Get-Content "$Out\DRY\B\TRAIN_${cfgName}_s10\manifest.json" -Raw | ConvertFrom-Json
  "$cfgName arm=$($mf.arm) obs=$($mf.obs_profile) actor=$($mf.actor_view) critic=$($mf.critic_view) window=$($mf.window) geometry=$($mf.geometry) k=$($mf.k_candidates) beta=$($mf.model_beta) sha=$($mf.git_head)" | Out-File -Encoding utf8 -Append "$Out\DRY\MANIFESTS.txt" }
```

**自检通过的标准：**
- pytest 退出码 0；
- `DRY\CHECK.json` 中 `complete: true`；
- `DRY\figures` 生成了 `fig_training`、`fig_final`、`fig_window` 与 `TABLES.json`；
- `DRY\MANIFESTS.txt` 六行与 `docs/final_method/ABLATION_MATRIX.md` 表 1 一致：
  - EACH：e_priv / reduced_priv / 24 / 48 / True / True；
  - SAC：e_sac / reduced_priv / 24 / 24；
  - SAC_ENG：e_sac / engineered / 24 / 24；
  - DUP：e_priv / reduced_dup / 24 / 48；
  - NO_GEOM：geometry False；
  - RIGID：window False；
  - 全部 k = 0、beta = 0。

dryrun 的数值没有任何意义，只检查流程。

## 1. 一次性脚本

存为 `$Out\run.ps1`（带 BOM 的 UTF-8），用 `powershell -ExecutionPolicy Bypass -WindowStyle Hidden -File "$Out\run.ps1"` 运行。变量名都不与 `$Out` 重名。

**机时依据（下层实测）：** 4 条并行时 2000 步需 6.0–7.7 小时（T2.17、T2.19），整机约 1040 决策 / 小时。6 条 × 2000 决策，训练约 11.5 小时；评估约 13 组 × 48 个开局，约 2.5 小时。

**时间规则（从脚本启动算起）：**
- 第 12 小时仍未训练完：结束全部训练进程，记录为"训练超时"（该轮有效性不过）；
- 评估按优先级依次进行：
  1. 6 个训练模型；
  2. EACH 刚性配对；
  3. 几何扫描；
  4. 固定策略；
  5. 固定策略刚性配对；
  6. 轨迹；
- 第 14 小时之后不再开始新的评估项，直接进入检查、汇总与出图；
- 第 15 小时是硬上限。

**启动检查：** 六条训练启动 3 分钟后都应仍在运行，否则全部结束并报错。

**训练崩溃（进程退出但没有 `finished.json`）：**
- 前 3 小时内：把该目录改名为 `*_crash1`，同配置同种子从零重启一次；
- 3 小时之后，或第二次仍崩溃：不再重启，如实记录。

```powershell
$py  = "D:\py\DRL2\.venv\Scripts\python.exe"; $Out = "C:\Users\35884\Documents\Spacecraft\过程文件\final_pretest"
Set-Location "D:\py\DRL2"; $env:OMP_NUM_THREADS = "1"; $env:MKL_NUM_THREADS = "1"; $env:OPENBLAS_NUM_THREADS = "1"
$T0 = Get-Date; $hours = { ((Get-Date) - $T0).TotalHours }
"start $T0" | Out-File -Encoding utf8 "$Out\timeline.log"
function Start-Run($cfgName, $seedNo) {
  $runDir = "$Out\B\TRAIN_${cfgName}_s$seedNo"
  Start-Process -FilePath $py -WorkingDirectory "D:\py\DRL2" -WindowStyle Hidden -PassThru `
    -ArgumentList "-B","-m","experiments.final","train","--stage","pretest","--config",$cfgName,"--seed",$seedNo,"--output",$runDir `
    -RedirectStandardOutput "$runDir.out.log" -RedirectStandardError "$runDir.err.log" }
$jobOf = @{}; $restarted = @{}
foreach ($cfgName in "EACH","SAC","SAC_ENG") { foreach ($seedNo in 10,11) { $jobOf["${cfgName}_s$seedNo"] = Start-Run $cfgName $seedNo } }
Start-Sleep -Seconds 180
$deadRuns = @($jobOf.Keys | Where-Object { $jobOf[$_].HasExited })
if ($deadRuns.Count -gt 0) {
  foreach ($job in $jobOf.Values) { if (-not $job.HasExited) { taskkill /T /F /PID $job.Id | Out-Null } }
  "launch failed: $($deadRuns -join ',')" | Out-File -Encoding utf8 -Append "$Out\timeline.log"; throw "训练启动失败：$($deadRuns -join ',')" }
"launch ok" | Out-File -Encoding utf8 -Append "$Out\timeline.log"
while (@($jobOf.Values | Where-Object { -not $_.HasExited }).Count -gt 0) {
  if ((& $hours) -ge 12) {
    foreach ($job in $jobOf.Values) { if (-not $job.HasExited) { taskkill /T /F /PID $job.Id | Out-Null } }
    "training cap 12 h reached" | Out-File -Encoding utf8 -Append "$Out\timeline.log"; break }
  foreach ($runName in @($jobOf.Keys)) {
    $runDir = "$Out\B\TRAIN_$runName"
    if ($jobOf[$runName].HasExited -and -not (Test-Path "$runDir\finished.json") -and -not $restarted[$runName] -and (& $hours) -lt 3) {
      Rename-Item $runDir "TRAIN_${runName}_crash1"; $restarted[$runName] = $true
      $parts = $runName -csplit '_s'; $jobOf[$runName] = Start-Run $parts[0] $parts[1]
      "restart $runName at $(& $hours) h" | Out-File -Encoding utf8 -Append "$Out\timeline.log" } }
  Start-Sleep -Seconds 300 }
"training end $(& $hours) h" | Out-File -Encoding utf8 -Append "$Out\timeline.log"
$evalSteps = @()
foreach ($cfgName in "EACH","SAC","SAC_ENG") { foreach ($seedNo in 10,11) {
  $evalSteps += ,@("eval", "--run", "$Out\B\TRAIN_${cfgName}_s$seedNo", "--label", "${cfgName}_s$seedNo", "--output", "$Out\C\${cfgName}_s$seedNo") } }
foreach ($seedNo in 10,11) { $evalSteps += ,@("eval", "--run", "$Out\B\TRAIN_EACH_s$seedNo", "--label", "EACH_s${seedNo}_rigid", "--output", "$Out\C\EACH_s${seedNo}_rigid", "--rigid") }
$evalSteps += ,@("geoscan")
foreach ($polName in "pure","rally","c_10_0") { $evalSteps += ,@("screen", "--policy", $polName, "--output", "$Out\S\$polName") }
foreach ($polName in "rally","c_10_0") { $evalSteps += ,@("screen", "--policy", $polName, "--output", "$Out\S\${polName}_rigid", "--rigid") }
$evalSteps += ,@("traj")
foreach ($stepArgs in $evalSteps) {
  if ((& $hours) -ge 14) { "evaluation stop before $($stepArgs -join ' ')" | Out-File -Encoding utf8 -Append "$Out\timeline.log"; break }
  if ($stepArgs[0] -eq "geoscan") {
    & $py -B -m experiments.final geoscan --openings 294200-294211 --output "$Out\G" --workers 10 *>> "$Out\geoscan.log" }
  elseif ($stepArgs[0] -eq "traj") {
    $trajJobs = @()
    foreach ($entrySpec in "EACH=$Out\B\TRAIN_EACH_s10", "SAC=$Out\B\TRAIN_SAC_s10", "pure=pure", "rally=rally") {
      $trajJobs += Start-Process -FilePath $py -WorkingDirectory "D:\py\DRL2" -WindowStyle Hidden -PassThru `
        -ArgumentList "-B","-m","experiments.final","traj","--entry",$entrySpec,"--openings","294200,294201,294202","--output","$Out\T" }
    $trajJobs | Wait-Process -Timeout 3600 }
  else {
    & $py -B -m experiments.final @stepArgs --stage pretest --workers 10 *>> "$Out\eval.log" }
  "done $($stepArgs[0]) $($stepArgs[2]) at $(& $hours) h" | Out-File -Encoding utf8 -Append "$Out\timeline.log" }
& $py -B -m experiments.final check --stage pretest --root $Out --output "$Out\CHECK.json" *> "$Out\check.log"
& $py -B -m experiments.final resolve --stage pretest --root $Out --output "$Out\REGISTRY_RESOLVED.json" *> "$Out\resolve.log"
& $py -B -m experiments.final_figures --stage pretest --root $Out --output "$Out\figures" --geoscan "$Out\G" --traj "$Out\T" *> "$Out\figures.log"
"all done $(& $hours) h" | Out-File -Encoding utf8 -Append "$Out\timeline.log"
```

## 实验有效性（只有这一类判定；不设性能门槛）

预实验**有效**，当且仅当以下全部成立：
1. **版本与配置可查：** `REGISTRY_RESOLVED.json` 生成成功，即 6 条训练同一科学 SHA、同一预算 2000、manifest 与注册表一致；`ENV_LOCK.txt` 存在。
2. **测试：** pytest 退出码 0，dryrun 自检通过。
3. **训练：** 6 条训练都有 `finished.json`，没有超时被停。
4. **评估：** `CHECK.json` 中 `complete: true`，即每个模型、每个固定策略、每个刚性配对各 48 个开局，没有 `ERROR_` 文件、没有缺漏与重复。
5. **执行器统计可核验：** 训练回合有 `exec_stats`；评估逐开局有 `entries_lost`、`window_recoveries` 与逐决策模式。
6. **图表可重建：** `experiments.final_figures` 退出码 0，生成 `fig_training`、`fig_final`、`fig_window`、`fig_geometry`、`fig_trajectory` 与 `TABLES.json`。

**发现 bug 时：** 由上层修复、重新冻结，并**重跑受影响的整个对照集合**；不得只补跑对某一方有利的部分。

**研究观察（据实记录，不作门槛，不据此改方法）：**
- EACH 对 SAC、对 SAC + planner features：曲线 AUC、2000 步模型的捕获、紧急捕获与回报、逐开局配对差；
- 窗口承诺对刚性承诺：EACH 与固定策略的入口丢失、捕获与配对回报差；
- 几何扫描中几何窗口与 G★ 可行区的占比。

## 交付

**科学分支 `evidence/final_pretest/`（附 SHA256SUMS）：**
- `B/TRAIN_*/`：`manifest.json`、`finished.json`、`episodes.json`、`model_stats.json`、`heartbeat.json`（模型 zip 不上传，SHA256 见 `REGISTRY_RESOLVED.json`）；
- `C/**`、`S/**`、`G/**`、`T/**`、`figures/**`；
- `TABLES.json`、`REGISTRY_RESOLVED.json`、`CHECK.json`、`ENV_LOCK.txt`、`pytest.log`、`timeline.log`、`DRY/CHECK.json`、`DRY/MANIFESTS.txt`。

**collab `lower/handoffs/final_pretest/RESULT.md`：**
- 有效性六项逐条写明（是 / 否，附证据文件）；
- 研究观察只摘自 `TABLES.json`；
- 墙钟与异常；
- 页首写明"预实验，2 种子 × 2000 决策，评估开局 294100–294147"。

**交付后停止。** 不启动正式实验。
