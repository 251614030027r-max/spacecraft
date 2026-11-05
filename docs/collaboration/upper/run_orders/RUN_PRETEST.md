# 执行书：RUN_PRETEST（EACH 预实验；下层本机；不限时，预计约 23 小时；自检约 40 分钟）

**状态：已批准（用户 11-05：本轮不限时，保留关键消融，下层立即开跑；依据研究层 11-05 定稿意见 `final_review`）。** 本轮是最终方法冻结后**唯一**一次预实验，只检验实验有效性并记录研究观察，不调参、不改方法。交付后停止开发；正式实验（完整消融，≥ 5000 决策 × 3 种子）由用户另行授权，见 `RUN_FORMAL.md`（草案，未授权）。

**可直接转发给下层的一句话：**

> 读 collab `docs/collaboration/upper/run_orders/RUN_PRETEST.md`，在 `D:\py\DRL2` detached 检出 `fd789126cc822d3650e636ff2169c8757ff3bdb1`，按第 0 节自检（含全流程 dryrun），全部通过后用带 BOM 的 UTF-8 存第 1 节脚本为 `run.ps1` 并隐藏运行（不限时），跑完按"交付"一节提交证据与 RESULT.md，不改任何代码、配置与判据。

**科学提交（冻结）：** `fd789126cc822d3650e636ff2169c8757ff3bdb1`（detached 检出）。
- 环境 t2-stage-0.13.0、学习器 t2-learner-7.0；
- 主方法路径与 T2.12 `e_priv`（`2e7d838`）逐位等价（`evidence/final_pretest/precheck/`）；
- 方法与配置：`docs/final_method/`（`METHOD_MINIMAL.md`、`METHOD_SPEC.md`、`C1_THEORY.md`、`CONFIG_LOCK.md`、`ABLATION_MATRIX.md`、`FIGURE_SPEC.md`、`METHOD_REGISTRY.json`）；
- 唯一入口：`experiments/final.py`、`experiments/final_figures.py`。它们只能启动注册表里的配置、本阶段的种子与开局，历史反事实分支不可能被误启动。

## 本轮做什么：关键消融，机制可验证

| 机制 | 检验 | 本轮 |
|---|---|---|
| U1 的学习收益 | EACH 对 SAC（同一 Actor 接口、同一 C1） | 训练 |
| U1 的收益来自执行证书，而不是 critic 更宽 | EACH 对 DUP（critic 同为 48 维，后 24 维重复精简观测） | 训练 |
| 部署信息边界 | EACH 对 SAC + planner features（策略部署时也读规划答案） | 训练 |
| C1 扫掠锥几何 | EACH 对 NO_GEOM（阶段目标只沿视线接近） | 训练 |
| C1 窗口承诺 | 同一 EACH 模型、同一开局，窗口承诺对刚性承诺的配对评估；固定策略的同样配对 | 不训练（RIGID 训练留到正式实验） |
| 系统基线 | pure MPC、Direct G*、Deadline rule、Basic layering、Fixed intent (1, 0) | 不训练 |

| 部分 | 内容 | 规模 |
|---|---|---|
| 训练 | 5 组 × 种子 10、11 × 2000 决策，分两批：第 1 批 EACH、SAC、SAC_ENG（6 条并行），第 2 批 DUP、NO_GEOM（4 条并行）；训练开局从 11,000,000 与 12,000,000 起，从未用过 | 10 条 |
| 评估 | 10 个 2000 步模型；EACH 两个模型另跑刚性承诺；固定策略 pure、gstar、rally、b_10、c_10_0；其中 gstar、rally、c_10_0 另跑刚性承诺 | 评估开局 294100–294147（48 个，从未用过），共 20 组 × 48 |
| 作图数据 | 几何扫描（图 A）；轨迹（图 B：EACH、NO_GEOM、SAC、pure）；开局 294200–294211，不用于任何模型选择 | 小 |
| 图与表 | `experiments.final_figures` 从原始日志一键生成 | — |

**机时估计**（下层实测：4 条并行时 2000 步需 6.0–7.7 小时，整机约 1040 决策 / 小时）：
- 第 1 批约 11.5 小时，第 2 批约 7.7 小时；
- 评估约 960 个回合，约 3.5 小时；
- 合计约 23 小时。**不设时间上限。**

## 0. 准备与自检（任一项不满足就停下报告）

```powershell
cd D:\py\DRL2; git fetch origin claude/sac-mpc-coupling-design-ns7g6i; git checkout --detach fd789126cc822d3650e636ff2169c8757ff3bdb1
$py  = "D:\py\DRL2\.venv\Scripts\python.exe"
$Out = "C:\Users\35884\Documents\Spacecraft\过程文件\final_pretest"
if ((git rev-parse HEAD) -ne "fd789126cc822d3650e636ff2169c8757ff3bdb1") { throw "HEAD 不是冻结 SHA" }
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
  - 全部 k = 0、beta = 0，sha 为冻结 SHA。

dryrun 的数值没有任何意义，只检查流程。

## 1. 一次性脚本

存为 `$Out\run.ps1`（带 BOM 的 UTF-8），用 `powershell -ExecutionPolicy Bypass -WindowStyle Hidden -File "$Out\run.ps1"` 运行。变量名都不与 `$Out` 重名。

**规则：**
- 不设时间上限。
- **启动检查：** 每批训练启动 3 分钟后都应仍在运行，否则全部结束并报错。
- **训练崩溃（进程退出但没有 `finished.json`）：** 把该目录改名为 `*_crash1`，同配置同种子从零重启一次；第二次仍崩溃就不再重启，如实记录。
- **停滞记录：** 某条训练的 `heartbeat.json` 超过 3 小时没有更新时，在 `timeline.log` 记一行，不停训。

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
foreach ($batchCfgs in @(@("EACH","SAC","SAC_ENG"), @("DUP","NO_GEOM"))) {
  $jobOf = @{}; $restarted = @{}
  foreach ($cfgName in $batchCfgs) { foreach ($seedNo in 10,11) { $jobOf["${cfgName}_s$seedNo"] = Start-Run $cfgName $seedNo } }
  Start-Sleep -Seconds 180
  $deadRuns = @($jobOf.Keys | Where-Object { $jobOf[$_].HasExited })
  if ($deadRuns.Count -gt 0) {
    foreach ($job in $jobOf.Values) { if (-not $job.HasExited) { taskkill /T /F /PID $job.Id | Out-Null } }
    "launch failed: $($deadRuns -join ',')" | Out-File -Encoding utf8 -Append "$Out\timeline.log"; throw "训练启动失败：$($deadRuns -join ',')" }
  "batch $($batchCfgs -join ',') launch ok at $(& $hours) h" | Out-File -Encoding utf8 -Append "$Out\timeline.log"
  while (@($jobOf.Values | Where-Object { -not $_.HasExited }).Count -gt 0) {
    foreach ($runName in @($jobOf.Keys)) {
      $runDir = "$Out\B\TRAIN_$runName"
      if ($jobOf[$runName].HasExited -and -not (Test-Path "$runDir\finished.json") -and -not $restarted[$runName]) {
        Rename-Item $runDir "TRAIN_${runName}_crash1"; $restarted[$runName] = $true
        $parts = $runName -csplit '_s'; $jobOf[$runName] = Start-Run $parts[0] $parts[1]
        "restart $runName at $(& $hours) h" | Out-File -Encoding utf8 -Append "$Out\timeline.log" }
      elseif ((-not $jobOf[$runName].HasExited) -and (Test-Path "$runDir\heartbeat.json") -and (((Get-Date) - (Get-Item "$runDir\heartbeat.json").LastWriteTime).TotalHours -gt 3)) {
        "stall? $runName heartbeat older than 3 h at $(& $hours) h" | Out-File -Encoding utf8 -Append "$Out\timeline.log" } }
    Start-Sleep -Seconds 300 }
  "batch $($batchCfgs -join ',') end $(& $hours) h" | Out-File -Encoding utf8 -Append "$Out\timeline.log" }
$evalSteps = @()
foreach ($cfgName in "EACH","SAC","SAC_ENG","DUP","NO_GEOM") { foreach ($seedNo in 10,11) {
  $evalSteps += ,@("eval", "--run", "$Out\B\TRAIN_${cfgName}_s$seedNo", "--label", "${cfgName}_s$seedNo", "--output", "$Out\C\${cfgName}_s$seedNo") } }
foreach ($seedNo in 10,11) { $evalSteps += ,@("eval", "--run", "$Out\B\TRAIN_EACH_s$seedNo", "--label", "EACH_s${seedNo}_rigid", "--output", "$Out\C\EACH_s${seedNo}_rigid", "--rigid") }
$evalSteps += ,@("geoscan")
foreach ($polName in "pure","gstar","rally","b_10","c_10_0") { $evalSteps += ,@("screen", "--policy", $polName, "--output", "$Out\S\$polName") }
foreach ($polName in "gstar","rally","c_10_0") { $evalSteps += ,@("screen", "--policy", $polName, "--output", "$Out\S\${polName}_rigid", "--rigid") }
$evalSteps += ,@("traj")
foreach ($stepArgs in $evalSteps) {
  if ($stepArgs[0] -eq "geoscan") {
    & $py -B -m experiments.final geoscan --openings 294200-294211 --output "$Out\G" --workers 10 *>> "$Out\geoscan.log" }
  elseif ($stepArgs[0] -eq "traj") {
    $trajJobs = @()
    foreach ($entrySpec in "EACH=$Out\B\TRAIN_EACH_s10", "NO_GEOM=$Out\B\TRAIN_NO_GEOM_s10", "SAC=$Out\B\TRAIN_SAC_s10", "pure=pure") {
      $trajJobs += Start-Process -FilePath $py -WorkingDirectory "D:\py\DRL2" -WindowStyle Hidden -PassThru `
        -ArgumentList "-B","-m","experiments.final","traj","--entry",$entrySpec,"--openings","294200,294201,294202","--output","$Out\T" }
    $trajJobs | Wait-Process }
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
1. **版本与配置可查：** `REGISTRY_RESOLVED.json` 生成成功，即 10 条训练同一科学 SHA、同一预算 2000、manifest 与注册表一致；`ENV_LOCK.txt` 存在。
2. **测试：** pytest 退出码 0，dryrun 自检通过。
3. **训练：** 10 条训练都有 `finished.json`。
4. **评估：** `CHECK.json` 中 `complete: true`，即 20 组评估各 48 个开局，没有 `ERROR_` 文件、没有缺漏与重复。
5. **执行器统计可核验：** 训练回合有 `exec_stats`；评估逐开局有 `entries_lost`、`window_recoveries` 与逐决策模式。
6. **图表可重建：** `experiments.final_figures` 退出码 0，生成 `fig_training`、`fig_final`、`fig_window`、`fig_geometry`、`fig_trajectory` 与 `TABLES.json`。

**发现 bug 时：** 由上层修复、重新冻结，并**重跑受影响的整个对照集合**；不得只补跑对某一方有利的部分。

**研究观察（据实记录，不作门槛，不据此改方法）：**
- **U1：** EACH 对 SAC、对 DUP（证书还是宽度）、对 SAC + planner features（部署信息边界）——曲线 AUC、2000 步模型的捕获、紧急捕获与回报、逐开局配对差；
- **C1 几何：** EACH 对 NO_GEOM；
- **C1 窗口：** EACH 与固定策略的窗口 / 刚性配对，看入口丢失、捕获与配对回报差；
- **几何扫描：** 几何窗口与 G★ 可行区的占比。

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
