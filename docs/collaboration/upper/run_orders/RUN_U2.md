# 执行书：RUN_U2（执行器模式探询 U2 的一轮训练验证；下层本机；预计约 10.5 小时，自检约 40 分钟）

**状态：用户 11-06 同意由下层执行以节省时间，转发即下发。**

训练与上层沙箱中的阶段 A 离线预检**并行**：
- 阶段 A 由上层完成，判据在数据产生前已登记于 `docs/final_method/U2_PROBE_SPEC.md`（提交 `0c4b97ec`）；中途 29 个回合 A1–A5 全过。
- 脚本在训练结束、评估之前自动读取科学分支上的 `evidence/U2_PROBE_PRECHECK/SUMMARY.json`：
  - `stop: false` 且 `defects: false` 才继续评估；
  - 否则停止，本轮训练作废，不报告任何训练结果。
- 上层若提前通知 A 未通过，下层立即结束全部训练。

依据：研究层 11-06《EACH-U2 上层最终设计建议》与用户批准（U2 是唯一新增方向，只做一次完整验证）。

本轮只训练两组新配置；SAC 与 EACH 复用预实验模型。不调参，不改方法。交付后停止；正式实验另行授权。

**可直接转发给下层的一句话：**

> 读 collab `docs/collaboration/upper/run_orders/RUN_U2.md`，在 `D:\py\DRL2` detached 检出 `e7acaa340cbff41a2e618c200cd7bdeb0438118f`，按第 0 节自检（含复用核验与冒烟），全部通过后用带 BOM 的 UTF-8 存第 1 节脚本为 `run.ps1` 并隐藏运行（不限时），跑完按"交付"一节提交，不改任何代码、配置与判据。

**科学提交（冻结）：** `e7acaa340cbff41a2e618c200cd7bdeb0438118f`（detached 检出）。
- 环境 t2-stage-0.14.0、学习器 t2-learner-7.1；
- U2 规格 `docs/final_method/U2_PROBE_SPEC.md`；注册表 `METHOD_REGISTRY.json` 中的配置 `PROBE`、`EACH_PROBE` 与阶段 `u2`；
- EACH 与 SAC 的训练路径与 `fd789126` 逐位一致（`evidence/U2_PROBE_PRECHECK/equivalence/`）。

## 本轮做什么

| 组 | 来源 | critic 输入 | 回答的问题 |
|---|---|---|---|
| SAC | 复用预实验种子 10、11 模型 | 24（无反馈） | 参照 |
| EACH（U1 + C1） | 复用预实验种子 10、11 模型 | 48（U1） | 参照 |
| **PROBE**（U2 + C1） | 新训练，种子 10、11 × 2000 | 24 + 27（U2 替代 U1） | **主判据：系统生成的执行器反馈能否独立优于无反馈 SAC** |
| **EACH_PROBE**（U1 + U2 + C1） | 新训练，种子 10、11 × 2000 | 48 + 27 | 次要：U1 与 U2 是否互补、性能代价 |

**统一设置：**
- 物理、Actor（24 维）、执行器、奖励、优化预算（每条转移 2 次更新）、熵系数、异步 2 worker、紧急截止 U(140, 200) s 与预实验相同；
- 训练开局流与预实验同种子相同（11,000,000、12,000,000 起），所以按种子与 SAC / EACH 配对。

**评估：**
- 8 个 2000 步模型在**新开局 294148–294195**（48 个，从未用过）上评估：PROBE、EACH_PROBE 各 2 个，复用的 EACH、SAC 各 2 个；
- 评估时探询关闭，Actor 只读 24 维。

**机时：**
- 训练：4 条并行。预实验第 2 批（4 条）用 7.7 小时，离线中途实测探询开销约为执行的 7%，估计约 8.5 小时；
- 评估：384 个回合，约 1.5 小时。
- **不设时间上限。**

## 0. 准备与自检（任一项不满足就停下报告）

```powershell
cd D:\py\DRL2; git fetch origin claude/sac-mpc-coupling-design-ns7g6i; git checkout --detach e7acaa340cbff41a2e618c200cd7bdeb0438118f
$py  = "D:\py\DRL2\.venv\Scripts\python.exe"
$Out = "C:\Users\35884\Documents\Spacecraft\过程文件\u2"
$Pre = "C:\Users\35884\Documents\Spacecraft\过程文件\final_pretest"
if ((git rev-parse HEAD) -ne "e7acaa340cbff41a2e618c200cd7bdeb0438118f") { throw "HEAD 不是冻结 SHA" }
if (git status --porcelain --untracked-files=no) { throw "科学工作树有改动" }
if ((Test-Path "$Out\B") -and @(Get-ChildItem "$Out\B" -Force).Count -gt 0) { throw "输出目录非空：$Out\B" }
New-Item -ItemType Directory -Force "$Out\B", "$Out\C", "$Out\REUSE", "$Out\DRY\B", "$Out\DRY\C" | Out-Null
$env:OMP_NUM_THREADS = "1"; $env:MKL_NUM_THREADS = "1"; $env:OPENBLAS_NUM_THREADS = "1"
& $py -B -m pytest -q tests/test_t2_stage.py tests/test_integrator.py *> "$Out\pytest.log"; $pt = $LASTEXITCODE   # 以退出码 0 判定
if ($pt -ne 0) { throw "pytest 退出码 $pt" }
& $py --version *> "$Out\ENV_LOCK.txt"; & $py -m pip freeze *>> "$Out\ENV_LOCK.txt"
$old = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'experiments\.' }
if ($old) { throw "有遗留的 python 进程：$($old.ProcessId -join ',')" }
# ---- 复用核验 1：预实验模型文件与预实验登记的 SHA256 一致 ----
$want = @{ "EACH_s10" = "cb9a8968868ae67ead341a9a99ea59bbff0bf81cad7917f7f5bbbfd4578dd388"; "EACH_s11" = "6895eb31bf0b46638236f0ba49860f75127560b2ec793b60edd55200906457d8";
           "SAC_s10"  = "3e2bd45e197889710eefeb35381beeb6b584f42a501257d6d8b7f53c202aa376"; "SAC_s11"  = "1e78cccd3df34a51581cc359045b85a246e30205cdb4218fcc381778ed429dad" }
foreach ($runName in $want.Keys) {
  $got = (Get-FileHash "$Pre\B\TRAIN_$runName\checkpoints\step_2000.zip" -Algorithm SHA256).Hash.ToLower()
  if ($got -ne $want[$runName]) { throw "复用模型 $runName 的 SHA256 不符：$got" }
  "$runName $got" | Out-File -Encoding utf8 -Append "$Out\REUSE\MODEL_SHA256.txt" }
# ---- 复用核验 2：新代码重评预实验模型的前 2 个预实验开局，与预实验记录逐项相同 ----
foreach ($runName in "EACH_s10","SAC_s10") {
  & $py -B -m experiments.final eval --stage pretest --run "$Pre\B\TRAIN_$runName" --label $runName --openings 294100-294101 --output "$Out\REUSE\$runName" --workers 2 *>> "$Out\REUSE\eval.log"
  & $py -B -m experiments.u2_readout reuse --new "$Out\REUSE\$runName" --old "$Pre\C\$runName" --output "$Out\REUSE\$runName.json" *>> "$Out\REUSE\reuse.log"
  if ($LASTEXITCODE -ne 0) { throw "复用核验失败：$runName（见 $Out\REUSE\$runName.json）" } }
# ---- 冒烟：两组新配置各 40 决策（smoke 学习器设置）+ dryrun 开局评估 ----
$dryJobs = @()
foreach ($cfgName in "PROBE","EACH_PROBE") {
  $dryJobs += Start-Process -FilePath $py -WorkingDirectory "D:\py\DRL2" -WindowStyle Hidden -PassThru `
    -ArgumentList "-B","-m","experiments.final","train","--stage","u2","--config",$cfgName,"--seed","10","--smoke-steps","40","--output","$Out\DRY\B\TRAIN_${cfgName}_s10" `
    -RedirectStandardOutput "$Out\DRY\B\TRAIN_${cfgName}_s10.out.log" -RedirectStandardError "$Out\DRY\B\TRAIN_${cfgName}_s10.err.log" }
$dryJobs | Wait-Process
foreach ($cfgName in "PROBE","EACH_PROBE") {
  if (-not (Test-Path "$Out\DRY\B\TRAIN_${cfgName}_s10\finished.json")) { throw "冒烟训练未完成：$cfgName" }
  & $py -B -m experiments.final eval --stage dryrun --run "$Out\DRY\B\TRAIN_${cfgName}_s10" --label "${cfgName}_s10" --openings 294300-294301 --output "$Out\DRY\C\${cfgName}_s10" --workers 2 *>> "$Out\DRY\eval.log"
  $mf = Get-Content "$Out\DRY\B\TRAIN_${cfgName}_s10\manifest.json" -Raw | ConvertFrom-Json
  $ep = Get-Content "$Out\DRY\B\TRAIN_${cfgName}_s10\episodes.json" -Raw | ConvertFrom-Json
  "$cfgName arm=$($mf.arm) obs=$($mf.obs_profile) actor=$($mf.actor_view) critic=$($mf.critic_view) env=$($mf.env_version) learner=$($mf.learner_version) k=$($mf.k_candidates) beta=$($mf.model_beta) sha=$($mf.git_head) probes=$($ep[0].exec_stats.probes)" | Out-File -Encoding utf8 -Append "$Out\DRY\MANIFESTS.txt" }
```

**自检通过的标准：**
- pytest 退出码 0；
- `REUSE\MODEL_SHA256.txt` 四行与上面一致；`REUSE\EACH_s10.json`、`REUSE\SAC_s10.json` 中 `identical: true`；
- `DRY\MANIFESTS.txt` 两行：
  - PROBE：`e_priv / reduced_probe / 24 / 51`；
  - EACH_PROBE：`e_priv / reduced_priv_probe / 24 / 75`；
  - 两行都应满足：env 0.14.0、learner 7.1、k = 0、beta = 0、sha 为冻结 SHA、probes > 0；
- `DRY\C\*` 各 2 个评估文件，没有 `ERROR_`。

## 1. 一次性脚本

存为 `$Out\run.ps1`（带 BOM 的 UTF-8），用 `powershell -ExecutionPolicy Bypass -WindowStyle Hidden -File "$Out\run.ps1"` 运行。变量名都不与 `$Out`、`$Pre` 重名。

**规则（同预实验）：**
- 不设时间上限；
- 启动 3 分钟后检查进程仍在运行；
- 训练崩溃时，同配置、同种子从零重启一次；第二次仍崩溃就如实记录；
- `heartbeat.json` 超过 3 小时未更新时，只记录，不停训。

```powershell
$py  = "D:\py\DRL2\.venv\Scripts\python.exe"; $Out = "C:\Users\35884\Documents\Spacecraft\过程文件\u2"; $Pre = "C:\Users\35884\Documents\Spacecraft\过程文件\final_pretest"
Set-Location "D:\py\DRL2"; $env:OMP_NUM_THREADS = "1"; $env:MKL_NUM_THREADS = "1"; $env:OPENBLAS_NUM_THREADS = "1"
$T0 = Get-Date; $hours = { ((Get-Date) - $T0).TotalHours }
"start $T0" | Out-File -Encoding utf8 "$Out\timeline.log"
function Start-Run($cfgName, $seedNo) {
  $runDir = "$Out\B\TRAIN_${cfgName}_s$seedNo"
  Start-Process -FilePath $py -WorkingDirectory "D:\py\DRL2" -WindowStyle Hidden -PassThru `
    -ArgumentList "-B","-m","experiments.final","train","--stage","u2","--config",$cfgName,"--seed",$seedNo,"--output",$runDir `
    -RedirectStandardOutput "$runDir.out.log" -RedirectStandardError "$runDir.err.log" }
$jobOf = @{}; $restarted = @{}
foreach ($cfgName in "PROBE","EACH_PROBE") { foreach ($seedNo in 10,11) { $jobOf["${cfgName}_s$seedNo"] = Start-Run $cfgName $seedNo } }
Start-Sleep -Seconds 180
$deadRuns = @($jobOf.Keys | Where-Object { $jobOf[$_].HasExited })
if ($deadRuns.Count -gt 0) {
  foreach ($job in $jobOf.Values) { if (-not $job.HasExited) { taskkill /T /F /PID $job.Id | Out-Null } }
  "launch failed: $($deadRuns -join ',')" | Out-File -Encoding utf8 -Append "$Out\timeline.log"; throw "训练启动失败：$($deadRuns -join ',')" }
"launch ok at $(& $hours) h" | Out-File -Encoding utf8 -Append "$Out\timeline.log"
while (@($jobOf.Values | Where-Object { -not $_.HasExited }).Count -gt 0) {
  foreach ($runName in @($jobOf.Keys)) {
    $runDir = "$Out\B\TRAIN_$runName"
    if ($jobOf[$runName].HasExited -and -not (Test-Path "$runDir\finished.json") -and -not $restarted[$runName]) {
      Rename-Item $runDir "TRAIN_${runName}_crash1"; $restarted[$runName] = $true
      $cut = $runName.LastIndexOf('_s'); $jobOf[$runName] = Start-Run $runName.Substring(0, $cut) $runName.Substring($cut + 2)
      "restart $runName at $(& $hours) h" | Out-File -Encoding utf8 -Append "$Out\timeline.log" }
    elseif ((-not $jobOf[$runName].HasExited) -and (Test-Path "$runDir\heartbeat.json") -and (((Get-Date) - (Get-Item "$runDir\heartbeat.json").LastWriteTime).TotalHours -gt 3)) {
      "stall? $runName heartbeat older than 3 h at $(& $hours) h" | Out-File -Encoding utf8 -Append "$Out\timeline.log" } }
  Start-Sleep -Seconds 300 }
"training end $(& $hours) h" | Out-File -Encoding utf8 -Append "$Out\timeline.log"
# ---- 阶段 A 关卡：只 fetch，不动工作树；判定文件未发布时每 10 分钟再查一次 ----
while ($true) {
  git fetch -q origin claude/sac-mpc-coupling-design-ns7g6i
  $gateText = git show "origin/claude/sac-mpc-coupling-design-ns7g6i:evidence/U2_PROBE_PRECHECK/SUMMARY.json" 2>$null
  if ($LASTEXITCODE -eq 0 -and $gateText) { break }
  "stage A verdict not published yet at $(& $hours) h" | Out-File -Encoding utf8 -Append "$Out\timeline.log"; Start-Sleep -Seconds 600 }
$gateText | Out-File -Encoding utf8 "$Out\STAGE_A_SUMMARY.json"
$gate = $gateText | Out-String | ConvertFrom-Json
if ($gate.stop -or $gate.defects) { "stage A failed: stop=$($gate.stop) defects=$($gate.defects); training void" | Out-File -Encoding utf8 -Append "$Out\timeline.log"; throw "阶段 A 未通过，本轮作废" }
"stage A passed at $(& $hours) h" | Out-File -Encoding utf8 -Append "$Out\timeline.log"
foreach ($cfgName in "PROBE","EACH_PROBE") { foreach ($seedNo in 10,11) {
  & $py -B -m experiments.final eval --stage u2 --run "$Out\B\TRAIN_${cfgName}_s$seedNo" --label "${cfgName}_s$seedNo" --output "$Out\C\${cfgName}_s$seedNo" --workers 10 *>> "$Out\eval.log"
  "done eval ${cfgName}_s$seedNo at $(& $hours) h" | Out-File -Encoding utf8 -Append "$Out\timeline.log" } }
foreach ($cfgName in "EACH","SAC") { foreach ($seedNo in 10,11) {
  & $py -B -m experiments.final eval --stage u2 --run "$Pre\B\TRAIN_${cfgName}_s$seedNo" --label "${cfgName}_s$seedNo" --output "$Out\C\${cfgName}_s$seedNo" --workers 10 *>> "$Out\eval.log"
  "done eval reused ${cfgName}_s$seedNo at $(& $hours) h" | Out-File -Encoding utf8 -Append "$Out\timeline.log" } }
& $py -B -m experiments.final check --stage u2 --root $Out --output "$Out\CHECK.json" *> "$Out\check.log"
& $py -B -m experiments.final resolve --stage u2 --root $Out --output "$Out\REGISTRY_RESOLVED.json" *> "$Out\resolve.log"
& $py -B -m experiments.u2_readout decide --root $Out --pretest-root $Pre --output "$Out\U2_DECISION.json" *> "$Out\decision.log"
"all done $(& $hours) h" | Out-File -Encoding utf8 -Append "$Out\timeline.log"
```

## 2. 判据（事先冻结；`experiments/u2_readout.py`，提交即冻结）

比较全部按**种子配对、开局配对**：PROBE_s10 对 SAC_s10、PROBE_s11 对 SAC_s11，共 96 对。真实回报为无塑形的任务回报。

**主判据：PROBE 对 SAC，五项全部成立才算 `pass`，否则 `no_gain`：**

| 项 | 内容 |
|---|---|
| G1 明确改善 | 配对真实回报差均值 ≥ 8.3，**或**捕获数 ≥ SAC + 4（96 个中） |
| G2 不是少数翻转 | 每个种子去掉最大的 2 个正差（共去掉 4 / 96）后，均值仍 > 0 |
| G3 两个种子 | 每个种子的配对差均值都 > 0 |
| G4 捕获不减少 | 捕获数 ≥ SAC |
| G5 安全 | PROBE unsafe = 0 |

参照：把预实验的 EACH 代入 PROBE 的位置，EACH 对 SAC 在预实验开局上满足全部五项（均值 +18.2，去掉每个种子最大 2 个正差后 +10.1，两种子 +24.7 / +11.7，捕获 90 / 80，unsafe 0）。这说明门槛是 U1 达到过的水平，不是放宽的门槛。

**次要：EACH_PROBE 对 EACH（不决定 U2 成败）：**
- `complementary`：上面五项都满足；
- `kept`：均值差 ≥ −8.3、捕获 ≥ EACH − 4、unsafe 0；
- 否则 `degraded`。

**只报告：**
- PROBE 对 EACH（与 U1 的差距）；
- 训练曲线 AUC 与后 500 步回报；
- 紧急捕获、中位差、入口丢失；
- 双方都捕获的开局上的时间与 Δv；
- 探询耗时。

**结论处理（研究层 11-06 已定）：**
- PROBE `pass`：系统生成的执行模式反馈有独立学习价值；即使弱于 U1 也成立为机制，但必须如实报告差距。
- EACH_PROBE：
  - `kept` / `complementary` → 可作为集成版本；
  - `degraded` → 保留原 EACH，U2 只作独立方法证据，不强行叠加。
- PROBE `no_gain`：直接停止 U2。不做第二轮网格、时域、校准或网络调整。

**发现 bug 时：** 由上层修复、重新冻结，重跑受影响的整组对照，不只补跑有利的一方。

## 交付

**科学分支 `evidence/U2_TRAIN/`（附 SHA256SUMS）：**
- `B/TRAIN_*/`：`manifest.json`、`finished.json`、`episodes.json`、`model_stats.json`、`heartbeat.json`（模型 zip 不上传，SHA256 见 `REGISTRY_RESOLVED.json`）；
- `C/**`（8 组 × 48）；
- `REUSE/**`、`STAGE_A_SUMMARY.json`；
- `DRY/MANIFESTS.txt`；
- `U2_DECISION.json`、`CHECK.json`、`REGISTRY_RESOLVED.json`、`ENV_LOCK.txt`、`pytest.log`、`timeline.log`。

**collab `lower/handoffs/u2/RESULT.md`：**
- 页首写明"U2 训练验证，2 种子 × 2000 决策，评估开局 294148–294195；SAC 与 EACH 为预实验模型"；
- 自检各项（是 / 否，附文件）；
- `U2_DECISION.json` 的主判据五项与结论、次要结论，原样摘录；
- 墙钟与异常。

**交付后停止。** 不启动正式实验。
