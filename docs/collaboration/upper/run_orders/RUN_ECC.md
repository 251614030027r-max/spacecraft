# 执行书：RUN_ECC（ECC 开发轮：3 组 × 2 种子 × 2000，统一信息边界、从零训练；下层本机，预计约 15 小时）

**状态：用户 11-07 批准（研究层"ECC 最终统筹意见"），要求最快推进；转发即下发。**

**可直接转发给下层的一句话：**

> 读 collab `docs/collaboration/upper/run_orders/RUN_ECC.md`，在 `D:\py\DRL2` detached 检出 `1cddbae26233b167d8c418614c467d21bf51b833`，按第 0 节自检，通过后用带 BOM 的 UTF-8 存第 1 节脚本为 `run.ps1` 并隐藏运行（不限时），跑完按"交付"提交，不改任何代码、配置与判据。

**科学提交（冻结）：** `1cddbae26233b167d8c418614c467d21bf51b833`（env 0.16.0、learner 7.2，106 项测试）。

- 算法与判据：`docs/final_method/ECC_SPEC.md`、`experiments/ecc_readout.py`（先于数据登记）。
- 预实验配置 EACH 与 `fd789126` 逐位一致：`evidence/ECC_PRECHECK/`。

## 本轮做什么

| 配置 | 决策层 | critic | 认证 | 作用 |
|---|---|---|---|---|
| `SAC_C` | 24 | 24 | 关 | 无反馈基线 |
| `EACH_C` | 24 | 48（过顶层证书） | 关 | 旧方法水平 |
| `ECC` | 24 | 58（另含动作层证书） | 训练与部署均开 | **主方法** |

- **统一信息边界：** 三组的决策层都不看规划答案和执行器内部执行状态（上一段残差、保持标志）。
- **训练：** 种子 10、11 × 2000 决策，6 条并行，从零训练。
- **评估：** 新开局 294320–294367（48 个），按训练方式各评一次；另加：
  - ECC 认证关闭（同一模型，`--no-certify`）；
  - EACH_C 部署认证（同一模型，`--certify`）。

  共 10 组 × 48。

**机时：** 预实验第 1 批 6 条并行用了 11.5 小时，ECC 的探询开销约 +10%，估计训练约 12 小时；评估 480 个回合约 3 小时。不设时间上限。

## 0. 准备与自检

```powershell
cd D:\py\DRL2; git fetch origin claude/sac-mpc-coupling-design-ns7g6i; git checkout --detach 1cddbae26233b167d8c418614c467d21bf51b833
$py  = "D:\py\DRL2\.venv\Scripts\python.exe"
$Out = "C:\Users\35884\Documents\Spacecraft\过程文件\ecc"
if ((git rev-parse HEAD) -ne "1cddbae26233b167d8c418614c467d21bf51b833") { throw "HEAD 不是冻结 SHA" }
if (git status --porcelain --untracked-files=no) { throw "科学工作树有改动" }
if ((Test-Path "$Out\B") -and @(Get-ChildItem "$Out\B" -Force).Count -gt 0) { throw "输出目录非空：$Out\B" }
New-Item -ItemType Directory -Force "$Out\B", "$Out\C", "$Out\DRY\B", "$Out\DRY\C" | Out-Null
$env:OMP_NUM_THREADS = "1"; $env:MKL_NUM_THREADS = "1"; $env:OPENBLAS_NUM_THREADS = "1"
& $py -B -m pytest -q tests/test_t2_stage.py tests/test_integrator.py *> "$Out\pytest.log"; if ($LASTEXITCODE -ne 0) { throw "pytest 退出码 $LASTEXITCODE" }
& $py --version *> "$Out\ENV_LOCK.txt"; & $py -m pip freeze *>> "$Out\ENV_LOCK.txt"
$old = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'experiments\.' }
if ($old) { throw "有遗留的 python 进程：$($old.ProcessId -join ',')" }
# 冒烟：三组各 40 决策 + dryrun 开局评估（ECC 另测认证关闭，EACH_C 另测部署认证）
$dryJobs = @()
foreach ($cfgName in "SAC_C","EACH_C","ECC") {
  $dryJobs += Start-Process -FilePath $py -WorkingDirectory "D:\py\DRL2" -WindowStyle Hidden -PassThru `
    -ArgumentList "-B","-m","experiments.final","train","--stage","ecc","--config",$cfgName,"--seed","10","--smoke-steps","40","--output","$Out\DRY\B\TRAIN_${cfgName}_s10" `
    -RedirectStandardOutput "$Out\DRY\B\TRAIN_${cfgName}_s10.out.log" -RedirectStandardError "$Out\DRY\B\TRAIN_${cfgName}_s10.err.log" }
$dryJobs | Wait-Process
foreach ($cfgName in "SAC_C","EACH_C","ECC") {
  if (-not (Test-Path "$Out\DRY\B\TRAIN_${cfgName}_s10\finished.json")) { throw "冒烟训练未完成：$cfgName" }
  & $py -B -m experiments.final eval --stage dryrun --run "$Out\DRY\B\TRAIN_${cfgName}_s10" --label "${cfgName}_s10" --openings 294300-294301 --output "$Out\DRY\C\${cfgName}_s10" --workers 2 *>> "$Out\DRY\eval.log"
  $mf = Get-Content "$Out\DRY\B\TRAIN_${cfgName}_s10\manifest.json" -Raw | ConvertFrom-Json
  "$cfgName arm=$($mf.arm) obs=$($mf.obs_profile) actor=$($mf.actor_view) critic=$($mf.critic_view) certify=$($mf.certify) env=$($mf.env_version) learner=$($mf.learner_version) sha=$($mf.git_head)" | Out-File -Encoding utf8 -Append "$Out\DRY\MANIFESTS.txt" }
& $py -B -m experiments.final eval --stage dryrun --run "$Out\DRY\B\TRAIN_ECC_s10" --label "ECC_s10_nocert" --openings 294300-294301 --output "$Out\DRY\C\ECC_s10_nocert" --no-certify --workers 2 *>> "$Out\DRY\eval.log"
& $py -B -m experiments.final eval --stage dryrun --run "$Out\DRY\B\TRAIN_EACH_C_s10" --label "EACH_C_s10_cert" --openings 294300-294301 --output "$Out\DRY\C\EACH_C_s10_cert" --certify --workers 2 *>> "$Out\DRY\eval.log"
```

**通过标准：**
- pytest 退出码 0；
- `DRY\MANIFESTS.txt` 三行：
  - SAC_C：`e_sac / cut_priv / 24 / 24`，certify 为空；
  - EACH_C：`e_priv / cut_priv / 24 / 48`，certify 为空；
  - ECC：`e_priv / cut_ecc / 24 / 58`，`certify=True`；
  - 三行都应满足：env 0.16.0、learner 7.2、sha 为冻结 SHA；
- `DRY\C\*` 五组各 2 个文件，没有 `ERROR_`；
- ECC 的评估文件中 `certify` 为 true，`ECC_s10_nocert` 为 false，`EACH_C_s10_cert` 为 true。

## 1. 一次性脚本

**规则（同预实验）：**
- 启动 3 分钟后检查进程仍在运行；
- 训练崩溃时，同配置、同种子从零重启一次；
- 心跳超过 3 小时未更新只记录，不停训；
- 不设时间上限。

```powershell
$py  = "D:\py\DRL2\.venv\Scripts\python.exe"; $Out = "C:\Users\35884\Documents\Spacecraft\过程文件\ecc"
Set-Location "D:\py\DRL2"; $env:OMP_NUM_THREADS = "1"; $env:MKL_NUM_THREADS = "1"; $env:OPENBLAS_NUM_THREADS = "1"
$T0 = Get-Date; $hours = { ((Get-Date) - $T0).TotalHours }
"start $T0" | Out-File -Encoding utf8 "$Out\timeline.log"
function Start-Run($cfgName, $seedNo) {
  $runDir = "$Out\B\TRAIN_${cfgName}_s$seedNo"
  Start-Process -FilePath $py -WorkingDirectory "D:\py\DRL2" -WindowStyle Hidden -PassThru `
    -ArgumentList "-B","-m","experiments.final","train","--stage","ecc","--config",$cfgName,"--seed",$seedNo,"--output",$runDir `
    -RedirectStandardOutput "$runDir.out.log" -RedirectStandardError "$runDir.err.log" }
$jobOf = @{}; $restarted = @{}
foreach ($cfgName in "SAC_C","EACH_C","ECC") { foreach ($seedNo in 10,11) { $jobOf["${cfgName}_s$seedNo"] = Start-Run $cfgName $seedNo } }
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
foreach ($cfgName in "SAC_C","EACH_C","ECC") { foreach ($seedNo in 10,11) {
  & $py -B -m experiments.final eval --stage ecc --run "$Out\B\TRAIN_${cfgName}_s$seedNo" --label "${cfgName}_s$seedNo" --output "$Out\C\${cfgName}_s$seedNo" --workers 10 *>> "$Out\eval.log"
  "done eval ${cfgName}_s$seedNo at $(& $hours) h" | Out-File -Encoding utf8 -Append "$Out\timeline.log" } }
foreach ($seedNo in 10,11) {
  & $py -B -m experiments.final eval --stage ecc --run "$Out\B\TRAIN_ECC_s$seedNo" --label "ECC_s${seedNo}_nocert" --output "$Out\C\ECC_s${seedNo}_nocert" --no-certify --workers 10 *>> "$Out\eval.log"
  & $py -B -m experiments.final eval --stage ecc --run "$Out\B\TRAIN_EACH_C_s$seedNo" --label "EACH_C_s${seedNo}_cert" --output "$Out\C\EACH_C_s${seedNo}_cert" --certify --workers 10 *>> "$Out\eval.log"
  "done paired evals s$seedNo at $(& $hours) h" | Out-File -Encoding utf8 -Append "$Out\timeline.log" }
& $py -B -m experiments.final check --stage ecc --root $Out --output "$Out\CHECK.json" *> "$Out\check.log"
& $py -B -m experiments.final resolve --stage ecc --root $Out --output "$Out\REGISTRY_RESOLVED.json" *> "$Out\resolve.log"
& $py -B -m experiments.ecc_readout --root $Out --output "$Out\ECC_DECISION.json" *> "$Out\decision.log"
"all done $(& $hours) h" | Out-File -Encoding utf8 -Append "$Out\timeline.log"
```

## 2. 判定顺序（`experiments/ecc_readout.py`，提交即冻结）

全部按种子配对、开局配对，共 96 对。G1–G5：
- G1：均值 ≥ 8.3，或捕获 ≥ +4；
- G2：去掉每种子最大的两个正差后 > 0；
- G3：两个种子都 > 0；
- G4：捕获不少于对照；
- G5：unsafe 为 0。

判定按顺序进行：
1. **ECC 对 SAC_C 通过 G1–G5，且 ECC 对 EACH_C 不退化**（均值差 ≥ −8.3、捕获 ≥ EACH_C − 4、unsafe 0）→ `final_method: ECC`。
2. **否则**，若 EACH_C + 部署认证 对 SAC_C 通过 G1–G5 → `final_method: ECC_deploy`（部署认证的 ECC）。
3. **否则** → `none`。

**只报告：** 同一 ECC 模型认证开 / 关、EACH_C 对 SAC_C、训练曲线、介入次数。

## 交付

**科学分支 `evidence/ECC_DEV/`（附 SHA256SUMS）：**
- `B/TRAIN_*/`：`manifest.json`、`finished.json`、`episodes.json`、`model_stats.json`、`heartbeat.json`（不上传模型 zip，SHA256 见 `REGISTRY_RESOLVED.json`）；
- `C/**`（10 组 × 48）；
- `DRY/MANIFESTS.txt`；
- `ECC_DECISION.json`、`CHECK.json`、`REGISTRY_RESOLVED.json`、`ENV_LOCK.txt`、`pytest.log`、`timeline.log`。

**collab `lower/handoffs/ecc/RESULT.md`：**
- 自检各项；
- `ECC_DECISION.json` 的 `final_method`、三组比较的门槛与数值，原样摘录；
- 墙钟与异常（含训练中 unsafe 的回合信息）。

**交付后停止。** 正式实验另行下发。
