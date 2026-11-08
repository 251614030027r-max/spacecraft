# 执行书：RUN_GOV_B（ECC 最终轮 B 步：3 组 × 2 种子 × 3000 决策，从零训练；接在 A 步之后自动开始；下层本机，预计约 30 小时）

**状态：用户 2026-11-08 批准（B 步 3000 决策，机时已确认；A 步过门后由下层自动开训练）。**

**可直接转发给下层的一句话：**

> 读 collab `docs/collaboration/upper/run_orders/RUN_GOV_B.md`。A 步（RUN_GOV_A）的 `run.ps1` 跑完、`gov_a\A_DECISION.json` 写出后，用带 BOM 的 UTF-8 存第 1 节脚本为 `run_b.ps1` 并隐藏运行（不限时；它会等 A 步结束、自检、按 A 步的 `b_coordination` 自动选在环配置并开训练），跑完按"交付"提交，不改任何代码、配置与判据。

**科学提交（冻结）：** `e0a617532008dac02e2d50f7e859aca613e03e5e`（env 0.17.0、learner 7.3，114 项测试）。

- 与 A 步的 `bc2d0250` 相比只多一项：注册表允许 B 阶段训练回退配置 `ECC_CERT_LOOP`（A 步弃用调节器时用），机制、条件、判据不变。
- 规格、判据、选型顺序：`docs/final_method/ECC_FINAL_SPEC.md` 第 7 节；读出 `experiments/gov_readout.py b`。

## 本步做什么

**自动门（按事先登记的规则，下层不做判断）：** 读 `gov_a\A_DECISION.json` 的 `b_coordination`：

| `b_coordination` | 在环配置 | 部署协调（EACH_C 上） | 在环模型的监测评估 |
|---|---|---|---|
| `gov`（A 步 M2、M3 通过） | `ECC_GOV`（EACH_C + 新调节器在环） | `--govern act` | `--govern monitor` |
| `cert`（M2 或 M3 不过） | `ECC_CERT_LOOP`（EACH_C + 旧认证在环） | `--certify` | `--no-certify --govern monitor` |

`A_DECISION.json` 不存在或 `complete` 为 false 时脚本停下，不开训练。

**训练：** `SAC_C`、`EACH_C`、在环配置 × 种子 10、11 × 3000 决策，6 条并行，从零训练；训练任务紧急 3/4、经济 1/4（`urgent_mix:0.75`，注册表 `gov_b`）。检查点 1000、2000 只用于学习曲线。

**评估（只用 3000 步最终模型，不挑检查点）：** 三个条件，开局与筛查、A 步互不相交：

| 条件 | 开局 | 个数 |
|---|---|---|
| 紧急（主终点） | 294800–294895 | 96 |
| 经济（不伤害） | 294900–294947 | 48 |
| 自然 1:1（外部有效性） | 295000–295047 | 48 |

| 标签（后缀 `_economy` / `_natural` 表示条件） | 内容 |
|---|---|
| `SAC_C_s<种子>` | 无反馈基线 |
| `EACH_C_s<种子>_monitor` | EACH_C 不协调（监测，与关闭逐位相同） |
| `EACH_C_s<种子>_gov` 或 `_cert` | EACH_C + 部署协调（第 2 档） |
| `EACH_C_s<种子>_cert` | EACH_C + 旧认证（第 3 档 `ECC_deploy`，只报告；协调为 cert 时即第 2 档） |
| `<在环配置>_s<种子>` | 在环方法按训练方式（第 1 档） |
| `<在环配置>_s<种子>_monitor` | 在环模型的原始行为（只报告） |

**完整性检查：** 协调为 `cert` 时 `CHECK.json` 会把 `EACH_C_*_gov*` 列为缺失，属预期（这些评估不跑）；`ECC_GOV` 的名字由检查自动换成 `ECC_CERT_LOOP`。

**机时：** 训练 18,000 决策，按下层实测整机约 1,000 决策 / 小时、调节器开销约 +7%，约 19–20 小时；评估约 (1 + 3 + 2) × 2 × 192 = 2,304 回合（协调为 cert 时 EACH_C 部署与旧认证相同，少 384 回合），约 9 小时。不设时间上限。

## 0. 自检（脚本内自动完成）

- 等 A 步：`gov_a\timeline.log` 出现 `all done` 且 `gov_a\A_DECISION.json` 存在；没有遗留的 `experiments.` python 进程；
- 检出冻结 SHA，工作树干净，pytest 退出码 0；
- 冒烟：三组各 40 决策（`--smoke-steps 40`），在 2 个开局上按本步方式各评一次；manifest 视图核对：
  - SAC_C：`e_sac / cut_priv / 24 / 24`；
  - EACH_C：`e_priv / cut_priv / 24 / 48`；
  - ECC_GOV：`e_priv / cut_priv / 24 / 48`、`govern=act`；或 ECC_CERT_LOOP：同上、`certify=True`；
  - 都是 `mission=urgent_mix:0.75`、env 0.17.0、learner 7.3、sha 为冻结 SHA。

任何一项不过，脚本停下并写 `B_SELFCHECK_FAILED.txt`，不开训练。

## 1. 一次性脚本 `run_b.ps1`

**规则（同前几轮）：** 启动 3 分钟后检查 6 条训练都在运行；训练崩溃时同配置、同种子从零重启一次；心跳超过 3 小时未更新只记录，不停训；不设时间上限。外部中断（重启、更新）后：已完成的训练不重跑；评估只补缺失的开局（结果文件存在的开局自动跳过），并核对已有文件的哈希，在交付中写明。

```powershell
$py   = "D:\py\DRL2\.venv\Scripts\python.exe"
$OutA = "C:\Users\35884\Documents\Spacecraft\过程文件\gov_a"
$Out  = "C:\Users\35884\Documents\Spacecraft\过程文件\gov_b"
$SHA  = "e0a617532008dac02e2d50f7e859aca613e03e5e"
$env:OMP_NUM_THREADS = "1"; $env:MKL_NUM_THREADS = "1"; $env:OPENBLAS_NUM_THREADS = "1"
New-Item -ItemType Directory -Force "$Out\B", "$Out\C", "$Out\DRY\B", "$Out\DRY\C" | Out-Null
function Fail($msg) { $msg | Out-File -Encoding utf8 "$Out\B_SELFCHECK_FAILED.txt"; throw $msg }
# ---- 等 A 步结束
while (-not ((Test-Path "$OutA\A_DECISION.json") -and (Test-Path "$OutA\timeline.log") -and (Select-String -Path "$OutA\timeline.log" -Pattern "all done" -Quiet))) { Start-Sleep -Seconds 300 }
$old = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'experiments\.' }
if ($old) { Fail "有遗留的 python 进程：$($old.ProcessId -join ',')" }
$A = Get-Content "$OutA\A_DECISION.json" -Raw | ConvertFrom-Json
if (-not $A.complete) { Fail "A_DECISION.json complete=false" }
$coord = $A.b_coordination
if ($coord -eq "gov") { $Loop = "ECC_GOV"; $DepFlag = @("--govern","act"); $DepTag = "gov"; $MonFlag = @("--govern","monitor") }
elseif ($coord -eq "cert") { $Loop = "ECC_CERT_LOOP"; $DepFlag = @("--certify"); $DepTag = "cert"; $MonFlag = @("--no-certify","--govern","monitor") }
else { Fail "b_coordination 未知：$coord" }
"A done; b_coordination=$coord; loop=$Loop" | Out-File -Encoding utf8 "$Out\timeline.log"
# ---- 检出、测试
Set-Location "D:\py\DRL2"; git fetch origin claude/sac-mpc-coupling-design-ns7g6i; git checkout --detach $SHA
if ((git rev-parse HEAD) -ne $SHA) { Fail "HEAD 不是冻结 SHA" }
if (git status --porcelain --untracked-files=no) { Fail "科学工作树有改动" }
& $py -B -m pytest -q tests/test_t2_stage.py tests/test_integrator.py *> "$Out\pytest.log"; if ($LASTEXITCODE -ne 0) { Fail "pytest 退出码 $LASTEXITCODE" }
& $py --version *> "$Out\ENV_LOCK.txt"; & $py -m pip freeze *>> "$Out\ENV_LOCK.txt"
# ---- 冒烟
$cfgs = @("SAC_C","EACH_C",$Loop)
$dry = @(); foreach ($c in $cfgs) { $dry += Start-Process -FilePath $py -WorkingDirectory "D:\py\DRL2" -WindowStyle Hidden -PassThru `
  -ArgumentList "-B","-m","experiments.final","train","--stage","gov_b","--config",$c,"--seed","10","--smoke-steps","40","--output","$Out\DRY\B\TRAIN_${c}_s10" `
  -RedirectStandardOutput "$Out\DRY\B\TRAIN_${c}_s10.out.log" -RedirectStandardError "$Out\DRY\B\TRAIN_${c}_s10.err.log" }
$dry | Wait-Process
foreach ($c in $cfgs) {
  if (-not (Test-Path "$Out\DRY\B\TRAIN_${c}_s10\finished.json")) { Fail "冒烟训练未完成：$c" }
  $mf = Get-Content "$Out\DRY\B\TRAIN_${c}_s10\manifest.json" -Raw | ConvertFrom-Json
  "$c arm=$($mf.arm) obs=$($mf.obs_profile) actor=$($mf.actor_view) critic=$($mf.critic_view) govern=$($mf.govern) certify=$($mf.certify) mission=$($mf.mission) env=$($mf.env_version) learner=$($mf.learner_version) sha=$($mf.git_head)" | Out-File -Encoding utf8 -Append "$Out\DRY\MANIFESTS.txt"
  if ($mf.git_head -ne $SHA -or $mf.mission -ne "urgent_mix:0.75") { Fail "冒烟 manifest 不符：$c" } }
& $py -B -m experiments.final eval --stage gov_b --run "$Out\DRY\B\TRAIN_${Loop}_s10" --label "${Loop}_s10" --openings 294800-294801 --output "$Out\DRY\C\${Loop}_s10" --workers 2 *>> "$Out\DRY\eval.log"
& $py -B -m experiments.final eval --stage gov_b --run "$Out\DRY\B\TRAIN_${Loop}_s10" --label "${Loop}_s10_monitor" --openings 294800-294801 --output "$Out\DRY\C\${Loop}_s10_monitor" @MonFlag --workers 2 *>> "$Out\DRY\eval.log"
& $py -B -m experiments.final eval --stage gov_b --run "$Out\DRY\B\TRAIN_EACH_C_s10" --label "EACH_C_s10_$DepTag" --openings 294800-294801 --output "$Out\DRY\C\EACH_C_s10_$DepTag" @DepFlag --workers 2 *>> "$Out\DRY\eval.log"
if (@(Get-ChildItem "$Out\DRY\C" -Recurse -Filter "ERROR_*").Count -gt 0) { Fail "冒烟评估出错" }
"selfcheck ok" | Out-File -Encoding utf8 -Append "$Out\timeline.log"
# ---- 训练
$T0 = Get-Date; $hours = { ((Get-Date) - $T0).TotalHours }
function Start-Run($cfgName, $seedNo) {
  $runDir = "$Out\B\TRAIN_${cfgName}_s$seedNo"
  Start-Process -FilePath $py -WorkingDirectory "D:\py\DRL2" -WindowStyle Hidden -PassThru `
    -ArgumentList "-B","-m","experiments.final","train","--stage","gov_b","--config",$cfgName,"--seed",$seedNo,"--output",$runDir `
    -RedirectStandardOutput "$runDir.out.log" -RedirectStandardError "$runDir.err.log" }
$jobOf = @{}; $restarted = @{}
foreach ($c in $cfgs) { foreach ($s in 10,11) { if (-not (Test-Path "$Out\B\TRAIN_${c}_s$s\finished.json")) { $jobOf["${c}_s$s"] = Start-Run $c $s } } }
Start-Sleep -Seconds 180
$deadRuns = @($jobOf.Keys | Where-Object { $jobOf[$_].HasExited -and -not (Test-Path "$Out\B\TRAIN_$_\finished.json") })
if ($deadRuns.Count -gt 0) { foreach ($j in $jobOf.Values) { if (-not $j.HasExited) { taskkill /T /F /PID $j.Id | Out-Null } }; Fail "训练启动失败：$($deadRuns -join ',')" }
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
# ---- 评估（3000 步最终模型）
$sfx = @{ "urgent" = ""; "economy" = "_economy"; "natural" = "_natural" }
foreach ($cond in "urgent","economy","natural") { $x = $sfx[$cond]
  foreach ($s in 10,11) {
    & $py -B -m experiments.final eval --stage gov_b --condition $cond --run "$Out\B\TRAIN_SAC_C_s$s" --label "SAC_C_s$s$x" --output "$Out\C\SAC_C_s$s$x" --workers 10 *>> "$Out\eval.log"
    & $py -B -m experiments.final eval --stage gov_b --condition $cond --run "$Out\B\TRAIN_EACH_C_s$s" --label "EACH_C_s${s}_monitor$x" --output "$Out\C\EACH_C_s${s}_monitor$x" --govern monitor --workers 10 *>> "$Out\eval.log"
    & $py -B -m experiments.final eval --stage gov_b --condition $cond --run "$Out\B\TRAIN_EACH_C_s$s" --label "EACH_C_s${s}_$DepTag$x" --output "$Out\C\EACH_C_s${s}_$DepTag$x" @DepFlag --workers 10 *>> "$Out\eval.log"
    if ($DepTag -ne "cert") {
      & $py -B -m experiments.final eval --stage gov_b --condition $cond --run "$Out\B\TRAIN_EACH_C_s$s" --label "EACH_C_s${s}_cert$x" --output "$Out\C\EACH_C_s${s}_cert$x" --certify --workers 10 *>> "$Out\eval.log" }
    & $py -B -m experiments.final eval --stage gov_b --condition $cond --run "$Out\B\TRAIN_${Loop}_s$s" --label "${Loop}_s$s$x" --output "$Out\C\${Loop}_s$s$x" --workers 10 *>> "$Out\eval.log"
    & $py -B -m experiments.final eval --stage gov_b --condition $cond --run "$Out\B\TRAIN_${Loop}_s$s" --label "${Loop}_s${s}_monitor$x" --output "$Out\C\${Loop}_s${s}_monitor$x" @MonFlag --workers 10 *>> "$Out\eval.log"
    "done eval $cond s$s at $(& $hours) h" | Out-File -Encoding utf8 -Append "$Out\timeline.log" } }
& $py -B -m experiments.final check --stage gov_b --root $Out --output "$Out\CHECK.json" *> "$Out\check.log"
& $py -B -m experiments.final resolve --stage gov_b --root $Out --output "$Out\REGISTRY_RESOLVED.json" *> "$Out\resolve.log"
& $py -B -m experiments.gov_readout b --root $Out --a-decision "$OutA\A_DECISION.json" --output "$Out\B_DECISION.json" *> "$Out\decision.log"
"all done $(& $hours) h" | Out-File -Encoding utf8 -Append "$Out\timeline.log"
```

## 2. 选型（`experiments/gov_readout.py b`，入库后不改；规格第 7 节）

G1–G5（`experiments/u2_readout.gates`）在紧急任务按种子、按开局配对的 192 对上：
- G1：均值 ≥ 8.3，或捕获 ≥ +4；
- G2：去掉每种子最大的两个正差后 > 0；
- G3：两个种子都 > 0；
- G4：捕获不少于对照；
- G5：unsafe 为 0。

判定按顺序进行：
1. **在环方法**对 SAC_C：紧急任务过 G1–G5，且自然分布（1:1）对 SAC_C 不劣（均值差 ≥ −8.3、unsafe 0）→ `final_method: in_loop`；
2. **否则 EACH_C + 部署协调**对 SAC_C 同样两条 → `final_method: deploy`（论文不主张训练与部署一致）；
3. **否则** → `final_method: ECC_deploy`（RUN_ECC 选定版本，保留）。

**本轮选型完成即冻结方法；正式实验前不再修改机制、条件或判据。**

只报告：经济任务对 SAC_C（不伤害）、期限分档（< 160 / 160–180 / ≥ 180 / 经济，分档依据来自 RUN_ECC 的探索性观察）剂量关系、关键决策比例、持续率、兑现率、破坏率与命题 4 预言对实测、训练曲线（含训练期超时与每回合介入次数）、在环模型监测模式下的原始行为（对比 EACH_C 监测）、调节器串行耗时（不作实时性主张）。

## 交付

**A 步结果先行：** A 步跑完时（B 步脚本开始训练后即可），在 collab 的独立检出写 `lower/handoffs/gov_a/RESULT.md`（`A_DECISION.json` 的 `criteria`、`mechanism_pass`、`b_coordination` 与各条件数值原样摘录）。科学工作树在 B 步运行期间不切换、不提交。

**B 步结束后一次提交到科学分支（附 SHA256SUMS）：**
- `evidence/GOV_A/`：按 RUN_GOV_A 的交付清单；
- `evidence/GOV_B/`：`B/TRAIN_*/`（`manifest.json`、`finished.json`、`episodes.json`、`model_stats.json`、`heartbeat.json`；不上传模型 zip，SHA256 见 `REGISTRY_RESOLVED.json`）、`C/**`、`DRY/MANIFESTS.txt`、`B_DECISION.json`、`CHECK.json`、`REGISTRY_RESOLVED.json`、`ENV_LOCK.txt`、`pytest.log`、`timeline.log`。

**collab `lower/handoffs/gov_b/RESULT.md`：** 自检各项；`b_coordination` 与在环配置；`B_DECISION.json` 的 `final_method`、两档的门槛与数值原样摘录；墙钟与异常（含训练中 unsafe 的回合信息）。

**交付后停止。** 方法冻结、文档统一重写与正式实验执行书由上层另行完成。
