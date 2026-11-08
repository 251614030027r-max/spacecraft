# 执行书：RUN_GOV_A（ECC 最终轮 A 步：调节器机制验证，不训练；下层本机，预计约 9–10 小时）

**状态：用户 2026-11-08 批准本轮方案（"方案批准，立即开始落实"）；上层第 0 步完成后下发。转发即执行。**

**可直接转发给下层的一句话：**

> 读 collab `docs/collaboration/upper/run_orders/RUN_GOV_A.md`，在 `D:\py\DRL2` detached 检出 `bc2d02509cd89000fba13a00bfeea1b11e908543`，按第 0 节自检，通过后用带 BOM 的 UTF-8 存第 1 节脚本为 `run.ps1` 并隐藏运行（不限时），跑完按"交付"提交，不改任何代码、配置与判据。

**科学提交（冻结）：** `bc2d02509cd89000fba13a00bfeea1b11e908543`（env 0.17.0、learner 7.3，113 项测试）。

- 规格、判据、选型顺序：`docs/final_method/ECC_FINAL_SPEC.md`（`4d78ce2d`，先于任何数据）；读出 `experiments/gov_readout.py`（`d6f973d6`，先于任何数据）。
- 上层第 0 步证据：`evidence/ECC_FINAL_PRECHECK/`（旧路径逐位等价、294323 / 294334 回归重放、条件筛查与定档）。
- 条件（第 0 步定档，用户 11-08 选定）：自然紧急分布为主轴——主终点全部紧急任务（期限 U(140, 200) s）；经济任务为不伤害检验；原比例 1:1 自然分布为外部有效性。筛查与沙箱配对检查结果：`evidence/ECC_FINAL_PRECHECK/screen/urgent/`、`check6b/round3/`。
- 调节器（冻结）：执行一步后关键机会仍在且机会时刻不后移则不改；替代只引向 G★ 规划的入口；没有合格替代不改。

## 本步做什么

**不训练。** 复用 RUN_ECC 的 4 个模型（`EACH_C`、`SAC_C` × 种子 10、11，2000 步检查点；目录 `C:\Users\35884\Documents\Spacecraft\过程文件\ecc\B\TRAIN_*`），每个模型在三个条件上各评三种协调：

| 条件 | 开局（从未用过，与筛查及 B 步互不相交） | 个数 |
|---|---|---|
| 紧急（主终点） | 294500–294595 | 96 |
| 经济（不伤害） | 294600–294647 | 48 |
| 自然 1:1（外部有效性） | 294700–294747 | 48 |

| 标签后缀 | 协调 | 作用 |
|---|---|---|
| `_monitor` | 调节器监测（只记录、不动作；轨迹与不协调逐位相同，自检核对） | 不协调对照；破坏率、关键决策比例 |
| `_gov` | 新调节器 | 机制主证据 |
| `_cert` | 旧认证（RUN_ECC 规则） | 对照，只报告 |

共 4 × 3 × 192 = 2304 回合。

## 0. 准备与自检

```powershell
cd D:\py\DRL2; git fetch origin claude/sac-mpc-coupling-design-ns7g6i; git checkout --detach bc2d02509cd89000fba13a00bfeea1b11e908543
$py  = "D:\py\DRL2\.venv\Scripts\python.exe"
$Src = "C:\Users\35884\Documents\Spacecraft\过程文件\ecc"
$Out = "C:\Users\35884\Documents\Spacecraft\过程文件\gov_a"
if ((git rev-parse HEAD) -ne "bc2d02509cd89000fba13a00bfeea1b11e908543") { throw "HEAD 不是冻结 SHA" }
if (git status --porcelain --untracked-files=no) { throw "科学工作树有改动" }
if ((Test-Path "$Out\C") -and @(Get-ChildItem "$Out\C" -Force).Count -gt 0) { throw "输出目录非空：$Out\C" }
New-Item -ItemType Directory -Force "$Out\C", "$Out\DRY" | Out-Null
$env:OMP_NUM_THREADS = "1"; $env:MKL_NUM_THREADS = "1"; $env:OPENBLAS_NUM_THREADS = "1"
& $py -B -m pytest -q tests/test_t2_stage.py tests/test_integrator.py *> "$Out\pytest.log"; if ($LASTEXITCODE -ne 0) { throw "pytest 退出码 $LASTEXITCODE" }
& $py --version *> "$Out\ENV_LOCK.txt"; & $py -m pip freeze *>> "$Out\ENV_LOCK.txt"
$old = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'experiments\.' }
if ($old) { throw "有遗留的 python 进程：$($old.ProcessId -join ',')" }
foreach ($m in "EACH_C_s10","EACH_C_s11","SAC_C_s10","SAC_C_s11") {
  if (-not (Test-Path "$Src\B\TRAIN_$m\checkpoints\step_2000.zip") -and -not (Test-Path "$Src\B\TRAIN_$m\final.zip")) { throw "缺少 RUN_ECC 模型 $m" } }
# 监测 = 关闭：同一模型、两个开局，事件、回报、时间逐位相同
& $py -B -m experiments.final eval --stage gov_a --run "$Src\B\TRAIN_EACH_C_s10" --label off --openings 294500-294501 --output "$Out\DRY\off" --govern off --workers 2 *>> "$Out\DRY\dry.log"
& $py -B -m experiments.final eval --stage gov_a --run "$Src\B\TRAIN_EACH_C_s10" --label monitor --openings 294500-294501 --output "$Out\DRY\monitor" --govern monitor --workers 2 *>> "$Out\DRY\dry.log"
foreach ($o in 294500,294501) {
  $a = Get-Content "$Out\DRY\off\off_$o.json" -Raw | ConvertFrom-Json; $b = Get-Content "$Out\DRY\monitor\monitor_$o.json" -Raw | ConvertFrom-Json
  if ($a.event -ne $b.event -or $a.return -ne $b.return -or $a.time_s -ne $b.time_s) { throw "监测与关闭不一致：$o" } }
"monitor == off ok" | Out-File -Encoding utf8 "$Out\DRY\MONITOR_EQUALS_OFF.txt"
```

**通过标准：** pytest 退出码 0；4 个模型都在；`DRY\MONITOR_EQUALS_OFF.txt` 写出。

## 1. 一次性脚本

```powershell
$py  = "D:\py\DRL2\.venv\Scripts\python.exe"
$Src = "C:\Users\35884\Documents\Spacecraft\过程文件\ecc"; $Out = "C:\Users\35884\Documents\Spacecraft\过程文件\gov_a"
Set-Location "D:\py\DRL2"; $env:OMP_NUM_THREADS = "1"; $env:MKL_NUM_THREADS = "1"; $env:OPENBLAS_NUM_THREADS = "1"
$T0 = Get-Date; $hours = { ((Get-Date) - $T0).TotalHours }
"start $T0" | Out-File -Encoding utf8 "$Out\timeline.log"
$sfx = @{ "urgent" = ""; "economy" = "_economy"; "natural" = "_natural" }
foreach ($m in "EACH_C_s10","EACH_C_s11","SAC_C_s10","SAC_C_s11") {
  foreach ($cond in "urgent","economy","natural") { $s = $sfx[$cond]
    & $py -B -m experiments.final eval --stage gov_a --condition $cond --run "$Src\B\TRAIN_$m" --label "${m}_monitor$s" --output "$Out\C\${m}_monitor$s" --govern monitor --workers 10 *>> "$Out\eval.log"
    & $py -B -m experiments.final eval --stage gov_a --condition $cond --run "$Src\B\TRAIN_$m" --label "${m}_gov$s" --output "$Out\C\${m}_gov$s" --govern act --workers 10 *>> "$Out\eval.log"
    & $py -B -m experiments.final eval --stage gov_a --condition $cond --run "$Src\B\TRAIN_$m" --label "${m}_cert$s" --output "$Out\C\${m}_cert$s" --certify --workers 10 *>> "$Out\eval.log" }
  "done $m at $(& $hours) h" | Out-File -Encoding utf8 -Append "$Out\timeline.log" }
& $py -B -m experiments.final check --stage gov_a --root $Out --output "$Out\CHECK.json" *> "$Out\check.log"
& $py -B -m experiments.gov_readout a --root $Out --output "$Out\A_DECISION.json" *> "$Out\decision.log"
"all done $(& $hours) h" | Out-File -Encoding utf8 -Append "$Out\timeline.log"
```

评估中出现 `ERROR_*.json` 时，只对缺失的开局补跑同一命令一次（结果文件已存在的开局会被跳过），并在交付中写明。

## 2. 判据（`experiments/gov_readout.py a`，入库后不改；规格第 7 节）

同一模型、同一开局：调节器（`_gov`）对监测（`_monitor`）。

- **M1**：紧急任务合并均值 > 0，且合并捕获数增加；
- **M2**：0 丢失（任一开局监测下捕获、调节器下未捕获；三个条件都算）；
- **M3**：0 个由调节器引入的 unsafe（同上）；
- **M4（不伤害）**：经济任务每回合替代 ≤ 0.05，且"非关键"决策上的替代为 0；
- **M5**：三个条件各自的同模型均值增益与命题 4 的预言（破坏率 × 兑现率 × 200）方向一致（小于一次翻转当量记为 0）。

**M2 或 M3 不过 → 弃用调节器，B 步的在环与部署协调改用旧认证（`A_DECISION.json` 中 `b_coordination: cert`）。** M1、M4、M5 不过只如实报告。

只报告：旧认证对监测、调节器对旧认证、期限分档（< 160 / 160–180 / ≥ 180 / 经济）剂量关系（分档依据来自 RUN_ECC 的探索性观察）、关键决策比例、持续率、兑现率、破坏率与预言对实测、每次决策的调节器串行耗时（不作实时性主张）。

## 纪律

- 运行期间不改被导入的模块，不切分支、不拉取；
- 外部中断（重启、更新）后只补缺失的确定性评估（结果文件已存在的开局自动跳过），并核对已有文件的哈希，在交付中写明；
- pytest 以退出码判定；
- G★ 是动作空间外的参照，本步不涉及。

## 交付

**科学分支 `evidence/GOV_A/`（附 SHA256SUMS）：** `C/**`（36 组：12 组 × 紧急 96、经济 48、自然 48）、`DRY/MONITOR_EQUALS_OFF.txt`、`A_DECISION.json`、`CHECK.json`、`ENV_LOCK.txt`、`pytest.log`、`timeline.log`、`eval.log`。

**collab `lower/handoffs/gov_a/RESULT.md`：** 自检各项；`A_DECISION.json` 的 `criteria`、`mechanism_pass`、`governor_rejected`、`b_coordination` 与各档数值原样摘录；墙钟与异常。

**交付后停止。** B 步执行书由上层按 A 步结果另行下发。
