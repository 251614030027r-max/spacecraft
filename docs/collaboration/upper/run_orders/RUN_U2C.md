# 执行书：RUN_U2C（U2 决策时认证的配对评估；下层本机；不训练，约 2.5 小时）

**状态：用户 11-07 批准，要求上下层最快推进；转发即下发。**

**做什么：**
- U2 的执行器探询（网格、时域、执行器规则都与 RUN_U2 相同）改为在**决策时**认证 SAC 的意图：
  - SAC 的意图在 48 s 内不会让执行器切换到入口 / 下降模式；
  - 某个探测意图会；
  - 错过这次过顶期限前就不会再有入口。
  
  三条同时成立时，改执行离 SAC 意图最近、且能赶上的探测意图；否则原样执行。
- 规格：`docs/final_method/U2_CERT_SPEC.md`；判据：`experiments/cert_readout.py`（随冻结提交登记，先于评估数据）。
- **不训练。** 用预实验的 EACH、SAC 模型（种子 10、11），在新开局 294220–294267 上各跑认证关闭 / 开启两遍，同一检查点配对。

**可直接转发给下层的一句话：**

> 读 collab `docs/collaboration/upper/run_orders/RUN_U2C.md`，在 `D:\py\DRL2` detached 检出 `28a4079907c4d7a18c5c44e17383315e0d4a9705`，按第 0 节自检，通过后运行第 1 节脚本（带 BOM 的 UTF-8 存为 `run.ps1`，隐藏运行），跑完按"交付"提交，不改代码、配置与判据。

**科学提交（冻结）：** `28a4079907c4d7a18c5c44e17383315e0d4a9705`（env 0.15.0；认证默认关闭，EACH 训练路径与 `fd789126` 逐位一致，见 `evidence/U2C_PRECHECK/`）。

## 0. 准备与自检

```powershell
cd D:\py\DRL2; git fetch origin claude/sac-mpc-coupling-design-ns7g6i; git checkout --detach 28a4079907c4d7a18c5c44e17383315e0d4a9705
$py  = "D:\py\DRL2\.venv\Scripts\python.exe"
$Out = "C:\Users\35884\Documents\Spacecraft\过程文件\u2c"
$Pre = "C:\Users\35884\Documents\Spacecraft\过程文件\final_pretest"
if ((git rev-parse HEAD) -ne "28a4079907c4d7a18c5c44e17383315e0d4a9705") { throw "HEAD 不是冻结 SHA" }
if (git status --porcelain --untracked-files=no) { throw "科学工作树有改动" }
if ((Test-Path "$Out\C") -and @(Get-ChildItem "$Out\C" -Force).Count -gt 0) { throw "输出目录非空：$Out\C" }
New-Item -ItemType Directory -Force "$Out\C", "$Out\REUSE" | Out-Null
$env:OMP_NUM_THREADS = "1"; $env:MKL_NUM_THREADS = "1"; $env:OPENBLAS_NUM_THREADS = "1"
& $py -B -m pytest -q tests/test_t2_stage.py tests/test_integrator.py *> "$Out\pytest.log"; if ($LASTEXITCODE -ne 0) { throw "pytest 退出码 $LASTEXITCODE" }
& $py --version *> "$Out\ENV_LOCK.txt"; & $py -m pip freeze *>> "$Out\ENV_LOCK.txt"
$want = @{ "EACH_s10" = "cb9a8968868ae67ead341a9a99ea59bbff0bf81cad7917f7f5bbbfd4578dd388"; "EACH_s11" = "6895eb31bf0b46638236f0ba49860f75127560b2ec793b60edd55200906457d8";
           "SAC_s10"  = "3e2bd45e197889710eefeb35381beeb6b584f42a501257d6d8b7f53c202aa376"; "SAC_s11"  = "1e78cccd3df34a51581cc359045b85a246e30205cdb4218fcc381778ed429dad" }
foreach ($runName in $want.Keys) {
  $got = (Get-FileHash "$Pre\B\TRAIN_$runName\checkpoints\step_2000.zip" -Algorithm SHA256).Hash.ToLower()
  if ($got -ne $want[$runName]) { throw "模型 $runName 的 SHA256 不符：$got" }
  "$runName $got" | Out-File -Encoding utf8 -Append "$Out\REUSE\MODEL_SHA256.txt" }
# 认证关闭时与预实验记录逐项相同（新代码不改原路径）
& $py -B -m experiments.final eval --stage pretest --run "$Pre\B\TRAIN_EACH_s10" --label EACH_s10 --openings 294100-294101 --output "$Out\REUSE\EACH_s10" --workers 2 *>> "$Out\REUSE\eval.log"
& $py -B -m experiments.u2_readout reuse --new "$Out\REUSE\EACH_s10" --old "$Pre\C\EACH_s10" --output "$Out\REUSE\EACH_s10.json" *>> "$Out\REUSE\reuse.log"
if ($LASTEXITCODE -ne 0) { throw "复用核验失败（见 $Out\REUSE\EACH_s10.json）" }
```

**通过标准：**
- pytest 退出码 0；
- 四个模型的 SHA256 与上面一致；
- `REUSE\EACH_s10.json` 中 `identical: true`。

## 1. 脚本

```powershell
$py  = "D:\py\DRL2\.venv\Scripts\python.exe"; $Out = "C:\Users\35884\Documents\Spacecraft\过程文件\u2c"; $Pre = "C:\Users\35884\Documents\Spacecraft\过程文件\final_pretest"
Set-Location "D:\py\DRL2"; $env:OMP_NUM_THREADS = "1"; $env:MKL_NUM_THREADS = "1"; $env:OPENBLAS_NUM_THREADS = "1"
$T0 = Get-Date; "start $T0" | Out-File -Encoding utf8 "$Out\timeline.log"
foreach ($cfgName in "EACH","SAC") { foreach ($seedNo in 10,11) {
  & $py -B -m experiments.final eval --stage u2c --run "$Pre\B\TRAIN_${cfgName}_s$seedNo" --label "${cfgName}_s${seedNo}_cert" --output "$Out\C\${cfgName}_s${seedNo}_cert" --certify --workers 10 *>> "$Out\eval.log"
  & $py -B -m experiments.final eval --stage u2c --run "$Pre\B\TRAIN_${cfgName}_s$seedNo" --label "${cfgName}_s$seedNo" --output "$Out\C\${cfgName}_s$seedNo" --workers 10 *>> "$Out\eval.log"
  "done ${cfgName}_s$seedNo at $(((Get-Date) - $T0).TotalHours) h" | Out-File -Encoding utf8 -Append "$Out\timeline.log" } }
& $py -B -m experiments.final check --stage u2c --root $Out --output "$Out\CHECK.json" *> "$Out\check.log"
& $py -B -m experiments.cert_readout --root $Out --output "$Out\U2C_DECISION.json" *> "$Out\decision.log"
"all done $(((Get-Date) - $T0).TotalHours) h" | Out-File -Encoding utf8 -Append "$Out\timeline.log"
```

## 2. 判据（`experiments/cert_readout.py`，提交即冻结）

全部按种子配对、开局配对，共 96 对。

| 比较 | 作用 | 判定 |
|---|---|---|
| EACH + 认证 对 SAC | 论文主张：反馈对无反馈 | G1–G5 → `pass` / `no_gain` |
| EACH + 认证 对 EACH（同一模型） | U2 认证的贡献 | `pass`（G1–G5）/ `improved` / `no_effect` |
| SAC + 认证 对 SAC（同一模型） | 不依赖 U1 时的贡献 | 同上 |

`improved` 须同时满足：
- 均值 > 0；
- 每个种子 ≥ 0；
- 紧急捕获 ≥ +1，总捕获不减少；
- unsafe 为 0；
- 经济任务均值差 ≥ −2。

**保底（事先登记，见规格第 4 节）：** 两个同模型比较都判为 `no_effect` 时，由上层另写执行书，统一信息边界后重训。

## 交付

**科学分支 `evidence/U2C/`（附 SHA256SUMS）：**
- `C/**`（8 组 × 48）；
- `REUSE/**`；
- `U2C_DECISION.json`、`CHECK.json`、`ENV_LOCK.txt`、`pytest.log`、`timeline.log`、`eval.log`。

**collab `lower/handoffs/u2c/RESULT.md`：**
- 自检各项；
- `U2C_DECISION.json` 的三项判定与介入次数，原样摘录；
- 墙钟。

**交付后停止。**
