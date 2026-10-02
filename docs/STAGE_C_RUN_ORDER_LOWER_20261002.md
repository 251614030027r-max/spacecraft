# 下层执行单：阶段 C（C1 → C2 → C3 → 判读 → 分支）（2026-10-02）

*上层窗口写。判据只以 `docs/STAGE_C_PREREGISTRATION_20261002.md` 为准；两者冲突时以预注册为准，并停下报上层。本执行单从 C1 一直写到"下一次训练开始"，中途不需要等上层，除非遇到停止条件。*

---

## 0. 这一阶段做什么（先读）

B1/B2 已经证明："学习策略先飞、在对的时刻交给 Pure MPC"在每条轨迹上都存在。阶段 C 训练一个**判断器**，只看当前状态，估计"现在交给 Pure MPC 能不能干净完成"；然后在全新的开局块上闭环检验：判断器有把握就交接，否则继续学习策略。

| 步 | 内容 | 预计 |
|---|---|---|
| C1 | 重跑三个模型在 262000 块上的学习策略回合，记录每一步的观测，并与 B1 逐位核对 | 1–2 小时 |
| C2 | 训练判断器，按预注册规则定阈值 τ | 约 5 分钟 |
| C3 | 在新块 266000–266047 上闭环跑 Pure MPC、学习策略单独、学习策略 + 判断器 | 3–6 小时 |
| 判读 | 按预注册给出 `A_GO_TO_E` / `B_GO_TO_D` / `C_STOP` | 几秒 |

**不做的事**：
- 不改 SAC、接口、奖励、任务、Pure MPC；
- 不调 τ，不换网络，不补数据；
- C3 跑完之前不看任何结果。

工程 `D:\py\DRL2`，PowerShell，`$py = "D:/py/DRL2/.venv/Scripts/python.exe"`。每个进程设置单线程：

```powershell
$env:OMP_NUM_THREADS=1; $env:MKL_NUM_THREADS=1; $env:OPENBLAS_NUM_THREADS=1
```

## 1. 准备

### 1.1 同步与核对（任一不过就停）

```powershell
Set-Location D:\py\DRL2
git pull --ff-only origin claude/sac-mpc-coupling-design-ns7g6i
git rev-parse HEAD                               # 抄进 REPORT.md；C1–C3 全程用这一个提交
git status --porcelain --untracked-files=no      # 必须没有输出
# 物理系统、扫描工具、主线配置相对 B 阶段扫描提交 a714c61 没有改动（必须没有输出）：
git diff a714c61 HEAD --stat -- env controllers dynamics train experiments/v3_common.py experiments/v3_handoff_scan.py experiments/evaluate_hybrid_policy.py
& $py -B -m pytest -q tests/test_v3_stage_c.py tests/test_v3_handoff_scan.py tests/test_v3_mainline.py
```

### 1.2 把 B2-lite 的扫描文件放到主工程

B2-lite 是在固定于 `a714c61` 的独立工作树里跑的。把它的 `eval/v3e/stage_b2/<模型>/seed_*.json` 原样复制到主工程的 `eval/v3e/stage_b2/<模型>/`，然后用 B2 交付包里的 `JSON_SHA256.txt` 逐个核对哈希，必须全部一致。B1 的扫描文件已经在主工程的 `eval/v3e/stage_b/<模型>/`。

核对每个目录都有 48 个文件：

```powershell
foreach ($m in "262420","262421","262422") { "{0}: B1 {1}/48, B2 {2}/48" -f $m, (Get-ChildItem eval/v3e/stage_b/$m -Filter "seed_262*.json").Count, (Get-ChildItem eval/v3e/stage_b2/$m -Filter "seed_262*.json").Count }
```

## 2. C1：记录观测（3 个进程，每个模型 1 个）

```powershell
New-Item -ItemType Directory -Force eval/v3e/stage_c, eval/v3e/stage_c_logs | Out-Null
foreach ($m in "262420","262421","262422") {
  Start-Process -FilePath $py -WorkingDirectory "D:\py\DRL2" -WindowStyle Hidden -PassThru `
    -ArgumentList "-u","-B","-m","experiments.v3_stage_c","record","--run-dir","logs/v3e_$m","--b1-dir","eval/v3e/stage_b/$m","--output","eval/v3e/stage_c/c1_$m.npz" `
    -RedirectStandardOutput "eval/v3e/stage_c_logs/c1_$m.out.log" -RedirectStandardError "eval/v3e/stage_c_logs/c1_$m.err.log"
}
```

每个开局完成时，日志打印 `seed …: N decisions, matches B1`。完成后应有 `eval/v3e/stage_c/c1_<模型>.npz` 和同名 `.json`。

**停止条件**：出现 `replay differs from the B1 learned episode` 或其他报错，就停下报上层，附开局号和报错。不要绕过核对。

## 3. C2：训练判断器（单进程，约 5 分钟）

```powershell
$C = "eval/v3e/stage_c"
& $py -B -m experiments.v3_stage_c fit `
  --record "262420=$C/c1_262420.npz" --record "262421=$C/c1_262421.npz" --record "262422=$C/c1_262422.npz" `
  --b1 262420=eval/v3e/stage_b/262420 --b1 262421=eval/v3e/stage_b/262421 --b1 262422=eval/v3e/stage_b/262422 `
  --b2 262420=eval/v3e/stage_b2/262420 --b2 262421=eval/v3e/stage_b2/262421 --b2 262422=eval/v3e/stage_b2/262422 `
  --output-dir $C/c2
```

屏幕会打印 `c2_verdict`：
- `TAU_FOUND`：生成了 `$C/c2/handoff_classifier.pt` 和 `c2_report.json`，进入第 4 步；
- `C_STOP`：找不到满足精度要求的阈值，判断器识别不了。交付 `c2_report.json`（第 7 节），停下。

## 4. C3：新块闭环（共 7 组，6 个计算槽）

开局块固定为 266000–266047，是工具的默认值，不用另外指定。每组输出到各自目录；同一组可以开多个进程，用法同 B 阶段的认领锁。

```powershell
$C = "eval/v3e/stage_c"
$clf = "$C/c2/handoff_classifier.pt"
function Start-C3($name, $args_) {
  Start-Process -FilePath $py -WorkingDirectory "D:\py\DRL2" -WindowStyle Hidden -PassThru `
    -ArgumentList (@("-u","-B","-m","experiments.v3_stage_c","evaluate") + $args_) `
    -RedirectStandardOutput "eval/v3e/stage_c_logs/c3_$name.out.log" -RedirectStandardError "eval/v3e/stage_c_logs/c3_$name.err.log"
}
# 第一批 6 个进程：每个模型的混合行与学习策略单独行
foreach ($m in "262420","262421","262422") {
  Start-C3 "hybrid_$m"  @("--run-dir","logs/v3e_$m","--row","hybrid","--classifier",$clf,"--output-dir","$C/c3/hybrid/$m")
  Start-C3 "learned_$m" @("--run-dir","logs/v3e_$m","--row","learned","--output-dir","$C/c3/learned/$m")
}
```

第一批中任意一组跑完、空出槽位后，启动 Pure MPC 行，可以开 2 个进程。Pure MPC 与模型无关，借用 262420 的运行目录只是为了取配置：

```powershell
Start-C3 "pure_1" @("--run-dir","logs/v3e_262420","--row","pure","--output-dir","$C/c3/pure")
Start-C3 "pure_2" @("--run-dir","logs/v3e_262420","--row","pure","--output-dir","$C/c3/pure")
```

查看进度（每组做完是 48）：

```powershell
Get-ChildItem $C/c3 -Recurse -Filter "seed_*.json" | Group-Object DirectoryName | ForEach-Object { "{0}: {1}/48" -f $_.Name, $_.Count }
Select-String -Path eval/v3e/stage_c_logs/*.err.log -Pattern "Traceback|Error"
```

**纪律**：判读之前不打开任何结果文件，查进度只数文件个数。进程异常退出时，先确认没有同组进程在运行，再删掉该组目录下残留的 `*.lock` 和 `*.tmp`，用同样的命令续跑。**C3 期间不拉代码。**

## 5. 判读

```powershell
& $py -B -m experiments.v3_stage_c readout --pure $C/c3/pure `
  --learned 262420=$C/c3/learned/262420 --learned 262421=$C/c3/learned/262421 --learned 262422=$C/c3/learned/262422 `
  --hybrid 262420=$C/c3/hybrid/262420 --hybrid 262421=$C/c3/hybrid/262421 --hybrid 262422=$C/c3/hybrid/262422 `
  --output $C/c3_readout.json
```

屏幕打印 `verdict`。下层不解释，按第 6 节办。

## 6. 按判定走

| verdict | 含义 | 下一步 |
|---|---|---|
| `A_GO_TO_E` | 协调成立，效率也已达标 | 交付 C 包（第 7 节），**直接开始 E 的训练**（6.1） |
| `B_GO_TO_D` | 协调成立，效率仍不达标 | 交付 C 包，**等上层改接口**，再按上层的新执行单重训 |
| `C_STOP` | 判断器在闭环里站不住 | 交付 C 包，停下 |
| `FIDELITY_FAIL` | 结果文件不全或版本不一致 | 交付 `problems` 全文，停下 |

### 6.1 仅当 `A_GO_TO_E`：开始 E 的训练（3 个进程，约 30 小时）

同一配置、全新种子，各从零训练 60k，命令与 v3e 原训练相同，只换种子和名字：

```powershell
foreach ($s in 262430, 262431, 262432) {
  Start-Process -FilePath $py -WorkingDirectory "D:\py\DRL2" -WindowStyle Hidden -PassThru `
    -ArgumentList "-B","-m","train.train_hybrid","--steps","60000","--seed","$s","--run-name","v3e_$s","--horizon","35","--parametrization","task_state_v3","--adaptive-task","--device","cpu" `
    -RedirectStandardOutput "logs/train_v3e_$s.out.log" -RedirectStandardError "logs/train_v3e_$s.err.log"
}
```

训练开始后，用下面的命令核对新运行的配置就是主线配置，打印 `True` 才算对：

```powershell
& $py -B -c "import json; from dataclasses import asdict; from train.mainline import mainline_v3e_configs; m=json.load(open('logs/v3e_262430/manifest.json')); print(asdict(mainline_v3e_configs()[1])==m['hybrid'])"
```

训练期间上层会写 E 的评估预注册：判断器原样冻结，评估块 267000–267047，在新块上重跑 Pure MPC。**训练期间不要拉代码**；训练结束、上层的 E 执行单入库后再拉取。

## 7. 交付

- **C 包（zip）**：
  - `REPORT.md`：提交号；1.1 各项输出；B2 文件哈希核对结果；各步起止时间；异常与处理；C2 和判读打印的结论；
  - C1：`c1_*.npz` 和 `c1_*.json`；
  - C2：`c2_report.json` 和 `handoff_classifier.pt`（体积很小，可以打包），附 SHA-256；
  - C3：`c3/**/seed_*.json` 全部逐开局文件；`c3_readout.json`；
  - 全部日志；`JSON_SHA256.txt`、`ALL_FILES_SHA256.txt`；
- **审查分支** `review/v3e-stage-c-<日期>`：只放 `REPORT.md`、`c2_report.json`、`c3_readout.json` 和 SHA 清单；
- **GitHub release**：放完整 zip。

模型文件只报路径和 SHA-256，不打包。并行耗时不作为实时性结论。
