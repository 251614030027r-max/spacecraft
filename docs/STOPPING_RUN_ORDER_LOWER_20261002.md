# 下层执行单：learned stopping option（开发 run → 正式 3×60k → 正式评估）（2026-10-02）

*上层窗口写。判据只以 `docs/STOPPING_METHOD_PREREGISTRATION_20261002.md` 为准；两者冲突时以预注册为准，并停下报上层。*

---

## 0. 这一轮做什么（先读）

方法：原 V3e 任务策略 + 可学习的"现在交给 Pure MPC"停止头，交接价值用真实 MPC 后段回报学习，折扣 0.999。代码 `train/stopping.py`、入口 `train/train_stopping.py`、评估 `experiments/v3_stopping.py`。旧的 `train.train_hybrid` 不用。

| 步 | 内容 | 预计 |
|---|---|---|
| 1 | 同步与核对 | 几分钟 |
| 2 | 开发 run：种子 262440，20k 仿真决策 | 约 10 小时 |
| 3 | 开发评估：266000–266047，stopping 行；devcheck | 1–2 小时 |
| 4 | 三项全过 → 正式训练 262430/262431/262432，各 60k | 约 30 小时 |
| 5 | 正式评估：267000–267047，7 组 | 数小时 |
| 6 | 判读、交付 | 几分钟 |

**不做的事**：不改任何代码和配置；不看成功率调参；不挑检查点；正式评估跑完之前不打开结果文件。

工程 `D:\py\DRL2`，PowerShell，`$py = "D:/py/DRL2/.venv/Scripts/python.exe"`。每个进程单线程：

```powershell
$env:OMP_NUM_THREADS=1; $env:MKL_NUM_THREADS=1; $env:OPENBLAS_NUM_THREADS=1
```

## 1. 同步与核对（任一不过就停）

```powershell
Set-Location D:\py\DRL2
git pull --ff-only origin claude/sac-mpc-coupling-design-ns7g6i
git rev-parse HEAD                               # 抄进 REPORT.md；从开发 run 到正式评估全程用这一个提交
git status --porcelain --untracked-files=no      # 必须没有输出
# 物理系统、控制器、接口相对 B/C 阶段没有改动（必须没有输出）：
git diff a714c61 HEAD --stat -- env controllers dynamics
& $py -B -m pytest -q tests/test_v3_stopping.py tests/test_v3_mainline.py tests/test_v3_stage_c.py tests/test_v3_handoff_scan.py
```

## 2. 开发 run（1 个进程）

```powershell
Start-Process -FilePath $py -WorkingDirectory "D:\py\DRL2" -WindowStyle Hidden -PassThru `
  -ArgumentList "-u","-B","-m","train.train_stopping","--steps","20000","--seed","262440","--run-name","stop_dev_262440","--device","cpu" `
  -RedirectStandardOutput "logs/train_stop_dev_262440.out.log" -RedirectStandardError "logs/train_stop_dev_262440.err.log"
```

进度：`logs/stop_dev_262440/train.monitor.csv` 的行数是回合数；屏幕日志里 `total_timesteps` 是仿真决策数。`stop/handoff_episodes`、`stop/beta_mean` 是停止头的统计，只记录，不据此做任何事。

结束后 `logs/stop_dev_262440/manifest.json` 的 `status` 应为 `completed`，并有 `final_model.zip`。

## 3. 开发评估与结构检查

开 3 个进程跑同一组（认领锁会自动分开局）：

```powershell
New-Item -ItemType Directory -Force eval/stopping, eval/stopping_logs | Out-Null
foreach ($i in 1..3) {
  Start-Process -FilePath $py -WorkingDirectory "D:\py\DRL2" -WindowStyle Hidden -PassThru `
    -ArgumentList "-u","-B","-m","experiments.v3_stopping","evaluate","--run-dir","logs/stop_dev_262440","--row","stopping","--seeds","266000-266047","--output-dir","eval/stopping/dev/stopping" `
    -RedirectStandardOutput "eval/stopping_logs/dev_$i.out.log" -RedirectStandardError "eval/stopping_logs/dev_$i.err.log"
}
```

48 个文件齐了以后：

```powershell
& $py -B -m experiments.v3_stopping devcheck --run-dir logs/stop_dev_262440 --eval-dir eval/stopping/dev/stopping --output eval/stopping/dev/devcheck.json
```

屏幕打印 D1/D2/D3 和 `all_pass`。

- `all_pass = true`：交付开发包（第 6 节），**同时立即开始第 4 步**，不用等上层；
- 任何一项 `false`：交付开发包，停下，等上层与用户讨论。

## 4. 正式训练（3 个进程，与开发 run 同一提交）

```powershell
foreach ($s in 262430, 262431, 262432) {
  Start-Process -FilePath $py -WorkingDirectory "D:\py\DRL2" -WindowStyle Hidden -PassThru `
    -ArgumentList "-u","-B","-m","train.train_stopping","--steps","60000","--seed","$s","--run-name","stop_$s","--device","cpu" `
    -RedirectStandardOutput "logs/train_stop_$s.out.log" -RedirectStandardError "logs/train_stop_$s.err.log"
}
```

训练期间不拉代码。三个都 `completed` 后进入第 5 步。

## 5. 正式评估（267000–267047，7 组）

```powershell
$E = "eval/stopping/formal"
function Start-E($name, $args_) {
  Start-Process -FilePath $py -WorkingDirectory "D:\py\DRL2" -WindowStyle Hidden -PassThru `
    -ArgumentList (@("-u","-B","-m","experiments.v3_stopping","evaluate","--seeds","267000-267047") + $args_) `
    -RedirectStandardOutput "eval/stopping_logs/$name.out.log" -RedirectStandardError "eval/stopping_logs/$name.err.log"
}
foreach ($s in 262430, 262431, 262432) {
  Start-E "stopping_$s" @("--run-dir","logs/stop_$s","--row","stopping","--output-dir","$E/stopping/$s")
  Start-E "learned_$s"  @("--run-dir","logs/stop_$s","--row","learned","--output-dir","$E/learned/$s")
}
```

任一组跑完空出槽位后，启动 Pure MPC 行（与模型无关，借用 262430 的运行目录只为取配置；可开 2 个进程）：

```powershell
Start-E "pure_1" @("--run-dir","logs/stop_262430","--row","pure","--output-dir","$E/pure")
Start-E "pure_2" @("--run-dir","logs/stop_262430","--row","pure","--output-dir","$E/pure")
```

查进度只数文件个数：

```powershell
Get-ChildItem $E -Recurse -Filter "seed_*.json" | Group-Object DirectoryName | ForEach-Object { "{0}: {1}/48" -f $_.Name, $_.Count }
Select-String -Path eval/stopping_logs/*.err.log -Pattern "Traceback|Error"
```

进程异常退出时，先确认同组没有进程在跑，再删掉该组目录下残留的 `*.lock` 和 `*.tmp`，用同样的命令续跑。

## 6. 判读与交付

```powershell
& $py -B -m experiments.v3_stopping readout --pure $E/pure `
  --stopping 262430=$E/stopping/262430 --stopping 262431=$E/stopping/262431 --stopping 262432=$E/stopping/262432 `
  --learned 262430=$E/learned/262430 --learned 262431=$E/learned/262431 --learned 262432=$E/learned/262432 `
  --output $E/readout.json
```

屏幕打印 `verdict`（`METHOD_HOLDS` / `METHOD_DOES_NOT_HOLD` / `FIDELITY_FAIL`）。下层不解释。

**开发包（zip）**：`REPORT.md`（提交号、第 1 节输出、起止时间、异常与处理、devcheck 打印）；`logs/stop_dev_262440/` 下的 `manifest.json`、`train.monitor.csv`、`tensorboard/`；`eval/stopping/dev/` 全部；全部日志；`ALL_FILES_SHA256.txt`。`final_model.zip` 只报路径和 SHA-256。

**正式包（zip）**：`REPORT.md`；三个运行的 `manifest.json`、`train.monitor.csv`、`tensorboard/`；`eval/stopping/formal/` 全部逐开局文件和 `readout.json`；全部日志；`JSON_SHA256.txt`、`ALL_FILES_SHA256.txt`；模型文件只报路径和 SHA-256。审查分支 `review/stopping-<日期>` 只放 `REPORT.md`、`devcheck.json`、`readout.json` 和 SHA 清单；完整 zip 放 GitHub release。

并行耗时不作为实时性结论。
