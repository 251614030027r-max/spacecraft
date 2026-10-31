# 最终主线：预注册与下层完整执行单（2026-10-06）

*上层窗口写。用户与另一窗口讨论后拍定（`CleanTraining_20261006` 意见）：先确认工况可行，修掉唯一已知的停止缺陷，然后从零干净训练、全新块正式评估。本文件与代码在同一次提交中入库，早于任何 269000 / 271000 数据。判据写在代码常数中，事后不改。下层从头执行到正式交付，中途只在写明的停止点停下。*

---

## 0. 一页看懂

| 步 | 内容 | 预计 | 停止点 |
|---|---|---|---|
| 1 | 同步与核对 | 几分钟 | 任一项不过 |
| 2 | 工况筛选第 1 阶段：9 格，只跑 Pure，块 269000 | 3–5 小时 | — |
| 3 | 工况筛选第 2 阶段：通过的格跑平缓推进，按规则选定工况 | 2–3 小时 | 没有选中的工况 |
| 4 | 正式训练：选定工况，种子 262450 / 262451 / 262452，各 60k，从零 | 40–50 小时 | — |
| 5 | 结构检查，嵌在训练中：30k 检查点，开发块 266000 | 与训练并行，约 2 小时 | 结构检查不过 |
| 6 | 正式评估：全新块 271000，Pure / learned / stopping 共 7 组 | 6–10 小时 | — |
| 7 | 判读、交付 | 几分钟 | — |

结构检查嵌在正式训练里做，不另跑开发 run，可以省掉约 15 小时。正式评估只用各运行的 `final_model.zip`。30k 检查点只用于结构检查，**不参与任何性能比较，也不能被挑选**。

## 1. 最终方法（冻结）

- **任务策略**：原 V3e 二维 SAC，接口、奖励各项、Pure MPC 都不变。γ = 0.999。
- **交接价值** Q_H：回归真实 Pure MPC 后段回报，只用 31 维与策略无关的状态。
- **停止规则**（与 10-02 轮的唯一区别）：
  - 不再有单独的停止头网络；
  - β(s) = σ((Q_H(s) − V_C(s)) / 0.005)，其中 V_C(s) = min_i Q_C^i(s, μ(s))，即"按确定性任务策略继续"的价值；
  - 同一个定义用在三处：Bellman 目标（用目标 critic）、训练时的交接采样、部署；
  - 部署时 β ≥ 0.5 等价于 Q_H(s) ≥ V_C(s)。代码中 `StoppingConfig.stop_rule = "value"`，测试已钉住二者逐点一致。
- **不变的部分**：
  - 训练期交接概率限制在 [0.002, 0.01]，部署不用；
  - 后段只给 Q_H 提供标签，不占预算，也不产生更新；
  - 预算按外层决策计。
- **为什么改**：10-02 轮的停止头和自己的价值不一致，127 次交接中有 59 次发生在 Q_H < Q_C 时。268000 块上改用价值比较后，毁掉 Pure 成功的数量从 5/4/5 降到 5/2/2（`docs/STOPPING_VALUE_RULE_RESULT_20261006.md`）。这是修缺陷，不加新模块。

## 2. 工况（由第 3 步按规则选定）

- 只改目标翻滚速率和初始距离下限，备用时改初始速度上限。MPC、推力、约束、奖励、接口、时间上限都不动。
- 筛选规则见 `docs/REGIME_SCREEN_20261006.md`：
  - 第 1 阶段通过：Pure 干净完成 29–36 / 48，超时占失败的比例 ≥ 2/3，违规回合 ≤ 2；
  - 第 2 阶段：平缓推进至少救回 3 个 Pure 失败的开局；
  - 在满足以上条件的格中，取离现主线最近的。
- 选择不看任何学习或停止结果。

## 3. 正式判据（与此前正式轮相同）

在选定工况、全新块 **271000–271047** 上：

| 判定 | 条件 |
|---|---|
| 每个模型通过 | stopping 干净完成 > 同块 Pure；违规回合 ≤ Pure；毁掉的 Pure 成功开局 ≤ 2 |
| `METHOD_HOLDS` | ≥ 2 个模型通过 |
| `METHOD_DOES_NOT_HOLD` | 否则 |
| `FIDELITY_FAIL` | 文件不全、提交不一致、工作树不干净、截断或模型混用 |

**必报，不作门槛**：
- 救回 / 毁掉；
- 交接时刻分布；
- 时间与 Δv；
- 协调增益（stopping 相对同一模型禁止交接的 learned 行）。

读法写在前面：
- learned 行如果和 stopping 一样好甚至更好，只能说"训练改善了任务策略"，不能说交接改善了闭环；
- "交接有贡献"至少要满足一条：
  - stopping 比 learned 完成更多；
  - 完成数不降，而时间或 Δv 明显下降；
  - 明确救回 learned 的失败，且很少毁掉。

## 4. 结构检查（第 5 步，嵌在训练中）

在每个种子的 30k 检查点上，用开发块 266000–266047 跑 stopping 行，然后跑 `devcheck`：

| 检查 | 通过条件 |
|---|---|
| D1 | 第 0 步就交接的开局 ≤ 43，且从不交接的 ≤ 43 |
| D2 | 训练最后 1/4 的回合里 ≥ 25% 飞了 ≥ 30 个学习决策；评估中 ≥ 5/48 个开局飞了 ≥ 30 个学习决策 |
| D3 | 中途交接出现在 ≥ 3 个不同的步；Q_H − V_C 在学习段内的变化幅度中位数 ≥ 1 |
| D4 | 每一个学习段决策上，"β ≥ 0.5"与"Q_H ≥ V_C"完全一致 |

- **继续训练**：三个种子 D4 全过，且 D1–D3 至少 2 个种子全过。
- **否则**：停止全部训练，交付，等上层。上层只修有名字的实现错误，不按成功率调参，修好后全部重训。

---

## 5. 下层执行单

工程 `D:\py\DRL2`，PowerShell，`$py = "D:/py/DRL2/.venv/Scripts/python.exe"`。每个进程单线程：

```powershell
$env:OMP_NUM_THREADS=1; $env:MKL_NUM_THREADS=1; $env:OPENBLAS_NUM_THREADS=1
```

### 5.1 同步与核对（任一不过就停）

```powershell
Set-Location D:\py\DRL2
git pull --ff-only origin claude/sac-mpc-coupling-design-ns7g6i
git rev-parse HEAD                               # 抄进 REPORT.md；第 2–6 步全程用这一个提交，期间不拉代码
git status --porcelain --untracked-files=no      # 必须没有输出
git diff 2e5c236 HEAD --stat -- env controllers dynamics   # 必须没有输出
& $py -B -m pytest -q tests/test_regime_screen.py tests/test_v3_stopping.py tests/test_v3_mainline.py
```

### 5.2 工况筛选第 1 阶段（9 格，只跑 Pure）

```powershell
$R = "eval/regime_screen"
New-Item -ItemType Directory -Force $R, eval/regime_logs | Out-Null
function Start-R($regime, $row, $i) {
  Start-Process -FilePath $py -WorkingDirectory "D:\py\DRL2" -WindowStyle Hidden -PassThru `
    -ArgumentList "-u","-B","-m","experiments.regime_screen","evaluate","--regime",$regime,"--row",$row,"--output-dir","$R/$regime/$row" `
    -RedirectStandardOutput "eval/regime_logs/${regime}_${row}_$i.out.log" -RedirectStandardError "eval/regime_logs/${regime}_${row}_$i.err.log"
}
$stage1 = "w2.36_r15","w2.36_r18","w2.36_r20","w3.00_r15","w3.00_r18","w3.00_r20","w3.50_r15","w3.50_r18","w3.50_r20"
foreach ($g in $stage1) { Start-R $g "pure" 1 }
```

- 核多的话，同一格可以再执行一次 `Start-R $g "pure" 2`，认领锁会自动分配开局。
- 查进度：

```powershell
Get-ChildItem $R -Recurse -Filter "seed_*.json" | Group-Object DirectoryName | ForEach-Object { "{0}: {1}/48" -f $_.Name, $_.Count }
Select-String -Path eval/regime_logs/*.err.log -Pattern "Traceback|Error"
```

9 格各 48 个文件齐了以后：

```powershell
& $py -B -m experiments.regime_screen readout --root $R --output $R/readout_stage1.json
```

### 5.3 工况筛选第 2 阶段（平缓推进）并选定

对 `passing` 中的每一格执行 `Start-R "<格名>" "nominal" 1`，可多开一个进程编号为 2。全部跑完后：

```powershell
& $py -B -m experiments.regime_screen readout --root $R --output $R/readout_final.json
```

屏幕打印 `selected`。

- **`selected` 是一个格名**：记为 `$G`，立即进入 5.4，不用等上层。
- **`passing` 为空，且所有格的 Pure 都 > 36**：对 `w3.00_r18_v0.15`、`w3.50_r18_v0.15` 依次做 5.2、5.3，再判读。
- **其他情况**（没有格通过，或通过的格平缓推进都救不到 3 个）：交付筛选包，停下。

### 5.4 正式训练（3 个进程）

```powershell
$G = "<selected 打印的格名>"
foreach ($s in 262450, 262451, 262452) {
  Start-Process -FilePath $py -WorkingDirectory "D:\py\DRL2" -WindowStyle Hidden -PassThru `
    -ArgumentList "-u","-B","-m","train.train_stopping","--steps","60000","--seed","$s","--run-name","final_$s","--regime",$G,"--device","cpu" `
    -RedirectStandardOutput "logs/train_final_$s.out.log" -RedirectStandardError "logs/train_final_$s.err.log"
}
```

开始后核对 manifest：`regime.name` 等于 `$G`，`stopping.stop_rule` 等于 `value`，`code_dirty` 为 `false`。

```powershell
foreach ($s in 262450, 262451, 262452) { & $py -B -c "import json; m=json.load(open('logs/final_$s/manifest.json')); print($s, m['regime']['name'], m['stopping']['stop_rule'], m['code_dirty'])" }
```

### 5.5 结构检查（30k 检查点出现后，与训练并行）

每个种子的 `logs/final_<种子>/checkpoints/stopping_30000_outer_decisions.zip` 出现后：

```powershell
$D = "eval/final/devcheck"
foreach ($s in 262450, 262451, 262452) {
  Start-Process -FilePath $py -WorkingDirectory "D:\py\DRL2" -WindowStyle Hidden -PassThru `
    -ArgumentList "-u","-B","-m","experiments.v3_stopping","evaluate","--run-dir","logs/final_$s","--model-name","checkpoints/stopping_30000_outer_decisions.zip","--allow-incomplete-run","--row","stopping","--seeds","266000-266047","--output-dir","$D/$s" `
    -RedirectStandardOutput "eval/regime_logs/dev_$s.out.log" -RedirectStandardError "eval/regime_logs/dev_$s.err.log"
}
```

每组 48 个文件齐了以后：

```powershell
foreach ($s in 262450, 262451, 262452) {
  & $py -B -m experiments.v3_stopping devcheck --run-dir logs/final_$s --eval-dir $D/$s --output $D/devcheck_$s.json
}
```

**按第 4 节判定**：
- 三个种子 D4 全过，且 D1–D3 至少 2 个种子全过：训练继续，什么都不用做；
- 否则：停止三个训练进程（`Stop-Process`），交付，等上层。

### 5.6 正式评估（三个训练都 `completed` 后，块 271000–271047）

```powershell
$E = "eval/final/formal"
function Start-E($name, $args_) {
  Start-Process -FilePath $py -WorkingDirectory "D:\py\DRL2" -WindowStyle Hidden -PassThru `
    -ArgumentList (@("-u","-B","-m","experiments.v3_stopping","evaluate","--seeds","271000-271047") + $args_) `
    -RedirectStandardOutput "eval/regime_logs/f_$name.out.log" -RedirectStandardError "eval/regime_logs/f_$name.err.log"
}
foreach ($s in 262450, 262451, 262452) {
  Start-E "stopping_$s" @("--run-dir","logs/final_$s","--row","stopping","--output-dir","$E/stopping/$s")
  Start-E "learned_$s"  @("--run-dir","logs/final_$s","--row","learned","--output-dir","$E/learned/$s")
}
Start-E "pure_1" @("--run-dir","logs/final_262450","--row","pure","--output-dir","$E/pure")
Start-E "pure_2" @("--run-dir","logs/final_262450","--row","pure","--output-dir","$E/pure")
```

- 核多的话，任何一组都可以多开进程。
- Pure 行与模型无关，借用 262450 的运行目录，只为取到选定工况的配置。
- 查进度只数文件个数；判读前不打开结果。

### 5.7 判读

```powershell
& $py -B -m experiments.v3_stopping readout --pure $E/pure `
  --stopping 262450=$E/stopping/262450 --stopping 262451=$E/stopping/262451 --stopping 262452=$E/stopping/262452 `
  --learned 262450=$E/learned/262450 --learned 262451=$E/learned/262451 --learned 262452=$E/learned/262452 `
  --seeds 271000-271047 --output $E/readout.json
```

屏幕打印 `verdict`。`readout.json` 里含 `coordination_gain`。下层不解释。

### 5.8 交付

审查分支 `review/final-mainline-<日期>`，放以下内容：
- `REPORT.md`：提交号、5.1 的输出、选定工况、各步起止时间、异常及处理、所有判读打印；
- 筛选的两份 readout，以及全部逐开局 JSON；
- 三份 `devcheck_*.json`，以及结构检查的逐开局 JSON；
- 三个运行的 `manifest.json`、`train.monitor.csv`；
- 正式评估的全部逐开局 JSON 和 `readout.json`；
- 全部日志；
- SHA 清单。

模型只报路径和 SHA-256，不打包。

**进程异常退出时**：先确认同组没有进程在跑，再删掉该组目录下残留的 `*.lock` 和 `*.tmp`，用同样的命令续跑。训练进程中断不能续跑，交付并等上层。
