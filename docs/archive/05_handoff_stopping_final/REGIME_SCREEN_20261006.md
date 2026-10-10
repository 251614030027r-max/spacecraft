# 最终工况筛选：预注册与下层执行单（2026-10-06）

*上层窗口写。完整执行流程以 `docs/FINAL_MAINLINE_RUN_ORDER_20261006.md` 为准，本文件只保留筛选的预注册。用户 2026-10-06 采纳另一窗口的工况调整意见：方法冻结，任务改到 Pure MPC 不再饱和、但仍物理合理的区域，然后从零干净重训。本文件与代码在同一次提交中入库，早于任何 269000 块数据。选择规则写在 `experiments/regime_screen.py` 中，事后不改。*

---

## 1. 为什么筛选

- 名义工况（2.36°/s，初始距离 15–28 m）下，Pure MPC 在三个块上是 37 / 41 / 39（共 48 个开局），只剩 7–9 个失败，方法几乎没有可提升的余地。
- 只改两类东西：**目标翻滚速率**和**初始状态分布**。
- MPC、推力和力矩上限、约束、完成条件、时间上限、奖励、接口都不动。
- MPC 用测得的目标状态做预测，所以翻滚变快是对它真实的输入变化，不是削弱它。

## 2. 候选工况

**第 1 阶段，只跑 Pure**：翻滚速率 × 初始距离下限，共 9 格。距离上限保持 28 m，初始速度上限保持 0.10 m/s。

| | 15–28 m | 18–28 m | 20–28 m |
|---|---|---|---|
| 2.36°/s（现主线） | `w2.36_r15`（同块参照） | `w2.36_r18` | `w2.36_r20` |
| 3.0°/s | `w3.00_r15` | `w3.00_r18` | `w3.00_r20` |
| 3.5°/s | `w3.50_r15` | `w3.50_r18` | `w3.50_r20` |

**第 2 阶段，只对第 1 阶段通过的格**：跑脚本化平缓推进（`nominal`：学习分支、动作恒为 +1，即正式评估器的 `v3_nominal`），用来看接口在该工况下能否覆盖 Pure 的失败。不涉及任何学习或停止策略。

**第 3 阶段，备用，仅当第 1 阶段没有格通过**：`w3.00_r18_v0.15`、`w3.50_r18_v0.15`，初始速度上限改为 0.15 m/s。

开局块：**269000–269047**，全新，所有格使用同一组种子。

## 3. 选择规则

**第 1 阶段通过**，以下三条同时满足：
1. Pure 干净完成 29–36 / 48，即 60–75%；
2. Pure 的失败中，超时（time_failure）占比 ≥ 2/3，说明失败是时机问题，不是物理不可达；
3. Pure 真值违规回合 ≤ 2。

**最终选定**（2026-10-06 修订，早于任何 269000 数据；只在通过、且已跑完 nominal 的格中选）：
- 先筛"Pure 失败、而平缓推进干净完成"的开局 ≥ 3 个的格，说明接口本身有可救空间；
- 在这些格中取**离现主线最近**的：先比翻滚更慢，再比距离下限更小；
- 不追求可救开局数最大。

**没有格通过时**：
- 如果全部格的 Pure 都 > 36，跑第 3 阶段，同一规则；
- 其他情况交上层。

选择过程**不看任何学习策略或停止方法的结果**。

## 4. 选定之后

1. **任务冻结**：选定工况写进训练配置，并加测试钉住。
2. **方法冻结**：停止感知 SAC–MPC。另有一处已测出的缺陷待用户决定是否修正：停止头与自身价值不一致，价值规则检验中改用价值比较后毁掉数下降。
3. **训练**：全新种子 3 × 60k，从零开始。
4. **正式评估**：在全新块 **271000–271047** 上跑 Pure、learned-only、stopping 三行。270000 块里有 B 阶段写过的旧文件，不使用。
5. **判据**：与停止头正式轮相同；协调增益必须报告。

---

## 5. 下层执行单

工程 `D:\py\DRL2`，PowerShell，`$py = "D:/py/DRL2/.venv/Scripts/python.exe"`。每个进程单线程：

```powershell
$env:OMP_NUM_THREADS=1; $env:MKL_NUM_THREADS=1; $env:OPENBLAS_NUM_THREADS=1
```

### 5.1 同步与核对

```powershell
Set-Location D:\py\DRL2
git pull --ff-only origin claude/sac-mpc-coupling-design-ns7g6i
git rev-parse HEAD                               # 抄进 REPORT.md；全部进程用这一个提交
git status --porcelain --untracked-files=no      # 必须没有输出
git diff 2e5c236 HEAD --stat -- env controllers dynamics   # 必须没有输出
& $py -B -m pytest -q tests/test_regime_screen.py tests/test_v3_stopping.py
```

### 5.2 第 1 阶段：9 格只跑 Pure

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

- 每格一个进程。CPU 核多的话，可以对同一格再执行一次 `Start-R $g "pure" 2`，认领锁会自动分配开局。
- 进度：

```powershell
Get-ChildItem $R -Recurse -Filter "seed_*.json" | Group-Object DirectoryName | ForEach-Object { "{0}: {1}/48" -f $_.Name, $_.Count }
Select-String -Path eval/regime_logs/*.err.log -Pattern "Traceback|Error"
```

全部 9 格各 48 个文件齐了以后：

```powershell
& $py -B -m experiments.regime_screen readout --root $R --output $R/readout_stage1.json
```

屏幕会打印 `passing`，即通过第 1 阶段的格。

### 5.3 第 2 阶段：只对通过的格跑平缓推进

`passing` 里的每一格执行：

```powershell
Start-R "<格名>" "nominal" 1     # 可同样多开一个进程，编号改为 2
```

跑完后：

```powershell
& $py -B -m experiments.regime_screen readout --root $R --output $R/readout_final.json
```

屏幕打印 `selected`，即选定的工况。

- **`passing` 为空，且所有格的 Pure 都 > 36**：对 `w3.00_r18_v0.15`、`w3.50_r18_v0.15` 做 5.2 的 Pure，通过的再做 5.3，然后同样判读。
- **`passing` 为空的其他情况**：交付，停下。

### 5.4 交付

- 审查分支 `review/regime-screen-<日期>`，放：
  - `REPORT.md`（提交号、5.1 输出、起止时间、判读打印）；
  - 两份 readout；
  - 全部逐开局 JSON；
  - 日志；
  - SHA 清单。
- 交付后等上层写最终训练执行单。不自行开始训练。
