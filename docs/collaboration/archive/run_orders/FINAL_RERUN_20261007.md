# 执行单：最终主线干净重训与正式评估（合并版）_20261007

状态：已授权待执行。
人类授权来源：用户 2026-10-07 原话："我误操作导致训练中断……如果干净重训……在上次的基础上有没有什么改动的，一并改动了，然后下发执行单给下层推进工作。"

**本单取代** `FINAL_MAINLINE_20261006.md` 中尚未执行的部分（训练、结构检查、正式评估），并**合并** `FINAL_EVIDENCE_20261007.md`（nominal 正式行、272000 第二正式块、确定性重放），后者不再单独执行。

保留不变的部分：
- 269000 工况筛选的结果：选中 `w2.36_r15`，即原主线工况。筛选不重跑。
- `FINAL_MAINLINE_20261006` 第 3 节的正式判据（在 271000 上）。

| 模板项 | 内容 |
|---|---|
| 目的 | 中断后从零干净重训最终方法，并一次跑完论文所需的全部正式证据 |
| 科学代码提交 | `f2f8169acd580ea96a578d45022dd8952447eca5`，分支 `claude/sac-mpc-coupling-design-ns7g6i`。本单全部步骤用这一个提交，含分析工具，中途不拉代码 |
| 训练 | 种子 **262460 / 262461 / 262462**，运行名 `final2_<种子>`，各从零 60,000 外层决策，工况 `w2.36_r15` |
| 样本块 | 结构检查 266000–266047（30k 检查点）；**官方判定 271000–271047**；第二个、也是最后一个正式块 272000–272047；反事实回放取自 271000 |
| 判定 | 官方结论：271000 上的 `readout`，规则同 FINAL_MAINLINE 第 3 节。96 开局配对表、nominal 归因和重放都是描述性的，**不新增门槛** |
| 停止条件 | 5.1 核对不过；30k 结构检查不过（停全部训练）；`FIDELITY_FAIL`；重放 `fidelity_ok` 为 `false`（只停重放这一段） |
| 交付 | `lower/handoffs/final_rerun_<日期>.md`；审查分支 `review/final-rerun-<日期>` |

## 1. 被中断的运行

- `logs/final_262450`、`final_262451`、`final_262452` 标记为 **INTERRUPTED**：原样保留，不删除、不移动、不续训，任何结果都不使用。
- 按规则，训练不能从检查点续跑：检查点不含回放缓冲区和交接标签缓冲区，续跑会改变训练过程。
- 这三个运行的名字已被占用，所以重训改用新种子 262460–262462 和新运行名 `final2_*`，评估目录也改为 `eval/final2/`，与中断前可能留下的文件分开。
- 在交接里写明：中断时间、各运行最后的检查点和 monitor 行数（只读）。

## 2. 相对被中断运行的改动

全部改动列在下面，此外什么都不改。

| # | 改动 | 性质 | 理由 |
|---|---|---|---|
| 1 | 训练目标中，停止权重改用与"继续"项相同的软继续价值（`bellman_stop_value="soft"`）。部署规则不变，仍是第一次 Q_H(s) ≥ Q_C(s, μ(s)) 时交接 | 修一处定义不一致。不是新模块；不改动停止规则、接口、奖励、任务 | 原实现在同一个 Bellman 目标里用了两种继续价值：停止权重用确定性目标网络值，继续项用 SAC 软值。现在目标就是 Q_H 与软继续价值的熵正则最大值，与理论一致。部署取确定性，和 SAC 本身"随机训练、确定性部署"一致。测试已钉住 |
| 2 | 分析工具（`stopping_replay.py`、`final_tables.py`）已在同一提交中 | 流程合并 | 不需要中途拉代码 |
| 3 | nominal 正式行、272000 第二块、确定性重放并入本单 | 合并 FINAL_EVIDENCE | 一次跑完 |
| 4 | 新种子、新运行名、新评估目录 | 记录清晰 | 避免与中断运行混淆 |
| 5 | 运行防护（第 3 节） | 操作 | 防止再次误中断 |

其余与 `e855bb7` 完全一致：价值停止规则、γ = 0.999、行为截断 [0.002, 0.01]、按外层决策计预算、60k、结构检查 D1–D4。

## 3. 运行防护（必读）

- 训练用 `Start-Process` 独立启动，不依赖任何终端窗口。关闭终端、注销远程桌面都不能影响它。
- 训练期间关闭睡眠和休眠：`powercfg /change standby-timeout-ac 0`、`powercfg /change hibernate-timeout-ac 0`。
- **禁止** `Stop-Process -Name python`，以及任何按进程名批量结束的命令。需要停某个进程时，只按 PID 停，并先核对它的命令行。
- 训练期间不运行清理、整理、盘点类脚本；不在 `D:/py/DRL2` 里执行任何 git 写操作。
- 监督脚本只读状态，不重启、不杀进程。

---

## 4. 下层执行单

工程 `D:\py\DRL2`，PowerShell。

```powershell
$py = "D:/py/DRL2/.venv/Scripts/python.exe"
$env:OMP_NUM_THREADS=1; $env:MKL_NUM_THREADS=1; $env:OPENBLAS_NUM_THREADS=1
function Start-X($name, $args_) {
  Start-Process -FilePath $py -WorkingDirectory "D:\py\DRL2" -WindowStyle Hidden -PassThru `
    -ArgumentList (@("-u","-B","-m") + $args_) `
    -RedirectStandardOutput "eval/final2_logs/$name.out.log" -RedirectStandardError "eval/final2_logs/$name.err.log"
}
```

### 4.1 同步与核对（任一不过就停）

```powershell
Set-Location D:\py\DRL2
git pull --ff-only origin claude/sac-mpc-coupling-design-ns7g6i
git rev-parse HEAD
# HEAD 可以比 f2f8169 新，但相对它的代码差异必须为空（必须没有输出）：
git diff f2f8169 HEAD --stat -- env controllers dynamics train experiments tests
git status --porcelain --untracked-files=no                     # 必须没有输出
& $py -B -m pytest -q tests/test_v3_stopping.py tests/test_final_evidence.py tests/test_regime_screen.py tests/test_v3_mainline.py
New-Item -ItemType Directory -Force eval/final2, eval/final2_logs | Out-Null
```

### 4.2 正式训练（3 个进程）

```powershell
foreach ($s in 262460, 262461, 262462) {
  Start-Process -FilePath $py -WorkingDirectory "D:\py\DRL2" -WindowStyle Hidden -PassThru `
    -ArgumentList "-u","-B","-m","train.train_stopping","--steps","60000","--seed","$s","--run-name","final2_$s","--regime","w2.36_r15","--device","cpu" `
    -RedirectStandardOutput "logs/train_final2_$s.out.log" -RedirectStandardError "logs/train_final2_$s.err.log"
}
```

启动后核对，每个运行都应打印 `w2.36_r15 value soft False`：

```powershell
foreach ($s in 262460, 262461, 262462) { & $py -B -c "import json; m=json.load(open('logs/final2_$s/manifest.json')); print($s, m['regime']['name'], m['stopping']['stop_rule'], m['stopping']['bellman_stop_value'], m['code_dirty'])" }
```

### 4.3 与训练并行：与模型无关的四组（优先级低于训练）

```powershell
$E = "eval/final2/formal"
Start-X "nominal_271" @("experiments.regime_screen","evaluate","--regime","w2.36_r15","--row","nominal","--seeds","271000-271047","--output-dir","$E/nominal")
Start-X "nominal_272" @("experiments.regime_screen","evaluate","--regime","w2.36_r15","--row","nominal","--seeds","272000-272047","--output-dir","$E/nominal")
Start-X "pure_271" @("experiments.v3_stopping","evaluate","--run-dir","logs/final2_262460","--allow-incomplete-run","--row","pure","--seeds","271000-271047","--output-dir","$E/pure")
Start-X "pure_272" @("experiments.v3_stopping","evaluate","--run-dir","logs/final2_262460","--allow-incomplete-run","--row","pure","--seeds","272000-272047","--output-dir","$E/pure")
```

CPU 紧张时，这四组排在训练之后再跑，不能拖慢训练。

### 4.4 30k 结构检查（与训练并行）

每个种子的 `logs/final2_<种子>/checkpoints/stopping_30000_outer_decisions.zip` 出现后执行：

```powershell
$D = "eval/final2/devcheck"
foreach ($s in 262460, 262461, 262462) {
  Start-X "dev_$s" @("experiments.v3_stopping","evaluate","--run-dir","logs/final2_$s","--model-name","checkpoints/stopping_30000_outer_decisions.zip","--allow-incomplete-run","--row","stopping","--seeds","266000-266047","--output-dir","$D/$s")
}
# 每组 48 个文件齐了以后：
foreach ($s in 262460, 262461, 262462) { & $py -B -m experiments.v3_stopping devcheck --run-dir logs/final2_$s --eval-dir $D/$s --output $D/devcheck_$s.json }
```

判定同 FINAL_MAINLINE 第 4 节：
- 三个种子 D4 全过，且 D1–D3 至少 2 个种子全过：训练继续；
- 否则：按 PID 停掉三个训练进程，交付，等上层。

### 4.5 训练完成后：模型相关的四组评估

三个运行的 manifest 都是 `completed` 之后：

```powershell
foreach ($s in 262460, 262461, 262462) {
  Start-X "stopping_$s" @("experiments.v3_stopping","evaluate","--run-dir","logs/final2_$s","--row","stopping","--seeds","271000-271047,272000-272047","--output-dir","$E/stopping/$s")
  Start-X "learned_$s"  @("experiments.v3_stopping","evaluate","--run-dir","logs/final2_$s","--row","learned","--seeds","271000-271047,272000-272047","--output-dir","$E/learned/$s")
}
```

核多的话，每组可以多开进程，认领锁会自动分配开局。查进度只数文件个数，判读前不打开结果。

### 4.6 官方判定（271000）

```powershell
& $py -B -m experiments.v3_stopping readout --pure $E/pure `
  --stopping 262460=$E/stopping/262460 --stopping 262461=$E/stopping/262461 --stopping 262462=$E/stopping/262462 `
  --learned 262460=$E/learned/262460 --learned 262461=$E/learned/262461 --learned 262462=$E/learned/262462 `
  --seeds 271000-271047 --output $E/readout_271000.json
```

屏幕打印 `verdict`，即官方结论。

### 4.7 96 开局配对表（描述性）

```powershell
& $py -B -m experiments.final_tables --pure $E/pure --nominal $E/nominal `
  --learned 262460=$E/learned/262460 --learned 262461=$E/learned/262461 --learned 262462=$E/learned/262462 `
  --stopping 262460=$E/stopping/262460 --stopping 262461=$E/stopping/262461 --stopping 262462=$E/stopping/262462 `
  --output $E/final_tables.json
```

`problems` 必须为空。

### 4.8 确定性重放与反事实（描述性）

```powershell
$P = "eval/final2/replay"
foreach ($s in 262460, 262461, 262462) {
  Start-X "replay_271_$s" @("experiments.stopping_replay","replay","--run-dir","logs/final2_$s","--seeds","271000-271047","--formal-dir","$E/stopping/$s","--counterfactual","--output-dir","$P/$s")
  Start-X "replay_272_$s" @("experiments.stopping_replay","replay","--run-dir","logs/final2_$s","--seeds","272000-272047","--formal-dir","$E/stopping/$s","--output-dir","$P/$s")
}
# 全部完成后：
& $py -B -m experiments.stopping_replay summarize --replay 262460=$P/262460 --replay 262461=$P/262461 --replay 262462=$P/262462 --output $P/replay_summary.json
```

每个开局都会和正式结果逐位核对；任一 `fidelity_ok=false` 就停这一段并报告。

### 4.9 交付

- 写交接 `lower/handoffs/final_rerun_<日期>.md`，并更新 `CURRENT.json`。
- 审查分支 `review/final-rerun-<日期>`，放：
  - `REPORT.md`：提交号、4.1 输出、各步起止时间、异常、全部打印、中断运行的登记；
  - 三个运行的 `manifest.json`、`train.monitor.csv`；
  - `devcheck_*.json`；
  - 两个正式块四行的全部逐开局 JSON；
  - `readout_271000.json`、`final_tables.json`；
  - `replay_summary.json` 与 `.trajectories.csv`、重放逐开局 JSON；
  - 日志、SHA 清单。
- 模型只报路径和 SHA-256。

资产盘点（`ASSET_INVENTORY_20261007`）在本单交付之后再执行。
