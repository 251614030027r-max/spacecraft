# 价值比较交接规则检验：预注册与下层执行单（2026-10-06）

*上层窗口写。用户 2026-10-06 批准。本文件与代码在同一次提交中入库，早于任何 268000 块数据。背景：`docs/STOPPING_FORMAL_REVIEW_20261006.md`。判据事后不改。*

---

## 1. 要回答的问题

正式轮里，停止头和它自己的价值估计不一致：127 次交接中有 59 次发生在 Q_H < Q_C 时，被毁的开局普遍交接偏晚。本检验只回答一件事：

> **三个冻结模型原样不动，只把交接规则换成方法定义本身——第一次 Q_H(s) ≥ Q_C(s, μ(s)) 时交接——能否在全新块上过正式门？**

- 不训练，不改模型、接口、奖励、Pure MPC。
- Q_C(s, μ(s)) 是确定性任务动作下双 critic 的较小值，即 `stopping_values` 已输出、正式轮逐步记录的那个量。
- 熵项（0.005）在比较中忽略。

## 2. 设置

| 项 | 内容 |
|---|---|
| 模型 | `logs/stop_262430`、`stop_262431`、`stop_262432` 的 `final_model.zip`。SHA-256 必须分别为 `a3f36bde…0b3b`、`4b5b53ee…1b82`、`8b56adbf…d74b`（代码中常数 `FROZEN_MODEL_SHA256`），不符即 `FIDELITY_FAIL` |
| 块 | **268000–268047**，全新，未在任何设计或评估中用过 |
| 行 | `pure`（一次）；每个模型的 `value`（被检验的规则）；每个模型的 `stopping`（原停止头，同块对照，只报告） |
| 工具 | `experiments/v3_stopping.py evaluate --row value`，`readout-value` |

## 3. 判据（与正式轮相同）

**每个模型通过**（三条同时满足）：
1. value 行干净完成数 **>** 同块 Pure MPC；
2. value 行真值违规回合数 **≤** Pure；
3. 被毁掉的 Pure 成功开局 **≤ 2**。

**总判定**：
- `FIDELITY_FAIL`：文件不全、工作树不干净、各行提交不一致、行名不对、被截断，或模型 SHA 不是冻结模型；
- `VALUE_RULE_HOLDS`：≥ 2 个模型通过；
- `VALUE_RULE_DOES_NOT_HOLD`：否则。

**同时报告（不作门槛）**：
- 停止头行在同块上的同样计数，用来归因；
- 共同干净完成开局上相对 Pure 的时间和 Δv 中位多用量；
- 交接时刻分布（k=0 / 中途 / 从不）。

## 4. 判定之后（事先写定）

| 判定 | 含义 | 下一步 |
|---|---|---|
| `VALUE_RULE_HOLDS` | 正式轮失败来自停止头的近似，价值本身有用 | 方法改为闭式 β = σ((Q_H − V_C)/α)，部署即价值比较。论文数字须来自该定义下的全新 3×60k 训练和另一个全新块，不直接用本块 |
| `VALUE_RULE_DOES_NOT_HOLD` | 修好停止头也不够，瓶颈在 Q_H 的区分能力和学习策略本身 | 这条线如实收口：交接相对学习策略有显著、安全的增益，但在强 Pure MPC 基线上没有净收益。不再加分类器、安全集或新规则 |
| `FIDELITY_FAIL` | — | 交付 `problems`，停下 |

本检验在正式轮失败之后提出，属于开发验证，不进论文性能表。

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
git rev-parse HEAD                               # 抄进 REPORT.md；全部 7 组用这一个提交
git status --porcelain --untracked-files=no      # 必须没有输出
# 物理、控制器、训练代码相对冻结科学提交没有改动（必须没有输出）：
git diff 2e5c236 HEAD --stat -- env controllers dynamics train
& $py -B -m pytest -q tests/test_v3_stopping.py
foreach ($s in 262430, 262431, 262432) { (Get-FileHash "logs/stop_$s/final_model.zip" -Algorithm SHA256).Hash }
```

最后一行打印的三个哈希必须是（大小写不论）：
- `A3F36BDE8589853EFEFC7ABB24F13F3B068974BADD3E9AD2B104742D272B0B3B`
- `4B5B53EE4903E1AF0D7534898360A3C0DB003EB88B9582D55DBFEFED33721B82`
- `8B56ADBF09D4A9753A040955324F3F75CDA487BFCC77695644F16F6FDCCCD74B`

### 5.2 评估（7 组，块 268000–268047）

```powershell
$V = "eval/stopping/value_check"
New-Item -ItemType Directory -Force $V, eval/stopping_logs | Out-Null
function Start-V($name, $args_) {
  Start-Process -FilePath $py -WorkingDirectory "D:\py\DRL2" -WindowStyle Hidden -PassThru `
    -ArgumentList (@("-u","-B","-m","experiments.v3_stopping","evaluate","--seeds","268000-268047") + $args_) `
    -RedirectStandardOutput "eval/stopping_logs/vc_$name.out.log" -RedirectStandardError "eval/stopping_logs/vc_$name.err.log"
}
foreach ($s in 262430, 262431, 262432) {
  Start-V "value_$s"    @("--run-dir","logs/stop_$s","--row","value","--output-dir","$V/value/$s")
  Start-V "stopping_$s" @("--run-dir","logs/stop_$s","--row","stopping","--output-dir","$V/stopping/$s")
}
```

任一组跑完、空出槽位后，启动 Pure 行（与模型无关，可开 2 个进程，认领锁自动分开局）：

```powershell
Start-V "pure_1" @("--run-dir","logs/stop_262430","--row","pure","--output-dir","$V/pure")
Start-V "pure_2" @("--run-dir","logs/stop_262430","--row","pure","--output-dir","$V/pure")
```

机器槽位多就一开始全部启动；同一组也可以多开进程加速，认领锁会自动分配开局。

查进度只数文件个数，判读前不打开结果：

```powershell
Get-ChildItem $V -Recurse -Filter "seed_*.json" | Group-Object DirectoryName | ForEach-Object { "{0}: {1}/48" -f $_.Name, $_.Count }
Select-String -Path eval/stopping_logs/vc_*.err.log -Pattern "Traceback|Error"
```

进程异常退出时，先确认同组没有进程在跑，再删掉该组目录下残留的 `*.lock` 和 `*.tmp`，用同样的命令续跑。评估期间不拉代码。

### 5.3 判读

```powershell
& $py -B -m experiments.v3_stopping readout-value --pure $V/pure `
  --value 262430=$V/value/262430 --value 262431=$V/value/262431 --value 262432=$V/value/262432 `
  --stopping 262430=$V/stopping/262430 --stopping 262431=$V/stopping/262431 --stopping 262432=$V/stopping/262432 `
  --output $V/readout_value.json
```

屏幕打印 `verdict`。下层不解释，不据此做任何改动。

### 5.4 交付

- 审查分支 `review/stopping-value-check-<日期>`，放：
  - `REPORT.md`（提交号、5.1 全部输出、起止时间、异常与处理、判读打印）；
  - `readout_value.json`；
  - `eval/stopping/value_check/` 下全部逐开局 JSON；
  - 全部 `vc_*` 日志；
  - SHA 清单。
- 模型只报路径和 SHA-256，不打包。
