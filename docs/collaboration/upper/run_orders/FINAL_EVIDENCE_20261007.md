# 执行单：论文闭环最小证据（nominal 正式行 + 第二正式块 + 确定性重放）_20261007

状态：已授权待执行。本单补充 `FINAL_MAINLINE_20261006.md`，不替代它。
人类授权来源：用户 2026-10-07 原话："同意补三项工作……请据此整理成最小执行单给下层……尽快做完。"其中三项是：
1. 现在并行跑 nominal 的 271000–271047 正式行；
2. 预注册 272000–272047 为第二且最后一个正式块，以 271000 + 272000 共 96 个配对开局做统计；
3. 准备确定性重放工具，只作机制分析，不形成新的通过 / 不通过门。

| 模板项 | 内容 |
|---|---|
| 目的与问题 | 补论文闭环所需的最小证据：nominal 对照、96 开局配对统计、交接的状态依赖、价值语义审计、ε_H / ε_C 实测 |
| 科学代码提交 | A、B 两部分：沿用正在跑的冻结提交 `867a3a13e4ad1125a3420bd7ca2bc65d67bb2ba9`（与 `e855bb7` 的代码差异为空），**期间不拉代码**。C 部分：分析工具提交 `ed8db6a542715a7a9cb3a5d7f1485a67747d1857`，相对 `e855bb7` 只新增 `experiments/stopping_replay.py`、`experiments/final_tables.py`、`tests/test_final_evidence.py`；env / controllers / dynamics / train 无差异 |
| 预注册 | 本文件（协作分支）。**272000–272047 是第二个、也是最后一个正式块**。除非导师明确要求，不再增加块 |
| 输入模型 | `logs/final_262450`、`final_262451`、`final_262452` 的 `final_model.zip`；只用最终模型 |
| 样本块 | 正式：271000–271047、272000–272047。重放：两块的全部 stopping 开局；反事实：271000 块中每个模型前 16 个中途交接开局（交接决策 k\* 及前一个决策 k\*−1） |
| 判定 | 官方结论仍是 `FINAL_MAINLINE_20261006` 在 271000 上的 `readout`，不变。本单新增内容全部是描述性的：96 开局表格、nominal 归因、机制分析。**不新增任何通过 / 不通过门** |
| 停止条件 | 重放的 `fidelity_ok` 出现 `false`：只停 C 部分并报告，不影响 A、B；任何报错导致结果不全 |
| 异常处理 | 评估可以删同组残留的 `*.lock` / `*.tmp` 后续跑；不覆盖已有结果 |
| 交付 | 写入 `lower/handoffs/final_evidence_<日期>.md`；审查分支放 `REPORT.md`、全部逐开局 JSON、`final_tables.json`、`replay_summary.json` 及其 `.trajectories.csv`、日志和 SHA 清单 |

---

## A. 现在执行（与训练并行，冻结提交，不拉代码）

这三组都与模型无关。CPU 紧张时排在训练后面，**不能拖慢训练**。

```powershell
Set-Location D:\py\DRL2
$env:OMP_NUM_THREADS=1; $env:MKL_NUM_THREADS=1; $env:OPENBLAS_NUM_THREADS=1
git rev-parse HEAD        # 必须是 867a3a1…（正在跑的冻结提交）；抄进 REPORT.md
$py = "D:/py/DRL2/.venv/Scripts/python.exe"
$E = "eval/final/formal"
function Start-X($name, $args_) {
  Start-Process -FilePath $py -WorkingDirectory "D:\py\DRL2" -WindowStyle Hidden -PassThru `
    -ArgumentList (@("-u","-B","-m") + $args_) `
    -RedirectStandardOutput "eval/regime_logs/x_$name.out.log" -RedirectStandardError "eval/regime_logs/x_$name.err.log"
}
# nominal：271000 与 272000 两块
Start-X "nominal_271" @("experiments.regime_screen","evaluate","--regime","w2.36_r15","--row","nominal","--seeds","271000-271047","--output-dir","$E/nominal")
Start-X "nominal_272" @("experiments.regime_screen","evaluate","--regime","w2.36_r15","--row","nominal","--seeds","272000-272047","--output-dir","$E/nominal")
# Pure：272000 块（271000 的 Pure 已在 FINAL_MAINLINE 5.6 中跑）
Start-X "pure_272" @("experiments.v3_stopping","evaluate","--run-dir","logs/final_262450","--allow-incomplete-run","--row","pure","--seeds","272000-272047","--output-dir","$E/pure")
```

说明：
- 271000 和 272000 的 nominal 写进同一个目录 `$E/nominal`，逐开局一个文件，不会冲突；Pure 同理，写进 FINAL_MAINLINE 已在使用的 `$E/pure`。
- `--allow-incomplete-run` 只是让 Pure 行读取正在训练的运行目录里的工况配置，Pure 不使用模型。
- 查进度只数文件个数；判读之前不打开结果。

## B. 训练完成、FINAL_MAINLINE 5.6 正式评估启动时一起执行（冻结提交）

```powershell
foreach ($s in 262450, 262451, 262452) {
  Start-X "stopping_272_$s" @("experiments.v3_stopping","evaluate","--run-dir","logs/final_$s","--row","stopping","--seeds","272000-272047","--output-dir","$E/stopping/$s")
  Start-X "learned_272_$s"  @("experiments.v3_stopping","evaluate","--run-dir","logs/final_$s","--row","learned","--seeds","272000-272047","--output-dir","$E/learned/$s")
}
```

- 输出目录和 271000 相同，逐开局文件按种子号区分，可以和 5.6 的进程同时跑。
- **271000 的官方 `readout` 照 FINAL_MAINLINE 5.7 原样执行**：它带 `--seeds 271000-271047`，只读 271000 的文件，同目录里的 272000 文件不影响它。

## C. 全部评估完成后（拉取分析工具提交）

```powershell
git pull --ff-only origin claude/sac-mpc-coupling-design-ns7g6i
git rev-parse HEAD     # 应为 ed8db6a… 或更新；抄进 REPORT.md
git diff e855bb7 HEAD --stat -- env controllers dynamics train   # 必须没有输出
& $py -B -m pytest -q tests/test_final_evidence.py tests/test_v3_stopping.py
```

### C1. 96 开局配对表（几秒）

```powershell
& $py -B -m experiments.final_tables --pure $E/pure --nominal $E/nominal `
  --learned 262450=$E/learned/262450 --learned 262451=$E/learned/262451 --learned 262452=$E/learned/262452 `
  --stopping 262450=$E/stopping/262450 --stopping 262451=$E/stopping/262451 --stopping 262452=$E/stopping/262452 `
  --output $E/final_tables.json
```

- `problems` 必须为空：文件齐全，四行都是同一冻结提交，learned 与 stopping 用的是同一个模型文件。
- 输出内容：
  - 各行干净完成数；
  - stopping 相对 Pure、nominal、learned 的配对差别：只有一方完成的开局、精确 McNemar 检验；
  - 共同完成开局上的时间和 Δv 差；
  - 交接分布。

### C2. 确定性重放（3 个模型 × 2 块，每组一个进程）

```powershell
$P = "eval/final/replay"
foreach ($s in 262450, 262451, 262452) {
  Start-X "replay_271_$s" @("experiments.stopping_replay","replay","--run-dir","logs/final_$s","--seeds","271000-271047","--formal-dir","$E/stopping/$s","--counterfactual","--output-dir","$P/$s")
  Start-X "replay_272_$s" @("experiments.stopping_replay","replay","--run-dir","logs/final_$s","--seeds","272000-272047","--formal-dir","$E/stopping/$s","--output-dir","$P/$s")
}
```

- 每个开局都会与正式结果逐位核对，打印 `fidelity_ok`。
- 271000 块带反事实回放：每个模型取前 16 个中途交接开局，在交接决策 k\* 和前一个决策 k\*−1 两个状态上各回放两条路径。
  - A：立即交给 Pure MPC；
  - B：先按确定性动作继续一步，再按最终停止规则运行。
- 反事实比单纯重放慢，约多出 64 条后段回放 / 模型。

完成后汇总：

```powershell
& $py -B -m experiments.stopping_replay summarize --replay 262450=$P/262450 --replay 262451=$P/262451 --replay 262452=$P/262452 --output $P/replay_summary.json
```

输出 `replay_summary.json` 与 `replay_summary.trajectories.csv`，内容：
- 交接时的距离、入口轴偏角、任务进展、承诺度、剩余时间的分布；
- 每个决策的 Q_H、V_C（四种定义）；
- 价值语义审计：训练目标中的软继续价值、目标网络确定性价值、部署用的 V_C，相互之间的数值差，以及 Q_H − V_C 是否因此变号；
- 反事实给出的 ε_H、ε_C 样本，以及估计优势 Â 与真实优势 A 的符号是否一致；
- 重放保真度。

**这些全部是论文机制分析材料，不据此做任何通过 / 不通过判断。**

## D. 交付

- 交接写入 `lower/handoffs/final_evidence_<日期>.md`，并更新 `CURRENT.json`。
- 审查分支 `review/final-evidence-<日期>`，放：
  - `REPORT.md`：各提交号、各步起止时间、异常、所有打印；
  - 271000 / 272000 全部逐开局 JSON，四行；
  - `final_tables.json`；
  - `replay_summary.json` 及 `trajectories.csv`；
  - 重放逐开局 JSON；
  - 日志、SHA 清单。
- 可以和 FINAL_MAINLINE 的正式交付合成一个包，但两个 `readout` / 汇总文件要分开放。
