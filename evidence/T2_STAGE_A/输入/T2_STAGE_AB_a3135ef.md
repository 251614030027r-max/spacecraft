# 执行单：T2_STAGE_AB_20261021（下层本机；阶段 A 固定策略筛选 → 过门后自动进入阶段 B 限额学习验证）

**科学提交（冻结）：** `31074206045cf62dba3d560e4442db4b00eea05a`。
- **不得修改：** 任务、奖励、期限、动作映射、G★ / 跟踪律 / MPC、门控阈值、开局、步数。
- **研究层意见（2026-10-21）：** 阶段 A 为 GO；阶段 B 须在模型标签一致性修复且 A 通过之后进行。
  - 本提交已修复三项一致性问题：候选姿态按阶段起点对齐 16 s；事实检查与候选共用同一份冻结校准器；SMDP 折扣与塑形一致。三项都有单元测试。
  - 0k 门槛已取消。
- **设计说明：** 科学分支 `evidence/T2_PRECHECK/README.md`，以及用户转交的 T2 设计文档。
- **正式多种子长训：** 仍须研究层另行批准，本执行单不授权。

## 0. 准备（约 5 min）

```powershell
cd D:\py\DRL2; git fetch origin claude/sac-mpc-coupling-design-ns7g6i; git checkout --detach 31074206045cf62dba3d560e4442db4b00eea05a
$py = "D:\py\DRL2\.venv\Scripts\python.exe"
$R  = "C:\Users\35884\Documents\Spacecraft\过程文件\T2_AB_20261021"
New-Item -ItemType Directory -Force $R | Out-Null
$env:OMP_NUM_THREADS = "1"; $env:MKL_NUM_THREADS = "1"; $env:OPENBLAS_NUM_THREADS = "1"
& $py -B -m pytest -q tests/test_t2_stage.py *> "$R\pytest.log"          # 须 9 passed
(Get-CimInstance Win32_Processor | Measure-Object NumberOfLogicalProcessors -Sum).Sum | Out-File "$R\logical_cores.txt"
```

pytest 未全部通过时停下，回报日志，不得继续。

## 1. 阶段 A：固定策略筛选（不训练；约 60–90 min）

**开局：** 开发块 `294000–294023`（24 个，T2 首次使用，预先登记，不得更换）。

**策略（9 个）：**
- 恒定动作 7 个：`c_-5_0`、`c_0_0`（中性，即未训练策略的确定性动作）、`c_5_0`、`c_10_0`、`c_0_5`、`c_5_5`、`c_10_10`；
- `rally`：截止期感知规则，是 B 段要比较的强规则；
- `gstar`：旧 G★ 从远场直接桥接，只作动作空间外的参照。

```powershell
& $py -B -m experiments.t2 screen --openings 294000-294023 --policies "c_-5_0,c_0_0,c_5_0,c_10_0,c_0_5,c_5_5,c_10_10,rally,gstar" `
    --workers 10 --output "$R\A_screen" *> "$R\A_screen.log"
& $py -B -m experiments.t2_readout screen --dir "$R\A_screen" --output "$R\A_READOUT.json" *> "$R\A_readout.log"
```

`--workers` 取逻辑核数减 2。中途中断就以同样命令重跑：已有的 JSON 会被跳过，结果仍是确定性的。

**判定**（`A_READOUT.json` 里的 `gates` 与 `passed`，预先登记）：

| 门 | 条件 |
|---|---|
| A1 机械 | 无 `ERROR_*.json`；保持 + 拒绝的阶段数 ≤ 全部阶段数的 10 % |
| A2 状态相关 | 逐开局最佳固定策略减最佳恒定动作：平均 ≥ 10，且去掉最大的两个开局增益后仍 ≥ 5 |
| A3 分散 | 至少 3 种策略各自在 ≥ 15 % 的开局上最优 |

- **只报告、不设门：** 相对 `rally` 规则的余量（平均值、去掉前两大后的值），以及被其他策略以 ≥ 2 超过的开局数。研究层要看这一项，是为了判断收益是否集中在个别救援开局。
- **`passed = true`** → 立即进入第 2 节。
- **`passed = false`** → 停止，不跑 B，交付 A 的结果。

## 2. 阶段 B：两臂限额学习验证（A 通过后自动进行；训练约 2.5–3 h，评估约 1 h）

**设置：** 两臂同时启动，各 5 个环境，每臂 3000 次阶段决策（SB3 的 num_timesteps 按所有环境累计），种子 0，从零训练。检查点为 0 / 1000 / 2000 / 3000。

```powershell
foreach ($a in "e_sac","e_mve") {
  Start-Process -FilePath $py -WorkingDirectory "D:\py\DRL2" -WindowStyle Hidden `
    -ArgumentList "-B","-m","experiments.t2","train","--arm",$a,"--seed","0","--steps","3000","--envs","5","--checkpoints","1000,2000","--output","$R\TRAIN_$a" `
    -RedirectStandardOutput "$R\train_$a.out.log" -RedirectStandardError "$R\train_$a.err.log"
}
# 两臂都写出 finished.json 后再评估（检查点：step_0、step_1000、step_2000、final = 3000）
foreach ($a in "e_sac","e_mve") {
  foreach ($k in @(@("step_0","$R\TRAIN_$a\checkpoints\step_0.zip"), @("step_1000","$R\TRAIN_$a\checkpoints\step_1000.zip"),
                   @("step_2000","$R\TRAIN_$a\checkpoints\step_2000.zip"), @("step_3000","$R\TRAIN_$a\final.zip"))) {
    & $py -B -m experiments.t2 eval --checkpoint $k[1] --label "${a}_$($k[0])" --openings 294000-294023 --workers 10 --output "$R\B_eval\$a\$($k[0])" *>> "$R\B_eval.log"
  }
}
& $py -B -m experiments.t2_readout curve --eval-root "$R\B_eval" --train-root "$R" --screen "$R\A_screen" --output "$R\B_READOUT.json" *> "$R\B_readout.log"
```

**B 段交付的内容**（只报告事实，研究层据此判断是否批准正式训练）：
- **学习曲线：** 每臂在 0 / 1000 / 2000 / 3000 四个检查点上的平均回报、捕获、unsafe、成功回合的 Δv 与用时；同一批 24 个开局，确定性评估。
- **对照：** A 段的 `rally` 规则、中性动作 `c_0_0`、逐开局最佳固定策略。
- **耦合：** E-MVE 对 E-SAC 的逐检查点差值；训练中模型项的使用情况，包括探针数、有模型样本的探针数、λ_eff 均值、带候选的转移比例（`finished.json` 的 `replay`）。
- **上层的预期读法（不作停止条件）：** 3000 步时任一臂比 0k 回报高 ≥ 20、捕获不降，即认为学习曲线成立；E-MVE 若在 2000 或 3000 步比 E-SAC 高 ≥ 5，即认为耦合有初步贡献。

## 3. 下层可以自行处理的情况

- **允许自行修复，事后在报告中写明：**
  - Windows 路径、编码、多进程启动等驱动层问题；
  - 单个回合报错：程序会记为 `ERROR_*.json` 并继续。确属驱动或读数脚本缺陷才修复，重跑只限那几个回合，结果不能自己编；
  - 中断后用同一命令补跑评估（已有文件会跳过）；
  - `--workers` 与 `--envs` 的并行度（对结果没有影响；`--envs` 两臂必须相同）。
- **训练进程崩溃：** 同一种子从零重启一次，不续训、不改步数。第二次仍崩溃，就停下回报。
- **不得做的事：**
  - 改 `env/`、`train/` 里的参数或逻辑；
  - 改门槛、开局、步数；
  - 删掉失败回合；
  - 挑选检查点。

## 4. 交付

- **科学分支** `evidence/T2_STAGE_A/`、`evidence/T2_STAGE_B/`（逐个文件 `git add -f`，附 SHA256SUMS）：
  - A 段：`A_READOUT.json`、`A_screen/*.json`；
  - B 段：`B_READOUT.json`、`B_eval/**/*.json`、两臂的 `manifest.json` / `finished.json` / `episodes.json` / `model_stats.json`；
  - 另附 `pytest.log` 与 `logical_cores.txt`。
  - 模型 zip 不入 git，在本机按实验 ID 归档。
- **collab** `docs/collaboration/lower/handoffs/t2_stage_ab_20261021/RESULT.md`：
  - A 的三道门，逐开局 9 种策略的结果表；
  - B 的学习曲线表；
  - 各阶段实际墙钟，以及自行处理过的问题清单。
