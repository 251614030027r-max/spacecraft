# 执行单：T1_L_OPTION_THREE_ARM_20261015（第 2 版，唯一有效；下层，独立工作树并行支线；自主执行到交付）

**第 2 版修订（2026-10-15，协作窗口审查 `da9be02` 后，用户指示"开始修正并交付"）：** 只改两处，其余一字不变。

1. **NΔ 相对项强度。**
   - 原因：冒烟中 ‖g_Δ‖/‖g_TD‖ ≈ 4×10⁻⁴，固定 λ_Δ = 0.1 时几乎无作用。
   - 现方案：保留 3000 步预热。在预热最后 500 步内，只读测量未加权梯度比，不进入损失，也不改变随机流。到 3000 步一次性冻结 λ_Δ = 0.05 / 中位比值，即目标为 TD 梯度的 5%，并限制在 [1e-3, 1e4] 内。
   - 之后每次更新，相对项梯度不超过该次 TD 梯度的 10%（超出即缩放）；‖g_TD‖ ≤ 1e-8 的更新不加相对项。
   - 只依据梯度，不读回报或成功率，冻结后不再调整。B2 与 N0 不进入这段代码（测试逐位核验）。
2. **资源调度。** 第 4.1 节的规则与 Pure 评估全部结束后（`Wait-Process`），才启动第 4.2 节的训练。v4 吞吐 15% 保护不变。

第 1 版（科学 `da9be02`，collab `887ec26`）作废，未启动。

**方案依据：** 研究层《T1-L：面向持续任务意图的 SAC–G★–MPC 协同学习》（V3，2026-10-15）及其给上层的简令。
**授权：** 用户 2026-10-15 指令："上层核验确认可行后一并交给下层并行推进"。用户把本单转交下层，即视为对冻结 SHA 的确认。

**科学提交（冻结）：** `166244bd4d08713473e17354626f10e40b496e01`
- 分支：`experiment/t1_l_option_relative_20261015`。
- 起点：`1d432c8`，即 v4 所用的提交。
- 上层核验证据：该分支的 `evidence/T1_L_PRECHECK/`。

**本单内容：**
- T1-L 工况三臂：B2、N0、NΔ，种子 0，每臂 40,000 个真实 2 s 步；
- T1-L 开发集上 6 条固定 G★ 规则；
- Pure MPC 描述性对照行。
- **不授权**：多种子；访问正式块 300000–999999；在本支线内改奖励、任务、共享 MPC、网络、超参数或接口。

**最高优先级：在训的 T1 v4（`D:\py\DRL2_T1`，执行单 `T1_FIX_FIRST_TRAIN_20261015` 第 2.1 版）不受任何影响。**
- 不切换、拉取、清理、修改该工作树、其日志、模型和进程。
- v4 的任何健康风险优先于本支线。

---

## 0. 三臂与冻结参数（只需核对，不得更改）

**三臂：**

| 臂 | 审议周期 | MPC 相对 Critic 项 | 归因 |
|---|---|---|---|
| `b2` | 每 2 s（`review_period_steps` = 1） | 关（λ_Δ = 0） | 基准 |
| `n0` | 每 6 s（= 3），计划事件后提前 | 关 | `n0 − b2`：持续审议 |
| `ndelta` | 同 `n0` | 开：3000 步后启用，Huber 10；λ_Δ 按梯度一次校准（目标 5%，每次更新上限 10%，见修订说明） | `ndelta − n0`：MPC 相对反馈 |

**共同参数：**

| 类别 | 参数 |
|---|---|
| 任务 | T1-L：初距 12–18 m，其余与 T1 相同 |
| 规划与控制 | G★ v0.6，共享 MPC h35，RK45 真值 |
| 奖励 | −0.05/s、−5Δv、+100 / −100 / −150，γ = 0.999（每 2 s） |
| 动作（3 维） | 无计划时：τ = τ_min_feasible(s, v_in) + 30(a0+1) s，延迟 0–60 s；有计划时：保持带 / ±30 s 平移；κ 0.3–1.0；v_in = 0.06 + 0.04·a2 |
| 观测 | 34 维（`env.t1_task_review.REVIEW_OBSERVATION_NAMES`） |
| SAC | lr 1e-4、batch 256、buffer 3e5、UTD 4（n_envs = 1 时每步 4 次更新）、τ 0.001、256×3、ReLU |
| 熵 | α 固定，= 0.005 × 本次审议所跨的 2 s 步数（b2 为 0.005；n0/ndelta 正常为 0.015） |
| 共享 critic 修复 | λ_mc = 0.03、λ_eq = 0.1（研究层冻结） |
| 预填 | 无 |
| 并行环境 | n_envs = 1 |
| 开发开局 | `experiments/t1l_dev_openings_297000.json`，16 个，按初始几何分层，未看任何运行结果；SHA-256 见准入 B |

## 1. 独立工作树（单独克隆，不与 DRL2_T1 共享 .git）

```powershell
cd D:\py
git clone https://github.com/251614030027r-max/spacecraft.git DRL2_T1_L
cd D:\py\DRL2_T1_L
git checkout --detach 166244bd4d08713473e17354626f10e40b496e01
git log -1 --format=%H                  # 必须是 166244bd4d08713473e17354626f10e40b496e01
git status --porcelain                  # 必须无输出
git diff --stat 1d432c8 166244bd4d08713473e17354626f10e40b496e01 -- requirements.txt requirements-mpc.txt requirements-dev.txt   # 应无输出：沿用 v4 的 Python 环境，不安装、不升级
$env:OMP_NUM_THREADS="1"; $env:MKL_NUM_THREADS="1"; $env:OPENBLAS_NUM_THREADS="1"
mkdir logs\t1_l_option_20261015 -ErrorAction SilentlyContinue
```

- 每个新窗口都要先 `cd D:\py\DRL2_T1_L`，再设三个线程变量。
- 本支线所有产物只写在 `D:\py\DRL2_T1_L\logs\t1_l_option_20261015\` 下。
- 与 v4 共用同一个 Python 解释器和包，只读使用。

## 2. 准入（A–C 必须通过）

```powershell
python -B -m pytest -q tests/test_t1_l.py tests/test_t1_fixes.py tests/test_t1_m1.py tests/test_t1_counterfactual.py tests/test_t1_gym.py tests/test_t1_termination.py
Get-FileHash experiments\t1l_dev_openings_297000.json -Algorithm SHA256
```

| 项 | 条件 |
|---|---|
| A 测试 | 全部通过 |
| B 开局文件 | SHA-256 = `0fd6c137f8aca1558f349d29713b7bb75d86452adc2db448b62c1efafe9376f3` |
| C 磁盘 | 剩余 ≥ 30 GB |

## 3. 算力与 v4 隔离（实测、记录）

1. **启动前基线：** 读 v4 两个运行目录的 `heartbeat.json`，间隔 30 分钟各读一次，算出 v4 每小时决策数（M0、M2 各一个），写入 `logs\t1_l_option_20261015\ISOLATION.md`。
2. **优先级：** 本支线所有进程以 **Idle 优先级**运行（启动方式见第 4 节）。Windows 只在正常优先级进程让出 CPU 时调度 Idle 进程。v4 结束后，本支线自动获得全部算力。
3. **启动后：** 启动后 1 小时再读一次 v4 的心跳。
   - v4 吞吐比基线下降超过 15%：把本支线全部进程挂起（`Suspend-Process` 或资源监视器"挂起进程"），等 v4 结束后再恢复；同时记录。
   - 挂起再恢复属于同一进程，不是续训，状态完整。
   - 本支线自己的吞吐（每小时步数）同样记录。
4. **内存或磁盘出现风险：** 先保 v4，挂起本支线。
5. 本支线**不宣称**与 v4 互不干扰；`ISOLATION.md` 只写实测吞吐。

## 4. 执行（下列全部以 Idle 优先级启动；可在 v4 运行期间开始）

**辅助函数**（每个窗口先定义一次）：

```powershell
function Start-Idle($argLine, $name) {
  $p = Start-Process python -ArgumentList $argLine -WindowStyle Hidden -PassThru `
       -RedirectStandardOutput "logs\t1_l_option_20261015\$name.out" -RedirectStandardError "logs\t1_l_option_20261015\$name.err"
  $p.PriorityClass = "Idle"; "$name $($p.Id)" | Add-Content logs\t1_l_option_20261015\PIDS.txt
  return $p
}
```

**4.1 固定规则与 Pure（开发集，16 开局）：**

```powershell
$evals = @()
foreach ($k in "0.7","1.0") { foreach ($v in "0.02","0.06","0.10") {
  $evals += Start-Idle "-B -m experiments.t1_l_eval --policy rule:earliest_k${k}_v${v} --output-dir logs/t1_l_option_20261015/rules/rule_k${k}_v${v} --workers 1" "rule_k${k}_v${v}"
} }
$evals += Start-Idle "-B -m experiments.t1_gstar_probe run --row pure --task t1l --seeds 297000,297002,297003,297004,297005,297007,297008,297010,297012,297013,297015,297016,297018,297020,297024,297051 --output-dir logs/t1_l_option_20261015/pure" "pure"
$evals | Wait-Process          # 等全部规则与 Pure 结束，才进入 4.2；检查 7 个 .err 无 Traceback，16 个开局齐全
```

**强固定参照（预注册选择）：**
- 6 条规则写完后，按以下顺序冻结第一名：平均折扣回报最高；并列时捕获数多；再并列时平均 Δv 低。
- 6 行全部写入 `ADMISSION.md`，并注明这是在开发集上选出的参照。
- 规则结果**不影响**三臂是否训练，也不改动任何参数。

**4.2 三臂训练与开发评估（必须在 4.1 全部结束之后）：** 三臂同时启动，条件对等，每臂 1 个进程。

```powershell
foreach ($a in "b2","n0","ndelta") {
  Start-Idle "-B -m train.train_t1_option --arm $a --steps 40000 --seed 0 --run-name ${a}_s0 --log-root logs/t1_l_option_20261015 --n-envs 1" "train_$a"
}
Start-Sleep 60
foreach ($a in "b2","n0","ndelta") {
  Start-Idle "-B -m experiments.t1_l_eval --watch logs/t1_l_option_20261015/${a}_s0 --workers 1" "watch_$a"
}
```

- 检查点在 0、2k，之后每 5k；开发评估在 0、2k、5k、10k、…、40k，每次 16 开局，确定性。
- **不得**加 `--allow-dirty`、`--learning-starts`、`--net-arch`、`--delta-warmup` 或 `--delta-calibration-steps`，这几个参数只用于冒烟。
- `ndelta` 的 λ_Δ 由程序在第 3000 步自动冻结，写入 `ndelta_s0/checkpoints/ndelta_5000_delta_calibration.json`（含探针数、中位比值、λ、冻结时刻）。若到 5k 时 `lambda` 仍为 null（探针不足 30 个），如实记录、不干预：程序会继续探测，直到满 30 个探针才启用相对项。
- 若下层工具会话结束会连带结束子进程，请用户在自己的 PowerShell 中执行上述命令。
- 写 `LAUNCH_RECEIPT.json`（PID、命令、UTC），并更新 collab `CURRENT.json` 的支线字段：
  - `t1_l_branch_topic` = `t1_l_option_three_arm`；
  - `t1_l_scientific_commit` = 166244bd4d08713473e17354626f10e40b496e01；
  - `t1_l_run_order` = 本文件。
  - **不得改动 v4 的字段。**

## 5. 运行中

- **不按成绩停机。** 0k 好坏、曲线下跌、规则更强、NΔ 暂时不如 N0，都不是停机理由。
- **20k 中途只读快照：** 三臂都写完 `dev_eval/20000/summary.json` 后：
  1. 运行第 7 节的 readout；
  2. 复制日志与评估到 `interim_20k\`；
  3. 打包推送 collab，`RESULT_INTERIM.md` 只写事实。
  训练不中断。
- **技术停机（只有这些）：**
  - `ABORTED.json`（参数非有限）；
  - 进程崩溃或断电：写 `INTERRUPTED.md`，**不续训、不重跑**，另两臂照常；
  - 记录的执行动作与实际执行矛盾，或 shadow 污染了真实转移：停该臂，上报；
  - 资源危险：按第 3 节挂起。
- 监视进程崩溃：重启同一条 `--watch` 命令，已完成的开局会自动跳过。

## 6. 本单不包含

- 续训。只加载模型 zip 不算无损续训，中断即如实记录。
- 多种子、正式块评估。
- 任何参数或接口改动。
- 修改 v4。

## 7. 最终交付（三臂都结束，40k 评估写完）

```powershell
foreach ($a in "b2","n0","ndelta") {
  python -B -m experiments.t1_pretrain_checks stats logs/t1_l_option_20261015/${a}_s0 --output logs\t1_l_option_20261015\${a}_s0\stats_final.json
  python -B -m experiments.t1_pretrain_checks reload logs/t1_l_option_20261015/${a}_s0 --output logs\t1_l_option_20261015\${a}_s0\reload.json
}
python -B -m experiments.t1_l_readout --runs b2=logs/t1_l_option_20261015/b2_s0,n0=logs/t1_l_option_20261015/n0_s0,ndelta=logs/t1_l_option_20261015/ndelta_s0 --rules logs/t1_l_option_20261015/rules --strong <第4.1节选出的规则，如 rule:earliest_k1.0_v0.02> --pure logs/t1_l_option_20261015/pure --output logs\t1_l_option_20261015\readout_final.json
Get-FileHash -Algorithm SHA256 logs\t1_l_option_20261015\*_s0\checkpoints\*.zip, logs\t1_l_option_20261015\*_s0\final.zip | Export-Csv logs\t1_l_option_20261015\MODEL_SHA256.csv -NoTypeInformation
```

**打包** `T1_L_THREE_ARM_S0_FINAL.zip`（不含模型 zip）：

| 来源 | 文件 |
|---|---|
| 每臂目录 | `manifest.json`、`episodes.jsonl`、`decisions.jsonl`、`fix_stats.json`、`delta_stats.json`、`checkpoints/*_stats.json`、`dev_eval/**`、`heartbeat.json`、`finished.json`、`stats_final.json`、`reload.json`，以及 `ABORTED.json`/`INTERRUPTED.md`（若有） |
| 支线目录 | `rules/**`、`pure/**`、`ADMISSION.md`、`ISOLATION.md`、`readout_final.json`、`MODEL_SHA256.csv`、`PIDS.txt`、`LAUNCH_RECEIPT.json`、`*.out`/`*.err` |
| 包外 | 逐文件 `SHA256SUMS.txt` |

**交付位置：**
- 推到 collab `docs/collaboration/lower/handoffs/t1_l_option_20261015/`。包 > 50 MB 时留本机，只交 SHA 与路径。
- 模型 zip 留本机。

**`RESULT.md` 只写事实，不写评价：**
- 三臂 0–40k 每个检查点的：捕获、超时、unsafe（附类型）、平均折扣回报、Δv、时间（失败按 300 s）；
- `ndelta` 的 λ_Δ 校准结果，以及实际施加的相对项梯度占比分布、被上限截断的比例（readout 中 `delta`、`delta_calibration` 两项）；
- 与强参照、Pure 的配对差，以及臂间配对差 `n0−b2`、`ndelta−n0`、`ndelta−b2`；
- 审议次数 / 步、首次延迟与锚点、采纳的入口速度与 κ；
- `ndelta` 的反事实分种类计数与相对项梯度占比；
- 每步墙钟、`ISOLATION.md` 的吞吐、任何中断或偏离。

**不能只报最好检查点。**

上层收到后，按 B2 → N0 → NΔ 分层审查，并写结论。
