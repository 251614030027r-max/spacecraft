# 执行书：T2_5_1_VALIDATE_20261026（下层本机，约 13–15 h；修复版验证训练，优先执行）

**状态：用户 2026-10-26 批准优先执行；正在运行的 T2.5 验证由用户决定停止。**

**用途：** 验证 T2.5.1 的修复是否把二阶段反事实信息真正用于学习，并体现在配对回报上。这不是论文正式结果。

**科学提交（冻结）：** `34f877e933dc78740674fc82ee0e36f918c541f4`。下层不得改任何代码。

**本轮唯一改动：** 二阶段反事实预测额外作为一条单步 Bellman 样本，写在预测状态上：Q(s̃₁, a₂) ← r̃₂ + γV̄(s̃₂)。
- 这类样本不需要重要性比值，不受标签龄期影响。
- 其余全部沿用 T2.5：真实回放、一阶段锚定、入口结局模型、Retrace 项、Cauchy 权重、β = 0.5、K = 8、同步频率。

**修改原因：** T2.5 验证训练 1000 步后，c ≈ 0.15。沙箱测得，龄期 50 步时 c 只有 0.38，但价值差小于 σ 的 2%。比值把有效信息截掉了。

## 0. 停止 T2.5 并保全

1. 按 PID 结束两个 T2.5 训练进程及其 worker。不要按命令行模式批量结束进程。
2. 原目录 `T2_5_VALIDATE_20261026` 保持不动，不删除。
3. 把以下文件发布到科学分支 `evidence/T2_5_VALIDATE/`，附 SHA256SUMS，作为 T2.5 的机制读数证据：
   - 两个训练目录中已有的 `manifest.json`、`episodes.json`、`model_stats.json`、`heartbeat.json`；
   - 已完成的 `A_LABEL_AUDIT.json` / `A_audit/*.json`（如有）。
4. 不评估、不判定。在 collab 的 `t2_5_validate_20261026/RESULT.md` 中补一行：停止时刻、两组的步数、"用户决定停止，T2.5.1 优先"。

## 1. 准备

```powershell
cd D:\py\DRL2; git fetch origin claude/sac-mpc-coupling-design-ns7g6i; git checkout --detach 34f877e933dc78740674fc82ee0e36f918c541f4
$py  = "D:\py\DRL2\.venv\Scripts\python.exe"
$R   = "C:\Users\35884\Documents\Spacecraft\过程文件\T2_5_1_VALIDATE_20261026"
$R23 = "C:\Users\35884\Documents\Spacecraft\过程文件\T2_3_TEST_20261023"
$R24 = "C:\Users\35884\Documents\Spacecraft\过程文件\T2_4_TEST_20261025"
New-Item -ItemType Directory -Force $R | Out-Null
$env:OMP_NUM_THREADS = "1"; $env:MKL_NUM_THREADS = "1"; $env:OPENBLAS_NUM_THREADS = "1"
& $py -B -m pytest -q tests/test_t2_stage.py *> "$R\pytest.log"          # 须 37 passed
```

## 2. 训练（与 T2.5 验证相同）

```powershell
foreach ($s in 0,1) {
  Start-Process -FilePath $py -WorkingDirectory "D:\py\DRL2" -WindowStyle Hidden `
    -ArgumentList "-B","-m","experiments.t2","train","--arm","e_mve","--seed",$s,"--steps","4000","--workers","3",
                  "--checkpoints","1000,2000,3000","--output","$R\TRAIN_e_mve_s$s" `
    -RedirectStandardOutput "$R\TRAIN_e_mve_s$s.out.log" -RedirectStandardError "$R\TRAIN_e_mve_s$s.err.log"
}
```

`manifest.json` 中应有 `learner_version: t2-learner-2.5.1`。

**1000 步机制早报（只记录，不中止训练）：** 两个训练都过 1000 步后，读 `model_stats.json` 中 1000 步之后的探针，报告以下字段的均值和样本数：
- `two_stage_share`、`decisive_share`；
- `stage2_rows`、`w2_mean`、`pi_gap_sigma_median`、`stage2_entry_share`；
- `c_mean`、`increment_abs_mean`；
- `grad_cos`、`loss_share`、`w_mean`。

若 `w2_mean` < 0.5，或 `pi_gap_sigma_median` > 0.25，或 `grad_cos` ≤ 0，立即在 collab 报告。

**SAC 对照：** 同 T2.5 验证 §2a。从 `T2_5_VALIDATE_20261026\B_eval` 复制；如果那里还没有，按原路径从 T2.3 / T2.4 复制，不重算。

**标签审计：** 本轮不跑。环境和标签生成与 T2.5 相同，T2.5 的审计照常适用。

## 3. 训练结束后：评估、读出、判定

```powershell
foreach ($s in 0,1) { foreach ($k in 0,1000,2000,3000,4000) {
  & $py -B -m experiments.t2 eval --checkpoint "$R\TRAIN_e_mve_s$s\checkpoints\step_$k.zip" --label "e_mve_s${s}_step_$k" `
    --openings 294000-294023 --workers 10 --output "$R\B_eval\e_mve_s$s\step_$k" *>> "$R\eval.log" } }
& $py -B -m experiments.t2_readout curve --eval-root "$R\B_eval" --train-root "$R" --screen "$R23\A2_window" --output "$R\B_READOUT.json" *> "$R\B_readout.log"
& $py -B -m experiments.t2_decide --criteria v251 --readout "$R\B_READOUT.json" --train-root "$R" --output "$R\DECISION.json" *> "$R\decide.log"
```

提示：训练进行中也可以先评估已写出的检查点，例如 1000、2000，以便及早看到配对曲线，但不得据此停训或改动。

## 预先登记的判据（`t2_decide --criteria v251`，事后不得修改）

| 项 | 标准 |
|---|---|
| C0 完整 | EACC 种子 0 有 0–4000，种子 1 有 1000–3000；SAC 如上；全部 24/24，且没有 ERROR 文件 |
| M 机制 | 两个 EACC 训练 1000 步之后的探针均值：two_stage_share ≥ 0.5，decisive_share > 0，w2_mean ≥ 0.5，pi_gap_sigma_median ≤ 0.25，grad_cos > 0；c_mean 只报告 |
| E1 效率 | 配对 EACC − SAC 的检查点平均：种子 0 取 1000–4000 > 0，种子 1 取 1000–3000 > 0 |
| E2 终点 | 种子 0 在 4000 步：配对差 ≥ −3，且 EACC > 53.79（T2.4 耦合臂） |
| S 安全 | 种子 0 在 4000 步：EACC 捕获 ≥ SAC − 1，且 EACC 不安全 ≤ SAC + 1 |

判定结果：
- `mechanism_failure`：M 不满足；
- `harmful`：S 不满足；
- `pass`：M、E1、E2、S 全部满足；
- `mixed`：只有一个种子满足 E1，或 E2 不满足；
- `no_gain`：两个种子都不满足 E1；
- `incomplete`：C0 不满足。

结论以 E1、E2 的配对回报为准；机制读数只说明反馈通道是否在工作。

## 交付（不跑后续分支）

- 科学分支 `evidence/T2_5_1_VALIDATE/`，附 SHA256SUMS：
  - `B_READOUT.json`、`DECISION.json`、`B_eval/**`；
  - 两个训练目录的 `manifest.json`、`finished.json`、`episodes.json`、`model_stats.json`、`heartbeat.json`；
  - `pytest.log`。
- collab `lower/handoffs/t2_5_1_validate_20261026/RESULT.md`：
  - 判定与各项取值；
  - 两个种子的配对曲线，按紧急 / 经济拆分；
  - 机制读数，与 T2.5 停止前的读数并列；
  - 墙钟。
- 只报告事实。

## 下层可以自行处理

- 驱动层问题、单个回合报错补跑、评估的 worker 数，事后写明。
- 训练崩溃：同配置从零重启一次；第二次仍崩溃就停下，交付已有结果。
- 不得改代码、步数、开局、判据，不得挑选检查点。
