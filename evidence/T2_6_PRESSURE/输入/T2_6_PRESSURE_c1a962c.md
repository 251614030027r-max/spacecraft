# 执行书：T2_6_PRESSURE_20261027（下层本机；A 筛查约 2 h，B 训练约 12 h，C 评估约 3 h）

**状态：研究层与用户 2026-10-27 同意按时间压力方向调整任务。A 立即执行；A 按预先登记的规则选出工况后，B、C 直接执行，不再等上层批准。**

**科学提交（冻结）：** `f4d660aa820a2e3b08c8be310aeac7a55d641318`。下层不得改任何代码。

## 本轮唯一的任务因素
- **改动：** 紧急任务的截止时间区间。标准任务是 170–210 s，本轮筛查两档：155–195 s 和 140–180 s。
- **不变：**
  - 各开局的紧急 / 经济分配不变，每个开局的截止时间分位也不变，所以各档可以逐开局配对比较；
  - 经济任务、动力学、初始距离、翻滚范围、执行器、二维动作、奖励、约束不变；
  - EACC 算法（t2-learner-2.5.1）不变。

## 0. 准备

```powershell
cd D:\py\DRL2; git fetch origin claude/sac-mpc-coupling-design-ns7g6i; git checkout --detach f4d660aa820a2e3b08c8be310aeac7a55d641318
$py   = "D:\py\DRL2\.venv\Scripts\python.exe"
$R    = "C:\Users\35884\Documents\Spacecraft\过程文件\T2_6_PRESSURE_20261027"
$SAC0 = "C:\Users\35884\Documents\Spacecraft\过程文件\T2_3_TEST_20261025\TRAIN_e_sac_s0\checkpoints\step_4000.zip"
$MVE0 = "C:\Users\35884\Documents\Spacecraft\过程文件\T2_5_1_VALIDATE_20261026\TRAIN_e_mve_s0\checkpoints\step_4000.zip"
New-Item -ItemType Directory -Force $R | Out-Null
$env:OMP_NUM_THREADS = "1"; $env:MKL_NUM_THREADS = "1"; $env:OPENBLAS_NUM_THREADS = "1"
& $py -B -m pytest -q tests/test_t2_stage.py *> "$R\pytest.log"          # 须 41 passed
```

## A. 无训练筛查（三档，含标准任务，同一提交、同一台机器）

```powershell
$P = @{ "p170" = "170,210"; "p155" = "155,195"; "p140" = "140,180" }
foreach ($t in "p170","p155","p140") {
  & $py -B -m experiments.t2 screen --openings 294000-294023 --policies rally,c_10_0,c_10_-5,c_6_5,c_10_5,gstar,pure `
     --urgent-deadline $P[$t] --workers 11 --output "$R\A\$t\screen" *>> "$R\A_screen.log"
  & $py -B -m experiments.t2 eval --checkpoint $SAC0 --label sac_s0_4000 --openings 294000-294023 --urgent-deadline $P[$t] --workers 11 --output "$R\A\$t\probe" *>> "$R\A_probe.log"
  & $py -B -m experiments.t2 eval --checkpoint $MVE0 --label mve_s0_4000 --openings 294000-294023 --urgent-deadline $P[$t] --workers 11 --output "$R\A\$t\probe" *>> "$R\A_probe.log"
}
& $py -B -m experiments.t2_readout pressure --dir "$R\A" --tags p170,p155,p140 --output "$R\A_PRESSURE.json" *> "$R\A_readout.log"
```

**筛查内容：**
- **固定策略（都在冻结的二维阶段动作空间内）：** rally（截止感知规则）、c_10_0、c_10_-5、c_6_5、c_10_5。
- **参考：** G★（只作内部参考，不作基线）、纯 MPC。
- **学习探针：** 标准任务上训练好的 SAC 种子 0 第 4000 步，以及 EACC（T2.5.1）种子 0 第 4000 步，用来检验"阶段动作空间内能否完成"和"时间压力是否真的起作用"。

**预先登记的选择规则（`t2_readout pressure`，事后不得修改）：**

| 项 | 标准（与标准任务 p170 相比） |
|---|---|
| F1 参考可行 | G★ 捕获 ≥ 22/24，且无 unsafe |
| F2 动作空间可达 | 由阶段动作空间内的策略（固定策略或学习探针）捕获的开局 ≥ 21/24，且紧急开局最多漏 2 个 |
| F3 安全 | 固定策略 unsafe 合计 ≤ 标准任务 + 2；纯 MPC unsafe ≤ 标准任务 + 1；学习探针 unsafe ≤ 标准任务 + 1 |
| D1 时间压力生效 | 两个学习探针的紧急捕获合计比标准任务少 ≥ 2；且去掉贡献最大的单个开局后，决策价值（逐开局最优固定策略 − 统一最优固定策略）仍 > 0 |

- **选择：** 在 F1–F3 与 D1 都满足的档里，取最紧的一档。
- **都不满足：** 不进入 B。交付 A 的结果，在 collab 报告 `tightest_feasible`。

## B. 选定工况上的配对训练（SAC 与 EACC 同时从零开始，2 个种子）

设 `$U` 为选中档的区间，例如 `"155,195"`，`$T` 为选中档的标签，例如 `"p155"`。

```powershell
foreach ($a in "e_sac","e_mve") { foreach ($s in 0,1) {
  Start-Process -FilePath $py -WorkingDirectory "D:\py\DRL2" -WindowStyle Hidden `
    -ArgumentList "-B","-m","experiments.t2","train","--arm",$a,"--seed",$s,"--steps","3000","--workers","2","--urgent-deadline",$U,
                  "--checkpoints","500,1000,1500,2000,2500","--output","$R\B\TRAIN_${a}_s$s" `
    -RedirectStandardOutput "$R\B\TRAIN_${a}_s$s.out.log" -RedirectStandardError "$R\B\TRAIN_${a}_s$s.err.log"
} }
```

- 4 个进程同时启动，每个进程 2 个 worker。
- 每个 `manifest.json` 中应有 `urgent_deadline_s` = 选中档，以及 `learner_version: t2-learner-2.5.1`。
- **1000 步机制早报（只记录，不中止训练）：** 内容同 T2.5.1：`two_stage_share`、`decisive_share`、`stage2_rows`、`w2_mean`、`pi_gap_sigma_median`、`c_mean`、`grad_cos`、`loss_share`。
  - 另外报告两个 EACC 训练中第一个出现模型行（`model_td` 非空）的步数，与 T2.5.1 的 512 / 537 步对照。

## C. 评估、读出、判定

```powershell
foreach ($a in "e_sac","e_mve") { foreach ($s in 0,1) { foreach ($k in 0,500,1000,1500,2000,2500,3000) {
  & $py -B -m experiments.t2 eval --checkpoint "$R\B\TRAIN_${a}_s$s\checkpoints\step_$k.zip" --label "${a}_s${s}_step_$k" --openings 294000-294023 --urgent-deadline $U `
    --workers 11 --output "$R\C_eval\${a}_s$s\step_$k" *>> "$R\C_eval.log" } } }
& $py -B -m experiments.t2_readout curve --eval-root "$R\C_eval" --train-root "$R\B" --screen "$R\A\$T\screen" --output "$R\C_READOUT.json" *> "$R\C_readout.log"
& $py -B -m experiments.t2_decide --criteria t26 --readout "$R\C_READOUT.json" --train-root "$R\B" --output "$R\DECISION.json" *> "$R\decide.log"
```

- 训练会自动写出 `step_0` 和 `step_3000`。
- 训练进行中可以先评估已写出的检查点，但不得据此停训或改动。

**预先登记的判据（`t2_decide --criteria t26`，事后不得修改）：**

| 项 | 标准 |
|---|---|
| C0 完整 | 两臂 × 两种子 × 500–3000 每个检查点都是 24/24，且没有 ERROR 文件 |
| M 机制 | 同 v251：two_stage_share ≥ 0.5，decisive_share > 0，w2_mean ≥ 0.5，pi_gap_sigma_median ≤ 0.25，grad_cos > 0 |
| E1 效率 | 配对 EACC − SAC 在 500–3000 各检查点的平均：两个种子都 > 0，且两种子平均 ≥ +5 |
| U 紧急捕获 | EACC − SAC 的紧急捕获数在各检查点上求和：两种子合计 ≥ +4，且任一种子都不 < 0 |
| E2 终点 | 第 3000 步配对差的两种子平均 ≥ −3 |
| S 安全 | 每个种子：EACC 在各检查点的 unsafe 合计 ≤ SAC + 1；第 3000 步 EACC 捕获 ≥ SAC − 1 |

**判定结果：**
- `incomplete`：C0 不满足；
- `mechanism_failure`：M 不满足；
- `harmful`：E2 或 S 不满足；
- `pass`：全部满足；
- `partial`：E1 在两个种子上都 > 0 但平均不到 +5；或 E1 满足而 U 不满足；或 U 满足而 E1 只有一个种子 > 0；
- `no_gain`：其余情况。

**对照：** 标准任务的结果（T2.5.1 与 T2.3 SAC）照常保留，本轮不重跑。

## 交付

- **科学分支 `evidence/T2_6_PRESSURE/`，附 SHA256SUMS：**
  - `A_PRESSURE.json`，以及 `A/**`（screen、probe）；
  - `C_READOUT.json`、`DECISION.json`、`C_eval/**`；
  - 4 个训练目录的 `manifest.json`、`finished.json`、`episodes.json`、`model_stats.json`、`heartbeat.json`；
  - `pytest.log`。
- **collab `lower/handoffs/t2_6_pressure_20261027/RESULT.md`：**
  - A：三档的 F1–F3 / D1 取值，以及选中的档；
  - B / C：判定与各项取值、配对曲线（紧急 / 经济分开）、紧急捕获曲线、机制读数、首个模型行出现的步数、墙钟。
- 只报告事实。A 完成后先推送一次 A 的结果和选档，再开始 B。

## 下层可以自行处理
- 驱动层问题、单个回合报错补跑、worker 数，事后写明。
- 训练崩溃：同配置从零重启一次；第二次仍崩溃就停下，交付已有结果。
- 不得改代码、截止区间、步数、开局、判据，不得挑选检查点或档位。
