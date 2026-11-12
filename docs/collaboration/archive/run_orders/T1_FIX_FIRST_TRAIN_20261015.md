# 执行单：T1_FIX_FIRST_TRAIN_20261015（第 2.1 版，唯一有效；下层本机自主一次性执行到交付）

**第 2.1 版修订（2026-10-15，回应 collab `872bc96` 的准入 C 暂停）：** 只改准入 C 的参照，其余条款一字不变。
- 原 C 要求与仓库证据 `evidence/T1_dev_rule_295000` 逐值相同。但该证据由上层 Linux 沙箱计算，下层在 Windows 本机运行，两台机器的浮点运算不同。
- 下层核查结果：事件和时间 16/16 相同；Δv、回报在 1e-8 量级不同，295010 的回报差最大，约 0.0066；没有 deadline 事件。
- 同一核查中，新版与本机已发布的旧版规则（v0.3.2 时代，`lower/handoffs/t1_round1_20261012/rule_local`）16/16 逐值相同。因此 v4 的零动作在本机与旧代码完全等价。
- C 的本意是"同一机器上，新旧代码给出相同物理结果"，这是上层原文的错误，不是放宽判据：逐值要求不变，只把参照换成同机旧结果。
- 下层已完成的测试（71 项）、配置检查和 `rule_k0.7_v0.02`、`rule_k0.7_v0.06` 两套规则（同一提交、同一机器、确定性）可直接复用，不必重算。
- 仓库参照与本机的差异作为跨机记录保留（`admission_exception/RULE_COMPATIBILITY.json`）。
- Pure MPC 参照 `evidence/T1_dev_pure_295000` 同样来自上层沙箱，只用于描述性对照。

**本版替代第 1 版**（collab `7456db8`，科学 `1282464`；未启动）。第 1 版连同其中"10k 止损"一节**全部作废**，不得混用接口 v3/v4、24/25 维模型。若本机已按第 1 版启动过任何运行，在 `ADMISSION.md` 中把它记为"旧接口诊断"，不接续、不混入本单。

**授权（用户 2026-10-15）：**
- 《T1 主线调整执行计划：SAC 接管部分 G★规划设计权限，修复后直接连续实训》；
- 《T1 SAC–G★规划权限重分配：补充修正指令》："技术准入全部通过，即自动冻结科学提交并下发执行单，授权下层连续完成 M0/M2 各 40k 配对训练、开发评估和结果交付，无须再次请示"。上层技术准入的结果见 `evidence/T1_ENTRY_SPEED_PRECHECK`；
- 此前的《可学习性修复与首轮实训》任务书；
- 用户原话："下层拿到自主一次性推进任务直到有实际交付，暂时不再反复询问协作窗口"。

**执行方式：** 从第 1 步连续做到第 7 步，**中途不请示**。只有第 6 节列出的技术情况才停下上报。成绩好坏、曲线下跌、成功率低，都**不是**停机或询问的理由。

**科学提交：** `1d432c8aa0e26e31354b8d43771e2a45c8d1caf5`（分支 `claude/sac-mpc-coupling-design-ns7g6i`）。

**本单训练范围：**
- 原 T1 上 M0/M2 一对、种子 0、各连续 40k 决策；
- 另有一次 2k 决策的 M0 λ 校准。
- **不授权**：多种子、S1、访问 300000–999999、修改奖励/Pure MPC/共享 MPC/任务。

**预计耗时（外推，非实测，以第 3 步实测为准）：**
- 准入约 1.5 h（含 7 套规则 × 16 开局）；
- 校准约 0.5 h；
- 两条同时到 40k 约 12–16 h。上一轮本机为 M1、6 环境，约 0.76 s/决策；两条同时跑时每条约 1.1–1.4 s/决策。

---

## 0. 改了什么（下层只需知道）

详见科学分支 `evidence/T1_ENTRY_SPEED_PRECHECK/IMPLEMENTATION_AND_CHECKS.md`。

| 改动 | 内容 |
|---|---|
| G★ v0.6 入口速度 | 新计划终端约束中的入口径向速度 v_in 改为输入；之前写死为 0.06 m/s |
| 接口 v4 | **3 维动作**：a0 = τ（v2 语义）、a1 = κ、a2 → v_in = 0.06 + 0.04·a2，范围 0.02–0.10 m/s。只有真正新建计划（初次规划、重规划、恢复）时才采用 v_in；保持计划时不受影响，也不会因 a2 而被迫重规划。**25 维观测**：在 v3 的 24 维上追加当前计划的入口速度 (v_in−0.06)/0.04，无计划时为 0。a2 = 0 与旧的 2 维动作逐位相同 |
| M2 反事实 | 在"真实动作会新建计划"的状态，按 (episode+decision) 奇偶交替，提供一个**入口速度候选**：τ、κ 相同，a2 移动 ±1（Δv_in = 0.04 m/s）。有效性按"两计划入口速度相差 ≥ 0.02 m/s"判定。安全、饱和、预测误差门控**不变**。其余状态沿用 τ 候选 |
| Critic 修复 | 沿用 `1282464`：MC 回写、保持带等价、M0/M2 同一个类 `M1SAC`。L_eq 只改变 a0，不改变 a2 |
| λ 校准第 2 版 | λ_eq 固定为 0.1。λ_mc 取 {0.3, 0.1, 0.03} 中"占比 ≤ 0.3"的最大值，下限 0.03，**不会为 0**。合成占比 > 0.5 时下调一档 |
| 训练 | 每条 40k；检查点在 0、2k，之后每 5k。**取消一切按成绩止损** |
| 预填 | 不采用（理由见实现说明） |

## 1. 现场

```powershell
Get-Process python -ErrorAction SilentlyContinue | Format-Table Id, StartTime, Path   # 应无输出；若有，停下上报
cd D:\py\DRL2_T1
git status --porcelain                  # 应无输出
git fetch origin
git checkout --detach 1d432c8aa0e26e31354b8d43771e2a45c8d1caf5
git log -1 --format=%H                  # 必须是 1d432c8aa0e26e31354b8d43771e2a45c8d1caf5
git status --porcelain                  # 必须无输出
git diff --stat 78eb63c 1d432c8aa0e26e31354b8d43771e2a45c8d1caf5 -- requirements.txt requirements-mpc.txt requirements-dev.txt   # 应无输出
$env:OMP_NUM_THREADS="1"; $env:MKL_NUM_THREADS="1"; $env:OPENBLAS_NUM_THREADS="1"
mkdir logs\t1\_v4 -ErrorAction SilentlyContinue
```

- 每个新 PowerShell 窗口都要先 `cd D:\py\DRL2_T1`，再设上面三个线程变量。
- 本单结束前，不得在这个工作树里切换、拉取、修改或提交。

## 2. 准入（A–C 必须通过；D–F 只记录）

```powershell
python -B -m pytest -q tests/test_t1.py tests/test_t1_gym.py tests/test_t1_counterfactual.py tests/test_t1_m1.py tests/test_t1_termination.py tests/test_t1_dev_eval.py tests/test_t1_fixes.py
python -B -m experiments.t1_frozen_config --output logs\t1\_v4\frozen_config_local.json
foreach ($k in "0.7","1.0") { foreach ($v in "0.02","0.06","0.10") {
  python -B -m experiments.t1_dev_eval --policy "rule:earliest_k${k}_v${v}" --openings experiments/t1_dev_openings_295000.json --output-dir "logs/t1/_v4/rule_k${k}_v${v}" --workers 6
} }
```

| 准入项 | 条件 |
|---|---|
| A 测试 | 全部通过 |
| B 冻结配置 | `code_commit` = 1d432c8aa0e26e31354b8d43771e2a45c8d1caf5，`code_dirty` = false；`gstar_version` = "0.6"；`gym_interface_version` = "4"；observation 共 25 项，最后一项为 `plan_entry_speed`；`gym.entry_speed_nominal_m_s` = 0.06，`entry_speed_span_m_s` = 0.04；`sac.updates_per_transition` = 4 |
| C 零动作等价（**同机**） | `rule_k0.7_v0.06` 的 16 个开局，与本机已发布的旧版规则 `docs/collaboration/lower/handoffs/t1_round1_20261012/rule_local/<opening>.json` 比较，事件、时间、Δv、折扣回报**逐值相同**（16/16）。只有 `decision_log` 中 `deadline` ≠ "none" 的开局允许不同：记录并继续。其他任何不同：停下上报。仓库证据 `evidence/T1_dev_rule_295000` 是上层 Linux 沙箱算的，跨机器存在浮点级差异（第 2.1 版修订说明），**只记录、不作判据** |
| D 固定规则网格 | 6 套规则（κ ∈ {0.7, 1.0} × v_in ∈ {0.02, 0.06, 0.10}）的 16 开局结果全部记录 |
| E 强固定参照（预注册选择） | 在 D 的 6 套中按以下顺序比较，冻结第一名为"强固定参照"：平均折扣回报最高；并列时捕获数多；再并列时平均 Δv 低。选择过程与全部 6 行写入 `ADMISSION.md`。这是在开发集上选出的参照，对比时对规则有利，**对本文方法是保守的** |
| F 磁盘 | 剩余空间 ≥ 25 GB |

## 3. λ 校准（一次 2k M0；λ = 0，只测梯度）

```powershell
python -B -m train.train_t1 --method m0 --steps 2000 --seed 0 --n-envs 6 --run-name t1v4_cal_m0_s0 --log-root logs/t1 --checkpoint-freq 1000 --grad-probe-every 10 --audit-every 100
python -B -m experiments.t1_fix_calibration logs/t1/t1v4_cal_m0_s0/fix_stats.json --output logs\t1\_v4\calibration.json
```

- 冻结值取 `calibration.json` 中的 `frozen.lambda_mc`、`frozen.lambda_eq`（下文记为 `LMC`、`LEQ`），原样写入 `ADMISSION.md`。
- 若输出中有 `defect`（梯度非有限或为零），属于代码缺陷：停下上报。
- 校准模型不作为任何训练检查点。
- 记下每决策墙钟（`heartbeat.json`：wall_s / decisions）。

## 4. 配对训练：M0、M2 同时从零开始，各 40k

```powershell
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
$common = "--steps 40000 --seed 0 --n-envs 6 --log-root logs/t1 --checkpoint-freq 5000 --extra-checkpoints 2000 --lambda-mc LMC --lambda-eq LEQ --audit-every 100"
Start-Process python -WindowStyle Minimized -ArgumentList "-B -m train.train_t1 --method m0 --run-name t1v4_m0_s0 $common" -RedirectStandardOutput logs\t1\_v4\m0.out -RedirectStandardError logs\t1\_v4\m0.err
Start-Process python -WindowStyle Minimized -ArgumentList "-B -m train.train_t1 --method m2 --run-name t1v4_m2_s0 $common" -RedirectStandardOutput logs\t1\_v4\m2.out -RedirectStandardError logs\t1\_v4\m2.err
Start-Sleep 30
$marks = "0,2000,5000,10000,15000,20000,25000,30000,35000,40000"
Start-Process python -WindowStyle Minimized -ArgumentList "-B -m experiments.t1_dev_eval --watch logs/t1/t1v4_m0_s0 --openings experiments/t1_dev_openings_295000.json --checkpoints $marks --workers 1" -RedirectStandardOutput logs\t1\_v4\watch_m0.out -RedirectStandardError logs\t1\_v4\watch_m0.err
Start-Process python -WindowStyle Minimized -ArgumentList "-B -m experiments.t1_dev_eval --watch logs/t1/t1v4_m2_s0 --openings experiments/t1_dev_openings_295000.json --checkpoints $marks --workers 1" -RedirectStandardOutput logs\t1\_v4\watch_m2.out -RedirectStandardError logs\t1\_v4\watch_m2.err
```

- `LMC`、`LEQ` 换成第 3 步的冻结数值，两条命令必须相同。
- **不得**加 `--allow-dirty`。
- 进程数为 2 × (1 + 6) + 2 = 16，本机 12 核。
  - 两条同时跑，条件对等。环境步进时各占 6 核，梯度更新时只占主进程 1 核，两条会自然交错。
  - 评估监视进程各 1 个，滞后于训练是正常的。
- 若下层工具会话结束时会连带结束子进程，请用户在自己的 PowerShell 中执行上述命令。下层负责核验进程存活，写 `LAUNCH_RECEIPT.json`（PID、命令、UTC）。
- 更新 collab `CURRENT.json`：
  - topic = `t1_v4_first_train`；
  - scientific_commit = 1d432c8aa0e26e31354b8d43771e2a45c8d1caf5；
  - run_order = 本文件；
  - training_active = true；
  - `lambda_mc` / `lambda_eq` = 冻结值；
  - `strong_fixed_reference` = 第 2 步 E 选出的规则。

## 5. 运行中

- **不按成绩停机。** 20k 只是中途检查点：0k 高、中途下跌、再恢复、后期再跌，都如实记录，训练照常跑到 40k。
- **中途交付（只读快照，不停训练）：** 两条都写完 `dev_eval/20000/summary.json` 后，复制两条运行目录中已写出的日志、评估和 stats 到 `logs\t1\_v4\interim_20k\`，按第 7 节格式打一个中间包，推到 collab。`RESULT_INTERIM.md` 只写事实，注明"中途快照，非稳定结论"。
- **每 5k：** 检查 `heartbeat.json`，`parameters_finite` 应为 true；检查新的 `dev_eval/<n>/summary.json` 是否写出。
- 本单不实现、也不需要续训。训练若中断，见第 6 节；只加载模型 zip 不算无损续训。

## 6. 只有这些技术情况才停下或上报

| 情况 | 处置 |
|---|---|
| 准入 A/B/C 不通过；校准 `defect` | 停，上报 |
| **M2 入口速度反事实长期无效** | 判定时机：M2 写出 10k 检查点后，运行一次（训练中可运行，只读）：`python -B -m experiments.t1_fix_readout --m0 logs/t1/t1v4_m0_s0 --m2 logs/t1/t1v4_m2_s0 --output logs\t1\_v4\cf_check_10k.json`。若 `curve4.m2.cf_by_kind_total.entry_speed.gate_valid` 为 0 或不存在，在 M2 目录写 `STOP_REQUESTED`（内容：`"rule":"T1_FIX r2 sec. 6: entry-speed counterfactual never valid by 10k"`）并上报；M0 照常跑完。这是补充指令规定的技术停机，不是成绩止损 |
| `ABORTED.json`（参数非有限） | 该条为技术停机，保留产物；另一条照常跑完 |
| 训练进程崩溃、断电、被结束 | 保留目录，写 `INTERRUPTED.md`（时间、最后心跳、原因）；**不续训、不重跑**。另一条照常跑完，一并交付并注明 |
| 监视进程崩溃 | 重启同一条 `--watch`（已完成的开局会自动跳过），不上报 |
| 内存不足、系统卡死、磁盘满 | 上报；不得自行减少 n-envs |

## 7. 最终交付（两条都结束，且 40k 的评估写完）

```powershell
foreach ($r in "t1v4_m0_s0","t1v4_m2_s0","t1v4_cal_m0_s0") {
  python -B -m experiments.t1_pretrain_checks stats logs/t1/$r --output logs\t1\$r\stats_final.json
  python -B -m experiments.t1_pretrain_checks reload logs/t1/$r --output logs\t1\$r\reload.json
}
$rules = "k07v06=logs/t1/_v4/rule_k0.7_v0.06,k10v06=logs/t1/_v4/rule_k1.0_v0.06,strong=<第2步E选出的规则目录>"
python -B -m experiments.t1_fix_readout --m0 logs/t1/t1v4_m0_s0 --m2 logs/t1/t1v4_m2_s0 --rules $rules --pure evidence/T1_dev_pure_295000 --output logs\t1\_v4\first_train_readout.json
Get-FileHash -Algorithm SHA256 logs\t1\t1v4_*\checkpoints\*.zip, logs\t1\t1v4_*\final.zip | Export-Csv logs\t1\_v4\MODEL_SHA256.csv -NoTypeInformation
```

**打包** `T1_V4_S0_FINAL_20261015.zip`（不含模型 zip 与 replay）：

| 来源 | 文件 |
|---|---|
| 三个运行目录 | `manifest.json`、`episodes.jsonl`、`decisions.jsonl`、`fix_stats.json`、`cf_stats.json`、`checkpoints/*_fix_stats.json`、`checkpoints/*_cf_stats.json`、`dev_eval/**`、`heartbeat.json`、`finished.json`、`stats_final.json`、`reload.json`，以及 `ABORTED.json`/`INTERRUPTED.md`（若有） |
| `logs/t1/_v4/` | 全部文件：`ADMISSION.md`、`frozen_config_local.json`、`calibration.json`、6 个 `rule_*` 目录、`first_train_readout.json`、`MODEL_SHA256.csv`、`*.out`/`*.err`、`LAUNCH_RECEIPT.json` |
| 包外 | `SHA256SUMS.txt`（逐文件） |

**交付位置：**
- 提交到 collab `docs/collaboration/lower/handoffs/t1_v4_s0_20261015/`。包 > 50 MB 时留在本机，只提交 SHA256 与本机路径。
- 一并提交 `RESULT.md` 与 `SHA256SUMS.txt`。

**`RESULT.md` 只写事实，不写评价，包括：**
- 冻结的 λ 与校准比值；强固定参照及其选择表；
- 两条运行的决策数与结束方式；
- 0–40k 每个检查点的：捕获、超时、unsafe（附违规类型）、平均折扣回报、Δv、时间（失败按 300 s 计）；
- 同一开局上与 0k、强固定参照、Pure MPC 的配对差；
- 入口速度、τ、κ 的请求与实际采用分布：`first_train_readout.json` 中曲线 2 的 `entry_speed_*`；规划更新时刻和重规划触发原因在 `decisions.jsonl`（`t`、`trig`、`v_req`、`v_applied`、`v_plan`）；
- 两类反事实的候选、物理不同、门控有效计数：曲线 4 的 `cf_by_kind_total`；有效样本的预测控制差、计划入口速度差、预测误差：`cf_valid_differences_by_kind`；CF 梯度占比：`cf_grad_share`；
- 每决策 CPU 墙钟；准入结果；偏离本单之处。

**不能只报最好的一个检查点。**

**最后：**
- 更新 `CURRENT.json`：training_active = false，phase = `t1_v4_first_train_delivered`；
- fetch 后提交推送 collab，不 force；
- 上层读包后写 `FIRST_TRAIN_READOUT.md`，区分四种结果：
  ① 超过零点或强参照，且后期能维持；
  ② 有改善，但 M2 不优于 M0；
  ③ Critic 高估或策略坍塌；
  ④ 新动作被忽略。
