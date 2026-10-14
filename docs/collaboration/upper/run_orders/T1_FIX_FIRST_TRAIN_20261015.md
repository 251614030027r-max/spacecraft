# 执行单：T1_FIX_FIRST_TRAIN_20261015（下层本机，自主一次性执行到交付）

**授权（用户 2026-10-15）：**《T1_SAC–MPC 可学习性修复与首轮实训——上层执行任务书》。用户原话："下层拿到自主一次性推进任务直到有实际交付，暂时不再反复询问协作窗口，拿到有效结果再讨论。"

**执行方式：** 下层按本单从第 1 步连续执行到第 7 步交付，中间**不向协作窗口请示**。只有第 6 节列出的情况才停下上报。

**科学提交：** `128246427322700c24b7fa7bd8a5eec5d9928d53`（分支 `claude/sac-mpc-coupling-design-ns7g6i`）。`78eb63c` 冻结为旧参照，本单不使用。

**预计耗时（外推，非实测）：**
- 上一轮在本机：M1、6 环境，32,088 决策用 6.8 h，约 0.76 s/决策。
- 本单两条同时跑会超订，预计每条 1.0–1.3 s/决策。
- 准入约 1 h（含两套 16 开局规则评估）；2k 校准约 0.5 h；到 10k 约 3–4 h；到 20k 约 6–8 h。

**本单训练内容：** 原 T1 上 M0/M2 一对、种子 0、每条上限 20k 决策；另有一次 2k 决策的 M0 权重校准。
- 不授权多种子；不授权 S1 训练；不访问 300000–999999。
- 不改奖励、Pure MPC、共享 MPC 或任务。

---

## 0. 改了什么（下层只需知道）

详见科学分支 `evidence/T1_FIX_PRECHECK/IMPLEMENTATION_DIFF_AND_CHECKS.md`。

| 改动 | 内容 |
|---|---|
| G★ v0.5 期限一致性 | 规划到达不晚于 300 − 60 = 240 s。晚于期限的请求截到期限（`clamped`）；仍在执行但到达晚于期限的计划要重规划（`plan_beyond`）；期限内不可行时取最早可行，并报告（`infeasible_within`）。规则（最早＋保持）的搜索与 v0.4 相同 |
| 接口 v3 | 动作语义与 v2 相同。观测追加 2 维，相对当前计划：位置误差/1 m、速度误差/0.3 m/s；无计划时为 0。共 24 维 |
| 动作诊断 | 每个决策的 `execution` 中新增四项：`request_kind`（keep/later/earlier/no_plan）、`new_plan_trigger`（deviation/deadline/request/no_plan）、`deadline`、`deviation_before_m`。`decisions.jsonl` 与开发评估都会记录 |
| Critic 修复（M0/M2 共用） | 同一个类 `M1SAC`：L_Q = L_TD + λ_mc·L_mc + λ_eq·L_eq + [M2] λ_cf·L_cf。λ_cf = 0.5，预热 3000，与原 M1 相同。M0 关闭 CF，用无反事实的环境 |
| 预填 | **本轮不采用**。理由：任务书允许跳过；预填会给 MC 回写和 M0/M2 字节一致性增加一条新路径；首轮先验证 critic 修复本身 |

---

## 1. 现场

```powershell
Get-Process python -ErrorAction SilentlyContinue | Format-Table Id, StartTime, Path   # 应无输出；若有，停下上报
cd D:\py\DRL2_T1
git status --porcelain                  # 应无输出
git fetch origin
git checkout --detach 128246427322700c24b7fa7bd8a5eec5d9928d53
git log -1 --format=%H                  # 必须是 128246427322700c24b7fa7bd8a5eec5d9928d53
git status --porcelain                  # 必须无输出
git diff --stat 78eb63c 128246427322700c24b7fa7bd8a5eec5d9928d53 -- requirements.txt requirements-mpc.txt requirements-dev.txt   # 应无输出
$env:OMP_NUM_THREADS="1"; $env:MKL_NUM_THREADS="1"; $env:OPENBLAS_NUM_THREADS="1"
mkdir logs\t1\_fix -ErrorAction SilentlyContinue
```

- 每个新 PowerShell 窗口都要先 `cd D:\py\DRL2_T1`，再设三个线程变量。
- 本单结束前，不得在该工作树切换、拉取、修改或提交。
- 旧目录 `logs/t1/t1v2_m1_s0` 等原样保留。

## 2. 准入（全部通过才继续）

```powershell
python -B -m pytest -q tests/test_t1.py tests/test_t1_gym.py tests/test_t1_counterfactual.py tests/test_t1_m1.py tests/test_t1_termination.py tests/test_t1_dev_eval.py tests/test_t1_fixes.py
python -B -m experiments.t1_frozen_config --output logs\t1\_fix\frozen_config_local.json
python -B -m experiments.t1_dev_eval --policy rule:earliest --openings experiments/t1_dev_openings_295000.json --output-dir logs/t1/_fix/rule_k07 --workers 4
python -B -m experiments.t1_dev_eval --policy rule:earliest_k10 --openings experiments/t1_dev_openings_295000.json --output-dir logs/t1/_fix/rule_k10 --workers 4
```

| 准入项 | 通过条件 |
|---|---|
| A 测试 | 全部通过 |
| B 冻结配置 | `code_commit` = 128246427322700c24b7fa7bd8a5eec5d9928d53，`code_dirty` = false；`gstar_version` = "0.5"；`gstar_deadline_margin_s` = 60；`gym_interface_version` = "3"；observation 共 24 项，最后两项为 `track_position_error_1m`、`track_velocity_error_0p3m_s`；`sac.updates_per_transition` = 4 |
| C 规则 κ0.7 | 应逐值等于 `evidence/T1_dev_rule_295000`（16/16，81.08945807364475）。若有开局不同，看该开局 `decision_log` 的 `deadline` 字段：不同只出现在 `deadline` ≠ "none" 的开局时，**记录并继续**；否则停下上报 |
| D 规则 κ1.0 | 只记录结果，作为强固定规则参照，无通过门槛 |
| E 磁盘 | 剩余 ≥ 15 GB |

把四项的结果和用时写入 `logs\t1\_fix\ADMISSION.md`。

## 3. 权重校准：一次 2k M0（λ = 0，只测梯度）

```powershell
python -B -m train.train_t1 --method m0 --steps 2000 --seed 0 --n-envs 6 --run-name t1fix_cal_m0_s0 --log-root logs/t1 --checkpoint-freq 1000 --grad-probe-every 10 --audit-every 100
python -B -m experiments.t1_fix_calibration logs/t1/t1fix_cal_m0_s0/fix_stats.json --output logs\t1\_fix\calibration.json
```

- 这次运行中 λ_mc = λ_eq = 0：辅助项只测梯度，不更新参数（单测已证明逐位等于纯 TD）。
- `learning_starts` = 1000，统计窗口为决策 1000–2000。
- 选择规则由脚本机械执行（任务书 D 节），不看任何成绩：
  - 在 {0.03, 0.1, 0.3} 中取使 λ·中位比值落在 [0.1, 0.3] 的最大值；
  - 太强时取更小的安全值 0.2/中位；
  - 太弱、样本太少或梯度异常时取 0；
  - 合并占比超过 0.4 时同比下调。
- **冻结值：** `calibration.json` 的 `frozen.lambda_mc` 与 `frozen.lambda_eq`，下文记为 `LMC`、`LEQ`。原样写入 `ADMISSION.md`，之后不改。
- 校准模型不作为任何训练检查点。
- 记下用时，换算每决策墙钟：`heartbeat.json` 中 wall_s / decisions。

**两项都为 0 是合法结果**，照常继续：此时 M0 就是 TD + 期限修复 + 新观测。

## 4. 配对训练（同时启动；独立最小化窗口；防睡眠）

```powershell
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
$common = "--steps 20000 --seed 0 --n-envs 6 --log-root logs/t1 --checkpoint-freq 1000 --lambda-mc LMC --lambda-eq LEQ --audit-every 100"
Start-Process python -WindowStyle Minimized -ArgumentList "-B -m train.train_t1 --method m0 --run-name t1fix_m0_s0 $common" -RedirectStandardOutput logs\t1\_fix\m0.out -RedirectStandardError logs\t1\_fix\m0.err
Start-Process python -WindowStyle Minimized -ArgumentList "-B -m train.train_t1 --method m2 --run-name t1fix_m2_s0 $common" -RedirectStandardOutput logs\t1\_fix\m2.out -RedirectStandardError logs\t1\_fix\m2.err
Start-Sleep 30
Start-Process python -WindowStyle Minimized -ArgumentList "-B -m experiments.t1_dev_eval --watch logs/t1/t1fix_m0_s0 --openings experiments/t1_dev_openings_295000.json --checkpoints 0,2000,5000,10000,15000,20000 --workers 2" -RedirectStandardOutput logs\t1\_fix\watch_m0.out -RedirectStandardError logs\t1\_fix\watch_m0.err
Start-Process python -WindowStyle Minimized -ArgumentList "-B -m experiments.t1_dev_eval --watch logs/t1/t1fix_m2_s0 --openings experiments/t1_dev_openings_295000.json --checkpoints 0,2000,5000,10000,15000,20000 --workers 2" -RedirectStandardOutput logs\t1\_fix\watch_m2.out -RedirectStandardError logs\t1\_fix\watch_m2.err
```

- 把 `LMC`、`LEQ` 换成第 3 步冻结的数值，两条命令必须相同。
- 训练脚本在有改动的检出上会拒绝运行。**不得**加 `--allow-dirty`。
- 进程数：2 × (1 + 6) + 2 × 2 = 18，本机 12 核，会超订。两条同时跑，条件对等；超订只会变慢，不影响结果，因为环境与评估都是确定的。
- 开发评估覆盖全部 16 个开局（任务书允许 2k/5k 只评前 8 个，这里统一评 16 个，便于比较）。
- 若下层工具会话结束时会连带结束子进程，就请用户在自己的 PowerShell 中执行上面的命令。下层负责核验进程存活，并写 `LAUNCH_RECEIPT.json`（PID、命令、UTC）。
- 更新 collab `CURRENT.json`：
  - `topic` = `t1_fix_first_train`；
  - `scientific_commit` = 128246427322700c24b7fa7bd8a5eec5d9928d53；
  - `run_order` = 本文件；
  - `training_active` = true；
  - `lambda_mc` / `lambda_eq` = 冻结值。

## 5. 预注册的停止规则（下层机械执行，不看其他成绩）

10k 两条的开发评估都写出后，对**每条运行分别**判定：

| 判定 | 条件（相对同一运行自己的 0k） | 处置 |
|---|---|---|
| **旧失败复现** | 10k 捕获数 ≤ 0k 捕获数 − 4，且 10k 失败中超时占多数；或 10k 平均折扣回报 < 0k − 25 | 在该运行目录写 `STOP_REQUESTED`，内容：`{"rule":"T1_FIX section 5: old failure recurred at 10k", ...两点的捕获、超时、回报}`。另一条运行不受影响 |
| **机器太慢** | 按 0→10k 的实际速度推算，到 20k 还需 > 14 h | 两条都在 10k 后写 `STOP_REQUESTED`，内容注明 `"rule":"T1_FIX section 5: too slow"` |
| 其余 | — | 继续到 20k |

- 2k、5k 只评估、只记录，**不触发停止**。
- 不得人工挑点；不得因成绩好坏另行停机或续训；不得改 λ、奖励、网络、种子或预算。
- 写 `STOP_REQUESTED` 时，训练会在下一个决策干净停止。已写出的检查点由监视进程继续评估完，全部保留。

## 6. 只有这些情况才停下上报（其余一律按本单继续）

| 情况 | 处置 |
|---|---|
| 准入 A/B/C 不通过 | 停，上报 |
| 训练进程崩溃、断电、被结束 | 保留目录，写 `INTERRUPTED.md`。**不续训、不重跑**；另一条照常跑完，再一起交付 |
| `ABORTED.json`（参数非有限） | 技术停机，保留产物；另一条照常 |
| 监视进程崩溃 | 重启同一条 `--watch`（已完成的开局会自动跳过），不上报 |
| 内存不足、系统卡死 | 上报；不得自行减少 n-envs |

## 7. 交付（两条都结束，且最后一个检查点的评估写完后）

```powershell
foreach ($r in "t1fix_m0_s0","t1fix_m2_s0","t1fix_cal_m0_s0") {
  python -B -m experiments.t1_pretrain_checks stats logs/t1/$r --output logs\t1\$r\stats_final.json
  python -B -m experiments.t1_pretrain_checks reload logs/t1/$r --output logs\t1\$r\reload.json
}
python -B -m experiments.t1_fix_readout --m0 logs/t1/t1fix_m0_s0 --m2 logs/t1/t1fix_m2_s0 --rules k07=logs/t1/_fix/rule_k07,k10=logs/t1/_fix/rule_k10 --pure evidence/T1_dev_pure_295000 --output logs\t1\_fix\first_train_readout.json
Get-FileHash -Algorithm SHA256 logs\t1\t1fix_*\checkpoints\*.zip, logs\t1\t1fix_*\final.zip | Export-Csv logs\t1\_fix\MODEL_SHA256.csv -NoTypeInformation
```

**打包** `T1_FIX_S0_FINAL_20261015.zip`，内容如下（不含模型 zip 与 replay）：

| 来源 | 文件 |
|---|---|
| 每条运行目录（含校准） | `manifest.json`、`episodes.jsonl`、`decisions.jsonl`、`fix_stats.json`、`cf_stats.json`（M2）、`checkpoints/*_fix_stats.json`、`checkpoints/*_cf_stats.json`、`dev_eval/**`、`heartbeat.json`、`finished.json`、`stopped.json`/`STOP_REQUESTED`（若有）、`stats_final.json`、`reload.json` |
| `logs/t1/_fix/` | 全部文件：`ADMISSION.md`、`frozen_config_local.json`、`calibration.json`、`rule_k07/**`、`rule_k10/**`、`first_train_readout.json`、`MODEL_SHA256.csv`、`*.out`/`*.err`、`LAUNCH_RECEIPT.json` |
| 包外 | `SHA256SUMS.txt`（逐文件） |

**交付位置：**
- 提交到 collab `docs/collaboration/lower/handoffs/t1_fix_s0_20261015/`：数据包（> 50 MB 时放本地，只提交 SHA256 与路径）、`RESULT.md`、`SHA256SUMS.txt`。
- 模型 zip 留在本机。

**`RESULT.md` 只写事实：**
- 冻结的 λ 与校准比值；
- 每条运行的停止原因与决策数；
- 各检查点开发评估的捕获/超时/unsafe/回报/Δv/时间，以及 κ0.7、κ1.0 规则；
- 每决策 CPU 墙钟；
- 准入结果；
- 偏离本单之处。

**不写评价。**

**最后：**
- 更新 `CURRENT.json`：`training_active` = false，phase = `t1_fix_first_train_delivered`；
- fetch 后提交推送 collab，不 force；
- 上层读包后写 `FIRST_TRAIN_READOUT.md`。
