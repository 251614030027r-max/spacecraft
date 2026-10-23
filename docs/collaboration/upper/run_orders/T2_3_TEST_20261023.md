# 自主执行书：T2_3_TEST_20261023（下层本机，约 8–10 小时无人值守；开发测试，非正式出图）

下层按本单自主推进：A 段机制测试（不训练）→ 中间交付 → B 段训练与评估 → 脚本判定 → 一个预先登记的分支 → 交付后停下。

**科学提交（冻结）：** `9fc96e82018a7358a9e866e9e90377bdcb5941e7`（分支 `claude/sac-mpc-coupling-design-ns7g6i`）。下层不得改任何代码。

## 本轮改了什么（背景，一页）

理论问题与场景问题不变。

**主机制：SAC–G★–MPC 双向耦合框架。**
- 下行通路不变：SAC 给出 16 s 阶段意图，G★ 生成可执行计划和入口计划，MPC 安全执行，真值 RK45 是唯一裁判。
- 本轮升级的是上行通路（执行 → 学习），由三步组成：
  1. **反事实生成：** 用 G★ 和 MPC 自己的规则，预测其它意图会怎样执行。
     - 入口结局按实测的"首次入口 → 终止"结果校准，丢失的入口不再记为失败（T2.2 的入口标签因此平均偏低 69）。
     - 校准门槛从 5 次入口提高到 20 次入口。
  2. **真实执行校正：** 标签在本阶段执行完之后才计算。
     - 目标姿态与追踪器动作无关，所有候选共用实测姿态轨迹。
     - 已执行意图每 2 s 的实测跟踪偏差，搬到每个候选的计划上；本步入口的实测结局偏差同理。
  3. **可信度融合：** w = 1/(1+(e_a/σ)²)，β = 0.5。
     - e_a 是对已执行意图的预测误差，σ 是真实 TD 残差的滑动 RMS。
     - 融合后的目标仍是 γ-收缩，且 w·|e_a| ≤ σ/2。
     - 这一项取代 T2.2 的四类门控、梯度上限和入口成功率估计。

**副机制：旋转同步的窗口制导与承诺。**
- 制导部分即 C1，锥面几何动作空间，保持不变。
- 本轮新增承诺部分：入口承诺的对象从"到达时刻"改为"端口的这一次通过"。
  - 固定时刻的重规划因跟踪滞后而失败时，先在同一次通过内后移 2 / 4 / 8 s；
  - 后移仍不可行时，动用推力储备（95%）；
  - 不越过截止期。
- 依据（上层探针）：
  - 23 个回合共 31 次承诺，丢失 17 次；τ = 60 的 11 次全部丢失；
  - 丢失时姿态预报误差的中位数为 2°；
  - 在丢失时刻，晚到 2–4 s 或动用储备即可行。
- 沙箱配对（8 个开局 × 2 种策略）：丢失次数 13 → 1，回报平均 +3.1/回合。

**提速：**
- 异步采集，SAC 臂不计算候选。
- 沙箱实测 4.1 s/决策（3 个 worker，CPU 有争用）；T2.2 锁步采集约 15 s/决策。
- 训练的到达顺序记录在 `finished.json` 的 `worker_order`。
- 异步训练不能逐位复现；评估仍然确定、串行。

## 0. 准备（约 5 min）

```powershell
cd D:\py\DRL2; git fetch origin claude/sac-mpc-coupling-design-ns7g6i; git checkout --detach 9fc96e82018a7358a9e866e9e90377bdcb5941e7
$py = "D:\py\DRL2\.venv\Scripts\python.exe"
$R  = "C:\Users\35884\Documents\Spacecraft\过程文件\T2_3_TEST_20261023"
New-Item -ItemType Directory -Force $R | Out-Null
$env:OMP_NUM_THREADS = "1"; $env:MKL_NUM_THREADS = "1"; $env:OPENBLAS_NUM_THREADS = "1"
& $py -B -m pytest -q tests/test_t2_stage.py *> "$R\pytest.log"          # 须 23 passed
```

pytest 未全部通过时停下，回报日志，不得继续。

```powershell
function Train($tag, $arm, $seed, $steps, $marks, $extra) {
  Start-Process -FilePath $py -WorkingDirectory "D:\py\DRL2" -WindowStyle Hidden `
    -ArgumentList (@("-B","-m","experiments.t2","train","--arm",$arm,"--seed",$seed,"--steps",$steps,"--workers","3",
                     "--checkpoints",$marks,"--output","$R\$tag") + $extra) `
    -RedirectStandardOutput "$R\$tag.out.log" -RedirectStandardError "$R\$tag.err.log"
}
function EvalRun($tag, $label, $root, $steps, $extra) {   # 检查点 step_<k>.zip（训练会保存每个标记和最终步）
  foreach ($k in $steps) {
    & $py -B -m experiments.t2 eval --checkpoint "$R\$tag\checkpoints\step_$k.zip" --label "${label}_step_$k" --openings 294000-294023 `
      --workers 10 --output "$R\$root\$label\step_$k" @extra *>> "$R\$root.log"
  }
}
```

## A 段：机制测试，不训练（约 1–1.5 h，两项并行）

若 `FIG_T2_20261023` 已跑完，它的 1a/1b 就是本段：把 `A1_*`、`A1_audit`、`A2_*` 复制到本单的 `$R`，跳过本段直接进入 B 段。

**A1 标签精度审计。** 在每个决策点分叉真值，每个候选在自己的分支上真飞一段（标签跨两段时飞两段）。比较两种标签：
- raw：只用校准，等同 T2.2；
- anchored：本轮 M2。

开局 2910000–2910023，行为策略为规则或按开局抽取的恒定意图。

```powershell
& $py -B -m experiments.t2_audit warmup --output "$R\A1_CAL.json" --episodes 40 --workers 10 *> "$R\A1_warmup.log"   # 直到阶段校准 ≥10 段、入口校准 ≥20 次
& $py -B -m experiments.t2_audit labels --calibration "$R\A1_CAL.json" --openings 2910000-2910023 --workers 6 --output "$R\A1_audit" *> "$R\A1_audit.log"
& $py -B -m experiments.t2_audit summary --input "$R\A1_audit" --output "$R\A1_LABEL_AUDIT.json" *> "$R\A1_summary.log"
```

**A2 副机制筛选。** 同样的开局和固定策略，分别在窗口承诺和刚性承诺（T2.2）下运行，配对比较。

```powershell
$pol = "c_10_0,c_6_5,c_10_-5,rally,gstar"
& $py -B -m experiments.t2 screen --openings 294000-294023 --policies $pol --workers 6 --output "$R\A2_window" *> "$R\A2_window.log"
& $py -B -m experiments.t2 screen --openings 294000-294023 --policies $pol --workers 6 --output "$R\A2_rigid" --no-window *> "$R\A2_rigid.log"
& $py -B -m experiments.t2_readout window --rigid "$R\A2_rigid" --window "$R\A2_window" --output "$R\A2_WINDOW.json" *> "$R\A2_readout.log"
```

**A 段中间交付（立即，不等 B 段）：**
- 科学分支 `evidence/T2_3_TEST/A/`，逐个文件 `git add -f`，附 SHA256SUMS：
  - `A1_CAL.json`、`A1_LABEL_AUDIT.json`、`A1_audit/*.json`；
  - `A2_WINDOW.json`、`A2_window/*.json`、`A2_rigid/*.json`；
  - `pytest.log`。
- collab `lower/handoffs/t2_3_test_20261023/RESULT_A.md`：
  - A1：raw 与 anchored 两列（候选数、入口事件一致率、TP/FP/FN、入口标签误差、阶段回报误差、位置误差中位数/p90）；
  - A2：配对回报、丢失入口数、各策略的配对差；
  - 墙钟。

A 段的数字只记录，**不作为 B 段的放行条件**；A 段一交付就开始 B 段。

## B 段：训练与评估（约 6 h）

**B1 训练。** 两臂 × 种子 0、1，共 4 个进程同时启动；每个进程 3 个 worker，4000 次决策，从零开始。两臂都用窗口承诺。

```powershell
foreach ($a in "e_sac","e_mve") { foreach ($s in 0,1) { Train "TRAIN_${a}_s$s" $a $s 4000 "1000,2000,3000" @() } }
```

训练中可以只读 `heartbeat.json` 看进度，不得打开或修改其他文件。

**B2 评估、读出、判定。** 4 个训练进程都写出 `finished.json` 后进行。

```powershell
foreach ($a in "e_sac","e_mve") { foreach ($s in 0,1) { EvalRun "TRAIN_${a}_s$s" "${a}_s$s" "B_eval" @(0,1000,2000,3000,4000) @() } }
& $py -B -m experiments.t2_readout curve --eval-root "$R\B_eval" --train-root "$R" --screen "$R\A2_window" --output "$R\B_READOUT.json" *> "$R\B_readout.log"
& $py -B -m experiments.t2_decide --criteria t23 --readout "$R\B_READOUT.json" --output "$R\DECISION_B.json" *> "$R\decide_B.log"
```

`verdict` 为 `incomplete` 时，用同一命令补跑缺失的评估（已有文件会跳过），然后再判定一次。

## 预先登记的判定标准（`experiments/t2_decide.py --criteria t23`，事后不得修改）

| 项 | 标准（4000 步，种子 0、1） |
|---|---|
| C0 完整 | 每个臂、每个检查点（0、1000、2000、3000、4000）都是 24/24，且没有 `ERROR_*.json` |
| C1 超过上一轮 | 每个种子的 EACC（`e_mve`）4000 步平均回报 > 54.32（T2.2 最好的臂：SAC 种子 1，3000 步） |
| C2 耦合（曲线面积） | 每个种子：配对 EACC − SAC 在 1000/2000/3000/4000 四个检查点上的平均 > 0 |
| C3 末点不落后 | 两种子 4000 步配对 EACC − SAC 的平均 ≥ −3 |
| C4 收敛 | 每个臂、每个种子：\|R(4000) − R(3000)\| ≤ 10 |
| C5 安全 | 每个种子 4000 步：EACC 捕获数 ≥ SAC − 1，且 EACC unsafe ≤ SAC + 1 |

- C1–C5 全部满足：`pass`；
- 否则 C4 不满足：`not_converged`；
- 否则两臂的（4000 − 0）平均增益都 < 60：`learning_failure`；
- 其余：`coupling_failure`。

**读法：**
- 耦合的因果证据只看同版本、同种子的 EACC − SAC 配对。
- 与 T2.2 的比较只说明系统总体是否变好：执行器变了（窗口承诺），两臂都受益。

## B 段之后：按判定只跑一个分支（约 5 h，可以超出 10 h，跑完为止）

| 判定 | 分支 | 内容 |
|---|---|---|
| `pass` | P | 两臂 × 种子 2 × 4000，评估 0–4000，再以种子 0、1、2 判定一次，作为第三个开发种子 |
| `not_converged` | N | 两臂 × 种子 0 × 6000（从零开始），检查点 1000–5000，评估 0–6000，以 `--final 6000` 判定 |
| `coupling_failure` | K | 只训 EACC，种子 0、1，β = 1.0（`--model-beta 1.0`），4000 步；SAC 复用 B 段（同代码、同种子，β 不影响 SAC） |
| `learning_failure` | L | 两臂 × 种子 0、1，每条样本更新 1 次（`--updates-per-transition 1`），4000 步 |

```powershell
# P
foreach ($a in "e_sac","e_mve") { Train "P_${a}_s2" $a 2 4000 "1000,2000,3000" @() }
# 全部 finished.json 后：
foreach ($a in "e_sac","e_mve") { EvalRun "P_${a}_s2" "${a}_s2" "B_eval" @(0,1000,2000,3000,4000) @() }
& $py -B -m experiments.t2_readout curve --eval-root "$R\B_eval" --train-root "$R" --screen "$R\A2_window" --output "$R\P_READOUT.json" *> "$R\P_readout.log"
& $py -B -m experiments.t2_decide --criteria t23 --readout "$R\P_READOUT.json" --output "$R\DECISION_P.json" *> "$R\decide_P.log"

# N
foreach ($a in "e_sac","e_mve") { Train "N_${a}_s0" $a 0 6000 "1000,2000,3000,4000,5000" @() }
foreach ($a in "e_sac","e_mve") { EvalRun "N_${a}_s0" "${a}_s0" "N_eval" @(0,1000,2000,3000,4000,5000,6000) @() }
& $py -B -m experiments.t2_readout curve --eval-root "$R\N_eval" --train-root "$R" --screen "$R\A2_window" --output "$R\N_READOUT.json" *> "$R\N_readout.log"
& $py -B -m experiments.t2_decide --criteria t23 --final 6000 --readout "$R\N_READOUT.json" --output "$R\DECISION_N.json" *> "$R\decide_N.log"

# K
foreach ($s in 0,1) { Train "K_e_mve_s$s" "e_mve" $s 4000 "1000,2000,3000" @("--model-beta","1.0") }
foreach ($s in 0,1) { EvalRun "K_e_mve_s$s" "e_mve_s$s" "K_eval" @(0,1000,2000,3000,4000) @()
                      Copy-Item -Recurse "$R\B_eval\e_sac_s$s" "$R\K_eval\e_sac_s$s" }
& $py -B -m experiments.t2_readout curve --eval-root "$R\K_eval" --train-root "$R" --screen "$R\A2_window" --output "$R\K_READOUT.json" *> "$R\K_readout.log"
& $py -B -m experiments.t2_decide --criteria t23 --readout "$R\K_READOUT.json" --output "$R\DECISION_K.json" *> "$R\decide_K.log"

# L
foreach ($a in "e_sac","e_mve") { foreach ($s in 0,1) { Train "L_${a}_s$s" $a $s 4000 "1000,2000,3000" @("--updates-per-transition","1") } }
foreach ($a in "e_sac","e_mve") { foreach ($s in 0,1) { EvalRun "L_${a}_s$s" "${a}_s$s" "L_eval" @(0,1000,2000,3000,4000) @() } }
& $py -B -m experiments.t2_readout curve --eval-root "$R\L_eval" --train-root "$R" --screen "$R\A2_window" --output "$R\L_READOUT.json" *> "$R\L_readout.log"
& $py -B -m experiments.t2_decide --criteria t23 --readout "$R\L_READOUT.json" --output "$R\DECISION_L.json" *> "$R\decide_L.log"
```

分支 P 的种子 2 评估写进同一个 `B_eval`，读出时三个种子一起判定。只跑这一个分支，交付后停下。

## 交付

**科学分支 `evidence/T2_3_TEST/B/` 与 `evidence/T2_3_TEST/<P|N|K|L>/`：**
- `*_READOUT.json`、`DECISION_*.json`、评估 json；
- 各训练目录的 `manifest.json`、`finished.json`、`episodes.json`、`model_stats.json`、`heartbeat.json`；
- 附 SHA256SUMS。

**collab `lower/handoffs/t2_3_test_20261023/RESULT.md`：**
- B 段的判定与 C0–C5 的取值；
- 4 条曲线（0–4000），按紧急 / 经济拆分；
- 配对比较；
- 训练统计：`replay.reason` 分布、`w_mean`、`abs_err_median`、`sigma_last`、`q_pi_last_decile`；
- 每个训练的墙钟（`finished.json` 的 `wall_s`）和决策/小时；
- 所走分支与它的判定；
- 自行处理过的问题清单。

**只报告事实。** 判定以 `DECISION_*.json` 为准。

## 下层可以自行处理的情况

- **允许自行修复，事后在报告中写明：**
  - Windows 路径、编码、进程启动；
  - 单个回合报错（记为 `ERROR_*.json`，C0 不完整，补跑）；
  - 中断后用同一命令补跑；
  - 评估、筛选、审计的 `--workers`。
- **训练进程崩溃：** 同一配置、同一种子从零重启一次；第二次仍崩溃，就停下当前段并交付已有结果。
- **CPU 过载：** 先降评估/筛选/审计的 workers；训练的 worker 数（3）不得改。
- **不得做的事：** 改代码、开关取值、步数、开局、判定标准；删掉失败回合；挑选检查点；在判定之外增加训练。
