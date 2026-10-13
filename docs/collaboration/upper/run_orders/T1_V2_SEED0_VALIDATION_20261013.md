# 执行单：T1_V2_SEED0_VALIDATION_20261013（下层本机执行）

**替代：** `T1_ROUND1_M0M1_20261012.md` 中尚未执行的部分（种子 0 重跑、种子 1/2）全部作废。该执行单已执行部分（准入、种子 0 至中断）保持原样。
**授权（用户 2026-10-13）：** 同意上层对方法做相应修改；**只跑满一个种子做验证**，不开启三种子长训练；可以提升训练速度；20k 不够可以适当增加，只要不是进入平台期。
**科学提交：** `78eb63c22615c9672c43338be4cec54da713812c`（分支 `claude/sac-mpc-coupling-design-ns7g6i`）。
**本轮只训练一条：M1、种子 0。** M0 和种子 1、2 都**未授权**。

---

## 0. 为什么改、改了什么（下层只需知道，不需判断）

上层对中断数据包（collab `709a3ec`）的审查结论如下：

- 学习开始后，SAC 把 a0 推到 ±1 饱和（1–3k 决策时约 75% 的决策是饱和的）。
- 在接口 v1 下，a0 只要不在 0 附近，就会请求重规划。所以饱和意味着几乎每 2 s 都重规划，随之出现滑移、10–20% 的惯性保持，以及远场距离失败或超时。
- 学习确实在发生，但很慢。

本次改动是 M0/M1 共用的接口和执行层，对两者同样生效。奖励、任务、Pure MPC、共享 MPC 都不动。

| 改动 | 内容 |
|---|---|
| G★ v0.4（计划稳定） | 已有可用计划时，不会因为请求"更早"而换成一个更晚的计划，也不会因为请求不可行而丢掉现有计划。没有计划且请求不可行时，先用 100% 推力在最早网格上做一次恢复求解，然后才进入惯性保持 |
| 接口 v2（保持带） | 有计划时：\|a0\| ≤ 0.5 表示保持；超出部分线性映射为 ±30 s 的平移。无计划时：照旧映射为绝对时间 0..160 s |
| 每个转移 4 次梯度更新（UTD 4） | gradient_steps = 4 × 并行环境数 |
| 并行环境 `--n-envs 6` | 第 i 个环境的开局号为 1_000_000 + i·100000 + 回合号，与评估块不重叠 |
| M1 预热 | 改为绝对 3000 次决策 |
| 检查点 | 每跨过一个 5000 的倍数保存一次。**另外保存未训练策略 `m1_0_decisions.zip` 作为学习曲线的零点**。在接口 v2 下，未训练策略的确定性动作 a0≈0 正好落在保持带内，本身就可能有不错的成绩，所以学习效果必须相对这个零点来读 |
| 停机 | 训练目录下出现 `STOP_REQUESTED` 时干净停止（写 `stopped.json`、`final.zip`、`finished.json`） |
| 开发评估 | `--workers` 并行跑开局；`--stop-on-plateau` 按第 4 节的预注册规则自动写 `STOP_REQUESTED` |

上层沙箱已核验以下各项：

- 规则 `rule:earliest` 在 16 个开发开局上，用 4 个进程并行，**逐值复现** `evidence/T1_dev_rule_295000`：16/16，81.08945807364475。说明 v0.4 对规则无影响，并行评估也是确定的。
- 3 环境 M1 冒烟：辅助损失生效；各环境的反事实样本互不串扰；干净停机正常；监视器按检查点评估后退出。
- 全部测试：398 通过，3 个预期失败。
- 平台期规则另有单元测试。

---

## 1. 现场与旧目录

- `D:\py\DRL2` 保持 `4795910`，不动。
- `logs/t1/t1_m0_s0`、`logs/t1/t1_m1_s0`（中断的两条）原样保留：不续训、不删除、不移动。
- 本轮使用新的 run-name：`t1v2_m1_s0`。

```powershell
Get-Process python -ErrorAction SilentlyContinue | Format-Table Id, StartTime, Path   # 应无输出；若有，停下上报
cd D:\py\DRL2_T1
git status --porcelain                  # 应无输出（logs/ 被忽略）
git fetch origin
git checkout --detach 78eb63c22615c9672c43338be4cec54da713812c
git log -1 --format=%H                  # 必须是 78eb63c22615c9672c43338be4cec54da713812c
git status --porcelain                  # 必须无输出
git diff --stat f36ac77 78eb63c -- requirements.txt requirements-mpc.txt requirements-dev.txt   # 应无输出：沿用现有环境，不升级不降级
$env:OMP_NUM_THREADS="1"; $env:MKL_NUM_THREADS="1"; $env:OPENBLAS_NUM_THREADS="1"
mkdir logs\t1\_v2 -ErrorAction SilentlyContinue
```

**之后每次新开 PowerShell 窗口，都先进入 `D:\py\DRL2_T1`，再设置上面三个线程变量。** 训练结束之前，不得在这个工作树里切换、拉取、修改或提交。

## 2. 准入（全部通过才能启动）

```powershell
python -B -m pytest -q tests/test_t1.py tests/test_t1_gym.py tests/test_t1_counterfactual.py tests/test_t1_m1.py tests/test_t1_termination.py tests/test_t1_dev_eval.py
python -B -m experiments.t1_frozen_config --output logs\t1\_v2\frozen_config_local.json
python -B -m experiments.t1_dev_eval --policy rule:earliest --openings experiments/t1_dev_openings_295000.json --output-dir logs/t1/_v2/rule_local --workers 4
```

| 准入项 | 通过条件 |
|---|---|
| A 测试 | 全部通过 |
| B 冻结配置 | `code_commit` 为 78eb63c…，`code_dirty` 为 false；`gstar_version` 为 "0.4"；`gym_interface_version` 为 "2"；`gym.keep_band` 为 0.5；`sac.updates_per_transition` 为 4；`m1.warmup_decisions` 为 3000 |
| C 规则基准 | 16/16，平均折扣回报与 81.0895 相差 ≤ 0.5（上次本机算出 81.089873，应当不变）。记下用时 |
| D 磁盘 | 剩余空间 ≥ 10 GB |

任何一项不通过，就停下上报，不要处理。

## 3. 启动（防睡眠；独立窗口，不随下层会话窗口关闭）

上次中断，是因为进程挂在会被关掉的窗口上。这次用 `Start-Process` 让每个进程有**自己的最小化窗口**（不加 `-NoNewWindow`）。关闭下层自己的 PowerShell 或会话窗口，不会结束这些进程；**不要手动关闭那两个最小化窗口。**

```powershell
powercfg /query SCHEME_CURRENT SUB_SLEEP STANDBYIDLE > logs\t1\_v2\powercfg_before.txt
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0

Start-Process python -WindowStyle Minimized -ArgumentList "-B -m train.train_t1 --method m1 --steps 60000 --seed 0 --n-envs 6 --run-name t1v2_m1_s0 --log-root logs/t1 --checkpoint-freq 5000" -RedirectStandardOutput logs\t1\_v2\train.out -RedirectStandardError logs\t1\_v2\train.err
Start-Sleep 20
Start-Process python -WindowStyle Minimized -ArgumentList "-B -m experiments.t1_dev_eval --watch logs/t1/t1v2_m1_s0 --openings experiments/t1_dev_openings_295000.json --checkpoints 0,5000,10000,15000,20000,25000,30000,35000,40000,45000,50000,55000,60000 --workers 3 --stop-on-plateau" -RedirectStandardOutput logs\t1\_v2\watch.out -RedirectStandardError logs\t1\_v2\watch.err
```

- **推荐由用户在自己新开的 PowerShell 窗口中执行上面的命令**（与 F01 相同的做法：设置线程变量后，执行 `powercfg` 和两条 `Start-Process`）。若下层的工具会话会在结束时连带结束它启动的子进程，就必须交给用户启动。下层负责准入、启动后核验进程是否存活，以及写回执。
- 训练脚本在有改动的检出上会拒绝运行。若报 `refusing to train from a modified or unknown checkout`，停下上报，**不得**加 `--allow-dirty`。
- 进程分配：训练占 1 个主进程和 6 个环境进程，评估占 3 个进程。共 10 个，本机 12 核。
- 记录：启动回执写入 `lower/handoffs/t1_v2_seed0_20261013/LAUNCH_RECEIPT.json`，内容包括 PID、命令和 UTC 时间。同时更新 `CURRENT.json`：topic 设为 `t1_v2_seed0`，scientific_commit 设为 78eb63c…，run_order 设为本文件。

## 4. 预算与停止规则（预先注册，由监视进程机械执行）

- **下限 20k，上限 60k**（`--steps 60000`）。
- **0k 和 5k…20k 只评估，不触发停止。**
- 从 25k 起，每个检查点做如下判断。若同时满足以下两条，该检查点记为"未改善"：
  1. 平均折扣回报不超过此前所有检查点的最好值加 2.0；
  2. 捕获数不超过此前所有检查点的最好值。
- **连续两个检查点"未改善"**，监视进程就写 `STOP_REQUESTED`（文件内附各检查点的捕获数和回报），训练随即干净停止。
- 评估滞后于训练。停止前已经写出的检查点，监视进程照样评估，这些结果全部保留。

成绩好坏都**不是**人工停机理由。不得改奖励、网络、门控、预算或种子；不得删除或挑选检查点；**不得手动写 `STOP_REQUESTED`**。

## 5. 运行中检查（只读，训练中可做）

| 时点 | 检查 | 处置 |
|---|---|---|
| 启动后 30 min | `heartbeat.json` 存在，`parameters_finite` 为 true；吞吐 = decisions / wall_s × 3600 | 吞吐低于 3000 次决策/h 时上报，**但不停止** |
| 0k 评估写出后 | 记下 `dev_eval/0/summary.json`（未训练策略的零点） | — |
| 5k 评估写出后 | 运行下面第一条命令，做一次快照 | 若 `ABORTED.json` 存在，或参数非有限，停下上报 |
| 每 5k | 新的 `dev_eval/<n>/summary.json` 是否写出 | 监视进程崩溃时，重启同一条 `--watch` 命令（已完成的开局会自动跳过） |

```powershell
python -B -m experiments.t1_pretrain_checks stats logs/t1/t1v2_m1_s0 --output logs\t1\_v2\stats_5k.json     # 已能读取正在写入的日志（末尾半行会被丢弃）
python -B -m experiments.t1_pretrain_checks reload logs/t1/t1v2_m1_s0 --output logs\t1\_v2\reload.json       # 结束后运行一次
```

## 6. 异常（预先写定）

| 情况 | 处置 |
|---|---|
| 训练进程崩溃、断电、被结束 | 保留目录，写 `INTERRUPTED.md`（时间、最后一次心跳、原因）。**不续训，不自行重跑**，上报后等上层 |
| `ABORTED.json` | 技术停机：保留全部产物，上报 |
| 评估监视进程崩溃 | 重启同一条 `--watch` 命令 |
| 内存不足或系统卡死 | 上报；不自行减少 n-envs |

## 7. 交付（训练停止或到 60k，且最后一个检查点的评估写完后）

```powershell
python -B -m experiments.t1_pretrain_checks stats logs/t1/t1v2_m1_s0 --output logs\t1\t1v2_m1_s0\stats_final.json
python -B -m experiments.t1_pretrain_checks reload logs/t1/t1v2_m1_s0 --output logs\t1\t1v2_m1_s0\reload.json
# 再把待机/休眠超时恢复为 powercfg_before.txt 中记录的原值（用户确认后执行）
```

**交付到** `lower/handoffs/t1_v2_seed0_20261013/REPORT.md`，以及同目录下的小文件。

**报告第一张表：** 每个检查点一行（0k、5k、10k…）。各行列出：

- 捕获数/16、平均折扣回报、失败按 300 s 计的平均时间、平均 Δv；
- 失败按类型分列（`summary.json` 中的 `failures`）；
- 保持（kept）次数和新计划次数（评估 JSON 中已有）。

表头下列出两行参照：

- **Pure MPC 开发块：11/16，平均折扣回报 0.91**（证据见科学分支 `evidence/T1_dev_pure_295000`，提交 62d6346，只加证据，代码与 78eb63c 相同）；
- **规则：16/16，81.0895。**

**附件：**

- 运行目录内：`manifest.json`、`heartbeat.json`、`finished.json`；有 `stopped.json` 时一并附上，`STOP_REQUESTED` 也一并附上；
- 日志：`episodes.jsonl` 和 `decisions.jsonl`，均 gzip 压缩，保留原字节；
- `cf_stats.json` 以及各检查点的 `*_cf_stats.json`；
- `dev_eval/` 下的全部 JSON；
- `stats_5k.json`、`stats_final.json`、`reload.json`；
- `_v2` 下的准入文件：冻结配置、`rule_local/summary.json`、`train.out`/`err`、`watch.out`/`err`；
- 全部文件的 `SHA256SUMS`。

模型 `.zip` 只列出 SHA-256，**不上传**。

**报告里只写事实和表格，不下"方法有效"的结论。** 是否与 Pure 判定优劣、是否接着跑同接口的 M0、是否开种子 1/2，都由上层和用户决定。**交付后停止。**

## 8. 表述限制（用户特别记录，继续有效）

M1 在开局择时阶段的辅助门控通过率只有约 4%，这是已知的研究局限。**任何交付和报告都不得把计划管理（途中保持/重规划）上的改善表述为开局择时机制的学习优势。** 门控通过率必须按决策类型分列（`stats` 中的 `gate_by_kind`），不得只报总体比例。

本轮改动是 M0/M1 共用的接口和执行层。所以即便单种子 M1 在开发块上超过 Pure，可以归因的也只是"本架构 + 本轮学习"，**不是 M1 相对 M0 的增量**。
