# 执行单：T1_B_V3_ROUND_20261019（下层本机；上层技术裁决，研究层 2026-10-19 授权上层主导）

**科学提交（冻结）：** `e33a0dd7b0bd0d73b7d3ad16561f1dd494d242fa`。不得修改任务、D/B、G★、MPC、真值判据、网络、初值、训练器、温度、ε、CF 门控、梯度比例、开局或步数。

**取代：** `T1_B_V2_PILOT_20261019`（已按 A1 停止并交付）。

**定位（科学 `evidence/T1B_V3_PRECHECK/README.md`）：** V2 每回合有 13.9 次学习决策，其中 12.7 次是“已有计划时每 6 s 再选一次 提前 15 / 保持 / 推迟 15”。
- 晚到达的意图要靠连续约 15 次随机选择一致才能维持，critic 学到的是噪声策略下局部平移的价值，均衡点就是“一直提前”。V2 的 A1 停止不是梯度饱和（梯度仍为 0k 的 77 %），而是学出来的偏好。
- 真值扫描是“选定后保持”，与 V1/V2 的 Q 不是同一个决策，所以排序 5/15 一致在意料之中。

**V3 只改决策结构：**
- 只有在没有在用计划时才是学习决策（首次，以及计划丢失后），候选是 G★ 核验的 [最早, +20, +40]；
- 计划在用时只能保持它自己的到达时刻。
- 结果：一次决策的真实区间一直延续到计划丢失或回合结束，结束于终止时就是该意图的 Monte Carlo 回报。MPC 反事实只在这些决策处配对，两侧都是新 G★ 计划。
- 学习器（α 0.5、ε 0.10、散列替代、20 % 专家 Replay、5 % / 10 %）与 V2 相同。

## 0. 同步与检查（约 20 min）

```powershell
cd D:\py\DRL2
git status --porcelain                 # 只允许被忽略的产物
git fetch origin claude/sac-mpc-coupling-design-ns7g6i
git checkout --detach e33a0dd7b0bd0d73b7d3ad16561f1dd494d242fa
$py = "D:\py\DRL2\.venv\Scripts\python.exe"
$env:OMP_NUM_THREADS="1"; $env:MKL_NUM_THREADS="1"; $env:OPENBLAS_NUM_THREADS="1"
$R = "C:\Users\35884\Documents\Spacecraft\过程文件\T1B_V3_ROUND_20261019"
New-Item -ItemType Directory -Force $R | Out-Null
& $py -B -m pytest -q tests/test_t1_b_v2.py tests/test_t1_b.py tests/test_t1_b_learning.py tests/test_t1_b_checkpoint_loading.py *> "$R\pytest.log"
& $py -B -m experiments.t1b_v2 contract --interface v3 --output "$R\contract"
& $py -B -m experiments.t1b_v2 expert --interface v3 --steps 1000 --output "$R\expert"   # V1 教师规则，在 V3 接口上重采，所有臂与种子共用
```

条件：测试全过（13 项）；`contract\CONTRACT.json` 中 `passed = true`；`expert\EXPERT_RECEIPT.json` 中 `interface = v3`、`transitions = 1000`。

## 1. 第一阶段：种子 0，E-SAC 与 E-TCPF 各 20k，同时启动（预计约 5–6 h）

```powershell
function Start-V3($a, $s) {
  Start-Process -FilePath $py -WorkingDirectory "D:\py\DRL2" -WindowStyle Hidden -PassThru `
    -ArgumentList "-B","-m","experiments.t1b_v2","train","--interface","v3","--arm",$a,"--seed",$s,"--steps","20000", `
                  "--expert","$R\expert\expert_replay.pkl","--output","$R\TRAIN_${a}_s$s" `
    -RedirectStandardOutput "$R\${a}_s$s.out.log" -RedirectStandardError "$R\${a}_s$s.err.log"
}
Start-V3 "e_sac" 0; Start-V3 "e_tcpf" 0
```

- 检查点 0 / 500 / 2000 / 5000 / 10000 / 20000。
- 技术停止只限 `Traceback`、非有限参数或数值失败，此时从零重启并加后缀 `_restart1`。不续训、不挑检查点。

**机制判定 M（2k 检查点出现后立即检查；任一项不过就停止两臂并上报）：**
1. `decisions.jsonl` 中，`support.learning_decision = true` 的行里 `observation_before[27]`（has_plan）全部为 0；
2. 已完成回合的平均学习决策数在 1–4 之间；
3. E-TCPF 的 `model_prefix_releases.jsonl` 里有效配对（`gate.weight > 0`）多于 0，且替代候选比实际候选更早、更晚的两种情况都出现；
4. 两臂 `audit_2000` 中，至少 4/8 个首审状态的 `entropy` > 0.05。

## 2. 读出（每个检查点出现后执行；确定性；两个开发块）

```powershell
foreach ($a in "e_sac","e_tcpf") { foreach ($m in 2000,5000,10000,20000) {
  foreach ($blk in "292140-292147","292120-292127") {
    & $py -B -m experiments.t1b_v2 eval --interface v3 --checkpoint "$R\TRAIN_${a}_s0\checkpoints\step_$m.zip" --label "${a}_s0_$m" --openings $blk --workers 4 --output "$R\eval_${a}_s0_$m"
  }
  & $py -B -m experiments.t1b_v2 audit --checkpoints "$R\TRAIN_${a}_s0\checkpoints\step_0.zip,$R\TRAIN_${a}_s0\checkpoints\step_$m.zip" `
      --episodes evidence\T1B_V1\episodes\interface_0k --output "$R\audit_${a}_s0_$m.json"
} }
```

在 `RESULT.md` 中逐检查点、逐臂报告（只写事实）：
- **学习：** 两块各自的捕获数、平均原折扣回报，以及首次决策选的候选序号。对照“总是最早”：292140–147 为 7/8、36.54；292120–127 为 6/8、16.38。真值最佳首选：292140–147 为 8/8、60.87；292120–127 为 6/8、18.70。真值来自科学 `evidence/T1B_V1_AUDIT/candidate_truth_scan/` 与 `evidence/T1B_V2_PILOT/scan_292120/`。
- **状态相关性：** 8 个首审状态上 Q 最高的候选与真值最佳首选一致的个数；经济型与急迫型各自的首选分布。
- **反馈：** E-TCPF 的有效配对数、`applied_gradient_share` 中位数，以及标签 Y 的即时部分与自举部分（方法同 V1 审计）；两臂首次决策不同的状态数；逐开局配对回报差。

**学习判定 L（20k）：** 至少一臂在 292140–147 上平均原折扣回报 > 36.54，且至少一个开局的首次决策不是最早。
- 通过：直接进入第 3 节。
- 不通过：完成读出后交付，不扩种子。

## 3. 第二阶段（L 通过即自动执行）：种子 1、2 × 两臂 × 20k

与第 1、2 节相同，`Start-V3` 的种子改为 1、2，运行名与读出标签跟着改。可以 4 个运行同时启动；如果 CPU 实测饱和，就按种子顺序排队，但不得减少臂、种子或步数。

## 4. 交付

- **collab** `docs/collaboration/lower/handoffs/t1b_v3_round_20261019/`：`RESULT.md`、`ADMISSION.md`（第 0 节与判定 M）。
- **科学分支** `evidence/T1B_V3_ROUND/`：每个文件单独 `git add -f`。内容包括：
  - 每个运行的 `manifest.json`、`finished.json`、`learning_audit.json`、`feedback_audit.json`、`delta_calibration.json`；
  - `decisions.jsonl` 与 `model_prefix_releases.jsonl`（不超过 50 MB）；
  - 全部 `eval_*` 与 `audit_*`；
  - 各检查点模型 zip 及其 SHA256；
  - `EXPERT_RECEIPT.json`。
- **`CURRENT.json`：** `topic = t1b_v3_round`，`scientific_commit = e33a0dd`，`run_order` 指向本文件。
