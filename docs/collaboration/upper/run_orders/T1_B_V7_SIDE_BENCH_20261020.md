# 执行单：T1_B_V7_SIDE_BENCH_20261020（下层本机；与 V7 并行或其后执行；不训练、不改 V7）

**目的：** 回应研究层 2026-10-20 对 V7 的审查。不再按“改一版、跑一轮”试错，而是把下一步判断需要的证据一次备齐。
1. 用一组 T1-B 从未接触过的独立开局，测出“与状态相关的择时收益”在整体上有多大，以及 V7 的 MPC 成对标签在这些新状态上有多准。现有开发集的潜在改善约 95 % 来自 292140 一个开局，单看它不可靠。
2. 用 V7 留下的 Replay 离线回答两个问题：三类 critic 损失的梯度方向是否冲突；有了正确标签，critic 能否学出与状态相关的排序。这样不必再开在线轮次。

**冻结代码：** A 部分用 V7 正在运行的 `54c2587`（不切换检出，不打扰 V7）；B 部分在 V7 结束后检出 `6fa8be8`（只新增离线脚本 `experiments/t1b_critic_bench.py`，训练代码与 `54c2587` 完全相同）。

## A. 独立确认块 292500–292531（32 个开局，T1-B 首次使用；预先登记，之后不得据此改方法）

以 Idle 优先级运行，工作进程 ≤ 2，避免拖慢 V7；若 V7 已结束，可用 6 个。

```powershell
cd D:\py\DRL2   # 保持在 54c2587，不切换
$py = "D:\py\DRL2\.venv\Scripts\python.exe"
$Q = "C:\Users\35884\Documents\Spacecraft\过程文件\T1B_V7_SIDE_BENCH_20261020"
New-Item -ItemType Directory -Force "$Q\truth","$Q\model" | Out-Null
# A1 真值：每个开局 × 每个首选候选的完整真实闭环（后续为 V1 教师：保持计划，无计划时选最早）
Start-Process -FilePath $py -WorkingDirectory "D:\py\DRL2" -WindowStyle Hidden -PassThru `
  -ArgumentList "-B","-m","experiments.t1b_v2","scan","--interface","v3","--openings","292500-292531","--workers","2","--output","$Q\truth" `
  -RedirectStandardOutput "$Q\truth.out.log" -RedirectStandardError "$Q\truth.err.log" | ForEach-Object { $_.PriorityClass = "Idle"; $_ }
# A2 模型：同一状态、同一候选，V7 的真实控制器（QP 每周期）全时域模型分支
$env:PYTHONPATH = "D:\py\DRL2"
foreach ($op in 292500..292531) {
  & $py -B evidence\T1B_V7_PRECHECK\full_horizon_mpc_counterfactual.py $op 1 >> "$Q\model\model_scan.jsonl" 2>> "$Q\model\model_scan.err.log"
}
```

（A2 是串行脚本；如果 V7 已结束，可以把开局分成 6 组同时跑，结果都追加到同一个 jsonl。）

报告（`SIDE_BENCH_RESULT.md`，只写事实，逐开局附表）：
- **学习空间：** 32 个开局里，首选非最早更优的个数，其中优势 > 1、> 5、> 50 的各有几个；“总是最早”与“逐开局最佳首选”的平均回报和捕获数；以上都按经济型、急迫型分开。
- **标签准确度：** 对每对候选比较模型差与真值差，报告方向一致数（真值差 > 0.5 的对）、|模型差 − 真值差| 的中位数和 95 分位，以及终止事件被模型判错的个数（成功与耗尽/超时互判）。
- **V7 部署：** 两臂的 5k 和 10k 检查点在 292500–292531 上做确定性评估（`experiments.t1b_v2 eval --interface v7 --openings 292500-292531`），给出捕获数、平均回报、首选分布及与真值最佳的一致数。

## B. 离线 critic 试验台（V7 结束后；在 `6fa8be8` 检出）

```powershell
git fetch origin claude/sac-mpc-coupling-design-ns7g6i; git checkout --detach 6fa8be8be2c1fdf8c7d1fe2dc9e9dcac7761bf94
foreach ($a in "e_sac","e_tcpf") { foreach ($m in 5000,10000) {
  & $py -B -m experiments.t1b_critic_bench --checkpoint "<V7 运行目录>\TRAIN_${a}_s0\checkpoints\step_$m.zip" `
      --replay "<V7 运行目录>\TRAIN_${a}_s0\replay_latest.pkl" `
      --truth "evidence\T1B_V1_AUDIT\candidate_truth_scan,evidence\T1B_V2_PILOT\scan_292120,$Q\truth" `
      --states "evidence\T1B_V1\episodes\interface_0k,evidence\T1B_V3_MIDTERM_20261019\eval_e_sac_2000,<A 部分 V7 评估目录>" `
      --updates 3000 --report-every 500 --probes 40 --output "$Q\bench_${a}_$m.json"
} }
```

- `replay_latest.pkl` 是 RecordV1 在最后一个检查点保存的 Replay；如果 10k 的 Replay 与 `step_10000.zip` 的相位不同，在报告中注明，不要拼接。
- 报告内容：
  - 三对梯度余弦（真实意图 vs MPC 优势、转移 TD vs MPC 优势、转移 TD vs 真实意图）的中位数与 10 / 90 分位；
  - 三种离线损失组合（只用真实意图 / 真实意图 + MPC / 在线 V7 损失）训练 3000 次后的曲线：最佳候选一致数、两两方向一致数、Q 贪婪选择的平均遗憾、Q 每档平均差。

## 交付

collab `docs/collaboration/lower/handoffs/t1b_v7_side_bench_20261020/`：`SIDE_BENCH_RESULT.md`。
科学分支 `evidence/T1B_V7_SIDE_BENCH/`：`truth/*.json`、`model/model_scan.jsonl`、A 部分评估 JSON、`bench_*.json`。每个文件单独 `git add -f`，并附 SHA256SUMS。

## C. 292126 专项审查（研究层 2026-10-21；只读，不训练；用 V7 本机资产）

背景：已有固定后续真值（`evidence/T1B_V2_PILOT/scan_292120/scan_292126_1.json`）表明，292126 首选 120 s、计划丢失后按最早候选重选，结局是超时（−137.0）。而 V7 E-TCPF 10k 同样首选 120 s，却成功了（+45.6）。所以成功很可能来自计划丢失后的重新决策，而不只是首次承诺。需要用轨迹确认。

1. 取 V7 两臂 10k 在 292126 上的确定性评估轨迹（eval 目录中的 `*_292126.json` / `.trace.jsonl`），逐条列出以下内容。只写事实，不推断。
   - 每次学习决策：时间、候选、选中序号、`has_plan`；
   - 每次计划丢失或重规划事件：时间、原因；
   - Δv 与剩余预算、剩余时间的时间曲线；
   - 终止时刻和终止事件。
2. 对 E-TCPF 的 5k 和 10k 检查点，在 292126 的每个学习决策状态（从轨迹中取 `observation_before`），用 `experiments.t1b_v2 audit` 的同样方法输出候选概率、Q1/Q2、Q 最佳候选和 Actor 确定性选择，以确认 Q 与 Actor 是否不一致。
3. 判定依据：
   - 首次决策之后再没有学习决策 → 收益归因于首次承诺；
   - 后续还有学习决策，而且选的不是最早 → 收益来自恢复决策。这类决策没有 MPC 配对（V7 只在首次决策配对），属于策略泛化，应如实标注。

## 执行顺序（研究层 2026-10-21 意见，上层同意）

先完成 A（包括研究层登记的 293100–293131 独立开发块）和 C。种子 1 按 `T1_B_V7_ROUND_20261020.md` 第 3 节，在 A 完成后再启动；资格已于 `35c8f79` 满足。只有在 C 查出评估或技术伪影（例如确定性评估不可复现）时才暂缓，届时上报。B 在种子 1 结束后做。期间不改 V7 代码。
