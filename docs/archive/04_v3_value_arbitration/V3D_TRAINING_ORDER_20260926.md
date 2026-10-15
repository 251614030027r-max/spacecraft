# v3d 正式训练指令（v3c 在 QP 门停止、背面半球闸门修复之后）

> **已被 `docs/V3D_EXECUTION_ORDER_20260926.md` 取代，从未开跑。** 开跑前的全面审查又发现并根治了多个问题（见 `docs/V3D_REVIEW_20260926.md`），训练代码提交随之更新。

*2026-09-26，上层窗口。分支 `claude/sac-mpc-coupling-design-ns7g6i`。**训练代码提交：`7555185346fc120d86ccf78f35705f801bbb6ff8`**（之后只有文档、证据与一个只读计数脚本的提交，不改训练路径）。全套测试 312 passed、3 xfailed。*
*本指令取代 `docs/V3C_TRAINING_ORDER_20260925.md`；第 1–7 节的流程与 v3c 相同，只改了代码提交与运行名。*

---

## 0a. v3c 为什么停、改了什么

v3c 学得很好（最近 100 回合训练完成：262420 81、262421 55、262422 59），参考跳变违例始终为 0。
262422 在约 22k 时触发了预注册的 QP 健康门（最近 100 回合零推力回退率 1.32 × 10⁻³ > 1 × 10⁻³），按规则停止。

**原因（沙箱复现；产物 `eval/adp/v3c_262422_qp_burst_diagnosis.json`）**

1. **回退是成批出现的，三个种子都一样。** 三个 monitor 里，86%–92% 的零推力步来自少数几个"≥ 20 步"的回合。
   262420 也有一个 104 步的回合，262421 有 39 步与 68 步的回合，只是它们没有凑够门槛。**所以这是三个种子共有的缺陷，262422 只是运气最差。**
2. **定位（用 262422 的 15k / 20k checkpoint 跑 72 个随机策略回合，共 163,420 控制步，回退 113 步）。**
   108 步来自同一个回合（15k，种子 264038）。**这 108 步全部在目标的"背面"**：
   - 在对接口平面之后（port 轴向 −4.4 → −3.0 m）；
   - 距离从 6.65 m 滑到 4.42 m；
   - 承诺 c = 0，参考停在 6.5 m 下限上；
   - 零推力期间追踪器一路滑进来，约 10 s 后以视场失败结束。
3. **机理。** MPC 的"预测终端"闸门只看两件事：离目标 ≤ 6 m，轴向 < 4.5 m。它**没有排除目标背面**。
   走廊是从对接口向前张开的锥，在背面没有几何意义，但闸门仍在背面 6 m 以内启用走廊约束。
   在回退起点，最差的走廊行需要 **2.94** 个单位的松弛，上限是 **2.0**（终点处 2.04），所以 QP 无解，零推力回退接连触发。
   真实判定在锁存之前从不考核走廊，所以这些约束管的是一块任务本身不打分的区域。
   这与 `docs/TERMINAL_GATE_DEFECT.md` 修过的是同一类缺陷：那次补的是距离上限，这次补背面。
4. **Pure MPC 为什么不触发**：它的参考始终在走廊轴上。实测 48 个评估回合里，它在背面离目标最近 10.13 m，从来没进过背面 6 m 以内（见下方计数）。

**修复（`7555185`，改 MPC 的约束闸门，属于"具名代码缺陷"）**：port 轴向 < 0（对接口平面之后）且未锁存时，不启用终端约束行。锁存之后的行为不变。新增 1 个测试。

**Pure MPC 不受影响（证据）**
- **逐次调用计数（决定性证据）**：在 48 个 Pure MPC 评估回合（262000–262047）里，对闸门的每一次调用，同时用修复前的规则再判一次。
  共 **2,444,785** 次调用，判定不同的次数为 **0**。Pure MPC 在背面离目标最近也有 **10.13 m**，从来没进过 6 m 以内的背面区域。
  所以 Pure MPC 的每一次求解看到的约束行与修复前完全相同。这个结论与平台无关（脚本 `experiments/count_pure_mpc_gate_differences.py`，计数在第 6 节的产物里）；
- 沙箱里修复前后同一回合对照（262012、262013），逐控制步推力/力矩逐位相等（沙箱自查，未入库）；
- 与用户机器上的 `eval/adp/pure_mpc_nominal.json` 比，48 回合的完成（36/48）、步数、存活时间全部相同。等效 Δv 最大差 3.4 × 10⁻⁷，这是 Windows 与沙箱的浮点差异：在同一台机器上，修复前后逐位相同（沙箱自查，未入库）。

**没修的一处（记录，不阻塞）**：20k checkpoint 另有一个 5 步的小回退段（种子 264005），在对接口平面**之前**、偏离走廊轴约 65°、距离 5.0–5.4 m。
这里最差走廊行需要 1.55 单位松弛（在上限内），无解来自多行叠加。它占 72 回合回退的 5/113。要修它得收窄正面的闸门，可能改变 Pure MPC，所以本轮不动，交给训练中的 QP 健康门继续盯。

**v3c 运行目录处置**：
- `v3c_262422` 已停，加侧车 `STOPPED_QP_GATE.md`（已有）；
- **`v3c_262420`、`v3c_262421` 现在也停**。它们没有违规，但：（1）三个种子共有同一缺陷，只是没触发门槛；（2）正式结论要求三个种子在同一版代码上训练；
  （3）修复改变了学习分支在背面近场的动力学，旧运行的后半程与新代码不可比。
- 三个目录原样保留（monitor、manifest、checkpoint），262420 / 262421 各加侧车 `SUPERSEDED_GATE_FIX.md`，写明"修复 7555185 之前的运行，仅作接口可学性参考，不续训、不作正式结论"。

**v3c 的学习曲线仍然有价值**：它说明 T0b + 近场下限的接口在三个种子上都学得起来。这可以作为开发记录引用，但不进论文结果表。

---

## 1. 预检（开跑前，在仓库根目录，PowerShell）

```powershell
git fetch origin claude/sac-mpc-coupling-design-ns7g6i
git checkout claude/sac-mpc-coupling-design-ns7g6i
git pull --ff-only origin claude/sac-mpc-coupling-design-ns7g6i
git rev-parse HEAD                 # 训练代码提交为 7555185346fc120d86ccf78f35705f801bbb6ff8；若 HEAD 更新，git diff 7555185346fc120d86ccf78f35705f801bbb6ff8 HEAD --stat 只能出现 docs/、eval/ 下的文件和 experiments/count_pure_mpc_gate_differences.py
git status --porcelain --untracked-files=no   # 必须为空
python -B -m pytest -q tests/test_terminal_gate.py tests/test_v3_task_interface.py tests/test_check_v3b_training_health.py
```

测试必须全绿；工作树必须干净（manifest 会记录 `code_dirty`，不干净的运行作废）。
旧运行 `v3_*`、`v3b_*`、`v3c_*` 都保持原样，**不续训、不复用运行名、不加载其 checkpoint**。

## 2. 启动（三个 PowerShell 窗口，一个种子一个）

```powershell
$env:OMP_NUM_THREADS="1"; $env:MKL_NUM_THREADS="1"; $env:OPENBLAS_NUM_THREADS="1"
python -u -B -m train.train_hybrid --steps 60000 --seed 262420 --run-name v3d_262420 --horizon 35 --parametrization task_state_v3 --adaptive-task --device cpu
```
```powershell
$env:OMP_NUM_THREADS="1"; $env:MKL_NUM_THREADS="1"; $env:OPENBLAS_NUM_THREADS="1"
python -u -B -m train.train_hybrid --steps 60000 --seed 262421 --run-name v3d_262421 --horizon 35 --parametrization task_state_v3 --adaptive-task --device cpu
```
```powershell
$env:OMP_NUM_THREADS="1"; $env:MKL_NUM_THREADS="1"; $env:OPENBLAS_NUM_THREADS="1"
python -u -B -m train.train_hybrid --steps 60000 --seed 262422 --run-name v3d_262422 --horizon 35 --parametrization task_state_v3 --adaptive-task --device cpu
```

- 超参全部用默认值，与 v3c 完全相同，**不加任何其他参数**；
- 种子仍用 262420/421/422：**不因 v3c 的表现换种子**；
- checkpoint 每 5,000 步自动保存；训练期间窗口不能关，系统不能睡眠。

**开跑 1 分钟后检查 manifest**（每个运行各一次）：
```powershell
python -c "import json,sys;m=json.load(open(sys.argv[1]));print({k:m.get(k) for k in ('code_commit','code_dirty','observation_dimension','action_dimension','training_branch','waypoint_parametrization','seed')})" logs/v3d_262420/manifest.json
```
必须是：`code_commit` 等于第 1 节的 HEAD，`code_dirty` 为 False，`observation_dimension` 42，`action_dimension` 2，`training_branch` "learned"，`waypoint_parametrization` "task_state_v3"。任何一项不对，立刻停掉该运行并报告。

## 3. 训练中的健康检查（预注册，停止规则不变）

在 **10k、20k、30k、40k、50k** 附近各跑一次（只读）：
```powershell
python -B -m experiments.check_v3b_training_health logs/v3d_262420 logs/v3d_262421 logs/v3d_262422
```

| 规则 | 条件 | 动作 |
|---|---|---|
| **S1 参考跳变** | 任一回合 `reference_jump_violations > 0` | 停止该运行并报告 |
| **S2 QP 健康** | 最近 100 个回合零推力回退率 > 1 × 10⁻³ | 停止该运行并报告 |

门槛不变（1 × 10⁻³）。**不因为修过一次就放宽**。停止时按 v3c 的做法打包：monitor、manifest、最近两个 checkpoint、侧车说明。

完成率、回报、最大参考跳变只做记录，不是停止依据；不改超参、奖励，不丢种子，不挑 checkpoint，不续训。

## 4. 训练完成后

1. 三个运行目录都有 `final_model.zip`，manifest 中 `status` 为 completed、`actual_decision_steps` 为 60000；
2. 最后跑一次第 3 节的健康检查；
3. 按 `docs/V3_METHOD_EXECUTION_ORDER_20260924.md` 的 **M2 → M3 → M6 → M5** 继续，该执行单不变。
   M2 数据与 V_L / V_B 的输入一律用 42 维观测；评估器用 manifest 里的 `evaluation_flags`，不要手写。
4. **最终 Pure MPC 行在 `7555185` 或之后的提交上干净重跑一次**（论文数字只用这次重跑），与第 0a 节的"逐次调用计数为 0"一起报告。

## 5. 交回内容（训练结束时）

1. 三个 manifest（完整文件）；
2. 至少 10k、30k、60k 三次健康检查的输出；
3. 每个种子末 100 回合的完成数、平均回报、最大参考跳变、QP 回退率，以及"≥ 20 步回退的回合"列表（回合序号、步数）；
4. 有无停止事件，若有，附当时的 monitor 与报错；
5. 三个 `train.monitor.csv` 的 SHA-256（模型 ZIP 不入库，只报路径与 SHA-256）。

## 6. 证据

| 产物 | SHA-256 | 内容 |
|---|---|---|
| `eval/adp/v3c_262422_qp_burst_diagnosis.json` | `10df2f1352786c5707d4d31b69c998999301282ae6d3a8568b55a5504d5e6933` | 三个 v3c monitor 的回退集中度；262422 15k/20k 的 72 个随机回合；背面走廊行的松弛需求；Pure MPC 48 回合闸门判定差异计数 |

生成：`python -B -m experiments.analyze_v3c_qp_bursts`（输入：`experiments.probe_v3_qp_policy` 的逐回合输出、诊断包里的 monitor、`experiments/count_pure_mpc_gate_differences.py` 的 4 个计数文件）。
诊断包 `V3C_262422_QP_DIAGNOSTIC_HANDOFF_20260926`（用户机器）内的 SHA-256 见其 `SHA256SUMS.txt`；其中 `v3c_262422_train.monitor.csv` 为 `A8EB7AB1…7390`。

## 7. 禁止事项

- 不续训旧 `v3_*`、`v3b_*`、`v3c_*` 运行，不复用旧运行名，不加载旧 checkpoint；
- 不改奖励、MPC（本次修复之外）、限速上界、任务参数、超参；
- 不因训练中的完成率调整任何东西；
- 工作树不干净时不开跑。
