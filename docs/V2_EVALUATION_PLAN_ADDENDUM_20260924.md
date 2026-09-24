# V2 评估方案补充说明（两处读法修正，不改任何跑动）

*2026-09-24，上层窗口。对应 `docs/V2_EVALUATION_PLAN_20260923.md`。基线 HEAD `7d850d5`。*

**本文只改"怎么读、怎么报"，不改任何命令、参数、模型、判据阈值以外的东西。**
**本文是纯文档提交；若评估正在跑，不要为了读它而 `git pull`**（会改变产物指向的 HEAD），
让用户直接转发内容即可。

---

## A1｜第 2 步 `max|Δwrench|` 的计算方法（原方案没写，必须补）

### 问题

原方案要求 `max|Δwrench| = 0.0`，但 `experiments/evaluate_hybrid_policy.py`
只输出逐回合**汇总量**（力/力矩范数统计、零推力计数），**不输出逐步 wrench**。
汇总量相等不等于逐步逐位相等。单测 T4
（`tests/test_v2_task_interface.py::test_t4_reject_all_proposals_is_bitwise_pure_mpc_closed_loop`）
的口径是**逐控制步 `np.array_equal`**，正式复验必须同一口径。

### 做法（按你当前进度选一条）

| 你的进度 | 动作 |
|---|---|
| 第 2 步还没跑 | 按下面"实现"加逐回合 wrench 哈希，再跑第 2 步 |
| 第 2 步已跑，且已经是逐步比较 | 在报告里写明比较方法（逐步数组 / 哈希 / 其他），不用重跑 |
| 第 2 步已跑，但只比了汇总量 | 按"实现"补哈希，**重跑第 2 步 3×4 回合 + 基线抽检 4 回合**（约 15 分钟），两种结果都报 |

### 实现（建议，约 10 行）

在 `evaluate_hybrid_policy.py` 取得每步 `wrench` 的位置（当前约 867 行
`wrench, diagnostics = env.controller.command(...)` 之后）：

```python
# 回合开始
wrench_hasher = hashlib.sha256()
wrench_step_count = 0
# 每个控制步
wrench_hasher.update(np.ascontiguousarray(wrench, dtype=np.float64).tobytes())
wrench_step_count += 1
# 写入 record
record["wrench_sha256"] = wrench_hasher.hexdigest()
record["wrench_step_count"] = wrench_step_count
```

- **所有**评估路径都记（Pure MPC 抽检、第 1 步、第 2 步），字段对非 V2 参数化同样存在；
- 加一个测试：同种子跑两次，哈希相同；改一步动作，哈希不同。

### 判读（替换原第 2 步的判读表）

对比对象：第 0.2 步基线抽检 `eval/adp/spot_pure_mpc_nominal.json`（种子 262000–262003，
与第 2 步同一种子块）。按 `record["seed"]` 对齐。

| 观测 | 结论 |
|---|---|
| 3 模型 × 4 回合，`wrench_sha256` 与 `wrench_step_count` **全部**等于基线 | 架构地板逐位成立，报 `max|Δwrench| = 0.0（按哈希，12/12）` |
| 任何一个不等 | **立即停，报上层**。这是 bug，不是结果。附上首个不等的回合种子 |

> 注意：强制全拒时策略输出被忽略，所以 V2 的观测 flag（`--phase-time-observation`
> 等）不影响 wrench；两边 wrench 不同只可能来自下层或参考构造的差异。

---

## A2｜Q2 的"是否顶边界"必须把 ρ 和 c 分开读

### 问题（代码推理，未经跑动验证）

- **c 的上界 1 就是捕获方向本身**。完成时位置要落在捕获位姿 0.25 m 内，
  方向必须基本对准，所以**完成回合 c ≈ 1 是预期，不是"撞墙"**。
- **ρ 的上界不是捕获位姿**。捕获位姿半径 3.0 m（`desired_position_target_m = (-3,0,0)`），
  而 ρ 上界 `v2_progress_max_m = hold_radius − 2.5` 对应的参考半径是 **2.5 m**，
  比捕获位姿还近 0.5 m（`env/hybrid_env.py` `v2_radius_min_m`、`reference_for_task_state`）。
- 所以原表"ρ/c 是否顶边界"：若按"ρ **或** c"读 → 几乎全判 B；
  按"ρ **且** c"读 → 几乎全判非 B。**两种读法都没有区分力。**

### 修正后的读法（只改报告字段，不需重跑，前提是 00.1 已逐决策记录 ρ、c、`v2_progress_max_m`）

对每个**完成**回合，取完成前最后 20 个决策，定义

- `ρ_max = v2_progress_max_m`（该回合自己的值）
- `ρ_pose = v2_progress_max_m − 0.5`（参考正好落在 3.0 m 捕获位姿的 ρ）

逐回合报：

| 字段 | 定义 |
|---|---|
| `step_last20_median_m` | 最后 20 决策参考步长中位数 |
| `rho_gap_to_pose_m` | 最后一个决策 `ρ − ρ_pose`（负 = 没推到位姿，正 = 推过了位姿） |
| `rho_at_max` | 最后一个决策 `ρ ≥ 0.99·ρ_max`（是/否） |
| `c_last` | 最后一个决策的 c（**只记录，不作为边界判据**） |

判读（替换原 Q2 表）：

| 观测 | 结论 |
|---|---|
| 步长 ≤ 0.05 m，`|rho_gap_to_pose_m|` ≤ 0.25 m，`rho_at_max = 否` | **A：策略自己停在了捕获位姿**（学会收手） |
| 步长 ≤ 0.05 m，`rho_at_max = 是` | **B：参考推到接口端墙、越过位姿，靠下层把它停住** |
| 步长始终 ≥ 0.3 m | 饱和与完成互斥被确认 |
| 其他 | 如实列出分布，不归类 |

报三个种子各自的 A/B/其他 回合数，**不合并**。

---

## 交回格式的对应改动

第 6 节第 2 条改为：

```
2. 架构地板：比较方法 = 逐步哈希 / 逐步数组 / 仅汇总量（后者须补跑）
   12/12 哈希一致 = 是/否 ；max|Δwrench| = ___
```

第 6 节第 3 条的 Q2 行改为：

```
Q2（每个种子分别）：A __ / B __ / 其他 __ 回合；
   rho_gap_to_pose_m 中位 ___ ；rho_at_max 占比 ___ ；c_last 中位 ___
```

其余条目不变。
