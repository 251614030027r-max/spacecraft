# V3 T0：接口修复执行单（给下层）

*2026-09-24，上层窗口。基线：分支 `claude/sac-mpc-coupling-design-ns7g6i` 当前 HEAD。*

**性质：修 bug + 搭骨架，不是方法创新，不训练。** 与 P2（上层沙箱在跑）并行，互不依赖。
依据：`docs/V2_EVALUATION_REPORT_20260924.md` 与 `docs/COUPLING_DESIGN_PROPOSAL_20260924.md` 第 0 节中的三个接口缺陷。

---

## 总原则

1. **新增参数化 `task_state_v3`，不改 `task_state_v2`。** V2 的三个模型与评估产物必须仍能逐位复现，现有 V2 测试一条不改。
2. **全拒 / 基线分支逐位等于 Pure MPC**，是硬约束，每项改动后都要重跑现有地板测试。
3. 每项一个提交，带测试；先写红测试再实现。
4. 不改 reward、MPC、任务参数、`v2_reference_step_max_m` 的默认值。

---

## T0.1 动作改成增量（修"全速进 / 全速退"）

**现象**：V2 策略提议绝对 ρ，被截成 ±0.4 m。提议与当前值只要差超过 0.4 m（约占动作范围 2%）就走满一步，末段来回换向 42–61%。

**实现**（`env/hybrid_env.py`，仅 `task_state_v3`）：
```
a ∈ [-1,1]^2
δρ = a0 · task_progress_advance_limit_m      (a0 ≥ 0)
     a0 · task_progress_retreat_limit_m      (a0 < 0)
δc = a1 · task_commit_advance_limit          (a1 ≥ 0)
     a1 · task_commit_retreat_limit          (a1 < 0)
z_candidate = clip(z + [δρ, δc])
```
之后照旧做度量上限（沿用 V2 的 8 步几何二分，上限 0.40 m）。
**a = 0 表示任务状态不动**，不是 Pure MPC（Pure MPC 是独立分支，见 T0.4）。

**ρ 的上界改为正好落在抓取位姿**：`ρ_max = hold_radius − |desired_position|`（当前是 `hold_radius − 2.5`，比抓取位姿多 0.5 m）。
这样 `z = (ρ_max, 1)` 对应的参考**严格等于** Pure MPC 的参考 `desired_position`，为 T0.4 的连续切换提供条件。

**测试**：
- `a = 0` 连续 10 步，施加状态不变；
- `a = (+1, +1)` 一步，δρ = 0.5、δc = 0.05（在度量上限内时）；
- `z = (ρ_max, 1)` 时参考与 `desired_position` 的差 < 1e-12。

---

## T0.2 方向插值不跨反向点跳变（修"参考一步跳 20 m"）

**现象**：惯性暂存方向在目标系里随翻滚转动；转到与抓取方向约 175° 时，最短弧 slerp 翻到另一侧。
262004 在 t = 80 s 一个决策跳 15 m 再跳 20 m，113 s 飞出 35 m（沙箱逐步重放复现，与记录逐位一致）。
262410 与 262412 在同种子同时刻同样大跳。

**实现**：把"最短弧"改成**带记忆的旋转轴插值**（上层已在沙箱验证，参考实现
`experiments/probe_reference_continuity.py` 的 `MemoryAxisBlend`，直接照搬到 env 里）：
```
w    = h − d
n_0  = normalize(h × d)                      # 第一个决策：普通测地线
n_k  = normalize(n_{k-1} − (n_{k-1}·w)/(w·w) · w)
       # 所有满足 (h−d)·n = 0 的轴都能把 h 转到 d；取离上一个轴最近的那个
h⊥ = h − (h·n_k) n_k ,  d⊥ = d − (d·n_k) n_k
θ_k  = atan2(n_k · (h⊥ × d⊥), h⊥ · d⊥)，相对 θ_{k-1} 解卷绕
dir  = Rodrigues(h, n_k, c · θ_k)            # c = 1 时严格等于 d
```
每回合 reset 时清空 `n`、`θ` 记忆。

> **上层先试过一个更简单的方案，失败了，不要用**：只用端点定义轴 `normalize(h × d)`，再做符号连续化。
> 在 262004 上仍然跳 20 m。原因是 h 接近 −d 时这个轴会快速旋转，而不是翻转符号，符号连续化管不到。

**沙箱实测**（记录的 V2 施加任务状态逐步重放，产物 `eval/adp/v3_t0_reference_continuity.json`）：

| 回合 | V2 最大相邻跳变 | 带记忆轴 | 暂存方向自然转动 |
|---|---:|---:|---:|
| 262410 / 262004 | 20.00 m | **1.68 m** | 1.69 m |
| 262412 / 262004 | 21.39 m | **1.68 m** | 1.69 m |
| 262411 / 262034 | 7.41 m | **2.45 m** | 1.79 m |
| 262410 / 262010 | 3.60 m | **1.34 m** | 1.15 m |

**测试**：
- 回归：上表 4 个回合的施加任务状态在 v3 映射下逐步生成参考，相邻差最大值 < 3 m；
- 构造测试：h 在目标系中匀速转过 d 的反向点（覆盖 170°–190°），c 取 0.3 / 0.5 / 0.7，相邻差 ≤ 自然转动 × 1.2 + 0.40 m；
- c = 1 时方向与 d 的差 < 1e-9；c = 0 时方向等于 h。

---

## T0.3 跨决策参考跳变的监测（不是再加一层截断）

T0.2 让映射在时间上连续之后，"同一时刻比较 + 度量上限"就足以限制跳变。这里只**记录**，用于验收和评估：

- 每个决策记录 `reference_jump_target_m = |r_{k+1} − r_k|`（目标系）和 `reference_jump_inertial_m`（惯性系）；
- 评估输出增加每回合最大值与 p95。

**验收**：V3 接口下用脚本臂（例如恒定 `a = (+0.5, +0.5)`）跑 8 回合，目标系最大跳变 < 3 m。

---

## T0.4 显式分支与切换（Pure MPC 是独立分支）

**状态**：新增 `mode ∈ {baseline, learned}`，以及 `z`、上一决策施加的参考 `r_last`。

**新接口**（沿用 V2 的 `step_with_proposal` 思路，改为显式选分支）：
```python
env.step_with_branch(action, branch="learned" | "baseline")
```
- `baseline`：施加 `desired_position`，逐位等于 Pure MPC；
- `learned`：按 T0.1–T0.2 施加 `r(z)`。

**切换规则**（这两条都是修 bug，不是新机制）：
1. **learned → baseline**：切换那一刻调用 `self.controller.reset()`。
   原因：下层用"相邻两次参考之差 / 2 s"估计参考速度。不重置的话，十几米的参考跳变会变成每秒数米的前馈速度指令。
   重置后，含义就是"从当前状态重新启动 Pure MPC"，与 Pure MPC 从初始状态启动完全同构。
   上层沙箱的 P2 rollout 已按这个定义实现（`experiments/probe_base_policy_rollout.py`）。
2. **baseline → learned**：`z` 初始化为 `(ρ_max, 1)`，对应的参考就是 Pure MPC 当前的参考，**切换瞬间参考不动**。
   策略之后可以从这里后退或保持。这一条依赖 T0.1 的 ρ 上界修改。

**测试**：
- 全程 `baseline`：逐控制步 wrench 与 Pure MPC 相等（复用现有地板测试写法，12 回合）；
- 第 k 个决策从 `learned` 切到 `baseline` 之后，controller 的内部状态与新建 controller 的状态逐字段相等；
- 从 `baseline` 切到 `learned` 的第一个决策，施加的参考与切换前相同（差 < 1e-12）；
- 切换计数写进 info（`hybrid_branch`、`hybrid_branch_switches`）。

---

## T0.5 观测

`task_state_v3` 的观测在 V2 的基础上加一维 `mode`（0 = baseline，1 = learned）。
任务状态 `(ρ/ρ_max, c)` 保留。维度守卫照旧：用错维度加载模型要拒绝。

---

## T0.6 评估器与记录

`experiments/evaluate_hybrid_policy.py` 支持 `--parametrization task_state_v3`，并逐决策记录：
`branch`、`task_state_applied`、`task_action_raw`、`reference_jump_target_m`、`reference_jump_inertial_m`、`branch_switches`。
沿用 V2 已有的 `control_wrenches` 记录方式，用于地板检查。

---

## 验收清单（全部满足才交回）

1. 现有全套测试仍绿：285 passed、3 xfailed，外加 V2 评估分支带来的新测试；
2. 新测试全部通过（T0.1–T0.5 列出的每一条）；
3. 262004 回归：v3 映射下相邻参考差最大值 < 3 m；
4. 全程 baseline：12/12 回合逐步 wrench 数组与 Pure MPC 相等；
5. 脚本臂 8 回合：无 NaN，无 QP 无解异常，目标系最大参考跳变 < 3 m。

**交回格式**：提交列表 + 每条验收的结果 + 测试输出的最后 5 行。**不训练、不做正式评估。**

## 不做的事

- 不实现仲裁器、Q_b 评估器、option 执行，这些等 P2 通过后另发执行单；
- 不改 `task_state_v2` 的任何行为；
- 不加驻留时间、干预惩罚等机制。
