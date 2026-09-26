# 耦合缺陷诊断执行单（下层窗口）— v2

> 日期：2026-09-21。分支 `claude/sac-mpc-coupling-design-ns7g6i`。
> 性质：**诊断单**，不是方法设计单。目标是用最少的机时**把根因判定到唯一**，然后进入优化阶段。
> **不改方向、不改任务参数、不改 reward、不调超参、不换算法、不重训。**
> 主线（自适应同步—进入 + 强 MPC + 基线锚定耦合）与 2026-09-18 全部锁定裁定继续有效。
>
> v2 相对 v1 的变化：新增 **D0（训练期可达性）** 与 **D4（脚本化暂存臂）**；修正 D2 的判据
> （v1 的判据会把"探索不可达"误判为"表达不够"）；把最贵的 D2 降级为条件执行；新增
> **第 5 节根因判定表**作为诊断的出口条件。

---

## 0. 这一轮到底测到了什么

已由 `ADAPTIVE_TWO_MODEL_INTERIM_REVIEW_20260921.md` 与代码核查共同确认：

- 有效介入 **0.114%（262410，4/3509）**、**0.484%（262411，17/3509）**；
- 因此本轮 nominal 三行的实际内容是 **Pure MPC 的重复**，不是被激活的高层策略。

**两个不得越过的否定**（写任何报告、任何幻灯片都不得违反）：

1. **不能说"耦合方法无效"。** 它没生效，所以有效性未知。
2. **不能说"已验证基线保持"。** retained 36 / rescued 0 / destroyed 0 配 0.1%–0.5% 的介入率，
   是**靠不作为**得到的；而"残差 0 逐位等于 Pure MPC"本来就有单测钉着
   （`tests/test_baseline_residual.py`）。**那是恒等式，不是实验结果。**

---

## 1. 缺陷机理（已在代码中逐行核实）

四条，前三条是结构，第四条是 v2 新增、也是最关键的一条。

| # | 位置 | 事实 |
|---|---|---|
| ① | `env/hybrid_env.py` 残差映射 | `a_commit = clip(1 + 2r)`。**所有 `r ≥ 0` 映射到同一个 `a_commit = +1`**，残差 commit 通道整整一半塌成一个点 |
| ② | `_arrival_condition_waypoint` 棘轮 | `blend = max(self._commit_blend, blend)`；`blend` 到 1 后走 `blend >= 1.0` 分支**直接 return 捕获位姿**。此后负残差被 max 吃掉，`a_radius` 被整条绕过——**锁存之后整个 2D 动作空间 100% 惰性**，不是一半 |
| ③ | `experiments/evaluate_hybrid_policy.py` 部署门 | 回退动作 `np.zeros_like(action)` 即 `r = 0` → `a_commit = +1` → `blend = 1` → **锁存**。门在策略首次暂存前回退过**一次**，暂存能力即永久消失 |
| ④ | `train/train_hybrid.py`（**v2 新增**） | **训练时没有部署门**（无 `--deployment-gate` 参数，门是纯评估侧的）。两个后果：**(a)** critic 从未被训练去做门要它做的那个比较，`Q(s,0)` 是离策略分布动作（2026-09-16 已自标此隐患），门的"本步安全回退"带不可逆模式副作用，两件事被混在一个动作里；**(b)** 更要紧——训练期探索是**随机采样**，tanh-高斯在 `r ≥ 0` 上有相当大的概率质量，**任一次正采样即锁死该 episode**，所以**大量训练回合本身就是 Pure MPC**，暂存分支在探索中可能近乎不可达 |

**④(b) 是 v1 漏掉的东西，它直接改变诊断的读法。** 如果暂存分支在训练中不可达，那么"策略没学到
暂存"既不是表达不够、也不是任务没决策，而是**没有数据可学**。三者修法完全不同，必须先分开。

---

## 2. 立即执行

### 2.1 停掉 262412 的 48-seed 评估

吸收态是代数结论不是概率结论（`max` + 零残差映射到 `blend = 1`），第三行几乎必然再给一行
Pure MPC；"三种子一致"由 **D1** 用更强的方式覆盖；且语义一修这一行必然重跑。

- 停进程，**保留已产出的 `*.json.partial`**，归档备查；
- 停止记录写明：**主动停止，理由是该行信息量已被 D1 取代**——不是失败、不是丢数据。

### 2.2 推送本地 3 个提交

评估基点 `32fe20ef` **不在远端**（远端 tip `4206794`），即这份决定框架走向的报告，其数字指向
一个他人不可见的提交，违反"任何进入决策的数字必须指向可复现 artifact"。另：你侧
`263 passed`、远端 `258`，差额应为新增的评估记录测试，一并推上来对齐。

```
git push -u origin claude/sac-mpc-coupling-design-ns7g6i
```

### 2.3 禁止事项

- **不删棘轮。** 它为一个实测原因而加：没有它时 0.09 的脚本动作噪声让 **8/8 能完成的种子
  变成 0/8**。直接砍掉是已知回归。
- **不开跨翻滚率大表。** 有效介入 0.1%–0.5% 时它测的是 Pure MPC，与自适应命题无关。
- **不改 reward、不调超参、不换算法、不重训。** 当前问题在接口，不在学习算法。
- **不在拿到第 5 节唯一根因之前动手设计新机制。** 旧任务（Waypoint / 24D）已经为此栽过一次。

---

## 3. 需要先补的工具能力（纯测试脚手架，不是方法改动）

这三条是 D0/D4 的前置，改动都很小，各自独立提交。

**C1｜`evaluate_hybrid_policy.py` 加 `--stochastic-policy`**
当前 `policy.predict(..., deterministic=True)` 硬编码。D0 需要按训练时的随机策略滚动，
才能估出训练期的锁存分布。加一个开关，默认仍是确定性。

**C2｜`evaluate_hybrid_policy.py` 加 `--control staged_residual --stage-decisions N`**
脚本化上层：前 N 个决策强制 `r_commit = -1`（完整惯性暂存），之后 `r = 0`。
不经策略、不经门，**直接走真实的残差接口 + 棘轮**。
（`evaluate_hybrid_scripted.py` 的 `--parametrization` 只支持 `absolute / radial_local`，
不支持 `arrival_condition`，也没有 `--adaptive-task`，所以脚本化上层这条路在主线接口上
目前是断的。走 `evaluate_hybrid_policy` 的 `--control` 加一个模式最省事。）

**C3｜`replay_hybrid_policy.py` 补 `--adaptive-task` 与 `--baseline-anchored-residual`**
当前**无法回放这批模型**（观测 38D 对不上，残差映射也缺）。逐决策诊断工具不能对主线模型失效。

### 统一的仪表字段（D0/D1/D2/D4 都要记）

| 字段 | 含义 |
|---|---|
| `raw_action` | 策略输出的原始残差 `(r_commit, r_radius)`（你侧已有） |
| `applied_action` | 经门之后实际使用的动作（你侧已有） |
| `blend_before_ratchet` | `0.5·(a_commit + 1)`，**过 max 之前** |
| `blend_after_ratchet` | 过 max 之后，真正生效的 blend |
| `reference_changed` | 本步生效参考是否异于固定设定点参考（容差 `1e-9`） |
| `latch_step` | 本 episode 中 `blend_after_ratchet` 首次达 1.0 的决策序号；未达到记 `null` |
| `latch_cause` | `"policy"`（该步 `raw_action[0] ≥ 0`）或 `"gate_fallback"`（该步被门换成零残差） |

**`latch_step` 与 `latch_cause` 是本次诊断的核心量。**

---

## 4. 诊断任务

顺序即优先级。**D0 / D1 / D4 都很便宜，先跑这三个；D2 是条件执行；D3 独立，可随时并行。**

### D0 — 训练期可达性【最便宜，信息量最大，v2 新增】

**问题：训练的时候，暂存分支到底被探索到过吗？**

```
for S in 262410 262411 262412; do
  python -B -m experiments.evaluate_hybrid_policy --episodes 12 --seed 262000 --horizon 35 \
    --parametrization arrival_condition --adaptive-task \
    --phase-time-observation --execution-feedback --baseline-anchored-residual \
    --stochastic-policy \
    --model logs/adp_rf_${S}/final_model.zip \
    --output eval/adp/d0_stochastic_${S}.json
done
```

只读 `latch_step` 分布（中位数、四分位、=0 的比例）。

**预注册判据：**
- `latch_step` 中位数 **≤ 2** → **暂存分支在训练中近乎不可达，训练数据本身被污染**
  → 根因含 **R2**（见第 5 节）。这是当前最可能的结果。
- `latch_step` 中位数**明显 > 2** → 训练期确实见过暂存，数据不是主因 → R2 排除。

> 这是对训练期的**事后估计**（用最终策略的随机分布近似训练分布），**标记为估计**，
> 不要写成"训练期实测"。它的作用是定性分流，不是精确复现训练。

### D1 — 门控确定性滚动的锁存归因【便宜，确认机理】

```
for S in 262410 262411 262412; do
  python -B -m experiments.evaluate_hybrid_policy --episodes 8 --seed 262000 --horizon 35 \
    --parametrization arrival_condition --adaptive-task \
    --phase-time-observation --execution-feedback --baseline-anchored-residual \
    --deployment-gate --gate-advantage-margin 0.0 \
    --model logs/adp_rf_${S}/final_model.zip \
    --output eval/adp/d1_gated_instrumented_${S}.json
done
```

**预注册预测（先写下，再看结果）：**
- `latch_step` 中位数 = **0**；
- `latch_cause` 绝大多数为 **`gate_fallback`**；
- `reference_changed` 占比 ≈ **0.1%–0.5%**，与中期报告一致。

三条全中 → 机理确认，**三种子一致性由此成立，不需要第三个 48-seed 行**。
任一条不中 → **机理理解有误，停下报上层**，不要自行改设计。

### D4 — 脚本化暂存臂【一次测两件事，v2 新增】

**问题：这个接口能不能表达暂存（管路）＋暂存能不能赢回东西（价值）。**

只跑那 **12 个两边都失败的种子**（从 `eval/adp/nominal_pairing_two_model_interim.json`
取 both-failed 列表），不用跑 48 个。不经策略、不经门，所以结果干净。

```
# SEEDS=<从 pairing JSON 取出的 12 个 both-failed 种子>
for N in 5 15 30; do            # 暂存 10 s / 30 s / 60 s
  python -B -m experiments.evaluate_hybrid_policy --episodes 12 --seed 262000 --horizon 35 \
    --parametrization arrival_condition --adaptive-task --baseline-anchored-residual \
    --control staged_residual --stage-decisions $N \
    --output eval/adp/d4_staged_n${N}.json
done
```

**预注册判据：**
- 参考确实随 `N` 变化（`reference_changed` 显著 > 0、`latch_step ≈ N`）→ **管路通**，
  接口能表达暂存 → R3 的"表达"分支排除；
- 参考不随 `N` 变化 → 管路仍有问题，**先修管路再谈别的**；
- 管路通的前提下，**任一 `N` 救回 ≥1 个 both-failed 种子** → **决策存在，暂存有价值**
  → R4 排除，且**这就是学习层本该拿到的那个增量**；
- 管路通但三个 `N` 一个都救不回、且资源代价更高 → **R4 抬头**，需与 D3 交叉判读。

> 这是**脚本化对照**，不是 RL、不是理论最优、不是严格上界。它赢 = 暂存有潜在资源优势；
> 它不赢 ≠ RL 没空间——但**连脚本化都赢不到任何东西**，是对 R4 的强证据。

### D2 — 关门 48 种子【条件执行】

**先跑 D0/D1/D4。** 如果 D0 中位数 ≤ 2 且 D1 三条全中，D2 的结果基本已被预测，它从
**判别器**降级为**确认器**；此时是否值得 48×2 由上层裁决，**不要默认开跑**。

需要跑时：同一 48 种子块、同一模型，**只去掉 `--deployment-gate`**，其余一字不改，
带全部仪表字段。Pure MPC 行复用 `eval/adp/pure_mpc_nominal.json`，**不要重跑**。
成对四象限沿用产出 `nominal_pairing_two_model_interim.json` 的同一份配对代码，不要另写。

```
for S in 262410 262411; do
  python -B -m experiments.evaluate_hybrid_policy --episodes 48 --seed 262000 --horizon 35 \
    --parametrization arrival_condition --adaptive-task \
    --phase-time-observation --execution-feedback --baseline-anchored-residual \
    --model logs/adp_rf_${S}/final_model.zip \
    --output eval/adp/d2_nogate_nominal_adp_rf_${S}.json
done
```

**判据（v2 修正，必须与 D0 交叉读，不可单独判）：**

| 关门后（确定性） | 配合 D0 | 读法 |
|---|---|---|
| `latch_step` 中位数 > 0，`reference_changed` 明显上升 | 任意 | 策略**能**暂存，是门+棘轮抹掉的 → **R1 为主因** |
| 同上，且 rescued > 0 | 任意 | R1 加强：被抹掉的是**真实收益** |
| 同上，但 rescued = 0 且 destroyed > 0 | 任意 | 会介入但介入是坏的 → 奖励/表达问题（R3） |
| 同上，但 rescued = destroyed = 0 | 任意 | 介入了但无后果 → 动作权限不足（R3 的幅度分支） |
| `latch_step` 中位数 = 0 | **D0 中位数 ≤ 2** | **R2 为主因**：策略的均值动作就是 commit，因为它训练时几乎没见过暂存 → **不要**据此判"表达不行" |
| `latch_step` 中位数 = 0 | **D0 中位数 > 2** | 训练见过暂存却学成了立即 commit → **R3 或 R4**，须由 D4/D3 分开 |

**v1 的判据缺最后两行，会把 R2 误判成 R3。这是 v2 最重要的修正。**

### D3 — 决策裕度标定【最根本的空白，可随时并行】

**任务里到底有没有两种代价结构不同的合理策略——这件事我们从来没测过。**
runsheet §1 写的是"先标定再训练"，实际跳过了。不依赖任何模型。

```
mkdir -p eval/cal2
for R in 0.10 0.20 0.30; do          # 1.18 / 2.36 / 3.54 deg/s
  for ARM in desired_pose timed_entry; do
    python -B -m experiments.evaluate_hybrid_policy --episodes 8 --seed 262000 \
      --horizon 35 --parametrization arrival_condition --adaptive-task --tumble-scale $R \
      --control $ARM --output eval/cal2/${ARM}_r${R}.json
  done
done
python read_cal.py
```

只读 completion / equivalent-Δv / time / worst margin 四列。

**判据（锁定裁定 #3，放宽，且绝不是方向门）：**
- 两种策略资源代价不同，**且差异随翻滚率变化** → 决策空间成立（R4 排除）；
- 差异很弱或不随翻滚率变化 → **按裁定 #3 调近场距离 / 初始范围 / 翻滚率范围，把物理决策
  空间做出来，再继续同一主线。** 不换方向、不扫窗口角、不加门。

---

## 5. 根因判定表（诊断的出口条件）

四个候选根因。**它们不互斥，很可能 R1 + R2 同时成立。**
诊断"完成"的定义是：**每一行都被证据置为"成立"或"排除"，没有"未知"。**

| 代号 | 根因 | 由什么证据判定 | 对应修复 |
|---|---|---|---|
| **R1** | **门语义**：安全回退带不可逆模式副作用 | D1 的 `latch_cause` 以 `gate_fallback` 为主；D2 关门后 `latch_step` 中位数 > 0 | **F1**（小修） |
| **R2** | **探索不可达**：棘轮 + 边界 nominal 让暂存分支在训练中近乎不可达，策略没有数据可学 | D0 `latch_step` 中位数 ≤ 2；且 D2 关门后仍 = 0 | **F2 + F3**（重参数化 + 棘轮处置），并需**重训** |
| **R3** | **动作几何/表达**：即使可达，表达不够或介入无后果 | D2 关门后有介入但 rescued = 0（destroyed > 0 或全 0）；且 D4 管路通 | **F2**（重参数化），可能需重训 |
| **R4** | **任务里没有决策**：暂存本身赢不到东西 | D4 管路通但三个 `N` 一个都救不回且代价更高；且 D3 差异弱/不随翻滚率变化 | **不是改机制**——按裁定 #3 **调任务参数范围**把决策空间做出来 |

**判定完成后，把结论写成一句话交上来**，形如：
> 根因 = R1 + R2（R3 排除、R4 排除）；证据 = D0 中位数 1、D1 三条全中、D4 在 N=15 救回 2 个 both-failed。

**只有拿到这句话，才进入优化改进阶段。**

---

## 6. 修复优先级（拿到唯一根因之后，一次一个因子，各自独立提交 + manifest）

**F1｜门的回退不得推进锁存**（最高，最像 bug）
把"本步力/力矩退回基线"与"任务模式不可逆转移"彻底分开。改动小，由 T-B 钉住。
**无论根因落在哪一条，这条都要做。**

**F2｜重参数化，让 nominal 不落在裁剪边界**
判据是 T-A：每个有效动作区间都必须产生可观测的参考变化。
候选（待文献调研补充后定）：剩余推进量 / 目标到达时间 / 持续时长 / 模式 hazard、
对数或 logit 域映射、直接的参考轨迹参数。

**F3｜棘轮处置**（最谨慎）
不可直接删（8/8 → 0/8 的实测回归）。三条候选：可撤销但规则明示、换成速率限制、
做成显式 option / 模式转移。**必须带上无棘轮的对照复现**再决定。

**F4｜学习与部署对象一致**
critic 比较原始残差，门却影响映射后的参考/模式。门至少要对"映射后参考/模式转移"去重，
并报告 **effective-reference deviation**，不再以 raw-action deviation 代替实际介入。

---

## 7. 缺陷的可执行证据（与诊断并行，纯代码）

用 `pytest.mark.xfail(strict=True)` 写：**当前记为预期失败，套件保持绿**；语义修好后
strict xfail 会转为报错，强制删标记——缺陷被钉住且不会被悄悄绕过。

**T-A `test_every_commit_residual_interval_changes_the_reference`**
`r_commit ∈ [-1, 1]` 等距取 21 点，统计生成的**不同**参考点数量。当前 `r ≥ 0` 的 11 个点
全塌成捕获位姿。判据：每个有效动作区间都必须产生可观测的参考变化。

**T-B `test_gate_fallback_does_not_advance_commit_latch`**
表达目标语义：一次"本步退回基线"不得把 commit 内部状态永久置 1。
（门目前在评估脚本里而不在 env 里，所以测试要针对 env 的模式状态写——**这本身就说明
安全回退与模式转移没有分离**，正是 F1 要解决的东西。）

**T-C `test_latched_action_space_is_inert`**
锁存后（`blend = 1`）对整个 2D 动作网格求参考，当前全部相同——**锁存之后动作空间
100% 惰性，不是一半**。这条把缺陷的严重程度也钉在测试里。

---

## 8. 记录与交付要求

- **真值 RK45 几何是安全/违约的唯一裁决者**；不用观测、不用 QP `solved`、不用预测裕度。
- **`maximum_slack` 在 fallback 步可能是上一次成功求解的陈旧值**（陷阱 3）：只从成功求解的
  步读，或与 fallback 标志配对。
- **孤立 QP 异常必须保留并报告**：262412 约 29,897 步处 9 次 zero fallback、回报 −17.43、
  一次非法进入。判为孤立求解异常（此后约 3,800 步未复现，另两种子全程 0），
  **但正式评估必须保留此记录**，不得因"看起来是噪声"而省略。
- 每个数字指向 artifact：JSON 路径 + SHA-256 + 产生它的提交。
- 不丢种子、不挑 checkpoint、不事后改判据；报分布。
- 结论需 ≥3 种子——**但实现缺陷的诊断不受此约束**，它是代数/机理问题，不是统计问题。
- D0 的训练期读数是**事后估计**，必须如此标注，不得写成训练期实测。

## 9. 回传清单

1. **C1–C3 的完成确认**（含 replay 能跑通主线模型）；
2. **D0**：三份 stochastic JSON + `latch_step` 分布（中位数/四分位/=0 比例）；
3. **D1**：三份 instrumented JSON + `latch_step` / `latch_cause` 分布表；
4. **D4**：三份 staged JSON + 12 个 both-failed 上的 rescue 计数与资源代价；
5. **D3**：`read_cal.py` 汇总表；
6. **D2**：仅在上层裁决后执行，两份关门 48-seed JSON + 成对四象限 + 有效介入率；
7. **T-A/T-B/T-C** 的 xfail 输出；
8. **第 5 节那句根因结论**；
9. 262412 的停止记录与保留的 partial 产物路径。
