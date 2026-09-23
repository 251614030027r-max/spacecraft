# V2 正式评估方案（一次跑完，判读全部预注册）

*2026-09-23。60k 训练完成后执行。基线 `9b14903`。*

**这份方案的目的是：跑完之后不需要再讨论一天。** 每一个数字的读法都在第 4 节预先写死了，
下层按表填结论，不需要临场解释。

**总量约 396 个 episode，串行单进程约 13 小时。** 按第 0→1→2→3 的顺序跑，
**越靠前越决定性**——中途若必须中断，前面的结果仍然可用。

---

## 第 00 步：先补三处脚手架，否则方案里有两项读不出来

**上层自查发现的缺口，不补则重演第一轮"判据写在纸上、数没记下来"。**

### 00.1 评估记录里加逐决策的任务状态（Q2 的前提）

`experiments/evaluate_hybrid_policy.py` **目前完全不记录任务状态**——
`grep task_state` 只命中参数化的 choices。而 **Q2 需要每个完成回合最后 20 个决策的
参考步长与 `ρ`/`c` 是否顶边界**。

**不补这一项，第 1 步跑完 144 个 episode 之后 Q2 答不出来，得整批重跑。**

在每回合记录里加（`task_state_v2` 下才出现）：

| 字段 | 内容 |
|---|---|
| `task_state_applied` | 逐决策的 `(ρ, c)`，与 `decision_times_s` 索引对齐 |
| `task_state_proposed` | 逐决策的策略提议 `(ρ, c)` |
| `v2_reference_step_m` | 逐决策的参考位移（米） |
| `v2_progress_max_m` | **本回合**的 `ρ` 上界（它依赖回合初始暂存半径，**每回合不同**） |

> **`v2_progress_max_m` 必须逐回合记录。** 它是 `hold_radius − v2_radius_min_m`，
> 随初始半径变化。用一个全局常数去判"是否顶边界"会判错。

env 侧的 `hybrid_task_state_applied` / `hybrid_task_state_proposed` /
`hybrid_v2_reference_step_m` 已经在 `info` 里，只需在评估循环里逐决策收集。

### 00.2 加强制全拒的 CLI 入口（第 2 步的前提）

`evaluate_hybrid_policy.py` **没有任何 force-reject 入口**（`grep` 无命中）。
第 2 步按现在的写法跑不起来。

加一个 `--force-reject-all`：调用
`env.step_with_proposal(action, proposal_accepted=False)`，与 T4 测试同一口径。
**它必须同时要求 `--model`**（要复验的是"最终模型在被全拒时"仍逐位等于基线）。

### 00.3 评估命令一律取自训练 manifest，不要手写

训练把观测 flag 硬编码为 `include_target_phase_and_time_observation=True`，
执行反馈默认开，所以 V2 的评估**必须**带 `--phase-time-observation --execution-feedback`。
**本文第 1、3 步里我手写的命令漏了这两个**（会被观测维度守卫拒绝——守卫是好的，
但说明手写命令不可信）。

**正确做法**：

```
python -c "import json;print(' '.join(json.load(open('logs/v2_262410/manifest.json'))['evaluation_flags']))"
```

**把打印出来的那串 flag 原样拼进评估命令**，再加 `--model` / `--output` / `--episodes` / `--seed`。
三个种子各自读自己的 manifest。这条是 CLAUDE.md 陷阱 6（"按 manifest 而不是猜 flag"）。

### 00.4 验收

- 加两个测试：记录字段在 `task_state_v2` 下存在且与逐决策统计一致 / 其他参数化下不出现；
- `--force-reject-all` 在一个 episode 上跑通；
- 全套回归仍绿。

**00 步不完成，不许开始第 0 步。**

---

## 第 0 步：MPC 等价性核对（必做，最先，否则整张主表作废）

### 为什么

提交 `9b14903`「reduce V2 MPC presolve overhead without changing formulation」**把
`controllers/mpc/constraints.py` 里的约束裕度计算手工重写了一遍**（原先调
`normalized_precapture_constraint_margins`，现在逐行内联手算）。行布局核对过是对应的
（`nominal[0..3]`、终端分支 `[4]`、走廊面、`[corridor+5]`、`[corridor+6]`），
**但这是 MPC 约束路径上的手工改写，不是缓存。**

**而且现成的基线行 `eval/adp/pure_mpc_nominal.json` 是 `32fe20ef` 跑的，早于这个提交。**
V2 的三个模型是在 `9b14903` 下训练的。

> **如果改写不是数值等价的，基线和 V2 就来自两个不同的下层，整张主表的对比无效。**

### 0.1 逐元素等价核对（几分钟，纯计算）

写一个一次性脚本：在若干采样状态上（建议覆盖终端区内/外、不同距离、不同相位，≥200 个状态），
把内联算出的 `nominal` 向量与 `normalized_precapture_constraint_margins` 的输出**逐元素比较**。

**预注册判读**：

| 观测 | 结论 |
|---|---|
| 最大绝对差 `< 1e-12` | 数值等价，**现成基线行可用** |
| 最大绝对差 `1e-12 ~ 1e-6` | 浮点重排，**可用**，但在报告里记下差值量级 |
| 最大绝对差 `> 1e-6`，或任何一行符号/活跃性不同 | **不等价 → 基线必须在 `9b14903` 重跑，且必须记下"V2 与旧基线来自不同下层"** |

### 0.2 基线在当前提交上重跑（48 episode，约 1.6 小时）

**无论 0.1 结果如何都跑。** 仓库规则本来就要求「Pure MPC 在最后干净重跑一次给最终数字」，
现在正是那个时候。

```
python -B -m experiments.evaluate_hybrid_policy --episodes 48 --seed 262000 --horizon 35 \
  --parametrization arrival_condition --adaptive-task --control desired_pose \
  --output eval/adp/final_pure_mpc_nominal.json
```

**预注册判读**：与 `pure_mpc_nominal.json` 逐 episode 比较 `completed`、
`equivalent_delta_v_m_s`、`survival_s`。

| 观测 | 结论 |
|---|---|
| 48/48 episode 全部一致 | 改写中性，历史数字全部继续有效 |
| 有任何 episode 不一致 | **以新行为准**，并在报告里列出变化的 episode 与变化量；旧行标为"前一下层" |

**这一行就是主表的基线行**，后面所有对比都对它，不对旧的。

---

## 第 1 步：主表，nominal，48 种子（144 episode，约 4.8 小时）

固定评估块 262000–262047、h35、adaptive task、真值 RK45 裁决、串行单进程。

```
for SEED in 262410 262411 262412; do
  FLAGS=$(python -c "import json;print(' '.join(json.load(open('logs/v2_'$SEED'/manifest.json'))['evaluation_flags']))")
  python -B -m experiments.evaluate_hybrid_policy --episodes 48 --seed 262000 $FLAGS \
    --model logs/v2_${SEED}/final_model.zip \
    --output eval/adp/final_v2_nominal_${SEED}.json
done
```

- **flag 一律取自 manifest**（见 00.3），不要手写；
- **确定性**（默认，不要加 `--stochastic-policy`）——这是唯一能和基线比的口径；
- **不要**加 `--deployment-gate`（这一轮没有仲裁层，策略提议一律接受）。

---

## 第 2 步：架构地板复验（12 episode，约 25 分钟）

**必须用最终模型重测，不许假设它还成立。**

强制拒绝全部提议时，闭环必须**逐控制步**等于 Pure MPC：

```
for SEED in 262410 262411 262412; do
  FLAGS=$(python -c "import json;print(' '.join(json.load(open('logs/v2_'$SEED'/manifest.json'))['evaluation_flags']))")
  python -B -m experiments.evaluate_hybrid_policy --episodes 4 --seed 262000 $FLAGS \
    --force-reject-all --model logs/v2_${SEED}/final_model.zip \
    --output eval/adp/final_v2_floor_${SEED}.json
done
```

（`--force-reject-all` 是 00.2 加的，走 `step_with_proposal(..., proposal_accepted=False)`，
与 T4 测试同一口径。）

**预注册判读**：`max |Δwrench|` 必须是 **0.0**。
**任何非零 → 立即停，报上层，这是 bug，不是结果。**

---

## 第 2b 步：算力行（1 次，约 2 分钟）

```
python -B -m experiments.profile_precapture_mpc --no-diagnostics
```

V2 的限速器是纯几何二分、不解 MPC，**预期不增加求解成本**。但这一行论文要用，
且能证实"零额外在线求解"这个设计承诺。报 p95/控制周期、超周期步数、以及 max 列
（**那是第 0 步冷启动，不是稳态尾部**，见陷阱 3）。

---

## 第 3 步：跨工况（条件执行，第二优先，192 episode，约 6.4 小时）

**只有在第 1 步的主表非退化时才跑**（判据：至少一个种子的有效参考改变率 > 50%，
且至少一个种子完成 ≥ 12/48）。否则跳过并说明原因。

缩减块：**24 种子（262000–262023）**，两个未见翻滚率。

```
for R in 0.10 0.30; do
  python -B -m experiments.evaluate_hybrid_policy --episodes 24 --seed 262000 --horizon 35 \
    --parametrization arrival_condition --adaptive-task --tumble-scale $R \
    --control desired_pose --output eval/adp/final_mpc_r${R}.json
  for SEED in 262410 262411 262412; do
    FLAGS=$(python -c "import json;print(' '.join(json.load(open('logs/v2_'$SEED'/manifest.json'))['evaluation_flags']))")
    python -B -m experiments.evaluate_hybrid_policy --episodes 24 --seed 262000 $FLAGS \
      --tumble-scale $R --model logs/v2_${SEED}/final_model.zip \
      --output eval/adp/final_v2_r${R}_${SEED}.json
  done
done
```

**这是 zero-shot 泛化**：同一个策略、同一个 reward、只改翻滚率。
**好 = 很强的结果；一般也不否定主线。**

---

## 第 4 步：预注册判读表（跑完按这张表填，不要临场解释）

### Q1｜V2 确定性水平相对基线

对比基线（第 0.2 步那一行，36/48 或其重跑值）：

| 观测（每个种子分别看） | 结论 | 下一步 |
|---|---|---|
| ≥ 36/48 | **策略单独就保住了基线** | 强结果，直接进 V2.5 设计 |
| 24–35/48 | 低于基线但可观 | 基线保持要靠**架构回退 + 仲裁**，V2.5 的仲裁层成为必需项 |
| 12–23/48 | 明显低于基线 | 接口给了用不好的权限 → 改**幅度自由度**（见 Q6），不是训更久 |
| < 12/48 | 远低于基线 | 回到接口/奖励设计，不要加机制 |

**同时必做的一个对照**：40k 时的**训练期**成功率是 31.3%。

> **若确定性水平明显高于 31.3%** → 确认"探索噪声压低了训练期读数"这个判断，
> **这是一个可写进论文的测量**（V1 的通道死了所以两者相等，V2 活了所以两者分离）。
> **若两者接近** → 说明噪声不是主要因素，策略本身就在这个水平。

### Q2｜完成是"策略收手"还是"撞到接口端墙"

对每个**完成**的 episode，取**完成前最后 20 个决策**，读两条：

1. 参考步长（米）随时间；
2. 同一时刻 `ρ` 是否 ≥ 0.99·`v2_progress_max_m`、`c` 是否 ≥ 0.99。
   **`v2_progress_max_m` 用该回合自己记录的那个值**（见 00.1），不是全局常数。

| 观测 | 结论 | 下一步 |
|---|---|---|
| 步长降到 ≤ 0.05 m，**且 ρ/c 未顶边界** | **A：策略学会了收手** | 平台化是学习速度问题 → 训练预算 / 奖励 |
| 步长降到 ≤ 0.05 m，**但 ρ/c 正好顶边界** | **B：靠撞端墙完成** | **接口仍在替策略决策** → 改幅度自由度 |
| 步长始终 ≥ 0.3 m | 饱和与完成互斥被确认 | **限速上界 0.40 是主要瓶颈** → 重新裁定 |

### Q3｜成对四象限（**这是本轮的头条**）

每个 V2 种子对基线逐 episode 配对，给 retained / rescued / destroyed / both-failed。

| 观测 | 结论 |
|---|---|
| **rescued > 0** | **学习层第一次真的救回了基线失败的状态**（第一轮是 rescued 0 / destroyed 0）→ 互补性在学出来的层上成立，这是论文最硬的一条 |
| rescued = 0 且 destroyed > 0 | 策略净有害 → 仲裁层是必需的，不是可选的 |
| rescued = 0 且 destroyed = 0 | 又是恒等式（不作为）→ 检查介入率，可能回到 V1 的病 |
| rescued > 0 **且** destroyed > 0 | 互补性 + 破坏同时存在 → **正是 T12 那个"救得回也毁得掉"的结构**，仲裁的价值可量化 |

**报告必须同时给出三个种子各自的四象限，不许合并成一个平均数。**

### Q4｜种子分布（规则风险）

| 观测 | 结论 |
|---|---|
| 三个种子都 > 0 完成 | 满足 ≥3 种子规则，可向上报结论 |
| **有种子 0/48** | **记为死种子，不得丢弃。** 按仓库规则"n=2 且一个死种子，噪声底约 ±37 点"，**任何正面结论只能标为"两种子上成立、第三种子未复现"** |

**262410 在 40k 时仍是 0%。这一格很可能触发，提前准备好这个表述。**

### Q5｜介入率与饱和

每个种子报：有效参考改变率、参考步长分布（中位 / p95 / 最小回合均值）、后退决策占比。

| 观测 | 结论 |
|---|---|
| 改变率 ≈ 1.000 且步长仍 100% 贴 0.39–0.40 | 饱和未缓解 → 策略只决定方向不决定幅度 |
| 出现明显低于 0.30 的回合均值 | **开始学会减速**，与 40k 时 262411 的 0.19 最低值一致 → 记录并与该种子的完成率对照 |
| 改变率 < 0.5 | 通道在评估口径下衰减 → 立即报上层 |

> **注意**：40k 时 262412 完全饱和（0.39）却有 43% 完成，262411 部分减速（0.36）有 45%，
> **所以"减速才能完成"目前不成立**。这一格只做记录与相关性观察，**不得当作结论**。

### Q6｜若 Q2 落在 B 或"饱和与完成互斥"

则下一轮的修复方向是**给策略幅度自由度**，具体候选（下一轮再定，本轮只记录）：
- 把限速上界本身变成策略的一个输出（学"走多快"）；
- 或按距离/剩余时间缩放上界；
- 或把 `v2_reference_step_max_m` 重新裁定（当前 0.40 是在四档扫描里取的宽端）。

### Q7｜跨工况（若第 3 步执行）

| 观测 | 结论 |
|---|---|
| 同一策略在 r=0.10 / 0.30 上行为明显不同（commit 时刻、staging 半径），且整体权衡不差于基线 | **跨工况自适应成立** → 论文头条证据 |
| 行为几乎不随翻滚率变化 | 未支持跨工况主张，**但不否定主线**（locked call #3）；正式版可在翻滚率分布上训练后再测 |
| 明显差于基线 | 记录为 zero-shot 局限，不掩饰 |

---

## 第 5 步：陷阱清单（读数时必须遵守）

1. **裕度只看 `minimum_truth_normalized_margin`。** 原始 per-key 裕度没有门控，
   在没违约的回合上也会是大负数，且混着米/弧度/米每秒。
2. **`maximum_slack` 在 fallback 步可能是陈旧值。** 只从成功求解的步读，或与 fallback 标志配对。
3. **任何算力测量的 `max` 列是第 0 步**（冷启动），次大的才是稳态。报告时写成"一次性冷启动"。
4. **`predicted_minimum_margin` 不是真值裕度**，且只在 `runtime_diagnostics=True` 下存在。
5. **不要重跑旧的 `pure_mpc_nominal.json`**——它是历史记录；新基线写成 `final_pure_mpc_nominal.json`，
   两者都留着。
6. **评估串行单进程。** 并行会污染算力列（任务级指标不受影响，但别混）。
7. **不挑种子、不挑 checkpoint、不丢失败种子。** 报分布。

---

## 第 6 步：交回格式（按这个填，别自由发挥）

```
0. MPC 等价性：逐元素最大绝对差 = ___ ；基线重跑 48/48 一致 = 是/否（不一致的 episode: ___）
   新基线：完成 __/48，零违约 __/48

1. 主表（每个种子一行）
   种子 | 完成/48 | 零违约完成/48 | 平均 Δv | 平均完成时间 | 全局最薄真值归一化裕度 | 非法进入回合数 | QP 无解 | 零推力回退

1b. 算力行（一次，任一种子，串行单进程）
   p95/控制周期 = ___ ；超周期步数 = __/300 ；max 列（= 第 0 步冷启动）= ___ ms
   与 Pure MPC 的 h35 基准 0.77x 对比：___（限速器是纯几何二分，预期无明显增加）

2. 架构地板：max|Δwrench| = ___（必须 0.0）

3. Q1 判定 = ___（对照训练期 31.3%：确定性 ___%）
   Q2 判定 = A / B / 互斥   （证据：完成前 20 决策步长 ___ m，ρ/c 是否顶边界 ___）
   Q3 四象限（三个种子分别）retained/rescued/destroyed/both-failed = ___
   Q4 种子分布 = ___（是否有 0/48 死种子）
   Q5 改变率 ___ ，步长中位 ___ ，最小回合均值 ___ ，后退占比 ___
   Q7 跨工况 = ___（或"未执行，原因 ___"）

4. 产物路径 + SHA-256（全部）

5. 一句话结论，形如：
   "V2 确定性 __/48 对基线 __/48；rescued __ / destroyed __；Q2 判定 __；
    种子分布 __；因此下一步应 ___（引用第 4 节对应格）"
```

---

## 附：开跑前 30 秒自检

```
ls -la logs/v2_26241{0,1,2}/final_model.zip          # 三个都在?
python -c "import json;[print(s, json.load(open(f'logs/v2_{s}/manifest.json'))['observation_dimension'], json.load(open(f'logs/v2_{s}/manifest.json'))['actual_decision_steps']) for s in (262410,262411,262412)]"
git rev-parse HEAD                                   # 记下来,所有产物都指向它
git status --porcelain                               # 必须干净
```

三个 `final_model.zip` 缺任何一个、或 `actual_decision_steps` 不是 60000、
或工作树不干净 → **先报上层，不要开跑**。

**产物报告前，生成它们的提交必须已推送远端**——第一轮出现过"数字指向一个别人看不到的提交"。

---

## 禁止事项

- **不训练、不续训、不挑 checkpoint。** 只用 `final_model.zip`。
- **不改** reward / SAC / MPC / 任务参数 / V2 接口 / `v2_reference_step_max_m`。
- **不加仲裁层**（本轮无 governor，提议一律接受）。
- **不合并三个种子的四象限**成平均数。
- **不丢弃 0/48 的种子。**
- **不在第 0 步不通过时继续往下跑**——那是整张表的前提。
- **跑完不要自行下方向结论**，按第 6 节填表交上层。
