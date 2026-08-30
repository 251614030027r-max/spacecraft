# 最小验证实验：短 horizon + 学习凸终端值（SAC-MPC 耦合第一步）

> 对应 `HANDOFF_sac_mpc_coupling.md` 第 4 节。本文件是**给用户在本机跑**的操作说明；
> 代码已随附。会话内已做端到端 smoke（拟合 R²=0.997、评测 0 QP fallback、0 predicted-safe
> 违约、compute 随 horizon 显著下降），但**结论性数字必须由本机 ≥3 seed 的完整 episode 跑出**。

## 这一步验证什么

唯一可解释因子 = **终端代价来源**（`terminal_cost_source`）。把现有约束凸 QP 的
horizon 从 50 缩短，终端代价从"固定对角二次 `terminal_weight·‖S⁻¹(x−r)‖²`"换成
**一个学习到的凸二次终端值 `V(x)=(x−c)ᵀH(x−c)+gᵀ(x−c)`（H 半正定）**。

判据（HANDOFF 第 4 节）：短 horizon+学习终端值能否在**不掉安全**（worst-margin 不转负）
下把完成/质量拉到接近长 horizon MPC，同时 **compute 明显下降**。若成立→脊柱成立，进完整
hybrid；若一接学习终端值就掉安全/掉完成→回陷阱 1，先修 surrogate。

## 三陷阱在本实现下的处置（对应 HANDOFF 第 3 节）

- 陷阱 2/3（NN 进求解器 + 流形求导 / 并行双解）：**已绕开**——终端项是一个 numpy 常数
  凸二次，`‖L(x−c)‖²+gᵀ(x−c)`，DCP/DPP 平凡；单次凸 QP。
- 陷阱 1 Critic Trap：**结构护栏成立**——终端值只进目标、不进约束（约束仍是
  `linearize_constraint_margins`+slack），终端值再烂最坏只是次优，horizon 内不会直接把
  truth-margin 打负；且 H 由**带 PSD 约束的凸回归**（不是事后裁剪）直接拟合成凸，造不出
  非凸包。跨 horizon 的间接安全由 worst-margin 实测背书。

## 改了哪些文件

| 文件 | 改动 |
|---|---|
| `controllers/mpc/terminal_value.py` | **新增**：`ConvexQuadraticTerminalValue`（凸二次终端值，含 save/load、Cholesky 因子、PSD 校验）+ `fit_convex_quadratic`（带 PSD 约束的凸回归拟合）。 |
| `controllers/mpc/config.py` | `MPCConfig` 加 `terminal_cost_source`（默认 `fixed_quadratic`，不改历史行为）与 `terminal_value` 字段 + 校验；新增便捷构造器 `learned_terminal_mpc_config()`。 |
| `controllers/mpc/controller.py` | 终端目标：`learned_convex` 时用学习凸二次替换固定对角二次；`predicted_cost` 诊断同步反映实际终端代价。 |
| `controllers/mpc/__init__.py` | 导出新符号。 |
| `experiments/fit_terminal_value.py` | **新增**：脚本控制器闭环 cost-to-go 离线拟合终端值。 |
| `experiments/evaluate_mpc.py` | 加 `--terminal-cost-source` / `--terminal-value-file`，注入 config。 |
| `tests/test_terminal_value.py` | **新增** 11 项测试（默认行为不变、PSD、拟合可复原、round-trip、learned QP 可解、evaluate 序列化）。 |

全套：`python -B -m pytest -q` → **123 passed**（原 112 + 新 11）。

## 复现命令（本机，按顺序）

**纪律**：拟合用的 seed 与评测的 262000 块**必须不相交**（默认 `--seed 990000`），
否则终端值被评测集污染。每个因子独立 commit + manifest；结论 ≥3 训练/拟合 seed 并报分布。

### 步骤 0：拟合终端值（3 个拟合 seed，报分布）

```bash
for s in 990000 990100 990200; do
  python -B -m experiments.fit_terminal_value \
    --episodes 12 --seed $s \
    --output models/terminal_value_scripted_seed$s.json
done
```

`models/` 未被追踪，正合适。留意打印的 `fit_r_squared`（应 >0.99）、`weight_eigenvalue_min`
（应 ≥0）。三个拟合各自喂给下面的 (c) 行，得到 (c) 的分布。

### 步骤 1：三行主表（同一 262000 块，20 episode，对齐）

```bash
# (a) 现有 Pure MPC 基线：长 horizon 50，corridor 参考，固定终端（参照，非因子）
python -B -m experiments.evaluate_mpc --task single_phase \
  --episodes 20 --seed 262000 --output logs/mpc_h50_fixed.json

# (b) 短 horizon 10，固定终端 —— 因子对比的对照臂
python -B -m experiments.evaluate_mpc --task single_phase \
  --episodes 20 --seed 262000 --horizon 10 \
  --terminal-cost-source fixed_quadratic \
  --output logs/mpc_h10_fixed.json

# (c) 短 horizon 10，学习凸终端 —— 因子对比的处理臂（对每个拟合 seed 各跑一次）
python -B -m experiments.evaluate_mpc --task single_phase \
  --episodes 20 --seed 262000 --horizon 10 \
  --terminal-cost-source learned_convex \
  --terminal-value-file models/terminal_value_scripted_seed990000.json \
  --output logs/mpc_h10_learned_seed990000.json
```

**真正的单因子对比是 (b) vs (c)，两者同为 horizon 10、corridor 参考，只差终端来源。**
(a) 同时变了 horizon 和终端来源，只作背景参照。

### 步骤 2：拼主表并贴回

```bash
python -B -m eval.main_table \
  --row "PureMPC_h50=logs/mpc_h50_fixed.json" \
  --row "MPC_h10_fixed=logs/mpc_h10_fixed.json" \
  --row "MPC_h10_learned=logs/mpc_h10_learned_seed990000.json"
```

把这张表（含 alignment 行）贴回即可。**不要传日志文件。** compute 列看 p95/max，不看均值。

## 看什么（判据落到具体列）

- **安全**：(c) 的 `worst_constraint_margin` 每一项是否 ≥0；`predicted_safe_truth_violation_count`
  是否为 0；QP `fallback_count` 是否为 0。掉负 = 学习终端值把系统导进后续不可行角落 → 回陷阱 1。
- **完成/质量**：(c) 的 completion、completion_time、force_impulse、discounted_return 是否
  接近 (a) 长 horizon MPC；(b) 短 horizon 无学习终端应明显更差（这正是"学习终端值补回被砍
  horizon"的证据）。
- **compute**：(b)/(c) 的 `controller_p95_over_budget`、max 应显著低于 (a)（短 horizon 砍掉
  沿 horizon 的开销）。(b) 与 (c) 的 compute 应彼此接近（终端来源不改算力）——会话内 smoke
  已见 (b)=0.31×、(c)=0.30× 对 (a) 的 ~1.75×。

## 诚实边界（写实验记录时带上）

- 会话内 smoke 只跑了 6 s / 单 episode，仅证明**管线通、指标能分离**，不构成结论。
- 首版终端值学的是**脚本策略**的 cost-to-go（离线、无训练），是 Bemporad 凸终端 surrogate 的
  监督回归种子；不是"混合策略"的 cost-to-go。闭环联合训练是脊柱成立后的下一步（HANDOFF 第 8 节
  决策 4，倾向两阶段起步）。
- 终端值锚定在期望位姿 x*（"离目标还差多少"的自然锚点），拟合用无折扣尾部代价和（与有限
  horizon MPC 目标不带折扣一致）；`--gamma` 可调但会引入第二个因子，改它要单开一轮。
- compute 是本机墙钟，跨机不可比；只用 (a)/(b)/(c) 的**相对**下降和 QP 占比论证，并注明星载慢
  1–2 数量级。
- horizon=10 是首版取值，非定论；(b)/(c) 若都掉太多，可扫 horizon∈{5,15,20}，但那是**另一个
  因子**，单开一轮，别和终端来源混跑。

---

## 结果与结论（20 episode 主表 + 会话内诊断）

用户本机跑出的三行主表（262000 块，20 episode，对齐确认）：

| method | completed | time s | force N·s | worst margin | compute | budget | return |
|---|---|---|---|---|---|---|---|
| PureMPC_h50 | 20/20 | 89.310 | 159.975 | +0.044 | 104 ms | 1.04× | 4.349 |
| MPC_h10_fixed | 20/20 | 89.345 | 160.010 | +0.044 | 32.8 ms | **0.33×** | 4.347 |
| MPC_h10_learned | **0/20** | — | — | **−0.016** | 34.8 ms | 0.35× | — |

三个拟合 seed（990000/990100/990200）R² 均 0.9997、H 半正定。

**两条硬结论：**

1. **算力问题靠缩 horizon 免费解决，学习终端值无空间可占。** horizon 50→10 把控制器算力从
   1.04× 砍到 **0.33×（进预算、可实时）**，且**质量零损失**——`MPC_h10_fixed` 与 `PureMPC_h50`
   在完成/时间/推力/裕度/回报上几乎逐位相同。是 **corridor 参考在做制导，不是 horizon 长度**；
   50 步是过剩。既然短 horizon+固定终端已 20/20+安全，学习终端值最多打平，没有质量/算力缺口
   可利用。

2. **学习凸二次终端值在本任务上被严格支配、且有害——是函数类问题，不是标定问题。** 会话内两组
   诊断（各 2–3 episode，解释 0/20 的机制）：

   - **λ 缩放扫描**（corridor 参考，学习终端权重 ×λ）：λ=0（无终端）3/3 完成且安全；此后**任何
     正权重只会单调变差**——λ=0.01 停滞不完成、λ=0.03–0.1 侧向拖出走廊、λ≥0.3 早期破 FOV。
     **没有能救活的 λ** → 排除"单纯标定错"。
   - **固定设定点救援探针**（去掉 corridor，让终端值当唯一的越视界制导）：固定设定点 h10 固定终端
     仍 2/2 完成安全（+0.081）；学习终端 λ=1 反而**卡在 ~9.7 m 进不去、破 FOV**。→ 学习终端连
     "替代缺失的路径规划"都做不到。

   **机制**：全局凸二次的梯度**径向指向中心 x***，而可行路径是**弯曲的共旋走廊 + FOV 锥**。所以
   "cost-to-go"把飞行器直直拽向目标，切过走廊/FOV。拟合 R²=0.9998 正是陷阱——**值拟合得好，
   驱动 QP 的梯度方向错**。trust-region/降权都改不了错误的梯度方向；唯一对症的函数类升级是 ICNN
   （能表达非径向、走廊形的下降方向）。

**对研究主线的影响（需与用户对齐）：** 值耦合（学习终端值）这条最小验证在 `single_phase` 上
**以负结果收官**——这正是"先便宜地打掉最大风险再决定进不进完整 hybrid"该有的产出。且结论 1 意味着
Pure MPC 短 horizon 已实时+安全，混合控制器原本的算力动机在本任务/本平台被削弱。下一步是战略选择，
不是继续调 surrogate（见会话汇报的四选项）。

（口径：主表结论 1 有用户 20-episode 背书；结论 2 的机制来自会话内 2–3 episode 诊断，方向一致、
内部自洽，任何据此的转向建议用更足 episode 复核。）

---

## 结论 1 修正（上层复核要求）：compute 必须报 p95/max，不报均值

上层复核指出：结论 1 报的 0.33× 是**均值**——正是 CLAUDE.md 明令禁止的口径。exact 线性化
刷新步（~250 ms）**与 horizon 无关**，每 `exact_linearization_refresh_steps` 步触发一次，主导
p95/max。复核为真。

**`experiments/profile_mpc_step.py` 分离刷新步/非刷新步（会话内测得，墙钟、只看比值）：**

| horizon | mean | p95 | 非刷新步 mean | 刷新步（1×/s） | QP | 约束线性化 |
|---|---|---|---|---|---|---|
| h10 | 54.9 ms (0.55×) | **276.6 ms (2.77×)** | 29.7 ms (0.30×) | **281.9 ms (2.82×)** | 9.7 ms | 6.9 ms |
| h50 | 190.4 ms (1.90×) | 443.0 ms (4.41×) | 162.6 ms | 440.5 ms (4.41×) | 54.8 ms | 38.2 ms |

**一个 exact 线性化点 h10 是 252 ms、h50 是 278 ms——基本相同，horizon 无关。** 缩 horizon 把
QP/rollout（非刷新步 162→30 ms）砍进预算，但**刷新步仍把 p95≈2.8×、max≈2.8× 顶在预算之上**。
实时受最坏情况约束，所以**短 horizon 并未实现实时；刷新步才是真命门。** 修正后的结论 1：
"缩 horizon 只解决均值与非刷新步的算力，p95/max 由 horizon 无关的刷新步决定，仍超预算。"

### 刷新 compute 实验（单因子：刷新策略；h10、corridor、固定终端、2 episode、262000）

`experiments/evaluate_mpc.py` 新增 `--linearization-source {exact,local}` 与 `--exact-refresh-steps N`：

| 刷新策略 | 完成 | 最差裕度 | mean | p95 | max | 预算 mean/p95/max |
|---|---|---|---|---|---|---|
| exact 每 10 步（基线） | 2/2 | **+0.0442** | 50.6 ms | 214 ms | 605 ms | 0.51× / **2.14×** / **6.05×** |
| exact 每 100 步 | 2/2 | **+0.0442** | 33.4 ms | 34.2 ms | 468 ms | 0.33× / 0.34× / 4.68× |
| local（无真值刷新） | 2/2 | **+0.0442** | 57.3 ms | 66 ms | 196 ms | 0.57× / 0.66× / **1.96×** |

**三点：**

1. **刷新是瓶颈，且是真·可调杠杆**：基线 p95/max 由刷新步顶起；`local` 去掉真值刷新 lump →
   p95 0.66×，`refresh=100` → p95 0.34×（但 max 仍 4.68×，因 lump 仍偶发一次）。
2. **在本任务上放松刷新不掉安全**：三档 worst-margin **逐位相同 +0.0442**。
3. **但都还没在 max 上完全进预算**（local max 1.96×，含冷启动残差），且**"放松刷新仍安全"是
   在单一确定翻滚上测的**——线性化陈旧/local 模型误差"从未遇到不同相位"（陷阱 9）。**放松刷新
   在采样相位下是否仍安全，是决定性实验（路线 A），而那个放松调度正是学习型自适应刷新层要决定的。**

（口径：会话内 2 episode smoke + 墙钟比值；安全需 ≥3 seed / 20 episode 复核；绝对 ms 跨机不可比。）

### 后续落点（据复核修正）

- **学习层落点从"终端代价"改为"自适应刷新 / SAC 供参考"**——刷新实验证明杠杆与价值都在这里。
- **路线 A（采样相位更难 regime）从"备选"升为决定性下一步**：它同时检验"放松刷新是否仍安全"与
  "手工 corridor 是否失效"，两者都是学习层能否加值的前提。
- 复现命令见会话汇报。
