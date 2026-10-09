# V2 第一轮训练单（给下层）

*2026-09-22。基线 `b0abd49`。上层与用户已拍定，可以开训——但**第 0 步必须先做**。*

**已拍定的三条**：
1. 第一轮**只上 V2**（修好的两轴管子），**不上 V2.5 人工参考**——需要"V2 单独"这一行当消融臂；
2. 预算 **60,000 decisions × 3 种子，从零**，与第一轮对齐才可比；
3. 早期门读**有效介入率**，**10k 时低于 5% 立即停**。

---

## 第 0 步（必做，不做则门是空的）：把门的那个数记下来

### 问题

`env/hybrid_env.py:1080` 已经把 `hybrid_v2_reference_step_m` 放进 `info`，
但 `train/train_hybrid.py:314` 的 `info_keywords` **没有记它**，只记了
`completed` / `terminal_region_active` / `illegal_terminal_entry_count` /
`hybrid_waypoint_radius_m` / `hybrid_qp_zero_fallbacks`。

**所以现在 `train.monitor.csv` 里读不到介入率，10k 的门无法执行。**

而且 `Monitor` 的 `info_keywords` **只取回合终止那一步的 info**，所以光把键名加进去也不够
——**必须在 env 里按回合汇总**。

### 要做的

在 `PrecaptureHybridEnv` 里累计，并在回合终止的 info 里给出**三个回合级标量**：

| 键 | 含义 |
|---|---|
| `hybrid_v2_episode_changed_fraction` | 本回合中**参考真实改变**的决策占比（判据用这个） |
| `hybrid_v2_episode_mean_reference_step_m` | 本回合参考位移的均值（米） |
| `hybrid_v2_episode_accepted_fraction` | `proposal_accepted` 为真的决策占比（架构级回退频率） |

"参考真实改变"的判定沿用活性探针的口径：**本次生效参考与固定设定点不同**
（容差 `1e-9`）。三个键**只在 `task_state_v2` 下产生**，其他参数化下不出现。

然后把这三个键加进 `train_hybrid.py` 的 `info_keywords`。

### 验收

加两个测试（会红→实现后绿）：

- **G1**：`task_state_v2` 跑完一个短回合，终止 info 里三个键都存在、取值在 [0,1] 或非负，
  且 `changed_fraction` 与逐决策统计一致；
- **G2**：其他参数化下这三个键**不出现**（不污染历史路径）。

**这一步是纯记录管道，不改 reward / MPC / 动作语义 / 任务参数。**

---

## 第 1 步：开训前的两个检查

### 1.1 活性检查（这是第一轮缺的那一环）

```
python -B experiments/probe_channel_liveness.py
```

**期望**（上层已在种子 262000 / 262001 / 262005 上实测）：

| | 有效改变率 | 后半段 | 40 次决策的不同参考点 |
|---|---:|---:|---:|
| V1 `arrival_condition` | 0.026 / 0.025 / 0.050 | **0.000 ×3** | 2 / 2 / 3 |
| V2 `task_state_v2` | **1.000 ×3** | **1.000 ×3** | **40 / 40 / 40** |

**V2 三条必须全过。任何一条不过就停，不要开训。**

### 1.2 全套回归

```
python -B -m pytest -q
```

**期望：`285 passed, 3 xfailed`**（283 + G1/G2）。三个 xfail 是 V1 的钉子，**不要动**。

---

## 第 2 步：训练，3 种子，从零

种子沿用第一轮的 262410 / 262411 / 262412（训练种子块对齐，便于和第一轮比），
**run-name 换新**，避免覆盖：

```
for SEED in 262410 262411 262412; do
python -B -m train.train_hybrid \
  --steps 60000 --seed $SEED --run-name v2_${SEED} \
  --horizon 35 --parametrization task_state_v2 \
  --adaptive-task --device auto
done
```

- **不要**加 `--baseline-anchored-residual`（那是 `arrival_condition` 专用，V2 会拒绝；
  基线保底已经是架构级的）；
- **三个独立进程/终端并行**最快（规则允许训练并行），瓶颈是 MPC 求解，**别上 GPU**；
- 上面的 `for` 只是串行循环，真并行要开三个终端。

---

## 第 3 步：门怎么读

### 10k：灾难门，**低于 5% 立即停**

从 `logs/v2_<seed>/train.monitor.csv` 读 `hybrid_v2_episode_changed_fraction`，
取最近 100 回合的均值。

| 读数 | 判定 | 动作 |
|---|---|---|
| **≥ 5%** | 管子在闭环里活着 | 继续跑 |
| **< 5%** | **正在走 V1 那条衰减曲线** | **立即停，报上层** |

5% 这个线的来历：V1 的实测衰减是 **5k 时 4–6% → 10k 时 2–15% → 60k 时精确 0%**。
低于 5% 说明已经在同一条路上。

> **10k 的低分只判"接口是否又死了"，不判方向。** 早期/smoke 从不是方向门，这是纪律明令。

### 30k：趋势确认

同一个数，看**有没有单调下降**。三个种子都在往下走 → 停下报上层，不要跑到 60k。

### 60k：正式完成

---

## 第 4 步：**不要**用什么当门

**回报和完成率不是门。** 第一轮的教训：三个种子的完成回报一路稳定在 **+15.5**、
hover **+4**、间隔清楚，完成率稳定 **73%**——**而能力正在归零**。
那时候拿完成率当门，门根本没响。

回报和完成率**继续记录**，但**不单独用来判断"RL 是否学起来"**。

---

## 第 5 步：跑完之后（先别做，等上层）

60k 跑完**先报数据，不要自行开评估**。正式评估（48 种子块 262000、h35、
Pure MPC 对照行）由上层定义判据后再开——第一轮的教训是评估判据必须先写死。

---

## 禁止事项

- **不上 V2.5**（人工参考 / 可还价）。这一轮只要"V2 单独"这一行。
- **不接 governor**，不增加任何在线额外 MPC 求解。
- **不改** reward / SAC 超参 / MPC / 任务参数 / 执行器 / 控制周期 / V2 动作语义。
- **不改** `v2_reference_step_max_m`（已裁定 0.40，理由见 `V2_RATE_LIMIT_DECISION_20260922.md`）。
- **不动**三个 V1 的 strict xfail。
- **不 resume、不 warm start、不挑 checkpoint。** 三个种子都从零。
- **不因为 10k 分数低就改设计**——那一步只判接口死活，先报上层。

---

## 交回

1. 活性检查三条判据的结果；
2. 全套回归结果；
3. **10k 时三个种子的 `hybrid_v2_episode_changed_fraction`（最近 100 回合均值）**；
4. 30k 同上 + 是否单调下降；
5. 60k 完成后：三份 `train.monitor.csv` + manifest + 最终模型路径；
6. 若中途触发停止门，停在第几步、读数多少。

**训练期间每到 10k / 30k 就报一次，不要跑完才报。**
