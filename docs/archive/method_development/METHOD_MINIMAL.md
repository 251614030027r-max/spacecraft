# EACH：最小闭环（定稿方法）

> **非最终论文算法（2026-11-10）：** 本文件是研发记录。论文方法以 `docs/paper/METHOD.md` 为唯一源，对照见 `docs/paper/FACT_MAP.md`。

**EACH = 执行感知的分层控制（Execution-Aware Hierarchical control）= C1 + U1。** 代码中的主方法是 `e_priv`（`obs_profile = reduced_priv`），与 T2.12 冻结版本 `2e7d838` 逐位一致（`evidence/final_pretest/precheck/`）。

**两项创新：**
- **C1｜旋转同步扫掠锥制导与窗口承诺（向下）：** SAC 的阶段意图被变成在旋转目标的过顶窗口内可执行的宏动作。
- **U1｜执行约束知识与真实宏观价值学习的非对称耦合（向上）：** 执行器的时序与可行性判据只进入训练期 critic 的输入；价值只由真实宏转移学习。

## 总图

```
 观测 o（24 维精简） ──► SAC Actor π(a|o) ──► 意图 a = (a_p, a_z)
                                                │
              ┌─────────────── C1：可执行规划 ───┴───────────────────────────────┐
              │ 扫掠锥阶段目标（a_p, a_z + 固定纬向调节）→ 限幅 + 安全投影 → 16 s 局部规划 / 远场跟踪   │
              │ → 可进入时：条件 G★ 入口（35°、12 m 预筛，τ∈{15…60} s）→ 同窗口重定时（+2/4/8 s，95% 权限） │
              │ → 冻结 MPC 下降与捕获                                                        │
              └──────────────────────────────┬───────────────────────────────────────┘
                                             ▼
                       真实宏转移 (R, o', d)，RK45 真值
                                             │
              ┌─────────────── U1：训练期 ──────┴───────────────┐
              │ 执行证书 z_E(o)：过境时间、窗口可达角、过境后截止余量（+ 其余同观测） │
              │ Critic Q(o, z_E, a)，只用真实 TD 目标                          │
              └──────────────────────────────┬──────────────────┘
                                             ▼
                         标准 SAC 策略改进（Actor 只读 o）
```

## 六步

| 步 | 内容 | 源码 | 可核验日志 |
|---|---|---|---|
| 1 决策状态 | 真值仿真给出相对状态、目标姿态与角速度、任务剩余时间与资源；Actor 读 24 维精简观测（3 个规划答案置零） | `env/t2_stage.py: observe`、`T2StageEnv._observe`（`PLANNER_FEATURES = (8, 17, 19)`） | `manifest.json`：`obs_profile`、`actor_view` |
| 2 旋转同步意图 | 学习两个自由度：接近 v_c = 0.05 + 0.15 a_p，沿扫掠 v_z = 0.15 a_z；纬向速度是固定的几何调节，不是学习动作 | `target_velocity`、`latitude_velocity`、`CLOSING_M_S`、`AZIMUTH_M_S`、`LAT_GAIN_PER_S`、`LAT_SPEED_M_S` | 评估逐决策 `a` |
| 3 可执行目标 | 速度增量限幅（16 s 内 70% 规划力），安全投影（阶段末 ≥ 6.8 m、不超过径向制动包络减 0.02 m/s）；局部阶段规划，远场（> 7.5 m）跟踪律；被拒或偏离 3 m 后 hold，≥ 2 s 后重新询问 Actor | `stage_goal`、`_safe_stage_end`、`T2StageEnv.step` | 训练逐回合 `exec_stats`（`stages`、`holds`、`rejected`、`rate_limited`） |
| 4 窗口进入与近场 | 距离 ≤ 12 m 且制动裕度 ≥ 0.02 m/s 时，对预测端口–视线夹角 ≤ 35° 的到达时间 τ 依次用条件 G★ 规划入口；已有入口重规划失败时，在同一次过顶内尝试 +2/+4/+8 s 与 95% 权限备用；之后由冻结 MPC 下降与捕获 | `_try_bridge`、`WindowGStar.decide`、`make_window_gstar` | `bridge_attempts`、`entries_lost`、`window_recoveries` |
| 5 真实结果与证书 | 记录宏步真实回报（时间、燃料、终局）、后继观测与终止；执行证书 z_E 只给训练期 critic | `T2StageEnv.step`、`_macro_to_end`、`observe` | `episodes.json`：`return_true`、`event`、`time_s`、`delta_v` |
| 6 真实价值学习 | Critic Q(o, z_E, a) 用真实 soft TD 目标；Actor 用标准 SAC 目标，只读 o | `T2SAC.train`（`feedback = None`、`model_beta = 0`）、`SplitViewPolicy` | `model_stats.json`：`real_td`、`sigma`、`q_pi_mean` |

## 训练与部署的信息边界

| | 训练期 | 部署 |
|---|---|---|
| Actor | 24 维精简观测 o | 同左；一张 24 → 2 的网络 |
| Critic | 48 维 [o, 完整工程观测]；新增独立信息只有 3 个规划量：过境时间、桥接（窗口可达）角、过境后截止余量 | 不使用 |
| 执行器 | 规划、跟踪、入口、MPC | 同左 |

**注意：** 后 24 维里其余 21 个分量与精简观测重复（编码实现），不能说成 24 个新的物理信息。重构检查表明，这 3 个规划量可由精简观测精确算出（误差约 1e-7，`evidence/T2_16_PRECHECK`），所以 U1 的作用是把高曲率、模式边界的组合运算显式交给 critic，而不是补足不可观测的状态。

## 模块分类

| 类型 | 内容 | 处理 |
|---|---|---|
| 主算法必需 | 两维意图、扫掠锥纬向引导、阶段规划、限幅与安全投影、条件 G★、同窗口重定时、远 / 近场执行、U1 critic | 保留；见上表与 `METHOD_SPEC.md` |
| 必要但不单列创新 | 势函数 shaping、固定 16 s、2 s 内层步、hold 后重询问、95% 权限备用、观测归一化 | 写入实现与参数表（`METHOD_SPEC.md` 第 5 节） |
| 历史试验，不属于最终方法 | M1–M3 反事实标签、U2 机会图、U3 动作修正、`e_acc / e_rel / e_term / e_pi / e_pull / e_cert / e_full` 的额外损失、门控、校准头 | 不进主图、伪代码和最终实验入口；代码保留以便复现。`experiments/final.py` 只能启动 `METHOD_REGISTRY.json` 中的配置 |

## 默认开关（主方法）

`arm = e_priv`、`obs_profile = reduced_priv`、`actor_view = 24`、`critic_view = 48`、`window = true`、`geometry = true`、`k_candidates = 0`、`model_beta = 0`、`pref = false`、`acc = false`、`guide = false`、`relative = false`、`feedback = null`、`terminal_horizon = 0`；紧急期限 U(140, 200) s，经济期限 U(280, 340) s；每决策固定折扣 γ_d = 0.999^8；固定熵系数 0.05；每条转移 2 次更新。启动入口：`python -B -m experiments.final train --stage S --config EACH --seed N --output DIR`。

## 伪代码（一个训练回合）

```
o ← reset(opening)                                   # 训练开局 1,000,000·(seed+1) + 100,000·worker + 回合
loop:
    a ~ π(·|o)                                        # SAC Actor，只读 o
    (R, o', d) ← Executor(a)                          # C1：阶段目标 → 规划 / 跟踪 → [G★ 入口 → 窗口重定时 → MPC] → 真值
    store (o, z_E(o), a, R + γ_d φ(o') − φ(o), o', z_E(o'), d)
    for 2 updates:  Q ← min over twin critics; y = r + γ_d (1−d)(min Q̄(o', z_E', a') − α log π(a'|o'))
                    critic ← minimize (Q(o, z_E, a) − y)²; actor ← maximize E[min Q(o, z_E, ã) − α log π(ã|o)]
    o ← o'; until d
```
