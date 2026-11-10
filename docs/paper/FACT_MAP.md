# 事实映射：论文对象 ↔ 源码

**基准：** 科学 `e0a61753`，配置 `ECC_CERT_LOOP`（`docs/final_method/METHOD_REGISTRY.json`：`e_priv` + `cut_priv` + `certify`）。

**核对方式：** 上层沙箱逐项阅读源码，并做数值检查（2026-11-10）。若论文表述与源码不一致，只改文档，不改实现。

| 论文对象（`METHOD.md`） | 源码入口 | 调用路径 | 训练中 | 执行中 | 备注 |
|---|---|---|---|---|---|
| 真值动力学（SE(3)、中心引力、J2、二阶质量矩、重力梯度） | `dynamics/gravity.py`、`dynamics/integrator.py` | `T1Env.step` | ✓ | ✓ | 0.1 s 控制、RK45 |
| 结局与约束判定 | `env/t1_task.py: compute_t1_metrics`；`env/t1_env.py: step` | 每 0.1 s | ✓ | ✓ | 硬违背：碰撞、操作球 / 锥、视场、径向包络、总速度、轴向包络；最大距离 30 m |
| 宏决策与结束条件 | `env/t2_stage.py: T2StageEnv.step`、`_macro_to_end` | 每次 `env.step(action)` | ✓ | ✓ | 16 s / 偏离保持 ≥ 2 s 提前 / 入口或下降后至终局 |
| γ_d | `GAMMA_STAGE = 0.999 ** 8` | 学习器 `gamma`，塑形 | ✓ | — | 固定每决策折扣 |
| 策略观测 o（18 项有效） | `observe` + `_observe`（`cut_priv`：第 8、14、15、16、17、19 维置 0） | `observation()` | ✓ | ✓ | 24 维张量 |
| 执行反馈 z_E（6 项） | `observe` 第 8（经过时间）、17（入口可达夹角）、19（经过后期限余量）、14 / 15（位置 / 速度残差）、16（保持）维 | `_observe` 的后 24 维 | ✓（评价器） | ✗（执行时不用评价器） | 第 18 维制动裕度两边都有 |
| 策略 / 评价器视图 | `train/t2_mve_sac.py: SplitViewPolicy`（`actor_dim = 24`，`critic_dim = 48`） | `make_model` | ✓ | 只用策略 | — |
| 扫掠几何（λ、β、Δλ、φ、e_r、e_pol、e_az、经过时间） | `spin_frame` | `observe`、`stage_goal`、预测器 | ✓ | ✓ | φ̇ = −‖ω‖ − v_z /(r sin λ)、λ̇ = −v_l / r 已数值核对 |
| 二维意图 → 目标速度 | `target_velocity`、`latitude_velocity`（`CLOSING_M_S`、`AZIMUTH_M_S`、`LAT_GAIN_PER_S`、`LAT_SPEED_M_S`） | `stage_goal` | ✓ | ✓ | `geometry = False` 为"w/o 旋转同步" |
| 阶段限幅与安全投影 | `stage_goal`、`d_authority`、`_safe_stage_end` | `step` | ✓ | ✓ | 6.8 m、制动包络 − 0.02 m/s |
| 阶段规划与跟踪 | `LocalStagePlanner`；`_far_field_force`（> 7.5 m，前馈 + PD，`TRACK_KP / KD`）；`_pointing_torque`；≤ 7.5 m 由冻结 MPC（`_reference`） | `env.controller.command` 包装 | ✓ | ✓ | 远场平移不是 MPC |
| 保持 | `T1PGenerator._enter_hold`；`STAGE_DEVIATION_HOLD_M = 3` | `_advance` | ✓ | ✓ | — |
| 入口判定 | `_try_bridge`（12 m、制动裕度 0.02、夹角 ≤ 35°、`BRIDGE_TAUS_S`） | 每 2 s（模式 S / H） | ✓ | ✓ | — |
| 入口规划（条件 G★，80% 力界，期限余量 45 s） | `env/t1_n_task.py: make_conditional_gstar`、`ConditionalGStar.decide` | `WindowGStar.decide` | ✓ | ✓ | 执行层估计惯量（`InfoEnv`） |
| 入口重规划扩展 | `WindowGStar.decide`（`WINDOW_SLIPS_S = (2, 4, 8)`、95% 备用） | 原到达时刻重规划失败时 | ✓ | ✓ | 只检查期限与规划可行，不显式检查相位窗口 |
| 模式预测 (ê, τ̂, Δv̂) | `env/t2_probe.py: ExecutorProbe.probe_intent`（3 段 = 48 s，2 s 节点，名义规划轨迹，恒速姿态预报，期限 / 燃料截止） | `_opportunity_eval`、`_certify` | ✓ | ✓ | 独立的规划器实例，不影响执行器求解器状态 |
| 可认证候选集 𝔅_c | `_opportunity_eval`：`event ∧ t + τ̂ + P > L ∧ used + Δv̂ + Δv̂_entry ≤ B_max` | `_certify` | ✓ | ✓ | P = 2π/‖ω‖ |
| 认证映射 𝒞 | `_certify`：原意图有事件即放行；否则取 (‖b − a‖, τ̂) 最小的可认证候选 | `step`（模式 S / H，`certify = True`） | ✓（训练进程 `env.certify`） | ✓（manifest 中 `certify: true`，评估按训练方式） | 原意图不套用关键性 / 燃料条件 |
| 存储提出动作 | `worker`：`transition = (obs, next_obs, action, ...)`，其中 `action` 是策略发来的提议 | 每宏步 | ✓ | — | 回报与后继来自认证后执行 |
| 塑形 r̃ | `step`：`reward + GAMMA_STAGE·φ(x′) − φ(x)`（终局 0）；`potential`（口部 24 s 预报，6.3 m，截断 25 m） | 每宏步 | ✓（`shaping = True`） | ✗（评估 `shaping = False`，报告 `return_true`） | — |
| 真实回报 | `T1Reward`（捕获 +100、超时 / 耗尽 −100、不安全 −150）+ 任务时间 / 燃料代价 | — | — | ✓ | 紧急 0.10 / 10，经济 0.02 / 40 |
| SAC 设置 | `make_model`：[256, 256]、3e-4、10⁵、256、起步 200、`gradient_steps = 2`、`ENT_COEF = 0.05`、τ = 0.005；`SYNC_UPDATES = 100`（采集进程每 100 次更新同步策略） | `train` | ✓ | — | 异步 2 个采集进程 |
| 任务分布（开发期 B 步） | `MixMission(urgent_share = 0.75)`；紧急 U(140, 200) s，经济 U(280, 340) s | 训练进程 | ✓ | 按评估条件 | 正式实验另定 |

**已核对的数值：** 用 `spin_frame` 在 r = 10 m、λ = 60° / 90° / 120°、v_z = ±0.15 m/s 下验证相位与纬向导数（`METHOD.md` 第 2.2 节），与解析式完全一致。
