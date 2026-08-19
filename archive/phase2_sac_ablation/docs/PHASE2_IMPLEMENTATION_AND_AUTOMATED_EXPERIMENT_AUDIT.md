# Phase-2 受约束近距离任务：实现、验证与自动训练全过程审计

> 日期：2026-08-12  
> 活动工程：`D:\py\DRL2`  
> 只读参考工程：`D:\py\DRL`（本阶段未写入）  
> 文档用途：供上层模型独立审查 Phase-2 的任务定义、实现真实性、MPC 验收、SAC 失败证据、实验缺陷和下一步选择。本文不把训练回报改善等同于任务成功，也不把尚未验证的原因写成定论。

---

## 1. 执行结论摘要

Phase-2 已完成任务几何、动态可恢复初态采样、环境/观测/奖励/终止、统一诊断以及 constrained MPC V2。MPC 在固定 5-seed 开发集上达到 **5/5 completion、5/5 constraint-success、3,802 控制步零 fallback、零 predicted-safe/truth-violation**，证明 nominal V1 在当前真值动力学、输入限制和完成条件下至少总体可行。

纯 SAC 方面，已执行 30k smoke、100k 主判别以及多轮 75k 高判别力实验，总计约 580k transition。所有有效的固定评估均为零 completion，最简单的永久 Stage-A 教学任务也未形成闭环成功。任务难度、warm-up 长度、终端失败代价、折扣时域和两种奖励尺度修改均未单独恢复能力；部分大尺度奖励实验反而造成 actor/critic 数值失稳。当前证据支持以下有限结论：

1. **任务本身不是已知物理不可行**，因为 constrained MPC 已完成；
2. **当前从随机策略、空 replay buffer 开始的纯 SAC 训练方案不可用**；
3. 证据不足以断言 SAC 算法本身永远无法解决该任务；更可能需要控制先验、示范数据、动作/状态归一化再设计或更系统的 RL 训练诊断；
4. 当前不应继续无边界增加纯 SAC 训练步数或重复随机种子；下一高信息增益路径是 MPC demonstrations / behavior cloning / demonstration replay warm-start，然后在相同环境中做短 SAC 微调。

---

## 2. 原始边界与执行原则

本轮以 `Phase2_受约束近距离任务重构与实现路径.md` 的审查版为执行基准。核心边界如下：

- `D:\py\DRL2` 是唯一活动代码、模型和日志工程；`D:\py\DRL` 严格只读；
- 不把旧质心趋零任务静默包装成新的受约束近距离任务；
- Pure SAC 与 Pure MPC 必须共享任务几何、真值约束指标、初态分布和输入限制；
- V1 不加入 KOZ、复杂障碍、扰动、质量/惯量失配、测量噪声或 safety filter；
- 普通 corridor/FOV/speed 违规记录但不立即终止；严重距离失控或数值异常才终止；
- `constraint_success` 必须由 truth margin 判定，slack 不能把违规伪装成安全；
- 在 MPC 未证明任务总体可行前，不投入 SAC 长训练；
- 训练采用固定窗口评估和自动早停，避免再次运行无信息增益的百万步试错。

---

## 3. Phase-2 nominal V1 任务定义

统一配置位于 `env/task.py` 的 `Phase2TaskConfig`。当前 nominal 定义为：

| 项目 | 数值/语义 |
|---|---|
| docking/capture port | 目标本体系 `[-1.5, 0, 0] m` |
| 期望 chaser 位姿 | 目标本体系位置 `[-3, 0, 0] m`，期望相对姿态为单位阵 |
| approach axis | `[-1, 0, 0]` |
| nominal 初始轴向剩余距离 q | `[3, 7] m` |
| nominal 最大相对姿态 | `60 deg` |
| target tumble scale | `0.25` |
| corridor half-angle | `35 deg` |
| camera FOV half-angle | `50 deg` |
| 总位置变化率上限 | `0.35 m/s` |
| closing-speed envelope | `min(0.30, 0.08 + 0.04 max(q,0)) m/s` |
| 输入限制 | 每轴 force `±5 N`，torque `±0.6 N·m` |
| 完成位置误差 | `≤0.25 m` |
| 完成姿态误差 | `≤10 deg` |
| 完成位置变化率 | `≤0.05 m/s` |
| 完成相对角速度 | `≤0.02 rad/s` |
| 保持时间 | `1.0 s`，dt=0.1 s 时连续10步 |

相对位姿继续使用 `T_rel = T_t^{-1} T_c`，relative twist 继续采用当前动力学定义。任务约束不直接使用 SE(3) logarithm 的平移坐标 `rho`，而使用重构后的 raw target-frame position、LOS 和 `p_dot = R_rel v_rel`。

---

## 4. 实际代码修改

### 4.1 统一任务几何和真值指标

新增 `env/task.py`：

- `Phase2TaskConfig`：冻结 port、desired pose、approach axis、FOV/corridor/speed、完成阈值；
- `TaskMetrics`：输出位置、目标误差坐标、corridor/FOV/速度 margins、完成状态与统一 `constraints_satisfied`；
- `compute_task_metrics()`：控制器无关、环境与评估共享的 truth 计算入口。

重要坐标约定经过数值测试：

- `p_dot_target = R_rel @ relative.velocity`；
- FOV 使用 chaser 本体系 camera boresight 与 port LOS；
- corridor 使用目标本体系 port displacement 的轴向和横向分解；
- desired error 使用 `T_d^{-1} T_rel`，但 MPC predictor 内部状态仍保持 raw `se3_log(T_rel)`，没有把两种状态语义混用。

### 4.2 公共相对状态逆变换

在 `dynamics/relative.py` 新增 `reconstruct_chaser_state()`，精确逆转 `relative_state(target, chaser)`。MPC prediction 和 Phase-2 sampler 共用该函数，避免控制器模块反向成为环境依赖。

### 4.3 初态采样

在 `env/scenarios.py` 新增 `sample_phase2_chaser_state()`：

- 在 q、corridor 横截面、姿态、位置变化率和相对角速度的任务空间直接采样；
- interior 采用面积覆盖；约 25% nominal 样本位于 corridor radial fraction 0.75--0.95；
- 姿态通过 rejection 保证 FOV truth-feasible；
- 所有返回样本必须通过统一 `compute_task_metrics()`；
- 在发现近边界位置与向外横向速度可能“几何可行但动力学来不及恢复”后，加入保守制动距离条件。该条件使用 `0.03 m/s²` 的 corridor recovery acceleration，小于单轴最大 `5/106≈0.0472 m/s²`，没有放宽物理约束。

10,000 次纯采样最终统计：

- feasible：`10000/10000`；
- q 范围：`3.00013--6.99997 m`；
- near-boundary fraction：`0.253`；
- 最小 FOV margin：约 `1.55e-5 rad`；
- 最小 total-speed margin：约 `0.0242 m/s`；
- 最小 closing-speed margin：约 `0.0313 m/s`；
- 最小制动恢复 residual：约 `7.90e-5`，无负值。

### 4.4 Phase-2 环境

在现有 `SE3RendezvousEnv` 中通过 `phase2_enabled` 和单独 task config 接入，没有复制第二套长期环境。主要变化：

- observation 从旧12维变为16维：desired pose error 6维、relative twist 6维、corridor/FOV/total-speed/closing-speed signed margins 4维；
- reset 使用 Phase-2 task-feasible sampler；
- reward 使用 desired-pose task potential、进展差分、持续状态代价、动作代价和透明 constraint hinge components；
- `info` 输出完整 task/constraint metrics、每类 violation steps、maximum violation、constraint-success、完成保持和 reward breakdown；
- 普通约束违规不中止；distance failure 终止；
- severe distance failure 当前保留固定 `-100` 终端代价，防止通过提前终止逃避剩余阶段代价；
- Phase-1 环境和旧测试仍保留为 regression 模式。

### 4.5 训练与评估入口

`train/train.py` 和 `eval/evaluate_policy.py` 已支持 `--phase2`：

- 训练/评估环境由 manifest 严格恢复，包括嵌套 `Phase2TaskConfig`；
- fresh initialization、空 replay buffer、不支持 resume；
- checkpoint、Monitor、TensorBoard、manifest 均按独立 run 保存；
- 固定 Stage-A 和 nominal seed 评估；
- 连续失败窗口自动早停；
- Phase-2 Monitor 保存 completion、constraint-success 和四类违规步数；
- 未达到门槛的模型只作为诊断产物，不标记为合格模型。

---

## 5. 测试和静态验证

新增/扩展测试覆盖：

- desired state 的 completion/constraint truth；
- corridor/FOV/total-speed/closing-speed margin 符号；
- `p_dot` 与有限差分一致；
- relative-state reconstruct round trip；
- 500样本 reproducibility、q/姿态/near-boundary 覆盖；
- 10k纯采样统计；
- Gymnasium Phase-2 API、16维 observation；
- zero/random action smoke；
- 普通违规记录但不立即终止；
- 1 s completion hold；
- Phase-2 manifest 嵌套配置恢复；
- MPC constraint margin/Jacobian 一致性；
- constrained MPC 单步求解和输入限制；
- distance-failure terminal cost；
- Stage-A 到 nominal 只在 episode boundary 切换。

测试数量随实现增长：首批几何后53项，环境/训练入口后59--60项，最终回退后的活动代码为：

```text
62 passed in 16.38s
```

当前目录不是 Git repository，因此没有 commit/diff/hash 证据；本轮遵循用户要求，没有用源码哈希或重复多种子兜底替代功能验证。

---

## 6. Constrained MPC V2 实现与问题定位

### 6.1 实现

主要文件：

- `controllers/mpc/config.py`：Phase-2 reference state、task、slack、constraint tightening；
- `controllers/mpc/constraints.py`：truth margins、保守多面体 corridor、逐节点局部 affine constraint linearization；
- `controllers/mpc/controller.py`：目标从相对质心零状态改为 `T_d` reference；加入逐节点 state constraints、显式 bounded slack、constraint timing、slack 和 predicted margin diagnostics；
- `experiments/evaluate_mpc.py`：Phase-2 评估、完整 command runtime、fallback、truth constraint-success 和 predicted-safe/truth-violation 对照。

约束表达：

- corridor：目标本体系圆锥的内接规则多面体近似；
- FOV、total speed、closing speed：统一 signed truth margin 的局部一阶线性化；
- 所有最终安全结论仍以 nonlinear truth metrics 判定；
- 使用 `constraint_tightening=0.02` 消除已观察到的少量速度节点真值越界；
- slack 保持显式统计，不把 slack 后的优化可行称为 constraint-success。

### 6.2 OSQP 失败过程

初版 5-seed 评估中4/5完成，但 seed `260814` 从初始步开始频繁 `user_limit`，最终380次 fallback、distance failure，总 fallback rate `10.23%`，并出现7次 predicted-safe/truth-violation。该 seed 初始 truth margins 全为正，最小约0.133，说明不是初始几何违规。

单步对照：

- `slack_weight=1e4`：OSQP 10,000 iterations，`user_limit`，zero fallback；
- 降到 `1e3` 或更低：能够返回解，但20 s闭环出现46--48个 corridor truth violation，不能作为合格修复；
- 保持同一 QP、同一 `slack_weight=1e4`，仅切 CLARABEL：单步13 iterations、optimal；20 s闭环零 fallback、零 truth violation。

因此 Phase-2 constrained MPC 默认使用 CLARABEL；旧无状态约束 MPC 的 OSQP regression 配置未删除。该变更记录也写入 Phase-2 路线文档。

### 6.3 最终 MPC 门槛

正式开发结果：`logs/phase2_mpc_v2_clarabel_5seeds.json`。

| 指标 | 结果 |
|---|---:|
| completion | `5/5 = 100%` |
| constraint-success | `5/5 = 100%` |
| distance/time failure | `0/5` |
| fallback | `0 / 3802 steps` |
| predicted-safe/truth-violation | `0 / 3802` |
| completion time | `63.2--83.7 s` |
| solver time median / P95 | `0.0272 / 0.0296 s` |
| full command time median / P95 | `0.3065 / 0.4334 s` |

解释边界：这只是5-seed开发门槛，不是论文级大样本验收；但足以证明当前 nominal V1 不是显然不可行，并允许进入 SAC 判别。

---

## 7. SAC 自动训练全过程

### 7.1 固定 SAC 基线

除明确列出的单因素实验外，SAC 使用：

- learning rate `2e-4`；
- buffer `300k`；
- learning starts `20k`；
- batch `256`；
- tau `0.005`；
- gamma 基线 `0.993`；
- train_freq `4`，gradient_steps `4`；
- entropy `auto_0.01`；
- network `[256,256,256,256] ReLU`；
- CUDA；
- fresh initialization、empty replay。

### 7.2 运行汇总

| run | 实际步数 | 主要变化 | 固定评估 | 结果解释 |
|---|---:|---|---|---|
| `phase2_sac_v2_smoke_30k_seed260812` | 30k | 25k Stage-A 后 nominal | nominal 0/5、全距离失败 | CUDA/replay/update/checkpoint/stage switch 链路可用；只有约10k更新，不作收敛判断 |
| `phase2_sac_v2_300k_seed260812` | 100k early-stop | 原始 Phase-2 SAC | 50k nominal 0/10；100k nominal 0/10；均全距离失败 | 训练回报改善未转化为任务指标；自动停止避免跑满300k |
| `phase2_sac_v2_warmup75k_seed260812` | 75k early-stop | warm-up 延至75k | 回调记录0/10 | **评估纯度有缺陷**：第1回合后误切 nominal，不能作为纯 Stage-A 0/10证据 |
| `phase2_sac_v2_terminalcost_seed260812` | 75k early-stop | 加剩余时域终端失败代价 | 回调记录0/10 | **同样受上述评估纯度缺陷影响**，只说明混合门槛失败 |
| 独立只读 checkpoint 评估 | 非训练 | 原50k/100k模型强制永久 Stage-A | 两个模型各0/5 constraint/completion | 支持原策略没有保留简单任务能力，但样本仅5 |
| `phase2_sac_v2_teachingA_seed260812` | 75k early-stop | q=0.5--1.5m、15deg、tumble0.10、宽约束、无near-boundary；评估修成永久 Stage-A | **纯 Stage-A 0/10、全距离失败** | 强证据：显著降低任务难度仍未形成基础闭环 |
| `phase2_sac_v2_dt_reward_seed260812` | 75k early-stop | task/constraint cost 改为 dt 积分 | 纯 Stage-A 0/10 | critic loss约`1e3--1e5`、actor loss约800，数值失稳；已回退 |
| `phase2_sac_v2_gamma999_seed260812` | 75k early-stop | 恢复稳定reward，gamma=0.999 | 纯 Stage-A 0/10 | 将有效折扣时域从约14.3s扩至约100s仍未恢复；已回退 |
| `phase2_sac_v2_densecost_seed260812` | 75k early-stop | 在gamma999基础上采用密集task/constraint cost、固定−100 failure | 纯 Stage-A 0/10 | actor/critic再次大尺度失稳；已回退 |

说明：最后一轮相对 `gamma999` 只改变 reward，但相对原始基线同时含 gamma999 和 dense reward，因此不能用它单独估计 dense reward 在 gamma0.993 下的效果。该轮的有效结论仅是“这个组合失败且数值不稳定”。

### 7.3 主失败轨迹

对 `phase2_sac_v2_300k_seed260812` 的50k和100k checkpoint，在固定 nominal seed `20288000` 上逐步检查：

- 初始 position error约`6.98 m`；
- 50k策略最佳接近到`2.57 m`，100k策略最佳接近到`4.24 m`；
- 通常前10--20s朝目标接近，随后 corridor/FOV/速度逐步越界；
- 最终均到约28--29m position error并触发 distance failure；
- 动作绝对均值约0.37--0.65 normalized，非“完全零动作”；
- 失败来自不会稳定制动/保持和长期约束恢复，而不是策略完全没有产生控制。

100k固定回合 reward breakdown 累计约：

- task dissipation：`-2.61`；
- task penalty：`-3.96`；
- actuation penalty：`-1.00`；
- corridor penalty：`-9.63`；
- FOV penalty：`-1.94`；
- total-speed penalty：`-2.57`；
- closing-speed penalty：`-0.006`。

这说明约束惩罚并非完全缺失，但也不能仅从累计值证明相对尺度最优，因为折扣、replay分布和函数逼近共同影响策略。

---

## 8. 已证实、被证伪与未解决问题

### 8.1 已证实

- Phase-2 nominal V1 在当前真值模型和输入限制下至少总体可行；
- task geometry、相对坐标、FOV/corridor/速度 truth metrics 和 sampler 通过数值/回归测试；
- constrained MPC 可在5个固定 seed 完成且保持约束；
- OSQP 在该高权重软约束 QP 上存在真实数值退化，CLARABEL 在同一 formulation 上稳定；
- SAC 训练链路、CUDA、replay、更新、checkpoint、manifest和固定评估均实际运行；
- 当前合格 SAC 模型不存在；所有训练模型只能作为失败诊断产物。

### 8.2 已被当前实验否定为“充分原因”

以下因素单独修改后仍未恢复成功，因此不能再把它们作为唯一主因：

- 25k curriculum switch 过早；
- Stage-A 任务仍太难；
- 缺少 distance failure terminal cost；
- gamma0.993 的折扣时域太短；
- 仅把 task/constraint penalty 放大到 dt 积分。

注意：“不是充分原因”不等于“完全无影响”。例如更长 warm-up 可能仍是示范预训练后的合理配置。

### 8.3 尚未解决、需要上层重点审查

1. **随机探索与受约束六自由度任务的匹配性**：SAC 在20k learning-starts前收集的大量随机失控 transition 是否使 replay 长期被远离任务的分布主导？
2. **动作/动力学时间尺度**：normalized action 直接对应±5N/±0.6Nm，随机策略对106kg航天器可快速注入大速度；是否需要控制增量、低通动作、较低训练专用动作范围或 policy action repeat 重新设计？这些会改变训练接口，尚未实验。
3. **observation 表达**：当前16维包含 desired SE(3) log error、body relative twist和4个margins；是否缺少 raw target-frame position/rate 或 task-axis分量，导致网络难以从log坐标恢复约束几何？
4. **reward 数值条件**：当前势能、差分、hinge和终端项的尺度虽透明，但没有做系统的Q-value/reward normalization、critic target范围和梯度统计；大尺度实验已显示critic可失稳。
5. **SAC超参数适配**：当前参数继承旧任务，未系统研究自动熵、learning starts、更新数据比、网络规模和reward scaling对Phase-2的适配；本轮刻意避免多因素试参。
6. **成功稀疏性**：即使任务势能连续，随机策略几乎不会进入1s完成保持集；是否需要MPC demonstrations、behavior cloning或HER类目标重标记？当前任务目标固定，HER是否适用需单独判断。
7. **MPC计算速度**：full command median约0.306s，超过0.1s控制周期；开发门槛证明可行但未达到实时性。solver本身median约0.027s，主要瓶颈仍是逐节点constraint Jacobian和外部计算。
8. **样本规模边界**：MPC只有5-seed开发验收；SAC固定门槛多为10 seed。它们足够用于开发决策，不足以支撑论文统计结论。

---

## 9. 当前活动代码状态

为了避免把失败实验留下作为默认值，最终已回退：

- Phase-2 `gamma` 回到基线 `0.993`；
- reward state/constraint step weight 回到稳定的 `dt/30s`；
- Stage-A 恢复审查版附近的 q=2--4m、max attitude45deg、tumble0.20、corridor45deg、FOV60deg、near-boundary0.15；
- dense-cost 与 teaching-A 极简数值不再是默认配置。

仍保留：

- fixed `-100` severe distance-failure terminal cost；
- 75k Stage-A配置和永久Stage-A固定评估口径；
- nominal固定评估和连续失败早停；
- 全部Phase-2环境、采样、task metrics和constrained MPC实现。

当前代码可通过62项测试，但**不建议直接再次启动当前纯SAC训练**；测试通过只说明实现一致性，不说明RL收敛。

---

## 10. 建议的下一最小实验路线

建议不再运行新的随机初始化纯SAC长训练。下一步应利用已验收 MPC 的同任务轨迹：

1. 用固定 nominal 与 near-boundary seeds 导出每步16维 observation、6维 normalized MPC action、truth margins、completion和solver diagnostics；
2. 检查示范数据动作饱和率、状态覆盖和constraint margins，不先训练；
3. 对现有 SAC actor architecture 做有界 behavior cloning，使其在固定示范验证集上能复现动作和闭环可行性；
4. 只有BC policy在独立固定seed上获得非零completion/constraint-success后，才进行短SAC微调；
5. SAC微调继续使用现有50k固定评估和早停；比较是否保持MPC可行性、是否降低推理时间或控制消耗；
6. 若BC闭环本身无法复现MPC，优先检查observation是否缺少raw geometry、actor容量和动作回归，而不是继续调SAC奖励。

该路线会从“纯SAC能否从随机探索学出任务”转向“在MPC证明可行且提供先验后，SAC能否学习可执行近似并产生互补价值”。这与当前证据和后续SAC--MPC研究主线一致。

---

## 11. 关键证据路径

- Phase-2执行方案：`C:\Users\35884\Documents\Spacecraft\Phase2_受约束近距离任务重构与实现路径.md`
- 简要训练结论：`docs/PHASE2_AUTOMATED_TRAINING_FINDINGS.md`
- 本过程审计：`docs/PHASE2_IMPLEMENTATION_AND_AUTOMATED_EXPERIMENT_AUDIT.md`
- MPC最终结果：`logs/phase2_mpc_v2_clarabel_5seeds.json`
- MPC失败对照：`logs/phase2_mpc_v2_dev_5seeds.json`
- 30k smoke：`logs/phase2_sac_v2_smoke_30k_seed260812/`
- 原100k早停：`logs/phase2_sac_v2_300k_seed260812/`
- 75k warm-up：`logs/phase2_sac_v2_warmup75k_seed260812/`
- terminal cost：`logs/phase2_sac_v2_terminalcost_seed260812/`
- teaching Stage-A：`logs/phase2_sac_v2_teachingA_seed260812/`
- dt reward：`logs/phase2_sac_v2_dt_reward_seed260812/`
- gamma999：`logs/phase2_sac_v2_gamma999_seed260812/`
- dense cost：`logs/phase2_sac_v2_densecost_seed260812/`

---

## 12. 给上层审查者的直接问题

请重点审查以下判断，而不是只看“0成功”：

1. 由5-seed MPC证明任务可行，再由永久Stage-A 0/10判断纯SAC训练结构失败，这一证据链是否足够支持转向demonstration warm-start？
2. 16维 observation 是否存在结构性不可辨识/不利学习问题，特别是SE(3) log translation与raw corridor geometry之间的映射？
3. 当前动作范围和随机探索是否使learning-starts前replay分布严重失真？应优先改动作参数化、训练动作限幅，还是优先使用MPC示范？
4. fixed `-100` distance-failure cost是否应保留，或应改为严格的absorbing-state remaining-horizon cost并配套reward normalization？
5. MPC demonstration应进入behavior cloning、replay prefill、Q-filter imitation，还是仅用于curriculum初态/动作尺度校准？哪一种是最小且最能区分原因的实验？
6. 在继续任何SAC实验前，是否应加入critic target、Q magnitude、gradient norm、entropy/action saturation和replay state-distribution统计？

这些问题仍是开放项；本文没有把它们伪装成已确认根因。
