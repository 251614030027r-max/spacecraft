# S1 learning-fix qualification — 2026-08-14

## 结论

当前 60k smoke 模型不是合格 S1：独立 20-seed deterministic canonical evaluation 为 Gate 0/20，17 次 catastrophic-speed failure。可交付的是下一次从零 S1 长训练方案，而不是 smoke 模型。该方案已通过 truth、reward、课程语义、回归和三轮受控 smoke 的因果筛选；不得把工作区 smoke 模型复制进 canonical `models/`，不得由 S1 smoke 进入 S2。

## 已定位并修正的因果问题

1. 原 Phase-I reward 使用速度模长，不能区分朝向/背离 Gate；现使用 Gate-directed desired-velocity field、bounded discount-consistent potential `C(s)-gamma*C(s')`，并保持 Phase-II reward、Gate、truth、约束和 Pure SAC 超参数不变。
2. 固定时间课程在 difficulty≈0.02 后仍继续扩展，明显跑赢 actor；现改为每 20 个非 rehearsal episode 按成功率推进/回退：`>=0.80` 加 `0.005`，`<0.50` 减 `0.005`，否则保持。
3. 旧的 2% early full rehearsal 在学习开始前实际占 replay transitions 的 92.1%，因为 full failure episode 远长于 easy success；现 early full probability 为 0，并随 frontier 平滑升至 0.10，canonical periodic evaluation 始终使用完整原分布。
4. 旧 easy envelope 的预置 0.05 m/s 朝 Gate 速度让随机/零动作几乎自动成功，critic 无法学习动作排序；truth sweep 选择中心距离 9.02–9.05 m、速度 0.03 m/s：固定 20 seeds 下 zero 6/20、逐步 uniform-random 14/20、Gate-PD 20/20，平均 episode 长度分别约 144、72、9.65 steps。

## 资格证据

- 全量回归：87 passed。
- 正式目录部署后 validate-only 通过，全量回归再次 87 passed；Gate-PD truth 20/20，平均到达 58.18 s、最大速度 0.1973 m/s，三组 reward preference 最小 discounted-return 余量 15.998 且全部通过。
- 最终隔离 smoke：`s1_learningfix_actioncontrast_smoke60k_seed260813`，474 episodes、350 Gate successes；curriculum frontier 峰值 0.05、终点 0.045，没有退回零难度。
- 同一组 periodic full-distribution deterministic 5-seed evaluations：Gate minimum-distance mean `10.606 -> 9.809 -> 8.950 m`（20k/40k/60k），60k best `0.801 m`；Gate rate仍为0，因此只能判定方向有效，不能判定 S1 成功。
- critic Q-minus-short-MC：25k `0.029`，50k full-state probe `1.908`；后者说明未覆盖完整分布仍有外推高估，正式长训必须依赖自适应课程扩大覆盖，不能恢复 smoke 模型。
- 独立 final 20-seed canonical evaluation（seed 264500）：Gate 0/20，minimum Gate distance mean 9.089 m、best 2.295 m，17 speed failures、2 distance failures，mean action saturation 0.135。

## 下一次且仅一次长训练

从零启动，不恢复模型或 replay，不中途调参：

```powershell
python -B -m train.train --steps 250000 --seed 260814 --run-name phase2_mission_s1_adaptive_actioncontrast_seed260814 --mode phase1_pretrain
```

训练中只读观察。最终必须用 canonical deterministic 20–30 seeds 核验 Gate acquisition、到达时间、Gate 入场姿态/速度/角速度、失败类型、actor/Q/entropy 与动作饱和。只有 Gate acquisition 稳定非零并呈上升趋势，且入场状态满足联合 Gate 条件，才允许 actor-only 进入 S2；否则停止在 S1 交由上层纠察，不继续堆长训练。
