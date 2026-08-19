# Phase-2 自动训练结论（2026-08-12）

## 结论

Phase-2 nominal 任务的物理可行性已经由 constrained MPC 证明，但当前从零、纯 SAC、空 replay buffer 的训练链在多个高判别力实验中均未形成最简单 Stage-A 闭环能力。继续增加训练步数、重复种子或微调课程数值没有证据支持，当前应停止纯 SAC 盲训，转入“利用已通过的 MPC 轨迹建立行为先验/示范数据，再检验 SAC 是否能保持和改进”的下一阶段。

这不是任务不可行：`logs/phase2_mpc_v2_clarabel_5seeds.json` 中 5/5 nominal 初态完成、5/5 全周期约束成功、3,802 控制步零 fallback、零 predicted-safe/truth-violation，完成时间 63.2--83.7 s。

## 已执行训练与判别

| 运行 | 实际步数 | 唯一主要变化 | 固定评估结果 | 判定 |
|---|---:|---|---|---|
| `phase2_sac_v2_smoke_30k_seed260812` | 30k | 全链 smoke | nominal 0/5 | 链路可用，不作收敛判断 |
| `phase2_sac_v2_300k_seed260812` | 100k early-stop | 原 Phase-2，25k 切 nominal | 50k、100k 均 nominal 0/10，全部距离失败 | 课程主体无学习 |
| `phase2_sac_v2_warmup75k_seed260812` | 75k early-stop | 只延长 Stage A | Stage A 0/10 | 切换过早不是充分原因 |
| `phase2_sac_v2_terminalcost_seed260812` | 75k early-stop | 加终端失败代价 | Stage A 0/10 | 提前终止套利不是充分原因 |
| `phase2_sac_v2_teachingA_seed260812` | 75k early-stop | q=0.5--1.5 m、15 deg、tumble=0.10 的教学任务 | 纯 Stage A 0/10 | 任务难度不是主因 |
| `phase2_sac_v2_dt_reward_seed260812` | 75k early-stop | 状态代价改为 dt 积分 | Stage A 0/10；critic loss 约 1e3--1e5 | 奖励尺度过大，已回退 |
| `phase2_sac_v2_gamma999_seed260812` | 75k early-stop | gamma=0.999 | Stage A 0/10 | 长期折扣不足不是充分原因，已回退 |
| `phase2_sac_v2_densecost_seed260812` | 75k early-stop | 密集状态/约束代价 | Stage A 0/10；actor/critic 大尺度失稳 | 已回退 |

所有训练均为从零初始化；没有复用失败模型或 replay buffer。每轮只在固定门槛失败后才实施下一项单因素变化。失败模型保留为诊断产物，不得标记为有效基线。

## 保留实现

- 统一 `Phase2TaskConfig`、truth task metrics、16 维 observation、完成保持和违规累计诊断；
- 10,000/10,000 几何可行且满足保守制动距离条件的任务空间初态采样；
- constrained MPC V2 和 CLARABEL 数值求解路径；
- Phase-2 SAC 训练清单、检查点、固定 Stage-A/nominal 评估和早停；
- 固定 `-100` 严重距离失败代价，避免通过提前结束逃避余下任务代价；
- 审查版 Stage A 数值已恢复；`gamma` 和 reward 积分尺度已恢复稳定基线。

## 下一最小路线

下一步不再从随机策略直接烧 1M SAC。应先把已验收 MPC 的 nominal/near-boundary 轨迹导出为统一 observation/action 数据，做一个有界的 actor 行为克隆预训练或 replay-buffer demonstration warm start；随后在完全相同 Phase-2 环境中短 SAC 微调，并继续使用现有固定门槛判断“能否保持 MPC 已有可行能力、是否在控制消耗或运行时上改善”。这将直接服务后续 SAC--MPC 互补研究，也比继续对纯 SAC 奖励和课程做无证据试参更有信息增益。

## V2.1更新（2026-08-12）

上层指导后的V2.1已完成，详见`docs/PHASE2_SAC_V2_1_EXECUTION_REPORT.md`。首次23D observation实现存在port平面后方向特征越界bug，会让critic在约8k直接达到10^29量级；该bug已修复并由极端逃逸测试覆盖。修正版同seed固定Stage-A训练前期数值稳定，150k仍连续3次固定10回合零完成、全距离失败；replay全约束满足比例由25k的21.0%降至150k的13.9%，`<1m`样本始终不超过0.22%，Q均值后期漂到约7800。6条分层MPC teacher轨迹全部完成、约束成功且0 fallback；新23D相对旧16D的整轨迹留出动作MAE改善约19.5%，但两种回归策略闭环均失败。因此下一步优先demo replay prefill或最小actor初始化，不把简单开环BC拟合等同于闭环可行性，也不再追加Pure SAC长训练。
