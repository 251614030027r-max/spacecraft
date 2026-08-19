# Phase-2 experiment reconciliation note

本说明只解释历史文件之间的关系，不修改任何原始 manifest、checkpoint、monitor、TensorBoard 或评估 JSON。

- `phase2_sac_v2_1_stagea_200k_seed260812`：目录名/请求为 200k，但 manifest 的 `actual_model_timesteps=150000`，并存在 50k/100k/150k checkpoint 与评估；因此最终有效停止点是 150k，不是请求值 200k。
- `phase2_sac_v2_1b_boundedobs_stagea_150k_seed260812`：请求、manifest actual、checkpoint 和候选评估均到 150k，一致。
- `phase2_sac_v2_2_mpc_actor_init_stagea_100k_seed260812`：当前磁盘 manifest 为 `completed`、requested/actual 均 100k，且存在 50k、100k checkpoint/evaluation。任何“在50k停止”的先前口头或中间报告都不是最终磁盘证据；正式引用应以当前 100k manifest 与同目录产物为准。
- `phase2_sac_v2_300k_seed260812`：请求 300k、实际 100k；一次性 densecost/dt_reward/gamma999/Teaching-A/terminalcost/warmup75k 均请求 300k、实际 75k。这些是历史 ablation，不是当前配置候选。
- Phase-1 `sac_final_*` 与 fixed-simple 结果不属于当前受约束 Phase-2 主路径。

当前工程没有 Git repository，因而不能真实提供 Git commit。后续 review package 必须从同一时刻的活动源码、canonical config、测试结果与本 note 构建，并在 package manifest 中记录文件清单；不得用虚构 commit 代替。

## 2026-08-12 历史模型清理

当前所有13个模型目录均属于失败、提前停止、中断或一次性消融运行，没有合格 Pure SAC 基线，也不被 constrained MPC 使用。经明确授权，已删除 `models/` 下68个模型/checkpoint文件，共599,903,219 bytes（572.1 MiB）：Phase-1 delayed curriculum/fixed-simple，以及 Phase-2 V2.0、V2.1、V2.1b、V2.2 actor-init和各75k单因素消融模型。`models/` 清空并保留目录供下一次有效训练使用。对应 `logs/`、manifest、monitor、TensorBoard、evaluation JSON和MPC teacher数据全部保留；因此历史数值结论仍可对账，但不能再从旧模型重新执行策略评估。
