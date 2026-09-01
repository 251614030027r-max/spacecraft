# 历史演进摘要

本文件压缩原 `docs/history/` 的逐轮阶段报告。它用于解释工程为什么演进到当前状态，不是当前结论来源；现行结论只认根 `CLAUDE.md`、`HANDOFF.md` 和 `docs/PROBE_RESULTS.md`。原始阶段报告可从整理前提交 `0665a81` 的 Git 历史恢复。

## 2026-08-12：初始受约束 Phase-2

工程加入走廊、FOV、速度包络和 MPC 约束，尝试 V2.0/V2.1、bounded observation、MPC actor initialization、dense cost、dt reward、gamma 与 warmup。多项因素混杂且 Gate 为零，促使后续改为严格单因素。

对应证据：`logs/phase2_sac_v2_*`、`logs/phase2_v2_1_mpc_observation_diagnostic/`、`logs/phase2_mpc_v2_*.json`。

## 2026-08-13：两阶段任务与语义修正

任务被拆成 Phase-I Gate acquisition 与 Phase-II terminal approach。旧 0.50 m/s 终止形成提前失败捷径，随后改为软速度代价与 1.0 m/s catastrophic guard；奖励改为有界、折扣一致的势函数。可达性脚本证明任务接口可达，但 SAC 仍未获取 Gate。

对应证据：`logs/phase2_mission_s1_*`、`logs/phase2_mission_s1_semanticfix_validation/`。

## 2026-08-14：干净 Pure SAC 单因素

退出自适应课程并从零训练，依次检查自动熵、局部进展、20°方向锥和 body-frame 平动观测。body-frame 表达首次产生约 1 m 级趋近，证明状态/动作坐标对齐是有效因素，但未形成完整 Gate 控制。

对应证据：`logs/phase2_mission_s1_fullcanonical_*`、`logs/phase2_mission_s1_cone20_*`。

## 2026-08-15 至 08-20：S1-v2 与固定熵

S1-v2 将任务集中为 10–14 m、窄方向锥、低初速和 acquisition Gate；显式速度误差提高低速率但牺牲位置进入率。后续证据识别自动熵温度增长为 critic 过估计驱动因素，固定 `ent_coef=0.005` 后三种子 Phase-I Waypoint 表现显著改善，但跨种子仍不稳定。

对应证据：`logs/phase2_mission_s1v2_*` 及相应本机模型。

## 2026-08-21 至 08-26：统一 single_phase 基准

Waypoint 被确认是诊断脚手架，不应成为主任务的一部分。共同基准改为约束从第一步生效的 `single_phase`。Pure SAC 在统一评价块上完成 `3/1/0`，无违约 `20/10/0`；Pure MPC 与脚本控制器均完成 20/20。主表口径统一为完成时间、完成回合推力冲量、所有回合最差约束裕度和控制器 p95/max 计算时间。

对应证据：`logs/gatefree_sac_400k_fix_seed26086*`、`logs/sac_seed26086*.json`、`logs/mpc_single_phase.json`、`logs/scripted_single_phase.json`。

## 2026-08-30 至 09-01：四个缺口探针

依次完成学习凸终端值、目标相位采样、目标模型失配和目标状态观测误差探针。观测误差轮发现并修正了姿态偏置错误旋转惯性速度的问题，修正后四轮均为负，hybrid 训练停止。详见 `docs/PROBE_RESULTS.md`。
