# 实验与证据索引

## 代表模型（保留）

| 目录 | 文件 | 用途 | 结论 |
|---|---|---|---|
| `models/phase2_mission_s1v2_acquisition_bodyobs_autoent_tau001_seed260815/` | `checkpoints/sac_400000_steps.zip` | 空间接近最佳对照 | 20 seeds中10次进入位置球、3次低速、Gate 0 |
| 同上 | `final_model.zip` | v2遗忘对照 | 500k平均靠近仅0.181 m |
| `models/phase2_mission_s1v2_velerror_bodyobs_autoent_tau001_seed260815/` | `checkpoints/sac_200000_steps.zip` | 速度控制最佳对照 | fixed Gate 1/10；独立20 seeds低速19/20、位置球0 |
| 同上 | `final_model.zip` | v3遗忘对照 | 500k平均靠近0.262 m，Gate 0 |

这些模型都不是合格S1，禁止用于S2初始化。

旧单因素模型与两组主线的非代表checkpoint没有永久删除，已迁入`legacy/model_history/`。它们仅用于历史追溯，不是当前候选模型；原始实验结论应优先查看对应`logs/`和`docs/history/`。

## 活动主线日志

- `phase2_mission_s1_seed260813`至`semanticfix/rewardfix/stable`：两阶段任务建立、终止与reward语义修正；
- `phase2_mission_s1_fullcanonical_*`：退出adaptive curriculum后的标准全分布对照、automatic entropy和local progress；
- `phase2_mission_s1_cone20_*`：方向锥、body-frame observation和tau稳定性单因素；
- `phase2_mission_s1v2_acquisition_*`：S1-v2空间获取实验；
- `phase2_mission_s1v2_velerror_*`：S1-v2显式速度误差实验。

每个正式目录保留manifest、monitor、`phase2_diagnostics.json`、TensorBoard event和周期evaluation JSON。日志总量很小，全部保留以支持复核，不用失败模型文件代替证据。

## 更早的Phase-2消融日志

`phase2_sac_v2_*`记录2026-08-12阶段的V2.0/V2.1、bounded observation、MPC actor initialization及多项75k消融。它们不是当前配置，但用于说明为什么项目转向干净单因素和两阶段mission。请求步数与实际步数差异见`history/EXPERIMENT_RECONCILIATION.md`。

## 历史文档与早期复现

`docs/history/`保存P0 truth/Gate/MPC验证和每次关键单因素的原始说明；它们按时间保留，不代表当前入口。`legacy/documents/`保存早期REPRODUCTION_SPEC、DEVIATION_LOG和复现脚本；`legacy/reference_baseline_evidence/`保存选定原始证据；`legacy/reference_model/`保存一个早期参考模型；`legacy/model_history/`保存从活动模型目录移出的失败模型和非代表checkpoint；`legacy/log_history/`保存8组更早的Phase-I完整日志。`legacy/`不参与当前import。
