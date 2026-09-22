# 日志目录地图

> 2026-09-22 静态盘点：`logs/` 下有 88 个一级子目录。这里只分组和标状态，不移动、不删除、不改名；路径仍以文档和 Git 中的原位置为准。

| 状态 | 目录组 | 数量 | 说明 |
|---|---|---:|---|
| **LIVE，勿动** | `adp_rf_262410/`, `adp_rf_262411/`, `adp_rf_262412/` | 3 | 第一轮正式模型、5k/10k checkpoint、训练曲线与 manifest；被 `ADAPTIVE_ROUND1_CLOSEOUT_20260921.md` 按路径/哈希引用。 |
| **EVIDENCE，勿动** | `upper_d2_fallback/`, `upper_d3_slack/`, `upper_na_f1/` | 3 | 第一轮耦合缺陷诊断的上层复核证据。 |
| **CLOSED** | `sac_mpc_v2_arrival_26220{0,1,2}/`, `sac_mpc_v3_bidirectional_26230{0,1,2}/` | 6 | T6–T12 的旧耦合训练线；保留证据，勿据此重开旧接口。 |
| **CLOSED** | `p0_horizons_20260910/`, `p1_a_20260910/`, `p1_b_20260910/`, `p1_c_20260910/`, `p2_feedback_20260910/` | 5 | P0–P2 horizon/脚本/反馈诊断。 |
| **CLOSED** | `attitude_probe_20260912/`, `compute_h35/`, `terminal_gate/`, `t0_amendment_20260910/` | 4 | 姿态、计算量、终端门与 T0 修订证据。 |
| **CLOSED** | `t7_launch_20260911/`, `t7_retry1_20260911/`, `t8_eval/` | 3 | T7/T8 训练与评估历史。 |
| **CLOSED** | `t10/`, `t10_gate_a/`, `t10_headroom/`, `t11/`, `t11_gate_b/`, `t11_interface/` | 6 | T10/T11 闸门与接口历史。 |
| **CLOSED** | `t12_train/`, `t12_eval/`, `t12_eval_v2/` | 3 | T12/S10 训练与正式评估；仍是可复核历史证据。 |
| **CLOSED** | `precap_noncoop/`, `precapture_planning_baseline/`, `precapture_planning_oracle/`, `precapture_planning_v2/` | 4 | 非合作探针、Pure MPC、oracle 与旧 planning 证据。 |
| **CLOSED** | `a2_guidance_free/`, `a3_deployable_planning/`, `g0_perception_closed_loop/` | 3 | A2/A3/G0 感知与规划历史线。 |
| **CLOSED** | `fullmission_sac_200k_seed26083{0,1,2}/` | 3 | `single_phase` 之前的 full-mission 历史线。 |
| **CLOSED** | `gatefree_sac_150k_seed26084{0,1,2}/`, `gatefree_sac_400k_seed26085{0,1,2}/`, `gatefree_sac_400k_fix_seed26086{0,1,2}/`, `gatefree_validation/` | 10 | `single_phase` gate-free 历史训练与验证。 |
| **CLOSED** | `phase2_mission_full_validation/`, `phase2_mission_s1_*/`, `phase2_mission_s1v2_*/` | 21 | Phase-2 mission/S1/S1-v2 历史训练、语义修正、熵与 replay 对照。 |
| **CLOSED** | `phase2_sac_v2_*/`, `phase2_v2_1_mpc_observation_diagnostic/` | 12 | 更早 Phase-2 SAC v2 与观测诊断历史。 |
| **CLOSED** | `hybrid/` | 1 | 旧 hybrid 汇总与逐局证据。 |
| **INVALIDATED** | `invalidated/` | 1 | 已知错误生成的隔离证据，只能用于复查错误机制。 |

合计 `3+3+6+5+4+3+6+3+4+3+3+10+21+12+1+1 = 88`。其中只有三个 `adp_rf_*` 目录属于当前活证据；其余保留原位，不等于相应研究方向仍然有效。

训练产物默认由 `.gitignore` 排除；确需发布的单个证据必须显式 `git add -f <path>`，并在文档中记录路径与 SHA-256，禁止整目录提交。
