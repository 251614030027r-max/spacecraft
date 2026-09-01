# 交接：观测误差探针的 bug 更正 与 路线 B（给下一个窗口）

> 首先通读仓库根 `CLAUDE.md`（权威）与 `HANDOFF_sac_mpc_coupling.md`（方法层）。本文件补
> 本窗口的进展与**一处推翻结论的 bug 更正**，确保衔接。分支
> `claude/sac-mpc-coupling-design-ns7g6i`，本会话无 push 权限，代码以 zip 交用户本地 commit。
> `python -B -m pytest -q` 应 **147 passed**。

## 0. 一句话现状

**四个"寻找 hybrid 缺口"的探针（值耦合 / 相位采样 / 目标模型失配 / 目标状态观测误差）在修完
一处 bug 后全部为负。** 短 horizon 约束 MPC（+ 经典 EKF predict 处理时延）对相位/惯量/姿态偏置/
时延都鲁棒、实时、安全；Pure SAC 是不安全弱基线。**当前没有留给 hybrid 的缺口 → 诚实落点是路线 B**
（两基线诚实论文 + Λ>1 regime 刻画 + 多个负结果）。**训练已叫停**（其依据的"部分可观缺口"是 bug）。

## 1. 本窗口关键事件：观测误差 bug 与更正（最重要）

- **探针4（观测误差）曾报"现实姿态偏置击穿纯 MPC、偏置不可观、任何滤波器补不掉" = 首个真实缺口**，
  上层据此批了训练。
- **随后在建"可行性前沿"工具时发现矛盾**（0.01° 偏置→total_speed 违约，但名义裕度 +0.14、机制估算
  仅 1e-4 m/s），查出 bug：`env/observation_error.py::TargetStateEstimator._biased` 旋转姿态却保留
  本体系速度 → **估计的惯性速度被偏置旋转，把 ~7.6 km/s 轨道速度的一部分注入成虚假相对速度**
  （0.05°→~6.6 m/s）→ 推力饱和 → total_speed 违约。**"偏置缺口"整条由此而来。**
- **已修**（commit `3470634`）：只旋转朝向，保持惯性平动/角速度不变；加回归测试。修后重测：

| 修正模型（MPC local, 3ep, seed 970000） | raw | EKF |
|---|---|---|
| 姿态偏置 1°（前沿工具到 5°） | **完成，0 违约** | — |
| 延迟3+更新率3 | 0 违约 | **完成，0 违约** |
| 现实档 1°+延迟+更新率 | 0 违约 | **完成，0 违约** |

- **结论**：正确建模的姿态偏置无害（≥5°）；唯一真实观测误差是**时延/低更新率**，且**经典 EKF
  predict 步完全补掉**。→ 无 hybrid 缺口。（此前的"偏置不可观证明" `experiments/bias_observability.py`
  数学仍成立，但它证的是"偏置不可观"，而修正后偏置本就无害，所以不再是缺口来源。）

## 2. 未决 / 下一步（等上层指令）

按优先级，等上层新指令后择一：
1. **量测噪声快验证（唯一没测的杆，~30min，无训练）**：零均值高斯位姿噪声注入，纯 MPC + EKF。
   用户此前已指出零均值噪声可滤，预期负；验完彻底收口"没有别的误差没试"。
2. **路线 B 诚实论文骨架**：四负结果 + 强基线（短 horizon 约束 MPC 实时+安全，EKF 处理时延）
   vs Pure SAC 不安全彩票 + Λ>1 regime 定量刻画（Λ=ωr/v_max，12m 处 1.41）+ 诚实边界
   （单一确定翻滚、速度限定义在旋转系等）。
3. **和上层重新对齐**：上层批训练的依据已被更正掉；备忘已更新（见第 5 节链接，顶部红色更正横幅）。

**不要**在上层未重新拍板前起训——训练命令是基于被推翻的缺口给的。

## 3. 本窗口新增/改动的代码（均已本地 commit 于分支）

- `controllers/mpc/`：`terminal_value.py`（学习凸终端值，探针1，负结果）、`config.py`/`controller.py`
  加 `terminal_cost_source` 与 `corridor_speed_fraction`。
- `env/scenarios.py`：`target_initial_state(phase_seed=)` 相位采样、`sample_target_parameters()` 惯量失配。
- `env/se3_rendezvous_env.py`：`phase2_target_phase_sampling` / `phase2_target_model_mismatch` /
  `phase2_observation_bias_rad|delay_steps|update_every`（env 层观测误差，训练用，默认关逐位不变）。
- `env/phase2_env.py`：新任务模式 `single_phase_phase_sampled`。
- `env/observation_error.py`：`TargetStateEstimator`（bias/delay/低更新率 + `filter_mode=propagate` 的
  EKF predict）。**bug 修复在此。**
- `experiments/`：`evaluate_mpc.py`（加 `--linearization-source/--exact-refresh-steps/--target-model-mismatch/
  --controller-model/--obs-*/--target-estimator/--corridor-speed-fraction`）、`fit_terminal_value.py`、
  `diagnose_terminal_value.py`、`refresh_staleness_probe.py`、`feasibility_frontier.py`（**发现 bug 的工具**）、
  `bias_observability.py`、`summarize_obs_grid.py`。
- `train/train.py`：加 `--obs-*`（POMDP 训练入口，未起训）。
- `eval/main_table.py`：compute 改报 p95/max（不报均值）。
- `tests/`：`test_terminal_value.py` / `test_phase_sampling.py` / `test_target_model_mismatch.py` /
  `test_observation_error.py`。147 passed。

## 4. 已确立的正式结论（可写进论文，均无训练）

- **compute 修正**：短 horizon 只砍均值；与 horizon 无关的 exact 刷新步（~250ms）使 p95/max≈2.8×，
  短 horizon 未实现实时；`local`/`refresh100` 可把 p95 压进预算且不掉安全。（`profile_mpc_step` +
  刷新实验；compute 一律报 p95/max。）
- **相位采样**：无缺口（陈旧量 ∝ 冻结 |ω|×horizon，方向改不了）。
- **模型失配**：惯量 ±30% 无缺口（预测误差 ∝ horizon²，1s 内 0.01°）。
- **观测误差**：偏置无害、时延被 EKF 补掉（本窗口更正后）。

## 5. 上层备忘（Artifact 链接，均已更新到最新/含更正）

- 训练投入决策（**顶部已加红色更正横幅、训练暂停**）：
  https://claude.ai/code/artifact/04957ae6-f585-44a9-8f06-468a6ed5a066
- 部分可观诊断（探针4，注意其正文为更正前，以决策备忘的更正为准）：
  https://claude.ai/code/artifact/bd0116cc-fd64-4727-b2c7-9a483069eb88
- 前序：值耦合负结果 https://claude.ai/code/artifact/c5ca92f9-5b7d-4767-bcb5-a523cc38ed56 ；
  模型失配 https://claude.ai/code/artifact/3d4c91c7-3928-4d74-8b3d-0eb429f1b0b4
  （**这两页也在更正前写就，涉及"观测误差缺口"处以决策备忘更正为准**。）

## 6. 纪律（不变）

严格单因子、每因子独立 commit+manifest；结论 ≥3 训练/评测 seed 且报分布不报最好 seed；每次从零；
S1-v2 任务参数冻结；compute 报 p95/max 不报均值；违约按真实几何判；长训练在用户本机、本会话只做
代码/pytest/≤5k smoke/诊断；代码 zip 交付本地 commit，本会话不 push；回答用中文。
