# 择时价值实验：本地 foundation 整合与启动闸门

日期：2026-09-16 JST。本文件记录已完成的代码整合和未完成的实验前置；不是 A/B 结果，也不授权训练。最新方向来自上层的 `下层执行_择时价值确认实验_20260916.md`，优先于仓库内较早的双向残差训练执行单。

## 已落地与验证

在 `claude/sac-mpc-coupling-design-ns7g6i` 上、已推送的探针 2 结论提交 `8654884` 之后，以 `git am -3` 整合上层 `task_restructure_foundation.patch` 的全部六个提交；没有 reset、覆盖或移动既有 T12 与探针 2 日志。补丁提供：baseline-anchored arrival 残差映射（默认关闭）；固定惯性 staging 方向下的入口相位有利度；默认关闭的 `entry_phase_gate_cos` 和 `precapture_staging_environment_config(90°)` 开发配置。全量回归在本机通过：`244 passed in 176.05s`。

原 `precapture_planning_environment_config()` 和 `evaluate_hybrid_policy --control desired_pose` 仍指向旧任务；因此当前代码不能直接按新任务跑 A 臂。现有 90° gate 会实际拒绝不满足相位条件的 terminal entry，只能按最新执行单作为**开发期合法性护栏**，不能将它或收紧为 60° 的版本冒充论文中的自然物理窗口。单独的相位有利度函数测试证明几何信号随目标转动变化，不证明等待能改善完成率或燃料。

## 正式 A/B 启动前尚缺上层交付

1. **统一 staging 实验配置：**两臂共同使用的 15–30 m 初始距离、更宽接近方位、20–30° 初始指向误差与预先固定的翻滚速率分布；冻结 ±5 N / ±0.6 N·m、终端精度、MPC 模型及 0.1 s 更新率、走廊/FOV/速度约束。需要明确每个参数的采样定义与记录字段，确保 A/B 同种子只差择时策略。
2. **A 臂接入：**固定 Pure MPC 直接进入，但必须显式选择上述同一 staging 配置；旧 `desired_pose` 默认命令不具备此配置选择。
3. **B 臂 harness：**固定 MPC 先在惯性暂存点 hold，按已冻结的 `entry_phase_favourability` 规则择时 commit；给出确切阈值/时限/回退行为与 provenance。它是脚本化 oracle 择时，不是学习策略。
4. **实验规格：**共同种子块、样本数、结果路径、相位分箱方法、成功/燃料/时间及约束压力的汇总口径、通过/灰区/停止门槛。当前执行单只给出“稳定、明显”的定性判据，正式跑前宜量化并锁定。

在这些前置到位之前，**不启动 A/B、不训练 SAC、不跑探针 3、不更改 MPC 或执行器、不收紧相位门控**。上层完成前置后，下层可复核实现与测试，再按同配置、同种子执行 A/B 并如实交付分布；如果结果很弱，先回上层处理任务几何，不转去调算法。
