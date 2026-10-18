# 任务规格（当前有效，2026-10-10）

高保真 6 自由度 SE(3) 预捕获：追踪星接近一个快速翻滚的非合作目标，在规定时间内以合法方式进入目标体固连的捕获走廊，并稳定在预捕获位姿。

## 物理与执行

| 项 | 值 |
|---|---|
| 追踪星 / 目标质量 | 106 kg / 225 kg |
| 轨道 | 500 km 圆轨道，倾角 45° |
| 真值积分 | RK45；中心引力、二阶矩、重力梯度、J2 |
| 控制周期 | 0.1 s |
| 执行器限幅 | 每轴 ±5 N、±0.6 N·m |
| 上层决策周期 | 2 s（20 个控制步） |
| MPC 预测时域 | 35 步（h35） |

## 目标翻滚

名义翻滚角速度为 0.041231 rad/s（2.362 °/s，周期 152.4 s），由 `phase2_target_tumble_scale = 0.20` 产生。

在大部分距离上 Λ = ω·r / v_max > 1，追踪星无法静止保持，只能持续同步旋转。这是本任务区别于普通平移交会的物理原因。

## 当前工况：`w2.36_r15`（`train/regimes.py`）

2026-10-06 由 269000 块筛选选定，见 `docs/archive/05_handoff_stopping_final/REGIME_SCREEN_20261006.md`。

| 项 | 值 |
|---|---|
| 翻滚 | 2.362 °/s（名义值） |
| 初始距离 | 15–28 m（原主线为 15–20 m） |
| 初始惯性速度上限 | 0.10 m/s |
| 时间上限 | 300 s |
| 距离失败 | 30 m |

工况只改变初始分布。MPC、约束、完成条件、奖励和接口都与 V3e 主线相同。

## 入口与完成

- **入口平面**：距目标中心 6 m（`entry_port_axial_distance_m = 4.5`），入口盘半径 3.151 m。
- **合法进入**：同时满足径向 < 3.151 m、目标系速度 ≤ 0.35 m/s、接近速度 ≤ 0.20 m/s。
- **非法穿越只计数**：不锁存、不结束回合、无奖励惩罚，可以重新合法进入。它是诊断量，不是安全违规。
- **完成**：位置 < 0.25 m、姿态 < 10°、速度 < 0.05 m/s、角速度 < 0.02 rad/s，保持 1 s，且终端区已锁存。

## 安全判定

只以 RK45 真值几何判定：禁区、视场、外区速度、外区径向、走廊、总速度、接近速度。

可比的裕度列是 `minimum_truth_normalized_margin`。不要读 `info` 里未门控的原始逐键裕度，也不要读 `predicted_minimum_margin`（那是优化器自己的预测）。

## 代码入口

| 用途 | 入口 |
|---|---|
| 任务配置 | `env.task.PrecaptureTaskConfig` |
| 环境 | `env.phase2_env.precapture_planning_environment_config()` |
| 耦合环境 | `env.hybrid_env.PrecaptureHybridEnv` |
| 主线配置（V3e） | `train.mainline.mainline_v3e_configs()` |
| 工况 | `train.regimes.REGIMES["w2.36_r15"]`、`regime_environment()` |
| 下层 MPC | `controllers.mpc.config.precapture_mpc_config()` |
