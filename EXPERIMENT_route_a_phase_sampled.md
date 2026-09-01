# 路线 A 阶段 0：采样相位更难 regime（无学习，纯诊断）

> 单因子 = 目标初始相位采样 开/关。新副任务 `single_phase_phase_sampled`，S1-v2 主基准不动。
> 本会话交付代码 + 会话内 smoke；结论性分布由本机 ≥3 seed × 20 episode 跑出。

## 实现了什么（单因子）

`single_phase_phase_sampled` = `single_phase` + 唯一一处改动:每 episode 采样目标**初始姿态
(均匀 SO(3)) + 角速度方向**,**翻滚幅值冻结**(|ω| = 0.0412 rad/s 不变,保 Λ>1 regime)。
`phase2_training_mode` 仍是 `"single_phase"`,所有任务语义(相位标志、终端约束从第一步起效、无 Waypoint)
逐位不变——改的只有相位采样这一个因子。

| 文件 | 改动 |
|---|---|
| `env/scenarios.py` | `target_initial_state(*, tumble_scale, phase_seed=None)`:None → 原确定态(单位阵+base,逐位不变);给 seed → 采样姿态+方向,|ω| 冻结,固定 seed 可复现。 |
| `env/se3_rendezvous_env.py` | 新增 `phase2_target_phase_sampling` 开关;每 episode 从 `np_random` 抽相位 seed(可复现),**相位 seed 进轨迹缓存 key**(否则复用一条),`info["episode_target_phase_seed"]` 记录。 |
| `env/phase2_env.py` | 新任务模式 `single_phase_phase_sampled`(= single_phase + 采样开)。 |
| `experiments/evaluate_mpc.py` | `--task single_phase_phase_sampled`。 |
| `eval/validate_single_phase_semantics.py` | `--task` 选项,脚本参考行可跑采样任务。 |
| `experiments/refresh_staleness_probe.py` | **新增**:点 3 的廉价陈旧信号(local 与 exact 单步预测偏差)对齐 margin,判断能否预测掉安全时刻。 |
| `tests/test_phase_sampling.py` | 6 项:nominal 逐位不变、采样幅值冻结+可复现、缓存按相位 key、语义不变、env 逐 episode 变、MPC 可跑采样任务。 |

`python -B -m pytest -q` → **129 passed**(123 + 6)。

## 会话内 smoke(非结论,仅证管线通 + 守卫)

- **nominal-phase 守卫(用户点 1)**:脚本控制器在 `single_phase` 与 `single_phase_phase_sampled`
  上各 4 episode **均 4/4 完成,worst-margin +0.047 vs +0.046**——脚本是活反馈律,采样不掉,
  **无隐藏"默认单位阵"假设**。这也预示"corridor 参考失效"分支大概率不触发。
- **MPC 可跑采样任务**(单相位 seed=990000,1 ep):exact r10 与 local 均完成,worst +0.044;
  local p95 0.48×/max 1.53×。**n=1,不作结论。**
- **陈旧信号探针**(2 ep):local 在这两个相位上守安全,无违约。

**这些都是 n≤4 的 smoke,只证管线与守卫。真正的分布由下面的批处理定。**

## 本机批处理(≥3 seed × 20 episode,PowerShell)

三个 seed 块(相位是 seed 的一部分,按相位报分布):

```powershell
foreach ($S in 990000,992000,994000) {
  # 脚本参考行
  python -B -m eval.validate_single_phase_semantics --task single_phase_phase_sampled --seed $S --episodes 20 --output "logs/ps_scripted_s$S.json"
  # Pure MPC 三档(h10、corridor)
  python -B -m experiments.evaluate_mpc --task single_phase_phase_sampled --episodes 20 --seed $S --horizon 10 --output "logs/ps_mpc_exact_s$S.json"
  python -B -m experiments.evaluate_mpc --task single_phase_phase_sampled --episodes 20 --seed $S --horizon 10 --linearization-source local --output "logs/ps_mpc_local_s$S.json"
  python -B -m experiments.evaluate_mpc --task single_phase_phase_sampled --episodes 20 --seed $S --horizon 10 --exact-refresh-steps 100 --output "logs/ps_mpc_refresh100_s$S.json"
}
```

每个 seed 块出一张同口径主表(compute 看 p95/max):

```powershell
python -B -m eval.main_table `
  --row "Scripted=logs/ps_scripted_s990000.json" `
  --row "MPC exact r10=logs/ps_mpc_exact_s990000.json" `
  --row "MPC local=logs/ps_mpc_local_s990000.json" `
  --row "MPC refresh100=logs/ps_mpc_refresh100_s990000.json"
```

三个块的表贴回即可(共 3 张)。看每行的 **完成率 + worst-margin + compute p95/max 预算比**,
按相位报分布(不报最好块)。

**点 3(仅当 local/refresh100 掉安全时再跑)**:陈旧信号能否预测掉安全时刻——
```powershell
python -B -m experiments.refresh_staleness_probe --episodes 20 --seed 990000
```
输出 `corr(error,tightness)` 与 `pre_violation_error_ratio`(>1 = 信号在掉安全前抬升,可作自适应刷新的门)。

## 预先锁定的分支判据(阶段 0 出数即决)

| 观察 | 缺口 | 阶段 1 |
|---|---|---|
| local/refresh100 在相位下**掉安全**,exact 刷新能救回 | "何时刷新" | **自适应刷新/调用率**学习层(首选,直连 compute 叙事) |
| 手工 corridor **完成率在相位下下降** | 参考 | **SAC 供参考** |
| 两者都不失效 | 无缺口 | **转路线 B** 两基线诚实故事,不硬造 hybrid |

会话内守卫已预示 corridor 分支大概率不触发;主看点是 **local 是否在某些相位掉安全**——
若掉,自适应刷新就地有据(配合陈旧信号探针)。

## 纪律

单因子 = 相位采样开/关;S1-v2 不动;compute 全程 p95/max;学习轮次独立、≥3 训练 seed、报分布;
代码 zip 交付本机 commit,不 push。
