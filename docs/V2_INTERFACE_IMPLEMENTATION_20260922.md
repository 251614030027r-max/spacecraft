# V2 高层任务接口实现与验证

> 日期：2026-09-22 JST
> 起点：`41ec7b6`
> 范围：离线信号审计、T1–T6、V2 接口与 correctness smoke；**未训练**。

## 1. 离线信号审计

固定抽样规则：8 个等间隔 nominal 种子 `262000, 262006, ..., 262042`，每个种子取决策点 `0/15/30`，共 24 个状态；每状态构造 8 个距离/承诺扰动加 1 个不动候选，每候选独立 reset 求解 3 次。预注册通过线是：至少 70% 状态在至少一个扰动方向上随幅度单调，且候选跨度大于同状态重复波动。

| 信号 | 有信息状态 | 结论 |
|---|---:|---|
| `maximum_successful_slack` | 24/24 | 通过 |
| `total_successful_slack` | 24/24 | 通过 |
| `objective` | 24/24 | 通过 |
| `actuator_usage` | 23/24 | 通过 |
| `predicted_minimum_margin` | 24/24 | 通过，仅离线 oracle |

三选一结论：**有便宜信号满足判据，governor 有离线信号地基；V2 第一版仍不接 governor。** 完整表见 `eval/adp/v2_execution_signal_audit.json`，SHA-256 `7c3b6fd427f337eff946d1b86884fd961d7d62541151a51644d70e839936424a`。

## 2. 红测试证据

实现前新增 `tests/test_v2_task_interface.py`，首次输出：

```text
xxxxxx [100%]
6 xfailed in 3.12s
```

六项分别钉住：距离轴有效且可有限后撤、承诺轴不锁死、拒绝无持久副作用、架构级 Pure MPC 地板、几何可接受性、限速器无非物理死区。使用 `xfail(strict=True)`，实现后逐项移除标记。

## 3. V2 实现

- 任务状态 `z=(ρ,c)`：`ρ` 是从 episode 初始惯性暂存半径向内推进的距离（m，后撤时可减小），`c` 是同步承诺（0–1）。两者以归一化值加入 observation，保持状态化限速器的 Markov 性。
- SAC action 表示绝对期望任务状态；第一版限制器分别限幅 `Δρ` 与 `Δc`。每 2 s 决策最大推进/后撤均为 `0.5 m`，承诺最大前进 `0.05`、回退 `0.02`，因此可回退但回退更慢；没有绝对棘轮。
- 参考生成把半径 `r_hold-ρ` 与方向同步 `slerp(hold_direction, capture_direction, c)` 分开。`c=1` 只使方向完全同步，不会让距离轴失效或自动跳到终端位姿。
- 几何保护只复用原 `minimum_waypoint_radius_m` 与 keep-out 下界；没有机会窗、远场走廊或相位合法性门。
- `step_with_proposal(..., proposal_accepted=False)` 是架构级拒绝路径：本步直接走原 desired-pose Pure MPC 参考，且不改写 V2 任务状态。强制全拒由自动测试逐控制步比较 wrench。
- 训练/评估 CLI 仅增加 `task_state_v2` 参数化入口；reward、SAC 超参、MPC、任务参数、执行器和控制周期均未修改。

## 4. 验证

T1–T6 最终输出：`6 passed in 70.36s`；关联旧接口回归：`48 passed, 3 xfailed`。三个 xfail 是 V1 死半边/棘轮的历史诊断钉子，V2 未继承该实现。

Correctness smoke（非方向判据）：

- 3 个接受决策中 proposal 与 applied 均按限幅分离，两轴实际改变参考；
- 参考有限且连续，最小参考半径 `2.5 m`；
- QP zero fallback `0`，真值约束违约步合计 `0`；
- 架构 fallback 已触发；强制拒绝相对 Pure MPC 的 `max |Δwrench| = 0.0`。

Smoke 产物：`eval/adp/v2_interface_smoke.json`；在默认 `0.40 m` 修复后刷新，SHA-256 `75d663131831166dbfd09abdc2939c3d2981d378044727ff2251f1bd864fa593`。

## 5. 停止边界

V2 接口通过 T1–T6 与 smoke。按执行单到此停止：未启动训练、未接 governor、未改 reward/SAC/MPC/任务参数，也未把本次接口修复表述为方法创新。

## 6. 后续米单位上界修复

复核发现承诺轴的无量纲限速不能约束实际参考弧长；大半径、大夹角下单步参考位移可达 `2.98 m`。在两轴独立限幅之后新增同相位参考欧氏位移上界，默认 `v2_reference_step_max_m=0.40`，超限时仅用 8 次纯几何二分共同缩放两轴增量，不增加 MPC 求解。T7 覆盖 12/15/19 m、45/90/120/近 180°及不同轴组合，钉住实际增量不超过上界。四档扫描与最终裁决见 `docs/V2_RATE_LIMIT_SWEEP_STOP_20260922.md`；全套最终为 `283 passed, 3 xfailed`，未训练。

## 7. 训练门记录管道与开训前检查

训练 CLI 已确认同时支持 `--parametrization task_state_v2` 与 `--adaptive-task`，并把两项写入 manifest。为避免 Monitor 只保存终止步快照而使 10k 活性门为空，V2 现按回合累计并仅在终止 info 发出三个标量：`hybrid_v2_episode_changed_fraction`（生效参考相对固定设定点改变的决策占比，绝对容差 `1e-9`）、`hybrid_v2_episode_mean_reference_step_m` 和 `hybrid_v2_episode_accepted_fraction`。`train_hybrid.py` 只在 `task_state_v2` 下把三键加入 Monitor，其他参数化不产生也不读取这些键。

G1/G2 定向结果为 `2 passed`，分别验证回合汇总与逐决策统计一致、其他参数化无键泄漏。活性探针同时纠正为每决策只经 `env.step()` 应用一次 action：V2 三种子整段改变率、后半段改变率均为 `1.000`，不同参考均为 `40/40`；V1 对照后半段均为 `0.000`。全套回归为 `285 passed, 3 xfailed`；三个 xfail 仍是既有 V1 诊断钉子。未启动训练。
