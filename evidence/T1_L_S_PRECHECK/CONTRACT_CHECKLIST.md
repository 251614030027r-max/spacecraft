# T1-L-S 合同核验（任务书第 7 节 A–D）

## A. 物理不变，旧在训分支不变：通过

- **未改动的核心文件：** `env/t1_env.py`、`env/t1_gstar.py`、`env/t1_task.py`、`env/t1_task_review.py`、`env/t1_option_cf.py`、`env/t1_counterfactual.py`、奖励、MPC 都未改动（见 `SCIENCE_DIFF.md`）。
- **`env/t1_l_task.py`：** 只新增 T1-R 评估档案，T1-L 档案逐字段不变（`test_t1l_changes_only_the_initial_range`）。
- **`OptionSAC`：** 只抽出 4 个钩子，默认行为与 166244b 逐位相同。跨版本回归：同一脚本分别在 166244b 和新分支上跑，覆盖普通路径和 λ_Δ 标定冻结路径，参数哈希一致（`0e23a95e61773e51`、`8474133868f1c2d2`）。
- **旧 T1-L 测试：** 17 项全部通过。新分支不接触 `D:\py\DRL2_T1_L` 或其产物。

## B. 审议区间数学合同：通过（`tests/test_t1_l_s.py`，手算值）

| 测试 | 结论 |
|---|---|
| `test_six_second_interval_folds_three_rewards_and_ends_at_the_next_review` | d = 3 时 R = r0 + γr1 + γ²r2，折扣 γ³，终点为下一审议态；continue 槽和最新未闭合审议不可训练 |
| `test_early_review_closes_a_shorter_interval` | 事件提前审议时 d = 2 |
| `test_episode_end_closes_with_the_terminal_observation`（2 组） | 回合结束时终点为终止观测；成功、超时、unsafe 时 D = 1；数值截断时 D = 0（自举） |
| `test_review_that_ends_the_episode_at_once_is_a_one_step_interval` | 审议当步即结束时 d = 1（修复了同一步内开启又闭合时被误丢弃的缺陷） |
| `test_every_step_review_degenerates_to_one_step` | B2 下 d ≡ 1，退化为一步 |
| `test_overwritten_start_slot_is_dropped_never_backfilled` | 环形覆盖时丢弃并计数，不回填 |
| `test_two_environments_never_mix_intervals` | 两个环境的区间互不串联 |
| `test_target_routing_review_folded_continue_one_step_open_review_masked` | 审议取折叠目标，continue 取一步目标，未闭合审议被掩码；用固定后继值逐项核对 |

## C. MPC 反馈与数学边界：通过

- **相对项关闭：** λ_Δ = 0 时，无论 Replay 中有无反事实数据，区间目标下参数都逐位相同（`test_relative_term_off_ignores_counterfactual_data_with_interval_targets`）。
- **相对项只用一步数据：** 继续使用 `OptionSAC.train` 里一步的 `v_next` 和 `data.rewards`，从不读取区间字段（代码路径没有改动，见 `SCIENCE_DIFF.md`）。
- **旧合同：** 校准只读、冻结、上限、预热的原有测试（`test_t1_l.py`）全部通过。
- **冒烟：** sndelta 探针 30 个，λ_Δ = 4.81，在第 218 步冻结；门控有效反事实 18 个；continue 步不产生反事实。

## D. 学习器时钟与资源：通过

- **Actor 时钟：** 按全局真实转移计数。1 环境下每 3 步 1 次；6 环境下每向量步正好 2 次；总数 = (T − learning_starts) / 3（`test_actor_clock_counts_global_real_transitions[1, 6]`）。
- **训练器 A：** 每次 Critic 更新配 1 次 Actor 更新，Actor 批次只含已闭合审议。
- **余弦学习率：** 0k 为 1e-4，中点为 5.5e-5，终点为 1e-5；SB3 实际写入 Actor 和 Critic 优化器（`test_cosine_learning_rate_runs_on_real_transition_progress`）。
- **真实环境冒烟（`SMOKE_HEALTH.json`）：**

  | 臂 | 配置 | Critic 更新（实际/理论） | Actor 更新（实际/理论） |
  |---|---|---|---|
  | sb2 | 训练器 A，1 环境 | 720 / 720 | 720 / 720 |
  | sn0 | 训练器 B，2 环境 | 180 / 180 | 60 / 60 |
  | sndelta | 训练器 B，2 环境 | 180 / 180 | 60 / 60 |

  - 区间：sb2 的 239 个全为 d = 1；sn0 有 78 个 d = 3 和 1 个 d = 2。没有被覆盖丢弃的区间，两个环境之间没有串联。
  - 余弦学习率逐步下降。
  - TCPF 检查点可重载，并能在 T1-R 上评估。
- **无损续训：** 不支持。只保存模型，不保存 Replay、优化器和随机状态。执行单如实声明：中断即记录，不续训。

## 三个重点风险的结论

1. **Bellman 口径：** 审议与 continue 的目标按各自真实时间跨度构造。不声称无偏，也不声称是"MPC 完整区间反事实"。
2. **Replay 生命周期：** 按环境封账、终止观测、覆盖丢弃、ready 掩码均已测试。
3. **反事实一跳与区间跨度并存：** 字段分离、相对项只用一步数据，测试与代码路径均确认。
