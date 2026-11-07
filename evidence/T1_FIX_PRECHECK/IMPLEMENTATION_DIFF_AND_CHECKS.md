# T1 可学习性修复：实现差异与检查（任务书 2026-10-15）

**范围：** 原 T1（9–15 m，0.10 m/s，300 s），开发环境。`78eb63c`（G★ 0.4、接口 v2）冻结为旧参照。
**不变：** 奖励、Pure MPC、共享 MPC（h35）、任务约束、真值 RK45、SB3 SAC 主体与超参数（lr 1e-4、batch 256、buffer 3e5、UTD 4、α 0.005、γ 0.999、τ 0.001、n_envs 6）。

## 1. 变更清单

| 文件 | 变更 | 原因（任务书条目） |
|---|---|---|
| `env/t1_gstar.py` | G★ **v0.5** 期限一致性。最晚到达 = 300 − `deadline_margin_s`（60 s）。`last_deadline` 有四种取值，见下表 | A（P0）：请求、滑移、保持计划都受同一期限约束，规则、M0、M2 共用 |
| `env/t1_gym.py` | 接口 **v3**：动作语义仍为 v2，观测末尾追加 `track_position_error_1m`、`track_velocity_error_0p3m_s`，共 24 维。`info["execution"]` 新增 `request_kind`、`deviation_before_m`、`new_plan_trigger`、`deadline` | A 观测；B 有效动作诊断 |
| `train/t1_m1.py` | `M1ReplayBuffer` 写回 MC 回报，提供 `monte_carlo()`。`M1SAC` 新增 L_mc、L_eq、梯度探针和 critic 审计 | D |
| `train/train_t1.py` | M0 与 M2 用同一类 `M1SAC`：M0 的 λ_cf = 0，环境不含反事实；`m1` 保留为 `m2` 的别名。新增参数 `--lambda-mc`、`--lambda-eq`、`--mc-window`、`--grad-probe-every`、`--audit-every`。`decisions.jsonl` 和 episodes 记录诊断字段。manifest 写入 `critic_fixes` 与 `algorithm_class`。检查点旁边写 `*_fix_stats.json` | D 共用类；B 诊断 |
| `experiments/t1_dev_eval.py` | 新增 `rule:earliest_k10`（κ = 1.0）和 `--first N`。每个决策记录请求类型、触发、期限、偏差、奖励，以及 critic 的 Q（实际动作、保持、±15 s）。监视进程的默认检查点为 0/2k/5k/10k/15k/20k | 训练评估与四条审计曲线 |
| `experiments/t1_fix_calibration.py` | 预注册的 λ 一次校准，规则见第 4 节 | D 权重校准 |
| `experiments/t1_fix_readout.py` | 四条审计曲线（开发表现、有效动作、critic 校准与排序、M2 门控与 CPU） | 审计曲线 |
| `experiments/t1_frozen_config.py`、`experiments/t1_pretrain_checks.py` | 输出 v0.5 期限和 critic 修复配置；重载检查识别共用类 | 冻结配置与重载 |
| `tests/test_t1_fixes.py` | 18 项新测试，见第 3 节 | 验收 |

### 1.1 G★ v0.5 的期限行为

**最晚到达：** horizon = 300 − 60 − t（自当前时刻起算）。

| `last_deadline` | 触发 | 动作 |
|---|---|---|
| `clamped` | 推迟请求 τ > horizon | τ 截到 horizon |
| `plan_beyond` | 可用计划的剩余时间 > horizon + 容差 | 该计划不再保持，重规划 |
| `infeasible_within` | 期限内没有可行解 | 取最早可行（在期限之后），并上报 |
| `none` | 其他 | — |

**期限没有切掉任何滑移时：** 搜索与 v0.4 完全相同。

**期限切掉了滑移时：**
- 有可用计划：请求不可行就保持原计划（v0.4 的保持保护）；
- 没有可用计划：依次尝试期限内的滑移、最早网格（80% 推力）、恢复求解（100%）、惯性保持。

**最早请求路径（规则）：** 最早网格搜索与 v0.4 相同。

**60 s 的来源：** 规则运行中，从最后一次计划到达到捕获的时间，中位 37.1 s，p90 63.9 s，p95 70.6 s。成功的规则回合计划到达最晚为 232 s，没有超过 240 s 的。证据：`evidence/T1_BETA_B1`、`evidence/T1_S1_SCREEN`。

### 1.2 Critic 修复（M0/M2 共用，任务书 D）

```
L_Q = L_TD + λ_mc · L_mc + λ_eq · L_eq + 1_M2 · λ_cf · L_cf
L_mc = 0.5 Σ_j mean_B[ m · Huber_10(Q_j(s,a) − G) ]
L_eq = 0.5 Σ_j mean_B[ e · (Q_j(s,(u,a1)) − sg Q_j(s,(0,a1)))² ],  u ~ U[−0.5, 0.5]
```

**G（MC 标签）：**
- 是本回合的精确折扣回报（γ = 0.999）。
- 只有回合正常结束（success/unsafe/timeout）时才按 `(env_index, 写入序号)` 回写。`numerical_failure`（SB3 的 `TimeLimit.truncated`）和未完成回合的 m = 0。
- 写回前若环形槽已被覆盖，则跳过；槽被新转移占用时，其标签清零。
- 采样时只用最近 `mc_window` = 10k 条转移。
- 随机暖启动的前 1000 条转移是 SB3 行为策略本身的一部分，与其他转移按同一窗口处理。
- 本轮不预填，所以没有需要排除的专家样本。

**e（等价掩码）：** `remaining_160s` > 0 且非下降。此时带内任意 a0 都请求同一剩余时间，转移逐位相同（单步测试见第 3 节）。以下情形不施加：0.5 < |a0| ≤ 0.55、无计划、下降。

**随机性：** ã0 来自独立的 `torch.Generator`。开启探针或审计不会移动全局随机流。

**λ = 0 且探针关闭时：** 不做额外计算，也不抽随机数。

**L_cf：** 沿用原 M1（λ_cf 0.5，预热 3000，下降时 w = 0，门控不变）。M2 的 shadow 观测由同一个 `T1GStarEnv._observation` 生成，所以 24 维自动一致。门控误差 e_o 改在 24 维上计算，阈值 0.07 不变（只会更保守）。

## 2. 奖励与终止审计（任务书 C：不改动）

- success / unsafe / timeout 是 terminated；numerical_failure 是 truncated。
- SB3 对终止不做自举，对截断做自举（`tests/test_t1_termination.py`）。
- MC 回写沿用同一语义：截断回合不打标签（`test_sb3_path_labels_terminals_and_not_truncations`，走真实 SB3 收集路径）。
- 没有任何奖励或塑形项鼓励保持、最早或 κ。

## 3. 检查结果（上层沙箱）

数字见 `precheck.json`。沙箱为 4 核且有争用，计时不代表用户机器。

| 验收项 | 结果 | 依据 |
|---|---|---|
| 状态/动作确定性；保持带单步等价 | 通过：同一状态下 a0 = 0 与 a0 = 0.3（a1 相同）执行一步，下一观测、奖励、终止标志逐位相同 | `test_keep_band_one_step_equivalence` |
| 期限行为 | 通过：t = 200 s 时 100 s 的请求被截到 40 s；到达超过期限的可用计划不再保持；最早请求的结果不受期限影响 | `test_deadline_*`、`test_earliest_request_search_is_unaffected_by_the_deadline` |
| 规则不变 | 295003、295004 的 κ0.7 规则在 v0.5 下与 v0.4 证据（`evidence/T1_BETA_B0/rule`）逐值相同（49、86 个决策逐步相同，`deadline` 全为 none）。16 开局的全块复核由下层准入 C 完成 | `precheck.json` |
| 新观测与诊断 | 通过：无计划时追加维为 0；新计划后有限且 < 1；`request_kind` 和 `new_plan_trigger` 正确区分 keep/later/no_plan | `test_tracking_errors_zero_without_plan_and_appended` |
| 新维度贯通 M0/M2、buffer、重载、评估 | 通过：两条 300 决策冒烟（n_envs 2）均正常结束；重载 4 个检查点，动作有限；`t1_dev_eval` 能加载并记录 critic Q，`t1_fix_readout` 能读出四条曲线 | `precheck.json`、`smoke_readout.json` |
| shadow 不污染真实转移 | 通过：沿用 M1 的逐位测试（shadow 闭环复现真实一步；真实分支不变），在 24 维下重新运行通过 | `tests/test_t1_counterfactual.py` |
| shadow 误差在原水平 | 实际动作预测误差 p95：e_o 0.051、e_p 0.011 m、e_r 0.052（236 个决策）；验证 B 为 0.071、0.016、0.059（22 维）。e_o 只关闭过 2 次门控 | `precheck.json` |
| MC 回写 | 通过：交错环境、正常终止（success/unsafe/timeout）、数值截断、半回合、环形覆盖、标签清零、10k 窗口；另走真实 SB3 收集路径核验 | `test_mc_*`、`test_sb3_path_labels_terminals_and_not_truncations` |
| eq 掩码 | 通过：只有"有计划且非下降"为 1 | `test_eq_mask_live_plan_only_not_descent` |
| λ = 0 回归 | 通过：M1SAC（λ 全为 0）与 SB3 SAC 参数逐位相同，探针和审计开启时同样逐位相同；λ_mc 或 λ_eq > 0 都会改变 critic | `test_lambda_zero_is_plain_sac_bitwise_even_with_probe_and_audit`、`test_each_fix_moves_the_critic` |
| 奖励/回报一致 | 由 `test_t1_termination`（四种结束方式的奖励与步数）和 MC 精确回报测试共同覆盖 | — |
| 冒烟数值 | 无 NaN、参数有限；探针值全部有限；M2 门控开启率 0.40（随机动作期）；冒烟中出现 18 次 `clamped`（随机动作请求晚于期限），没有 `plan_beyond` 或 `infeasible_within` | `precheck.json` |
| 每决策 CPU | 4 核争用下 M0 约 1.6 s、M2 约 1.8 s 墙钟（2 环境，含更新）。上一轮在用户机器上，M1、6 环境：32,088 决策用 6.8 h，约 0.76 s/决策 | `precheck.json`、`evidence/T1_V2_S0` |
| 全部测试 | 427 通过，3 个预期失败（xfail，均为旧耦合诊断，与 T1 无关），在 d8b185f 上运行 | `python -B -m pytest -q` |


## 4. 权重校准规则（预注册，由下层执行，不看成绩）

`experiments/t1_fix_calibration.py`：

1. 运行一次 2k 决策的 M0（n_envs 6，learning_starts 1000，λ = 0，`--grad-probe-every 10`）。
2. 在决策 1000–2000 窗口内，计算每次探针的 ‖g_aux‖/‖g_TD‖（未加权，在 critic 参数上），取中位。
3. 对 λ_mc 和 λ_eq 分别取值：
   - 在 {0.03, 0.1, 0.3} 中，取 λ·中位落在 [0.1, 0.3] 的最大值；
   - 0.03 仍过强时，取 0.2/中位；
   - 0.3 仍过弱时，取 0；
   - 探针少于 30 次、标签比例 < 0.02 或梯度非有限时，取 0。
4. 合并占比 ‖λ_mc g_mc + λ_eq g_eq‖/‖g_TD‖ 的中位若超过 0.4，两个 λ 同比下调。这一项由点积精确计算。
5. 得到的值冻结，M0 与 M2 共用。

## 5. 预填

**本轮不采用。** 理由：
- 任务书允许跳过；
- 预填会给 MC 回写（专家样本须排除在 L_mc 之外）和 M0/M2 字节一致性增加一条新代码路径；
- 首轮先单独验证 critic 修复本身。

若首轮表明早期采样成本是主要瓶颈，可作为下一轮的唯一改动。
