# T8 执行单：正式评估、补种子、死种子诊断（2026-09-12）

> 读者：下层执行窗口（`D:\py\DRL2`）。本文自包含。
> 前置：远端 `9cd503a` + 本次增量补丁 `T8_delivery.patch`。
> **授权范围：S1–S5。不改奖励、不改 MPC、不改任务定义、不加新动作通道。**

---

## 0. 上一轮发生了什么（一句话）

T7 四组训练全部跑完 60,000 决策步。**耦合第一次学会了完成任务**——
最好的种子最后 100 回合完成 75%，而上一代接口三种子 25k 步是 0/1308。
但四个种子里**两个完成率恒为 0**，而且它们不是不动：回报在改善，
改善来源是超时回合占比从 35% 涨到 68%——**它们学会了拖到 300 秒上限**。

本轮要做三件事：把四个模型正式评估出来、把种子补到 3 个、搞清楚死种子在干什么。

---

## S1：应用增量补丁

补丁放仓库根目录（与 `.git` 同级）：

```powershell
cd D:\py\DRL2
git status --short          # 必须干净
git am -3 T8_delivery.patch
python -B -m pytest -q
```

补丁含 3 条提交，改动三处：

| 文件 | 改动 |
|---|---|
| `experiments/evaluate_hybrid_policy.py` | 支持 `--parametrization arrival_condition` 与 `--execution-feedback`；**加载检查点时校验观测/动作维数**，不匹配直接报错 |
| `experiments/replay_hybrid_policy.py` | **新增**，逐决策回放诊断脚本（只读） |
| `docs/` | T7 裁决、本执行单 |

**回归预期**：本地原有数 + 0（本补丁未加测试）。上一轮你那边是 217 passed，
若仍是 217 即通过；**数字不同先报告**。

### 自洽性核验（上游已做，本机可复核）

评估路径没有改变下层行为：`--control desired_pose` 在 262000 完成于 **113.6 s**、
262001 失败于 **38.7 s**，与仓库记录的 113.5 / 38.6 s 一致。

---

## S2：正式评估四个模型（**串行单进程**）

同一 12 种子块 262000–262011，关闭探索。**保留全部四个种子，包括两个零成功的。**

**注意观测开关必须与 manifest 一致**：V3 训练时开了反馈（34 维），V2 没开（31 维）。
写错现在会直接报错，不会静默错配。

```powershell
# V3（有反馈，34 维）
python -B -m experiments.evaluate_hybrid_policy --episodes 12 --seed 262000 --horizon 20 `
    --parametrization arrival_condition --phase-time-observation --execution-feedback `
    --model logs\t7_retry1_20260911\sac_mpc_v3_bidirectional_262300\final_model.zip `
    --output logs\t8_eval\eval_v3_262300.json

python -B -m experiments.evaluate_hybrid_policy --episodes 12 --seed 262000 --horizon 20 `
    --parametrization arrival_condition --phase-time-observation --execution-feedback `
    --model logs\t7_retry1_20260911\sac_mpc_v3_bidirectional_262301\final_model.zip `
    --output logs\t8_eval\eval_v3_262301.json

# V2（无反馈，31 维）
python -B -m experiments.evaluate_hybrid_policy --episodes 12 --seed 262000 --horizon 20 `
    --parametrization arrival_condition --phase-time-observation `
    --model logs\t7_retry1_20260911\sac_mpc_v2_arrival_262200\final_model.zip `
    --output logs\t8_eval\eval_v2_262200.json

python -B -m experiments.evaluate_hybrid_policy --episodes 12 --seed 262000 --horizon 20 `
    --parametrization arrival_condition --phase-time-observation `
    --model logs\t7_retry1_20260911\sac_mpc_v2_arrival_262201\final_model.zip `
    --output logs\t8_eval\eval_v2_262201.json

# 同场景对照行：经包装器交付的固定设定点下层（= Pure MPC）
python -B -m experiments.evaluate_hybrid_policy --episodes 12 --seed 262000 --horizon 20 `
    --parametrization arrival_condition --phase-time-observation `
    --control desired_pose --output logs\t8_eval\eval_control_desired_pose.json
```

**这五条必须串行、单进程、不与训练同时跑。** 其中的 compute 列是实时性证据，
任何并行运行的耗时都作废。

**报回来**：每个 JSON 的完成数、完成时间、力冲量、最差真值裕度、
`controller p95/budget`。

---

## S3：补第三个种子（可与 S2 之外的时间并行）

项目规矩是 ≥3 训练种子报分布。当前每版只有 1 个活种子，噪声底线约 ±37 个百分点，
在这个底线上比较任何机制都没有意义。

```powershell
python -B -m train.train_hybrid --steps 60000 --seed 262302 `
    --parametrization arrival_condition --run-name sac_mpc_v3_bidirectional_262302

python -B -m train.train_hybrid --steps 60000 --seed 262202 `
    --parametrization arrival_condition --no-execution-feedback `
    --run-name sac_mpc_v2_arrival_262202
```

从零开始、不载检查点、参数一字不改。训练可以并行，**但 S2 的评估必须单独串行跑**。

---

## S4：死种子诊断（本轮最关键的一步）

T7 裁决给出一个**可证伪的假设**：

> 零成功种子收敛到「长期等待、拖到上限」，
> 也就是它的 `a_commit` 长期停在**负半轴**（负＝在外面惯性冻结地等，正＝往里切）。

补丁新增的 `experiments/replay_hybrid_policy.py` 就是验这一条的。
它逐决策记录动作、真实距离、参考半径、各几何裕度、进入事件、
反馈三量、回退数、终止原因，并直接打印 `a_commit` 落在负半轴的比例。

```powershell
# 死种子
python -B -m experiments.replay_hybrid_policy --episodes 4 --seed 262000 `
    --parametrization arrival_condition --phase-time-observation --execution-feedback `
    --model logs\t7_retry1_20260911\sac_mpc_v3_bidirectional_262301\final_model.zip `
    --output logs\t8_eval\replay_v3_262301_dead.json

python -B -m experiments.replay_hybrid_policy --episodes 4 --seed 262000 `
    --parametrization arrival_condition --phase-time-observation `
    --model logs\t7_retry1_20260911\sac_mpc_v2_arrival_262201\final_model.zip `
    --output logs\t8_eval\replay_v2_262201_dead.json

# 活种子（配对对照，同样的 4 个场景）
python -B -m experiments.replay_hybrid_policy --episodes 4 --seed 262000 `
    --parametrization arrival_condition --phase-time-observation --execution-feedback `
    --model logs\t7_retry1_20260911\sac_mpc_v3_bidirectional_262300\final_model.zip `
    --output logs\t8_eval\replay_v3_262300_live.json

python -B -m experiments.replay_hybrid_policy --episodes 4 --seed 262000 `
    --parametrization arrival_condition --phase-time-observation `
    --model logs\t7_retry1_20260911\sac_mpc_v2_arrival_262200\final_model.zip `
    --output logs\t8_eval\replay_v2_262200_live.json
```

**要回答的四个问题**（照 JSON 直接答，不要推测）：

1. 死种子的 `fraction_commit_negative` 是不是明显高于活种子？
   `action_mean[0]`、`action_min/max[0]` 分别是多少？
2. 死种子的 `ended_by` 主要是 `time_failure` 还是别的？
3. 死种子在整条轨迹上 `true_range_m` 停在哪个范围？活种子推到多近？
4. **反馈三量在死/活种子上是不是近乎常量？**
   （`hybrid_feedback_*` 三列的取值范围。若三列几乎不动，
   说明这条通道在部署时携带的信息很少，对解释 T7 的消融结果直接相关。）

---

## S5：一次性耗时分解（不在正式运行中加持续诊断）

一个上层决策约 1.3 秒，其中 MPC 求解约 0.84 秒。要知道这 0.84 秒里
**真正解 QP 占多少、CVXPY 每次调用的组装开销占多少**。

做法：短窗口（例如 300 个控制步）分别计时
参考构造 / 线性化 / CVXPY 装配 / 求解器求解 / RK45 传播 / 网络更新，
覆盖 `learning_starts=2000` **之后**的阶段，不能只测随机采样阶段。

**这只是测量，本轮不做任何优化实现。** 若组装开销占大头，
下一轮再考虑绕过 CVXPY 直接更新求解器数据——那属于纯实现改动，
必须与现实现逐位对照后才能采用。

**明确不算加速**（会改变控制行为，等于换实验）：放大仿真步长、
降低 MPC 调用频率、缩短预测时域、放松求解容差、用近似控制器替代 MPC。

---

## 红线（不变）

1. **算力只认串行单进程。** 并行仅用于训练，任何并行耗时不构成实时结论。
2. **违约按 RK45 真值 + 真实几何判**，不用 MPC 的预测裕度代替。
3. **不看结果调参数**：不换种子、不挑检查点、不删失败种子。四个种子全部报。
4. **每次训练从零开始**，不续训、不载检查点。
5. **任务参数冻结**：几何、约束、时限、奖励一律不动。
6. 不加第三个动作通道、不加姿态通道、不给「正确进入时机」额外奖励。

---

## 什么情况停下来报告

- S1 回归数与本地上一轮不一致
- S2 任一评估报观测/动作维数不匹配（说明 manifest 与命令开关对不上，**不要靠试**，去读 manifest）
- S4 的四个问题里，死种子的 `fraction_commit_negative` **并不**明显高于活种子
  （那说明 T7 裁决的假设被证伪，下一个单因子要重选，别自行推进）
- 补种子时又出现全程零成功，或出现此前那种进程无故停止
- 任何看起来需要改任务／奖励／MPC 才能推进的情况

---

## 交回来的东西

1. S2 的五个评估 JSON + 一张汇总表
2. S3 两个新训练的日志目录（跑完再报，中途不用报）
3. S4 的四个回放 JSON + 上面四个问题的直接回答
4. S5 的耗时分解结果
5. 推送本轮全部提交

**不要**在本轮修改任何运行代码、不要自行启动第三轮训练方案、
不要因为 V3 落后就移除反馈通道——每版只有 1 个活种子，不足以下这个结论。
