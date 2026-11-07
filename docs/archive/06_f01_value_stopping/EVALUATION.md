# 评估协议（当前有效，2026-10-10）

## 行

| 行 | 含义 | 命令 |
|---|---|---|
| Pure MPC | 固定设定点，冻结基线 | `experiments.v3_stopping evaluate --row pure` |
| nominal | 学习分支的常数动作 +1，用于归因，不作门槛 | `experiments.regime_screen evaluate` |
| learned | 同一模型关闭交接 | `experiments.v3_stopping evaluate --row learned` |
| stopping | 方法：价值规则交接 | `experiments.v3_stopping evaluate --row stopping` |

全部是确定性动作。评估串行进行；并行运行得到的计时不能作为实时性结论。

## 样本块（每块 48 个开局，按用途固定，不复用）

| 块 | 用途 |
|---|---|
| 262000 | B1/B2 交接窗口扫描（原工况） |
| 266000 | 开发块：结构检查 D1–D4 |
| 267000 | 停止头正式轮（原工况，METHOD_DOES_NOT_HOLD） |
| 268000 | 价值规则检验（原工况，VALUE_RULE_DOES_NOT_HOLD） |
| 269000 | 工况筛选（Pure 与 nominal） |
| **271000** | **最终官方判定** |
| 272000 | 第二块，仅描述，与 271000 合成 96 组配对开局 |

## 官方判据（`experiments/v3_stopping.py` 中的 `gate_verdict`）

每个模型单独和 Pure MPC 比，以下三条同时满足才算通过：
- stopping 的干净完成数**严格多于** Pure；
- 真值违规回合数不多于 Pure；
- 毁掉的 Pure 成功开局不超过 2 个（`MAX_DESTROY = 2`）。

**3 个模型里至少 2 个通过**（`MIN_PASSING_MODELS = 2`），方法才算成立。只读 271000 块。

## 描述性证据（不设门槛）

- 交接增益：stopping 相对同一模型 learned 行多完成的数量。
- 96 组配对开局表（`experiments/final_tables.py`）：只在 A 成功、只在 B 成功的数量，精确 McNemar 检验，时间与 Δv 差值的中位数。
- 确定性重放与反事实（`experiments/stopping_replay.py`）：交接点的状态；V_C 在 4 种定义下的取值；在 k* − 1 和 k* 两个时刻测 ε_H、ε_C。

## 不可违反

- 只以 RK45 真值判定安全。
- Pure MPC 冻结。
- 不人为制造差距。
- 不换种子，不挑检查点，不丢弃失败种子。
- 任何进入决策或论文的数字，都必须指向仓库中的产物（JSON、CSV、manifest、提交或脚本），否则标为探索性。
- 训练日志里的完成率是行为策略（随机动作加随机时刻交接）的统计，不是部署性能，不作判据。
