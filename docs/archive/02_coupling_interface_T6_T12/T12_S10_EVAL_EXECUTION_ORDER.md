# T12 / S10 正式评估执行单(下层执行)

写于 2026-09-15,上层沙箱,基点 `41474cf`。T12 六个 final 模型训练完成后的唯一下一步:
六模型 × 48 种子确定性评估,产出主表所需的完成率、燃料、违约。命令与维数已对
`experiments/evaluate_hybrid_policy.py` @ `41474cf` 核对;沙箱无模型/数值栈,未能实跑,
仅静态核对 flag、维数与棘轮激活逻辑。

## 0. 前置核对(每个模型跑前确认一次)

- 模型路径见 `T12_TRAINING_COMPLETION` §六个正式模型路径,只用 `final_model.zip`,
  **不挑中间 checkpoint**。
- 维数:arrival 观测 32 / 动作 2;radial 观测 31 / 动作 4。评估器首局自动做维数校验,
  不符会 `raise ValueError`——报错即停,核对 flag,**不得靠改维数或猜 flag 绕过**。
- 棘轮:评估器 `PrecaptureHybridConfig` 用 `monotone_commit` 默认 `True`,与训练一致,
  arrival 的棘轮观测(+1 维)与混合动力学(`hybrid_env.py:517`)自动生效,无需额外 flag。
  评估器**没有** `--no-monotone-commit`,本轮六模型都不需要它。

## 1. 命令(每模型一条,48 局固定种子块 262000–262047)

arrival(三个种子 262400/401/402 各一条):

```
python -B -m experiments.evaluate_hybrid_policy \
  --model D:/py/DRL2/logs/t12_train/sac_mpc_arrival_<seed>/final_model.zip \
  --episodes 48 --seed 262000 \
  --horizon 35 --parametrization arrival_condition --phase-time-observation \
  --output D:/py/DRL2/logs/t12_eval/arrival_<seed>.json
```

radial(三个种子 262500/501/502 各一条):

```
python -B -m experiments.evaluate_hybrid_policy \
  --model D:/py/DRL2/logs/t12_train/sac_mpc_radial_<seed>/final_model.zip \
  --episodes 48 --seed 262000 \
  --horizon 35 --parametrization radial_local --phase-time-observation \
  --output D:/py/DRL2/logs/t12_eval/radial_<seed>.json
```

- 两臂都**不加** `--execution-feedback`(本轮无反馈)。
- `--seed 262000` + `--episodes 48` 即种子 262000–262047,与 Pure MPC / oracle 同块。
- `deterministic` 是评估器默认行为(用策略均值动作);若脚本暴露开关须确认为 True。

## 2. 必须保留的指标(勿填 0 缺字段)

每个输出 JSON 至少留:完成局数 /48;完成局的均时与力冲量(force impulse),另附全局版;
七类真值违约步(keepout/fov/outer_speed/outer_radial/corridor/total_speed/closing_speed);
非法穿越计数 + 预登记 60% 归因(`primary_illegal_cause_by_preregistered_60pct_rule`);
`qp_infeasible_steps_total`;`max_consecutive_zero_wrench_steps`;超时(满 150 决策 / 到 300 s)
局数。**旧基准缺某字段的,留空标缺,不得填 0 冒充测量。**

## 3. 并行与算力

- **结果型评估可并行**(依据:沙箱并行与下层串行在共同种子上逐条一致的既往结论)。
  六个模型可并行跑完成率/燃料/违约。
- **compute(每步耗时)必须独占串行、单进程**,在所有结果评估结束后单独测,`--no-diagnostics`,
  确认机器无其他计算负载。并行测出的毫秒**不是**实时性证据。

## 4. 报告规则(交回上层)

- 三个 arrival 种子**报分布,不择优**;radial 三种子同样列出(预期 0/48)。
- 主表三行对齐:Pure MPC(已核验 32/48、78.825 s、187.721 N·s,固定设定点)、
  SAC-MPC arrival(本轮)、radial 作可学性对照。用 `eval/main_table.py` 组装,先打印对齐行
  确认 task/种子块/局数一致再比。
- **oracle 40/48 不上主表**(事后逐种子选时的参考,非控制器、非上界)。
- Pure SAC 本轮不做。旧 T8 负结果(7/12 对 8/12、配对燃料贵 40%)保留为历史缺陷证据。

## 5. 判据分叉(预登记,勿事后改)

1. 三个 arrival 在 48 种子上**稳定**超过 Pure MPC 的 32/48,且燃料 ≤ 187.7 N·s、违约/非法穿越/
   QP 不可行代价可接受 → 支持"学习层带来实际净正贡献"。**完成率赢、燃料不赢,不算赢。**
2. arrival 明显优于 radial 但**低于** MPC → 只支持"接口改善训练可学性",不支持"学习控制优于
   固定设定点"。按止损承认,回上层重设计,不找理由。
3. 只有单个种子好、其余明显落后 → 报告种子敏感性,不得拿最好种子替代方法分布。

## 6. 护栏

不重训、不加第四轮、不挑 checkpoint、不换种子、不丢失败种子、不据训练回报冒充成功率、
不改奖励/MPC/任务参数。全零 radial 种子照常评估并报告。
