# 主线运行单:机会任务(外层冻结接近走廊 + 内层旋转捕获)+ 强耦合

写于 2026-09-17,上层沙箱。这是**已冻结方向**的完整可训练闭环。方向纪律(2026-09-17):做成主线、
不因小样本否决;MPC 用普通即可,重点是耦合有明显优势;两臂同一 MPC、同一任务约束,只差"学习参考
vs 固定参考"。命令直接在 `D:\py\DRL2` 跑,**结果不必提交**,回传我迭代。

## 0. 任务是什么 / 为什么这次有真决策

- reset 时冻结 chaser 从远处带进来的**惯性接近方向 `s0`**。终端捕获前,chaser 必须待在围绕 `s0`
  的 **40° 外层接近锥**内(不能为追端口绕目标转)。这条锥**不随目标转**。
- 捕获端口/终端走廊/入口面继续**随目标翻滚**。于是两者几何衔接**周期性对齐**:对齐时进入容易,
  不对齐时合理动作是**在锥内等**、调半径、降速,而不是硬追。**机会由几何自然产生,没有相位硬门。**
- 外层接近锥是**计数的安全裕度约束**(不终止回合、不阻断完成),pre-entry 生效、合法 latch 后失效。

**机制已验证(2 集 smoke,非结果):** 固定设定点 Pure MPC 在机会任务上 **0/2 合法完成**——
seed 262000 完成但非法(277 步 outer_approach 违约:追旋转端口时冲出锥);seed 262001 直接超时
(2172 步违约、2 次非法进入)。即**固定参考的 MPC 守不住接近锥**,给了"学习择时进入"真实空间。

## 1. 训练(本地,3 种子,从零,h35,预算对齐 T12)

```
for SEED in 262410 262411 262412; do
python -B -m train.train_hybrid \
  --steps 60000 --seed $SEED --run-name opp_${SEED} \
  --horizon 35 --parametrization arrival_condition \
  --baseline-anchored-residual --opportunity-task --device auto
done
```

- `--opportunity-task` 自动:切到机会任务(外层锥 40°、初始 15–28 m、指向 25°、max_dist 35 m)+
  给策略加 **staging 方向观测**(`R_target^T s0`,3D)——策略靠它判断"锥相对端口转到哪了"。
- 观测 **38D**、动作 2D、残差锚定(残差 0 = 固定设定点 Pure MPC 逐位)。manifest 的
  `evaluation_flags` 会带全 `--opportunity-task --baseline-anchored-residual`。
- `final_model` 为正式模型;checkpoint 只排错不择优;从零、不 warm-start。

## 2. 评估(同一 48 种子块 262000,h35,deterministic,串行)

```
OUT=eval/results/opp_YYYYMMDD
# 1) Pure MPC(固定设定点,机会任务)
python -B -m experiments.evaluate_hybrid_policy --episodes 48 --seed 262000 --horizon 35 \
  --parametrization arrival_condition --opportunity-task --control desired_pose \
  --output $OUT/pure_mpc.json

# 2) 提案:残差 + critic 优势门(主方法),每个训练种子一次
python -B -m experiments.evaluate_hybrid_policy --episodes 48 --seed 262000 --horizon 35 \
  --parametrization arrival_condition --opportunity-task \
  --phase-time-observation --execution-feedback --baseline-anchored-residual \
  --deployment-gate --gate-advantage-margin 0.0 \
  --model logs/opp_262410/final_model.zip \
  --output $OUT/proposed_gated_262410.json

# 3)(可选诊断)提案不加门,看门的作用
#   去掉 --deployment-gate 即可,输出 proposed_ungated_262410.json
```

组表:`python -B -m eval.main_table --row "PureMPC=$OUT/pure_mpc.json" --row "Proposed=$OUT/proposed_gated_262410.json" ...`。

## 3. 看什么(头条指标)

**headline = 合法(零违约)完成率** `truth_geometry_zero_violation_completed`,而不是裸完成率——
因为 Pure MPC 会"非法完成"(冲出接近锥)。并排看:
- `outer_approach_violation_steps`(Pure MPC 应很高,提案应接近 0);
- completion / 合法 completion;成功集 `equivalent_delta_v_m_s`、`survival_s`;
- 配对 rescue/destroy(共同成功、仅提案成功、仅 MPC 成功);
- `worst_truth_normalized_margin`、非法进入、QP 不可行/零推力串。

**预登记(出结果前锁死,不设单一硬阈值):** 提案在 3 训练种子上整体稳定满足:合法完成率明显高于
Pure MPC;baseline retention 高(少破坏 nominal 成功);rescue>destroy;安全不退。任一项明显成立
且安全不退即继续。

## 4. 第一轮不好怎么调(沿这条线,不重新开题)

- **策略不学 / 一直冲进去(outer_approach 违约高):** 先调 **pre-entry reward**——当前 v1 只在
  pre-entry 安全项里加了"离开接近锥"的惩罚(potential 未改,因 potential shaping 是策略不变量、
  不改最优)。若不够,把 potential 改成两区域(pre-entry 拉向入口面半径 + 锥对齐,post-entry 才拉
  终端位姿),这是**首选旋钮**。
- **会等但大量超时:** 调 time penalty / commit gain / staging 半径范围。
- **rescue 有了但 destroy 多:** 调 critic 门(`--gate-advantage-margin` 提高)/ 残差幅值。
- **MPC 频繁 infeasible:** 查 shift fallback / slack;这是 formulation,不归因方向失败。

第二轮 3 种子仍完全无收益,才回上层重审机制;任务方向不因小样本撤销。

## 5. 已验证(sanity,非结果)

- 全套回归通过;新增机会任务单测(几何/守护/观测维数)全绿。
- 机会任务:Pure MPC / 提案(残差+门)/ 训练 三条链端到端跑通;obs 38D;outer_approach 计数生效;
  历史任务逐位不变。
- 上面 §0 的 0/2 与违约计数是 2 集 smoke,证明**机制成立**,不是性能结论——性能由 §1–§2 正式 3
  种子闭环给出。

## 6. 冻结底线

truth RK45 几何裁决安全;不削弱执行器/更新率/终端精度/翻滚率来造难度(外层锥对两臂同等生效,是
公平任务约束,不是 blind-MPC 假 gap);reward 不泄露答案;不 cherry-pick 种子/checkpoint;≥3 种子
看分布;每个数字指向 artifact;训练从零。
