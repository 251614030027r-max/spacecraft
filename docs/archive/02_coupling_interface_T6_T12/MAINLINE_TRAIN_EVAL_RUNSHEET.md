# 主线闭环运行单:双向基线锚定 SAC–MPC(可训练/可比较/可迭代)

写于 2026-09-17,上层沙箱。落实 2026-09-17 方向纪律:**目标是把这条主线做成完整闭环并跑起来,
不再以小规模探针否决方向。** 择时 A/B 的脚本启发式为负(B 17/35 < A 26/35),但那不证伪
**学习的**残差方法——本运行单就是把学习残差方法做成 `train → evaluate → compare → iterate`
的完整闭环,交由本地正式训练裁决。命令可直接在 `D:\py\DRL2` 跑,**结果不必提交**,回传给我迭代。

## 0. 现在实现到哪一步(本轮补丁)

| 部件 | 状态 | 入口 |
|---|---|---|
| 基线锚定残差(残差 0 = Pure MPC 逐位) | ✅ 已实现、已测 | `train_hybrid --baseline-anchored-residual` |
| 残差模型的评估路径(残差接口对齐) | ✅ 本轮新增 | `evaluate_hybrid_policy --baseline-anchored-residual` |
| **critic 优势部署门 + nominal 回退** | ✅ 本轮新增(V1:仅 critic 优势) | `evaluate_hybrid_policy --deployment-gate` |
| manifest 评估 flag 补齐 `--baseline-anchored-residual` | ✅ 本轮修复(原缺失=评估会用错映射) | train 侧 |
| MPC 前瞻可行性证书 + 反向通道(观测) | ⏳ 未做(下一个单因子 V2,§4 有设计) | — |

部署门(V1):每决策比较 `A_task = Q_min(s, a_policy) − Q_min(s, a_nom=0)`,`A_task > margin`
才偏离,否则回 nominal(固定设定点 Pure MPC)。用训练好的 SAC critic(携带长时域回报),解决
"用短时 MPC 仲裁长期价值"的循环。残差锚定保证偏离有界、门错时仍安全回退。**MPC 可行性合取项
是 V2**(见 §4);V1 先把 critic 优势门这条主机制跑通、可比较。

## 1. 训练:3 种子,从零,预算对齐 T12(本地跑)

```
# 主接口:arrival 残差,3 种子,从零,60000 决策(与 T12 同量级,便于对比)
for SEED in 262410 262411 262412; do
python -B -m train.train_hybrid \
  --steps 60000 --seed $SEED --run-name residual_${SEED} \
  --horizon 35 --parametrization arrival_condition \
  --baseline-anchored-residual --device auto
done
```

- 每个 run 的 `final_model` 是正式模型;checkpoint 只排错不择优。
- `logs/residual_<seed>/manifest.json` 的 `evaluation_flags` 现在会带 `--baseline-anchored-residual`
  (本轮修复;此前缺失会导致按 manifest 评估时用错动作映射)。观测 35D、动作 2D。
- horizon 用 **35**(与 Pure MPC 行、最终评估一致);若想省机时先探,可先跑 h20 一个种子看曲线,
  但正式对比用 h35。

## 2. 评估:同一 48 种子块的四行对照(h35,deterministic,串行)

种子块 262000–262047,`--episodes 48 --seed 262000 --horizon 35`。四行走**同一** `main_table`/
`entry_channel`/真值裕度路径:

```
OUT=eval/results/mainline_YYYYMMDD
# 1) Pure MPC(固定设定点)——最终 Pure MPC 数字
python -B -m experiments.evaluate_hybrid_policy --episodes 48 --seed 262000 --horizon 35 \
  --parametrization arrival_condition --control desired_pose \
  --output $OUT/pure_mpc.json

# 2) 直接接管对照(arrival,无残差锚、无门 = T12 式)——用已有 T12 arrival checkpoint
python -B -m experiments.evaluate_hybrid_policy --episodes 48 --seed 262000 --horizon 35 \
  --parametrization arrival_condition --phase-time-observation --execution-feedback \
  --model <T12_arrival_final_model.zip> \
  --output $OUT/direct_takeover.json

# 3) 提案(残差,无门)——每个种子一行,报分布
python -B -m experiments.evaluate_hybrid_policy --episodes 48 --seed 262000 --horizon 35 \
  --parametrization arrival_condition --phase-time-observation --execution-feedback \
  --baseline-anchored-residual \
  --model logs/residual_262410/final_model.zip \
  --output $OUT/proposed_ungated_262410.json

# 4) 提案 + critic 优势门(主方法)——同一模型加门
python -B -m experiments.evaluate_hybrid_policy --episodes 48 --seed 262000 --horizon 35 \
  --parametrization arrival_condition --phase-time-observation --execution-feedback \
  --baseline-anchored-residual --deployment-gate --gate-advantage-margin 0.0 \
  --model logs/residual_262410/final_model.zip \
  --output $OUT/proposed_gated_262410.json
```

第 3、4 行对每个训练种子各跑一次(3 个模型)。门的 `--gate-advantage-margin` 默认 0.0;若要看
门的敏感性,扫几个 margin(0 / 0.5 / 1.0),但主结果先用 0.0。

组表:`python -B -m eval.main_table --row "PureMPC=$OUT/pure_mpc.json" --row "Direct=$OUT/direct_takeover.json" --row "Proposed=$OUT/proposed_gated_262410.json" ...`。

## 3. 预登记裁决(与 `CLAUDE.md`/执行单一致,判在正式 3 种子闭环上)

**主判据**——3 训练种子整体稳定:
- `completion(Proposed 门控) > completion(Pure MPC)`;
- **baseline retention** `R = |S_proposed ∩ S_MPC| / |S_MPC|` 高(少破坏 nominal 成功);
- **rescue > destroy**,净完成增益为正;
- 燃料分三类报(common-success / rescued / overall),common-success 不得普遍恶化;不要求
  overall 低于 Pure MPC;
- truth 安全裕度不明显下降;门不得靠更激进撞约束换成功。

参照天花板(T12 每回合基线保持上界)≈ **38/48(+6)**:这是机制论文的适度真实增益,不是碾压。
**判在这个正式闭环上,不在任何中途小样本上**(2026-09-17 纪律)。若门控提案稳定不超过 Pure MPC
或大量 destroy → 回上层讨论机制(不是直接砍方向)。

## 4. 下一个单因子(V2):MPC 前瞻可行性证书 + 反向通道

现在部署门只用 critic 优势。方法的完整形态要加 MPC 侧的**前瞻可行性证书**(对提案意图做一次
horizon 预测,压成 2–4 维预测可行性/安全裕度),两处用途:
1. **部署门合取项**:`偏离 ⇔ A_task>margin 且 证书判局部可行`(现在只有前半);
2. **观测反向通道**:证书进下一步高层观测,形成"提案↔可行性"双向协商(区别于 T7 已证无益的
   原始执行遥测)。

这会改观测维数(需在 manifest 记新维数),是独立单因子,单独一个 commit + smoke + 重训。
现在**先不混进主线残差训练**,让 V1(残差 + critic 门)先形成可比较闭环;V2 作为 Phase D 之前的
机制增强,消融时可回答"去掉证书/去掉门各损失多少"。

## 5. 不变的底线(引 `CLAUDE.md`)

truth RK45 几何裁决安全;Pure MPC 冻结,最终统一重跑;不造 gap;reward 不泄露答案;不 cherry-pick
种子/checkpoint;≥3 训练种子看分布;每个数字指向 artifact。训练从零、不 warm-start。

## 6. 已验证(sinity,非结果)

- 定向测试全绿(残差映射、门 critic 查询与回退、缓存一致性等);全套回归通过。
- 残差 smoke 训练(60 决策)跑通、存 `final_model`、manifest 评估 flag 已含 `--baseline-anchored-residual`。
- 门控评估 smoke(1 集,60 决策模型)端到端跑通、记录 `gate_deviation_fraction`/`gate_mean_advantage`。
  这些只证明闭环正确,**不是**性能结果;性能由 §1–§2 的正式 3 种子闭环给出。
