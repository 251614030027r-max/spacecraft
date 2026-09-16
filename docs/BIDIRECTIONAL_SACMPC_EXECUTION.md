# 历史执行单:双向基线锚定任务级 SAC–MPC

**状态更新（2026-09-16）：**本文件的“直接开工 / Phase B 三种子训练”已被更新的择时价值确认闸门覆盖。当前仅保留 baseline-anchored residual 代码作为 foundation；在新的零训练 A/B 证明择时有价值且上层给出后续指令前，不执行本文件的训练命令或评估计划。见 `TIMING_VALUE_FOUNDATION_HANDOFF_20260916.md`。

写于 2026-09-16,上层沙箱,基点 `8654884`。方向不再讨论。目标:把 T12 已测到的互补性
(SAC 救回 6 / 破坏 4)变成**稳定净收益**,不破坏强基线。纪律见 `CLAUDE.md`"Research execution
discipline"——`implement → smoke → 3-seed train → evaluate`,不再堆 probe。

## 0. 方法结构(锁定,不再改)

- **上层残差**:SAC 输出 `arrival_condition` 上的低维残差 Δa,候选 `a_cand = Π(a_nom + D·Δa)`;
  **Δa=0 逐位回到 nominal(固定设定点 Pure MPC)**。学"要不要偏离、偏多少",不重学整个任务。
- **critic 优势门(部署)**:`A_task = Q_min(s,a_cand) − Q_min(s,a_nom)`,由 SAC critic(完整回报
  训练,携带长时域)判断"长期值不值得偏离"。
- **MPC 证书(下行)**:对候选任务意图,MPC 返回 2–4 维预测可行性/安全裕度证书,进下一步观测。
- **执行门**:`偏离 ⇔ A_task>δ 且 MPC 证书判局部可行`,否则回 nominal。

## 1. 分工(务必按此)

- **Phase A(实现 + smoke)= 上层沙箱(我)做**,交付代码补丁 + 单元测试。下层**不自行实现**,
  等补丁。
- **Phase B(3 seed 正式训练)= 下层做**,拿到我的补丁后开训。
- **Phase C(统一正式评估)= 下层跑,上层组表**。
- **Phase D(最小消融)= 结果为正后再做**。

## 2. Phase A —— 上层沙箱实现清单(我负责)

1. baseline task action `a_nom` 明确定义(arrival 全提交 = 固定设定点);
2. arrival 残差映射(Δa=0 → nominal bitwise,残差幅值 D、合法域投影 Π);
3. proposal-conditioned MPC 证书(复用 controller 现有预测诊断,压到 2–4 维);
4. 证书进 high-level 观测(新观测标志,维数写进 manifest);
5. critic-advantage 部署门 + nominal 回退(在 evaluate 路径实现,用 SAC critic);
6. 单元测试 + 一个极短 smoke episode(闭环能跑、无 NaN、fallback 触发、Δa=0 复现 nominal)。

**允许的验证仅限正确性**(shape/语义/fallback/无 NaN/solver 正常/残差 0 复现)。**禁止**:为每个
小模块跑 48 seed、再建 oracle 上界、每个阈值大网格、因小 smoke 不理想改方向。Pure MPC 不动。

交付:代码补丁(下层 `git am -3`)+ 一份"训练接口就绪"说明,含**最终训练命令的确切 flag 与观测维数**。

## 3. Phase B —— 下层正式训练(等我的补丁)

- **一个主接口**(arrival 残差),3 个训练种子,**从零**,budget 与 T12 同量级(60000 决策)便于对比;
- `final_model` 为正式模型,checkpoint 只排错不择优;
- truth-state 主方法先训;estimated-state 是否单独训由上层按时间定,优先保证主线完成;
- 训练时残差候选经合法映射 + MPC 硬可行性保护;证书进下一步观测;**按正常 episode reward 学**,
  **不给"被接受/拒绝"人工 reward**;replay 记录 proposal / 实际执行 / gate / 证书;
- **单个 seed 曲线难看不改方向**。只有以下才停下做结构排错:三种子全程完全不学 / 系统性 NaN /
  结构上产生不了非 nominal 动作。
- 确切启动命令由我在 Phase A 交付时给全(含观测维数、残差/证书 flag);现在不要凭猜开训。

## 4. Phase C —— 统一正式评估(下层跑,上层组表)

配置冻结后,同一 48 种子块 262000–262047、h35、deterministic,重跑五行:

1. Pure MPC — truth(此时才生成"最终 Pure MPC 数字");
2. Direct task-level SAC–MPC — truth(无残差锚 / 无门,= T12 式直接接管,作对照);
3. **Proposed 双向基线锚定 — truth**;
4. Pure MPC — estimated;
5. Proposed — estimated。

指标:completion、success fuel/力冲量、success time、truth 安全裕度/违约、非法入口、
QP 不可行/零推力串;compute 待主线为正后单独串行测,不提前 profiling。

## 5. 通过判据(预登记,B/C 出结果前锁死)

**主判据**——三训练种子**整体稳定**满足(不要求每个 seed 都超):

- `completion(Proposed) > completion(Pure MPC=32/48)`;
- **baseline retention** `R = |S_proposed ∩ S_MPC| / |S_MPC|` 高(新方法少破坏 nominal 成功);
- **rescue > destroy**,净完成增益为正;
- 燃料分三类报(common-success / rescued-new-success / overall);困难 seed 允许多花燃料,
  但 common-success 不得普遍恶化;**不要求 overall fuel 低于 187.7 N·s**;
- truth 安全裕度不明显下降,完成局零关键真值违约;门不得靠"更激进撞约束"换成功率。

**参照天花板**:每回合基线保持上界 ≈38/48。Proposed 明显靠近且 retention 高、rescue>destroy →
方法成立,进 Phase D + 写论文;若三种子稳定**不超过或大量 destroy** → 才回上层讨论收束。

## 6. Phase D —— 最小消融(仅主结果为正后)

优先两个:① 去掉下行 MPC 证书;② 去掉 critic-advantage 门 / 残差锚其一。必要时加一个简单
heuristic 门作对照。不一开始就训练一排消融。

## 7. 护栏(引 `CLAUDE.md`)

truth 几何裁决安全;Pure MPC 冻结;不造 gap;reward 不泄露答案;不 cherry-pick seed/checkpoint;
论文数字指向 artifact;不加与主 claim 无关的模块(新 selector 网络、Q/R 学习、actor warm-start、
critic terminal cost、rollout+Newton、多抓取点、人工障碍等一律不加)。

## 8. 现在的状态

上层沙箱开始 Phase A 实现,完成 smoke 后交付代码补丁 + 确切训练命令。**下层在收到该补丁前,
不训练、不自行实现、不动 Pure MPC 与 T12 模型。**
