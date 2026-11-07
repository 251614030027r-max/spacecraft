# 主线运行单:自适应同步—进入决策 + 强 MPC + 基线锚定耦合

写于 2026-09-18,上层沙箱。**方向已冻结**(见 `CLAUDE.md` 2026-09-18 纪律块)。命令在
`D:\py\DRL2` 跑,结果不必提交,回传迭代。核心:**RL 决定"怎么做任务"(现在同步多少、何时进入),
MPC 决定"怎样把当前意图安全飞出来"。** 机会是连续代价结构,不是硬门;Pure MPC 始终是合法强基线。

## 0. 这版相对上一版改了什么

- **去掉了外层硬走廊**(上一版会让 Pure MPC 大量违约=造 gap)。远场靠合理初始化 + 宽松参考不乱绕,
  真安全仍由 keep-out/FOV/速度/终端承担。
- 新任务 `precapture_adaptive_capture_environment_config()`:只放开初始分布(15–28 m、指向 25°、
  失败边界 35 m),控制难度全冻结;策略加 **staging 方向观测**(38D)。
- 动作沿用 2D `arrival_condition`,**正名为 capture progress / reference blend**(V2 再考虑显式
  sync_level)。
- **残差动作范围已修正(2026-09-18)**:commit 通道用增益 2,`a_commit=clip(1+2·residual)`——
  residual=0→完整 commit(=Pure MPC 逐位)、residual=−1→**完整惯性暂存**、中间连续。此前 residual
  只能退到中间 blend、表达不了完整暂存,现已修好并加单测(`test_most_negative_residual_recovers_full_inertial_hold`)。
- 新增 `--tumble-scale`,用于决策裕度标定和跨工况泛化。

## 1. 第 0 步:决策裕度标定(唯一一次,不是生死门)

**目的:** 只确认"任务里存在两种代价结构不同的合理策略"。**不是**证明 RL 能赢。理想结果是**交叉**:
慢翻滚时"较早共旋"更省,翻滚加快后"暂存后进入"的燃耗优势出现(代价是时间)。

3 个翻滚率 × 两种极端策略(同一 MPC),小样本:
```
mkdir -p eval/cal2
for R in 0.10 0.20 0.30; do          # 1.18 / 2.36 / 3.54 deg/s
  for ARM in desired_pose timed_entry; do   # 早共旋 / 暂存后进入
    python -B -m experiments.evaluate_hybrid_policy --episodes 8 --seed 262000 \
      --horizon 35 --parametrization arrival_condition --tumble-scale $R \
      --control $ARM --output eval/cal2/${ARM}_r${R}.json
  done
done
python read_cal.py            # 汇总表;默认读 eval/cal2/,也可传目录
```
只看 completion / equivalent-Δv / time / worst margin。判据放宽:**不要求漂亮交叉、不要求等待策略赢、
不要求某速率后突然模式切换**;只要"两种策略资源代价不同、且差异随翻滚率有变化",就足以说明任务有长期
决策空间 → **立即停,进入训练**。若确实太弱 → **调初始距离/初始范围/翻滚率范围**把物理决策空间做出来,
再继续同一主线(不换方向、不扫窗口角、不加门)。

> **`timed_entry` 定位**:truth-future-assisted scripted timing comparator(用真值未来相位辅助的
> 脚本择时对照),**不是 RL、不是理论最优、不是严格上界**。它好=暂存有潜在资源优势;它一般≠RL 没空间;
> 更不能因它某些 seed 失败再推翻任务。
>
> **reward 量级顺手核对(标定后,不做 grid)**:确认没有"物理上晚一点明显省很多油、但 reward 几乎
> 看不见这个差别"的情况。量级明显不合理才做**一次**物理尺度修正。

> 沙箱预跑(小样本,directional,非结论):见文末 §6。以你机器上的 8 种子为准。

## 2. 训练:3 种子,从零,新任务

```
for SEED in 262410 262411 262412; do
python -B -m train.train_hybrid \
  --steps 60000 --seed $SEED --run-name adp_${SEED} \
  --horizon 35 --parametrization arrival_condition \
  --baseline-anchored-residual --adaptive-task --device cpu
done
```
### 2b. 重训(奖励单位修正后,2026-09-19)

第一轮 262410/411/412 因**奖励单位 bug** 停训(合法完成 ~-75、hover ~+4,目标被反转;
详见 `docs/REWARD_UNITS_FIX_20260919.md`)。已修 `env/reward.py`(safety 项补 `time_step_s`,
不动任何权重),258 tests 通过。**从零重训**,换新 run-name 以免覆盖被污染的旧日志:

```
for SEED in 262410 262411 262412; do
python -B -m train.train_hybrid \
  --steps 60000 --seed $SEED --run-name adp_rf_${SEED} \
  --horizon 35 --parametrization arrival_condition \
  --baseline-anchored-residual --adaptive-task --device cpu
done
```
- 仍是 3 个独立进程/终端并行(每终端一个 seed)最快;瓶颈是 MPC 求解,别上 GPU。
- **健康检查有两个节点,用途不同,别混**(2026-09-19 用户裁定,替代此前 handoff/runsheet
  的 10k-vs-30k 不一致):
  - **10k = 灾难检查,纠错性质,不是方向门。**只问一件事:合法完成的回报是否已经爬到 hover 之上?
    若仍是"完成 ≤ hover",奖励信号还有问题 → **立刻停、别烧到 60k**。10k 的低分**不得**当作
    "方向失败"来否决主线 —— 早期/smoke 不是方向门,这是纪律明令禁止的。
  - **30k = 健康/趋势确认**(完成 ~+17、干净完成 +26 vs hover +4-6);**60k = 正式完成**。
  - 两处都看:读 `train.monitor.csv` 不花钱。

- `--adaptive-task` 自动:新任务 + staging 观测(38D)。残差 0 = 固定设定点 Pure MPC 逐位。
- **加速(重要):3 个种子并行跑**(规则允许训练并行)→ 墙钟 ~1/3;瓶颈是 MPC 求解,**别上 GPU**。
  **注意上面的 `for SEED` 只是串行循环**;真正并行要开 **3 个独立进程/终端**(每个跑一个 seed),
  或写并行启动脚本。调试可先 h20 看曲线,但**最终正式训练/评估统一 h35**(不要拿 h20 训练/h35 评估
  当最终论文配置)。

## 3. 评估:两个层次

**(a) nominal 保底(baseline retention):** 同一 48 种子块 262000、h35、nominal 翻滚:
```
# Pure MPC(强基线)
python -B -m experiments.evaluate_hybrid_policy --episodes 48 --seed 262000 --horizon 35 \
  --parametrization arrival_condition --adaptive-task --control desired_pose \
  --output eval/adp/pure_mpc_nominal.json
# 提案(残差 + critic 优势门)
python -B -m experiments.evaluate_hybrid_policy --episodes 48 --seed 262000 --horizon 35 \
  --parametrization arrival_condition --adaptive-task \
  --phase-time-observation --execution-feedback --baseline-anchored-residual \
  --deployment-gate --gate-advantage-margin 0.0 \
  --model logs/adp_rf_262410/final_model.zip \
  --output eval/adp/proposed_nominal_adp_rf_262410.json
```
判据:提案**保住** Pure MPC 的 completion 和安全(不破坏基线),而不是要求 nominal 上省油。

**(b) 跨工况泛化(真正的头条证据,定位为 zero-shot robustness / adaptation):**
nominal 上训出的**同一策略、同一 reward**,只改 `--tumble-scale` 到未见速率,看行为是否自适应、
整体权衡是否更好。**这是 zero-shot 测试**:好=很强的结果;一般也**不否定**自适应主线(正式 TAES 版
可再在合理 tumble-rate 分布上训练、在未见/边界速率上测,reward/policy/MPC 不变;当前先把 nominal V1
跑通,不为泛化提前扩大训练复杂度)。
```
for R in 0.10 0.20 0.30; do
  python -B -m experiments.evaluate_hybrid_policy --episodes 48 --seed 262000 --horizon 35 \
    --parametrization arrival_condition --adaptive-task --tumble-scale $R \
    --phase-time-observation --execution-feedback --baseline-anchored-residual \
    --deployment-gate --model logs/adp_rf_262410/final_model.zip \
    --output eval/adp/proposed_r${R}_adp_rf_262410.json
  python -B -m experiments.evaluate_hybrid_policy --episodes 48 --seed 262000 --horizon 35 \
    --parametrization arrival_condition --adaptive-task --tumble-scale $R \
    --control desired_pose --output eval/adp/mpc_r${R}.json
done
```
看:同一策略在慢/中/快翻滚下,**同步—进入行为是否随状态改变**(commit 时刻、staging 半径),
并在 completion/Δv/time/margin 上保持更好的整体权衡;而固定参考 MPC 在偏离其"舒适工况"时退化。
**这是 MPC 和手写规则做不到、只有学习层能给的——论文主结果。**

## 4. 指标(独立报告,不要用单一总 reward)
completion、legal completion(零违约完成)、equivalent Δv、completion/survival time、
normalized safety margin、constraint violations、peak force/torque、rescue/destroy、
QP fallback/infeasible、common-success 资源代价、以及**策略行为**(commit 时刻分布、staging 半径)。

## 5. 第一轮不好怎么调(沿主线,不重新开题)
- 一直早共旋 → 查 physical fuel penalty 权重、action mapping、近场自由度;
- 一直等/超时 → 查 time cost、progress shaping、terminal latch credit、residual 幅值;
- 乱切/不稳 → 查 bounded reference update、residual scaling、观测 Markov 性、MPC 反馈;
- Pure MPC 与 hybrid 几乎一样 → 查高层参考是否真改变长期轨迹、近场是否仍太单目标跟踪;
- MPC 频繁 infeasible → 查 formulation/约束激活,不归因任务失败;
- 3 种子完全不学 → 允许一轮 reward/action/coupling 针对性优化再训;
- **决策裕度弱 → 调任务参数范围(近场距离/初始范围/翻滚率),不换方向。**

## 6. 沙箱已验证(sanity,非结论)
- 全套回归通过(新增 adaptive/opportunity/timing/gate 测试);新任务 obs 38D、无外层硬约束、
  历史任务逐位不变。
- 训练/评估/门端到端 smoke 跑通(train→save→load→eval;38D adaptive 模型 + critic 门 + Pure MPC)。
- `--tumble-scale` 已接通(标定/泛化用)。**决策裕度标定(§1)沙箱太慢,已交给你机器跑**——命令在
  §1,8 种子几十分钟即可;它只用于校准参数范围,不是方向门。**先跑 §1,看到 trade-off/交叉再开 §2 训练;
  若弱就按 §5 调范围。**

## 7. 冻结底线
truth RK45 几何裁决安全;不削弱执行器/更新率/终端精度/翻滚率造难度;不加会让 Pure MPC 违约的
远场硬约束;reward 不奖励等待/共旋/有利相位;不 cherry-pick;≥3 种子看分布;数字指向 artifact;
训练从零。架构保持可分析(RL=有界参考生成器、慢时标;MPC=快跟踪子系统;残差有界、有 baseline 回退)。
