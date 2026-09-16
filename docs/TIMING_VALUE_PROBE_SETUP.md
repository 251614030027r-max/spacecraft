# 择时价值 A/B 探针:上层交付的两件前置 + 运行说明

写于 2026-09-16,上层沙箱。基点 = `8654884` + 任务重构 6 提交。本文交付下层执行单
`下层执行:择时价值确认实验` 第 3 节点名的两件前置,并给出 A/B 的确切命令、冻结边界、
记录量与**待两窗口锁定**的预登记判据。**本轮仍不训练 SAC。**

## 0. 这两件是什么

| 前置 | 交付物 | 位置 |
|---|---|---|
| ① 参数放开的 staging 初始分布配置 | `precapture_timing_probe_environment_config()` | `env/phase2_env.py` |
| ② B 臂脚本择时 harness | `--control timed_entry`(`ScriptedEntryTiming`) | `experiments/evaluate_hybrid_policy.py` |

两者都在**共享测量路径** `evaluate_hybrid_policy` 上,A/B 两臂走同一套 metrics /
entry-channel / 真值裕度代码,唯一差别是 commit 时机。没有 fork 测量路径。

## 1. 关键设计决定(我做的判断,附理由,可被两窗口否决)

1. **相位门默认关(gate 180°)。** 主 A/B 用 `--timing-probe` 时 `entry_phase_gate_deg`
   默认 180 → `gate_cos ≤ -1` → 门关。**理由**:择时"值钱"的证据必须来自物理
   ——端口正对你时终端几何/相机视场天然更容易——**绝不能来自"硬门拒绝了 A 臂的穿越"**。
   门制造的差异不是择时有价值的合法证据(下层交接自己也这么写)。90° 门仍可用
   (`--entry-phase-gate-deg 90`),但**只作开发期诊断护栏,不是论文机制、永不收紧到 60°**。
2. **只放开"任务选择空间",冻结"控制难度"。** 配置只改初始范围(15→28 m,严格低于
   30 m 距离失败边界,留 2 m 漂移余量;字面"15–30"不可用——起点即在失败面上会被首步外漂
   打失败)和初始指向误差(5°→25°,仍在 50° FOV 内)。**接近方位角本就在采样器里全开**
   (`sample_precapture_planning_chaser_state` 方位角 uniform[-π,π]、极角 ~40° 离轴外的锥),
   **初始目标相位本就逐集采样**(`phase2_target_phase_sampling`)——那正是让"进入时刻"分散的来源,
   不需要另改。**冻结不动**:±5 N/±0.6 N·m、0.1 s 更新率、MPC 模型、走廊/FOV/速度/终端约束、
   0.041 rad/s 翻滚速率。
3. **翻滚速率变化(执行单 §2 第 4 项)本轮不实现,建议推迟。** 它是四个放开旋钮里
   **唯一触碰控制难度**的(共旋力 ∝ ω²r);而回答"择时值不值钱"**不需要**它——相位分散
   已由逐集相位采样给出(8 种子探针在固定 ω 下就已测到 [-0.68,+0.55])。把它留作后续
   稳健性变体、且要单独论证物理诚实,而不是混进主 A/B 制造 headroom。**这是我主动省略的一项,
   请两窗口确认是否接受。**
4. **B 臂是真值未来相位辅助的择时探针,不是严格最优上界、反应式阈值或恒体率外推。** 见 §3。理由:一次接近
   ≈0.58 个翻滚周期,"现在有利就 commit"会在到达时早已转走;必须按转移时间提前 commit。
   且预测未来端口朝向**用环境已有的真值目标轨迹**(`cache_target_trajectory`,真实 6-DoF 自由
   刚体翻滚:非球惯量、重力梯度、J2),不是当前角速度长时外推——后者在几十~上百秒 lead 上
   不可信,会让"预测器本身"而非"择时"限制结果。B 使用非部署可得的未来真值,但 hold 策略、
   转移时间估计和阈值仍是固定启发式,**不能称严格最优上界或可部署控制器**;本探针只测
   "在这一明确策略下,知道未来有利相位后择时是否有收益"。

## 2. 运行命令(下层执行,同一 48 种子块,h35,deterministic)

种子块与 episode 数**对齐 Pure MPC 行 / T12 / 探针 2**:`--seed 262000 --episodes 48 --horizon 35`。

```
# A 臂:立即进入(= 现固定设定点 Pure MPC,一来就 commit)
python -B -m experiments.evaluate_hybrid_policy \
  --episodes 48 --seed 262000 --horizon 35 \
  --parametrization arrival_condition --timing-probe \
  --control desired_pose \
  --output eval/results/timing_ab_20260916/A_immediate.json

# B 臂:脚本择时(hold 在惯性暂存点,预测到达相位有利再 commit)
python -B -m experiments.evaluate_hybrid_policy \
  --episodes 48 --seed 262000 --horizon 35 \
  --parametrization arrival_condition --timing-probe \
  --control timed_entry \
  --output eval/results/timing_ab_20260916/B_timed.json
```

诊断变体(**可选,非主判据**):加 `--entry-phase-gate-deg 90` 看硬门下的对照——仅诊断,
不得作为"择时值钱"的证据。B 臂的两个旋钮:`--commit-favourability-cos`(默认 cos45°≈0.707)、
`--commit-lead-speed-m-s`(默认 0.20,转移 lead = range/speed)。若主结果对这两个旋钮敏感,
按 §5 报敏感性,别只报一个点。

## 3. B 臂真值未来相位辅助逻辑(`ScriptedEntryTiming`)

每个决策(2 s):未 commit 时,取当前距离估转移 lead `τ = range / lead_speed`,**从本集缓存的
真值目标轨迹取 t+τ 时刻的目标姿态**(`_target_trajectory[step+round(τ/dt)]`,即真实 6-DoF 翻滚
到那一刻端口朝向,不是恒体率外推)算预测到达相位有利度 `pred`;`pred ≥ commit_cos` 即 commit
(棘轮锁定,之后恒 commit)。若剩余时间不足以再等一个转移窗口则强制 commit(永不因从未 commit
而超时)。commit 后执行的就是 A 臂那条固定设定点指令,逐位相同——**B 从不削弱 MPC,只改 when**。
主 A/B **均开启目标轨迹缓存**,并按同一索引步进相同种子的目标自由翻滚;仅 B 读取未来索引。
这避免了 A 逐步积分、B 缓存步进带来的实现路径差异。A/B 的差别是择时动作和 B 的未来真值辅助,
不是控制器强度差别;B 并非经过全时刻/全策略优化的上界。

smoke(2 集,h20,gate off,恒体率版)已验证闭环正确:两集均完成;commit 在 t=48 s / 4 s,
**实际合法穿越相位 = 0.707 / 0.854**——提前 commit 让 chaser 恰在有利相位抵达入口面,机制符合
预期;compute p95 0.52x 在预算内。真值-轨迹版按此更换预测源,行为更干净。

## 4. 每集记录的量(A/B 都记,便于对照)

`records[i]` 新增:`commit_time_s`、`favourability_at_commit`、`predicted_favourability_at_commit`、
`peak_force_n`、`peak_torque_nm`(峰值控制负荷,供相位→峰值负荷分析);每个 `entry_crossings[k]`
新增 `favourability`(穿越时刻的相位有利度)。既有列不变:completion、成功燃料/力冲量、成功时间、
`worst_truth_normalized_margin`、非法穿越、QP 不可行/零推力串、`main_table`。payload 顶层记
`timing_probe`、`entry_phase_gate_deg`、`timed_entry_settings`,产物自描述。

## 5. 预登记判据(**提案,B/C 出结果前由两窗口锁死**)

主对照:同一配对种子块(262000–262047)上 B vs A。择时"值钱"需**同时**满足三层
(**不设单一硬阈值**——降幅/涨幅由分布相对种子噪声判"稳定、明显",不把某个 10% 之类的点当生死线):
- **机制成立**:B 的合法穿越相位显著高于 A(A 是 naive 自然穿越,8 种子均值 0.16);
  即 B 确实把进入挪到了有利相位——这是"择时起作用"的直接证据。
- **且有净收益(任务级,二选一即可,非都要)**:配对种子下 B 相对 A 出现**明显**任务级改善——
  **要么完成率提高**(如 32/48→更高),**要么共同成功集上控制消耗明显下降**(等效 Δv / 峰值负荷);
  两者不必同时。例:32→40 而 common-success fuel 仅降 6% 是很强的证据;成功率不变而 fuel 稳定降 15%
  也成立。**等待的任务时间代价必须完整并排报告**(commit_time、survival_s),不得藏起来;燃料按
  common-success / rescued / overall 三类分报,common-success 不得普遍恶化。
- **安全不退**:B 的 `worst_truth_normalized_margin` 不明显低于 A;不靠更激进撞约束换成功。

分支(与下层执行单一致):
- 差异**明显** → 择时值钱,主线成立 → 上层再建外层几何物理化 + 耦合;
- 差异**弱** → 外层几何还不够"有任务性" → **继续强化外层几何(加物理诚实约束),不是调算法**;
- 强化后仍弱 → **停这条线**,不靠调门控角度制造 headroom。

## 6. 不做 / 冻结边界

不训练 SAC;不把硬门当机制、不收紧到 60°;不削弱 MPC/执行器/终端精度/更新率;
不靠调门控角度造 headroom;truth RK45 几何是安全唯一裁决;进决策/论文的数字指向本仓库产物;
≥3(此处 48)种子看分布。

## 7. 交付回上层

A/B 各自 completion/燃料/时间/安全裕度分布 + **按相位分箱的穿越相位对照**(A 自然穿越相位
vs B 择时穿越相位)。下层不做裁决,分布照报。

**特别要出的一张图(立意级证据,比"B 38>32 A"更重要)**:把 **A 的自然进入相位有利度 →
结果** 画出来——横轴 `favourability`(A 每集合法/非法穿越时的相位有利度),纵轴分别是
完成与否、成功燃料(等效 Δv)、峰值控制负荷(`peak_force_n`)。若出现清楚的连续趋势
(相位越有利→越易成功/越省油/峰值负荷越低),那就直接证明**捕获相位本身是任务性能的重要状态
变量**,而不只是"B 比 A 好"。这张图不需要额外跑,`records` 里的 `entry_crossings[].favourability`、
`completed`、`equivalent_delta_v_m_s`、`peak_force_n` 已足够画。
