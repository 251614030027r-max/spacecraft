# 非合作预捕获 S1:观测接口 + 控制接口已落地,观测窗口已测实,下层执行单

写于 2026-09-15,上层沙箱。数字均指向可复现产物与沙箱 venv 下的 `pytest`。本文替换早前
同名初稿——**其中"观测窗口不会免费涌现"的判断已被整周扫描推翻,见 §3**。

---

## 1. 已完成(沙箱,单元测试通过)

### 1.1 非合作观测接口(单因子:观测来源)

- `precapture_perception_environment_config()`:在冻结的 `precapture_planning` 上,唯一改动是
  打开 A1 相机 + 相对 EKF,上层观测从 24D 真值全状态换成 29D 估计态
  (`precapture_planning_estimated_v1_29d`:EKF 估计相对状态构建的 24D 核心 + 4 个协方差
  不确定通道 + 可见特征比例)。真值仍裁决动力学/奖励/终止/几何/评估;`perception=None` 逐位
  复现原任务。复用 A1/A3 已有 EKF/相机机件。

### 1.2 非合作控制接口(单因子:控制器所用状态)

- `experiments/evaluate_hybrid_policy.py` 新增 `--perception` 与 `--control-source {truth,estimated}`
  (默认 truth,逐位不变)。`estimated` 时,MPC 的相对状态与目标位姿改用 EKF 估计
  (目标位姿由已知 chaser + 估计相对量经 `reconstruct_target_state` 反解),**目标出视场时
  控制器只能靠停滞的估计滑行**——这就是非合作 operational baseline 的退化机制。
- 违约裁决仍用真值(`normalized_precapture_truth_margins` 保持真值)。

### 1.3 测试

- 全套 `python -B -m pytest -q`:**235 passed**(228 基线 + 7 项非合作观测测试)。
- 估计控制路径已用行内 smoke 复跑 40 控制步 × truth/estimated,MPC 每步给出有限非零 wrench,
  估计误差达 0.7 m,无异常。

---

## 2. 观测接口的就绪证据:估计误差真实且大

零动作、8 种子 × 150 步、`precapture_perception`(N=1200):估计位置误差中位 **0.33 m**、
p90 0.57 m、max 1.5 m;姿态中位 1.35°。**中位数就超过 0.25 m 的完成容差**,最大约为入口盘
半径(3.15 m)的一半。非合作注入的是真实、决策相关的不确定性,不是人为噪声。

## 3. 观测窗口:整周扫描证实其存在(推翻初稿判断)

初稿因零动作只扫过约 35° 相位,曾误判"观测窗口不会免费涌现"。用**整周静态扫描**
(固定 chaser,目标绕自旋轴转满 2π,72 个相位采样/种子,直接 `measure_visible_features`)重测:

| 种子 | 可见特征 min–max | 均值 | 一周中 0 可见的比例 |
|---|---|---|---|
| 262000 | 0–5 | 2.99 | 40% |
| 262001 | 0–5 | 2.36 | 53% |
| 262002 | 0–5 | 2.08 | 58% |
| 262003 | 0–5 | 2.92 | 42% |
| 262004 | 0–5 | 2.01 | 60% |
| 262005 | 5–5 | 5.00 | 0%(自旋轴近视线,特征面不扫出去) |

**结论(实测)**:多数初始条件下,目标特征面在一个 152 s 自旋周期里只有约 **40–60% 时间可见**,
其余时间 **0 特征可见**——追踪器周期性"失明"。因此"当前相位可不可观测、要不要现在进入还是等
下一个可观测窗口"是**物理上真实存在**的决策,由体固定相机 × 翻滚特征面自然产生,**无需手加
捕获机会 latch**。少数种子(如 262005)因自旋轴指向而全程可见——决策内容对多数而非全部初始
条件存在,这符合真实且提供了种子间多样性。

这一条同时兑现了收敛方向文的 §5 因果链(部分可观测 ⇒ 机会不确定 ⇒ 需要长期任务决策),
且是免费的,不膨胀任务。

---

## 4. 下层执行单(决定方向,零/少训练)

### 探针 1 —— 整周观测窗口:**已由上层沙箱完成**(见 §3),无需下层重跑。

### 探针 2 —— 非合作 Pure MPC(估计) vs 真值,48 种子:**接口已就绪,直接可跑**

对照已验证的真值 Pure MPC(32/48、78.8 s、187.7 N·s):

```
# 非合作固定设定点 MPC(飞在估计上)
python -B -m experiments.evaluate_hybrid_policy \
  --control desired_pose --perception --control-source estimated \
  --episodes 48 --seed 262000 --horizon 35 --parametrization arrival_condition \
  --output D:/py/DRL2/logs/precap_noncoop/pure_mpc_estimated_48.json

# 对照:同配置但飞在真值上(应≈复现 32/48,作为同代码路径的锚)
python -B -m experiments.evaluate_hybrid_policy \
  --control desired_pose --perception --control-source truth \
  --episodes 48 --seed 262000 --horizon 35 --parametrization arrival_condition \
  --output D:/py/DRL2/logs/precap_noncoop/pure_mpc_truth_48.json
```

- 串行单进程;保留完成数、成功均时/力冲量、七类真值违约、非法穿越、QP 不可行、最长零推力串。
- **判据**:估计行完成率明显低于真值行(且真值行≈32/48)→ 非合作真造出 fixed MPC 打不动的
  困难态,**新方向的承重假设成立**,值得继续到探针 3 + 训练;基本不掉 → EKF 滑行足够好,
  当前几何下非合作对强 MPC 无实质冲击,须重设观测/相位耦合,别急着训练。
- 注:`--control-source truth` 行是同代码路径的一致性锚(应与历史 32/48 一致);若不一致先查
  `--perception` 是否改变了初始条件采样(理应不改,已 `perception=None` 逐位验证)。

### 探针 3 —— 基线保持离线上界:**待探针 2 确认有余量后再建**

需要一个**新的反事实重放**工具(每个决策点分叉,分别沿 SAC 提议与 nominal a=(+1,·) 滚到
回合结束,取更优者),且必须对 T12 训练模型运行(模型在下层机器)。现有 `replay_hybrid_policy`
只存逐决策航点、不存可反事实滚动的状态,故这是新工具。**上层沙箱可在探针 2 确认余量后编写并
用哑策略做单元测试,再交下层对 T12 模型跑。** 探针 2 是当前紧前项,探针 3 不阻塞它。

---

## 5. 尚未做、且此刻不应做的(附理由)

- **耦合 arrival 接口(hybrid_env)改用估计**:非合作**训练**时,上层动作→航点的映射也应基于
  估计而非真值(现为 `self.env.target_state` 真值)。这动到冻结的 T6 输入,且只有在决定训练
  非合作耦合策略时才需要——属探针 2 确认余量之后的事,现在做是投机。已定位改点
  (`env/hybrid_env.py:399/506` 读 `target_state`),留待方向确认。
- **手加相位相关捕获机会项**:§3 已证明机会从可观测性免费涌现,**不需要**再往任务里加 latch
  (也避免"自造 gap")。

## 6. 复现入口

- 观测配置 `env.phase2_env.precapture_perception_environment_config()`;观测构建
  `env.observation.build_precapture_estimated_observation`;目标反解 `dynamics.relative.reconstruct_target_state`。
- 控制接口 `experiments/evaluate_hybrid_policy.py --perception --control-source`。
- 测试 `tests/test_precapture_perception.py`;整周扫描见 §3(72 相位/种子静态几何)。
