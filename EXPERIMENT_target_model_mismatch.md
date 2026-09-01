# 探针：目标模型不确定性 — 纯 MPC 抗失配诊断（无学习）

> 单因子 = 目标动力学模型失配幅度（truth 用采样真值,controller 用名义)。S1-v2 主基准不动。
> 本轮只做纯 MPC 诊断,不碰任何 SAC 训练。阶段① smoke 出数即判。

## 动机

当前任务号称"非合作目标",却把目标真值惯量喂给 MPC 做预测——这与"非合作"矛盾,也是让 MPC
无敌的根因之一。真实非合作目标的惯量只能由成像/雷达在线估计、有误差(南航将"未考虑惯量参数
不确定性"列为 future work)。本探针检验:注入现实量级失配后,纯 MPC 是否因"按错模型预测未来"
而失效——若失效且拿真值者仍可行,则 hybrid(SAC 分布鲁棒 + MPC 约束安全)有真实用武之地。

## 实现(接线活,pytest 全绿)

truth/controller 拆分:**env 的真值目标轨迹用采样惯量/质量传播;MPC 的 `reference_model`
用名义参数**。控制器每步读到的是真值当前状态(如同完美观测),但用错惯量向前预测。

| 文件 | 改动 |
|---|---|
| `env/scenarios.py` | `sample_target_parameters(mismatch, seed)`:惯量按主轴特征值 ±mismatch 乘性扰动(特征分解→缩放→重构,保 SPD),质量 ±mismatch;mismatch=0 逐位名义。惯量是有效杠杆(自由轨道目标翻滚由惯量驱动,质量在引力加速度里抵消)。 |
| `env/se3_rendezvous_env.py` | 新增 `phase2_target_model_mismatch`;每 episode 抽失配 seed(在相位 seed 之后,不扰动相位流),真值目标参数=采样值,**失配幅度+seed 进轨迹缓存 key**(必修,否则被复用),`info` 记录失配幅度/seed/惯量迹。 |
| `experiments/evaluate_mpc.py` | `--target-model-mismatch`(0/0.1/0.2/0.3)+ `--controller-model {nominal,truth}`;truth 模式每 episode 用真值参数重建控制器(可行性对照)。 |
| `tests/test_target_model_mismatch.py` | 6 项:采样 SPD/可复现/界内、mismatch=0 名义、缓存按失配 seed key、env 用真值而名义不变、evaluate 接受两种控制器模型。 |

`python -B -m pytest -q` → **135 passed**(129 + 6)。

## 阶段① 结果(mismatch=0.3 即惯量 ±30%,h10 corridor,10 episode,seed 970000)

| 配置 | 完成 | worst margin | 违约 | 推力饱和 |
|---|---|---|---|---|
| 名义(错)模型 · local | **10/10** | +0.0441 | 无 | 0.000 |
| 名义模型 · exact r10 | 10/10 | +0.0442 | 无 | 0.000 |
| 真值模型 · local(可行性对照) | 10/10 | +0.0441 | 无 | 0.000 |

**纯 MPC 用错 30% 的惯量,仍 10/10、0 违约、worst-margin 与真值逐位相同。判据落"杠杆偏弱":
不扩训练。**

## 机制(定量,airtight)

±30% 惯量误差导致的**目标姿态预测误差 ∝ horizon²**,实测:

| 预测跨度 | 姿态预测误差(均值 / 最大) |
|---|---|
| 1 horizon(1 s,h10) | **0.00° / 0.01°** |
| 5 horizons(5 s) | 0.09° / 0.15° |
| 10 horizons(10 s) | 0.37° / 0.61° |

在 h10 的 1 s 窗口里,错 30% 惯量只造成 **0.01° 的姿态预测误差**——走廊/FOV/速度裕度是度/厘米
量级,0.01° 动不了它。这与相位那轮同一结构原因:**短 horizon + 每步真值状态反馈**把模型误差
压到可忽略(一个 horizon 只翻 2.4°,误差 0.01°),且误差不跨步累积。

## 结论与下一步(交上层/用户定)

- **参数失配(惯量 ±30%)在"完美状态观测 + 短 horizon"下不构成缺口**——预测误差按 horizon² 缩,
  一个 horizon 内 ~0.01°,结构性太小。**不进阶段②(仅在阶段①退化时才做),不扩训练。**
- **机制直接指向更狠的杠杆:丢失真值状态反馈**,即测量噪声 / 部分可观 / 状态估计误差。状态误差
  **不按 horizon 缩**——当前步的目标位姿错 1° 立即整体平移走廊/FOV 几何,是结构上更强的不确定性。
  这正是用户预设的 plan B。
- 也可考虑"参数误差 × 拉长 horizon"(误差按 horizon² 放大,10 s 才到 0.37°),但那与"短 horizon
  已够用/实时"的既有结论冲突,且改任务难度量级,须独立单因子 + 上层批。

## 纪律

严格单因子(只动模型失配);S1-v2 冻结;compute 报 p95/max;本轮不做 SAC 训练;代码 zip 交付
本机 commit,不 push。阶段① smoke = 会话内 3 配置 × 10 episode,负结果;阶段② 未跑(判据未触发)。
