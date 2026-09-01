# 探针4 对照：余量控制 + EKF 输出反馈基线（无训练，定论文立论）

> 承接探针4(部分可观=首个真实缺口)。投训练前的两个对照,判"缺口是根本难点还是可被经典手段
> 补掉"。严格单因子(只动余量系数 / 只动滤波,分开跑),违约按真实几何判,S1-v2 不动,不训练。

## 加了什么(可控旋钮)

| 旋钮 | 位置 | 作用 |
|---|---|---|
| `corridor_speed_fraction` | `MPCConfig` + `--corridor-speed-fraction` | MPC 共旋参考瞄准 total_speed 限的比例(0.6→0.5→0.4),只改 MPC 自己的参考,不动任务/reward——余量对照 |
| EKF 滤波 | `TargetStateEstimator(filter_mode="propagate")` + `--target-estimator ekf` | 用名义目标模型把"延迟/保持的量测"前向传播到当前时刻(EKF predict 步),去滞后 |

10 项测试,`pytest` 145 passed。

## 会话内预览结论(各 6 episode,h10 local,seed 970000;确认 run 见下)

### 对照A:放宽速度余量能否救回?(1° 偏置)

| 余量系数 | 完成 | 违约 |
|---|---|---|
| 0.6 / 0.5 / 0.4 / 0.3 @ 1° 偏置 | **0/6(全档)** | 6/6(全档) |
| 0.6 obs off(对照) | 6/6 | 0/6(用时 86.5 s) |
| 0.4 obs off(对照) | 6/6 | 0/6(用时 91.3 s,更慢但完成) |

**结论:放宽余量救不回 1° 偏置**(放到 0.3× 仍 0/6),而余量旋钮本身有效(obs off 在 0.4× 仍
6/6、只是更慢)。**缺口不是"贴边调参",是根本的。** 机制:偏置经 Adjoint 与 ω 耦合,污染的是
控制器**感知到的共旋速度**本身——飞得再保守也没用,因为它按错的旋转系估自己的速度。

### 对照B:EKF 输出反馈能否补回?(predict 步去滞后)

| 误差类型 | raw | EKF(predict) |
|---|---|---|
| 仅时延(延迟3+更新3,无偏置) | 0/6 违约 | **6/6,0 违约,+0.0441** |
| 仅偏置 1° | 0/6 违约 | **0/6 违约(不变)** |
| 现实档(1°+延迟3+更新3) | 0/6 | **0/6(不变)** |

**结论:EKF predict 步完全补回"时延"类误差(6/6,回到名义),但对常值姿态偏置无能为力(0/6 不变)。**
——正是预期的"缓解但补不满",且边界清晰:时延可滤,常值偏置不可滤(predict-only 无偏置状态)。

## 立论(两对照共同指向)

**残留缺口 = 持续的目标姿态估计偏置**,它同时**免疫于保守余量**(对照A)**和 EKF predict 步**
(对照B)。在 Λ>1 body-fixed 共旋下,该偏置经 ω 耦合直接转成 total_speed 违约。这是 hybrid 的
精确立足点:**学习跨"估计偏置分布"的谨慎共旋 + MPC 硬约束兜底**,做经典滤波+固定余量做不到的
"对不可滤估计偏置鲁棒"。

## 投训练前必答的最后一个审稿人问题(诚实关口)

本 EKF 基线是 **predict-only(无量测更新、无偏置状态)**。一个**偏置增广 EKF** 理论上能估掉
偏置——**当且仅当偏置可观**。而非合作位姿估计的**常值传感器偏置在无独立参照下通常不可观**
(无法区分"目标在 X"与"目标在 R·X 且我的传感器偏了 R")。所以:
- 若确认偏置不可观(常值传感器偏置)→ 任何 EKF 都补不掉 → hybrid 立论最强;
- 确认 run 建议补一档 **偏置增广 EKF**,实测它是否真的补不掉(堵死"你没试更强的滤波器"这一问)。
这决定 hybrid 的卖点表述,须在投最贵的训练前想清。

## 本机确认 run(挂机,无训练,PowerShell)

seed 块 {970000, 972000, 974000};误差档:0 / 小(0.2°,延迟1) / 中(0.5°,延迟2,更新2) /
大(1°,延迟3,更新3)。

```powershell
$seeds = 970000,972000,974000
# 误差档定义:label -> 参数串
$errs = @{
  "e0"   = "";
  "small"= "--obs-bias-deg 0.2 --obs-delay-steps 1";
  "mid"  = "--obs-bias-deg 0.5 --obs-delay-steps 2 --obs-update-every 2";
  "big"  = "--obs-bias-deg 1.0 --obs-delay-steps 3 --obs-update-every 3";
}
# 1) 误差扫描 × 余量对照(estimator raw):threshold 曲线 + 余量是否救回
foreach ($s in $seeds) { foreach ($e in $errs.Keys) { foreach ($f in 0.6,0.5,0.4) {
  $args = $errs[$e]
  python -B -m experiments.evaluate_mpc --task single_phase --episodes 20 --seed $s --horizon 10 `
    --corridor-speed-fraction $f $args.Split(" ") `
    --output "logs/obs_${e}_f${f}_raw_s$s.json"
}}}
# 2) EKF 对照(余量固定 0.6):raw vs ekf,各误差档
foreach ($s in $seeds) { foreach ($e in $errs.Keys) { foreach ($est in "raw","ekf") {
  $args = $errs[$e]
  python -B -m experiments.evaluate_mpc --task single_phase --episodes 20 --seed $s --horizon 10 `
    --target-estimator $est $args.Split(" ") `
    --output "logs/obs_${e}_${est}_s$s.json"
}}}
```
（PowerShell 里空参数串 `$args.Split(" ")` 传空数组即可；若报错就把 e0 档单列不带误差 flag。）

拼表看每档:**完成率 + 违约率 + worst-margin(真实几何) + 首次越界时刻 + compute p95/max**,
按 seed 块报分布。关注两条曲线:① 误差→失效阈值;② 各误差档下"raw vs 各余量 vs EKF"谁能回到零违约。

## 训练闸门

只有确认:①放宽余量在误差分布上仍救不回(对照A 已强烈预示);②EKF(含偏置增广那档)补不满偏置;
才投 **SAC 跨误差分布训练 + MPC 兜约束**。届时四方对比:纯MPC / MPC+EKF / 纯SAC / hybrid。

## 纪律

严格单因子(余量、滤波分开跑);违约按真实几何判;compute 报 p95/max;本轮不训练;
代码 zip 交付本机 commit、会话不 push。
