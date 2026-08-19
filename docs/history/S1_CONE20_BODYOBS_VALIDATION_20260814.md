# S1 cone20 body-frame translational observation 单因素验证 — 2026-08-14

## 实验边界与动机

上一轮`phase2_mission_s1_cone20_autoent_seed260814`完成250k但Gate仍为0。10个固定seeds在250k有8个minimum Gate distance与reset初值相差不足1 mm；独立20-seed final为0/20，minimum Gate mean 7.427 m，平均只比初值靠近0.112 m。100k曾短暂改善，随后伴随alpha、critic Q和actor loss增长而遗忘。由此停止继续收窄sampler，本轮只检验target-frame平动状态与body-frame force action的坐标错位。

唯一语义修改是canonical 24D mission observation的两个三维通道：

```text
position error: target frame -> chaser body frame
position rate:  target frame -> chaser body frame
```

物理action仍为`[body torque; body force]`，姿态、角速度、target omega、corridor/FOV/margin/phase channels、24D维数及全部normalization尺度不变。任务保持cone20、12–18 m、target tumble 0.20和Gate `[-8,0,0] m`；reward/termination及Pure SAC参数全部冻结。

## 坐标方向核验

`dynamics.relative.relative_state()`构造：

```text
relative.rotation = R_target^T @ R_chaser
```

因此它把chaser-body向量映射到target frame，target→chaser-body必须使用其转置：

```text
v_chaser_body = relative.rotation.T @ v_target
```

这与工程现有FOV路径`relative.rotation.T @ line_of_sight_target`一致。新增回归覆盖已知绕z轴90°的相对姿态及姿态一致两种情形，避免依赖变量名猜测方向。

新canonical schema为`phase2_mission_v2_body_translation_24d`。旧`phase2_mission_v1_24d`保留target-frame只读兼容路径；评估入口同时核对schema，禁止将旧模型按新observation语义静默评估。

## 必要验证

- 已知旋转、姿态一致、task/env/config定向回归：39 passed。
- 全量回归：89 passed in 20.38s。
- 5200-step fresh CUDA smoke：completed，actual 5200，actor init null、critic fresh、replay 0。
- observation shape `(24,)`且finite；training/evaluation config完全相同。
- manifest：`translational_observation_frame=chaser_body`、`force_action_frame=chaser_body`、schema正确、cone=20°、`auto_0.005`、target entropy=-6。
- alpha：0.005→0.004975；actor/critic参数、sample action/log-prob/Q均finite，sample action saturation为0。

smoke临时models/logs已清理，证据位于工作区：

- `phase2_s1_cone20_bodyobs_autoent_smoke_manifest_5200_seed260814.json`
- `phase2_s1_cone20_bodyobs_autoent_smoke_validation_5200_seed260814.json`

## 唯一长训练命令

在`D:\py\DRL2`执行：

```powershell
python -B -m train.train --steps 250000 --seed 260814 --run-name phase2_mission_s1_cone20_bodyobs_autoent_seed260814 --mode phase1_pretrain --checkpoint-freq 50000 --eval-freq 50000 --eval-episodes 10 --eval-seed 20288000 --device cuda
```

不得包含`--actor-init`，不得中途修改参数。除NaN/Inf或明确程序数值崩溃外完整运行250k。结束后使用相同独立20 seeds，并重点比较50k/100k与cone20 baseline的实际靠近量、lateral error、最近点速度/姿态/角速度、catastrophic-speed、alpha、Q、critic loss和动作饱和。
