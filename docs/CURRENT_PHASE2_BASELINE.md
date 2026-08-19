# 当前 Phase-2 基线与结论

## 当前任务

当前canonical任务是S1-v2 acquisition。Phase I从target-center 10–14 m、15°方向锥、20°姿态误差、0.05 m/s总相对速度和0.01 rad/s逐分量相对角速度包络出发，target tumble scale为0.20；目标Gate为target-frame `[-8,0,0] m`，成功要求位置误差不超过1.5 m、总相对速度不超过0.30 m/s、FOV不超过45°，姿态与角速度只参与观测、shaping和报告。Phase II仍使用35° corridor、50° FOV和正式速度约束，到达`[-3,0,0] m`后满足0.25 m、10°、0.05 m/s、0.02 rad/s并保持1 s。

truth固定为高保真六自由度SE(3)、J2和RK45；物理action是chaser-body torque/force。curriculum关闭，训练和评估使用相同固定全分布。

## 当前观测与算法

当前源码schema为`phase2_mission_v3_body_velocity_error_24d`。position error在chaser body frame；三维速度通道为`actual velocity - active desired velocity`后再转换至chaser body frame。Phase-I desired velocity直接复用reward正式0.15 m/s cruise/braking field；Phase-II active pose reference静止，对应desired velocity为零。旧v2仍表示body-frame实际速度，旧v1仍表示target-frame实际速度，历史模型不会被静默重解释。

Pure SAC固定为learning rate 1e-4、gamma 0.997、tau 0.001、buffer 300k、batch 256、learning starts 5k、train frequency 4、gradient steps 1、`auto_0.005`、target entropy -6、4x256 ReLU；没有PER/HER、demo、BC、MPC监督、Q clipping或rollback。

## 两个关键S1-v2实验

### v2实际速度观测：空间接近强、不会制动

`phase2_mission_s1v2_acquisition_bodyobs_autoent_tau001_seed260815`完成500k。最佳400k checkpoint在独立20 seeds上平均实际靠近2.365 m，16/20靠近超过1 m、10/20进入1.5 m位置球、16/20 FOV合格，但仅3/20最近点速度不超过0.30 m/s，Gate 0/20；500k final退化为平均靠近0.181 m。结论：空间方向可学，Gate前主动制动缺失且后期遗忘。

### v3速度误差观测：制动改善、位置获取变弱

`phase2_mission_s1v2_velerror_bodyobs_autoent_tau001_seed260815`完成500k。200k固定评估首次出现Gate 1/10，成功轨迹入场为1.490 m、0.254 m/s、FOV 8.1°、姿态9.0°、角速度0.006 rad/s。200k独立20 seeds仍Gate 0/20，但19/20最近点速度合格、20/20 FOV合格、12/20靠近超过1 m；没有轨迹进入位置球。500k final平均靠近仅0.262 m。训练随机Gate由v2的14次增加到33次，但400k–500k窗口降为0。

结论：显式速度误差确实让低速接近可学，却削弱了足够深的位置推进，也没有解决到达后的稳定保持；前200k出现局部有效结构，随后再次遗忘。

## 当前裁决

目前没有合格S1模型，禁止进入S2。保留v2 400k作为“空间接近最佳”证据，保留v3 200k作为“速度控制最佳”证据；两轮final仅用于证明遗忘。下一研究问题是如何在不降低Gate、不缩小sampler的前提下联合形成位置推进、主动制动和到达后保持，并把前期学习能力与后期actor-critic遗忘分开处理。当前不应继续做无方向的observation、tau、entropy或sampler sweep。
