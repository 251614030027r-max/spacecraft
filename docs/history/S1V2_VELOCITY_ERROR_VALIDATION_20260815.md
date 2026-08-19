# S1-v2 速度误差观测单因素启动验证（2026-08-15）

## 实验问题与对照

上一轮`phase2_mission_s1v2_acquisition_bodyobs_autoent_tau001_seed260815`从零完成500k，Gate始终为0。其400k checkpoint在独立20 seeds上16/20靠近超过1 m、10/20进入1.5 m位置球、16/20最近点FOV不超过45°，但仅3/20最近点速度不超过0.30 m/s，9次catastrophic-speed、10次premature-entry。空间获取已经形成，主要缺口是Gate前主动制动；500k final又退化为平均实际靠近0.181 m。

本轮唯一变量是24D observation的三维速度通道：

```text
position_rate_body -> velocity_tracking_error_body
```

新schema为`phase2_mission_v3_body_velocity_error_24d`，位置误差、姿态、角速度、future-task features、phase flag、维数和normalization全部不变。

## 统一速度reference语义

环境不复制guidance law，而是从同一`Phase2MissionReward`实例调用正式实现：Phase I使用现有`phase1_desired_velocity()`及冻结的0.15 m/s cruise/braking law；Phase II当前正式active pose reference是静止目标，reward对实际相对速度范数计价，因此desired velocity为零。observation统一计算target-frame `actual - desired`，再使用已经验证的`relative.rotation.T`转换为chaser body frame。

旧schema保持原义：`phase2_mission_v2_body_translation_24d`仍输出body-frame实际位置率，`phase2_mission_v1_24d`仍输出target-frame实际位置率。新manifest schema version 5明确记录：

```text
translational_position_frame = chaser_body
translational_velocity_observation = tracking_error
force_action_frame = chaser_body
observation_schema = phase2_mission_v3_body_velocity_error_24d
```

## 冻结项

S1-v2 sampler、1.5 m/0.30 m/s/45° Gate、Phase-II、reward、termination、desired-velocity参数、动作与尺度均未修改。SAC继续为learning rate 1e-4、gamma 0.997、tau 0.001、buffer 300k、batch 256、learning starts 5k、train frequency 4、gradient steps 1、`auto_0.005`、target entropy -6、4x256 ReLU；curriculum关闭，actor/critic/replay全部fresh，不加载上一轮400k。

## 启动验证

- 实际速度等于Phase-I desired velocity时，三维tracking error为零；
- 90°已知相对姿态下，target到chaser body使用`relative.rotation.T`，方向回归通过；
- 沿Gate方向超速时，body-frame error符号明确指示反向制动力；
- Phase-II active desired velocity为当前正式静止reference的零速度；
- 旧v2 schema能够构造环境且没有被解释为tracking error；
- 受影响回归及全量回归结果：95 passed，重复run保护另行直接验证通过；
- canonical CUDA `--validate-only`通过；fresh CUDA 5200-step smoke中observation保持24D且finite，replay从0到5200，alpha从0.00500000更新到0.00497505，无NaN/Inf；未生成正式run目录。

## 唯一长训练命令

在PowerShell中：

```powershell
$env:PYTHONPATH='D:\py\DRL2\.venv\Lib\site-packages'
```

随后在`D:\py\DRL2`执行：

```powershell
& 'C:\Users\35884\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -B -m train.train --steps 500000 --seed 260815 --run-name phase2_mission_s1v2_velerror_bodyobs_autoent_tau001_seed260815 --mode phase1_pretrain --device cuda --checkpoint-freq 50000 --eval-steps 50000 100000 200000 300000 400000 500000 --eval-episodes 10 --eval-seed 20288000
```

训练中配置冻结。评价重点是位置球进入率、进入时/最近点速度不超过0.30 m/s的比例、Gate success及FOV，并直接对照上一轮400k的10/20位置球、3/20低速、Gate 0/20。若早中期出现制动/Gate而后期遗忘，应保留最佳deterministic checkpoint并分别裁决representation有效性与后期稳定性。
