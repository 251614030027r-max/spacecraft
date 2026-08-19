# S1 bounded local-progress single-factor validation — 2026-08-14

## 单因素定义

上一轮automatic entropy显著抑制critic/Q膨胀和动作饱和，但canonical S1仍Gate 0。本轮只测试Phase-I局部信用分配：

```text
before: progress = k_p * (C_t - gamma * C_t+1)
after:  progress = k_p * (C_t - C_t+1)
```

`C(s)`仍为当前bounded Gate-directed cost；position、attitude、Gate-directed velocity error、angular-rate组成、normalization、权重和`progress_weight`全部未变。Phase-II reward路径未变。没有persistent state penalty、distance/time penalty或额外reward项。automatic entropy继续为`ent_coef="auto_0.005"`、`target_entropy=-6.0`，canonical sampler、termination、网络和其余SAC参数全部冻结。

与上一轮审查ZIP中的46个核心源码文件逐文件SHA-256比较，只有`env/reward.py`和对应`tests/test_reward.py`变化；测试文件只将旧“静止因discount获得正progress”断言替换为local-progress符号和有界性断言。

## 最低成本检查

### A. Reward单元检查

固定初态与Gate构造closer、stationary和farther next state：

```text
progress_close > 0
progress_stationary = 0 within 1e-12
progress_farther < 0
|progress| <= progress_weight * 2.1
```

`tests/test_reward.py`共12项通过。

### B. 原reward-preference sanity

复用`eval.validate_phase1_semantics`、seeds 262000–262002及Gate-PD/zero/high-speed三种rollout：Gate-PD Gate 3/3，maximum speed 0.179317 m/s；三seed discounted returns分别为：

```text
seed 262000: Gate-PD 2.3745, zero -12.5281, high-speed -13.8972
seed 262001: Gate-PD 1.6887, zero -11.3041, high-speed -14.1387
seed 262002: Gate-PD 2.5840, zero  -8.7664, high-speed -13.8396
```

`Gate-PD - high-speed`最小余量15.8273，排序通过；zero与high-speed均未获取Gate，没有发现静止或高速shortcut。验证JSON保存于工作区`phase2_s1_localprogress_reward_sanity_3_seed262000.json`，待训练结束纳入审查包。

## 工程回归

- full regression：`87 passed in 18.11s`。
- 正式命令`--validate-only`：通过。
- 正式run的model/log目录尚不存在，命令不含`--actor-init`；长训练必须actor/critic fresh、replay empty。
- 遵守“仅两个最低成本reward检查”边界，未增加seed矩阵、reward sweep或学习性短训练。

## 唯一长训练命令

```powershell
python -B -m train.train --steps 250000 --seed 260814 --run-name phase2_mission_s1_fullcanonical_localprogress_autoent_seed260814 --mode phase1_pretrain --checkpoint-freq 50000 --eval-freq 50000 --eval-episodes 10 --eval-seed 20288000
```

训练期间参数冻结；除NaN/Inf或程序明确数值故障外不提前停止。训练结束后用相同20个独立seeds和约5条代表轨迹与上一轮auto-ent直接比较，不自行续训或修改其他变量。
