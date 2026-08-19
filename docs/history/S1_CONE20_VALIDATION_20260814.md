# S1 initial-direction cone20 single-factor validation — 2026-08-14

## 单因素边界

本轮从`full-canonical + automatic entropy`稳定基线建立。已否决的local progress完全退出：`env/reward.py`和`tests/test_reward.py`与上一版auto-ent审查快照逐字节相同，Phase-I恢复`k_p(C_t-gamma*C_t+1)`。唯一任务变量为：

```text
initial_direction_half_angle: 55 deg -> 20 deg
```

初始距离12–18 m、速度0.2 m/s上限、姿态45°、相对角速度分量0.02 rad/s、target tumble 0.20、Gate、success、termination及所有reward权重不变。SAC仍为`auto_0.005`、target entropy -6.0及原全部参数；curriculum/adaptive false。

## Reset sampler与配置验证

现有sampler/config测试扩充一个20°canonical断言，未新增测试树。固定seeds 268000–268099进行100次真实environment reset：

```text
configured half angle: 20.0 deg
observed angle range: 2.104–19.735 deg
cone violations: 0
distance range: 12.006–17.986 m
maximum speed: 0.199903 m/s
maximum attitude: 44.465 deg
maximum omega component: 0.019893 rad/s
```

Training/evaluation config完全相同；automatic entropy参数正确。相关测试27 passed；full regression 87 passed in 18.17s。

## Fresh CUDA smoke

正式训练入口完成5200-step fresh CUDA smoke，跨过learning_starts：manifest completed，actual 5200，actor initialization null、critic fresh、initial replay 0；manifest方向锥20°，training/evaluation config相同，`ent_coef="auto_0.005"`、target entropy -6.0。保存模型检查alpha从0.005更新到0.004975；actor、critic、alpha、sample action、log-prob和Q均finite，sample action saturation 0。smoke临时models/logs验证后已清理，证据副本位于工作区：

- `phase2_s1_cone20_autoent_smoke_manifest_5200_seed260814.json`
- `phase2_s1_cone20_autoent_smoke_validation_5200_seed260814.json`

## 唯一长训练命令

```powershell
python -B -m train.train --steps 250000 --seed 260814 --run-name phase2_mission_s1_cone20_autoent_seed260814 --mode phase1_pretrain --checkpoint-freq 50000 --eval-freq 50000 --eval-episodes 10 --eval-seed 20288000
```

命令不得包含`--actor-init`。训练期间参数冻结，除NaN/Inf或程序异常外不提前终止。结束后使用相同20个独立seeds和约5条代表轨迹与55°auto-ent基线比较，不自行继续缩小cone或修改其他变量。
