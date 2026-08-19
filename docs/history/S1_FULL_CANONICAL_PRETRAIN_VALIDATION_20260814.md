# S1 full-canonical pretrain validation — 2026-08-14

## 上层指令落实

本轮只撤销adaptive curriculum主训练路径，保留当前reward/termination、truth、Gate、observation和Pure SAC参数。`training_environment_config("phase1_pretrain")`现在直接使用当前canonical factory，与evaluation config逐字段相同：

```text
phase1_curriculum_enabled = false
phase1_curriculum_adaptive = false
```

旧adaptive代码和历史回归保留，但新canonical S1命令不调用。训练必须actor fresh、critic fresh、empty replay，不允许`--actor-init`、旧checkpoint、demonstration或PD数据。

## 最小diagnostics修正

- `Phase2DiagnosticsCallback`直接接收实际`training_config`，不再自行选择另一个probe环境。
- episode记录新增minimum Gate distance、terminal Gate/姿态/速度/角速度与phase1-speed、premature-entry、terminal-constraint、distance、time failure标志。
- monitor增加相同核心failure字段。
- 原`short_soft_mc_return`改名为`short_soft_bootstrapped_return`，父字段为`short_horizon_bootstrapped_value_probe`；JSON显式写入`is_independent_monte_carlo_ground_truth=false`。它仅是64步soft rollout加末端critic bootstrap，不作为真实MC标定。

## 验证

1. full regression：87 passed。
2. canonical validate-only：通过。
3. 训练/评估config：dataclass完全相等，curriculum/adaptive均false。
4. 30 fixed-seed initial-state truth：center distance实测12.384–17.647 m，均在canonical 12–18 m；direction half-angle 0.959931 rad（55°），attitude limit 0.785398 rad（45°），initial speed limit 0.2 m/s，target tumble scale 0.20；所有episode metadata为difficulty=1/full sample。
5. 1k fresh SAC chain：manifest completed；actor initialization null、critic fresh、initial replay 0；training/evaluation config相同；monitor failure列齐全；diagnostics schema 4记录minimum Gate distance；2-episode periodic canonical evaluation成功生成。
6. runtime probe：使用同一training config；Q/std与actor/std可输出；bootstrapped estimate标签和非MC声明正确。

上述只是接口与低成本链路验证，不是学习成功。未进行difficulty sweep、sampler对照、reward sweep或长训练。

## 唯一长训练命令

```powershell
python -B -m train.train --steps 250000 --seed 260814 --run-name phase2_mission_s1_fullcanonical_seed260814 --mode phase1_pretrain --checkpoint-freq 50000 --eval-freq 50000 --eval-episodes 10 --eval-seed 20288000
```

50k/100k/150k/200k/250k使用相同canonical fixed seeds。参数全程冻结；除NaN/Inf或Q失控与长期actor全饱和同时出现外不提前终止。训练结束报告只提交Gate/minimum-distance趋势、failure distribution、actor/Q是否与行为退化同步、20–30 seed final evaluation和3–5条代表轨迹时序。

