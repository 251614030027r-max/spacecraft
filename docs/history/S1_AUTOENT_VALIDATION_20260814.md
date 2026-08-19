# S1 automatic-entropy single-factor validation — 2026-08-14

## 单因素边界

基于已完成但失败的`phase2_mission_s1_fullcanonical_seed260814`，本轮只修改SB3 SAC entropy mechanism：

```text
ent_coef = "auto_0.005"
target_entropy = -6.0
```

canonical full-S1 sampler、training/evaluation config、reward、termination、observation/action scaling、网络、learning rate、buffer、learning starts、batch、tau、gamma、train frequency、gradient steps和target update interval均保持不变。不使用裸`auto`，不实现自定义entropy controller，不增加alpha上下限、schedule或独立alpha learning rate。

## 代码与参数传递

- `PureSACConfig.ent_coef`为`"auto_0.005"`，新增`target_entropy=-6.0`。
- `model_kwargs()`通过dataclass序列化后将二者原样交给SB3 2.7.1 `SAC`。
- manifest的`hyperparameters`会记录这两个值。
- 现有`Phase2DiagnosticsCallback`从`model.log_ent_coef`读取实际alpha(t)，并继续记录actor action std/saturation、critic Q、Gate、minimum Gate distance和failure type；SB3 TensorBoard继续记录critic/actor/ent-coef loss。

## 最小验证证据

1. 参数回归：`model_kwargs()["ent_coef"] == "auto_0.005"`且`target_entropy == -6.0`。
2. full regression：`87 passed in 20.14s`。
3. canonical正式命令`--validate-only`：通过。
4. 5200-step canonical Phase-I CUDA smoke，使用正式`model_kwargs()`、seed 260814和fresh SAC：
   - initial replay transitions：0；final：5200；
   - SB3 target entropy：-6.0；
   - initial alpha：0.004999999888；final alpha：0.004975066520，确认alpha实际更新；
   - actor、critic、Q、log-prob与log-ent-coef检查均finite；
   - sampled deterministic-mean action saturation fraction：0.0；
   - 未写入正式`models/`或`logs/`，不是学习有效性实验。

## 唯一长训练命令

```powershell
python -B -m train.train --steps 250000 --seed 260814 --run-name phase2_mission_s1_fullcanonical_autoent_seed260814 --mode phase1_pretrain --checkpoint-freq 50000 --eval-freq 50000 --eval-episodes 10 --eval-seed 20288000
```

命令不得添加`--actor-init`。actor、critic和replay从零开始；50k/100k/150k/200k/250k使用与fixed-alpha run相同evaluation seeds。除NaN/Inf或训练程序明确数值崩溃外不提前停止、不在途中改参。训练结束后才进行20-seed同种子final evaluation和3–5条代表轨迹，并与fixed-alpha run直接比较Gate、minimum Gate distance、100k后退化、catastrophic speed、actor saturation、critic Q/loss及alpha(t)。
