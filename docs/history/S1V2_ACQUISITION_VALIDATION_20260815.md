# S1-v2 Acquisition 启动验证（2026-08-15）

## 结论

S1-v2已作为新的唯一canonical Phase-I任务落地，旧S1仍可按`legacy_regulation`恢复但不再由新训练入口调用。改动没有触碰truth、Phase-II约束、body-frame 24D observation、物理action、reward结构或SAC算法；500k正式训练尚未启动。

## 冻结配置

- sampler：target center 10–14 m，方向半角15°，姿态误差不超过20°，初始总相对速度不超过0.05 m/s，相对角速度逐分量不超过0.01 rad/s，target tumble scale 0.20；
- Gate：`[-8,0,0] m`，位置误差不超过1.5 m、target-frame总相对速度不超过0.30 m/s、FOV angle不超过45°；姿态和角速度仅作观测、shaping与报告，不是Phase-I硬Gate；
- reward：保留bounded discounted-consistent progress与原权重，只把Phase-I直接巡航参考改为0.15 m/s；soft overspeed 0.25 m/s与catastrophic guard 1.0 m/s不变；
- Phase II：35° corridor、50° FOV、terminal velocity constraints及0.25 m/10°/0.05 m/s/0.02 rad/s/1 s final条件逐项未变；
- SAC：fresh actor/critic/replay，learning rate 1e-4、gamma 0.997、tau 0.001、buffer 300k、batch 256、train_freq 4、gradient_steps 1、`auto_0.005`、target entropy -6、4x256 ReLU。

## 代码与诊断

`env.phase2_env.phase2_s1v2_mission_config()`集中生成新任务，避免散落近似参数。`gate_semantics`显式区分`legacy_regulation`与`acquisition_v2`；旧manifest缺少该字段时按legacy恢复。evaluation schema v4为Gate entry增加FOV，并增加`closest_gate_approach`，包含最近点时间、位置误差、速度、FOV、姿态误差和角速度。训练入口支持显式递增的`--eval-steps`，本轮只在50k/100k/200k/300k/400k/500k评估。

## 低成本验证证据

- 500个固定seed reset逐个验证新距离、方向、姿态、速度及角速度包络；24D observation和一步truth transition全部finite；
- canonical Phase-II数值检查确认35°/50°及0.25 m/10°/0.05 m/s/0.02 rad/s/1 s未变；
- 环境Gate切换、终止、任务、sampler和config定向用例通过；pytest唯一ERROR来自受限Windows临时目录的`tmp_path`夹具，单独复现发生在测试体进入前，不是工程断言失败；
- canonical `--validate-only`在CUDA配置下通过；
- fresh CUDA 5200-step SAC smoke通过：replay从0增长到5200，alpha从0.00500000更新到0.00497505，policy参数、observation与reward均finite，未产生模型或日志目录。

## 唯一正式训练命令

先在PowerShell设置解释器依赖：

```powershell
$env:PYTHONPATH='D:\py\DRL2\.venv\Lib\site-packages'
```

再在`D:\py\DRL2`执行：

```powershell
& 'C:\Users\35884\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -B -m train.train --steps 500000 --seed 260815 --run-name phase2_mission_s1v2_acquisition_bodyobs_autoent_tau001_seed260815 --mode phase1_pretrain --device cuda --checkpoint-freq 50000 --eval-steps 50000 100000 200000 300000 400000 500000 --eval-episodes 10 --eval-seed 20288000
```

200k为第一次正式判断点；Gate=0但已有明显靠近时继续，位置已有获取但速度/FOV未满足时继续，只有NaN/Inf或明确数值崩溃可提前停止。500k后用独立20 seeds做final deterministic evaluation，再按S1-v2标准决定是否进入S2。
