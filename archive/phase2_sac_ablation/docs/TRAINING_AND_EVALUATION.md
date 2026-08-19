# 训练、评估与下一轮执行约束

解释器固定为 `D:\py\DRL2\.venv\Scripts\python.exe`。当前恢复配置已经冻结：奖励、观测、动作、终止和 SAC 超参数保持不变；自动熵保留 `auto_0.01`，课程改为基于近期掌握度推进、保持或回退，并保留自动早停保护。已失败的 `run_final_pipeline.py` 和 `run_sac_recovery.py` 已删除。

全套测试：

```powershell
Set-Location D:\py\DRL2
& ".\.venv\Scripts\python.exe" -B -m pytest -p no:cacheprovider -q
```

活动训练入口仍是 `python -m train.train`，`--steps` 现在必须显式给出，不再提供可能早于课程结束的默认步数。下一次长训练必须满足：新 run name、随机网络、空回放池、配置写入 manifest、提前停止门槛已声明、候选筛选种子与正式 100 回合种子隔离。不得使用已删除失败模型或正式评估结果反向挑选检查点。`completed=1` 表示连续 5 步联合达标，`distance_failure=1` 和 `time_failure=1` 分别表示越界与超时。

唯一固定简单任务判别入口为 `--fixed-simple-diagnostic`：它冻结 1–3 m、30°、±0.05 m/s 和翻滚尺度 0.25，禁止超过 200k 步，默认每 50k 用固定 10 个候选种子评估；连续两次检查均为零完成且距离失效率大于 50% 时自动停止。该模式只用于区分固定任务训练退化与课程分布问题，其 manifest 状态不会被正式 SAC 验收入口接受。

2026-08-11 判别实验 `sac_fixed_simple_diagnostic_s20260823` 已完成 200k。固定候选在 50k/100k/150k/200k 的完成率均为 0%，但距离失效率依次为 90%/20%/10%/0%，均值回报依次为 -56.34/-32.10/-12.56/-16.63；训练 monitor 后半程完成率约 30%–35.5%，没有持续退化。由此下一轮只把课程固定简单阶段延长到 300k，1.3M 达到完整任务、1.4M 结束回放、1.5M 结束；其他物理语义、奖励和 SAC 参数不变。

延迟课程运行 `sac_final_delayed_curriculum_s20260824` 在 977,908 步手动中断：300–400k 完成率 80.6%，最后成功在 468,640 步，500k 后持续零成功且距离失效率长期超过 50%。细化时序显示 450k 前自动熵仍约 0.0021–0.0023，完成率已经开始下降；随后 Critic/actor 与自动熵共同恶化，因此自动熵上升是策略低熵和 Q 失稳的响应，不能认定为首因。下一轮恢复 `auto_0.01`，只把开环时间课程改成掌握度课程：300k 后每 25k 统计训练结果，完成率至少 70% 且距离失效率不高于 20% 时将难度上限推进 0.025；完成率低于 50% 或距离失效率高于 30% 时回退 0.025；其他情况保持。每个难度上限下 80% 回合集中采样其顶部 20% 前沿，20% 回放 `0…ceiling`，确保门控主要衡量当前前沿而不是旧简单任务。连续三个窗口零完成且距离失效率大于 50%，或连续八个窗口未刷新最高难度时，保存模型并提前结束，因此无需人工频繁监视。

最终长训练由用户手动启动，建议新名称和命令为：

```powershell
Set-Location D:\py\DRL2
& ".\.venv\Scripts\python.exe" -B -m train.train --steps 1500000 --seed 20260825 --device cuda --run-name sac_final_mastery_curriculum_s20260825 --checkpoint-freq 50000
```

MPC-only 已通过正式验收，复核入口和结果分别为：

```powershell
& ".\.venv\Scripts\python.exe" -B -m experiments.run_mpc_recovery
```

- 实现与单项评估：`experiments/evaluate_mpc.py`
- 正式结果：`logs/mpc_recovery_pipeline_exact_v1/official_100.json`
- 验收摘要：`logs/mpc_recovery_pipeline_exact_v1/acceptance.json`

不要为了重现“通过”而覆盖上述文件；如需复核，使用新的输出目录和种子。SAC正式验收仍要求独立、确定性100回合，并按 `BASELINE_SPEC.md` 的全部门槛判定。
