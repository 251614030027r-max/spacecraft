# 当前环境与复现入口（2026-09-19）

## 已验证解释器

本机回归使用系统 Python 3.12，并复用仓库 `.venv` 的依赖：

```powershell
$env:PYTHONPATH = 'D:\py\DRL2\.venv\Lib\site-packages'
& 'C:\Users\35884\AppData\Local\Programs\Python\Python312\python.exe' -B -m pytest -q
```

训练可从已配置好的 PyCharm 终端直接运行 `python -B -m ...`。不要覆盖旧 `.venv`，也不要因启动器路径问题重建整个工程环境；先确认 `python -c "import numpy, scipy, cvxpy, gymnasium, stable_baselines3"`。

## 当前唯一训练入口

正式命令见 `docs/ADAPTIVE_MAINLINE_RUNSHEET.md` 第 2b 节。必须满足：

- commit 与远端一致，且包含奖励单位修复；
- 三个新目录 `logs/adp_rf_262410/411/412` 均不存在；
- 从零初始化，不加载旧 checkpoint/replay buffer；
- h35、`arrival_condition`、`--baseline-anchored-residual --adaptive-task`；
- 三个种子独立进程运行，10k/30k/60k 按 runsheet 检查；
- 旧 `adp_262410/411/412` 的模型与 critic 已失效，不得恢复。

训练会在每个 run 目录生成 manifest、Monitor CSV、5k checkpoint、TensorBoard 和最终模型。manifest 记录完整环境、hybrid/MPC 配置、观测与动作维度、命令和状态；正式评价必须读取 manifest 中的 `evaluation_flags`，不得凭名称猜配置。

## 当前代码入口

- 自适应任务：`env.phase2_env.precapture_adaptive_capture_environment_config()`；
- 耦合环境：`env.hybrid_env.PrecaptureHybridEnv`；
- 训练：`python -B -m train.train_hybrid`；
- 统一评价：`python -B -m experiments.evaluate_hybrid_policy`；
- 校准汇总：`python read_cal.py`（默认读取 `eval/cal2/`）；
- 当前回归：`python -B -m pytest -q`。

安全与完成始终由 truth RK45 几何裁决。日志、模型和旧说明不能单独覆盖当前 commit、runsheet 和 manifest。
