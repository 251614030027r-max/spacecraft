# 环境重建与复现入口

## Python环境

本机保留的`.venv`使用Python 3.12.6，已验证可正常启动；核心依赖为PyTorch 2.9.1+cu128、Stable-Baselines3 2.7.1和Gymnasium 1.2.2。该环境约4.86 GiB，并通过`pyvenv.cfg`绑定本机Python安装路径，因此只作为当前机器的复核环境，不能视为可移植交付。导师或其他机器应依据`requirements.txt`、`requirements-mpc.txt`和`requirements-dev.txt`重建：

```powershell
Set-Location 'D:\py\DRL2'
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-mpc.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

GPU训练需安装与目标机器CUDA兼容的PyTorch wheel。复制工程时可保留`.venv`作为本机快照，但不能以它替代依赖清单；若启动器路径失效，应重建环境，不要修改工程代码规避。

## 回归

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

2026-08-17在保留的原项目`.venv`中完成最终回归：96项通过。若只验证canonical环境和SAC接口：

```powershell
.\.venv\Scripts\python.exe -B -m train.train --steps 500000 --seed 260815 --run-name validation_only --mode phase1_pretrain --device cpu --eval-steps 50000 100000 200000 300000 400000 500000 --validate-only
```

`--validate-only`会检查run名称未占用，但不会创建模型或日志。

## 正式入口

- 环境：`env.phase2_env.phase2_environment_config("phase1_pretrain" | "full_mission")`；
- SAC配置：`train.configs.PURE_SAC`；
- 训练：`python -B -m train.train`；
- manifest驱动评估：`python -B -m eval.evaluate_policy`；
- MPC：`controllers.mpc.config.constrained_mpc_nominal_config()`和`experiments.evaluate_mpc`。

旧schema模型应使用其原manifest恢复环境；canonical评估会拒绝与当前任务/schema不一致的manifest，这是防止静默重解释的预期行为。
