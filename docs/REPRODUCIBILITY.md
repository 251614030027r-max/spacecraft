# 环境与只读复现入口

## 当前解释器事实

仓库 `.venv` 的启动器仍绑定已经不存在的本机 Python 3.12.6，直接运行 `.venv\Scripts\python.exe` 会报“Unable to create process”。这不是代码错误，也不应通过修改工程来规避。

当前已验证的只读回归使用 Codex 随附 Python 3.12.13，并挂载原虚拟环境依赖目录：

```powershell
$runtimePython = 'C:\Users\35884\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$env:PYTHONPATH = 'D:\py\DRL2\.venv\Lib\site-packages'
& $runtimePython -B -m pytest -q
```

2026-09-01 的现场结果为 `146 passed`。旧交接中的 `147 passed` 与当前仓库不一致；姿态偏置不旋转惯性速度的实现存在，但测试尚未单独断言该不变量。

## PyCharm 环境重建

不要覆盖旧 `.venv`。如需恢复 PyCharm 可直接选择的解释器，应建立并验证一个新环境，再由用户决定是否切换：

```powershell
Set-Location 'D:\py\DRL2'
py -3.12 -m venv .venv-new
& '.\.venv-new\Scripts\python.exe' -m pip install --upgrade pip
& '.\.venv-new\Scripts\python.exe' -m pip install --upgrade cffi
& '.\.venv-new\Scripts\python.exe' -m pip install -r requirements-dev.txt
& '.\.venv-new\Scripts\python.exe' -B -m pytest -q
```

必须先升级 `cffi`，再安装 `requirements-dev.txt`。验证完成前保留旧环境，不改 PyCharm 项目解释器。

## 入口

- 环境：`env.phase2_env.phase2_environment_config("single_phase")`；
- 任务：`env.phase2_env.phase2_s1v2_mission_config()`；
- SAC：`train.configs.PURE_SAC`；
- 评价：`python -B -m eval.evaluate_policy`；
- 单相位语义：`python -B -m eval.validate_single_phase_semantics`；
- Pure MPC：`python -B -m experiments.evaluate_mpc --task single_phase`；
- 主表：`python -B -m eval.main_table`；
- 运行摘要：`python -B -m eval.digest_run --run logs/<run>`。

训练目前暂停。以上入口仅用于代码阅读、回归和经授权的诊断。
