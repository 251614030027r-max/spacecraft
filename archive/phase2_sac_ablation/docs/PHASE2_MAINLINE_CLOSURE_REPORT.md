# Phase-2 主干收口报告（2026-08-12）

## 结果

当前代码已明确分为五层：`dynamics/` 与 `env/se3_rendezvous_env.py` 是高保真 truth/step 内核；`env/phase2_env.py`、`task.py`、`observation.py`、`reward.py` 是唯一 Phase-2 任务路径；`train/` 是标准 Pure SAC；`controllers/mpc/` 与 `experiments/evaluate_mpc.py` 是 Pure MPC；`archive/phase2_sac_ablation/` 和历史 docs/logs/models 是证据层。未来 SAC-MPC 尚无活动实现。

## 唯一入口与默认加载

- 环境：`env.phase2_env.make_phase2_env` / `phase2_environment_config`。
- Pure SAC：`python -m train.train`；`train/configs.py::PURE_SAC`；默认 `two_stage`。
- Pure SAC evaluation：`python -m eval.evaluate_policy`；从 run manifest 恢复并核验任务，明确 Stage-A 或 nominal，deterministic 默认开启。
- constrained MPC nominal：`controllers.mpc.constrained_mpc_nominal_config`；`python -m experiments.evaluate_mpc`。
- 训练默认 callback：checkpoint、长期 Phase-2 actor/Q/entropy diagnostics、固定种子 deterministic evaluation。没有 BC、demonstration、mastery/rollback、early-stop 或 reward schedule。

## 归档与清理

旧 `train.py`、callbacks、configs、actor-only BC、V2.1 observation/probe、旧 MPC recovery/evaluator 和四份旧规范已复制到 `archive/phase2_sac_ablation/`。活动目录删除 BC 与一次性 diagnose/recovery 脚本；16D `build_phase2_observation_v1` 已从 observation 模块移除；旧 `phase2_mpc_config` 只在底层定义模块保留兼容别名，活动代码不 import。原始 logs/models/TensorBoard/manifest/evaluation 未删除、未重写。

## 发现的真实 bug

本轮发现并修复 manifest 恢复遗漏：`asdict(SE3RendezvousConfig)` 会把 `SuccessThresholds` 变成 dict，旧 evaluator 只重建 `Phase2TaskConfig`，导致 evaluation config 不再与训练对象严格等价。现已显式重建 `SuccessThresholds`，并由回归测试覆盖。此前已修复的 23D corridor direction 越界问题保持不变。本轮未发现或修改动力学、RK45、任务几何、reward step、终止或 MPC 数学 bug。

## Canonical 参数来源

物理/输入常量在 `dynamics/constants.py`；任务/约束/completion 在 `env/task.py::Phase2TaskConfig`；阶段/初态 envelope/observation scales 在 `env/phase2_env.py` 与底层 frozen config；reward 公式在 `env/reward.py`；SAC 在 `train/configs.py::PureSACConfig`；MPC 在 `controllers/mpc/config.py::constrained_mpc_nominal_config`。没有 YAML/JSON 第二来源。

## 验证

使用 Python 3.12 bundled runtime 加载工程 `.venv/Lib/site-packages`：完整回归 `68 passed in 21.52s`；canonical `--validate-only` 通过；23D nominal reset 得到可行初态；零/随机动作有限性、reward NaN/Inf、margin consistency、success、MPC regressions、训练/评估 task equality 与 legacy import 隔离均有测试。工程 `.venv/Scripts/python.exe` 自身仍指向已卸载的 Python 3.12.6，这是下一次实际运行前需修复的运行时风险，不是代码回归失败。

## 下一步接口

fixed-small entropy 只改 `train/configs.py::PureSACConfig.ent_coef`；简化 progress reward 只改 `env/reward.py::Phase2TaskReward` 并保持 `compute_task_metrics` 不动；未来 SAC-MPC 新建独立模块并复用 canonical env/task/controller，不应重新把 demonstration 或 hybrid callback 注入 Pure SAC 入口。
