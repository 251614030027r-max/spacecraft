# DRL2：六自由度快速翻滚非合作目标预捕获

本仓库实现高保真 SE(3) 六自由度预捕获控制：目标自由翻滚，真值传播包含中心引力、J2、引力梯度、二阶矩与刚体耦合，采用 RK45 积分，控制周期 0.1 s。

## 当前主线（2026-09-19）

当前唯一主线是 **adaptive sync-entry task + 强约束 MPC + baseline-anchored SAC-MPC**。RL 每 2 s 输出有界的捕获进度/参考残差，MPC 每 0.1 s 安全执行；残差零逐位恢复固定设定点 Pure MPC。机会来自同步/共旋与暂存后进入之间的连续资源代价，不使用相位硬门，也不人为削弱 Pure MPC。

第一轮 `adp_262410/411/412` 因奖励量纲错误在约 31k 步停止：安全接近惩罚漏乘 `dt`，造成合法完成回报低于悬停。该错误已经修复并加入防回归测试；旧 critic 和 checkpoint 禁止续训。下一步只允许从零运行 `adp_rf_262410/411/412`，在 10k 做奖励灾难检查、30k 看趋势、60k 完成正式训练。

## 当前入口

1. `CLAUDE.md`：冻结的研究问题、边界和执行纪律；
2. `docs/ADAPTIVE_MAINLINE_RUNSHEET.md`：当前训练与正式评估命令；
3. `docs/REWARD_UNITS_FIX_20260919.md`：奖励错误、修复和证据；
4. `docs/TRAINING_PREFLIGHT_AUDIT_20260919.md`：本机训练前代码审计；
5. `docs/FAST_TUMBLING_CAPTURE_MOTIVATION.md`：物理动机与论文口径；
6. `docs/EVIDENCE_INDEX.md`、`docs/REPRODUCIBILITY.md`：证据与复现索引；
7. `docs/HISTORY.md`、`docs/handoffs/`：历史材料，不是当前执行授权。

其他 T6–T12、感知、机会硬走廊和 timing probe 文档均为演进记录；不得覆盖以上入口或重新开启已否决机制。

## 目录

- `dynamics/`、`env/`：SE(3) 动力学、任务、奖励和环境；
- `controllers/`：约束 MPC；
- `train/`、`eval/`、`experiments/`：训练、正式评价和诊断；
- `tests/`：回归与训练前契约测试；
- `docs/`：当前入口及历史证据；
- `logs/`、`eval/cal2/`、`local_artifacts/`：本机运行产物，默认不入 Git。

训练前运行：

```text
python -B -m pytest -q
```

当前经审计基线为 `263 passed`（加入 5 个奖励单位防回归测试后）；正式启动命令以 `docs/ADAPTIVE_MAINLINE_RUNSHEET.md` 第 2b 节为准。
