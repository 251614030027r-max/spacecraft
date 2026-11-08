# ECC 最终轮：第 0 步（上层沙箱，2026-11-08）

规格：`docs/final_method/ECC_FINAL_SPEC.md`。本目录是冻结前的全部检查。

| 项 | 结果 | 文件 |
|---|---|---|
| 测试 | 113 项通过，退出码 0 | `pytest.log` |
| 旧路径逐位等价（对 `36994a1f`） | 环境层（`reduced_priv`、`cut_priv`、`cut_ecc` + 旧认证）逐步一致；EACH_C、SAC_C、ECC 冒烟训练 12 个检查点参数逐位相同，回合与训练统计一致 | `ENV_EQUIVALENCE.txt`、`TRAIN_EQUIVALENCE.txt`、`scripts/env_equiv.py`、`scripts/train_equiv.py` |
| 条件筛查 | `scarcity`（太易）、`scarcity:inside`（太难）、第 3 种自然紧急分布通过 L1–L3（G★ 0.917；五个阶段策略按开局取最好 0.875；单个最好 c_10_0 0.729）；pure 只报告 | `screen/base/`、`screen/inside/`、`screen/urgent/SCREEN_DECISION.json` |
| 6B 沙箱配对检查 | 第 1 轮：机会后推，兑现率 0.23 → 修正为机会时刻承诺；第 2 轮：下降类替代引入 1 次 unsafe（c_6_5 294427，FOV 违背于冻结 MPC 下降）→ 修正为替代只引向 G★ 入口；第 3 轮：0 丢失、0 新增 unsafe、救回 23（旧认证 24，且在 rally 294427 引入 1 次 unsafe），兑现率 0.73、持续率 0.96；用户接受并冻结 | `check6b/round1..3/`、`scripts/diag_prediction.py`、`scripts/replay_truth.py` |
| 回归案例（294320–294367 只作回归） | 最终规则：294323 不再过早介入（策略意图判为 successor，结局同不协调：捕获）；294334 不再引向下降（判为 lost，结局同不协调：超时，无 unsafe）；294334 原 unsafe 为冻结 MPC 下降中的视场违背 1.06e-4，接管时预测与真值都满足制动包络 | `REPLAY_294323_294334.json`、`REPLAY_294334_TRUTH.json`、`scripts/replay.py`、`scripts/replay_294334_truth.py` |
| 端到端冒烟 | 在环训练（ECC_GOV，2 worker，紧急 3/4 混合）与 EACH_C 冒烟；A 步全部评估命令（三个条件 × 监测 / 调节器 / 旧认证）、B 步按训练方式与监测评估、`gov_readout a` 跑通；监测与关闭逐位相同。冒烟模型几乎未训练，其判据数值无意义 | `smoke/` |
