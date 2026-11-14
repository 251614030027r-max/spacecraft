# ECC：冻结前的检查（上层沙箱，2026-11-07）

- `EACH_EQUIVALENCE.txt`：EACH（预实验配置 `reduced_priv` / `e_priv`）在 env 0.16.0 / learner 7.2 下的 dryrun 训练与 `fd789126` 逐位一致（第 20、40 步策略参数，回合，训练与执行统计）。
- `pytest.log`：106 项测试通过。新增测试覆盖：
  - ECC 接口（规划答案与执行内部状态从决策层去掉，critic 部分完整）；
  - 认证集与认证映射；
  - 资源条件；
  - 学习器视图。
- `smoke/`：`SAC_C`、`EACH_C`、`ECC` 各 30 决策冒烟训练（1 worker），`ECC_2workers` 为 40 决策、2 worker。manifest 视图为 24/24、24/48、24/58，ECC 带 `certify: true`；ECC 训练回合的 `exec_stats.cert_interventions` 记录到了介入。
- 评估路径（另行冒烟）：ECC 按训练方式（认证开）与 `--no-certify`（认证关）都能评估，结果文件记录 `certify` 与介入次数。
- 实现中发现并修复：新方法名与已有的 T2.7 `_opportunity` 冲突，已改名为 `_opportunity_eval`，`test_track_intent` 通过。
