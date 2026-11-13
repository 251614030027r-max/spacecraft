# SPEEDUP_20261113：不改结果的提速核验

**对象：** 科学 `52d673b5`（`dynamics/fastmath.py`、`controllers/mpc/cvxpy_fastpath.py` 及其调用点、`dynamics/lie.py`）。参照为冻结记录，即 `evidence/CLEANUP_EQUIV/` 中 `fcd8d2ee` 的输出。脚本同 `evidence/CLEANUP_EQUIV/scripts/`。改动说明见 `docs/paper/AUDIT.md` 第 5 节。

| 文件 | 结果 |
|---|---|
| `ENV_COMPARISON.json`（原始记录 `env_new_*.json`） | 7 / 7 种情形逐步逐位一致 |
| `TRAIN_COMPARISON.json` | 6 / 6 个配置（种子 10、40 决策、单进程）：检查点全部张量、回合记录、训练统计、`worker_order` 逐位一致 |
| `EVAL_COMPARISON.json` | 旧代码训练的 ECC 模型，2 / 2 回合逐决策一致 |
| `SCREEN_COMPARISON.json` | 10 / 10 个固定策略回合一致 |
| `pytest.log` | 160 passed，退出码 0（含锁定逐位一致的两项新测试） |
| `TIMING.json` | 同一沙箱上 7 种情形合计墙钟：536 s → 461 s（−14%） |
