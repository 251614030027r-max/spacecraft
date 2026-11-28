# CLEANUP_EQUIV：工程收尾的逐位等价核验（2026-11-12）

**对象：** 收尾瘦身后的科学代码，对照收尾前的 `fcd8d2ee`（其冻结路径与 `e0a61753` 逐位等价，`evidence/FORMAL_PREP/`）。结论与改动说明见 `docs/engineering/AUDIT.md` 第 4 节。

## 脚本（`scripts/`；每个脚本在两份检出中各运行一次，再比较）

| 脚本 | 用法 | 内容 |
|---|---|---|
| `env_equiv.py` | `python -B env_equiv.py ROOT CASE OUT.json` | 环境逐步记录。CASE 为 `ecc_urgent`、`each_urgent`、`nogeo_economy`、`rigid_urgent`、`ecc_natural`、`ecc_resource`、`ecc_lowfuel` 之一。比较：`compare_env.py REF_DIR NEW_DIR`，文件名为 `env_ref_<CASE>.json` / `env_new_<CASE>.json` |
| `train_equiv.py` | `python -B train_equiv.py ROOT CONFIG OUTDIR` | 种子 10、40 决策、单进程、正式阶段只读日志。比较：`compare_train.py REF_RUN NEW_RUN` |
| `eval_equiv.py` | `python -B eval_equiv.py ROOT RUN_DIR OUT [OPENINGS]` | 用 ROOT 的代码评估 RUN_DIR/final.zip（紧急、审计）。比较：`compare_eval.py REF_OUT NEW_OUT` |
| `screen_equiv.py` | `python -B screen_equiv.py ROOT OUT` | 固定策略 pure / gstar / rally / c_10_0 / b_10 × 紧急 294300、经济 R 294301。比较：`compare_screen.py REF_OUT NEW_OUT` |

参照检出用 `git worktree add <目录> fcd8d2ee`，在各自的检出目录中运行（训练要读 `git rev-parse HEAD`）。

## 上层沙箱结果（`sandbox/`）

| 文件 | 结果 |
|---|---|
| `ENV_COMPARISON.json`（原始记录 `env_*.json`） | 7 / 7 种情形逐位一致 |
| `TRAIN_COMPARISON.json` | 6 / 6 个配置：检查点全部张量、回合记录、训练统计、`worker_order` 逐位一致 |
| `EVAL_COMPARISON.json` | 旧代码训练的 ECC 模型在新代码下评估，2 / 2 回合逐决策一致 |
| `SCREEN_COMPARISON.json` | 10 / 10 个固定策略回合逐位一致（保留字段；结局含捕获、超时、燃料耗尽） |
| `pytest.log` | 158 passed，退出码 0 |

Windows 复核由下层按 collab `RUN_CLEANUP.md` 执行，结果放 `windows/`。
