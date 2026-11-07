# T12 / S10 精简审查证据包

日期：2026-09-15 JST。该目录用于 Git 审查与复核，不替代本机完整原始目录 `logs/t12_train/`、`logs/t12_eval/` 和 `logs/t12_eval_v2/`；原始日志、模型、checkpoint 与 TensorBoard 均继续本地保留。

## 审查入口

- `docs/T12_S10_FORMAL_EVALUATION_REPORT_20260915.md`：正式结论、分布、安全证据、限制与上层待裁决项。
- `docs/T12_S10_EVAL_EXECUTION_ORDER.md`：评估前已登记的六条命令、判据与护栏。
- `final_summary.json`：六个训练种子与 Pure MPC 的机器可读汇总。
- `main_table_seed_rows.txt`、`main_table_seed_rows.json`：排除无效并行 compute 列后的审计主表。
- `episodes/*.json`：六个运行在固定种子块 262000–262047 上的逐局结果，仅保留 JSON，不含 stdout/stderr 副本。
- `training_manifests/*.json`：六个 60000 决策训练的配置与完成记录。

## 边界

- 训练与评估均为完整真值状态 `perception=None`，不是局部视觉或 EKF 结果。
- Pure MPC 来源为既有 48 种子结果；旧记录没有新增真值安全字段，缺失值保持缺失。
- 两轮评估均为并行运行，compute 数字不得作为实时性证据。
- oracle 40/48 是事后逐种子选择，不进入正式主表；Pure SAC 本轮未执行。
- 首轮结果未入该精简包，但仍完整保留在本机；补字段后的第二轮与首轮逐局确定性一致，差别仅为新增的七类真值违约计数。

## 提交前验证

使用项目 `.venv` 的 site-packages 与 Python 3.12.6 运行：

```text
python -m pytest tests/test_hybrid_env.py tests/test_train_eval_config.py
41 passed in 82.98s
```
