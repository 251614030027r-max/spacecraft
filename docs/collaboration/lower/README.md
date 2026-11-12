# 下层入口（2026-11-12）

下层职责：在授权范围内于本机执行训练、评估与交付；维护 [CURRENT.json](../CURRENT.json) 与本机资产。

新交付放 `handoffs/<主题>/`，写 `RESULT.md`，大文件与模型留本机、登记 SHA。

| 交付 | 对应执行书 |
|---|---|
| `handoffs/formal_prep/` | RUN_FORMAL_PREP |
| `handoffs/gov_a/`、`handoffs/gov_b/` | RUN_GOV_A、RUN_GOV_B |
| `handoffs/ecc/` | RUN_ECC |
| `handoffs/u2/`、`handoffs/u2c/` | RUN_U2、RUN_U2C |
| `handoffs/final_pretest/` | RUN_PRETEST |
| `handoffs/assets_20261103/` | 11-03 本机资产盘点（下一次盘点由 RUN_CLEANUP 交付） |

更早的交付（F01、T1、T2.2–T2.19、旧清理与旧图）在 [../archive/handoffs/](../archive/handoffs/)，流水线脚本在 [../archive/pipeline_scripts/](../archive/pipeline_scripts/)。
