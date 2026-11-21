# 证据目录索引

本目录只放已执行实验的事实证据（读出、判定、清单、汇总；模型 ZIP 与大量原始回合文件不入库，只登记 SHA-256）。2026-11-12 工程收尾时，整理前的完整目录在科学提交 `fcd8d2ee`（标签 `archive/pre-cleanup-20261112`）中原样保存；移出工作树的目录见 `docs/archive/PATH_MAP_20261112.csv`，取回方法：`git checkout fcd8d2ee -- evidence/<目录>`。

## 0　论文正式结果（时限捕获，第一稿）

| 目录 / 文件 | 内容 | 论文用途 |
|---|---|---|
| `PAPER_READOUT/` | 论文全部数字与作图数据（`experiments/paper_readout.py` 生成，含 SHA256SUMS） | 第 V 节、`docs/paper/facts/08_实验结果.md` |
| `FORMAL/` | 正式评估逐回合表与核验：第一轮四组（`ECC_SESSION1` 等）、`ECC_LE_SESSION6`（最终 ECC）、`SAC_LE_SESSION7`、`TD3_C_CLOUD_20261118_7e55a65d/`（TD3–MPC，云端运行）；`B/` 为训练日志与清单 | 原始数据 |
| `PAPER_DATA_ROLLUP_20261119/` | 下层的数据统筹交接：结果分组、模型与训练索引、来源 SHA | 资产索引 |
| `LE_CLOUD_PARTIAL_20261119/` | SAC_LE 验证曲线的云端补跑（192 回合）原始记录 | 验证曲线来源 |
| `ECC_LE_PROBE/` | 第一轮 ECC 模型加远场入口的只评估检验（253/288） | 事实稿 08 的补充说明 |
| `ECC_LE_DEV/` | 远场入口的开发期检验（固定意图 40/48 → 43/48） | 方法设计依据 |
| `PAPER_FIGS/` | 两张案例图的轨迹数据（第一轮模型） | 第 V 节案例图 |

## A　工程核验与方法设计依据（保留）

| 目录 | 内容 |
|---|---|
| `final_pretest/` | EACH 预实验：含 NO_GEOM、窗口 / 刚性配对等；论文 RWG 的开发期证据（旋转引导 35/48 对 21/48） |
| `ECC_FINAL_PRECHECK/` | 最终轮等价性脚本与重放（`experiments/paper_readout.py` 读取） |
| `FORMAL_PREP/` | 正式实验准备：对 `e0a61753` 的环境与训练逐位回归、冒烟、资源条件预筛 |
| `CLEANUP_EQUIV/` | 2026-11-12 工程收尾的逐位等价核验（脚本与结果） |
| `SPEEDUP_20261113/` | 不改结果的提速改动的逐位核验 |
| `R2_EQUIV/` | 第二轮：收敛后的协调与第一稿 ECC_LE 执行逐位一致的核验（`docs/final_method/ROUND2.md` 第 2 节） |

## 已移出工作树（历史，按需取回）

**2026-11-19（第二轮清理）：** 开发期证据（调节器 GOV_A / GOV_B、ECC_DEV、U2 / U2C、T2.7–T2.19 负结果台账）共 27 个目录、12,222 个文件，从科学提交 `c15ece12` 取回：`git checkout c15ece12 -- evidence/<目录>`。清单见 `docs/archive/PATH_MAP_20261119.csv`；DEAD_ENDS 中的证据路径不变。

**2026-11-12（工程收尾）：** 从 `fcd8d2ee` 取回。

| 组 | 目录 | 说明 |
|---|---|---|
| 早期 T2（T2.0–T2.6W1） | `T2_PRECHECK`、`T2_STAGE_A/B`、`T2_2_TEST`、`T2_3_TEST`、`T2_4_TEST`、`T2_5_VALIDATE`、`T2_5_1_VALIDATE`、`T2_6_PRESSURE`、`T2_6W_WIDE_140_200`、`T2_6W1_CUTOFF` | 分阶段接口的建立、时间压力与截断语义；结论已并入 `docs/HISTORY.md` 与 DEAD_ENDS |
| T1 线 | `T1_*`、`T1B_*`、`T1N_*`、`T1O_*`、`T1P_*` | 单层审议接口、反事实 MPC、区间 Bellman 等；结论见 DEAD_ENDS D1x–D3x |
| F01 / V3 线 | `E01`–`E08`、`historical/` | 价值仲裁与停止选项；结论见 DEAD_ENDS D01–D1x |
| 开题 PPT 图 | `ppt_dev/` | 不用于论文 |

**第二轮**的证据：反馈机制开发放 `R2_DEV/`，正式实验放 `R2_FORMAL/`（执行书在 collab 分支）。
