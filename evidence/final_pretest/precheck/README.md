# 预实验前的沙箱自检（上层，2026-11-05）

**目的：** 证明最终方法 EACH（`e_priv`，U1 + C1）在本次收尾改动之后，与 T2.12 冻结版本 `2e7d838` 逐位等价；并在沙箱中把预实验的全流程走通。

## 1. 主方法等价审计（在全部收尾改动之后完成）

本次改动：
- 新观测配置 `reduced_dup`；
- 训练开关 `--no-geometry`；
- 评估从 manifest 读取窗口与几何配置；
- 逐回合执行统计 `exec_stats`；
- manifest 附加字段；
- 唯一入口 `experiments/final.py`。

| 审计 | 做法 | 结果 | 文件 |
|---|---|---|---|
| 环境层 | 4 个开局（1000001、1000002、1000003、1000005），固定随机动作序列，`reduced_priv`，shaping 开，紧急期限 U(140, 200) s；`2e7d838` 与最终代码各跑一次 | 36 个决策的观测哈希、shaped 与真实回报、终止事件、模式、时间逐步一致 | `scripts/equivalence_env.py`；`outputs/equivalence_env_2e7d838.json`、`equivalence_env_final.json` |
| 训练层（旧命令行） | `experiments.t2 train --arm e_priv --seed 3 --steps 80 --workers 1 --smoke --obs reduced_priv --urgent-deadline 140,200`，新旧各一次 | 第 40、80 步策略参数逐位一致；回合日志与训练统计一致 | `scripts/equivalence_train_compare.py`；`outputs/equivalence_train_cli.txt` |
| 训练层（新入口） | `experiments.final train --stage dryrun --config EACH --seed 10 --workers 1`，对照 `2e7d838` 的同参数旧命令 | 第 20、40 步策略参数逐位一致；回合日志与训练统计一致 | `outputs/equivalence_train_final_entry.txt` |

## 2. 全流程 dryrun（沙箱）

**做法：** dryrun 阶段（`METHOD_REGISTRY.json`）：6 组配置 × 种子 10、11 × 40 决策（smoke 学习器设置）；每个模型在开局 294300–294301 上评估；EACH 另跑刚性承诺；固定策略 pure、rally、c_10_0，以及 rally 的刚性承诺；然后 `check`、`resolve`、出图。

**结果：**
- 12 条训练全部退出码 0；manifest 与注册表一致（各组的 arm、obs、Actor / Critic 维度、window、geometry、k = 0、beta = 0）；
- 每条训练回合带有 `exec_stats`；
- `CHECK.json`：`complete: true`，没有缺漏与错误；
- `resolve` 通过；
- `experiments.final_figures` 生成全部 5 张图与 `TABLES.json`。

**注意：** dryrun 的数值没有任何意义。`REGISTRY_RESOLVED.json` 中的 SHA 是运行时的 HEAD 加未提交改动；以下层在冻结 SHA 上的自检 dryrun 为准。

**文件：** `outputs/dryrun_sandbox/`（`CHECK.json`、`REGISTRY_RESOLVED.json`、`dryrun.log`）、`outputs/figures_dryrun_sandbox/`。

## 3. 几何扫描与轨迹（图 A、B 的流程检查）

**开局 294200 的扫描**（`outputs/geoscan_294200.json`）：
- β = 93.2°，自旋 2.16 °/s；
- 构造误差 < 1e-13°；
- 几何窗口约占平面 22.4%；G★ 可行区在 8 / 10 / 12 m 时约 16.7% / 14.3% / 12.3%，全部在 |Δλ| ≤ 30° 内；
- 单开局（5° 网格）约 1 分钟。

**轨迹：** rally 与 pure 在 294200 上各飞两次，结果完全相同（rally 115.4 s 捕获，pure 195.2 s 捕获）。

## 4. 测试

`python -B -m pytest -q tests/test_t2_stage.py tests/test_integrator.py`：全部通过，退出码 0（日志 `outputs/pytest.log`）。

新增测试：
- `reduced_dup` 配置；
- 最终入口拒绝未登记配置、阶段外种子与偏离注册表的 manifest；
- 评估读取 manifest；
- 保留开局互斥；
- 扫掠锥构造还原执行器自身的纬向误差与相位。
