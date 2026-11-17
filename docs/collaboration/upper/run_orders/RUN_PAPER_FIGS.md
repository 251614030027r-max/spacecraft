# 执行书：RUN_PAPER_FIGS（论文案例图的数据导出）

**状态：用户要求（2026-11-18）。只重放第一轮已有模型，不训练，不改代码；几分钟可完成，可以与 RUN_ECC2_GATE0 同时进行。**

用途：论文第 V 节的两张案例图：典型捕获（Fig. case）和认证案例（Fig. screen-case）。图由写作侧根据导出的数据自行绘制。

## 0. 前提

| 项 | 值 |
|---|---|
| 代码 | 本机 `D:\py\DRL2` 当前检出的科学分支提交（`868fe5e6` 或之后；方法代码与 `e8474e21` 相同）。不切换、不拉取 |
| 模型 | `$R\B\TRAIN_ECC_CERT_LOOP_s20`、`$R\B\TRAIN_SAC_C_s20`（`$R` = `C:\Users\35884\Documents\Spacecraft\过程文件\FORMAL`） |
| 输出 | `C:\Users\35884\Documents\Spacecraft\过程文件\PAPER_FIGS\`（记作 `$T`） |

## 1. 选定认证案例的初始条件

按下列顺序，在 `$R\C\ECC_CERT_LOOP_s20_urgent\ECC_CERT_LOOP_s20_urgent_<编号>.json` 中找**第一个**满足条件的编号：`decisions` 中任一项的 `cert.intervened` 为 true。

300005, 300010, 300011, 300019, 300023, 300034, 300043, 300054, 300056, 300060, 300068, 300072, 300076, 300080, 300095

这些都是 ECC（种子 20）捕获、而 SAC–MPC（种子 20）未捕获的初始条件，由科学 `evidence/FORMAL/ROUND1_READOUT/URGENT_EPISODES_WIDE.csv` 得出。选中的编号记作 `X`。如果列表里没有满足条件的，写明情况，停在这一步。

## 2. 重放导出

```
python -B -m experiments.final traj --entry ECC=$R\B\TRAIN_ECC_CERT_LOOP_s20 --entry SAC=$R\B\TRAIN_SAC_C_s20 --openings 300000,X --output $T --condition urgent
```

- **输出：** 4 个文件 `traj_ECC_300000.json`、`traj_SAC_300000.json`、`traj_ECC_X.json`、`traj_SAC_X.json`。
- **文件内容：**
  - `rows`：每 0.1 s 一行，包括相对位置与速度、目标姿态与角速度、距离、端口–视线夹角、纬向偏差、相位、执行模式、力与力矩、终端误差与各约束裕度；
  - `decisions`：每次决策的提议意图；认证记录中有 9 个候选的预测事件、类型、时刻、Δv，以及是否可认证、是否介入、执行意图。
- **复现检查：** 每个文件的 `event` 和 `return` 必须与正式评估一致：
  - `traj_ECC_300000`：success，64.89486154337825；
  - `traj_SAC_300000`：success，66.31917365095221；
  - X 的两条与 `$R\C\` 下对应评估文件的值一致。

  跨机器只比较结局，回报允许末位差异。不一致就停下报告，不交付。

## 3. 交付

- **科学分支** `evidence/PAPER_FIGS/`，逐个文件强制添加：
  - 4 个 `traj_*.json`；
  - `SHA256SUMS`；
  - `README.md`：命令、选定的 X 及选择依据、复现检查结果。
- **协作分支** `lower/handoffs/` 写一行交付说明（科学提交号、X）。
