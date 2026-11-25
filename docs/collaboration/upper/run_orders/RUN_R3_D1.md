# 执行书：RUN_R3_D1（第三轮诊断，只评估，不训练）

**状态：** 等用户开场指令。

**依据：** 科学分支 `docs/final_method/ROUND3.md` 第 1 节。用第二轮正式最终模型（`B\TRAIN_*`），在开发块中未用过的初始条件 296000–296095（96 个）上评估。结果只用于第三轮方法设计，不进论文。

## 0. 固定设置

| 项 | 值 |
|---|---|
| 科学提交 | `4ee3fc47`（所有机器检出同一提交） |
| 阶段 | `diag3`（模型取自 `formal`） |
| 模型 | ECC、ECC_NO_CORE、SAC_MPC × 种子 30、31、32 |
| 初始条件 | 296000–296095（96 个） |
| 结果目录 | 每台机器新建 `<ROOT_D>`；正式结果目录 `<ROOT_F>` 只读 |

**全部工作量（10 组 × 3 种子 × 96 = 2880 回合，约 50 核·小时；32 核约 2 小时）：**

| 模型 | 评估（每个种子一条命令；`--switch` 为空表示不加开关） |
|---|---|
| ECC | 无开关；`r3_launch_first`；`r3_no_correction`；`r3_launch_lead32`；`r3_hs`；`r3_hs_slips`；`r3_hs_force07`；`r3_hs_replan05` |
| ECC_NO_CORE | 无开关 |
| SAC_MPC | 无开关 |

## 1. 开跑前（每台机器一次）

1. 确认没有遗留的 Python 进程；检出 `4ee3fc47`。
2. `python -B -m experiments.final preflight --root <ROOT_D>`，退出码为 0 才继续。

## 2. 评估

模型在哪台机器上就在哪台机器上跑：

```
python -B -m experiments.final eval --stage diag3 --run <ROOT_F>\B\TRAIN_<CFG>_s<S> [--switch <SW>] --output <ROOT_D> --workers W
```

PowerShell 一次排完一个模型（以 ECC 为例；`''` 表示不加开关）：

```
foreach ($sw in '', 'r3_launch_first','r3_no_correction','r3_launch_lead32','r3_hs','r3_hs_slips','r3_hs_force07','r3_hs_replan05') {
  if ($sw) { python -B -m experiments.final eval --stage diag3 --run <ROOT_F>\B\TRAIN_ECC_s<S> --switch $sw --output <ROOT_D> --workers W }
  else     { python -B -m experiments.final eval --stage diag3 --run <ROOT_F>\B\TRAIN_ECC_s<S> --output <ROOT_D> --workers W }
}
```

ECC_NO_CORE 与 SAC_MPC 只跑无开关的一条。评估逐回合确定；中断后用同一命令续跑。

## 3. 读出与提交（汇总到一台机器）

读出前检出科学 `304bb6df`（评估仍在 `4ee3fc47`，不要在评估运行中切换；读出只多一列 `timeout_in_cone`）。

```
python -B -m experiments.final diag --root <ROOT_D> [--root <ROOT_D2> ...] --root <ROOT_F> [--root <ROOT_F2> ...] --output evidence\R3_DIAG
```

（`<ROOT_F>` 只为找到 `B\TRAIN_*` 的运行目录。）退出码非 0 时看 `evidence\R3_DIAG\READOUT.json` 的 `missing_or_error`，补跑后再读出。

`git add -f evidence/R3_DIAG`，提交并推送科学分支；协作分支 `docs/collaboration/lower/handoffs/r3_d1/RESULT.md` 写一两行（提交号、是否完整、异常）。

**不做：** 训练；结论归因（由上层判定）；自制额外核验文件。
