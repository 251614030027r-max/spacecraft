# 执行书：RUN_R3_DEV_C（第三轮开发轮 C，最后一轮开发，本机）

**状态：** 等用户开场指令。

**依据：** 科学分支 `docs/final_method/ROUND3.md` 第 4 节（论文要点、开发轮 C 与采用规则、正式实验设计）。结果用于定型，不进论文。

## 0. 固定设置

| 项 | 值 |
|---|---|
| 科学提交 | `0451183e`（检出这一提交） |
| 训练 | 阶段 `dev3`；`ECC_V4F` × 种子 40、41、42，每个 3000 次决策，从零训练；输出到开发轮 A / B 的同一个 `<ROOT_A>` |
| 评估 | ① `ECC_V4F` 无开关（`--stage dev3`），576 回合；② 阶段 `diag3d`，`ECC_V4F` 加开关 `dev_hs`，576 回合 |
| 初始条件 | 296200–296391（与开发轮 A / B 相同） |

**资源：** 3 个训练运行，约 7 个核、8.5–10 小时；评估约 20 核·小时。

## 1. 开跑前

1. 确认没有遗留的 Python 进程；检出 `0451183e`。
2. `python -B -m experiments.final preflight --root <ROOT_A>`，退出码为 0 才继续。

## 2. 训练与评估

```
python -B -m experiments.final train --stage dev3 --config ECC_V4F --seed <S> --output <ROOT_A>\B\TRAIN_ECC_V4F_s<S>
python -B -m experiments.final eval  --stage dev3 --run <ROOT_A>\B\TRAIN_ECC_V4F_s<S> --output <ROOT_A> --workers W
python -B -m experiments.final eval  --stage diag3d --run <ROOT_A>\B\TRAIN_ECC_V4F_s<S> --switch dev_hs --output <ROOT_C> --workers W
```

`<ROOT_C>` 用开发轮 B 的同一个目录（里面已有 V3B、SAC_MPC 的高自旋结果）。

## 3. 读出与提交

1. `python -B -m experiments.final pack --stage dev3 --root <ROOT_A> [--root <云端结果副本>] --output evidence\R3_DEV`。开发轮 A、B、C 一起打包，共 18 个运行，开发轮 B 待补交的包并入这一个。
2. `python -B -m experiments.final diag --stage diag3d --root <ROOT_C> --root <ROOT_A> --output evidence\R3_DIAG4`
3. `git add -f evidence/R3_DEV evidence/R3_DIAG4`，提交并推送科学分支；协作分支 `docs/collaboration/lower/handoffs/r3_dev_c/RESULT.md` 写一两行。

**不做：** 判定（由上层做）。
