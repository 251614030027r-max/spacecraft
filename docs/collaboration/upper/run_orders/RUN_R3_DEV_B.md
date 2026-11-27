# 执行书：RUN_R3_DEV_B（第三轮开发轮 B：方法 v4，本机运行）

**状态：** 等用户开场指令。

**依据：** 科学分支 `docs/final_method/ROUND3.md` 第 3 节（方法 v4、开发轮 B、采用规则）。结果用于第三轮方法定型，不进论文。

## 0. 固定设置

| 项 | 值 |
|---|---|
| 科学提交 | `5432ecce`（检出这一提交） |
| 机器 | 本机。开发轮 A 的模型 `<ROOT_A>\B\TRAIN_*` 在哪台机器上，就在哪台机器上评估；不在本机的，从云端把整个运行目录（含 `final.zip`）带到本机 `<ROOT_A>\B\` 下 |
| 训练 | 阶段 `dev3`；`ECC_V4` × 种子 40、41、42，每个 3000 次决策，从零训练；输出到开发轮 A 的同一个 `<ROOT_A>` |
| 评估 | ① `ECC_V4` 的 3 个模型，无开关（`--stage dev3`），576 回合；② 阶段 `diag3c` 的开关，3456 回合（见第 3 节） |
| 初始条件 | 296200–296391（与开发轮 A 相同，直接配对） |

**资源：** 训练 3 个运行，约 7 个核、8.5–10 小时。`diag3c` 中 V3B、SAC_MPC 的评估不依赖新训练，开跑即可用空闲核同时进行；V4 的两项等训练完成后再跑。评估约 60 核·小时。

## 1. 开跑前

1. 确认没有遗留的 Python 进程；检出 `5432ecce`。
2. `python -B -m experiments.final preflight --root <ROOT_A>`，退出码为 0 才继续。

## 2. 训练与基本评估

```
python -B -m experiments.final train --stage dev3 --config ECC_V4 --seed <S> --output <ROOT_A>\B\TRAIN_ECC_V4_s<S>
python -B -m experiments.final eval  --stage dev3 --run <ROOT_A>\B\TRAIN_ECC_V4_s<S> --output <ROOT_A> --workers W
```

## 3. 只评估的检查（阶段 `diag3c`，输出到 `<ROOT_C>`）

| 模型 | 开关 |
|---|---|
| `ECC_V3B` × 40–42 | `dev_window`、`dev_hs` |
| `SAC_MPC` × 40–42 | `dev_window`、`dev_hs` |
| `ECC_V4` × 40–42（训练完成后） | `dev_fixed`、`dev_hs` |

```
python -B -m experiments.final eval --stage diag3c --run <ROOT_A>\B\TRAIN_<CFG>_s<S> --switch <SW> --output <ROOT_C> --workers W
```

## 4. 读出与提交

1. `python -B -m experiments.final pack --stage dev3 --root <ROOT_A> [--root <云端结果副本>] --output evidence\R3_DEV_B`（开发轮 A 的 12 个运行与新的 3 个运行都要在场）
2. `python -B -m experiments.final diag --stage diag3c --root <ROOT_C> --root <ROOT_A> --output evidence\R3_DIAG3`（`<ROOT_A>` 提供无开关的结果）
3. `git add -f evidence/R3_DEV_B evidence/R3_DIAG3`，提交并推送科学分支；协作分支 `docs/collaboration/lower/handoffs/r3_dev_b/RESULT.md` 写一两行。

**不做：** 按采用规则判定（由上层做）；自制额外核验文件。
