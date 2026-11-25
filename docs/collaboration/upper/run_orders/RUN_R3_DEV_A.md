# 执行书：RUN_R3_DEV_A（第三轮开发轮 A + 诊断 D2）

**状态：** 等用户开场指令。

**依据：** 科学分支 `docs/final_method/ROUND3.md` 第 2 节（D1 判定、方法 v3 候选、采用规则、D2）。结果用于第三轮方法定型，不进论文。

## 0. 固定设置

| 项 | 值 |
|---|---|
| 科学提交 | `89b1e784`（所有机器检出同一提交） |
| A：开发训练 | 阶段 `dev3`；`ECC_V3A`、`ECC_V3B`、`ECC_NC3`、`SAC_MPC` × 种子 40、41、42 = 12 个运行，每个 3000 次决策，从零训练 |
| A：评估 | 每个最终模型在 296200–296391（192 个）上评估，共 2304 回合 |
| D2：只评估 | 阶段 `diag3b`；第二轮正式 `ECC`、`SAC_MPC` × 种子 30–32；4 个开关 × 96 个初始条件（296000–296095），共 2304 回合 |
| 结果目录 | 开发训练 `<ROOT_A>`；D2 `<ROOT_D2>`；D1 结果 `<ROOT_D>` 与正式结果 `<ROOT_F>` 只读 |

**资源：** 每个训练运行约 2.3 个核，12 个运行约 28 个核，约 8.5–10 小时。D2 用剩下的核，约 40 核·小时，和训练同时跑。

## 1. 开跑前（每台机器一次）

1. 确认没有遗留的 Python 进程；检出 `89b1e784`。
2. `python -B -m experiments.final preflight --root <ROOT_A>`，退出码为 0 才继续。

## 2. A：开发训练与评估

```
python -B -m experiments.final train --stage dev3 --config <CFG> --seed <S> --output <ROOT_A>\B\TRAIN_<CFG>_s<S>
python -B -m experiments.final eval  --stage dev3 --run <ROOT_A>\B\TRAIN_<CFG>_s<S> --output <ROOT_A> --workers W
```

`<CFG>` 取 `ECC_V3A`、`ECC_V3B`、`ECC_NC3`、`SAC_MPC`；`<S>` 取 40、41、42。每个模型训练完立即在同一台机器上评估。

## 3. D2：只评估（空闲核，和训练同时进行）

```
foreach ($sw in 'r3b_hs','r3b_aligned','r3b_hs_aligned','r3b_hs_aligned_slips') {
  python -B -m experiments.final eval --stage diag3b --run <ROOT_F>\B\TRAIN_<CFG>_s<S> --switch $sw --output <ROOT_D2> --workers W
}
```

`<CFG>` 取 `ECC`、`SAC_MPC`；`<S>` 取 30、31、32。

## 4. 读出与提交

1. **A：** `python -B -m experiments.final pack --stage dev3 --root <ROOT_A> [--root ...] --output evidence\R3_DEV_A`
2. **D2：** `python -B -m experiments.final diag --stage diag3b --root <ROOT_D2> --root <ROOT_D> --root <ROOT_F> --output evidence\R3_DIAG2`（`<ROOT_D>` 提供 D1 中无开关的结果）。
3. 两项都完整后，`git add -f evidence/R3_DEV_A evidence/R3_DIAG2`，提交并推送科学分支；协作分支 `docs/collaboration/lower/handoffs/r3_dev_a/RESULT.md` 写一两行。

**不做：** 按采用规则判定（由上层做）；自制额外核验文件。
