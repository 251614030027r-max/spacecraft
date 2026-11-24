# 执行书：RUN_R2_ASSETS（论文补充资产，只评估，不训练）

**状态：** 用户 2026-11-24 授权，立即开跑。

**依据：** 科学分支 `docs/final_method/FORMAL.md` 第 10 节。只用第二轮正式实验的最终模型（`B\TRAIN_*`），不训练，不改动 `evidence/R2_FORMAL/`。

## 0. 固定设置

| 项 | 值 |
|---|---|
| 科学提交 | `9cde45f1`（所有机器检出同一提交） |
| 阶段 | `assets`（模型取自 `formal` 阶段） |
| 模型 | ECC、SAC_MPC × 种子 30、31、32，共 6 个最终模型 |
| 初始条件 | 301000–301191（与正式测试相同） |
| 结果目录 | 每台机器新建 `<ROOT_A>`，例如 `C:\Users\35884\Documents\Spacecraft\过程文件\R2_ASSETS`；正式结果目录 `<ROOT_F>` 只读 |

**全部工作量：**

| 项 | 内容 | 数量 |
|---|---|---|
| 泛化评估 | 6 个模型 × 4 个开关（`gen_spin_low`、`gen_spin_high`、`gen_deadline_tight`、`gen_nutation`）× 192 | 4608 回合 |
| 案例逐步重放 | 种子 30 的 ECC 与 SAC_MPC × 301001、301025，`--fine` | 4 条 |
| 终端误差 | 读正式评估已有结果，不运行 | — |

## 1. 开跑前（每台机器一次）

1. 确认没有遗留的 Python 进程；检出 `9cde45f1`。
2. 运行下面的命令，退出码为 0 才继续：

   ```
   python -B -m experiments.final preflight --root <ROOT_A>
   ```

## 2. 泛化评估

模型在哪台机器上，就在哪台机器上评估。每个模型、每个开关一条命令：

```
python -B -m experiments.final eval --stage assets --run <ROOT_F>\B\TRAIN_<CFG>_s<S> --switch <SW> --output <ROOT_A> --workers W
```

`<CFG>` 取 `ECC`、`SAC_MPC`；`<S>` 取 30、31、32；`<SW>` 取上面 4 个开关。PowerShell 一次排完一个模型：

```
foreach ($sw in 'gen_spin_low','gen_spin_high','gen_deadline_tight','gen_nutation') {
  python -B -m experiments.final eval --stage assets --run <ROOT_F>\B\TRAIN_<CFG>_s<S> --switch $sw --output <ROOT_A> --workers W
}
```

评估逐回合确定，并行数不影响结果；中断后用同一命令续跑。参考耗时：每回合约 1 分钟（单核），4608 回合约 80 核·小时。

## 3. 案例逐步重放（种子 30 的模型所在机器）

```
python -B -m experiments.final traj --entry ECC=<ROOT_F>\B\TRAIN_ECC_s30 --entry SAC_MPC=<ROOT_F>\B\TRAIN_SAC_MPC_s30 --openings 301001,301025 --output <ROOT_A>\traj_fine --fine
```

生成 4 个 `traj_<NAME>_fine_<opening>.json`。核对：每条的结局、完成时间、Δv 与 `evidence\R2_FORMAL\traj\` 中同名轨迹一致（跨机器只比结论，不比末位）。

## 4. 读出与提交

1. **汇总到一台机器：**
   - 其他机器的 `<ROOT_A>` 整个带过来；
   - 正式结果 `<ROOT_F>`（各机器的）也要在场：读出用其中的 `C\`、`S\` 原始结果和 `B\TRAIN_*\manifest.json`。正式打包时已经带到一台机器的，直接用那一份。
2. **读出：**

   ```
   python -B -m experiments.final assets --root <ROOT_F> [--root <ROOT_F2> ...] --root <ROOT_A> [--root <ROOT_A2> ...] --output evidence\R2_ASSETS
   ```

   退出码非 0 时看 `evidence\R2_ASSETS\READOUT.json` 的 `missing_or_error`，用第 2、3 节同一命令补跑后再读出。
3. **提交：** `git add -f evidence/R2_ASSETS`，提交并推送到科学分支；在协作分支 `docs/collaboration/lower/handoffs/r2_assets/RESULT.md` 写一两行（科学提交号、读出是否完整、异常）。

**不做：** 训练；改动 `evidence/R2_FORMAL`；自制额外核验文件；跨机器比较比特。
