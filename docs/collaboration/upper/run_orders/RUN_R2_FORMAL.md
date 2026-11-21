# 执行书：RUN_R2_FORMAL（第二轮正式实验：论文最终数据）

**状态：** 等用户开场指令。

**依据：** 科学分支 `docs/final_method/FORMAL.md`（组、协议、预登记的报告量、案例规则、图表来源）。交付方式见 [WORKFLOW.md](../../WORKFLOW.md)。

| 项 | 值 |
|---|---|
| 科学提交 | `81d43f55`（本机与云端必须相同） |
| 阶段 | `formal`：种子 30、31、32；6000 次决策；测试初始条件 301000–301191（192 个） |
| 训练配置（12 个运行） | `ECC`、`SAC_MPC`、`ECC_NO_CORE`、`TD3_MPC` × 3 个种子 |
| 只评估 | `ECC` 的 3 个模型加 `rigid_entry` |
| 非学习 | `pure`、`c_10_0`、`b_10` |
| 结果目录 | 本机 `C:\Users\35884\Documents\Spacecraft\过程文件\R2_FORMAL\`（`$L`）；云端 `<云端>/R2_FORMAL`（`$C`） |

## 0. 开跑前（两台机器各一次）

1. 确认没有遗留的 Python 进程。第一轮出现过旧等待任务重复评估、争抢资源的情况。
2. 检出 `81d43f55`。
3. 本机运行 `python -B -m experiments.final preflight --root $L`，云端运行 `... --root $C`。退出码为 0 才继续（沙箱 167 passed）。

## 1. 分工（原则：两台机器始终满载，主比较最先完成）

| 批次 | 本机（4 个运行） | 云端（2 个运行） |
|---|---|---|
| 第一批 | `ECC` s30、s31、s32；`SAC_MPC` s30 | `SAC_MPC` s31、s32 |
| 第二批 | `ECC_NO_CORE` s30、s31、s32；`TD3_MPC` s30 | `TD3_MPC` s31、s32 |

- 每个运行 2 个采集进程，约占 2.3 个核。
- 某台机器的运行一结束，立刻在该机器开下一批的运行，不等另一台。
- 云端核数多于 6 个时，可以把第二批的运行提前放到云端（每多 2.5 个空闲核加 1 个运行）。
- 种子 30 的 `ECC` 与 `SAC_MPC` 都在本机，案例重放在本机完成。

**训练命令：**

```
python -B -m experiments.final train --stage formal --config <CFG> --seed <S> --output <ROOT>\B\TRAIN_<CFG>_s<S>
```

`<ROOT>` 是本机的 `$L` 或云端的 `$C`。

## 2. 空闲核不闲着

- **非学习策略（本机，第一批开跑即开始）：** 不需要模型，用空闲核跑（约 2 个 worker）：

  ```
  python -B -m experiments.final screen --stage formal --policy pure   --output $L --workers 2
  python -B -m experiments.final screen --stage formal --policy c_10_0 --output $L --workers 2
  python -B -m experiments.final screen --stage formal --policy b_10   --output $L --workers 2
  ```

- **评估：** 每个模型训练完，在训练它的机器上用空闲核立即评估：

  ```
  python -B -m experiments.final eval --stage formal --run <ROOT>\B\TRAIN_<CFG>_s<S> --output <ROOT> --workers W
  ```

  `ECC` 的模型再加一条：

  ```
  python -B -m experiments.final eval --stage formal --run <ROOT>\B\TRAIN_ECC_s<S> --switch rigid_entry --output <ROOT> --workers W
  ```

- 全部训练结束后，剩余评估用满所有核（`W` = 空闲核数）。评估逐回合确定，并行数不影响结果。

## 3. 合并交付（本机）

1. **云端带回到本机某个目录（`$CB`）：**
   - 每个运行目录 `B\TRAIN_*` 里的 `manifest.json`、`finished.json`、`episodes.json`、`tb\`、`final.zip.sha256`。哈希文件只有一行 `<SHA-256>  final.zip`，模型本身不必回传；
   - 云端 `C\` 下它评估的全部结果目录。

2. **打包：**

   ```
   python -B -m experiments.final pack --stage formal --root $L --root $CB --output evidence\R2_FORMAL
   ```

   失败时看 `evidence\R2_FORMAL\CHECK.json`，用同一命令补跑缺项（已有结果会跳过），然后再 pack。

3. **案例重放（本机）：** 运行 `evidence\R2_FORMAL\CASE_COMMANDS.txt` 里的命令，`<OUT>` 换成 `evidence\R2_FORMAL`。然后再执行一次第 2 步的 pack，把 `traj\` 计入哈希。

4. **提交：**
   - 只有出现异常时，才在 `evidence\R2_FORMAL\RESULT.md` 末尾写一两句；
   - `git add -f evidence/R2_FORMAL`，提交并推送到科学分支；
   - 把 `RESULT.md` 复制到协作分支 `docs/collaboration/lower/handoffs/r2_formal/RESULT.md`，提交并推送。

**不做：** 自制额外核验文件、重算旧证据哈希、跨机器比较比特、打归档标签以外的其他记录。归档标签只打一次：`archive/r2-formal-code` 指向 `81d43f55`。

## 4. 预计耗时

- 每个 6000 步的训练约 17–20 小时（参照开发轮 3000 步约 8.5–10 小时）。
- 两批训练合计约 36–40 小时。非学习策略与第一批模型的评估在训练期间用空闲核完成。
- 训练全部结束后，剩余评估约 1–2 小时；打包与案例重放约 1 小时。

## 5. 交付后（上层）

上层读 `READOUT.json` 等文件，更新事实稿 08，并按 `FORMAL.md` 第 4 节给出全部图表数据。泛化蒙特卡洛另写规格，只评估这批最终模型。
