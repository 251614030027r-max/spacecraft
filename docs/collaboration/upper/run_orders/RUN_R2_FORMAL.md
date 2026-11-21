# 执行书：RUN_R2_FORMAL（第二轮正式实验，论文最终数据）

**状态：** 等用户开场指令。一次性完成，结果直接用于论文。

**依据：** 科学分支 `docs/final_method/FORMAL.md`（组、协议、预登记的报告量、案例规则、图表来源）。短检查已完成，ECC 位用 `ECC`（第 8.1 节）。交付方式见 [WORKFLOW.md](../../WORKFLOW.md)。

## 0. 固定设置

| 项 | 值 |
|---|---|
| 科学提交 | `41601520`。所有机器检出同一提交；不同提交的结果打包时会被拒绝 |
| 阶段 | `formal` |
| 种子 | 30、31、32 |
| 训练 | 6000 次高层决策，从零开始，每个运行 2 个采集进程 |
| 测试初始条件 | 301000–301191（192 个） |
| 评估 | 只评估最终模型，确定性策略 |

**全部工作量：**

| 类别 | 内容 | 数量 |
|---|---|---|
| 训练 | `ECC`、`SAC_MPC`、`ECC_NO_CORE`、`TD3_MPC` × 种子 30、31、32 | 12 个运行 |
| 评估 | 12 个模型各 192 个初始条件 | 2304 回合 |
| 只评估对照 | `ECC` 的 3 个模型加 `rigid_entry` | 576 回合 |
| 非学习 | `pure`、`c_10_0`、`b_10` 各 192 个初始条件 | 576 回合 |
| 案例重放 | 种子 30 的 `ECC` 与 `SAC_MPC`，各 2 个初始条件 | 4 条轨迹 |

## 1. 开跑前（每台机器一次）

1. 确认没有遗留的 Python 进程。
2. 检出 `41601520`。
3. 运行下面的命令，退出码为 0 才继续（沙箱 171 passed）：

   ```
   python -B -m experiments.final preflight --root <ROOT>
   ```

   `<ROOT>` 是这台机器的结果目录，如本机 `C:\Users\35884\Documents\Spacecraft\过程文件\R2_FORMAL`。

## 2. 训练

```
python -B -m experiments.final train --stage formal --config <CFG> --seed <S> --output <ROOT>\B\TRAIN_<CFG>_s<S>
```

**分配：** 由用户与下层商定，只需遵守三条：
- 每个运行占约 2.3 个核（2 个采集进程加学习进程）；
- 训练优先，空闲核用于评估和非学习策略；
- 同一个运行的训练、评估、轨迹重放在同一台机器上完成。

## 3. 评估与非学习策略（用空闲核，随时可以开始）

**每个模型训练完，立即在同一台机器上评估：**

```
python -B -m experiments.final eval --stage formal --run <ROOT>\B\TRAIN_<CFG>_s<S> --output <ROOT> --workers W
```

**`ECC` 的模型再加一条：**

```
python -B -m experiments.final eval --stage formal --run <ROOT>\B\TRAIN_ECC_s<S> --switch rigid_entry --output <ROOT> --workers W
```

**非学习策略（不需要模型，开跑即可开始，放在任一台机器上）：**

```
python -B -m experiments.final screen --stage formal --policy pure   --output <ROOT> --workers W
python -B -m experiments.final screen --stage formal --policy c_10_0 --output <ROOT> --workers W
python -B -m experiments.final screen --stage formal --policy b_10   --output <ROOT> --workers W
```

`W` 按空闲核数取。评估逐回合确定，并行数不影响结果。中断后用同一命令续跑，已有结果会被跳过。

## 4. 合并、案例重放与提交

1. **汇总到一台机器：** 其他机器的结果带到这台机器的某个目录，每台机器一个，记为 `<ROOT2>`、`<ROOT3>`……需要带的内容：
   - 每个 `B\TRAIN_*` 目录里的 `manifest.json`、`finished.json`、`episodes.json`、`tb\`；
   - 模型：`final.zip`，或者一行哈希的 `final.zip.sha256`（`<SHA-256>  final.zip`），模型本身不必搬运；
   - `C\` 与 `S\` 下的全部结果；
   - 各自的 `PREFLIGHT.json`。

2. **打包：**

   ```
   python -B -m experiments.final pack --stage formal --root <ROOT> --root <ROOT2> [...] --output evidence\R2_FORMAL
   ```

   失败就看 `evidence\R2_FORMAL\CHECK.json`，用第 2、3 节的同一命令补跑缺项，再 pack。

3. **案例重放：** `evidence\R2_FORMAL\CASE_COMMANDS.txt` 里每个模型一条命令，在该模型所在的机器上运行。
   - `<OUT>` 换成那台机器上的目录；
   - 生成的 `traj\*.json` 放进 `evidence\R2_FORMAL\traj\`；
   - 然后再 pack 一次，把轨迹计入哈希。

4. **提交：**
   - 把短检查的 `PAIRED_extended_lattice.json` 复制到 `evidence\R2_FORMAL\`；
   - 只有出现异常时（缺项、报错、中断、重复进程），才在 `evidence\R2_FORMAL\RESULT.md` 末尾写一两句；
   - `git add -f evidence/R2_FORMAL`，提交并推送到科学分支；
   - 把 `RESULT.md` 复制到协作分支 `docs/collaboration/lower/handoffs/r2_formal/RESULT.md`，提交并推送；
   - 打归档标签 `archive/r2-formal-code` 指向 `41601520`。

**不做：** 自制额外核验文件、重算旧证据哈希、跨机器比较比特。

## 5. 参考耗时

- 一个 6000 步的训练约 17–20 小时（开发轮 3000 步约 8.5–10 小时）。
- 总时长 ≈ 一台机器上排队的训练批数 × 约 20 小时，再加评估约 1–2 小时、打包与重放约 1 小时。

## 6. 交付后（上层）

上层读交付中的 `READOUT.json`、`PAIRED.csv` 等，更新事实稿 08，并按 `FORMAL.md` 第 4 节给出全部论文图表数据。泛化蒙特卡洛另写规格，只评估这批最终模型。
