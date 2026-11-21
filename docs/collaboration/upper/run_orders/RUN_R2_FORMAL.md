# 执行书：RUN_R2_FORMAL（第二轮正式实验：论文最终数据）

**状态：** 等用户开场指令。

**依据：** 科学分支 `docs/final_method/FORMAL.md`：
- 组、协议、预登记的报告量、案例规则、图表来源；
- 第 8 节：按整组分工，以及只影响 ECC 的短检查。

交付方式见 [WORKFLOW.md](../../WORKFLOW.md)。

| 项 | 值 |
|---|---|
| 科学提交 | `c6f6cc3c`（本机与云端必须相同；短检查结束后不换提交） |
| 阶段 | `formal`：种子 30、31、32；6000 次决策；测试初始条件 301000–301191（192 个） |
| 训练（12 个运行） | ECC 位（`ECC` 或 `ECC_FINE`，由短检查决定）、`SAC_MPC`、`ECC_NO_CORE`、`TD3_MPC`，各 3 个种子 |
| 只评估 | ECC 位的 3 个模型加 `rigid_entry` |
| 非学习 | `pure`、`c_10_0`、`b_10` |
| 结果目录 | 本机 `C:\Users\35884\Documents\Spacecraft\过程文件\R2_FORMAL\`（`$L`）；云端 `<云端>/R2_FORMAL`（`$C`） |

## 0. 开跑前（两台机器各一次）

1. 确认没有遗留的 Python 进程。
2. 检出 `c6f6cc3c`。
3. 本机运行 `python -B -m experiments.final preflight --root $L`，云端运行 `... --root $C`。退出码为 0 才继续（沙箱 171 passed）。

## 1. 分工：每台机器负责完整的组

| 机器 | 立即开始 | 之后 |
|---|---|---|
| 云端 | `SAC_MPC` s30、s31、s32 与 `TD3_MPC` s30、s31、s32，按云端可并行数排队 | 每个模型训练完，在云端评估 |
| 本机 | ① 第 2 节的短检查（约 1 小时）；② 同时开 `ECC_NO_CORE` s30、s31、s32；③ 非学习策略用空闲核 | 短检查定案后开 ECC 位 s30、s31、s32 |

- 每个运行 2 个采集进程，约占 2.3 个核。本机 12 核最多同时 5 个运行；短检查期间同时跑 3 个 `ECC_NO_CORE`。
- 每台机器只评估自己训练的模型。

**训练命令：**

```
python -B -m experiments.final train --stage formal --config <CFG> --seed <S> --output <ROOT>\B\TRAIN_<CFG>_s<S>
```

## 2. 短检查：细格机会（本机种子 0、1 + 云端种子 2；只评估，不训练）

用反馈机制开发轮已训练的 `ECC` 模型。它们的已有结果在本机 `R2_FB` 与云端，作为配对基线，不重跑。

```
python -B -m experiments.final eval --stage dev --run <R2_FB>\B\TRAIN_ECC_s<S> --switch extended_lattice --output <R2_FB> --workers W
```

- `<S>`：本机 0、1，云端 2。
- 结果写到 `<R2_FB>\C\ECC_s<S>_extended_lattice\`。
- 云端把 `C\ECC_s2\` 和 `C\ECC_s2_extended_lattice\` 带回本机目录 `$FB_CLOUD`。

然后在本机运行：

```
python -B -m experiments.r2_dev_readout --root <本机 R2_FB> --root $FB_CLOUD --paired-switch extended_lattice --output <本机 R2_FB>\READOUT
```

读 `PAIRED_extended_lattice.json` 的 `formal_ecc`：

| `formal_ecc` | ECC 位用的配置 |
|---|---|
| `ECC_FINE` | `ECC_FINE` |
| `ECC` | `ECC` |

判据事先写死（`FORMAL.md` 第 8 节），下层只按结果执行。把这个 JSON 一并放进正式交付（第 4 节）。

## 3. 空闲核与评估

**非学习策略（本机）：**

```
python -B -m experiments.final screen --stage formal --policy pure   --output $L --workers 2
python -B -m experiments.final screen --stage formal --policy c_10_0 --output $L --workers 2
python -B -m experiments.final screen --stage formal --policy b_10   --output $L --workers 2
```

**评估（每个模型训练完立即在它所在的机器上做）：**

```
python -B -m experiments.final eval --stage formal --run <ROOT>\B\TRAIN_<CFG>_s<S> --output <ROOT> --workers W
```

ECC 位的模型再加一条：

```
python -B -m experiments.final eval --stage formal --run $L\B\TRAIN_<ECC位配置>_s<S> --switch rigid_entry --output $L --workers W
```

评估逐回合确定，并行数不影响结果。全部训练结束后，剩余评估用满所有核。

## 4. 合并交付（本机）

1. **云端带回到本机 `$CB`：**
   - 每个 `B\TRAIN_*` 目录里的 `manifest.json`、`finished.json`、`episodes.json`、`tb\`，以及一行哈希的 `final.zip.sha256`（`<SHA-256>  final.zip`），模型不必回传；
   - 云端 `C\` 下的全部评估结果。
2. **打包：**

   ```
   python -B -m experiments.final pack --stage formal --root $L --root $CB --output evidence\R2_FORMAL
   ```

   失败就看 `CHECK.json`，补跑缺项后再 pack。
3. **案例重放：** `evidence\R2_FORMAL\CASE_COMMANDS.txt` 里有两条命令。
   - ECC 那条在本机跑，SAC–MPC 那条在云端跑；`<OUT>` 换成各自的输出目录；
   - 云端把 `traj\` 带回，放到 `evidence\R2_FORMAL\traj\`；
   - 然后再 pack 一次，把轨迹计入哈希。
4. **提交：**
   - 把短检查的 `PAIRED_extended_lattice.json` 复制到 `evidence\R2_FORMAL\`；
   - 只有出现异常时，才在 `RESULT.md` 末尾写一两句；
   - `git add -f evidence/R2_FORMAL`，提交并推送；
   - 把 `RESULT.md` 复制到协作分支 `docs/collaboration/lower/handoffs/r2_formal/RESULT.md`，提交并推送；
   - 归档标签 `archive/r2-formal-code` 指向 `c6f6cc3c`。

**不做：** 自制额外核验文件、重算旧证据哈希、跨机器比较比特。

## 5. 预计耗时

- 每个 6000 步的训练约 17–20 小时。
- **云端：** 6 个运行。能同时跑 6 个，约 20 小时；只能同时跑 3 个，约 40 小时；只能同时跑 2 个，约 60 小时。
- **本机：** 无核心 3 个立即开，ECC 位 3 个在短检查后（约 1 小时）开；两组同时跑约 20 小时，评估用空闲核。
- 总时长取决于云端可并行数：云端只能同时跑 2 个时，建议把 `TD3_MPC` 的一个种子移回本机。
- 训练结束后，评估约 1–2 小时，打包与重放约 1 小时。

## 6. 交付后（上层）

上层读 `READOUT.json` 等，更新事实稿 08，按 `FORMAL.md` 第 4 节给出全部图表数据。泛化蒙特卡洛另写规格，只评估这批最终模型。
