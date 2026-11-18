# 执行书：RUN_ECC_LE（远程入口候选：先只评估检验，达标即训练）

**状态：用户要求尽快提升捕获率（2026-11-18），可立即执行。**

依据：科学分支 `docs/ecc2/ECC_LE.md`（诊断、改动、判据与采用规则，先于结果入库）。ECC_EVT 未采用，原 ECC 保留。

## 0. 前提

| 项 | 值 |
|---|---|
| 科学提交 | `7ba4c90d`（新开关 `long_entry` 默认关闭；原有配置的行为不变） |
| 第一轮模型 | `$R\B\TRAIN_ECC_CERT_LOOP_s20/21/22`（`$R` = `C:\Users\35884\Documents\Spacecraft\过程文件\FORMAL`） |
| 输出 | 检验写到 `C:\Users\35884\Documents\Spacecraft\过程文件\ECC_LE_PROBE\`（`$P`）；训练与评估写到 `C:\Users\35884\Documents\Spacecraft\过程文件\ECC_LE\`（`$L`） |

**第 0 步：** 确认没有任务在运行；检出 `7ba4c90d`；`python -B -m pytest -q` 退出码应为 0（168 项）；把提交号、退出码、开始时刻写入 `$L\PREFLIGHT.json`。

## A. 只评估检验（预计 2–3 小时）

**配对检查（先做）：** 关闭远程入口，重放种子 20 的前 12 个初始条件。每个结局和回报必须与 `$R\C\ECC_CERT_LOOP_s20_urgent\` 中的结果一致（跨机器只比较结局），否则停止报告。

```
python -B -m experiments.ecc_le_probe --policy $R\B\TRAIN_ECC_CERT_LOOP_s20 --label ECC_s20_off --openings 300000-300011 --output $P\off --workers W --off
```

**检验：** 三个种子依次运行。

```
python -B -m experiments.ecc_le_probe --policy $R\B\TRAIN_ECC_CERT_LOOP_s20 --label ECC_s20 --openings 300000-300095 --output $P\on --workers W
python -B -m experiments.ecc_le_probe --policy $R\B\TRAIN_ECC_CERT_LOOP_s21 --label ECC_s21 --openings 300000-300095 --output $P\on --workers W
python -B -m experiments.ecc_le_probe --policy $R\B\TRAIN_ECC_CERT_LOOP_s22 --label ECC_s22 --openings 300000-300095 --output $P\on --workers W
```

**判据：** 三个 `SUMMARY_*.json` 的捕获合计 ≥ 250/288，且不安全合计 ≤ 1。

- 达到：立即进入 B；
- 达不到：停止，交付检验结果即可。

## B. 训练与评估 ECC_LE（一场，三个种子并行）

```
python -B -m experiments.final train --stage formal --config ECC_LE --seed 20 --output $L\B\TRAIN_ECC_LE_s20
python -B -m experiments.final train --stage formal --config ECC_LE --seed 21 --output $L\B\TRAIN_ECC_LE_s21
python -B -m experiments.final train --stage formal --config ECC_LE --seed 22 --output $L\B\TRAIN_ECC_LE_s22
```

每个种子训练完即评估：

```
python -B -m experiments.final eval --stage formal --run $L\B\TRAIN_ECC_LE_s<S> --condition urgent --output $L --workers W
python -B -m experiments.final eval --stage formal --run $L\B\TRAIN_ECC_LE_s<S> --condition validation --checkpoint-step <K> --output $L --workers W
```

`<K>` 依次取 1000、2000、3000、4000、5000。只做紧急任务和验证曲线。

## 交付

**科学分支：**

- `evidence/ECC_LE_PROBE/`：
  - 三个 SUMMARY 与配对检查的 SUMMARY；
  - 逐回合汇总表 `PROBE_EPISODES.csv`，每行一个回合：种子、初始条件、结局、回报、时间、Δv、远程入口次数；
  - SHA256SUMS。
- 若完成 B：按第一轮格式交付 `evidence/FORMAL/ECC_LE_SESSION6_EPISODES.csv`、`ECC_LE_SESSION6_VERIFIED.json` 和训练日志。

**协作分支** `lower/handoffs/formal/ECC_LE_RESULT.md`：A 的判据结果；若完成 B，再加各种子的结局构成。

是否采用由上层按 `docs/ecc2/ECC_LE.md` 第 5 节判定。
