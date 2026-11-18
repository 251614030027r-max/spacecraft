# 执行书：RUN_ECC_LE（远程入口候选：先只评估检验，达标即训练）

**状态：用户要求尽快提升捕获率（2026-11-18），可立即执行。**

依据：科学分支 `docs/ecc2/METHOD_FINAL.md`（统一方法定义与实验安排）与 `docs/ecc2/ECC_LE.md`（诊断、开发检验、采用规则）。ECC_EVT 未采用，原 ECC 保留。

**2026-11-18 更新：** 检验加入第一轮 "ECC w/o execution feedback"（`SAC_CERT_LOOP`）模型；训练加入强基线 `SAC_LE`（第二场）。

**2026-11-18 加急（用户：尽快出结果、立即训练）：** 取消 A 对训练的门槛，第 0 步后直接开训。

- **逻辑核数 ≥ 16：** `ECC_LE` 与 `SAC_LE` 的 6 条训练同时启动（每条 2 个采集进程）。
- **核数不足 16：** 先启动 `ECC_LE` 的 3 条，结束后立即启动 `SAC_LE` 的 3 条。
- **检验 A：** 照做，但不再决定是否训练。用剩余的核运行（`--workers` 取空闲核数，至少 1），只跑三个 ECC 模型；NOFB 的检验可省略。若核已占满，等训练结束后再跑。
- **评估：** 每个种子训练结束后立即评估，评估命令不变。

## 0. 前提

| 项 | 值 |
|---|---|
| 科学提交 | `d3fd8b5a`（新开关 `long_entry` 默认关闭；原有配置的行为不变；配置 `ECC_LE`、`SAC_LE`） |
| 第一轮模型 | `$R\B\TRAIN_ECC_CERT_LOOP_s20/21/22`、`$R\B\TRAIN_SAC_CERT_LOOP_s20/21/22`（`$R` = `C:\Users\35884\Documents\Spacecraft\过程文件\FORMAL`） |
| 输出 | 检验写到 `C:\Users\35884\Documents\Spacecraft\过程文件\ECC_LE_PROBE\`（`$P`）；训练与评估写到 `C:\Users\35884\Documents\Spacecraft\过程文件\ECC_LE\`（`$L`） |

**第 0 步：** 确认没有任务在运行；检出 `d3fd8b5a`；`python -B -m pytest -q` 退出码应为 0（168 项）；把提交号、退出码、开始时刻写入 `$L\PREFLIGHT.json`。

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
python -B -m experiments.ecc_le_probe --policy $R\B\TRAIN_SAC_CERT_LOOP_s20 --label NOFB_s20 --openings 300000-300095 --output $P\on --workers W
python -B -m experiments.ecc_le_probe --policy $R\B\TRAIN_SAC_CERT_LOOP_s21 --label NOFB_s21 --openings 300000-300095 --output $P\on --workers W
python -B -m experiments.ecc_le_probe --policy $R\B\TRAIN_SAC_CERT_LOOP_s22 --label NOFB_s22 --openings 300000-300095 --output $P\on --workers W
```

先跑三个 ECC，再跑三个 NOFB。判据只看 ECC 的三个；NOFB 只报告，供上层判断评价器输入。

**判据：** 三个 `SUMMARY_ECC_*.json` 的捕获合计 ≥ 250/288，且不安全合计 ≤ 1。

- 达到：立即进入 B；
- 达不到：停止，交付检验结果即可。

## B. 训练与评估（两场，每场一个配置 × 三个种子并行；先 ECC_LE，再 SAC_LE）

```
python -B -m experiments.final train --stage formal --config ECC_LE --seed 20 --output $L\B\TRAIN_ECC_LE_s20
python -B -m experiments.final train --stage formal --config ECC_LE --seed 21 --output $L\B\TRAIN_ECC_LE_s21
python -B -m experiments.final train --stage formal --config ECC_LE --seed 22 --output $L\B\TRAIN_ECC_LE_s22
```

第二场把配置换成 `SAC_LE`，输出 `$L\B\TRAIN_SAC_LE_s<S>`，其余相同。

每个种子训练完即评估：

```
python -B -m experiments.final eval --stage formal --run $L\B\TRAIN_<CFG>_s<S> --condition urgent --output $L --workers W
python -B -m experiments.final eval --stage formal --run $L\B\TRAIN_<CFG>_s<S> --condition validation --checkpoint-step <K> --output $L --workers W
```

`<K>` 依次取 1000、2000、3000、4000、5000。只做紧急任务和验证曲线。

## 交付

**科学分支：**

- `evidence/ECC_LE_PROBE/`：
  - 六个 SUMMARY 与配对检查的 SUMMARY；
  - 逐回合汇总表 `PROBE_EPISODES.csv`，每行一个回合：种子、初始条件、结局、回报、时间、Δv、远程入口次数；
  - SHA256SUMS。
- 若完成 B：按第一轮格式交付 `evidence/FORMAL/ECC_LE_SESSION6_*`、`SAC_LE_SESSION7_*`（EPISODES.csv、VERIFIED.json）和训练日志。

**协作分支** `lower/handoffs/formal/ECC_LE_RESULT.md`：A 的判据结果；若完成 B，再加各种子的结局构成。

是否采用 ECC_LE、是否保留评价器输入，由上层按 `docs/ecc2/METHOD_FINAL.md` 第 4、5 节判定。
