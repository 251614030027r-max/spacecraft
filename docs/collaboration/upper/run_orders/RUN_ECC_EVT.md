# 执行书：RUN_ECC_EVT（ECC 最终版开发训练，今晚一场）

**状态：用户要求今晚开跑（2026-11-18）。RUN_ECC2_GATE0 停止，不再执行。**

依据：科学分支 `docs/ecc2/ECC_EVT.md`（算法与采用规则，先于结果入库）。

## 0. 前提

| 项 | 值 |
|---|---|
| 科学提交 | `017f3a2e`（新配置 `ECC_EVT`；原有配置的代码路径不变） |
| 输出根目录 | `C:\Users\35884\Documents\Spacecraft\过程文件\ECC_EVT\`（记作 `$E`）。**不要写进第一轮的 `$R`**：每个根目录只能有一个科学提交 |
| 本机设置 | 同第一轮：高性能电源、禁止睡眠与自动重启、杀毒排除 |

**第 0 步：**

1. 确认没有任务在运行（停掉第 0 关的残留进程）；
2. 在 `D:\py\DRL2` 拉取科学分支并检出 `017f3a2e`，工作区干净；
3. `python -B -m pytest -q`，退出码应为 0（167 项）；
4. 把 `git rev-parse HEAD`、pytest 退出码、开始时刻写入 `$E\PREFLIGHT.json`。

之后到交付前不切换、不拉取、不改工作区。

## 1. 训练（三个种子并行，与第一轮每场相同）

```
python -B -m experiments.final train --stage formal --config ECC_EVT --seed 20 --output $E\B\TRAIN_ECC_EVT_s20
python -B -m experiments.final train --stage formal --config ECC_EVT --seed 21 --output $E\B\TRAIN_ECC_EVT_s21
python -B -m experiments.final train --stage formal --config ECC_EVT --seed 22 --output $E\B\TRAIN_ECC_EVT_s22
```

每个种子 5000 次决策，2 个采集进程（注册表默认），检查点 1000–4000，最终模型 5000。

## 2. 评估（每个种子训练完即评估）

```
python -B -m experiments.final eval --stage formal --run $E\B\TRAIN_ECC_EVT_s<S> --condition urgent --output $E --workers W
python -B -m experiments.final eval --stage formal --run $E\B\TRAIN_ECC_EVT_s<S> --condition validation --checkpoint-step <K> --output $E --workers W
```

- `<K>` 依次取 1000、2000、3000、4000、5000；
- 只评估紧急任务（300000–300095）和验证曲线（295200–295223）；
- 不评估经济、自然混合，不加任何执行开关。

## 3. 交付（与第一轮第 1–4 场相同的格式）

**科学分支 `evidence/FORMAL/`：**

- `ECC_EVT_SESSION5_EPISODES.csv`：列与 `ECC_SESSION1_EPISODES.csv` 相同；
- `ECC_EVT_SESSION5_VERIFIED.json`：结构与 `ECC_SESSION1_VERIFIED.json` 相同；
- `B/TRAIN_ECC_EVT_s20/21/22/` 下的 `episodes.json`、`manifest.json`；
- SHA256SUMS 追加以上文件，以及本地模型与逐回合文件的摘要。

**协作分支 `lower/handoffs/formal/ECC_EVT_SESSION5_RESULT.md`：** 各种子紧急任务的结局构成、墙钟时间、异常。

**判读：** 是否采用由上层按 `docs/ecc2/ECC_EVT.md` 第 5 节判定。下层只报告数字。
