# 执行书：RUN_ECC2_GATE0（ECC 第二版离线可行性审查）

**状态：用户已认可方向并要求优先推进离线审查（2026-11-18）。只做分析与分支重放，不训练、不改方法代码。**

依据：科学分支 `docs/ecc2/GATE0.md`（规格与判据，先于结果入库）；脚本 `experiments/ecc2_gate0.py`。

## 0. 前提

| 项 | 值 |
|---|---|
| 科学提交 | `868fe5e6` 或之后的科学分支 HEAD（相对 `e8474e21` 只新增文档、只读脚本与测试，方法代码不变） |
| 第一轮正式实验 | 已结束：四个训练组的结果此后作为开发证据。剩下的 TD3–MPC、ECC w/o 旋转引导，以及尚未运行的固定策略评估，都不再运行（用户 2026-11-17）。第一轮的本地目录原样保留，不删除 |
| 输入 | 第一轮根目录 `C:\Users\35884\Documents\Spacecraft\过程文件\FORMAL\`（记作 `$R`）：`$R\C\` 下的审计评估，`$R\B\TRAIN_ECC_CERT_LOOP_s20/21/22\final.zip` |
| 输出 | `C:\Users\35884\Documents\Spacecraft\过程文件\ECC2_GATE0\`（记作 `$G`） |

**第 0 步：** 确认没有任务在运行；在 `D:\py\DRL2` 拉取并检出科学分支 HEAD；运行 `python -B -m pytest -q`，退出码应为 0（162 项）。把 `git rev-parse HEAD`、pytest 退出码和本机逻辑核数写入 `$G\PREFLIGHT.json`。

## 1. A 部分：已有日志（几分钟）

```
python -B -m experiments.ecc2_gate0 logs --root $R --output $G\A_LOGS.json
```

## 2. B 部分：分支重放（预计 4–8 小时）

三个种子依次运行，写到同一个输出目录。`--workers` 取逻辑核数减 1，最多 8。

```
python -B -m experiments.ecc2_gate0 branch --policy $R\B\TRAIN_ECC_CERT_LOOP_s20 --label ECC_s20 --openings 294000-294063 --output $G\B --workers W
python -B -m experiments.ecc2_gate0 branch --policy $R\B\TRAIN_ECC_CERT_LOOP_s21 --label ECC_s21 --openings 294000-294063 --output $G\B --workers W
python -B -m experiments.ecc2_gate0 branch --policy $R\B\TRAIN_ECC_CERT_LOOP_s22 --label ECC_s22 --openings 294000-294063 --output $G\B --workers W
python -B -m experiments.ecc2_gate0 report --output $G\B --result $G\B_REPORT.json --table $G\B_OUTCOMES.csv
```

- **中断：** 中断后用同样的命令重跑，已完成的文件会被跳过。
- **失败的任务：** 写在 `$G\B\ERRORS.json`，命令以非零退出。不要删除失败记录、不要挑选，原样交付并说明。
- **约束：** 运行期间不切换分支、不拉取、不改工作区；不使用正式初始条件段（300000 以上）；不使用新种子。

## 3. 交付

- **科学分支** `evidence/ECC2_GATE0/`，逐个文件强制添加：
  - `A_LOGS.json`、`B_REPORT.json`、`B_OUTCOMES.csv`、`PREFLIGHT.json`；
  - 若有失败，加 `ERRORS.json`；
  - `SHA256SUMS`（覆盖以上文件，以及 `$G\B\base\`、`$G\B\branch\` 下全部 JSON）；
  - `README.md`：命令、开始 / 结束时刻、核数、墙钟、失败数。
- **不提交** `base\`、`branch\` 原始目录，原地保留，按 SHA256SUMS 可核对。
- **协作分支** `lower/handoffs/` 写一份交付说明（科学提交号、B0–B4 是否通过、异常）。
- **判读：** 结论由上层按 `docs/ecc2/GATE0.md` 第 3 节的判据作出，下层只报告数字，不调整判据。
