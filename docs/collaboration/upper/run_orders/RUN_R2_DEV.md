# 执行书：RUN_R2_DEV（第二轮开发检查：新势函数不伤性能）

**状态：** 已写好，**等用户开场指令**。没有明确 GO 不启动训练。

**依据：** 科学分支 `docs/final_method/ROUND2.md`（第 1–3 节是改动，第 4 节是本检查与判据）。

**这次检查是什么：**
- 第二轮的清理与方法收敛已经实现。具体是：去掉评价器输入 z、只用限时任务、17 项观测、协调统一成一条规则（与第一稿 ECC_LE 的执行逐位一致，`evidence/R2_EQUIV/`），以及新的机会可达性势函数。
- 本检查确认新势函数不会大幅损害性能。A 组（`ECC`）同时是之后反馈机制开发的基线。
- 不进论文，不是消融归因。

## 0. 前提

| 项 | 值 |
|---|---|
| 科学提交 | `3a8fc970`（分支 `claude/sac-mpc-coupling-design-ns7g6i`） |
| 阶段 | `dev`（注册表）：种子 0、1、2；3000 次高层决策；评估初始条件 295300–295395（96 个） |
| 配置 | 第一场 `ECC`（新势函数），第二场 `ECC_LEAD24`（第一稿势函数），其余完全相同 |
| 输出根目录 | `C:\Users\35884\Documents\Spacecraft\过程文件\R2_DEV\`，下文记为 `$D` |

**第 0 步：**
1. 确认没有任务在运行，`$D` 不存在；
2. 检出 `3a8fc970`，运行 `python -B -m pytest -q`，退出码应为 0（沙箱 159 passed）；
3. 把提交号、退出码、开始时刻写入 `$D\PREFLIGHT.json`；
4. 运行期间不切换分支、不拉取、不改工作树。

## 1. 第一场：`ECC`（三个种子并行，每个种子训练完立即评估）

```
python -B -m experiments.final train --stage dev --config ECC --seed 0 --output $D\B\TRAIN_ECC_s0
python -B -m experiments.final train --stage dev --config ECC --seed 1 --output $D\B\TRAIN_ECC_s1
python -B -m experiments.final train --stage dev --config ECC --seed 2 --output $D\B\TRAIN_ECC_s2
```

每个种子训练完：

```
python -B -m experiments.final eval --stage dev --run $D\B\TRAIN_ECC_s<S> --output $D --workers W
```

- `W` 按空闲核数取，评估是逐回合确定的，并行数不影响结果。
- 每个训练目录都必须有 TensorBoard 日志 `tb\events.out.tfevents.*`。没有就停止，报告上层。
- 训练中途可以用 `tensorboard --logdir $D\B` 查看，只看不改。

## 2. 第二场：`ECC_LEAD24`

把第 1 节命令里的 `ECC` 换成 `ECC_LEAD24`，输出 `$D\B\TRAIN_ECC_LEAD24_s<S>`，其余相同。

## 3. 收尾

```
python -B -m experiments.final check   --stage dev --root $D --output $D\CHECK.json
python -B -m experiments.final resolve --stage dev --root $D --output $D\REGISTRY_RESOLVED.json
python -B -m experiments.r2_dev_readout --root $D --output $D\READOUT
```

`CHECK.json` 的 `complete` 必须为 true。否则：
- 缺的回合用同一命令补跑（已存在的结果会被跳过）；
- 有 `ERROR_*.json` 的回合，原样交付并在交接里写明。

## 4. 判据（预先登记，事后不改；`ROUND2.md` 第 4 节）

- **采用新势函数：** 捕获(ECC) ≥ 捕获(ECC_LEAD24) − 10，且 不安全(ECC) ≤ 不安全(ECC_LEAD24) + 1；
- **否则：** 正式实验用第一稿势函数；
- **判定方式：** `READOUT\SUMMARY.json` 的 `verdict` 按此规则直接给出；下层只交付，不自行判定。

## 5. 交付

**科学分支** `evidence/R2_DEV/`（逐个文件强制添加，不提交整个目录）：
- `PREFLIGHT.json`、`CHECK.json`、`REGISTRY_RESOLVED.json`；
- `READOUT\SUMMARY.json`、`READOUT\EPISODES.csv`；
- 每个运行的 `manifest.json`、`finished.json`、`episodes.json` 和 `tb\` 下的事件文件；
- `SHA256SUMS`：覆盖上述全部文件，以及 6 个 `final.zip` 和 576 个评估结果 JSON（这两类只登记 SHA，不入库）。

**协作分支** `docs/collaboration/lower/handoffs/r2_dev/R2_DEV_RESULT.md`，写明：
- 两组逐种子的捕获、超时、不安全；
- `verdict`；
- 训练是否出现悬停信号（`hover_signal`）；
- 每场的墙钟时长；
- 任何异常或补跑。

**本机：** 6 个训练目录（含 `final.zip` 与 `tb\`）保留在 `$D\B\`。A 组是反馈机制开发的基线，不删除。

## 6. 预计耗时

- 第一轮 ECC_LE 的 5000 步约 12 小时（三个种子并行）。3000 步约 7–8 小时。
- 评估 3 × 96 回合，约 1–2 小时。
- 两场合计约 18 小时。
