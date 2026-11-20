# 执行书：RUN_R2_FB（第二轮反馈机制开发：执行落地的价值仲裁与执行等价经验）

**状态：** 已写好，**等用户开场指令**。没有明确 GO 不启动训练。

**依据：** 科学分支 `docs/final_method/FEEDBACK_DEV.md`（机制定义、配置、预先登记的判据），`docs/final_method/ROUND2.md`（基线 ECC）。

**这一轮测什么：** 反馈机制（`ECC_FB`）在同样设置下能不能比基线 `ECC` 多捕获。结果决定正式实验的 ECC 用哪个形式；开发数据不进论文。

## 0. 前提

| 项 | 值 |
|---|---|
| 科学提交 | `9d4d1382`（分支 `claude/sac-mpc-coupling-design-ns7g6i`） |
| 阶段 | `dev`：种子 0、1、2；3000 次高层决策；评估初始条件 295300–295395（96 个） |
| 配置 | 第一场 `ECC`（基线），第二场 `ECC_FB`（本机制），第三场 `ECC_EQ`（有条件，见第 3 节） |
| 输出根目录 | `C:\Users\35884\Documents\Spacecraft\过程文件\R2_FB\`，下文记为 `$D` |

**第 0 步：**
1. 确认没有任务在运行，`$D` 不存在；
2. 检出 `9d4d1382`，运行 `python -B -m pytest -q`，退出码应为 0（沙箱 171 passed）；
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

- `W` 按空闲核数取，评估逐回合确定，并行数不影响结果。
- 每个训练目录必须有 TensorBoard 日志 `tb\events.out.tfevents.*`，没有就停止并报告。
- 可以用 `tensorboard --logdir $D\B` 查看，只看不改。

## 2. 第二场：`ECC_FB`

把第 1 节命令里的 `ECC` 换成 `ECC_FB`，输出 `$D\B\TRAIN_ECC_FB_s<S>`，其余相同。

**注意：**
- `ECC_FB` 在关键窗口用学习进程的评价器给候选排序。采集进程会多发"排序请求"，属于正常现象。
- 训练日志 `episodes.json` 的 `exec_stats` 里应出现 `value_arbitrations`、`arbitration_changed`、`equivalent_samples`。
- TensorBoard 里有对应的 `episode/*` 曲线。

## 3. 读数与有条件的第三场

两场都评估完后：

```
python -B -m experiments.r2_dev_readout --root $D --output $D\READOUT
```

看 `READOUT\SUMMARY.json` 的 `verdict`：

| `verdict` | 下一步 |
|---|---|
| `formal_ecc` = `ECC_FB` | 结束，不跑第三场 |
| `next` = `run ECC_EQ` | 按第 1 节的命令跑第三场 `ECC_EQ`（输出 `$D\B\TRAIN_ECC_EQ_s<S>`），评估后重新运行读数命令 |

下层不自行判定，只按表执行；判据见 `FEEDBACK_DEV.md` 第 4 节，事后不改。

## 4. 收尾

```
python -B -m experiments.final check   --stage dev --root $D --output $D\CHECK.json
python -B -m experiments.final resolve --stage dev --root $D --output $D\REGISTRY_RESOLVED.json
python -B -m experiments.r2_dev_readout --root $D --output $D\READOUT
```

`CHECK.json` 的 `complete` 必须为 true。
- 第三场未触发时，`ECC_EQ` 列在 `conditional_not_run`，不影响完整性；
- 缺的回合用同一命令补跑（已有结果会被跳过）；
- 出现 `ERROR_*.json` 的回合原样交付，并在交接中写明。

## 5. 交付

**科学分支** `evidence/R2_FB/`（逐个文件强制添加，不提交整个目录）：
- `PREFLIGHT.json`、`CHECK.json`、`REGISTRY_RESOLVED.json`；
- `READOUT\SUMMARY.json`、`READOUT\EPISODES.csv`；
- 每个运行的 `manifest.json`、`finished.json`、`episodes.json` 和 `tb\` 下的事件文件；
- `SHA256SUMS`：覆盖上述文件，以及各 `final.zip` 和全部评估结果 JSON（这两类只登记 SHA，不入库）。

**协作分支** `docs/collaboration/lower/handoffs/r2_fb/R2_FB_RESULT.md`，写明：
- 各组逐种子的捕获、超时、不安全；
- `verdict`；
- `ECC_FB` 每回合的价值仲裁次数、改变选择的次数、等价样本数；
- 悬停信号；
- 每场墙钟时长，以及任何异常或补跑。

**本机：** 训练目录（含 `final.zip` 与 `tb\`）保留在 `$D\B\`，正式实验前不删除。

## 6. 预计耗时

- 每场（三种子并行，3000 步）约 7–8 小时，加评估 1–2 小时；
- 两场约 18 小时，触发第三场再加约 9 小时。
