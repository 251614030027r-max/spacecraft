# 执行书：RUN_R2_FB2（第二轮反馈机制开发，修订版；替代 RUN_R2_FB）

**状态：执行中（2026-11-20 起；本地 4 个运行 + 云端 2 个运行）。**

**追加（2026-11-20）：** 试验内容不变，只简化交付。
- 训练和评估照第 1–2 节进行，运行中不拉取。
- 全部完成后，再拉取科学分支 `431073ab` 或更新的提交。这些提交只新增交付命令，训练和评估代码与 `b47eb660` 相同。
- 用下面一条命令交付，替代第 3–4 节：
  ```
  python -B -m experiments.final pack --stage dev --root <本地 R2_FB> --root <云端带回的目录> --output evidence\R2_FB
  ```
- 云端带回什么、提交方式，见 [WORKFLOW.md](../../WORKFLOW.md) 第 2 节。
- 不自制额外核验文件；只有异常才在 RESULT.md 末尾写明。

**依据：** 科学分支 `docs/final_method/FEEDBACK_DEV.md`（第 4 节为修订后的判据）、`docs/final_method/ROUND2.md`。

**与 RUN_R2_FB 的区别：**
- 去掉有条件的第三场 `ECC_EQ`；
- 新增同一模型上的配对机制检验（只评估开关 `nearest_arbitration`）；
- 评估初始条件从 96 个增至 192 个；
- 两场训练可以并行。

RUN_R2_FB 未执行，作废。

## 0. 前提

| 项 | 值 |
|---|---|
| 科学提交 | `b47eb660`（分支 `claude/sac-mpc-coupling-design-ns7g6i`） |
| 阶段 | `dev`：种子 0、1、2；3000 次高层决策；评估初始条件 295300–295491（192 个） |
| 训练配置 | `ECC`（基线）、`ECC_FB`（价值仲裁 + 等价经验） |
| 输出根目录 | `C:\Users\35884\Documents\Spacecraft\过程文件\R2_FB\`，下文记为 `$D` |

**第 0 步：**
1. 确认没有任务在运行，`$D` 不存在；
2. 检出 `b47eb660`，运行 `python -B -m pytest -q`，退出码应为 0（沙箱 171 passed）；
3. 把提交号、退出码、开始时刻、本机逻辑核数写入 `$D\PREFLIGHT.json`；
4. 运行期间不切换分支、不拉取、不改工作树。

## 1. 训练（6 个运行）

```
python -B -m experiments.final train --stage dev --config ECC    --seed <S> --output $D\B\TRAIN_ECC_s<S>
python -B -m experiments.final train --stage dev --config ECC_FB --seed <S> --output $D\B\TRAIN_ECC_FB_s<S>
```

`<S>` 取 0、1、2。

**并行方式：**
- 每个运行占 2 个采集进程，再加 1 个轻量的学习进程；
- 空闲逻辑核 ≥ 12：6 个运行同时启动；
- 否则：先同时跑 3 个 `ECC`，再同时跑 3 个 `ECC_FB`。

**检查：**
- 每个训练目录都必须有 `tb\events.out.tfevents.*`，没有就停止并报告；
- `ECC_FB` 的 `episodes.json` 里，`exec_stats` 应有 `value_arbitrations`、`arbitration_changed`、`equivalent_samples`；`finished.json` 应有 `equivalence_samples`；
- 训练中可以用 `tensorboard --logdir $D\B` 查看，只看不改。

## 2. 评估（每个种子训练完即可开始，9 组 × 192 回合）

```
python -B -m experiments.final eval --stage dev --run $D\B\TRAIN_ECC_s<S>    --output $D --workers W
python -B -m experiments.final eval --stage dev --run $D\B\TRAIN_ECC_FB_s<S> --output $D --workers W
python -B -m experiments.final eval --stage dev --run $D\B\TRAIN_ECC_FB_s<S> --switch nearest_arbitration --output $D --workers W
```

- 第三条是同一 `ECC_FB` 模型改用最近距离规则的配对评估，结果写到 `$D\C\ECC_FB_s<S>_nearest_arbitration\`；
- `W` 按空闲核数取。评估逐回合确定，并行数不影响结果。

## 3. 收尾与读数

```
python -B -m experiments.final check   --stage dev --root $D --output $D\CHECK.json
python -B -m experiments.final resolve --stage dev --root $D --output $D\REGISTRY_RESOLVED.json
python -B -m experiments.r2_dev_readout --root $D --output $D\READOUT
```

- `CHECK.json` 的 `complete` 必须为 true；
- 缺的回合用同一命令补跑，已有结果会被跳过；
- 出现 `ERROR_*.json` 的回合原样交付，并写明；
- `READOUT\SUMMARY.json` 的 `verdict` 按预先登记的规则直接给出（P：方法层面；M：配对机制）。下层只交付，不自行判定。

## 4. 交付

**科学分支** `evidence/R2_FB/`（逐个文件强制添加）：
- `PREFLIGHT.json`、`CHECK.json`、`REGISTRY_RESOLVED.json`；
- `READOUT\SUMMARY.json`、`READOUT\EPISODES.csv`；
- 每个运行的 `manifest.json`、`finished.json`、`episodes.json` 和 `tb\` 下的事件文件；
- `SHA256SUMS`：覆盖上述文件，以及 6 个 `final.zip` 和 1728 个评估结果 JSON（这两类只登记 SHA，不入库）。

**协作分支** `docs/collaboration/lower/handoffs/r2_fb/R2_FB_RESULT.md`，写明：
- 逐种子的捕获、超时、不安全；
- `verdict`（P、M 与结论）；
- 配对翻转数；
- 仲裁改变选择的次数、等价样本数；
- 悬停信号；
- 训练与评估的墙钟时长，以及是否并行；
- 任何异常或补跑。

**本机：** 训练目录保留在 `$D\B\`，正式实验前不删除。

## 5. 预计耗时

- **训练：** 6 个运行并行约 8–9 小时；分两批约 16–18 小时。
- **评估：** 1728 回合；按 12 核约 2–4 小时。
