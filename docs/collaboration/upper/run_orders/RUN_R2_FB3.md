# 执行书：RUN_R2_FB3（第二轮反馈机制开发；替代 RUN_R2_FB2，试验内容不变，交付流程简化）

**状态：撤回（2026-11-20）。** RUN_R2_FB2 已在执行，本书的交付简化已作为 FB2 的追加写入，不单独执行。

**依据：** 科学分支 `docs/final_method/FEEDBACK_DEV.md`（判据在第 4 节）。

**与 FB2 的区别：** 试验内容完全相同，只把开跑前记录和交付改成两条命令。RUN_R2_FB2 未执行，作废。

| 项 | 值 |
|---|---|
| 科学提交 | `3f7b9bec` |
| 阶段 | `dev`：种子 0、1、2；3000 次决策；评估初始条件 295300–295491 |
| 训练配置 | `ECC`、`ECC_FB` |
| 输出根目录 | `C:\Users\35884\Documents\Spacecraft\过程文件\R2_FB\`（`$D`） |

## 交付纪律：只做下面这些

- 不要自制额外的核验文件：
  - 不写 VERIFIED / EXIT_RECEIPTS / SCOPE / 历史对照哈希 / TensorBoard 与原生记录的比对；
  - 不重算旧证据的哈希。
- 判定、完整性、哈希全部由命令生成。
- 只有出现异常时，才在 RESULT.md 末尾写一两句：缺项、报错、中断、重复进程。

## 1. 开跑前（一条命令）

检出 `3f7b9bec`，然后：

```
python -B -m experiments.final preflight --root $D
```

- 退出码为 0：继续；
- 否则停止，把 `$D\PREFLIGHT.json` 交给上层。

`PREFLIGHT.json` 记录提交、工作树是否干净、逻辑核数、pytest 结果。

## 2. 训练

```
python -B -m experiments.final train --stage dev --config ECC    --seed <S> --output $D\B\TRAIN_ECC_s<S>
python -B -m experiments.final train --stage dev --config ECC_FB --seed <S> --output $D\B\TRAIN_ECC_FB_s<S>
```

- `<S>` 取 0、1、2；
- 逻辑核 ≥ 12：6 个同时启动；否则先跑 3 个 `ECC`，再跑 3 个 `ECC_FB`；
- 运行期间不切换分支、不拉取、不改工作树。

## 3. 评估（每个种子训练完即可开始）

```
python -B -m experiments.final eval --stage dev --run $D\B\TRAIN_ECC_s<S>    --output $D --workers W
python -B -m experiments.final eval --stage dev --run $D\B\TRAIN_ECC_FB_s<S> --output $D --workers W
python -B -m experiments.final eval --stage dev --run $D\B\TRAIN_ECC_FB_s<S> --switch nearest_arbitration --output $D --workers W
```

`W` 按空闲核数取。

## 4. 交付（一条命令 + 提交）

在科学检出目录里：

```
python -B -m experiments.final pack --stage dev --root $D --output evidence\R2_FB
```

- **命令失败（不完整）：** 看 `evidence\R2_FB\CHECK.json`，用第 2、3 节的同一命令补跑缺项（已有结果会被跳过），再执行一次 pack。
- **命令成功：**
  1. 如有异常，在 `evidence\R2_FB\RESULT.md` 末尾"异常与补跑"一节写明；
  2. `git add -f evidence/R2_FB`，提交并推送到科学分支；
  3. 把同一份 `RESULT.md` 复制到协作分支 `docs/collaboration/lower/handoffs/r2_fb/RESULT.md`，提交并推送。

**本机：** `$D` 保留，正式实验前不删除。

## 5. 预计耗时

- **训练：** 并行约 8–9 小时；分两批约 16–18 小时。
- **评估：** 1728 回合，按 12 核约 2–4 小时。
- **交付：** 几分钟。
