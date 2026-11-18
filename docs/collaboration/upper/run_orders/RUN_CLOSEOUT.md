# 执行书：RUN_CLOSEOUT（收尾的两组训练与参考线）

**状态：** 等上层对 ECC_LE / SAC_LE 的判定写入本页第 0 节后执行；判定前不得启动。依据：科学 `docs/ecc2/METHOD_FINAL.md` 第 7 节。

## 0. 判定（上层填写）

| 项 | 值 |
|---|---|
| 最终方法 | 待定：`ECC_LE` / `SAC_LE` 结构 / 原 ECC |
| 旋转引导消融配置 | 待定：`ECC_LE_NOGEO` / `SAC_LE_NOGEO` / `ECC_CERT_LOOP_NOGEO` |
| 科学提交 | `de078417` 或之后（与当前训练所用的 `d3fd8b5a` 对已有配置行为相同） |

## 1. 训练两场（每场一个配置 × 种子 20、21、22 并行）

- 第一场：`TD3_C`；
- 第二场：第 0 节确定的旋转引导消融配置。

命令格式与 RUN_ECC_LE 相同，输出写到 `过程文件\ECC_LE\B\TRAIN_<CFG>_s<S>`，与 ECC_LE / SAC_LE 放在同一个根目录。每个种子训练完立即评估紧急任务 300000–300095；`TD3_C` 另评估验证曲线（检查点 1000–5000）。

## 2. 参考线（与学习方法同一批初始条件，可与训练并行）

```
python -B -m experiments.final screen --stage formal --policy pure  --condition urgent --output 过程文件\ECC_LE --workers W
python -B -m experiments.final screen --stage formal --policy gstar --condition urgent --output 过程文件\ECC_LE --workers W
```

## 3. 交付

- **科学分支：** 按第一轮格式交付 `evidence/FORMAL/<CFG>_SESSION<N>_EPISODES.csv`、`VERIFIED.json` 与训练日志；参考线交付 `evidence/FORMAL/REFERENCE_URGENT.csv`（每行一个回合：策略、初始条件、结局、回报、时间、Δv）。
- **协作分支：** `lower/handoffs/formal/CLOSEOUT_RESULT.md`。

交付后上层统一出读数，更新论文事实稿和作图数据。
