# 执行书草案：RUN_FORMAL（EACH 正式实验；**草案，未授权，不得启动**）

**状态：草案。** 预实验交付并经用户确认后，由用户单独授权；授权时冻结 SHA、批次与删减（若有）一并写入本页并改为"已批准"。方法、配置、开局、统计口径都已在 `docs/final_method/` 锁定，本页只排机时。

## 规模与机时

| 项 | 内容 |
|---|---|
| 训练组 | `ABLATION_MATRIX.md` 表 1 全部 6 组：EACH、SAC、SAC_ENG、DUP、NO_GEOM、RIGID |
| 种子 | 20、21、22（训练开局 21,000,000–23,999,999，从未用过） |
| 预算 | 每条 5000 决策；检查点 1000 / 2000 / 3000 / 4000 只报告学习过程；主指标取 5000 步模型 |
| 评估开局 | 300000–300095（96 个，正式块，从未访问），按开局配对 |
| 固定策略 | pure、gstar、rally、b_10、c_10_0、c_10_-5、c_6_5；除 pure 外都另跑刚性承诺 |
| 配对窗口消融 | EACH 三个 5000 步模型另跑刚性承诺 |

**机时估计**（按下层实测的整机吞吐约 1040 决策 / 小时：T2.12、T2.17、T2.19 均为 4 条并行，2000 步 6.0–7.7 小时）：
- 训练：18 条 × 5000 = 90,000 决策，约 87 小时；
- 评估：（18 + 3）个模型 × 96，加 13 组固定策略 × 96，约 3300 个回合；按实测约 275 回合 / 小时，约 12 小时。

分三批，每批 6 条并行：

| 批 | 训练 | 估计 |
|---|---|---|
| 1 | EACH、SAC × 种子 20、21、22 | 约 29 小时 |
| 2 | SAC_ENG、DUP × 种子 20、21、22 | 约 29 小时 |
| 3 | NO_GEOM、RIGID × 种子 20、21、22 | 约 29 小时 |
| 4 | 全部评估、几何扫描、轨迹、出图 | 约 12 小时 |

- 每批结束后只做完整性检查（`experiments.final check`），不看评估结果、不做任何选择；评估统一放在第 4 批。
- 机时不足需要删减时，必须**授权前**由用户写明删哪一组，不能训练后再选。

## 命令（与预实验相同，阶段换成 formal）

```powershell
& $py -B -m experiments.final train  --stage formal --config <CONFIG> --seed <20|21|22> --output "$Out\B\TRAIN_<CONFIG>_s<SEED>"
& $py -B -m experiments.final eval   --stage formal --run "$Out\B\TRAIN_<CONFIG>_s<SEED>" --label <CONFIG>_s<SEED> --output "$Out\C\<CONFIG>_s<SEED>" --workers 10
& $py -B -m experiments.final eval   --stage formal --run "$Out\B\TRAIN_EACH_s<SEED>" --label EACH_s<SEED>_rigid --output "$Out\C\EACH_s<SEED>_rigid" --rigid --workers 10
& $py -B -m experiments.final screen --stage formal --policy <POLICY> --output "$Out\S\<POLICY>" [--rigid → "$Out\S\<POLICY>_rigid"] --workers 10
& $py -B -m experiments.final geoscan --openings 294200-294211 --output "$Out\G" --workers 10
& $py -B -m experiments.final traj --entry EACH="$Out\B\TRAIN_EACH_s20" --entry SAC="$Out\B\TRAIN_SAC_s20" --entry pure=pure --entry rally=rally --openings 294200,294201,294202 --output "$Out\T"
& $py -B -m experiments.final check   --stage formal --root $Out --output "$Out\CHECK.json"
& $py -B -m experiments.final resolve --stage formal --root $Out --output "$Out\REGISTRY_RESOLVED.json"
& $py -B -m experiments.final_figures --stage formal --root $Out --output "$Out\figures" --geoscan "$Out\G" --traj "$Out\T"
```

## 统计与报告（预先登记）

**主指标：**
- 紧急捕获率、总体捕获率、unsafe、超时 / 资源耗尽、真实回报、训练曲线 AUC；
- 时间与 Δv 只在双方都捕获的开局上比较，并单列只有一方捕获的开局数。

**统计单位：** 开局是配对单位，种子是训练重复。报告种子均值、范围与逐开局配对差，附中位数、去掉前 2 名后的均值，以及紧急 / 经济、截止期限分档（< 160、160–180、≥ 180 s，仅用于分析）。

**图表：** 训练曲线为 3 个种子的真实滚动结果加汇总带；正式评估用每种子散点加配对差；窗口表用同一开局的差异；失败开局按事先规则选例，不按结果挑选。

**有效性：** 与预实验相同的六项。
