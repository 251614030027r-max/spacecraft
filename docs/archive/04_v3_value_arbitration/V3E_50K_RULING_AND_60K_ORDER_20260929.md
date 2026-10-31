# v3e 50k 裁定与 60k 正式评估指令

*2026-09-29，上层窗口。依据：下层夜间交付 `V3E_OVERNIGHT_DELIVERY_20260929`（训练提交 `a03632a`，验证提交 `e2c6f0b`）。*

---

## 1. 裁定：**安全门不构成程序缺陷，放行 60k 正式评估**

下层按执行单第 5 节停在"learned-only 的真值违规回合明显多于 Pure MPC"（3 / 5 / 4 对 0），没有自行设阈值，做法正确。上层核对后的判断：

| 事实（来自 `eval/v3e/early_50k_verification_checklist.json` 与 50k 原始 JSON） | 说明 |
|---|---|
| 每个违规回合都**只有 1 步违规**，就是回合终止的那一步（视场 10 次、走廊 2 次） | 真值判定在第一次越界时结束回合；这是"硬失败"本身，不是持续违规 |
| 违规余量 −0.0001 到 −0.025（归一化） | 贴边失败，不是飞出 |
| 每个违规回合之前都有一次非法进入（圆盘外穿过入口平面） | 失败机理：非法进入后在近场丢视场 |
| 这些回合的 QP 零推力回退基本为 0（只有 262421 的一个回合 12 步） | 不是求解器问题 |
| 262006 在 v3e、v3d、平缓推进上都以同样方式失败 | 仓库早有记录的"视场极限开局"（CLAUDE.md，262006） |
| 所有完成回合零违规；训练健康门全程通过；Pure MPC 复现 37/48 | 实现正确 |

**读法**：Pure MPC 的失败方式是"超时"，学习策略的失败方式是"贴着视场边缘终止"。这是**学习策略单独运行时的行为特征**，不是代码缺陷；而这正是 V3 仲裁要处理的情况——任务效用里硬失败记 −20，价值会把这类状态交还 Pure MPC。

**正式标准不放松**：安全只按预注册规则在**仲裁行**上判：仲裁行的真值违规回合数不得多于 Pure MPC（即 0）。learned-only 行照实报告，不设通过条件。

## 2. 50k 读数（观察，不作结论；48 个正式评估开局）

| 行 | 完成 | 违规回合 | 救回 / 丢失（对 Pure MPC） | 共同完成：多用时 / 多耗 Δv |
|---|---:|---:|---|---|
| Pure MPC | 37 | 0 | — | — |
| 平缓推进 | 40 | 1 | 9 / 6 | +78 s / +1.17 m/s |
| v3d@50k（旧参考实现） | 36 / 38 / 38 | 1 / 2 / 0 | 9/10、9/8、7/6 | +95–109 s / **+2.9–3.2 m/s** |
| v3e@50k | 39 / 32 / 30 | 3 / 5 / 4 | 8/6、3/8、3/10 | +80–102 s / **+0.8–1.4 m/s** |

1. **参考实现修复的效果清楚**：同为 50k，v3e 在共同完成回合上的多耗燃料从 v3d 的 +2.9–3.2 m/s 降到 +0.8–1.4 m/s（降 55%–75%），并且在 5–6 个开局上比 Pure MPC 更省燃料（v3d 基本为 0）。这是 E6 消融要的证据。
2. **互补空间大**（上层用交付 JSON 的探索性计算，未另立产物）：Pure MPC 与 v3e 262420 按开局取能完成者，最多 **45/48**；与 421 / 422 各为 40/48；Pure MPC 与三个 v3e 任一取并集 46/48。这是仲裁的上限，不是结果。
3. **种子差异大**：262420 单独 39/48，421、422 为 32、30。按规则三个种子都要做完整流程，分别判读，不挑种子。
4. 训练 50–60k 段三种子完成率 74% / 81% / 77%，421 与 422 仍在上升；50k 快照不代表 60k。

## 3. 60k 正式评估指令（给下层，在 `D:\py\DRL2_v3e`）

先拉取本文档所在提交（只有文档与 `eval/v3e/` 小文件，训练代码不变）：
```powershell
git pull --ff-only origin claude/sac-mpc-coupling-design-ns7g6i
git diff a362d4398d35405a83dce7f8ba7de19c0e6d9497 HEAD --stat -- . ':!docs' ':!eval' ':!CLAUDE.md'   # 只应出现 experiments/v3e_early_readout.py
```
**Pure MPC 与平缓推进两行直接复用 50k 时的结果**（`eval/v3e/pure_mpc.json`、`eval/v3e/nominal.json`），不重跑。

**A. learned-only @60k × 3**（3 个进程并行）：
```powershell
python -B -m experiments.evaluate_hybrid_policy --episodes 48 --seed 262000 --horizon 35 --parametrization task_state_v3 --phase-time-observation --execution-feedback --adaptive-task --model logs/v3e_262420/final_model.zip --output eval/v3e/262420/learned_only.json
```
（262421、262422 同理。）

**B. M2 价值数据**（与 A 同时开始；每个模型 4 个进程，共 12 个；机器核数不够就排队）：
```powershell
python -B -m experiments.v3_collect_value_data --run-dir logs/v3e_262420 --seeds 270000-270011 --output eval/v3e/262420/m2_a
python -B -m experiments.v3_collect_value_data --run-dir logs/v3e_262420 --seeds 270012-270023 --output eval/v3e/262420/m2_b
python -B -m experiments.v3_collect_value_data --run-dir logs/v3e_262420 --seeds 270024-270035 --output eval/v3e/262420/m2_c
python -B -m experiments.v3_collect_value_data --run-dir logs/v3e_262420 --seeds 270036-270047 --output eval/v3e/262420/m2_d
```
（262421、262422 同理。）试运行（dryrun_50k）的价值**不得复用**。

**C. M3 → M6 → M5**（每个模型的 M2 四块都完成后）：
```powershell
python -B -m experiments.v3_fit_values --data eval/v3e/262420/m2_a.npz eval/v3e/262420/m2_b.npz eval/v3e/262420/m2_c.npz eval/v3e/262420/m2_d.npz --output-dir eval/v3e/262420/values
python -B -m experiments.v3_calibrate_values --run-dir logs/v3e_262420 --values eval/v3e/262420/values --output eval/v3e/262420/m6.json
python -B -m experiments.evaluate_hybrid_policy --episodes 48 --seed 262000 --horizon 35 --parametrization task_state_v3 --phase-time-observation --execution-feedback --adaptive-task --model logs/v3e_262420/final_model.zip --arbiter one_way --values eval/v3e/262420/values --output eval/v3e/262420/arbitrated.json
```
- M6 为 `PASS` 或 `INCONCLUSIVE` 都跑 M5（`INCONCLUSIVE` 交回时单列，不得声称已校准）；`STOP` 则该模型不跑 M5，报上层。

**D. 判读**：
```powershell
python -B -m experiments.v3_readout --pure-mpc eval/v3e/pure_mpc.json --model 262420=eval/v3e/262420/learned_only.json,eval/v3e/262420/arbitrated.json --model 262421=eval/v3e/262421/learned_only.json,eval/v3e/262421/arbitrated.json --model 262422=eval/v3e/262422/learned_only.json,eval/v3e/262422/arbitrated.json --output eval/v3e/readout.json
python -B -m experiments.v3e_early_readout --pure-mpc eval/v3e/pure_mpc.json --nominal eval/v3e/nominal.json --v3e eval/v3e/262420/learned_only.json eval/v3e/262421/learned_only.json eval/v3e/262422/learned_only.json eval/v3e/262420/arbitrated.json eval/v3e/262421/arbitrated.json eval/v3e/262422/arbitrated.json --output eval/v3e/table_60k.json
```

**本轮停止条件**（其余情况一律跑完）：任何命令报错；M2 数据里出现 NaN；M5 仲裁行的 QP 零推力回退明显异常（每个模型 > 50 步）。**learned-only 或仲裁行的违规数不再作为中途停止条件**——它们是结果，按预注册规则在判读中处理。

## 4. 交回内容

1. `eval/v3e/<seed>/` 下全部 JSON（learned_only、m3_report、m6、arbitrated）、`readout.json`、`table_60k.json` 及 SHA-256；
2. 两个判读脚本打印的表；
3. 每个模型 M6 的 `summary` 段；
4. M2 的 `.npz`、`values/*.pt`、模型 ZIP 不交，只报路径与 SHA-256。

## 5. 50k 产物登记

入库（小文件）：`eval/v3e/early_50k_readout.json`（`72d6548c…ef06`）、`eval/v3e/early_50k_verification_checklist.json`（`1ec65936…7c`）、`eval/v3e/final_training_audit_60k.json`（`1607e5d0…b0`）。
留在用户机器（`D:\py\DRL2_v3e\eval\v3e\`，SHA-256 见交付包 `50k/JSON_SHA256.txt`）：`pure_mpc.json`（`0a849ad8…e1`）、`nominal.json`（`29aae24b…18d41`）、v3e 50k ×3、v3d 50k ×3。
