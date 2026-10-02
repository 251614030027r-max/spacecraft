# v3e 60k 校准结果裁定与"价值第二轮"指令

*2026-09-29，上层窗口。依据：下层交付 `V3E_60K_CALIBRATION_UPPER_FEEDBACK_20260929`（评估提交 `5a2abe6`）。*
*本文件在任何第二轮数据产生之前提交；其中的数据量、种子块、判定门槛都在此写死。*

---

## 1. 结果读数

**只用学习策略（60k，48 个正式开局，Pure MPC 37/48）：**

| 模型 | 完成 | 救回 / 丢失 | 违规回合 | 共同完成：多用时 / 多耗 Δv |
|---|---:|---|---:|---|
| 262420 | **42** | 8 / 3 | 2 | +90 s / +1.12 m/s |
| 262421 | 33 | 3 / 7 | 3 | +91 s / +0.92 m/s |
| 262422 | **38** | 7 / 6 | 3 | +90 s / +1.28 m/s |
| 平缓推进（对照） | 40 | 9 / 6 | 1 | +78 s / +1.17 m/s |

两个模型单独运行已不低于 Pure MPC；互补性（救回与丢失同时存在）在三个模型上都稳定出现。

**M6 校准（校准块 262100–262111）：262420 PASS（33/36，91.7%，误选 0）；262421 STOP（48.6%）；262422 STOP（53.1%）。**
下层按规则只让 262420 进入 M5，做法正确。

## 2. 诊断：问题在 V_L 的数据量，不在方法

| 证据 | 含义 |
|---|---|
| V_L 在留出开局上的 R² 为 −0.39 / −10.8 / −0.43；V_B 为 0.28 / 0.46 / 0.53 | V_L 在新开局上不如直接猜平均值；V_B 正常 |
| 判错的检查点两头都错：学习策略实际失败处估得很高（262421 的 262104：估 +17、实际 −43），实际成功处估得偏低（估 0–7、实际 +12～+18） | V_L 分不出"这个开局学习策略会成还是会败"，只给出居中的数 |
| V_L 各头在第 0–8 轮就停止改进（袋外误差最好点在开头） | 网络几乎没学到可泛化的东西 |
| V_L 的训练数据只有 40 个"学习策略全程"回合；策略是确定性的，同一回合所有状态共享一个结局 | **只有 40 个独立的成/败标签**，去学 42 维状态上的判断，样本太少 |
| V_B 有 Pure MPC 全程 40 条 + 交回探针 80 条，结局也更好预测 | 所以 V_B 正常 |

**结论**：这是第二步（学价值）数据不足，不是策略、接口或仲裁规则的问题，也不需要重新训练 SAC。补 V_L 的独立结局数据即可。

**按预注册规则的现状**：总体判定要求"至少两个模型两层都通过"。本轮只有一个模型能进 M5，所以**本轮按规则不能宣称方法成立**；这不是"方法无效"，而是价值估计还没达标。

## 3. 价值第二轮（修订 M2–M6，策略不动）

**修订内容（全部模型统一执行，写死）：**

1. **补 V_L 数据**：每个模型在新种子块 **271000–271191**（192 个开局）上各采 1 条"学习策略全程"回合（`--learned-only`），使 V_L 的独立结局从 40 条增加到 200 条以上。原 M2 数据全部保留、一起使用；
2. **留出集**：`270040-270047,271160-271191`（原 8 个 + 新 32 个），只用于报告 MAE / R²；
3. **拟合方法不变**：同样的 5 头集成、同样的网络、同样按任务效用拟合。不换网络、不调超参——只改数据量，这样结果的变化可以归因于数据；
4. **校准改用新的独立校准块 262112–262123**（12 个开局，检查点 0/10/20）。旧校准块 262100–262111 已被第一轮看过，不能再用来判第二轮；
5. **判定门不变**：决定性检查点少于 5 为 INCONCLUSIVE；至少 5 个且符号一致率 ≥ 80% 为 PASS；否则 STOP；
6. **仲裁最终结果统一用第二轮价值**：三个模型都按第二轮价值重新判 M6、跑 M5。262420 正在跑的第一轮 M5 继续跑完，作为"第一轮价值"的补充结果报告，不作主结果；
7. 总体判读规则不变（`docs/V3_METHOD_EXECUTION_ORDER_20260924.md` 第 6 节 + 基线门槛为同代码 Pure MPC 37/48）。

## 4. 给下层的命令（在 `D:\py\DRL2_v3e`）

先拉取（只新增一个采集选项、测试与文档；训练代码不变）：
```powershell
git pull --ff-only origin claude/sac-mpc-coupling-design-ns7g6i
```
262420 的第一轮 M5 继续运行，不要中断。

**A. 补采 V_L 数据**（每个模型 4 个进程，共 12 个；与 262420 的 M5 并行）：
```powershell
python -B -m experiments.v3_collect_value_data --run-dir logs/v3e_262420 --seeds 271000-271047 --learned-only --output eval/v3e/262420/m2x_a
python -B -m experiments.v3_collect_value_data --run-dir logs/v3e_262420 --seeds 271048-271095 --learned-only --output eval/v3e/262420/m2x_b
python -B -m experiments.v3_collect_value_data --run-dir logs/v3e_262420 --seeds 271096-271143 --learned-only --output eval/v3e/262420/m2x_c
python -B -m experiments.v3_collect_value_data --run-dir logs/v3e_262420 --seeds 271144-271191 --learned-only --output eval/v3e/262420/m2x_d
```
（262421、262422 同理。）

**B. 第二轮拟合**（每个模型的 8 块数据都齐后）：
```powershell
python -B -m experiments.v3_fit_values --data eval/v3e/262420/m2_a.npz eval/v3e/262420/m2_b.npz eval/v3e/262420/m2_c.npz eval/v3e/262420/m2_d.npz eval/v3e/262420/m2x_a.npz eval/v3e/262420/m2x_b.npz eval/v3e/262420/m2x_c.npz eval/v3e/262420/m2x_d.npz --holdout 270040-270047,271160-271191 --output-dir eval/v3e/262420/values_r2
```
**C. 第二轮校准（新校准块）**：
```powershell
python -B -m experiments.v3_calibrate_values --run-dir logs/v3e_262420 --values eval/v3e/262420/values_r2 --seeds 262112-262123 --output eval/v3e/262420/m6_r2.json
```
**D. 第二轮仲裁**（M6 为 PASS 或 INCONCLUSIVE 的模型；STOP 的不跑）：
```powershell
python -B -m experiments.evaluate_hybrid_policy --episodes 48 --seed 262000 --horizon 35 --parametrization task_state_v3 --phase-time-observation --execution-feedback --adaptive-task --model logs/v3e_262420/final_model.zip --arbiter one_way --values eval/v3e/262420/values_r2 --output eval/v3e/262420/arbitrated_r2.json
```
**E. 判读**：
```powershell
python -B -m experiments.v3_readout --pure-mpc eval/v3e/pure_mpc.json --model 262420=eval/v3e/262420/learned_only.json,eval/v3e/262420/arbitrated_r2.json --model 262421=eval/v3e/262421/learned_only.json,eval/v3e/262421/arbitrated_r2.json --model 262422=eval/v3e/262422/learned_only.json,eval/v3e/262422/arbitrated_r2.json --output eval/v3e/readout_r2.json
```
（M5 被 STOP 跳过的模型，从 `--model` 里去掉该项，并在交回中注明。）

**停止条件**：只在程序故障时停（命令报错、NaN、仲裁行 QP 零推力回退每模型 > 50 步）。

## 5. 交回内容

1. 262420 第一轮 M5 完整结果（`arbitrated.json`）与它的第一轮判读；
2. 三个模型的 `values_r2/m3_report.json`、`m6_r2.json`、`arbitrated_r2.json`（若有）、`readout_r2.json`，及全部 SHA-256；
3. M2 补采的 `.json` 元数据（`.npz`、`.pt` 只报路径与 SHA-256）；
4. **补充诊断（第一轮与第二轮仲裁行都要）**：每个"有真值违规"或"丢失 Pure 成功"的回合，列出种子、开局选的分支、是否交回及交回决策序号、失败时所在分支、失败类型，以及开局时的 μ_L、σ_L、μ_B、σ_B（都在 `arbitrated*.json` 的 `branch`、`handback_decision`、`arbiter_values_muL_sdL_muB_sdB` 字段里）。

## 6. 如果第二轮 V_L 仍然不达标

这时就不再是"数据太少"能解释的了，上层会在看到第二轮结果之后，再决定是否改变价值估计方法（例如把"成/败"作为分类问题单独估计）。**不在本轮预先加任何备用方案。**
