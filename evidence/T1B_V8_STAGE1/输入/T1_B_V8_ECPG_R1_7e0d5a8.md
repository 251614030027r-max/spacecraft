# 执行单：T1_B_V8_ECPG_20261021（修订 R1；下层本机；只批准阶段 I，限时 2 h）

**研究层决定（2026-10-21）：** 阶段 I 按 `181f5b4` 的冻结协议执行，限时 2 h；**不批准阶段 II**。阶段 I 结束后，上层根据真实闭环结果提交一次明确的论文主张和正式训练 GO/NO-GO 建议。不再扩展任务验证，不再添加新机制，不再修改开局。辅助机制（旋转目标动态进场许可）另行处理，不参与本轮耦合收益归因。

**R1 修正（研究层指出的三个问题）：**
1. ECPG 训练期贡献（P4）恢复为论文主张的**必要条件**，不再只报告。P4 不过，就不得声称学习耦合成功（第 4 节主张梯级）。
2. G1 除 AUC 外，还必须说明固定阈值实际触发了哪些状态、为什么触发：按真值类别和触发原因拆分，报告最早失败召回率、触发率、各类 Q 值，以及其它阈值下的结果（只报告，阈值固定为 0，不调）。
3. 计算收益必须报告真实核验延迟（每个触发决策的墙钟时间，与所执行意图的提前量对比；另报单分支串行成本），并做**等计算预算对照**：同一检查点、同样开局，随机触发，触发率取 S1 离线值；按模型分支数核对两臂花费的计算是否相当。

**代码：** 执行和训练代码 = `181f5b4b19488bb131e944a40705f9b87f4c62d7`（学习器、部署协议、阈值不变）。读数只用 `9183da21a518dfc4611383d064cbaa9ba9c75fde`：相对 181f5b4 只新增 `experiments/t1b_v8_readout.py` 和 1 个单元测试（`git diff --stat 181f5b4 9183da2` 只列这两个文件），所以全程检出 9183da2 即等同于冻结协议。
**设计依据：** 科学分支 `evidence/T1B_V8_PRECHECK/README.md`。

## 1. 方法（与 181f5b4 相同，不改）

- **V8 学习器（`train/t1_b_v8_sac.py`）**
  - critic 只用真实转移 TD 和真实意图 SMDP 目标，两臂相同。
  - 意图头：掩码分类头，最多 3 个 logits；末层零初始化，所以 0k 时确定性选择 = 最早候选。
  - Actor 损失 = Σπ(log π − Q/α) + λ·Σw·CE(p_M, π)。
  - p_M：预测失败的候选置 0；预测成功的候选按 softmax(G/2) 排序；全部候选都预测失败的决策跳过。
  - 偏好标签来自 V7 真实控制器分支，只在训练开局上生成，并经 V5.1 门控。
  - λ：E-ECPG = 1，E-SAC = 0。
- **部署协议（`experiments/t1b_v8.py`）**
  - 触发 `risk`：提议偏离最早候选，或 min(Q1,Q2)(提议) < 0。
  - 触发后：从真实状态快照出发，对全部候选做全时域 MPC 推演，执行预测成功候选中回报最高的那个。
  - `random`：按 ρ 的概率触发（以开局号和决策序号做哈希，可复现）。
  - 部署协议只在评估时使用，不进入训练。

## 2. 阶段 I（总墙钟 ≤ 2 h；不训练正式模型；不碰 300000 起的正式块）

```powershell
cd D:\py\DRL2; git fetch origin claude/sac-mpc-coupling-design-ns7g6i; git checkout --detach 9183da21a518dfc4611383d064cbaa9ba9c75fde
$py = "D:\py\DRL2\.venv\Scripts\python.exe"
$Q = "C:\Users\35884\Documents\Spacecraft\过程文件\T1B_V8_STAGE1_20261021"
$V7 = "C:\Users\35884\Documents\Spacecraft\过程文件\T1B_V7_ROUND_20261020_restart1"
$CK = "$V7\TRAIN_e_sac_s0\checkpoints\step_10000.zip"
$EXPERT = "C:\Users\35884\Documents\Spacecraft\过程文件\T1B_V3_ROUND_20261019\expert\expert_replay.pkl"
$F = "evidence\T1B_V7_SIDE_BENCH\final"
New-Item -ItemType Directory -Force $Q | Out-Null
(Get-CimInstance Win32_Processor | Measure-Object NumberOfLogicalProcessors -Sum).Sum | Out-File "$Q\logical_cores.txt"
```

**S0 单元与契约（≈ 5 min）**
```powershell
& $py -B -m pytest -q tests/test_t1_b_v8.py tests/test_t1_b_v2.py *> "$Q\pytest.log"
& $py -B -m experiments.t1b_v2 contract --interface v8 --output "$Q\contract"
```

**S3 V8 机械冒烟（与 S0 之后立即开始，后台运行；≤ 30 min，超时即停并报告）**
```powershell
Start-Process -FilePath $py -WorkingDirectory "D:\py\DRL2" -WindowStyle Hidden `
  -ArgumentList "-B","-m","experiments.t1b_v2","train","--interface","v8","--arm","e_ecpg","--seed","0","--steps","400","--smoke","--expert",$EXPERT,"--output","$Q\smoke_e_ecpg" `
  -RedirectStandardOutput "$Q\smoke.out.log" -RedirectStandardError "$Q\smoke.err.log"
```

**S1 离线触发行为（≈ 5 min）**：E-SAC 10k（真实 TD critic），确认块 + 独立块共 64 个初态。
```powershell
& $py -B -m experiments.t1b_v8 offline --interface v7 --checkpoint $CK `
  --episodes "$F\eval\e_sac_10000,$F\indep_eval\e_sac_10000" `
  --truth "$F\truth\*.json,$F\indep_truth\*.json" --model "$F\model,$F\indep_model" --output "$Q\offline_e_sac_10000.json"
$rho = (Get-Content "$Q\offline_e_sac_10000.json" -Raw | ConvertFrom-Json).summary.first_decision_recombination_truth_continuation_V1_teacher.verify_risk.triggered_fraction
$rho | Out-File "$Q\rho.txt"
```

**S2 真实闭环：风险触发 vs 等计算随机触发（两臂同时启动；从 S2 开始计时，85 min 截止）**
- 开局顺序预先登记：先独立块 293100–293131，再确认块 292500–292531，均按升序。
- 两臂使用同一检查点、同样的开局和并发设置（各 `--workers 2 --branch-workers 3`）。
- 到截止时间停止两臂，只用两臂都完成的开局配对；不补跑、不挑选。
```powershell
foreach ($arm in @(@("risk","0"), @("random",$rho))) {
  $m = $arm[0]; $r = $arm[1]
  Start-Process -FilePath $py -WorkingDirectory "D:\py\DRL2" -WindowStyle Hidden `
    -ArgumentList "-B","-m","experiments.t1b_v8","deploy","--interface","v7","--checkpoint",$CK,"--verify",$m,"--rho",$r, `
                  "--label","e_sac_10000_$m","--openings","293100-293131","--workers","2","--branch-workers","3","--output","$Q\deploy_$m" `
    -RedirectStandardOutput "$Q\deploy_${m}_indep.log" -RedirectStandardError "$Q\deploy_${m}_indep.err.log"
}
# 两臂的独立块都结束后，按同样方式以 --openings 292500-292531 启动确认块（同一个 --output 目录）
```
- 延迟是在“两臂各 2 个回合并行、每回合 3 个分支”的负载下测得的，按此报告，并附 `logical_cores.txt`。
- 不得据此声称实时可用。

**S4 读数（≈ 2 min）**
```powershell
& $py -B -m experiments.t1b_v8_readout --offline "$Q\offline_e_sac_10000.json" --risk "$Q\deploy_risk" --random "$Q\deploy_random" `
  --baseline "$F\eval\e_sac_10000,$F\indep_eval\e_sac_10000" --serial-model "$F\model,$F\indep_model" --smoke "$Q\smoke_e_ecpg" `
  --output "$Q\STAGE1_READOUT.json"
```

### 阶段 I 门（预登记；数字只取自 `STAGE1_READOUT.json`）

| 门 | 条件 |
|---|---|
| G0 机械 | pytest 和契约全部通过；冒烟 `preference_complete ≥ 1`，CE 有限 |
| G1 风险触发 | AUC ≥ 0.80；救援状态 293117 被触发；最早失败召回率 ≥ 0.80；首决策触发率 ≤ 60 % |
| G2 闭环收益 | risk 臂与已交付的 E-SAC 10k 无核验评估同开局配对：平均回报 ≥ +2.0；捕获不降；unsafe 不增；学习决策触发率 ≤ 60 % |
| G3 等计算 | 随机臂与 risk 臂的每回合分支数之比在 0.75–1.25 之间（否则判“不可比”，按未通过处理）；risk − random 配对回报 ≥ +1.0 |

**报告中必须写出（不设门）：**
- G1：四类状态（救援 / 全失败 / 有更优候选 / 最早最佳）各自的触发数、触发原因、Q 值范围；被触发开局清单；其它阈值下的触发率和召回率。用一段话说明阈值 0 实际起作用的方式：触发主要落在哪一类，是否把大量成功状态也拉进了核验。
- 延迟：
  - 触发决策的墙钟 p50 / p95 / max；
  - 延迟与提前量之比；
  - 单分支串行成本 p50 / p95；
  - 每回合分支数；
  - 核验后改变选择的次数；
  - 第二次及以后决策的触发数。

## 3. 阶段 I 之后：上层提交论文主张与 GO/NO-GO（规则预先登记）

| 结果 | 上层建议 |
|---|---|
| G0–G3 全过 | 建议 GO 阶段 II（由研究层和用户决定）。必须同时写明：阶段 I 没有训练 V8，因此 **对 ECPG 训练期贡献（P4）没有任何证据**；阶段 II 的主张按第 4 节梯级 |
| G2 过、G3 不过 | NO-GO。收益来自 MPC 核验，学习在等计算下不比随机好；论文最多写“MPC 核验混合决策”，不得写学习耦合 |
| G1 或 G2 不过 | NO-GO。如实报告，不改阈值、不换开局、不补跑 |
| G0 不过 | 只修代码缺陷，重跑 G0 |

## 4. 阶段 II 草案（未批准；仅作 GO/NO-GO 的对象，R1 已按研究层意见修正）

**训练：** 2 个种子 × 两臂（E-SAC-V8 λ = 0，E-ECPG-V8 λ = 1）× 10k，从零开始，检查点 2k / 5k / 10k；V8 学习器与 181f5b4 相同。

**评估开局（本执行单登记）：**
- 验证块 294000–294031：所有检查点 × {off, risk}，给出真实任务学习曲线。
- 最终块 300000–300063：方案冻结后只打开一次。10k 检查点 ×：
  - {off, risk}；
  - 每个臂 × 种子各加一组 random，ρ 取该检查点在验证块上 risk 的实测触发率（等计算）；
  - 首选候选扫描（fixed:0 与首决策 oracle）；
  - E-SAC 种子 0 的 `--verify all`（计算参照）。

**主张梯级（预登记；逐种子报告，不挑种子）：**

| 主张 | 必要条件（最终块；学习曲线取验证块） |
|---|---|
| **A. 训练期 SAC–MPC 耦合有效（学习耦合主张）** | **P4：** 两个种子在同一协议下，E-ECPG 相对 E-SAC：risk 下平均回报 ≥ +1.0 且捕获不降；off 下不差于 −1.0。**学习曲线：** E-ECPG 在验证块 risk 下 10k ≥ 0k + 1.0，且 10k 高于 E-SAC 10k |
| B. 学习决定 MPC 计算分配 | **P2：** risk 相对等计算随机 ≥ +1.0（分支数之比 0.75–1.25），两种子都满足。**学习曲线：** 验证块上 risk 与随机的差随训练不减小 |
| C. 混合决策任务收益 | **P1：** E-ECPG + risk 相对 fixed:0，平均回报 ≥ +2.0，捕获不降，unsafe 不增。**P3：** 触发 ≤ 60 %，保留 verify-all 增益 ≥ 70 % |

- **论文主张：** A + C 成立，才可写“SAC–MPC 耦合学习成功”。只有 B + C 成立时，只能写“计算受限的学习—预测混合决策”：性能来自 MPC 核验，学习负责计算分配；并明确写出“未观察到训练期耦合的独立收益”。
- **延迟：** 报告与阶段 I 相同的各项。只有在 p95 延迟 ≤ 执行意图提前量 p5（同一硬件、同一负载）时，才可称“在该硬件上可在线完成”；否则一律写成“计算辅助参照，非实时”。
- **辅助机制（动态进场许可）：** 另行独立处理，不进入 A–C 的任何对比。

## 5. 交付

- **collab：** `docs/collaboration/lower/handoffs/t1b_v8_stage1_20261021/STAGE1_RESULT.md`，包括：
  - G0–G3 逐项结果；
  - 上面列出的 G1 说明和延迟各项；
  - 两臂逐开局表；
  - 截止时各臂完成的开局数。
- **科学分支 `evidence/T1B_V8_STAGE1/`：**
  - `STAGE1_READOUT.json`、`offline_e_sac_10000.json`、`rho.txt`、`logical_cores.txt`；
  - `deploy_risk/*.json`、`deploy_random/*.json`；
  - 冒烟的 `manifest.json` / `finished.json` / `learning_audit.json`；
  - pytest 和契约日志；
  - 每个文件单独 `git add -f`，附 SHA256SUMS。
