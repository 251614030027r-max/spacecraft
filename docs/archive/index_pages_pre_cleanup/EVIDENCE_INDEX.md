# 日志、模型与论文证据索引

## 原则

日志是正式审计证据；模型只是可复核策略快照。失败日志保留并不等于失败结论仍然有效，必须结合生成它的代码版本和 `CLAUDE.md` 判断。`models/` 被 `.gitignore` 排除，不能依靠 Git 恢复，因此本次整理不删除任何模型。

## 当前 adaptive sync-entry 主线（2026-09-19）

- 执行单：`docs/ADAPTIVE_MAINLINE_RUNSHEET.md`；
- 奖励修复：`docs/REWARD_UNITS_FIX_20260919.md`；
- 训练前审计：`docs/TRAINING_PREFLIGHT_AUDIT_20260919.md`；
- 第一轮错误奖励运行的最小原始证据：`local_artifacts/reward_units_bug_20260919/`（三份 manifest + 三份 Monitor CSV）；
- 有效重训产物将写入本机 `logs/adp_rf_262410/411/412/`，未完成前不得形成论文结论。

旧 `adp_262410/411/412` 的 checkpoint、模型和 TensorBoard 已删除：其 critic 学到“悬停优于合法完成”的反向目标，不能用于恢复训练或部署门。旧校准重复目录 `eval/cal/` 已删除；当前小样本校准保留于本机 `eval/cal2/`。

### 2026-09-21/22 第一轮收尾与 V2（13 份）

1. `docs/ADAPTIVE_ROUND1_CLOSEOUT_20260921.md`：第一轮收尾、R5 根因终判与模型哈希；
2. `docs/COUPLING_DEFECT_DIAGNOSIS_LOG_20260921.md`：D0–D4 诊断日志与复现实物；
3. `docs/COUPLING_DEFECT_DIAGNOSIS_ORDER_20260921.md`：缺陷诊断预登记执行单；
4. `docs/DECK_CLEARING_ORDER_20260922.md`：进入耦合设计前的零机时收尾边界；
5. `docs/ROUND1_EXTERNAL_REVIEW_20260922.md`：第一轮外部审查件；
6. `docs/V2_EXECUTION_ORDER_20260922.md`：V2 实现预登记执行单；
7. `docs/V2_EXECUTION_SIGNAL_AUDIT_20260922.md`：离线执行信号审计；
8. `docs/V2_FIX_ORDER_RATE_LIMIT_20260922.md`：米单位限速缺陷修复单；
9. `docs/V2_INTERFACE_IMPLEMENTATION_20260922.md`：V2 实现、测试与架构地板；
10. `docs/V2_RATE_LIMIT_DECISION_20260922.md`：`0.40 m` 上界最终裁决；
11. `docs/V2_RATE_LIMIT_SWEEP_STOP_20260922.md`：四档扫描、探针纠偏与停止记录；
12. `docs/V2_REVIEW_RATE_LIMIT_20260922.md`：米位移无界缺陷复核；
13. `docs/V3_LITERATURE_AND_COUPLING_DIRECTION_20260922.md`：当前耦合方向与核心文献。

文档状态与新窗口阅读顺序以 `docs/INDEX.md` 为权威地图；上述文件只提供证据或当前裁决，不自动授权训练。

## 当前正式基线证据

### 非合作探针 2（2026-09-16）

正式配对结果见 `docs/PROBE2_NONCOOP_RESULT_20260916.md`：同一感知环境、同一 48 种子下，EKF 估计控制 Pure MPC 为 31/48，真值控制为 32/48，`ΔC=1`，命中预签的“基本持平”停止分支。原始 A/B JSON 仅保留于本机 `logs/precap_noncoop/`，不推入公开仓库。

### T12 / S10（2026-09-15）

正式结论见 `docs/T12_S10_FORMAL_EVALUATION_REPORT_20260915.md`，预登记规则见 `docs/T12_S10_EVAL_EXECUTION_ORDER.md`。可随 Git 审查的精简机器证据位于 `eval/results/t12_s10_20260915/`，包括六组逐局 JSON、训练 manifest、最终汇总和审计主表；本机完整训练、首轮评估与补字段复评仍分别保留于 `logs/t12_train/`、`logs/t12_eval/` 和 `logs/t12_eval_v2/`。该结果属于 full-state `perception=None`，并行运行产生的 compute 数字不构成实时性证据。

预登记结论为分叉 2：`arrival_condition` 三训练种子完成 16/48、34/48、28/48，Pure MPC 为 32/48，`radial_local` 三种子均为 0/48；不得以单个最好种子或事后 oracle 替代三种子分布。

### 较早基线

| 主张 | 日志/模型 |
|---|---|
| Pure SAC `single_phase` 完成 3/1/0、无违约 20/10/0 | `logs/sac_seed260860.json` 至 `sac_seed260862.json`；`models/gatefree_sac_400k_fix_seed26086*/checkpoints/sac_400000_steps.zip` |
| Pure MPC 名义基准 | `logs/mpc_single_phase.json`、`logs/mpc_h50_fixed.json` |
| 脚本可达性参考 | `logs/scripted_single_phase.json` |
| Pure SAC 全任务 150k 三种子历史结论 | `logs/fullmission_sac_200k_seed26083*`；对应 `models/` 目录中的 150k checkpoint |
| 固定熵 Phase-I 三种子 | `logs/phase2_mission_s1v2_*fixedalpha*` 与相应 `models/` |

## 四探针证据

- 终端值：`logs/mpc_h10_*`、`logs/mpc_h50_fixed.json`、`models/terminal_value_scripted_seed*.json`；
- 相位采样：`logs/ps_*.json`，三个种子块，每块 20 回合；
- 目标模型失配：结果摘要在 `docs/PROBE_RESULTS.md`，正式 JSON 尚未保存；
- 观测误差：错误版本网格隔离在 `logs/invalidated/observation_velocity_rotation_bug/`；修正后仅有快验证结论，尚无正式三种子 JSON。

## 历史训练日志

- `phase2_sac_v2_*`：早期受约束任务与多因素探索；
- `phase2_mission_s1_*`：两阶段任务、语义修正和单因素演进；
- `phase2_mission_s1v2_*`：S1-v2 acquisition、速度误差、固定熵和 replay 对照；
- `gatefree_sac_*`：统一 `single_phase` 前后的训练与三种子正式弱基线；
- `fullmission_sac_*`：两阶段 full-mission 历史证据。

正式训练目录通常保留 `manifest.json`、`train.monitor.csv`、`phase2_diagnostics.json`、TensorBoard 标量和固定种子周期评价。不得只拿 checkpoint 或单个最好种子替代这些证据。

## 模型保留边界

当前 `models/` 含多个历史训练目录，约 1 GiB；这与旧 `models/README.md` 所称“只保留四个模型”不一致。它们均不是通过统一成功门槛的合格策略，但部分 checkpoint 支撑 `CLAUDE.md` 的正式结果。本轮不做模型裁剪。后续若要缩减，必须先按本索引列出每个正式主张所需的确切 checkpoint，再删除其余文件。

## 论文

`References/` 中论文全部保留。读取 PDF 时使用 `pdf2txt.py` 抽取到被忽略的 `scratchpad/` 后检索，不把整本 PDF 注入上下文。各论文的用途和禁区见 `CLAUDE.md` 的 “What each reference is for”。
