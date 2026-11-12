# 执行单：T1_ROOTCAUSE_DIAG_20261013（下层本机执行，第一阶段）

**授权：** 用户 2026-10-13 下达《T1 SAC–MPC 学习失败根因诊断指令》，并说明"需要模型时由上层给指令，下层执行"。本单为其中需要检查点的 P1（策略干预）和 P2/P3（Critic 价值排序、M1 目标）的**第一阶段**。

**性质：** 研究诊断。**不训练、不改动任何冻结代码**；G★、MPC、奖励、物理环境和终止条件都保持不变；不访问正式块。

**代码：** 科学提交 `ca82c637c0110e96910d87e3302f4fca18e03215`（分支 `claude/sac-mpc-coupling-design-ns7g6i`）。相对 `78eb63c`，只新增了以下内容：

- `diagnostics/t1_rootcause/`（诊断脚本）；
- `tests/test_t1_rootcause_diag.py`；
- `experiments/t1_v2_review.py` 的记账更正；
- `evidence/T1_V2_S0/` 的更正文件。

**环境、动力学和评估代码与 `78eb63c` 逐字节相同**（可用第 1 节命令核对）。

**使用的检查点：** `logs/t1/t1v2_m1_s0_r1/checkpoints/` 中的 `m1_0_decisions.zip`、`m1_20000_decisions.zip`、`m1_30000_decisions.zip`。SHA-256 必须与交付的 `MODEL_SHA256.json` 一致。

**预算：** 第一阶段预计约 **5–6 CPU·h**，12 核机器上墙钟约 40 分钟。第 2 节的试跑若估算超过 **6.5 CPU·h**，先停下上报，**不得自行扩展**。

---

## 1. 现场与检出

```powershell
Get-Process python -ErrorAction SilentlyContinue | Format-Table Id, StartTime, Path   # 应无训练或评估进程
cd D:\py\DRL2_T1
git status --porcelain                  # 应无输出
git fetch origin
git checkout --detach ca82c637c0110e96910d87e3302f4fca18e03215
git log -1 --format=%H                  # 必须是 ca82c637c0110e96910d87e3302f4fca18e03215
git diff --stat 78eb63c ca82c637c0110e96910d87e3302f4fca18e03215 -- env train experiments/t1_dev_eval.py   # 应无输出
$env:OMP_NUM_THREADS="1"; $env:MKL_NUM_THREADS="1"; $env:OPENBLAS_NUM_THREADS="1"
$R = "logs/t1/t1v2_m1_s0_r1"
$D = "logs/t1/_diag"
mkdir $D -ErrorAction SilentlyContinue
Get-FileHash "$R/checkpoints/m1_0_decisions.zip","$R/checkpoints/m1_20000_decisions.zip","$R/checkpoints/m1_30000_decisions.zip" -Algorithm SHA256 | Format-List > "$D/model_sha.txt"
python -B -m pytest -q tests/test_t1_rootcause_diag.py tests/test_t1_gym.py
```

检查点哈希必须与 `MODEL_SHA256.json` 一致，测试必须全部通过，否则停下上报。

## 2. 试跑测时（决定是否继续）

```powershell
python -B -m diagnostics.t1_rootcause.interventions --checkpoint "$R/checkpoints/m1_30000_decisions.zip" --label 30k --modes A --openings 295001 --output-dir "$D/p1"
```

这一回合同时用于两项检查：

- **抽查：** 结果必须与 `$R/dev_eval/30000/295001.json` 完全相同（同为 timeout，`discounted_return` 逐位相等）。
- **测时：** 用 `wall_s / decisions` 得到每决策秒数 s。

第一阶段估算总量 ≈ 9500 决策 × s（P1 约 4800 决策，P2 约 4700 决策）。

- 若 9500 × s / 3600 > 6.5，**停下上报**；
- 否则继续第 3 节。

## 3. P1 策略干预（8 个开局，预注册）

**开局：** 295001、295002、295010、295015、295019（30k 的全部 5 个失败开局），以及 295000、295003、295004（30k 成功开局中按开局文件顺序取前 3 个）。

**为什么不重跑某些模式：**

- 模式 A（原策略）直接用本机已有的开发评估，二者同机且结果确定，只做第 2 节和下面两条抽查。
- 模式 E（完整规则）用本机已有的 `logs/t1/_v2/rule_local`。
- 0k 在有计划时始终请求保持（开发日志中 558 次全是保持），所以 0k 的 B 与 A 完全相同、D 与 C 完全相同，0k 只需跑 C。

```powershell
$O = "295001,295002,295010,295015,295019,295000,295003,295004"
python -B -m diagnostics.t1_rootcause.interventions --checkpoint "$R/checkpoints/m1_30000_decisions.zip" --label 30k --modes B,C,D --openings $O --output-dir "$D/p1" --workers 10
python -B -m diagnostics.t1_rootcause.interventions --checkpoint "$R/checkpoints/m1_20000_decisions.zip" --label 20k --modes A,B,C,D --openings 295001 --output-dir "$D/p1"
python -B -m diagnostics.t1_rootcause.interventions --checkpoint "$R/checkpoints/m1_20000_decisions.zip" --label 20k --modes B,C,D --openings $O --output-dir "$D/p1" --workers 10
python -B -m diagnostics.t1_rootcause.interventions --checkpoint "$R/checkpoints/m1_0_decisions.zip" --label 0k --modes A,C --openings 295001 --output-dir "$D/p1"
python -B -m diagnostics.t1_rootcause.interventions --checkpoint "$R/checkpoints/m1_0_decisions.zip" --label 0k --modes C --openings $O --output-dir "$D/p1" --workers 10
```

（第二条和第四条命令中的 A，就是 20k、0k 在 295001 上的抽查。）

## 4. P2/P3 Critic 排序与 M1 目标（10 个状态，预注册）

**状态清单：** `diagnostics/t1_rootcause/states.json`。它由已发布的开发日志按固定规则生成，规则写在 `critic_probe.select` 中。第一阶段只跑序号 **0,1,4,5,8,9,12,13,16,17**：30k 每个失败开局的"第一次实际推迟"，以及对应的 20k 对照。

**每个状态的计算内容：**

- 0k、20k、30k 三个 Critic 对"保持 / 提前 15 s / 推迟 15 s"的 Q 值，a0 网格上的 Q，以及 Actor 输出处的梯度；
- M1 目标 y_cf；
- 三个动作从克隆状态出发的闭环回报，后续策略分别为固定规则（`rule`）和该轨迹检查点的确定性 Actor（`traj_det`）。

```powershell
python -B -m diagnostics.t1_rootcause.critic_probe probe --states diagnostics/t1_rootcause/states.json --only 0,1 --checkpoints "0k=$R/checkpoints/m1_0_decisions.zip,20k=$R/checkpoints/m1_20000_decisions.zip,30k=$R/checkpoints/m1_30000_decisions.zip" --followups rule,traj_det --verify-branch --output-dir "$D/p2" --workers 2
python -B -m diagnostics.t1_rootcause.critic_probe probe --states diagnostics/t1_rootcause/states.json --only 4,5,8,9,12,13,16,17 --checkpoints "0k=$R/checkpoints/m1_0_decisions.zip,20k=$R/checkpoints/m1_20000_decisions.zip,30k=$R/checkpoints/m1_30000_decisions.zip" --followups rule,traj_det --output-dir "$D/p2" --workers 8
```

每个输出 JSON 的 `replay_check.match` 都必须为 true，即重放到达的状态与开发日志一致。若有 false，照常交付，并在报告中列出。

## 5. 汇总与交付

```powershell
python -B -m diagnostics.t1_rootcause.analyze --interventions "$D/p1" --dev-eval "$R/dev_eval" --rule logs/t1/_v2/rule_local --probe "$D/p2" --output "$D/diag_readout.json"
```

**交付到** `lower/handoffs/t1_rootcause_20261013/`：

- 一个 ZIP，内含 `$D` 整个目录（`p1/`、`p2/`、`diag_readout.json`、`model_sha.txt`）和控制台输出；
- 一页 `RECEIPT.md`，写明：检出 SHA、检查点哈希核对、抽查结果、每决策秒数、实际 CPU·h（各输出的 `wall_s` 之和）、`replay_check` 与 `verify_branch` 的结果，以及有无异常；
- `SHA256SUMS`。

**只交付事实，不做解读。** 解读由上层写入诊断报告。

## 6. 禁止事项

- 不训练；
- 不修改任何代码或参数；
- 不跑第二阶段（其余 10 个状态、随机后续策略、扩到 16 个开局），这些由用户另行决定；
- 不访问正式块；
- 不删除或移动任何已有日志和模型。
