# v3d 执行指令：训练 + 训练后方法流水线（给下层）

*2026-09-26，上层窗口。分支 `claude/sac-mpc-coupling-design-ns7g6i`。**代码提交：`e4ddf6cd43ba4649d753189633e1e501f9171e2a`**（之后只有文档与证据提交）。全套测试 328 passed、3 xfailed。*
*取代 `docs/V3D_TRAINING_ORDER_20260926.md`（那份从未开跑）和 `docs/V3C_TRAINING_ORDER_20260925.md`。审查全文见 `docs/V3D_REVIEW_20260926.md`。*
*方法定义不变：`docs/V3_METHOD_EXECUTION_ORDER_20260924.md`（M1–M6 与第 6 节判读）；本指令只把它落成可直接运行的命令，并修正第 6 节里写死的"36/48"。*

---

## 0. 相比 v3c 改了什么（都在审查文档里有证据与测试）

| # | 改动 | 影响训练 | 影响评估 | 影响 Pure MPC |
|---|---|---|---|---|
| 1 | MPC 终端约束闸门：只在"合法进入圆柱 ∩ 入口半径"内启用（取代 v3c 的背面补丁） | 是 | 是 | 36 个完成回合逐位不变；1 个"非法进入后"的回合变为合法重入完成：**36 → 37/48** |
| 2 | 300 s 超时按真实终止处理（`timeout_is_terminal`） | 是（SAC 目标） | 否 | 否 |
| 3 | `controller.reset()` 清空求解器缓存（此前重置不彻底） | 数值上极小 | 否（评估器每回合新建环境） | 否 |
| 4 | 评估器每个决策更新"执行反馈"观测（此前恒为 0） | 否 | **是**：评估观测与训练逐位一致（有测试） | 否 |
| 5 | M6 一致率只计真实回报差 ≥ 1 的检查点（数据产生前修正） | 否 | M6 判读 | 否 |

另外：M2–M6 全部有了脚本（第 4 节）；V2 评估报告已加勘误（其模型完成数作废，原因见审查第 2 节）。

---

## 1. 预检（PowerShell，仓库根目录）

```powershell
git fetch origin claude/sac-mpc-coupling-design-ns7g6i
git checkout claude/sac-mpc-coupling-design-ns7g6i
git pull --ff-only origin claude/sac-mpc-coupling-design-ns7g6i
git rev-parse HEAD
git diff e4ddf6cd43ba4649d753189633e1e501f9171e2a HEAD --stat        # 只允许出现 docs/ 与 eval/ 下的文件
git status --porcelain --untracked-files=no   # 必须为空
python -B -m pytest -q
```
全套测试必须全绿（预期 328 passed、3 xfailed）。`v3_*`、`v3b_*`、`v3c_*` 目录原样保留，不续训、不复用运行名、不加载其 checkpoint；`v3c_262420/421` 各加侧车 `SUPERSEDED_GATE_FIX.md`（"闸门与超时修复之前的运行，仅作接口可学性参考"）。

## 2. M1：训练（三个窗口，一个种子一个）

```powershell
$env:OMP_NUM_THREADS="1"; $env:MKL_NUM_THREADS="1"; $env:OPENBLAS_NUM_THREADS="1"
python -u -B -m train.train_hybrid --steps 60000 --seed 262420 --run-name v3d_262420 --horizon 35 --parametrization task_state_v3 --adaptive-task --device cpu
```
另两个窗口把 262420 换成 262421、262422。不加任何其他参数；窗口不关、系统不睡眠。

**开跑 1 分钟后检查 manifest**：
```powershell
python -c "import json,sys;m=json.load(open(sys.argv[1]));h=m['hyperparameters'];print({k:m.get(k) for k in ('code_commit','code_dirty','observation_dimension','action_dimension','training_branch','waypoint_parametrization','seed')}, 'timeout_is_terminal=',h.get('timeout_is_terminal'))" logs/v3d_262420/manifest.json
```
必须：`code_commit` = 第 1 节 HEAD、`code_dirty` False、观测 42、动作 2、`training_branch` "learned"、`task_state_v3`、`timeout_is_terminal= True`。

**健康检查**（10k / 20k / 30k / 40k / 50k 附近，只读）：
```powershell
python -B -m experiments.check_v3b_training_health logs/v3d_262420 logs/v3d_262421 logs/v3d_262422
```
| 规则 | 条件 | 动作 |
|---|---|---|
| S1 | 任一回合参考跳变违例 > 0 | 停该运行并报告 |
| S2 | 最近 100 回合零推力回退率 > 1 × 10⁻³ | 停该运行并报告；打包 monitor、manifest、最近两个 checkpoint |

完成率、回报只记录，不作停止依据；不改超参 / 奖励、不丢种子、不挑 checkpoint、不续训。

## 3. 训练结束后第一件事：Pure MPC 行（同一提交、同一台机器）

判读要和"同一代码、同一机器上的 Pure MPC"比较，所以先重跑它（单进程，约与一个模型评估同时长）：
```powershell
python -B -m experiments.evaluate_hybrid_policy --episodes 48 --seed 262000 --horizon 35 --parametrization arrival_condition --adaptive-task --control desired_pose --output eval/v3d/pure_mpc_262000.json
```
预期 **37/48**：与 `eval/adp/pure_mpc_nominal.json`（36/48）相比，只有 262034 由失败变为完成，其余 47 个回合的完成标志不变（沙箱证据 `eval/adp/v3d_gate_pure_mpc_count.json`）。如果不是这样，**停，报上层**，不继续。

## 4. M2 → M3 → M6 → M5（每个模型分别做，三个模型可并行）

以下以 262420 为例；`R=logs/v3d_262420`，`O=eval/v3d/262420`。

**M2 数据**（每个模型 4 个进程并行，每进程 12 个种子 × 4 回合）：
```powershell
python -B -m experiments.v3_collect_value_data --run-dir logs/v3d_262420 --seeds 270000-270011 --output eval/v3d/262420/m2_a
python -B -m experiments.v3_collect_value_data --run-dir logs/v3d_262420 --seeds 270012-270023 --output eval/v3d/262420/m2_b
python -B -m experiments.v3_collect_value_data --run-dir logs/v3d_262420 --seeds 270024-270035 --output eval/v3d/262420/m2_c
python -B -m experiments.v3_collect_value_data --run-dir logs/v3d_262420 --seeds 270036-270047 --output eval/v3d/262420/m2_d
```
**M3 价值拟合**（单进程，分钟级）：
```powershell
python -B -m experiments.v3_fit_values --data eval/v3d/262420/m2_a.npz eval/v3d/262420/m2_b.npz eval/v3d/262420/m2_c.npz eval/v3d/262420/m2_d.npz --output-dir eval/v3d/262420/values
```
**M6 校准**（单进程）：
```powershell
python -B -m experiments.v3_calibrate_values --run-dir logs/v3d_262420 --values eval/v3d/262420/values --output eval/v3d/262420/m6.json
```
输出最后一行 `"gate": "PASS"` 才进 M5（一致率按真实回报差 ≥ 1 的检查点计算，门槛 80%）；`"STOP"` 则该模型停在这里，报上层（其余模型照常）。

**M5 正式评估**（每个模型两行，各单进程）：
```powershell
python -B -m experiments.evaluate_hybrid_policy --episodes 48 --seed 262000 --horizon 35 --parametrization task_state_v3 --phase-time-observation --execution-feedback --adaptive-task --model logs/v3d_262420/final_model.zip --output eval/v3d/262420/learned_only.json
python -B -m experiments.evaluate_hybrid_policy --episodes 48 --seed 262000 --horizon 35 --parametrization task_state_v3 --phase-time-observation --execution-feedback --adaptive-task --model logs/v3d_262420/final_model.zip --arbiter one_way --values eval/v3d/262420/values --output eval/v3d/262420/arbitrated.json
```
（评估参数与 manifest 的 `evaluation_flags` 一致；评估器会拒绝观测维度不匹配的模型。）

**判读**（三个模型都跑完后，一条命令）：
```powershell
python -B -m experiments.v3_readout --pure-mpc eval/v3d/pure_mpc_262000.json --model 262420=eval/v3d/262420/learned_only.json,eval/v3d/262420/arbitrated.json --model 262421=eval/v3d/262421/learned_only.json,eval/v3d/262421/arbitrated.json --model 262422=eval/v3d/262422/learned_only.json,eval/v3d/262422/arbitrated.json --output eval/v3d/readout.json
```
判读规则就是 `V3_METHOD_EXECUTION_ORDER` 第 6 节，唯一改动：第一层"完成 ≥ 36/48"改为"完成 ≥ 第 3 节 Pure MPC 的完成数"。脚本逐模型给出两层结果与总判定，**不合并种子**。

## 5. 交回内容

1. 训练：三个 manifest、至少 10k / 30k / 60k 的健康检查输出、三个 `train.monitor.csv` 的 SHA-256；
2. 第 3 节 Pure MPC JSON；
3. 每个模型：`m3_report.json`、`m6.json`、`learned_only.json`、`arbitrated.json`；
4. `eval/v3d/readout.json` 与脚本打印的末 4 行；
5. 所有 JSON 的 SHA-256。模型 ZIP、M2 的 `.npz`、`values/*.pt` 不入库，只报路径与 SHA-256（`m3_report.json` 里已记录 `.pt` 的 SHA）。

## 6. 本指令依据的证据

| 产物 | SHA-256 | 生成脚本 | 内容 |
|---|---|---|---|
| `eval/adp/v3d_gate_pure_mpc_count.json` | `69f3725e7279c00d632df692ba5ca530368367a907e9926ef71ea3de0b625d19` | `experiments/count_pure_mpc_gate_differences.py` + `experiments/summarize_gate_change_pure_mpc.py` | 闸门修复前后 48 回合 Pure MPC 逐步对照与闸门判定计数 |
| `eval/adp/v3d_evaluator_feedback_impact.json` | `da3df6bcdc2f77f7667cf8d87b29f6b3e3f251518bc4c1ef826878416dafc4f7` | `experiments/compare_evaluator_feedback_fix.py` | 评估器修复前后同一策略、同一种子的对照 |
| `eval/adp/v3c_262422_qp_burst_diagnosis.json` | `10df2f1352786c5707d4d31b69c998999301282ae6d3a8568b55a5504d5e6933` | `experiments/analyze_v3c_qp_bursts.py` | v3c 262422 的零推力段定位（背面 108 步、环带 5 步） |

沙箱与用户机器的浮点结果在 10⁻⁷ 量级上可能不同（已知现象），所以第 3 节只比较完成标志与步数，不比较 Δv 的全部位数。

## 7. 禁止事项

- 不续训旧运行，不复用旧运行名，不加载旧 checkpoint；
- 不改奖励、MPC、任务参数、超参、限速上界、近场下限；
- 不因训练中的完成率改任何东西；M6 未过不跑 M5；
- 工作树不干净时不开跑；评估不挑 checkpoint，只用 `final_model.zip`。
