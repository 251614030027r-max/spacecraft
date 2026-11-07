# 下层执行单：v3d 收尾测试 + v3e 训练与正式评估

*2026-09-28，上层窗口。分支 `claude/sac-mpc-coupling-design-ns7g6i`。本文件是唯一需要执行的指令，取代 `docs/V3E_PLAN_AND_EXECUTION_ORDER_20260927.md` 的 B 部分（该文件 A 部分是原因说明，可不读）。*
*v3e 代码提交：`a362d4398d35405a83dce7f8ba7de19c0e6d9497`。之后只有文档提交。全套测试 335 passed、3 xfailed。*

每个进程开跑前都先设置：
```powershell
$env:OMP_NUM_THREADS="1"; $env:MKL_NUM_THREADS="1"; $env:OPENBLAS_NUM_THREADS="1"
```

---

## 第 1 步：v3d 收尾测试（**不要拉代码**，仍在 `b05e389` 上）

v3d 已在约 50k 手动停止。它只作消融对照，统一使用 50k checkpoint（该规则在看到任何 50k 结果之前定下）。

1. 确认三个目录都有 `logs/v3d_2624{20,21,22}/checkpoints/sac_mpc_50000_steps.zip`；**不使用** `interrupted_model.zip`。
2. 健康检查一次（记录停止时状态）：
```powershell
python -B -m experiments.check_v3b_training_health logs/v3d_262420 logs/v3d_262421 logs/v3d_262422
```
3. 三个 learned-only 评估（三个窗口并行，各约 2–3 小时）：
```powershell
python -B -m experiments.evaluate_hybrid_policy --episodes 48 --seed 262000 --horizon 35 --parametrization task_state_v3 --phase-time-observation --execution-feedback --adaptive-task --model logs/v3d_262420/checkpoints/sac_mpc_50000_steps.zip --output eval/v3d/learned_only_50k_262420.json
python -B -m experiments.evaluate_hybrid_policy --episodes 48 --seed 262000 --horizon 35 --parametrization task_state_v3 --phase-time-observation --execution-feedback --adaptive-task --model logs/v3d_262421/checkpoints/sac_mpc_50000_steps.zip --output eval/v3d/learned_only_50k_262421.json
python -B -m experiments.evaluate_hybrid_policy --episodes 48 --seed 262000 --horizon 35 --parametrization task_state_v3 --phase-time-observation --execution-feedback --adaptive-task --model logs/v3d_262422/checkpoints/sac_mpc_50000_steps.zip --output eval/v3d/learned_only_50k_262422.json
```
4. 每个 v3d 目录加侧车 `STOPPED_AT_50K.md`："消融用途（旧参考实现），按预先决定统一使用 50k checkpoint；不续训"。
5. v3d **不跑** M2–M6。

> **为了尽快推进**：第 1 步的三个评估与第 3 步的 v3e 训练可以同时进行——但必须先把评估进程启动起来（它们在启动时已加载 `b05e389` 的代码），**再**在另一个工作目录拉取新代码开训。最简单的做法：
> ```powershell
> git worktree add ..\DRL2_v3e claude/sac-mpc-coupling-design-ns7g6i   # 新目录用于 v3e
> ```
> v3d 评估留在 `D:\py\DRL2`（`b05e389`，不要 pull），v3e 在 `D:\py\DRL2_v3e` 里 pull 并训练。两个目录互不影响。

## 第 2 步：v3e 预检（在 v3e 工作目录）

```powershell
git fetch origin claude/sac-mpc-coupling-design-ns7g6i
git checkout claude/sac-mpc-coupling-design-ns7g6i
git pull --ff-only origin claude/sac-mpc-coupling-design-ns7g6i
git rev-parse HEAD
git diff a362d4398d35405a83dce7f8ba7de19c0e6d9497 HEAD --stat   # 只允许 docs/、eval/、CLAUDE.md
git status --porcelain --untracked-files=no                     # 必须为空
python -B -m pytest -q                                          # 预期 335 passed, 3 xfailed
```
若用新 worktree，需要先准备 Python 环境（沿用 `D:\py\DRL2\.venv` 的解释器即可，在 v3e 目录里运行）。

## 第 3 步：v3e 训练（三个窗口，从零，60k）

```powershell
python -u -B -m train.train_hybrid --steps 60000 --seed 262420 --run-name v3e_262420 --horizon 35 --parametrization task_state_v3 --adaptive-task --device cpu
python -u -B -m train.train_hybrid --steps 60000 --seed 262421 --run-name v3e_262421 --horizon 35 --parametrization task_state_v3 --adaptive-task --device cpu
python -u -B -m train.train_hybrid --steps 60000 --seed 262422 --run-name v3e_262422 --horizon 35 --parametrization task_state_v3 --adaptive-task --device cpu
```
不加任何其他参数；窗口不关、系统不睡眠。

**开跑 1 分钟后查 manifest**（每个运行一次）：
```powershell
python -c "import json,sys;m=json.load(open(sys.argv[1]));print({k:m.get(k) for k in ('code_commit','code_dirty','observation_dimension','action_dimension','training_branch','waypoint_parametrization','seed')}, 'timeout_is_terminal=',m['hyperparameters'].get('timeout_is_terminal'))" logs/v3e_262420/manifest.json
```
必须：`code_commit` = 第 2 步 HEAD，`code_dirty` False，42，2，learned，task_state_v3，`timeout_is_terminal= True`。

**健康检查**（10k / 20k / 30k / 40k / 50k 附近）：
```powershell
python -B -m experiments.check_v3b_training_health logs/v3e_262420 logs/v3e_262421 logs/v3e_262422
```
- S1：任一回合参考跳变违例 > 0 → 停该运行并报告；
- S2：最近 100 回合零推力回退率 > 1 × 10⁻³ → 停该运行并报告（打包 monitor、manifest、最近两个 checkpoint）；
- 完成率、回报只记录，不作任何调整依据。

**可选早读**（20k、40k；只看，不作决定依据；机器太忙就跳过）：
```powershell
python -B -m experiments.evaluate_hybrid_policy --episodes 12 --seed 264100 --horizon 35 --parametrization task_state_v3 --phase-time-observation --execution-feedback --adaptive-task --model logs/v3e_262420/checkpoints/sac_mpc_20000_steps.zip --output eval/v3e/early/262420_20k.json
python -B -m experiments.evaluate_hybrid_policy --episodes 12 --seed 264100 --horizon 35 --parametrization task_state_v3 --phase-time-observation --execution-feedback --adaptive-task --control v3_nominal --output eval/v3e/early/nominal_264100.json
```

## 第 4 步：训练后的基准行（可并行，每行一个进程）

```powershell
# Pure MPC（预期 37/48）
python -B -m experiments.evaluate_hybrid_policy --episodes 48 --seed 262000 --horizon 35 --parametrization arrival_condition --adaptive-task --control desired_pose --output eval/v3e/pure_mpc.json
# 平缓推进（参考实现本身的效果）
python -B -m experiments.evaluate_hybrid_policy --episodes 48 --seed 262000 --horizon 35 --parametrization task_state_v3 --phase-time-observation --execution-feedback --adaptive-task --control v3_nominal --output eval/v3e/nominal.json
```
每个模型两行 learned-only（60k 主结果；50k 用于和 v3d 同预算对比）：
```powershell
python -B -m experiments.evaluate_hybrid_policy --episodes 48 --seed 262000 --horizon 35 --parametrization task_state_v3 --phase-time-observation --execution-feedback --adaptive-task --model logs/v3e_262420/final_model.zip --output eval/v3e/262420/learned_only.json
python -B -m experiments.evaluate_hybrid_policy --episodes 48 --seed 262000 --horizon 35 --parametrization task_state_v3 --phase-time-observation --execution-feedback --adaptive-task --model logs/v3e_262420/checkpoints/sac_mpc_50000_steps.zip --output eval/v3e/262420/learned_only_50k.json
```
（262421、262422 同理。）Pure MPC 若不是 37/48，停，报上层。

## 第 5 步：方法流水线（每个模型分别做；以 262420 为例）

**M2 价值数据**（4 个进程并行）：
```powershell
python -B -m experiments.v3_collect_value_data --run-dir logs/v3e_262420 --seeds 270000-270011 --output eval/v3e/262420/m2_a
python -B -m experiments.v3_collect_value_data --run-dir logs/v3e_262420 --seeds 270012-270023 --output eval/v3e/262420/m2_b
python -B -m experiments.v3_collect_value_data --run-dir logs/v3e_262420 --seeds 270024-270035 --output eval/v3e/262420/m2_c
python -B -m experiments.v3_collect_value_data --run-dir logs/v3e_262420 --seeds 270036-270047 --output eval/v3e/262420/m2_d
```
**M3 价值拟合**（分钟级；默认按任务效用拟合）：
```powershell
python -B -m experiments.v3_fit_values --data eval/v3e/262420/m2_a.npz eval/v3e/262420/m2_b.npz eval/v3e/262420/m2_c.npz eval/v3e/262420/m2_d.npz --output-dir eval/v3e/262420/values
```
**M6 校准**：
```powershell
python -B -m experiments.v3_calibrate_values --run-dir logs/v3e_262420 --values eval/v3e/262420/values --output eval/v3e/262420/m6.json
```
- `PASS` → 跑 M5；
- `INCONCLUSIVE`（决定性检查点少于 5 个）→ 可以跑 M5，但交回时单列，不得声称已校准；
- `STOP` → 该模型不跑 M5，报上层（其他模型照常）。

**M5 仲裁评估**：
```powershell
python -B -m experiments.evaluate_hybrid_policy --episodes 48 --seed 262000 --horizon 35 --parametrization task_state_v3 --phase-time-observation --execution-feedback --adaptive-task --model logs/v3e_262420/final_model.zip --arbiter one_way --values eval/v3e/262420/values --output eval/v3e/262420/arbitrated.json
```

**判读**（三个模型都完成后）：
```powershell
python -B -m experiments.v3_readout --pure-mpc eval/v3e/pure_mpc.json --model 262420=eval/v3e/262420/learned_only.json,eval/v3e/262420/arbitrated.json --model 262421=eval/v3e/262421/learned_only.json,eval/v3e/262421/arbitrated.json --model 262422=eval/v3e/262422/learned_only.json,eval/v3e/262422/arbitrated.json --output eval/v3e/readout.json
```

**并行建议**：第 4 步全部行（2 + 6）与第 5 步各模型的 M2（12 个进程）互不依赖，按机器核数排队即可；M3 → M6 → M5 按模型顺序进行。

## 第 6 步：交回内容

1. v3d：三个 `learned_only_50k_*.json`、停止时的健康检查输出；
2. v3e：三个 manifest、各次健康检查输出、三个 `train.monitor.csv` 的 SHA-256；
3. `eval/v3e/` 下全部 JSON（pure_mpc、nominal、learned_only ×3、learned_only_50k ×3、m3_report ×3、m6 ×3、arbitrated ×3、readout）及其 SHA-256；
4. `v3_readout` 打印的末 4 行；
5. 模型 ZIP、M2 的 `.npz`、`values/*.pt` 不入库，只报路径与 SHA-256。

## 禁止事项

- v3d 评估必须在 `b05e389`（不 pull 的目录）上完成；
- 不续训、不复用运行名、不加载旧 checkpoint 训练；
- 不改奖励、MPC、任务、超参、限速、近场下限、仲裁 z；
- 不按中途结果调整任何东西，不挑 checkpoint（主结果只用 `final_model.zip`；50k 只用于与 v3d 同预算对比）；
- 工作树不干净时不开跑。
