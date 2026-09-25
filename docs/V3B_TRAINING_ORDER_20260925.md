# v3b 正式训练指令（T0b 完成后）

*2026-09-25，上层窗口。分支 `claude/sac-mpc-coupling-design-ns7g6i`。**训练代码提交：`85dd37ac7fbf3fb47e17791a8b7b652be49e3231`**（之后只有文档提交）。全套测试 308 passed、3 xfailed。*

---

## 0. 训练前已完成的事（上层接管后处理完毕）

| 项 | 状态 | 依据 |
|---|---|---|
| T0b 实现（下层 `f50eb78`：方向状态限速、候选只读、单次提交、42 维观测、旧模型适配器仅限压测） | 已审查、已合并（`46d787c`） | 代码审查逐条对照 `docs/V3_T0B_BLEND_FIX_ORDER_20260925.md` |
| 参考连续性 | 通过 | 下层真实压测 36/36 回合、动态上界违例 0（产物在用户机器上，见第 6 节）；上层离线压测 108 条序列、违例 0（`eval/adp/v3_t0b_blend_stress.json`） |
| Pure MPC 架构地板 | 通过 | 下层 12 回合逐控制步 wrench 完全相等（同上，产物在用户机器上） |
| **QP 回退门** | **上层裁定：T0b 不增加 QP 回退，门以新形式保留为训练健康监测** | 见下 |
| 全套回归 | 通过 | 本指令的提交上 `pytest` 全绿（数字见提交说明） |
| 训练入口 smoke | 通过 | 42 维观测、2 维动作、`training_branch=learned`、manifest 记录 `code_commit` |

### QP 门的裁定

下层原门槛是"QP 回退率不高于 V2 评估水平"，拿来比较的是：旧 39 维模型经适配器跑在 42 维接口上，对比 V2 的训练好模型。
**这不是同条件比较**：旧模型看不到新增的方向状态，也不是在这个接口上训练出来的。

上层做了同条件比较（`experiments/probe_v3_qp_matched.py`，产物 `eval/adp/v3_t0b_qp_matched.json`）：
同一组 12 个种子，同一串随机任务动作，分别送进 T0 环境（`41c59bb`）和 T0b 环境（`f50eb78`），不用模型。

| 动作类型 | T0 QP 零推力回退 | T0b QP 零推力回退 | T0 最大参考跳变 | T0b 最大参考跳变 |
|---|---:|---:|---:|---:|
| 均匀随机（训练起点的探索） | 1 / 28,920 | 1 / 32,420 | 24.78 m | 2.87 m |
| 偏向接近与承诺的随机 | 1 / 29,000 | 1 / 24,320 | 14.02 m | 2.40 m |

**结论**：T0b 没有提高 QP 回退率（两者都在 3–4 × 10⁻⁵，与 V2 评估的 2.3 × 10⁻⁵ 同量级），同时消除了大跳变。
262421 旧模型的 283 次回退，是"旧策略经适配器跑在它没训练过的接口上"造成的，不是接口本身的问题。
因此不做 263002 的进一步匹配诊断。QP 健康改为**训练中持续监测**（第 3 节），有硬停止规则。

---

## 1. 预检（开跑前，在仓库根目录，PowerShell）

```powershell
git fetch origin claude/sac-mpc-coupling-design-ns7g6i
git checkout claude/sac-mpc-coupling-design-ns7g6i
git pull --ff-only origin claude/sac-mpc-coupling-design-ns7g6i
git rev-parse HEAD                 # 训练代码提交为 85dd37ac7fbf3fb47e17791a8b7b652be49e3231；若 HEAD 更新，git diff 85dd37ac7fbf3fb47e17791a8b7b652be49e3231 HEAD --stat 只能出现 docs/ 下的文件
git status --porcelain --untracked-files=no   # 必须为空
python -B -m pytest -q tests/test_v3_task_interface.py tests/test_check_v3b_training_health.py
```

测试必须全绿；工作树必须干净（manifest 会记录 `code_dirty`，不干净的运行作废）。

旧运行 `v3_262420 / 421 / 422` 保持原样（已有 `CONTAMINATED.md`），**不续训、不复用运行名、不加载其 checkpoint**。

## 2. 启动（三个 PowerShell 窗口，一个种子一个）

每个窗口先钉线程，再运行对应命令：

```powershell
$env:OMP_NUM_THREADS="1"; $env:MKL_NUM_THREADS="1"; $env:OPENBLAS_NUM_THREADS="1"
python -u -B -m train.train_hybrid --steps 60000 --seed 262420 --run-name v3b_262420 --horizon 35 --parametrization task_state_v3 --adaptive-task --device cpu
```
```powershell
$env:OMP_NUM_THREADS="1"; $env:MKL_NUM_THREADS="1"; $env:OPENBLAS_NUM_THREADS="1"
python -u -B -m train.train_hybrid --steps 60000 --seed 262421 --run-name v3b_262421 --horizon 35 --parametrization task_state_v3 --adaptive-task --device cpu
```
```powershell
$env:OMP_NUM_THREADS="1"; $env:MKL_NUM_THREADS="1"; $env:OPENBLAS_NUM_THREADS="1"
python -u -B -m train.train_hybrid --steps 60000 --seed 262422 --run-name v3b_262422 --horizon 35 --parametrization task_state_v3 --adaptive-task --device cpu
```

- 超参全部用默认值（与 V2 相同：γ 0.99、`ent_coef` 0.005 固定、`n_critics` 2、学习率、batch、网络都不改），**不加任何其他参数**；
- checkpoint 每 5,000 步自动保存；
- **训练期间窗口不能关**。上一批就是因为终端关闭，停在了 13.1k。建议开跑后把窗口最小化，不要关闭，也不要让系统睡眠。

**开跑 1 分钟后检查 manifest**（每个运行各一次）：
```powershell
python -c "import json,sys;m=json.load(open(sys.argv[1]));print({k:m.get(k) for k in ('code_commit','code_dirty','observation_dimension','action_dimension','training_branch','waypoint_parametrization','seed')})" logs/v3b_262420/manifest.json
```
必须是：`code_commit` 等于第 1 节记下的值，`code_dirty` 为 False，`observation_dimension` 42，`action_dimension` 2，`training_branch` "learned"，`waypoint_parametrization` "task_state_v3"。有任何一项不对，立刻停掉该运行并报告。

## 3. 训练中的健康检查（预注册，只有两条停止规则）

在 **10k、20k、30k、40k、50k** 附近各跑一次（只读，不影响训练）：
```powershell
python -B -m experiments.check_v3b_training_health logs/v3b_262420 logs/v3b_262421 logs/v3b_262422
```

| 规则 | 条件 | 动作 |
|---|---|---|
| **S1 参考跳变** | 任一回合 `reference_jump_violations > 0` | **停止该运行并报告**：T0b 的上界是可证明的，出现违例就是 bug |
| **S2 QP 健康** | 最近 100 个回合的零推力回退率 > 1 × 10⁻³ | **停止该运行并报告**（同条件对照约 4 × 10⁻⁵，这个门槛有 25 倍余量） |

脚本还会打印最近 100 回合的完成数、平均回报、最大参考跳变，**这些只做记录，不是停止依据**：
- 完成率低、某个种子长期 0%，都**不停、不改超参、不改奖励、不丢种子**。训练噪声是预期内的，结论只在 3 个种子的正式评估上下；
- 不挑 checkpoint，不续训，不因为中途数字好看或难看而改动任何东西。

## 4. 训练完成后

1. 确认三个运行目录都有 `final_model.zip`，manifest 中 `status` 为 completed、`actual_decision_steps` 为 60000；
2. 最后跑一次第 3 节的健康检查；
3. 按 `docs/V3_METHOD_EXECUTION_ORDER_20260924.md` 的 **M2 → M3 → M6 → M5** 继续（数据收集、价值拟合、校准检查、正式评估），该执行单不变。
   注意：M2 数据与 V_L / V_B 的输入一律用 42 维观测；评估器用 manifest 里的 `evaluation_flags`，不要手写。

## 5. 交回内容（训练结束时）

1. 三个 manifest（完整文件）；
2. 三次以上健康检查的输出（至少 10k、30k、60k）；
3. 每个种子末 100 回合的完成数、平均回报、最大参考跳变、QP 回退率；
4. 有无任何停止事件，若有，附当时的 monitor 与报错；
5. 三个 `train.monitor.csv` 的 SHA-256（模型 ZIP 不入库，只报路径与 SHA-256）。

## 6. 待补的证据（不阻塞训练）

下层的 T0b 真实压测与地板产物目前只在用户机器上，尚未入库：
`eval/adp/v3_t0b_acceptance_26241{0,1,2}.json`（旧 checkpoint 名为 262420/421/422）、`eval/adp/v3_t0b_baseline_pure.json`、`eval/adp/v3_t0b_baseline_v3.json`、`eval/adp/v3_t0b_matched_263002.json`。
方便时逐个 `git add -f` 并记下 SHA-256，补进本文件。**在入库前，这些数字只作内部依据，不进论文。**

## 7. 禁止事项

- 不续训旧 `v3_*` 运行，不复用旧运行名，不加载旧 checkpoint；
- 不改奖励、MPC、限速上界、任务参数、超参；
- 不因训练中的完成率调整任何东西；
- 工作树不干净时不开跑。
