# v3e 改进方案与执行指令

*2026-09-27，上层窗口。分支 `claude/sac-mpc-coupling-design-ns7g6i`。**v3e 代码提交：`a362d4398d35405a83dce7f8ba7de19c0e6d9497`**。全套测试 335 passed、3 xfailed。*
*v3e 是主实验；v3d 保留为"参考传递修复前"的对照。依据：审查窗口两份讨论（9-27）与本文第 A 部分的三项检测。*

---

# A 部分：为什么这样改（给用户与审查窗口）

## A1. 三项检测的结论

**检测 A：参考传递本身能解决多少 Pure MPC 失败？**（`eval/adp/v3_headroom_readout.json`，开发块 265000–265047）
- Pure MPC 38/48。平缓推进（接口全速推进、不等待、不学习）救回 10 个失败中的 8 个，抽样的 12 个成功开局也全部完成；
- 代价：成功开局上平均慢 92 s、多耗 0.93 m/s；
- 两个开局任何策略都救不回；暂存类策略在一半成功开局上能省 14%–37% 燃料。

**检测 B：任务对目标翻滚相位是否敏感？**（`eval/adp/v3e_preflight_probes.json`）
同一开局只改变目标初始相位（绕自身转轴转 0°–315°，其余不变），分别用 Pure MPC 和平缓推进飞：

| 基础开局 | Pure MPC 完成 | 平缓推进完成 | 谁能完成（按相位 0,45,…,315°） | 燃料随相位变化 |
|---|---|---|---|---|
| 265000 | 6/8 | 7/8 | 两者 / 两者 / **仅平缓** / 都不行 / 两者 ×4 | Pure 1.9–5.7，平缓 5.6–10.3 m/s |
| 265012 | 0/8 | 6/8 | 仅平缓 ×3 / 都不行 ×2 / 仅平缓 ×3 | 平缓 5.3–7.1 m/s |
| 265030 | 5/8 | 8/8 | **仅平缓** / 两者 ×5（Pure 快约 50 s） / **仅平缓** ×2 | Pure 1.3–2.0，平缓 1.2–5.2 m/s |

读法：
- **成败随相位翻转**，两种控制都一样；
- **燃料随相位差 2–5 倍**；
- **哪种控制更好取决于相位**：有的相位只有平缓推进能完成；有的相位两者都能完成，但 Pure MPC 快约 50 s、燃料相当；平缓推进在某些相位也会失败。

所以平缓推进不是"横扫"：它有时救命，有时又慢又费油，有时也失败。
**任务本身存在真实的、随状态变化的长期决策（审查窗口的"情况 A"）→ 任务不改，直接做 v3e。** 不引入旋转安全接近区或非球形外形。

**检测 C：修好闸门和参考传递后，近场下限还需要吗？**（同一产物）
脚本"全速推进但承诺停在 c"的策略，最容易触发近场无解：

| 设置 | 回合 | 零推力回退率 | 出现回退的回合 |
|---|---:|---:|---:|
| c = 0.3，有下限 | 6 | 0 | 0 |
| c = 0.3，无下限 | 6 | **1.9 × 10⁻³** | 2（13、12 步） |
| c = 0.6，无下限 | 6 | 7 × 10⁻⁵ | 1（1 步） |

去掉下限后，"推进但不对准"仍会在 6 m 内造成可复现的 MPC 无解（高于训练健康门 1 × 10⁻³）。
→ **保留下限，但重新定位**：它不是策略规则，而是"MPC 可行参考范围"——承诺不足时，参考点不进入入口球。它不规定等多久、何时进入、是否后退，这些仍由 SAC 决定。

## A2. v3e 相对 v3d 改了什么（只动接口正确性与价值定义，不加策略规则）

| # | 改动 | 性质 | 影响 Pure MPC |
|---|---|---|---|
| 1 | **参考实现**：MPC 拿到参考点在预测窗内的真实轨迹（按接口自身规则连续计算），并修正决策内"当前参考冻结"的问题 | 执行一致性（此前惯性暂存被迫多耗约 ω²r 推力） | 否，逐位不变（已验证） |
| 2 | **价值目标**：M3/M6 改用"任务效用"——不打折、不含势能整形的任务奖励之和（完成 ±20、时间、执行器、安全告警） | 在任何 v3e 价值数据产生前定死；不引入新权重 | 否 |
| 3 | 评估程序新增 `--control v3_nominal`（平缓推进行） | 用来把"参考实现的效果"与"学习策略的效果"分开 | 否 |
| 4 | 近场下限保留，重新定位为可行参考范围 | 无代码变化 | —— |

**不变**：任务、奖励、MPC、超参、训练步数（60k）、种子（262420/421/422，与 v3d 对应，便于对照）、健康门、仲裁规则（z = 1）。

## A3. 实验证据链（对应审查窗口的 E1–E6）

| | 内容 | 来源 |
|---|---|---|
| E1 互补性 | Pure MPC 与只用学习策略的四象限 | M5 的 learned-only 行 |
| E2 长期性 | 短时偏离后交回救不回，长期继续才能保住 | 已有 P2（`eval/adp/v3_p1_p2_readout.json`） |
| E3 理想协调 | 精确价值下的策略选择 | 已有 P2′（`eval/adp/v3_p2prime_readout.json`） |
| E4 学到的价值 | 符号一致率、关键状态、误选 | M6 |
| E5 任务收益 | 成功率 / 燃料 / 时间 / 安全余量 | M5 仲裁行 + 读数 |
| E6 消融 | Pure MPC；平缓推进；v3d 只用学习策略（旧参考实现）；v3e 只用学习策略；v3e 完整方法 | 本指令第 B3、B4 节 |

平缓推进一行的作用只是分清归因，**不是论文要打败的对象**。主张仍然是：长期价值协调利用学习策略与 MPC 的互补，在任务指标上取得净收益。

---

# B 部分：执行指令（给下层）

## B0. 先收尾 v3d（不拉新代码，用当前 `b05e389`）

1. 确认三个 `logs/v3d_*` 都有 `final_model.zip`，最后跑一次健康检查；
2. 在正式评估块上各跑一次"只用学习策略"（三个进程可并行）：
```powershell
python -B -m experiments.evaluate_hybrid_policy --episodes 48 --seed 262000 --horizon 35 --parametrization task_state_v3 --phase-time-observation --execution-feedback --adaptive-task --model logs/v3d_262420/final_model.zip --output eval/v3d/learned_only_262420.json
```
（262421、262422 同理。）
3. 简要交回：三个模型的完成数、Pure MPC 救回 / 弄丢、共同完成回合的耗时与 Δv 差、健康检查末次输出、末 100 训练回合完成率。
4. **v3d 不跑 M2–M5。** 目录与模型原样保留，作为消融对照。

## B1. 预检（v3d 评估完成后）

```powershell
git fetch origin claude/sac-mpc-coupling-design-ns7g6i
git checkout claude/sac-mpc-coupling-design-ns7g6i
git pull --ff-only origin claude/sac-mpc-coupling-design-ns7g6i
git rev-parse HEAD
git diff a362d4398d35405a83dce7f8ba7de19c0e6d9497 HEAD --stat        # 只允许 docs/、eval/ 与 CLAUDE.md
git status --porcelain --untracked-files=no   # 必须为空
python -B -m pytest -q                 # 预期 335 passed、3 xfailed
```

## B2. 训练 v3e（三个窗口，从零，60k）

```powershell
$env:OMP_NUM_THREADS="1"; $env:MKL_NUM_THREADS="1"; $env:OPENBLAS_NUM_THREADS="1"
python -u -B -m train.train_hybrid --steps 60000 --seed 262420 --run-name v3e_262420 --horizon 35 --parametrization task_state_v3 --adaptive-task --device cpu
```
另两个窗口换成 262421、262422。不加其他参数。

- manifest 检查：`code_commit` = B1 的 HEAD、`code_dirty` False、观测 42、动作 2、`training_branch` learned、`timeout_is_terminal` True；
- 健康检查与停止规则同 v3d（S1 参考跳变违例 > 0；S2 最近 100 回合零推力回退率 > 1 × 10⁻³），10k–50k 各查一次；
- 可选早读：20k、40k 在早读块 264100 上跑 learned-only，并加一行 `--control v3_nominal` 作参照。只看，不作任何决定依据。

## B3. 训练后：三行非学习基准 + 三行只用学习策略（可并行）

```powershell
# Pure MPC（与 v3d 时代逐位相同，可复用第 3 节旧结果；若没跑过就跑这一次）
python -B -m experiments.evaluate_hybrid_policy --episodes 48 --seed 262000 --horizon 35 --parametrization arrival_condition --adaptive-task --control desired_pose --output eval/v3e/pure_mpc.json
# 平缓推进
python -B -m experiments.evaluate_hybrid_policy --episodes 48 --seed 262000 --horizon 35 --parametrization task_state_v3 --phase-time-observation --execution-feedback --adaptive-task --control v3_nominal --output eval/v3e/nominal.json
# 只用学习策略（每个模型一行）
python -B -m experiments.evaluate_hybrid_policy --episodes 48 --seed 262000 --horizon 35 --parametrization task_state_v3 --phase-time-observation --execution-feedback --adaptive-task --model logs/v3e_262420/final_model.zip --output eval/v3e/262420/learned_only.json
```
Pure MPC 预期 37/48。

## B4. M2 → M3 → M6 → M5（每个模型分别做）

与 `docs/V3D_EXECUTION_ORDER_20260926.md` 第 4 节相同，只把 `v3d` 换成 `v3e`、输出目录换成 `eval/v3e/<seed>/`：
- M2：`experiments.v3_collect_value_data`，种子 270000–270047，分 4 个进程；
- M3：`experiments.v3_fit_values`（默认 `--target task`，即任务效用）；
- M6：`experiments.v3_calibrate_values`：`PASS` 才跑 M5；`INCONCLUSIVE` 可以跑，但不得声称已校准；`STOP` 报上层；
- M5：`--arbiter one_way --values eval/v3e/<seed>/values`；
- 读数：`experiments.v3_readout --pure-mpc eval/v3e/pure_mpc.json --model 262420=...learned_only.json,...arbitrated.json ...`。

## B5. 交回内容

1. v3d 收尾（B0 第 3 项）；
2. v3e：三个 manifest、健康检查输出、monitor 的 SHA-256；
3. `eval/v3e/` 下全部 JSON（Pure MPC、平缓推进、learned-only ×3、M3 报告 ×3、M6 ×3、arbitrated ×3、readout）及其 SHA-256；
4. 模型 ZIP、M2 的 `.npz`、`values/*.pt` 不入库，只报路径与 SHA-256。

## B6. 禁止事项

- 不续训、不复用运行名、不加载旧 checkpoint；v3d 的 learned-only 评估必须在 `b05e389` 上做（拉取新代码之前）；
- 不改奖励、MPC、任务、超参、限速、近场下限、仲裁 z；
- 不按中途结果调任何东西，不挑 checkpoint；
- 工作树不干净时不开跑。
