# 执行书：RUN_FORMAL_PREP（正式实验前的全部准备：代码适配、日志、读出、出图、回归与冒烟、燃料压力预筛）

**状态：用户 2026-11-10 批准（"开始测试交付下层，在正式实验之前统筹好所有工作"）。本执行书不包含正式训练，也不访问正式块（300000 以上）。**

**可直接转发给下层的一句话：**

> 读 collab `docs/collaboration/upper/run_orders/RUN_FORMAL_PREP.md`。在 `D:\py\DRL2` 从科学分支 `claude/sac-mpc-coupling-design-ns7g6i` 最新提交（不早于 `6c76c2c9`）开始，按 WP1–WP8 顺序实现与测试。每个工作包单独提交，提交前 pytest 退出码为 0；WP7 的回归与冒烟全部通过后再跑 WP8 预筛。完成后按"交付"推送并写 RESULT.md。不改冻结方法的任何行为。

---

## 依据与硬约束

**依据：**
- `docs/paper/EXPERIMENTS.md`（预注册的实验规格）；
- `docs/paper/METHOD.md`、`docs/paper/FACT_MAP.md`；
- 冻结实现 `e0a617532008dac02e2d50f7e859aca613e03e5e`。

**硬约束（违反任何一条，工作包作废重做）：**
1. **冻结行为不变。** 不改奖励、观测、网络、学习器、认证规则、RWG、MPC、超参数。ECC_CERT_LOOP、SAC_C、EACH_C 三条已有路径对 `e0a61753` 逐位等价（环境层与冒烟训练，WP7）。
2. **日志只读。** 新增记录只能读取已经算出的量（例如 `_opportunity_eval()` 的缓存结果、`T1Env._min_margins`、`env.metrics()`），不得为记录而额外调用预测、求解器或随机数。
3. **不访问正式块。** 300000 以上的初始条件，在本执行书中一个都不跑。
4. **pytest 以退出码判定。** 每个工作包新增的测试写进 `tests/test_t2_stage.py`（或新文件），全部测试退出码 0 才提交。
5. **只用科学分支。** 每个工作包一个提交，提交信息写明工作包编号；不开新分支，不强推。

---

## WP1　注册表：新增配置与正式阶段

在 `docs/final_method/METHOD_REGISTRY.json`：

1. **新增训练配置**（展示名用论文名）：

| 键 | 展示名 | arm | obs | window | geometry | certify | actor / critic |
|---|---|---|---|---|---|---|---|
| `SAC_CERT_LOOP` | ECC w/o execution feedback | `e_sac` | `cut_priv` | true | true | true | 24 / 24 |
| `ECC_CERT_LOOP_NOGEO` | ECC w/o rotation guidance | `e_priv` | `cut_priv` | true | **false** | true | 24 / 48 |
| `TD3_C` | TD3–MPC | `e_td3`（WP2） | `cut_priv` | true | true | false | 24 / 24 |

   已有配置的展示名改为论文名：`ECC_CERT_LOOP` → "ECC"，`SAC_C` → "SAC–MPC"，`EACH_C` → "ECC w/o certification"。只改 `display` 字段，不改其他字段。

2. **原 `formal` 阶段**（EACH 时期）改名为 `formal_each_obsolete`，加注"作废，见 docs/paper/EXPERIMENTS.md"。

3. **新 `formal` 阶段**（按 `EXPERIMENTS.md` 第 2–4 节）：

| 字段 | 值 |
|---|---|
| `steps` | 5000 |
| `seeds` | [20, 21, 22] |
| `checkpoints` | [1000, 2000, 3000, 4000] |
| `mission`（训练） | `urgent_mix:0.75` |
| `trained` | `ECC_CERT_LOOP`、`SAC_C`、`TD3_C`、`EACH_C`、`SAC_CERT_LOOP`、`ECC_CERT_LOOP_NOGEO` |
| `conditions` | `urgent`: 300000–300095 / `urgent`；`economy`: 300100–300195 / `economy`；`natural`: 300200–300295 / `natural`；`resource`: 300100–300195 / `economy` + `budget_m_s: null`（WP8 冻结后填值，或删除） |
| `validation` | 295200–295223，`urgent`（检查点验证曲线，只用于 ECC、SAC_C、TD3_C） |
| `fixed` | `pure`、`gstar` |
| `eval_switches` | `ECC_CERT_LOOP`: `nocert`（U、E）、`rigid`（U）；`EACH_C`: `cert`（U、E）；`SAC_C`: `cert`（U、E） |
| `evaluations` | 全部评估目录名（命名规则见 WP4） |

4. **新阶段 `formal_prescreen`**（WP8 用）：
   - 初始条件 295100–295147，`economy`；
   - 固定策略：`rally`、`c_10_0`、`c_10_-5`、`c_6_5`、`b_10`、`gstar`；
   - 预算候选 [2.0, 1.8, 1.6]。

**测试：**
- 新配置通过 `verify_manifest`；
- `formal` 各条件块互不重叠，且与 `formal_prescreen`、`validation`、所有开发阶段块不重叠；
- `formal_prescreen` 不含 300000 以上编号。

---

## WP2　TD3–MPC 基线（arm `e_td3`）

在 `train/t2_mve_sac.py` 中实现，**不改 SAC 任何代码路径**。

**模型：** `T2TD3`，基于 stable-baselines3 的 `TD3`，策略为 `MlpPolicy`，`net_arch = [256, 256]`。参数：
- 学习率 3e-4，经验池 100,000，批量 256（冒烟 64），起步 200（冒烟 16）；
- γ = `GAMMA_STAGE`，τ = 0.005；
- `gradient_steps = 2`（每条转移），`policy_delay = 2`，`target_policy_noise = 0.2`，`target_noise_clip = 0.5`。

**与训练主循环的衔接：**
- 主循环访问的属性（`acc`、`feedback`、`model_beta`、标签相关字段等）在 `T2TD3` 上置为"关闭"的取值，不进入任何标签 / 反馈分支；
- **探索噪声：** 起步前与 SAC 相同，用均匀随机动作。之后动作 = clip(`predict(obs, deterministic=True)` + N(0, 0.1²), −1, 1)，噪声来自按种子初始化的独立随机数生成器（例如 `np.random.default_rng([seed, 3003])`），不得使用全局随机数。**原因：** SB3 的 `TD3.predict` 本身不加探索噪声；
- 采集进程的策略同步、检查点保存、`manifest.json` 写出与 SAC 相同；manifest 记录 `arm: e_td3` 以及上述 TD3 参数。

**评估：** `experiments/t2.py` 的 `eval_job` 按 manifest 的 `arm` 选择加载类（`T2TD3.load` 或 `T2SAC.load`），评估一律用确定性动作。

**测试：**
- 起步后动作含噪声、噪声可复现（同种子同序列）；
- 评估为确定性；
- 每条转移 2 次更新，策略每 2 次更新才更新一次；
- 训练 40 步（1 个采集进程）并评估 2 个开发初始条件，用命令行跑通（smoke）。

---

## WP3　只读日志

### 3.1 每次高层决策（写入 `info`，由 `experiments/t2.py: run_episode` 写进 `decisions`）

**所有运行都记录：**

| 字段 | 内容 |
|---|---|
| `t_start` | 本次决策开始时刻 |
| `remaining_s` | 期限 − t_start |
| `remaining_dv` | 预算 − 已用 Δv |
| `phase_rad`、`time_to_pass_s`、`lat_err_deg` | 决策开始时 `spin_frame` 的值（只读，用 `frame_now`） |
| `macro_end` | `stage_complete`、`deviation_hold`、`entry`、`descent`、`terminal` 之一 |
| `true_mode_event_t` | 本宏步内第一次进入 `bridge` 或 `descent` 的真实时刻；没有则为 null |

**认证开启时，扩展 `info['cert']`：**

| 字段 | 内容 |
|---|---|
| `cands` | 9 项，逐项来自 `self._opp['rows']`（`_opportunity_eval` 的缓存）：`b`、`event`、`kind`、`t`、`dv`、`dv_entry`、`certified`，以及 `binding`（该行的 t + τ̂ + P > L 判定，用已缓存的 `period`、`latest` 计算，不重新预测） |
| `own_kind`、`own_dv` | 原意图预测的事件类别与燃料 |
| `wall_s` | 本次认证的串行耗时（`perf_counter`） |

**禁止：** 不得为了记录而多调用一次 `probe_intent`。原意图的预测结果在 `_certify` 内已经计算，直接写出即可。

### 3.2 每回合（`run_episode` 返回值与训练 `episodes.json`）

| 字段 | 内容 |
|---|---|
| `condition` | `urgent`、`economy`、`natural`、`resource`、`validation`、`prescreen` |
| `budget_m_s` | 本回合实际使用的预算 |
| `min_margins` | `T1Env._min_margins` 的拷贝（各约束分别的最小带符号裕度） |
| `terminal` | 回合结束时 `env.metrics()` 的位置误差、速度、姿态误差、角速度误差、是否在锥内 |
| `cert_wall_s_total`、`cert_decisions` | 认证耗时累计与次数 |

训练 `episodes.json` 增加 `min_margins` 与 `terminal`，不改已有字段。

### 3.3 案例时序（`experiments/final.py: fly`）

`rows` 每 2 s 再增加：
- 推力与力矩：当步实际下发的指令，从控制器返回值读取；
- 姿态误差、角速度误差：从 `metrics()` 读取；
- 各约束的当前带符号裕度；
- 累计 Δv。

`decisions` 增加与 3.1 相同的字段。

**测试：**
- 对同一初始条件、同一固定动作序列，日志开 / 关时观测哈希、回报、事件、时间逐步相同（日志无副作用）；
- `cands` 的 9 项与 `_opportunity_eval()` 返回值逐项一致；
- 第 3.2 节字段齐全。

---

## WP4　评估入口的选项

在 `experiments/final.py` 中：

1. **`eval --condition {urgent, economy, natural, resource, validation}`：** 按 `formal` 阶段的块与任务。`resource` 使用冻结预算（WP8 之后才可用；未冻结时报错）。
2. **`eval --budget B`：** 只允许与 `formal_prescreen` 或冒烟一起使用。实现为评估时覆盖任务的 `budget_m_s`；不改训练。
3. **`eval --checkpoint-step N`：** 评估指定检查点。只允许 `--condition validation`（学习曲线用）。
4. **`screen --condition ...`** 与 **`screen --budget B`：** 固定策略（含 `pure`、`gstar`）按条件或预算筛查。
5. **目录命名规则（写进注册表的 `evaluations`）：**
   - `C/<配置>_s<种子>_<条件>`；
   - 开关评估加后缀：`_nocert`、`_cert`、`_rigid`；
   - 验证曲线：`V/<配置>_s<种子>_step<N>`；
   - 固定策略：`S/<策略>_<条件>`。
6. **`check --stage formal`：** 按以上清单统计完整性。燃料压力 R 被取消时，`resource` 不计入。
7. **中断后补跑：** 结果文件已存在的初始条件自动跳过（已有行为）。另加 `--verify-existing`：对已有文件核对 SHA256 清单，不一致时报错。

**测试：** 各选项的拒绝规则（例如 `--checkpoint-step` 不能用于 `urgent`，正式条件不能带 `--budget`）。

---

## WP5　读出与统计（新文件 `experiments/formal_readout.py`）

按 `EXPERIMENTS.md` 第 5、6 节实现，全部只读原始 JSON。

1. **主终点：**
   - U 上 ECC 对 SAC–MPC 的配对捕获率差：先给各种子，再给合并值；
   - 区间：95% 自助法，10,000 次，按种子分层、按初始条件成对重采样，随机数种子固定为 20261110；
   - 判定：区间下界 > 0，且 ≥ 2 个种子的差为正 → `significant_increase`；否则 `not_established`。
2. **次终点：**
   - U 真实回报差（均值、中位数、区间）；
   - E 共同成功配对的 Δt、Δv（附样本数），以及各方法捕获数与回报；
   - N 汇总（含实际 U / E 比例）；
   - R：成功率、耗尽率、共同成功的 Δv。
3. **对照：**
   - TD3–MPC 与纯 MPC 对 ECC；纯 MPC 每个初始条件只计一次，不按种子重复；
   - 2 × 2 四格的绝对捕获率与相对 SAC–MPC 的效应；
   - 只评估开关的救回 / 丢失翻转（按最终结局定义）；
   - w/o 旋转引导；
   - 重规划开 / 关：入口恢复、丢失、捕获翻转。
4. **物理可信度：** 成功回合的终端误差分布；各约束最小裕度分布（分开报告）；不安全计数；认证耗时（均值 / P95 / 最大）。
5. **解释轴：**
   - **N_geo：** 用精确相位半宽 Δφ(λ, β)。在 `env/t2_stage.py` 新增**纯函数** `window_half_width(lam, beta, theta_g)`，处理 c* > 1、c* ≤ −1、sin λ sin β = 0；t_arr 沿用 `opportunity_windows` 的名义接近模型。分层 0 / 1 / ≥ 2，某层 < 10 个时并入相邻层。
   - **η：** 只用于 ECC 的认证日志 `cands`：取 `event = 1` 且燃料够用者的最早 τ̂；分层 η < 0、[0, 1)、≥ 1，另列"预测时域未覆盖"；入口与下降分开。
   - **机制三项：** 修改率 = 修改数 / 访问数；合格候选数 / 9 与事件类型；同模型开 / 关的成功翻转。
6. **学习曲线与收敛读数：** 从 `episodes.json` 取训练真实回报的滑动均值，计算最后 1000 次决策的回报斜率与紧急捕获率变化；验证曲线读 `V/`。
7. **燃料压力预筛判定：** 按 `EXPERIMENTS.md` 第 2 节的规则输出 `freeze_budget` 或 `cancel`。

**测试：** 用合成数据检验配对、分层自助法（已知答案）、主终点判定、翻转计数、纯 MPC 不重复计数、N_geo 的边界情形、预筛规则。

---

## WP6　出图（新文件 `experiments/formal_figures.py`）

由 `formal_readout.py` 的 JSON 与原始日志一键生成 F1–F7、RWG 几何小图、T1–T3（表格写 CSV 与 Markdown），按 `EXPERIMENTS.md` 第 7 节的面板与规范：
- 色盲友好配色；形状 / 线型区分方法；
- 曲线输出矢量 PDF / SVG，三维图输出 300 dpi PNG；
- 图内标单位、约束边界、样本数；
- 每张图旁写出同名 JSON（图中每个数字）。

**案例选取：** 按第 7 节规则实现为函数，并输出所选编号；集合为空时按退路规则选取，并在输出中注明。

**测试：** 在冒烟数据（WP7）上一键生成全部图表不出错；案例规则在合成数据上选对编号。

---

## WP7　回归与冒烟（全部通过才能进入 WP8）

**输出目录：** `C:\Users\35884\Documents\Spacecraft\过程文件\formal_prep`。

1. **pytest：** 全部测试退出码 0，日志存 `pytest.log`。

2. **逐位回归（对 `e0a61753`）：**
   - 在单独目录 detached 检出 `e0a61753`（例如 `D:\py\DRL2_ref`，用完删除）；
   - 用 `evidence/ECC_FINAL_PRECHECK/scripts/env_equiv.py`（环境层）与 `train_equiv.py`（冒烟训练：`--stage ecc --config {ECC, EACH_C, SAC_C} --seed 10 --smoke-steps 40 --workers 1`），以及 `gov_b` 阶段的 `ECC_CERT_LOOP`，比较新旧两侧；
   - **要求：** 环境层逐步一致；冒烟训练各检查点参数逐位一致、回合与训练统计一致。
   - **说明：** 日志新增字段会使 `episodes.json` 多出字段。比较时只比对 `e0a61753` 已有的字段，并在报告中注明。

3. **训练冒烟（非正式块）：** 6 个正式配置各 `--smoke-steps 40`，其中 ECC_CERT_LOOP 与 TD3_C 各再用 2 个采集进程跑一次；核对 manifest 的 arm、obs、视图、certify、geometry、mission。

4. **评估冒烟：** 用冒烟模型，在开发初始条件（294300–294301）上，每种评估形式各跑 2 个初始条件：
   - 按训练方式；
   - `--no-certify`、`--certify`、`--rigid`；
   - `--checkpoint-step`；
   - `--budget`，其中至少一例用很低的预算（例如 0.3 m/s）制造燃料耗尽；
   - 固定策略 `pure`、`gstar`。

5. **完整终局：** 至少一例回合以捕获或超时自然结束（冒烟模型足够）。

6. **中断补跑：** 一次评估在中途强行终止，再次运行只补缺失的初始条件，`--verify-existing` 通过。

7. **端到端：** 对冒烟输出运行 `formal_readout.py` 与 `formal_figures.py`，全部图表生成。

8. **已有模型的日志无副作用：** 用开发期 B 步的 ECC 模型（`gov_b\B\TRAIN_ECC_CERT_LOOP_s10`），在 294800–294801 上分别用 `e0a61753` 与新代码评估，事件、回报、时间、决策序列一致。

---

## WP8　燃料压力 R 的预筛（只用非学习策略）

- **筛查条件：** `formal_prescreen`：295100–295147，经济任务。
- **执行：** 按 2.0 → 1.8 → 1.6 m/s 的顺序，对 `rally`、`c_10_0`、`c_10_-5`、`c_6_5`、`b_10`、`gstar` 运行 `screen --stage formal_prescreen --budget B`。
- **判定：** 每个预算跑完后，立刻用 `formal_readout.py prescreen` 判定。第一个同时满足以下两条的预算即冻结，后面的预算不再跑：
  1. 五个固定阶段策略合计 ≥ 10% 的回合燃料耗尽，或用掉 ≥ 90% 预算；
  2. 逐开局取最好的固定阶段策略，捕获率 ≥ 0.80。

  三档都不满足则取消 R。
- **冻结：** 把结果写进注册表 `formal.conditions.resource.budget_m_s`（或删除 `resource`）。这一项单独提交，提交信息写明判定。
- **纪律：** 不运行任何学习模型；不看任何学习方法的结果。

---

## 交付

**科学分支：**
- WP1–WP8 的提交；
- `evidence/FORMAL_PREP/`：`pytest.log`、`ENV_EQUIVALENCE.txt`、`TRAIN_EQUIVALENCE.txt`、`B_MODEL_LOG_EQUIVALENCE.json`、`DRY/`（冒烟 manifest 与评估文件）、`figures_smoke/`、`PRESCREEN/`（原始结果与 `PRESCREEN_DECISION.json`）、`ENV_LOCK.txt`、`SHA256SUMS`。

**collab：** `docs/collaboration/lower/handoffs/formal_prep/RESULT.md`，逐项写明：
- 每个工作包的提交号；
- 测试数与退出码；
- 逐位回归结果；
- 冒烟清单（逐项通过 / 失败）；
- 预筛判定与冻结预算；
- 遇到的问题与处理方式。

**交付后停止。** 上层审代码与证据后，编写 `RUN_FORMAL`（冻结 SHA、命令、输出路径、验收与补跑规则），交用户单独授权。
