# 执行单：T1_O_V2_GATES_LOCAL_20261017（第 1 版；下层本机执行；跑完交上层核实）

**授权：** 用户 2026-10-17 指令：
> 写执行指令给下层，后续测试都本地跑然后让你核实结果，确认后本地最终跑训练。确保下层理解我们当前任务状态。

**科学提交（冻结）：** `d3902a9dd62dbef6cb318e36ff4c2527d7340ec9`
- 分支：`experiment/t1_o_tcpf_20261017`，起点 `672ba79`。
- 全部判据在 `89b779a` 中预注册：`experiments/t1o_v2.py` 的模块说明。本单只按冻结规则机械执行，不得修改任何判据或参数。

**本单内容：**
- 在本机复现 Gate A；
- 依次跑 Gate B、Gate C（含零步检查）、Gate E；
- 遇到第一个 FAIL 即停；
- 交上层核实。

**不授权：**
- Gate D 试学（`--through-d`）；
- 除零步所需 1 步模型之外的任何训练；
- 正式四臂训练；
- 访问正式块 300000–999999；
- 改动任何规则、参数或开局。

---

## 0. 当前任务状态（下层必读）

### 已结束

| 项 | 结论 |
|---|---|
| 旧 T1-L 三臂（`166244b`，交付 `68680eb`，上层审查 `71d6b82`） | 6 s 审议的 N0/NΔ 相对 0k 有学习，但没有超过最强固定规则；NΔ − N0 未证明；2 s 审议的 B2 后期退化。六条规则逐开局事后最优只比统一最强规则高 0.21，说明原任务缺少状态相关的决策余量 |
| T1-L-S（`672ba79`，执行单 `T1_L_S_TCPF_20261016`） | 正式长训取消，未启动。区间 Bellman、训练器等代码保留，作为新线的基础 |
| 原 T1-O 工况（15–19 m，2.4–3.0°/s） | 搁置：20 个试探开局中 6 个远场外漂 unsafe |

### 物理筛选（证据 `e8c0f08`，`evidence/T1O_SCREEN/`）

| 工况 | 原规则判定 | 结果 |
|---|---|---|
| P1：15–19 m，2.1–2.7°/s | FAIL | 1 次远场外漂 unsafe |
| P2：14–18 m，2.1–2.7°/s | FAIL（原 S3） | 32/32 捕获，0 unsafe，无 recovery / hold |

原 S3 判定保留，不得改写为 PASS。

### 研究层 V2 方案（2026-10-17）

- **工况：** P2 为唯一待诊断工况。物理、G★ v0.6、共享 MPC 均与 T1 相同。
- **SAC 动作：** 首次延迟（相对最早可行锚点）、在役 ±30 s 调整、κ、v_in。审议周期 6 s，计划事件提前审议。
- **燃耗权重：** λ_f 从 {5, 10, 15, 20} 中在发现集上选择。
- **准入：** Gate A（可学习空间）→ B（途中审议与 G★ 职责）→ C（独立确认集与零步）→ E（MPC 反事实可信度）→ D（12k 单臂试学）。全部通过、研究层 GO 后，才进行正式 2×2 四臂训练（O-R1 / O-R3 / O-Δ1 / O-TCPF）。

### Gate A：已在上层沙盒跑完（证据 `f315440`，`evidence/T1O_V2/gate_a/`）

**判定：** PASS，λ_f = 20，最强统一候选 U = A15（首次延迟 15 s、κ 1.0、v_in 0.02）。

**同时记录的事实：**
1. "逐开局最优 − U" 为 2.62，去掉开局 298319 后为 1.94（门槛 2.5）。
2. κ 与 v_in 有影响力，但几乎总偏向同一端（κ = 1.0、v_in = 0.02）。
3. 随状态变化的选择基本只在首次延迟上，与端口相位有关。

**是否固定 κ / v_in（SAC 降为时间维）由研究层决定。** 因此本单停在 Gate D 之前。

## 1. 独立工作树

```powershell
cd D:\py
git clone https://github.com/251614030027r-max/spacecraft.git DRL2_T1_O
cd D:\py\DRL2_T1_O
git checkout --detach d3902a9dd62dbef6cb318e36ff4c2527d7340ec9
git log -1 --format=%H        # 必须是 d3902a9dd62dbef6cb318e36ff4c2527d7340ec9
git status --porcelain        # 必须无输出
git diff --stat 166244bd4d08713473e17354626f10e40b496e01 d3902a9dd62dbef6cb318e36ff4c2527d7340ec9 -- requirements.txt requirements-mpc.txt requirements-dev.txt   # 必须无输出：沿用 T1-L 的 Python 环境，不安装、不升级
$env:OMP_NUM_THREADS="1"; $env:MKL_NUM_THREADS="1"; $env:OPENBLAS_NUM_THREADS="1"
```

- 每个新窗口都先 `cd D:\py\DRL2_T1_O`，再设三个线程变量。
- 产物只写在 `D:\py\DRL2_T1_O\logs\` 下。
- 不改动 `D:\py\DRL2_T1`、`D:\py\DRL2_T1_L` 及其产物。

## 2. 准入（A–C 必须通过）

```powershell
python -B -m pytest -q tests/test_t1_o.py tests/test_t1_l_s.py tests/test_t1_l.py tests/test_t1_counterfactual.py tests/test_t1_gym.py tests/test_t1_termination.py tests/test_t1_fixes.py tests/test_t1_m1.py
Get-FileHash experiments\t1o_p2_discovery_openings.json, experiments\t1o_p2_confirmation_openings.json, experiments\t1o_p2_development_openings.json -Algorithm SHA256
```

| 项 | 条件 |
|---|---|
| A 测试 | 106 项全部通过（上层沙盒在本提交上的结果：106 passed） |
| B 开局 SHA-256 | discovery `663e8ed9cbbfcc30f07fb04419fb6f74c5f5d887e6b9387e2c7c32450ec45070`；confirmation `2ff5e9cc45592f3534878e08fb259d9a68ed1577bb23b54361220df842c03e47`；development `79d1505ce0b838da347f18363b2f06f1c89b03be59630ac0f927e932cd542f9e` |
| C 磁盘 | 剩余 ≥ 20 GB |

结果写入 `logs\t1o_v2\ADMISSION.md`。任一项不通过即停止上报。

## 3. 并行度

- **W** = 本机可用物理核心数 − 1，例如 12 核取 11。各阶段都按回合（或按开局）并行，结果与串行一致。
- 写入 `logs\t1o_v2\SCHEDULE.md`：W、CPU 型号、开始时间（UTC）。
- 墙钟只作资源记录，不作实时性声明。

## 4. 执行（一条命令，按顺序，遇到第一个 FAIL 即停）

```powershell
python -B -m experiments.t1o_v2_chain --root logs/t1o_v2 --workers W --reproduce-gate-a *> logs\t1o_v2_chain.log
```

| 阶段 | 内容 | 粗估（12 核） |
|---|---|---|
| A_reproduction | 本机重跑 Gate A 的 160 个回合并判定。判定、λ、U 必须与上层 `evidence/T1O_V2/gate_a/gate_a.json` 一致，否则停止。逐回合逐值相同的比例只记录，不作门槛 | 约 30 min |
| 状态规则 | 在本机 Gate A 数据上按冻结规则拟合：最多 2 个观测特征，深度 ≤ 2，叶节点 ≥ 3 | 数秒 |
| B | 每个发现开局取在途快照：keep 与 +15 s 两分支各飞到底；再取近场快照：κ 1.0 与 0.7 两分支各飞到底。两分支都从同一快照、用新建求解器出发；keep 分支与原轨迹之差记为数值重复性 | 约 15 min |
| C | 确认集 32 开局上跑 U、状态规则、G★ 预测代价规则、A0；做配对读出 | 约 20 min |
| C_zero_step | 生成未训练的 O-R3 模型（`train.train_t1_option --arm or3 --steps 1`，learning_starts 1000，所以不更新），与状态规则一起在 dev32 上评估 | 约 15 min |
| E | 发现集轨迹作标定、确认集 U 轨迹作验证；在每个审议点做 MPC 反事实，并用两个影子各飞 2 s 真值分支 | 约 30 min |

- **记录：** 每个阶段写入 `logs\t1o_v2\CHAIN.json`（通过与否加关键数），以及各自的 JSON。
- **中断：** 若机器中断，重跑同一条命令。已写出 JSON 的回合会自动跳过，因为评估是确定性的，不是训练。如实记入 `RESULT.md`。
- **不得添加：** `--through-d`，或任何其他参数。

## 5. 交付（链条停止后，无论 PASS 还是 FAIL）

**打包** `T1O_V2_GATES_LOCAL.zip`：

| 来源 | 文件 |
|---|---|
| `logs\t1o_v2\` | `CHAIN.json`、`ADMISSION.md`、`SCHEDULE.md`、`gate_a_reproduction.json`、`gate_a.json`、`gate_a\**`、`state_rule.json`、`gate_b.json`、`gate_c.json`、`gate_c\**`（含 `assignments.json`）、`zero_step.json`、`zero_step\**`、`gate_e_calibration.json`、`gate_e_calibrate.json`、`gate_e_validation.json`、`gate_e.json`（实际产生了哪些就交哪些） |
| 其他 | `logs\t1o_v2_chain.log`、`logs\t1_o_tcpf_20261017\or3_zero\manifest.json` |
| 包外 | 逐文件 `SHA256SUMS.txt` |

**交付位置：**
- 推到 collab `docs/collaboration/lower/handoffs/t1o_v2_gates_20261017/`。包 > 50 MB 时留本机，只交 SHA 与路径。
- 模型 zip 不上传，只交 SHA。

**`RESULT.md` 只写事实：**
- 各阶段 PASS / FAIL 与 `CHAIN.json` 中的关键数；
- 停在哪一阶段；
- 本机逐回合逐值相同的比例；
- 墙钟、中断与偏离。

**CURRENT.json：** 增加支线字段：
- `t1o_topic` = `t1o_v2_gates`；
- `t1o_scientific_commit` = 本单 SHA；
- `t1o_run_order` = 本文件；
- `t1o_phase` 按实际状态填写。

## 6. 之后

1. 上层核实本单交付：复现一致性、各阶段数值，必要时抽查重算。
2. 研究层决定：是否接受 Gate A 的单开局敏感性；动作维度是否保留 κ / v_in。
3. 动作空间冻结后，另出执行单跑 Gate D（12k 单臂试学），仍在本机执行，仍由上层核实。
4. 全部通过、研究层 GO 后，再出正式四臂 40k 执行单，在本机训练。
