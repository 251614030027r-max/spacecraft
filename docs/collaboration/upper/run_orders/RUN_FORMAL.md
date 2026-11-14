# 执行书：RUN_FORMAL（ECC 正式实验）

**状态：待用户单独授权；授权前不得启动任何一步。前提已满足（2026-11-13）：RUN_CLEANUP 已交付并复核通过（collab `fa7506d4`），代码在 Windows 上对冻结行为逐位等价，本机已就绪（`READY.json`）。** 授权时用户在本页写明"已批准"和日期；若要删减组别，必须在授权前写明删哪一组，不能训练后再选。

设计依据：科学分支 `docs/paper/EXPERIMENTS.md`（事先登记）；方法 `docs/paper/METHOD.md`；准备工作 RUN_FORMAL_PREP 已交付（科学 `e47a1165`，collab `5496cb51`），上层复核见本页第 7 节；工程收尾 RUN_CLEANUP 须先交付并复核通过（第 7 节）。

## 0. 冻结

| 项 | 值 |
|---|---|
| 科学提交 | `e8474e21`（标签 `archive/formal-code-20261114`，下层已打并回读；`e8474e213c5cd98d01e413907c18ecf662a30648`；工程收尾后的正式实验代码，冻结路径对 `fcd8d2ee` / `e0a61753` 逐位等价，见第 7 节） |
| 注册表 | `docs/final_method/METHOD_REGISTRY.json`，阶段 `formal`（条件 U / E / N；预算按任务类型：紧急 3.0、经济 2.0 m/s，训练与评估相同） |
| 唯一入口 | `python -B -m experiments.final`（train / eval / screen / traj / geoscan / check / resolve），读出 `experiments.formal_readout`，作图 `experiments.formal_figures` |
| 输出根目录 | `C:\Users\35884\Documents\Spacecraft\过程文件\FORMAL\`（下文记作 `$R`）；训练目录 `$R\B\TRAIN_<配置>_s<种子>` |

**启动前（第 0 步）：** 在 `D:\py\DRL2` 检出 `e8474e21`，工作区干净；全量 `python -B -m pytest -q` 退出码 0（160 项通过，无预期失败）；把退出码、开始 / 结束时刻和 `git rev-parse HEAD` 写入 `$R\PREFLIGHT.json`。之后到交付前不切换分支、不拉取、不改工作区。

## 0A. 本机设置（启动前，不影响结果）

- 把 `D:\py\DRL2`、当前 venv 与 `$R` 加入杀毒软件实时扫描的排除项；
- 电源计划设为"高性能"，禁止睡眠与自动重启；
- 运行期间不开其他重负载程序。

## 1. 规模

| 项 | 内容 |
|---|---|
| 训练 | 6 组 × 种子 20、21、22 × 5000 决策 = 18 条，全部从零开始：`ECC_CERT_LOOP`、`SAC_C`、`TD3_C`、`EACH_C`、`SAC_CERT_LOOP`、`ECC_CERT_LOOP_NOGEO` |
| 检查点 | 1000 / 2000 / 3000 / 4000 只用于学习曲线；**最终模型固定为 5000 步** |
| 主测试 | 18 个最终模型 × 条件 urgent（300000–300095，预算 3.0 m/s）、economy（300100–300195，预算 2.0 m/s）、natural（300200–300295，按任务类型 3.0 / 2.0 m/s） |
| 只评估对照 | `ECC_CERT_LOOP` 认证关（urgent、economy）与重规划关（urgent）；`EACH_C`、`SAC_C` 执行时认证开（urgent、economy） |
| 固定策略 | pure（urgent、economy、natural），gstar（urgent、economy） |
| 验证曲线 | `ECC_CERT_LOOP`、`SAC_C`、`TD3_C` × 3 种子 × 检查点 1000–5000，开局 295200–295223 |

机时（下层实测吞吐，按原排程）：训练约 70–75 小时，评估约 40 小时（含已取消的 R 组约 1,800 回合），整体约 4.5–5 天。按第 2、3 步的队列排程，加上 2026-11-13 的不改结果提速（环境墙钟约 −14%），取消 R 组后评估少约 17%，预计约 3–3.5 天。

## 2. 训练（任务队列，始终 6 条并行）

18 条训练按"种子 20 的 6 组 → 种子 21 的 6 组 → 种子 22 的 6 组"排成一个队列，同时运行 6 条。任一条结束，立即从队首补下一条，不等同批其他条。带认证的配置（`ECC_CERT_LOOP`、`SAC_CERT_LOOP`、`ECC_CERT_LOOP_NOGEO`）每决策约慢 1.5 倍，所以同一种子内先启动这三条。这样核不会在批尾空转，中途停机时已完成的条目仍按种子成组。

```bash
python -B -m experiments.final train --stage formal --config <配置> --seed <种子> --output $R\B\TRAIN_<配置>_s<种子>
```

- 每条结束后 `final.py` 自动核对 manifest；另记 `final.zip` 与各检查点的 SHA-256。
- **训练中断：** 不续训、不热启动。删除该条目录前先把它整体改名为 `...__aborted_<n>` 保留，再用同一配置、同一种子从零重跑，并在 RESULT.md 写明原因。
- 每条结束只核对 manifest 与检查点，**不看评估结果、不做任何选择**。

## 2B. 夜间分段运行（用户 2026-11-14：每晚约 10 小时，白天不占用机器）

单条 5000 步训练在 6 条并行时需 15–24 小时，跨不止一个晚上；而训练中断只能从零重跑。因此白天用**挂起**，不用停止：

- **早上：** 对每条训练的整棵进程树（学习进程与 2 个采集进程）和评估进程执行挂起（Python `psutil.Process(pid).suspend()`，或 Sysinternals `pssuspend`）。**晚上：** 原样恢复。
- 挂起只暂停 CPU，进程内存状态（回放、网络、优化器、随机数、求解器缓存）完整保留，所以不是续训，也不改变结果：训练本来就是多进程异步，评估是确定的。
- 白天机器保持开机，不睡眠、不重启、不注销。挂起期间内存照常占用；若白天内存不够用，把同时训练的条数减到 4（第 2 步的队列照常）。
- 每次挂起 / 恢复的时刻写入 `$R\SCHEDULE.log`。`manifest` / `finished.json` 的 `wall_s` 因而包含挂起时长，不用作任何结论。第 5 步的串行计时必须在一个晚上内不间断完成。
- **意外重启或断电：** 进行中的训练按第 2 步"训练中断"规则处理（目录改名保留，同配置同种子从零重跑）；评估只补缺失文件（`--verify-existing`）。
- **本机前提：**
  - Windows Update 在运行期间设为仅通知或暂停更新；
  - Defender 路径排除由管理员落实（预检时尚未完成，见 collab `lower/handoffs/formal_preflight_20261114/`）。
- **预计：** 计算量约 72–84 小时，即 7–9 个晚上。

## 3. 评估（训练完即评估）

每条训练一结束，就把它的全部登记评估排入评估队列：四个条件、登记的只评估开关；属于 ECC / SAC–MPC / TD3–MPC 的，还有验证检查点。固定策略（pure、gstar）在第一批训练期间就可开始。

- 评估进程的优先级低于训练（Windows "低于正常"）；
- 评估的 `--workers` 取当时空闲的核数，使训练与评估合计的进程数约等于逻辑核数（12）。

评估是确定的、逐回合串行的，何时运行不影响结果；耗时不作实时性结论（第 5 步单独串行计时）。

```bash
python -B -m experiments.final eval --stage formal --run $R\B\TRAIN_<配置>_s<种子> --condition <urgent|economy|natural> --output $R
python -B -m experiments.final eval --stage formal --run $R\B\TRAIN_ECC_CERT_LOOP_s<种子> --condition <urgent|economy> --no-certify --output $R
python -B -m experiments.final eval --stage formal --run $R\B\TRAIN_ECC_CERT_LOOP_s<种子> --condition urgent --rigid --output $R
python -B -m experiments.final eval --stage formal --run $R\B\TRAIN_<EACH_C|SAC_C>_s<种子> --condition <urgent|economy> --certify --output $R
python -B -m experiments.final eval --stage formal --run $R\B\TRAIN_<ECC_CERT_LOOP|SAC_C|TD3_C>_s<种子> --condition validation --checkpoint-step <1000..5000> --output $R
python -B -m experiments.final screen --stage formal --policy pure --condition <urgent|economy|natural> --output $R
python -B -m experiments.final screen --stage formal --policy gstar --condition <urgent|economy> --output $R
```

- 结果目录由入口自动命名：`$R\C\<配置>_s<种子>_<条件>[_nocert|_cert|_rigid]`、`$R\V\<配置>_s<种子>_step<步数>`、`$R\S\<策略>_<条件>`；正式评估自动开启只读日志。
- **补跑：** 只补缺失文件；补跑前对已有文件用 `--verify-existing $R\SHA256SUMS`（先生成）核对，不覆盖、不重算已有结果。
- **`error` 记录：** 表示回合没有跑起来。先把该文件改名保留，再补跑该开局；在 RESULT.md 列出。
- **`numerical_failure`：** 是正式结局（未捕获），不重跑、不剔除，读出单独计数。

## 4. 图用数据（不产生新的比较）

- **几何扫描（RWG 几何小图）：** 只用开发开局，不读任何回报：
  ```bash
  python -B -m experiments.final geoscan --openings 294020,294031 --output $R\G
  ```
- **案例轨迹（F2 / F3 / F5）：** 第 6 步读出完成后，按 `experiments.formal_figures.select_cases` 的固定规则选出开局（物理案例：ECC、SAC–MPC、纯 MPC 都捕获的最小编号；机制案例：ECC 捕获而认证关未捕获的最小编号；没有则用 ECC 捕获、SAC–MPC 未捕获的最小编号），用种子 20 的模型重飞：
  ```bash
  python -B -m experiments.final traj --entry ECC=$R\B\TRAIN_ECC_CERT_LOOP_s20 --entry SAC=$R\B\TRAIN_SAC_C_s20 --entry pure=pure --openings <所选开局> --condition urgent --output $R\TRAJ
  ```
  重飞结果须与 `$R\C` 中同一回合的结局、时间、Δv 一致；不一致则停下报告。

## 5. 串行计时（物理可信度表）

所有训练与评估结束、机器空闲时，单进程重跑 `ECC_CERT_LOOP` 种子 20 在 urgent 前 24 个开局：
```bash
python -B -m experiments.final eval --stage formal --run $R\B\TRAIN_ECC_CERT_LOOP_s20 --condition urgent --openings 300000-300023 --workers 1 --output $R\TIMING
```
结果只取认证耗时（`cert.wall_s`）与回合耗时；回合耗时除以 0.1 s 控制步数，作为 MPC 每步求解耗时的上界（`EXPERIMENTS.md` 第 5 节）。结局须与 `$R\C` 中对应回合逐项一致（确定性核对），不进入任何比较。

## 6. 完整性检查、读出与作图

```bash
python -B -m experiments.final check --stage formal --root $R --output $R\CHECK.json
python -B -m experiments.final resolve --stage formal --root $R --output $R\REGISTRY_RESOLVED.json
python -B -m experiments.formal_readout formal --root $R --output $R\FORMAL_READOUT.json
python -B -m experiments.formal_readout episodes --root $R --output $R\EPISODES.csv
python -B -m experiments.formal_figures --root $R --readout $R\FORMAL_READOUT.json --output $R\FIGURES
```

**验收（全部满足才算交付）：**
1. `check` 报告 complete；`resolve` 不报告混合的 SHA、预算或配置。
2. 18 个训练 manifest：`steps = 5000`、种子 ∈ {20, 21, 22}、`git_head = e8474e213c5cd98d01e413907c18ecf662a30648`、`final_stage = formal`。
3. 每个结果目录的开局集合与注册块完全一致，无重复、无 `error`。
4. 本轮访问过的开局只有：注册的正式条件块、验证块 295200–295223、几何扫描用的开发开局；训练开局只来自三个种子各自的训练流。
5. 串行计时与案例重飞的确定性核对通过。

**看结果的顺序（事先写定）：** 先看 `FORMAL_READOUT.json` 的 `primary_verdict` 与 urgent 下 ECC 对 SAC–MPC 的配对捕获差及区间，再看真实结局与回报，最后看机制统计与案例。不按结果改组别、种子、检查点或统计口径；不加种子。

## 7. 上层复核（RUN_FORMAL_PREP 交付）

- **逐位回归：** 环境逐步回归（6 个案例）、四组 40 步短训检查点与回合（含正式阶段只读日志开启）、B 步模型 294800–294801 逐决策日志，均与 `e0a61753` 逐位一致。只读日志只读取已有结果，没有改变求解顺序、随机数或动作。
- **代码：** TD3 按登记的超参实现（策略延迟 2，目标噪声 0.2 / 截断 0.5，探索噪声 σ = 0.1，独立随机流 `[seed, 3003]`，评估按 manifest 加载）。主终点、分层 bootstrap（10,000 次，种子 20261110）与预筛规则都与 EXPERIMENTS.md 一致。
- **预筛：** 约束激活与可解性的计算与事先登记的规则一致；2.0 m/s 即满足（57/240，48/48），按规则冻结。
- **复核时修正（`fcd8d2ee`，只改读出与测试）：**
  1. η 的候选燃料条件漏算了已用 Δv（原为"预测 Δv ≤ 预算"），改为"已用 + 预测 Δv ≤ 预算"，与 EXPERIMENTS.md 第 6 节 B 和认证集一致；
  2. `numerical_failure` 原会使读出报错，改为按未捕获计入并单独计数（只有 `error` 记录阻止读出）；
  3. 新增 `episodes` 子命令，导出一回合一行的紧凑表，供入库。
- **作图缺项（不影响运行）：** F4 经济面板尚未画逐对分布，RWG 几何图尚未叠画离散可行点与意图箭头。两者只读已保存数据，写论文时由上层补齐，不需要重跑。

### 设计修订：经济任务预算 2.0 m/s，取消 R（2026-11-13，用户）

- 经济任务在训练与评估中都用 2.0 m/s，紧急任务 3.0 m/s；不再单设 R 条件（`docs/paper/EXPERIMENTS.md` 第 2 节，规格先入库 `ff9f06e7`，实现 `e8474e21`）。
- 核验（沙箱）：紧急路径与已用 2.0 预算的经济路径对冻结记录逐位不变；原 3.0 预算的经济路径按设计变化；pytest 160 passed。

### 不改结果的提速（2026-11-13）

- 改动：3 维叉积 / 范数快速实现，李群工具去掉重复校验，CVXPY 无属性叶子赋值跳过必然通过的检查（`docs/paper/AUDIT.md` 第 5 节）。
- 沙箱核验：环境 7 / 7、训练 6 / 6、评估 2 / 2、固定策略 10 / 10 与冻结记录逐位一致，pytest 160 passed；环境墙钟约 −14%（`evidence/SPEEDUP_20261113/`）。

### 工程收尾（2026-11-12）

- 上层把科学分支瘦身为正式实验实际运行的代码（`e8474e21`）。冻结路径在上层沙箱对 `fcd8d2ee` 逐位等价，覆盖：
  - 环境 7 种情形；
  - 6 个配置的单进程短训；
  - 旧模型在新代码下的评估。

  详见 `evidence/CLEANUP_EQUIV/`、`docs/paper/AUDIT.md` 第 4 节。下层在 Windows 上的复核见 RUN_CLEANUP。
- 收尾中修正的缺陷：
  - `traj` 对认证模型只在给了 `--condition` 时才开认证，现在一律按 manifest；
  - `resolve` 把训练任务名当作截止区间元组处理，现在按任务名核对；
  - 冒烟训练可以误用正式种子，现在拒绝。
- 文档更正：`FACT_MAP` 原写"采集进程每 100 次更新同步策略"。在冻结路径中，ECC / SAC / TD3 的每次动作都由学习进程用当前策略作答，原表述只适用于历史的反事实标签臂。
- 可复现性：多进程训练的转移顺序取决于进程快慢（`worker_order`），种子相同的两次 2 进程训练不保证逐位相同；单进程训练与全部评估逐位可复现。

## 8. 交付

- **科学分支 `evidence/FORMAL/`（逐个文件强制加入）：** `PREFLIGHT.json`、`CHECK.json`、`REGISTRY_RESOLVED.json`、`FORMAL_READOUT.json`、`EPISODES.csv`、18 个训练 manifest 与 `episodes.json`、`FIGURES/` 全部图表、`TIMING/` 汇总、`TRAJ/` 所选案例、全部原始结果的 `SHA256SUMS`。
- **原始结果和模型**（`$R\C`、`$R\V`、`$R\S`、模型 ZIP）只在本机存档，不入 git；`SHA256SUMS` 作为索引。
- **collab：** `docs/collaboration/lower/handoffs/formal/RESULT.md`：各批起止时间、中断与补跑记录、验收 1–5 逐条结果、主终点判定一行。不写解读，解读由上层复核后写入论文材料。

**转发给下层的一句话（授权后）：**

> 读 collab `docs/collaboration/upper/run_orders/RUN_FORMAL.md`（已批准）。按第 2B 节夜间分段运行（每晚约 10 小时，白天挂起不停止）。在 `D:\py\DRL2` 检出 `e8474e21`，全量 pytest 退出码 0 后，按第 2–6 步执行：三批训练（每批一个种子的 6 组）、全部登记评估、图用数据、串行计时、完整性检查与读出作图；按第 8 节交付。中途不看结果做选择，训练中断从零重跑。
