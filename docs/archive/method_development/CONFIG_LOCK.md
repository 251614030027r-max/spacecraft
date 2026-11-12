# 配置冻结清单

> **非最终论文算法（2026-11-10）：** 本文件是研发记录。论文方法以 `docs/paper/METHOD.md` 为唯一源，对照见 `docs/paper/FACT_MAP.md`。

**冻结科学提交：** 执行书 `RUN_PRETEST`（collab `docs/collaboration/upper/run_orders/`）中写明的 SHA。本文件、`METHOD_REGISTRY.json` 与代码在同一次提交中冻结。

## 主方法与 T2.12 冻结版本的等价性

基准是 T2.12 `e_priv` 两条训练 manifest 中的 `git_head`：`2e7d838b680f88dbf16b4820e4862a7f9899ac99`。

**审计**（`evidence/final_pretest/precheck/`，沙箱）：
1. **环境层：** 4 个开局、固定随机动作序列、`reduced_priv`、shaping 开、紧急期限 U(140, 200) s，共 36 个决策。新旧版本的观测哈希、shaped 与真实回报、终止事件、模式、时间逐步一致。
2. **训练层：** `--arm e_priv --seed 3 --steps 80 --workers 1 --smoke`。新旧版本第 40 步与第 80 步的策略参数逐位一致，回合日志与训练统计一致。
3. 以上在本次全部改动（新增 `reduced_dup`、`--no-geometry`、评估读 manifest、逐回合执行统计日志）之后重做。

## 主方法开关

| 项 | 值 |
|---|---|
| arm / obs_profile | `e_priv` / `reduced_priv` |
| Actor / Critic 输入 | 24 / 48 |
| window / geometry | 开 / 开 |
| k_candidates / model_beta | 0 / 0 |
| pref / acc / guide / relative | 均为 false |
| feedback / terminal_horizon | null / 0 |
| 决策折扣 γ_d | 0.999^8 |
| 阶段 | 16 s |
| 熵系数 | 0.05（固定） |
| 每条转移的更新次数 | 2 |
| 采集 | 2 个异步 worker |
| 任务 | 紧急 / 经济各 50%；紧急期限 U(140, 200) s，经济 U(280, 340) s；预算 3 m/s |
| 冻结 MPC | 不变 |
| 奖励 | 时间与燃料代价按任务；捕获 +100、超时 / 耗尽 −100、不安全 −150 |
| 信息边界 | 规划器、观测、入口都用恒角速度 / 各向同性惯量估计；真实惯量只在真值里 |

## 阶段与开局

| 阶段 | 训练种子（训练开局） | 预算 | 评估开局 | 训练组 |
|---|---|---|---|---|
| dryrun（流程检查） | 10、11（smoke 学习器设置） | 40 | 294300–294301 | 6 组 |
| 预实验 | 10、11（11,000,000 起、12,000,000 起，从未用过） | 2000 | 294100–294147（48 个，从未用过） | EACH、SAC、SAC_ENG、DUP、NO_GEOM（两批：先 EACH、SAC、SAC_ENG，再 DUP、NO_GEOM） |
| 正式实验 | 20、21、22（21,000,000–23,999,999） | 5000 | 300000–300095（正式块，从未访问） | 6 组 |

- 已用过的开局：开发评估 294000–294047、开发训练种子 0 / 1、探针 2,900,000 以上。
- 作图开局 294200–294211：只用于几何扫描与轨迹图，不用于任何模型选择。
- `experiments.final` 拒绝在保留开局上评估、拒绝不在本阶段名单内的种子与配置。

## 依赖

沙箱：Python 3.13.18、numpy 2.5.4、scipy 1.18.1、torch 2.15.0、stable-baselines3 2.9.2、gymnasium 1.4.0、cvxpy 1.9.3、clarabel 0.11.1。

下层本机在预实验自检时写出 `ENV_LOCK.txt`（`pip freeze` 与 Python 版本）。跨机器比较结论，不比较最后几位数值。
