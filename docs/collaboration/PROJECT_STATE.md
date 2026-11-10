# 项目当前状态（2026-11-08）

*先读 [HANDOFF.md](HANDOFF.md)（第 0B 节是 11-08 速读）与 [研究演进](RESEARCH_EVOLUTION_AND_COUPLING_DESIGN_20261012.md)（阶段 32–39）。本文件只记事实：正在运行什么、最近完成什么、东西在哪里、还有什么没做。*

## 1. 当前状态

| 项 | 状态 |
|---|---|
| 运行中 | **RUN_FORMAL_PREP 已下发（2026-11-10）**：正式实验前的全部准备交下层（注册表 formal 重写、TD3、只读日志、评估选项、正式读出与作图、回归与冒烟、资源条件 R 预筛），不训练、不碰 300000 以上。方法冻结 = `ECC_CERT_LOOP`（科学 `322f1cec` 起；论文方法唯一来源 `docs/paper/METHOD.md`，`fdc164ef`；正式实验设计事前登记 `docs/paper/EXPERIMENTS.md`，`6c76c2c9`）。B 步结果见第 2 节：紧急对 SAC_C +36.0，捕获 145 / 111，0 unsafe。下一步：上层复核交付 → 写 RUN_FORMAL → 用户单独授权 |
| 现行方法 | **ECC（执行机会证书耦合）**，规格见科学 `docs/final_method/ECC_SPEC.md`。RUN_ECC 按事先登记的判定顺序选中 **`ECC_deploy`**：critic 读过顶层证书（原 U1），部署时执行器认证意图（原 U2 探询）；决策层统一去掉规划答案与执行内部状态 |
| RUN_ECC 结果（新开局 294320–294367，2 种子 × 2000，96 对，对 SAC_C） | **EACH_C + 部署认证：+12.2，捕获 84 / 78，种子 +1.5 / +23.0，去掉每种子最大两个正差后 +3.8，0 unsafe，G1–G5 全过**。<br>完整 ECC（认证在环、动作层证书进 critic）：+5.7，81 / 78，1 次 unsafe，未过。<br>EACH_C（不认证）：+5.9，种子 −7.1 / +18.9。<br>同模型认证（EACH_C 开 / 关）：+6.3，两种子 +8.5 / +4.0，捕获 +3，0 丢失，"改善"档。<br>训练曲线 AUC：ECC 26.3 / 23.2，EACH_C 5.8 / 14.3，SAC_C 13.2 / −10.7 |
| 需正视的点 | ① 通过幅度偏薄（种子 10 只有 +1.5，中位差约 0，靠 8 比 2 的捕获翻转）；<br>② 完整 ECC 的认证在 294334 引发 1 次 unsafe（"下降"类认证），而认证救回的捕获大多也来自"下降"类；<br>③ 训练与部署一致（ECC 规格性质 P1）只对完整 ECC 成立，选中的 `ECC_deploy` 训练时未开认证；<br>④ 动作层证书进 critic 的增益没有在最终评估中体现；<br>⑤ 只有 2 个种子 |
| 下一阶段（用户 11-08） | 正式实验前再统筹优化一版最终方法；三窗口重开收尾：先完整阅读、统筹优化方法，再计划正式实验 |
| 正式实验 | 未授权。[RUN_FORMAL](upper/run_orders/RUN_FORMAL.md) 与注册表 `formal` 阶段仍是 EACH 时期的 6 组，需按最终方法重写 |
| 汇报稿 | 开发期汇报图 `evidence/ppt_dev/`（EACH 时期），出图计划 [PPT_FIGURES](upper/run_orders/PPT_FIGURES.md) |
| 最新科学提交 | `f2fd0c8e`（文档）；ECC 冻结 `1cddbae2`（106 项测试）；RUN_ECC 交付 `cc1a0954` |

### 执行书记录（T2.12 起）

| 状态 | 执行书 |
|---|---|
| 已下发（2026-11-10，用户：开始测试交付下层） | [RUN_FORMAL_PREP](upper/run_orders/RUN_FORMAL_PREP.md)：正式实验准备 WP1–WP8（注册表新配置 SAC_CERT_LOOP / ECC_CERT_LOOP_NOGEO / TD3_C 与新 formal 阶段、旧 formal 改名 `formal_each_obsolete`；TD3 学习器；只读决策 / 回合 / 轨迹日志；评估选项与命名；`experiments/formal_readout.py`、`experiments/formal_figures.py`；对 `e0a61753` 逐位回归与 6 配置冒烟；R 预筛 295100–295147）。冻结方法行为不变，每个工作包单独提交、pytest 退出码 0；交付 `evidence/FORMAL_PREP/` 与 `lower/handoffs/formal_prep/RESULT.md` |
| 已交付（collab `7467923d`，科学 `cc1a0954`） | [RUN_ECC](upper/run_orders/RUN_ECC.md)：`final_method = ECC_deploy`（见第 1 节） |
| 已下发并交付（11-07 – 11-08） | [RUN_ECC](upper/run_orders/RUN_ECC.md)：ECC（执行机会证书耦合：一个执行机会评估同时给出 critic 证书与认证映射，训练与部署一致）。SAC_C、EACH_C、ECC × 种子 10、11 × 2000，决策层统一去掉规划答案与执行内部状态；评估新开局 294320–294367，另评 ECC 认证关闭与 EACH_C 部署认证；判定顺序 ECC → 部署认证 ECC → none（`experiments/ecc_readout.py`）；冻结科学 `1cddbae2`（106 项测试）；规格 `docs/final_method/ECC_SPEC.md` |
| 已交付（collab `a60b0d8c`，科学 `bcab9207`） | [RUN_U2C](upper/run_orders/RUN_U2C.md)：主比较 `pass`，同模型认证 `no_effect` |
| 已交付（collab `a60b0d8c`） | [RUN_U2C](upper/run_orders/RUN_U2C.md)：U2 探询改为决策时认证（SAC 意图赶不上过顶、探测意图能赶上、且过顶约束成立时，改执行最近的认证意图）。预实验 EACH、SAC 模型在新开局 294220–294267 上认证开 / 关配对，不训练；主判据 EACH+认证 对 SAC（G1–G5），同模型分档 pass / improved / no_effect；冻结科学 `28a40799`。保底（两同模型比较都 no_effect）：统一信息边界后重训，已登记于 `docs/final_method/U2_CERT_SPEC.md` 第 4 节 |
| 已交付（collab `93657249`，科学 `83bc7f6a`） | [RUN_U2](upper/run_orders/RUN_U2.md)：主判定 `no_gain`，次要 `kept`；U2 作为 critic 输入按规则结束 |
| 已交付（collab `93657249`） | [RUN_U2](upper/run_orders/RUN_U2.md)：`PROBE`（U2 替代 U1）与 `EACH_PROBE`（U1 + U2）各种子 10、11 × 2000；复用预实验 EACH、SAC 模型；评估新开局 294148–294195；主判据 PROBE 对 SAC（G1–G5，种子与开局配对）；冻结科学 `e7acaa34`；与阶段 A 并行，A 未过则作废 |
| 已下发（2026-11-03） | [ASSETS_20261103](upper/run_orders/ASSETS_20261103.md)：下层本机资产整理与核验（不训练、不改代码） |
| 已下发（2026-11-03，重新启动） | [T2_12_EACC_20261104](upper/run_orders/T2_12_EACC_20261104.md)：EACC（执行锚定的相对反事实价值学习；候选按执行器阶段目标间隔、共同随机数；训练代码 `0de3dd4`）与消融组（特权 critic、无反事实反馈）各两种子 × 3000 步，用户给 12 小时；判定在 2000 步预算上与 T2.11 比较，3000 步 final 另报；SAC 与 EACC-A 复用 T2.11。第一次启动停在测试门槛（Windows 77 passed / 1 failed，科学 `0190faf`），上层复核为测试写法问题，冻结改为科学 `2e7d838`（只改该测试，78 项）。用户 11-03 已指示推进 |
| 已批准（2026-11-04，用户） | [T2_12_STOP2000_20261104](upper/run_orders/T2_12_STOP2000_20261104.md)：T2.12 在四条都到 2000 步时检查，`e_rel` 的 1500–2000 步真实回报若未追到 `e_priv` − 8.3 以内就停训，评 2000 步、收尾交付，数据全留 |
| 已交付（2026-11-04，collab `58779aa6`） | [T2_14_TERM_20261104](upper/run_orders/T2_14_TERM_20261104.md)：T2.12 跑完后接着跑。依据 T2.12 千步中期（特权 critic 消融 `e_priv` 曲线 AUC 14.86，相对标签 `e_rel` −5.74，T2.11 12.05）：学习器 5.0 只保留执行器能独立算到终局的反事实（`e_term` 一段、`e_term2` 两段，常数目标，不在模型状态上读 critic），各两种子 × 3000 步，判定在 2000 步预算上对 `e_priv`（复用 T2.12 本机）。冻结科学 `03bf0d3`（80 项测试）。结果：`e_term` 回退（曲线 13.25、final 11.64），`e_term2` 判 unsafe（1 次不安全，曲线 9.27、final 2.37），`e_priv` 25.43 / 37.69 |
| A 已交付（`pass`，科学 `55a32e4`）；B 未进行，由 T2_17 取代 | [T2_16_PI_20261104](upper/run_orders/T2_16_PI_20261104.md)：A 真值方向审计 pass（转向决策中修正方向胜 14/20，均值 +2.64，90% 区间 [−0.11, 7.68]，unsafe 0/0）。B 首次启动失败：`323b8c8` 命令行不接受 `--arm e_pi`（上层缺陷，`42cf666` 修复）。A 数据促成学习器 6.2（等长转动）与 6.3（非局部） |
| 上层起草（2026-11-04，按用户指示：加快、以有效为先）；用户转发即下发 | [T2_17_PULL_20261104](upper/run_orders/T2_17_PULL_20261104.md)：学习器 6.3。critic 同 `e_priv`；执行器与 critic 的融合优势只转动策略这一步（critic 步长）：`e_pull` 局部 + 非局部（转向超过 TD 噪声的最优候选），`e_pi` 仅局部（消融）；各两种子 × 2000 步，不设训练前门槛，判据同 T2.12–T2.16（对 `e_priv`，MARGIN 8.3），训练后跑方向审计 A2 作图。冻结科学 `af0b915`（93 项测试） |
| 未执行（1500 步条件未过：`e_pull` 臂均 31.53 < 38.9）；原条件：仅当 T2.17 的 `e_pull` 在 1000–1500 步臂均 ≥ 38.9 且两种子 ≥ 33.9 时执行，并停掉 T2.17；用户转发即同意 | [T2_18_PULL3_20261104](upper/run_orders/T2_18_PULL3_20261104.md)：学习器 6.5。`e_pull3`（远距降权 + 500 步后启用反馈）与 `e_pull2`（只远距降权）各两种子 × 2000 步，判据同前（对 `e_priv`，MARGIN 8.3）。冻结科学 `d091a10`（95 项测试） |
| 上层起草（2026-11-04）；研究层意见支持方案甲；用户要求今夜开跑，转发即下发（T2.17 交付后） | [T2_19_CERT_20261104](upper/run_orders/T2_19_CERT_20261104.md)：最后一轮开发训练。环境 0.13.0 / 学习器 7.0：执行校准的反事实机会图（9 个参考意图 × 可行 / 校准成功率 / 价值 + 就绪标志）作 critic 输入，TD 只用真实转移。`e_cert`（U1 + U2）与 `e_full`（+ U3）各两种子 × 2000 步；对 `e_priv` 判定（MARGIN 8.3），另判 `e_full` 对 `e_cert`；最终方法选取规则事前登记。冻结科学 `0017f80`（97 项测试） |
| 已批准（2026-11-05，用户：不限时、保留关键消融、下层立即开跑） | [RUN_PRETEST](upper/run_orders/RUN_PRETEST.md)：**最终方法 EACH = U1 + C1（`e_priv`）冻结**，与 `2e7d838` 逐位等价。唯一入口 `experiments/final.py`、注册表 `docs/final_method/METHOD_REGISTRY.json`。预实验（关键消融）：EACH、SAC、SAC + planner features、DUP（U1 等宽对照）、NO_GEOM（C1 几何）各 2 种子（10、11）× 2000，两批；窗口承诺用 EACH 模型与固定策略的窗口 / 刚性配对评估检验；固定策略 pure、gstar、rally、b_10、c_10_0；评估开局 294100–294147；只判实验有效性，不设性能门槛；预计约 23 小时。冻结科学 `fd78912`（代码同 `d1523ce`，100 项测试，沙箱全流程 dryrun 完整） |
| 作废（EACH 时期草案；按 `docs/paper/EXPERIMENTS.md` 在 FORMAL_PREP 交付后重写） | [RUN_FORMAL](upper/run_orders/RUN_FORMAL.md)：旧草案 6 组 × 3 种子（20–22）× 5000，评估 300000–300095；约 87 小时训练 + 12 小时评估，分 4 批；须用户在预实验交付后单独授权 |

## 2. 最近完成的工作（按时间倒序）

| 工作 | 结论 | 证据 |
|---|---|---|
| RUN_ECC（`1cddbae2` → `cc1a0954`；ECC 开发轮：SAC_C、EACH_C、ECC × 2 种子 × 2000，统一信息边界；评估 294320–294367，另评 ECC 认证关闭与 EACH_C 部署认证） | `final_method = ECC_deploy`（+12.2，G1–G5 全过）；完整 ECC 未过（+5.7，1 次 unsafe）；同模型认证"改善"档 | `evidence/ECC_DEV/`、`ECC_PRECHECK/` |
| RUN_U2C（`28a40799` → `bcab9207`；U2 探询改为决策时认证；预实验 EACH、SAC 模型配对评估，不训练） | EACH + 认证 对 SAC `pass`（+23.3）；同模型 `no_effect`（EACH +6.3、SAC +5.5，种子 11 约为 0） | `evidence/U2C/`、`U2C_PRECHECK/` |
| RUN_U2（`e7acaa34` → `83bc7f6a`；U2 执行器模式探询作训练期 critic 输入；离线 A1–A5 先过） | 主判定 `no_gain`（PROBE − SAC +1.34），次要 `kept`；同批 EACH − SAC +10.6 | `evidence/U2_PROBE_PRECHECK/`、`U2_TRAIN/` |
| 预实验（下层，冻结 `fd789126`，5 组 × 种子 10、11 × 2000，评估 294100–294147） | 有效性六项全过；EACH 捕获 90/96 对 SAC 80/96，回报 45.7 对 27.5；等宽 critic 15.1，无几何 −54.2；窗口配对入口丢失 85 → 2；G★ 50.1 | 科学 `56841a04`，`evidence/final_pretest/`；collab `79497f85`，`lower/handoffs/final_pretest/RESULT.md` |
| 方法冻结与收尾工具（上层，`d1523ce`、`fd789126`） | EACH = U1 + C1；与 `2e7d838` 环境层、训练层、新入口三处逐位等价；沙箱全流程 dryrun 完整 | `docs/final_method/`、`evidence/final_pretest/precheck/` |
| T2.17 + T2.19 联合复核（下层） | 反事实 / 动作级 / 机会图路线全部未过 `e_priv`；按预登记规则回到 U1 + C1 | collab `lower/handoffs/review_t217_t219_20261105/` |
| T2.19 机会图（`0017f80`） | `e_cert` revert（AUC 15.25，2000 步回报 11.23）、`e_full` revert（2.63 / 6.46）；一个种子机会图乐观、与 `e_priv` 持平，三条悲观、critic 价值一路下降：历史依赖的校准作输入形成自我印证 | `evidence/T2_19_CERT/` |
| T2.17 动作级反馈（`af0b915`） | `e_pi` revert（14.39 / 18.69），`e_pull` no_increment（15.20 / 29.23）；反馈启动期（250–500 步）落后，之后追近 `e_priv` 但不超过 | `evidence/T2_17_PULL/` |
| T2.16 A 真值方向审计（`323b8c8`） | pass：转向决策中修正方向胜 14/20；B 启动失败（命令行漏项）后由 T2.17 取代 | `evidence/T2_16_PI/` |
| T2.14 终局反事实（`03bf0d3`） | `e_term` revert（13.25 / 11.64），`e_term2` unsafe 1（9.27 / 2.37）；终局标签真值审计是准的，仍更差 | `evidence/T2_14_TERM/`、`T2_14_PRECHECK/` |
| T2.12 相对反事实（`2e7d838`） | `e_rel` revert（AUC 10.46，回报 17.42）；消融 `e_priv`（无反事实）25.43 / 37.69，成为最终方法 | `evidence/T2_12_EACC/` |
| T2.11 EACC-A（下层，`ec394ef`，两臂从零 × 种子 0、1 × 2000，reduced_priv） | `t28` pass：AUC 25.79 对 −15.24；紧急捕获 15–20 对 6–11（/25）；unsafe 0；执行校准第 512 / 550 步可信、准确率 0.945 / 0.930；优势辅路每批 0.3–1.1 个状态 | 科学 `5e42c47`，`evidence/T2_11_EACCA/`；上层复核 `evidence/T2_11_EACCA_UPPER/`；collab `dd5fa17` |
| T2.11 设计与自检（上层，`ec394ef`） | 校准模块在 P4 数据复现 0.929；真实环境 48 维观测与标签端到端；视图分离由测试保证；自检发现优势辅路很少触发，主路改为执行器判断上传 | `evidence/T2_11_SELFCHECK/` |
| T2.8 P4（下层，`8016b16` / `0dfdb8a`，不训练） | 旧标签、`opp`、`opp2` 均未过预注册判据；上层复核：原接口把规划答案交给 SAC；规划器与真实执行不一致（210/221），决策级校准留出块 0.93 | 科学 `27d59a5`，`evidence/T2_8_P4_INFO/`、`T2_8_P4_INFO_UPPER/`；collab `34fc0c4` |
| T2.8 夜间（下层，`24a342f`） | `t28` harmful：AUC 20.76 对 25.78；P3：带 / 不带标签的 critic 排序相同 | 科学 `3d215b6`，`evidence/T2_8_NIGHT/` |
| T2.7（下层，`2208a51`） | `harmful`；引导 −4.6、标签 +6.8（按臂）；执行层安全缺口修复 `be596d9` | 科学 `ae96fa2`，`evidence/T2_7_GUIDE/`、`T2_8_PRECHECK/` |
| T2.6W1（下层，`e2d4c14`） | `no_gain`：−8.21 / +19.69 | 科学 `7dac3dd`；collab `c92acd4` |
| T2.6.1 补充修复（`e2d4c14`，研究层 10-30 要求的事件先后核验） | 预测入口时刻之前若 Δv 预算已用完，标签按真值在耗尽处终止（资源耗尽，kind 0），不再记为入口；第一、二阶段都适用。罕见但可达（T2.6W 训练中每个训练约 1–3 次资源耗尽）。新增两项功能测试（在 `8e612e7` 上失败），共 48 项 | 科学 `e2d4c14` |
| T2.6.1 修复（`ca96793`、`8e612e7`） | 第一、二阶段预测在期限 / 燃料预算内到期时按真值终止（代价 − 超时罚分，不自举，kind 0）；入口统计与审计不再把它算作入口；新读数 `stage1_cutoff_share`、`stage2_cutoff_share`；t26 / t26w 支持 `--checkpoints`。46 项测试 | 科学 `8e612e7` |
| T2.6W（下层，`e1166eb`，两臂从零 × 种子 0、1 × 4000，紧急 U(140, 200)，48 开局，四点评估） | EACC − SAC 配对：种子 0 为 −20.1 / −9.4 / −8.8 / −0.2，种子 1 为 −6.4 / −9.8 / −21.9 / −14.5（1000–4000）；紧急 −17 至 −43，经济 +1 至 +13；机制全过；反馈首步 337 / 512；入口偏差 −6.4 / −8.2。下层定位：第二阶段标签无截止终止 | 科学 `605f3e4`，`evidence/T2_6W_WIDE_140_200/`（含 `诊断/问题定位报告.md`）；collab `4ad969a` |
| T2.6W 准备（`fc5a07b`、`e1166eb`） | 分档、双方成功时的燃料 / 时间、末期窗口读数，判定 t26w；校准跨 worker 共享（沙箱首个模型步 512 → 362）；48 开局完整性；`fc5a07b` 的读出测试失败 1 项，已修复 | 科学 `e1166eb` |
| T2.6 压力筛查（下层，`f4d660a`，648 例） | 无档入选：155–195 的 D1 不足（−1，需 ≥ 2）；140–180 的 F1 不足（G★ 21/24）。T2.6B（148–188）撤回 | 科学 `c761433`，`evidence/T2_6_PRESSURE/` |
| T2.5.1 验证（下层，`34f877e`，EACC 种子 0、1 × 4000，SAC 复用） | `mixed`：四点配对 +3.94 / −0.82；种子 0 在 4000 步 54.58 对 54.48；机制全过（w2 0.96 / 0.94，价值差 0.027 / 0.038σ，grad_cos 0.33 / 0.34，c 0.23 / 0.17） | 科学 `62d2d72`，`evidence/T2_5_1_VALIDATE/`；collab `e9cf679` |
| T2.5 验证（下层，`84ee9c1`） | c ≈ 0.15 / 0.17 < 0.5；用户决定停止（约 2084 / 1888 步），不评估 | collab `docs/collaboration/lower/handoffs/t2_5_validate_20261026/`；科学 `evidence/T2_5_VALIDATE/` |
| T2.4（下层，`e13c8b1`，EACC 种子 0、1 × 4000，SAC 复用 T2.3） | 种子 0 在 4000 步 53.79 对 54.48（−0.68）；种子 1 在 3000 步配对 +10.8，4000 步 55.24；捕获 23/24；描述性读数（用户缩减评估范围） | 科学 `632b633`，`evidence/T2_4_TEST/`；collab `docs/collaboration/lower/handoffs/t2_4_test_20261025/` |
| T2.3（下层，`9fc96e8`，种子 0 × 4000） | `coupling_failure`：MVE 42.36 对 SAC 54.48（配对 −12.1）；窗口承诺 A2 入口丢失 110 → 3、+9.1 / 回合；未锚定入口标签偏低约 23 | 科学 `evidence/T2_3_TEST/`；collab `docs/collaboration/lower/handoffs/t2_3_test_20261023/` |
| T2.2（下层，`fa54ed1`，种子 1 × 3000） | SAC 54.32（23/24），MVE 13.39（19/24），配对 −40.9；G★ 60.9；pure −48.9（14/24，unsafe 4） | 科学 `evidence/T2_2_TEST/`；collab `docs/collaboration/lower/handoffs/t2_2_test_20261022/` |
| T2.2 自主测试执行书（`1745c28`）与冻结（`fa54ed1`） | 研究层有条件 GO，两处修正完成（`ddd6778`：近场物理刹车判据；入口事件一致性门）。新增备用开关与判定脚本（`fa54ed1`），判定脚本在 B 段证据上复现研究层结论（`coupling_failure`）。沙盒冒烟：入口候选 100/622 | 科学 `ad5f3ad`、`ddd6778`、`fa54ed1`；上层设计文档已转研究层（未入 git） |
| T2.1 与 pure 行修复（`b597521`、`e8a96cf`） | 远场不解 MPC（远场阶段约快 5 倍）；pure 行原被远场 PD 接管，修复后 8 个训练开局 4/8（3 次燃料耗尽）；T2.1 下 c_10_0 在 24 个开发开局上 16.2，T2.0 为 16.9 | 科学 `e8a96cf`；沙盒（探索性） |
| T2 B 段（下层，单种子，3000 次决策） | 0k −51.15 → E-SAC +12.90（19/24）、E-MVE +23.42（20/24）。MVE − SAC +10.52，剔除最大两例后 −2.40，紧急任务 −0.26。G★ 直连 24/24、+59.56。训练墙钟 12.8 h | 科学 `evidence/T2_STAGE_B/`（`26fd421`、`28c96e0`）；collab `c3d403c`、`96324c2` |
| T2 A 段（下层，24 开局 × 9 策略） | 三道门全过：相对最佳恒定增益 38.8（去掉前两大后 23.4），最佳策略分散于 8 种 | 科学 `evidence/T2_STAGE_A/`（`4d3c35b`） |
| T2 分阶段进场上层预检（`79b658a`、`7d9b30b`，沙盒，探索性） | 冻结 MPC 在远场保持惯性点时外推（16 m：20 s 外漂 1.84 m、0.19 m/s），改为远场惯性跟踪律后为 0，链路首次贯通；8 开局筛选：最佳恒定动作外的逐开局余量明显，但相对截止期感知规则的余量 90 % 来自 1000005（研究层指出），去掉后约 +3；E-MVE 80 步冒烟机制通过，研究层指出的候选时间错位与校准器不一致已在 `3107420` 修复并单测 | 科学 `evidence/T2_PRECHECK/` |
| T1-B V8 阶段 I（`9c55ddc`）与上层复核（`468afb1`） | G0/G1/G2 过，G3 不过 → NO-GO。36 个配对开局：risk +5.32、random +5.57（分支数比 1.35），收益全部来自救援开局 293117（两臂都救回，去掉后 +0.19 / +0.45）；risk 触发 40/54 分支花在全失败开局；critic 风险 AUC 0.90，但备选排序全反；核验延迟 p50 195 s，为意图提前量的 1.8 倍（非实时）。未得到 SAC–MPC 学习耦合有效的证据 | 科学 `evidence/T1B_V8_STAGE1/`、`evidence/T1B_V8_STAGE1_UPPER_REVIEW/`；collab `reviews/T1B_V8_STAGE1_UPPER_REVIEW_20261021.md` |
| T1-B V8 上层预检与设计（`181f5b4`） | 79 个有全候选真值的状态：oracle 增益 6.28 中 4.69 来自 2 个救援状态（292140、293117），oracle 捕获余量 +2.5 个百分点（确认块 0）；用真值标签做 CE 偏好蒸馏（ECPG 上界）在留出块不超过总是最早（确认 0.4–11.0 对 20.96，独立 12.4–16.4 对 16.68）；“最早会失败”可学（AUC 0.83–0.98；V7 E-SAC critic 在开发块 1.0）；风险触发核验离线重组：独立块 23.45（触发 15/32，同率随机 20.19，全核验 24.17），确认 21.68（随机 21.47，全核验 22.12）；沙盒闭环 293117 核验后由 −131.7 变为 +53.13 | 科学 `evidence/T1B_V8_PRECHECK/` |
| T1-B V7 种子 0（`35c8f79`，collab `44a1bee`） | 首次出现 MPC 反馈的独立收益迹象（单种子）：E-TCPF 10k 主块 7/8、36.77，扩展块 7/8、39.72；E-SAC 两块 7/8、36.54 与 6/8、16.22，2k 起不变。差别来自状态相关首选：292126（急迫）选中间候选，由资源耗尽 −140.7 变为成功 +45.6；292123 +2.3、292145 +1.6 均为真值最佳；292125 −0.5。292140 改选 120 s，仍资源耗尽。Q 最佳一致 7/15 对 5/15。288/302 配对被采信，事实误差中位 0.085。种子 1 的三项条件全部满足 | 科学 `evidence/T1B_V7_ROUND/final_recovered/` |
| T1-B V6 种子 0 最终（`0cf973c`）与上层 V7 | E-SAC 2k/5k/10k 不变（5/15，全选最早）；E-TCPF 7/15→6/15→7/15，主块 +0.05，未满足种子 1 条件。MPC 优势在捕获/耗尽/超时边界出错，根因：反事实 MPC 每 5 周期才解 QP，Δv 多算 0.1–0.25 m/s。改为每周期解 QP 后 4 个边界开局中 3 个被复现（292140 与真值一致）。V7：真实控制器 + 异步推演 | 科学 `evidence/T1B_V7_PRECHECK/`（`54c2587`） |
| T1-B V5.1 中期（种子 0，`9e4afff`、`39d75ed`）与上层 V6 | 两臂 5k 逐例相同、全选最早；E-TCPF 139 个有效全时域配对，斜率随状态变化（−0.9..−5）但未翻转。根因：决策只占转移 ~2.5 %（每批 ~6 个），决策 Q 与 MPC 标签都被稀释；奖励不是原因。V6：意图级回放 + MPC 成对结局优势 | 科学 `evidence/T1B_V6_PRECHECK/` |
| T1-B V4 种子 0（下层，主机崩溃停于 5k，`2c7283b`）与上层判断 | 覆盖修好（非最早约 40 %），两臂 2k/5k 逐例相同、全选最早；15/15 状态 Q 最早第一，E-SAC 斜率加陡；97 % 的 6 s 标签随自举。判断：设计问题——缺长时信号，不是训练量；V5 从未执行。V5.1：首次决策全部候选的全时域 MPC 反事实、真实终止事件门控、份额 10/20 % | 科学 `evidence/T1B_V4_RESOURCE_RESTART1_20261020/`、`evidence/T1B_V5_PRECHECK/` 第 5 节（`b6f5c8a`） |
| 上层 V4 沙盒 5k 与 V5 定位 | V4 机制修好（非最早 53 %），但 5k critic 仍全选最早。到达时刻价值多为下降段 Δv 的 +2..+3（另有罕见成败），被 ±100..200 回报方差淹没；6 s 前缀看不到。MPC 模型全时域推演与真值：最佳 11/15、两两 31/38、事件 37/45。V5：E-TCPF 首次决策全时域成对反事实，事实分支须复现真实结局 | 科学 `evidence/T1B_V5_PRECHECK/`（`22046f0`） |
| T1-B V3 中期（种子 0，10k，`b2a65c5`）与上层 V4 定位 | M 通过；10k 两臂首选全部回到最早。原因：非最早只占真实决策 11 %（经济型×最晚 7 个），critic Q 对到达坐标线性外推（E-SAC 每档 −1.0→−17.6）；E-TCPF 单反事实使斜率平 5–8 倍但不足以翻转。V4：决策处 ε 0.5 + 反事实覆盖全部未选候选 | 科学 `evidence/T1B_V3_MIDTERM_20261019/`、`evidence/T1B_V4_PRECHECK/`（`8684cca`） |
| T1-B V2 试跑（种子 0，`a1698ac`）与上层 V3 定位 | V2 2k 熵恢复，5k E-SAC 重新集中最早（2/8 熵达标），A1 停止；梯度仍为 0k 的 77 %。根因：已有计划时每 6 s 的相对平移决策（12.7/13.9 次每回合），不是温度或梯度。MPC 6 s 前缀奖励差 0.0–0.5、在 292140 方向与真值相反：长期信息只能来自承诺状态的价值自举 | 科学 `evidence/T1B_V2_PILOT/`（`5c86561`）、`evidence/T1B_V3_PRECHECK/`（`e33a0dd`） |
| T1-B V1 上层离线审计（`bee2d43` 资产 + 292140–147 候选真值扫描） | 塌缩根因 = 温度量级（α 0.015 vs Q 差 ≥ 0.5）+ 覆盖饥饿（1k 后 100 % 最早）+ 饱和后梯度 ×1e-3..1e-4；MPC 反馈无部署差异 = 塌缩后果 + 标签 95 % 随自举符号 + 替代规则不配对“推迟”。真值：最佳首选 最早 3 / 中 2 / 最晚 3，8/8、60.87；总是最早 7/8、36.50 | 科学 `evidence/T1B_V1_AUDIT/`（`a1698ac`） |
| T1-P 上层扩展测试（用户追加约 30 min，`08f52b6`） | 规则 4 个修订版仍 FAIL 3/3；最终版几何就绪 23 / 20 / 0 步，但条件 G★ 全部拒绝（`beyond_deadline` 41 次、`infeasible` 2 次，期限 240 s）；放宽到 280 s 的诊断：292000 到达下降（最近 3.23 m）仍超时 | 科学 `evidence/T1P_ROUND1_PRECHECK/extended_08f52b6/`（`2a2ba3a`）；测试记录第 4 节 |
| T1-P 上层限时测试（12.2 min） | 接口 / 阶段动作 / 配对 / 冒烟 PASS；完整任务链 FAIL ×3（292000 超时、292001 unsafe 距离、292002 超时；从未触发桥接）；阶段执行误差约 0.5 m / 0.08 m/s，与阶段权限 0.761 m / 0.127 m/s 同量级，纯漂移阶段也如此 | 科学 `evidence/T1P_ROUND1_PRECHECK/`（`8296c59`）；`upper/T1_P_UPPER_TEST_RECORD_20261018.md` |
| 研究层裁决后的排序项实现与最终诊断（34 对开发状态） | 实现正确（134 项测试通过；冒烟中排序项被冻结并施加）；排序信号与真实长期差方向一致 15/25，纯计划同为 15/25，相关约 0.1；MPC 校正使前 2 s 误差 0.166 → 0.006，但 0 次改变排序。确定性 0k ≈ fixed:0 | 科学 `evidence/T1N_R2_RANK/`（`2c504c1`）；审查第 8 节 |
| R2 上层实质可行性审查（T1-N，开发开局 292000–293999，沙盒） | 物理与接口无阻塞：t = 0 合法 2000/2000；中性请求真实改变计划与闭环；拒绝后保持 300 s 0 unsafe；12 开局闭环：条件最早 63.12、原完整 G★-earliest 62.43、固定请求 65.96、随机 19.62。MPC 配对预测准确（h = 2 误差中位 0.021），但 6 s 奖励差与长期差符号一致仅 2/11（G★ 计划级 9/11）→ 主创新风险，建议计划展开标签（B）。预算：方案 A 墙钟约 2.7 天，B 约 2.2 天 | 科学 `evidence/T1N_R2_PRECHECK/`（`6d73894`）；审查 `reviews/R2_UPPER_FEASIBILITY_REVIEW_20261018.md` |
| T1-O V2 本机门控（执行单 `T1_O_V2_GATES_LOCAL_20261017` 第 2 版，工况 P2，λ_f = 20） | Gate A PASS（上层沙盒）；B PASS；**C FAIL**：状态规则 − A15 = +1.21 [0.01, 2.42]；状态规则 − A0（总是最早）= −0.03。零步、E、D 与四臂训练均未运行 | 科学 `evidence/T1O_V2/`；collab `lower/handoffs/t1o_v2_gates_20261017/`（`be5727d`）；上层审查 `reviews/T1O_V2_GATES_UPPER_REVIEW_20261017.md` |
| T1-O 物理筛选 | P1（15–19 m，2.1–2.7°/s）FAIL：1 次远场外漂 unsafe。P2（14–18 m，2.1–2.7°/s）原 S3 FAIL：F1 32/32 捕获、0 unsafe。原判定保留 | 科学 `evidence/T1O_SCREEN/`（`e8c0f08`） |
| 原 T1-O 工况（15–19 m，2.4–3.0°/s） | 搁置：未登记的试探开局中 6/20 远场外漂 unsafe | `evidence/T1O_SCREEN/README.md` 机理节 |
| T1-L-S（区间 Bellman，`672ba79`） | 正式训练取消，代码保留 | `evidence/T1_L_S_PRECHECK` |
| T1-L 三臂 40k（`166244b`） | N0 / NΔ 相对 0k 学习成立，但低于最强规则；NΔ − N0 未证明；B2 退化；规则族余量 0.21 | 交付 `68680eb`；审查 `reviews/T1_L_S0_UPPER_REVIEW_20261017.md` |
| T1 v4 M0 / M2 40k（`1d432c8`） | 学习未确立，"later"漂移，critic 高估 | 交付 `2242f1b`；审查 `reviews/T1_V4_S0_UPPER_REVIEW_20261016.md` |

更早的 T1 工作（R0 / R1、v2、根因、β、S1）和 F01 线见科学分支 `docs/HISTORY.md` 与 `registry/EXPERIMENT_REGISTRY_20261007.csv`。

## 3. 东西在哪里

| 内容 | 位置 |
|---|---|
| 科学代码、测试、证据、当前文档 | 分支 `claude/sac-mpc-coupling-design-ns7g6i`（`CLAUDE.md` 已更新到 EACH 冻结与预实验）；最终方法材料 `docs/final_method/` |
| 分支、标签、证据目录与本机产物总表 | 科学分支 `docs/archive/GIT_ASSETS_20261018.md`（T2 段未补，见 HANDOFF 6.1） |
| 执行单 | `upper/run_orders/`（T2 线：`T2_STAGE_AB` 至 `T2_19_CERT`；最终阶段：`RUN_PRETEST`、`RUN_FORMAL`（草案）、`PPT_FIGURES`） |
| 下层交付包 | 全部在 `docs/collaboration/lower/handoffs/<主题>/`（T2 的 7 个目录已于 10-29 从仓库根迁入；旧路径对照见 `lower/README.md`） |
| 上层审查 | `reviews/`（T2 线的审查结论直接写在本文件、HANDOFF 与研究演进中） |
| 用户汇报稿 | `wang23.pdf`（本目录） |
| 实验总账 | `registry/EXPERIMENT_REGISTRY_20261007.csv`（T2 段待补） |
| 本机工程 | 只有 `D:/py/DRL2`；过程目录 `Spacecraft/过程文件/<主题>/`（见 HANDOFF 6.5） |
| 历史文档与索引 | 科学分支 `docs/archive/` |

## 4. Git 结构（2026-10-18）

**只保留三条分支：**

| 分支 | 作用 |
|---|---|
| `claude/sac-mpc-coupling-design-ns7g6i` | 科学 |
| `collab/spacecraft` | 协作：只改 `docs/collaboration/`；**不得从该分支运行代码** |
| `main` | 实验结束后统一并入，现在不动 |

**已并入的实验分支：** `experiment/t1_l_option_relative_20261015`、`experiment/t1_l_s_tcpf_20261016`、`experiment/t1_o_tcpf_20261017` 是一条直线，已快进并入科学分支，并已由下层删除（远端核验只剩三条分支）。

**标签：** `archive/*` 全部保留，见同一文件第 3 节。历史从未重写。

## 5. 未完成

| 项 | 前提 |
|---|---|
| ~~最终方法统筹优化一版~~ | 已完成：`ECC_CERT_LOOP` 冻结；论文层 `docs/paper/`（METHOD、FACT_MAP、AUDIT、BASELINE、EXPERIMENTS） |
| ~~方法文档更新~~ | 已完成：`docs/final_method/` 旧规格标"非最终论文算法"，论文方法以 `docs/paper/METHOD.md` 为准 |
| 正式实验（`RUN_FORMAL`） | RUN_FORMAL_PREP 交付并经上层复核（逐位回归、冒烟、R 预筛判定）后上层重写；用户单独授权 |
| 汇报稿 2.9 节奖励与实现的差异 | 写论文时按实现写（势函数为到 24 s 后口部位置的距离） |
| `registry`、`GIT_ASSETS` 的 T2 段 | 上层补齐 |
| 正式块 300000–999999 的评估 | 未授权 |
