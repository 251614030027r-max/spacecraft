# 参考文献分析与取材笔记（ECC–RWG 论文，目标 IEEE TAES）

> 用途：写正文时从这里找"每一段话的文献来源与写法参照"。事实仍以 `docs/paper/facts/` 为准；本文件只管"引谁、怎么写"。
> 遵守 `References/README.md` 的"不要声称"列与 facts/05 §5.6 的"不主张"清单。
> 阅读范围：References/ 下全部 27 个 PDF（两份综述为同一篇）。哈工大、南航两篇逐页通读；其余全文提取后精读引言（脉络与贡献）、问题表述/建模、方法呈现（框架、命题、算法）与实验组织；Reiter 综述精读第 5、6、8、10.3 节（与本文定位直接相关的部分），其余章节只看结构。死区迟滞一篇只看组织形式，不引用。
> 写作时先看文首"★ 主参考模板"，再按第 2 节找每一段的文献来源。
> 书目信息按 PDF 首页记录；标 TODO 的卷期/页码/DOI 写 BibTeX 前需再核。

## ★ 主参考模板（优先对照）：Ren, Zhong, Gui, "Learning-Based Model Predictive Control for Cooperative Spacecraft Swarm Reconfiguration", IEEE TAES, vol. 62, 2026, pp. 818–833, DOI 10.1109/TAES.2025.3624705（文件：北航编队.pdf）

### 为什么选它
- **权威与同刊：** TAES 2026 正刊，北航宇航学院；投稿目标期刊的最新一篇，版式、章节命名、定理与算法框的排法就是 TAES 当前的写法。
- **方法类型最接近：** RL 不直接输出推力，而是给优化层提供"时机/调度"（机动时间序列），MPC 负责带约束地求解当前控制——与我们"RL 给阶段意图，执行层（规划器 + MPC）带约束执行"同属一类（Reiter 综述的 hierarchical architecture）。
- **行文顺序正是"先建模、再观测"：** II 坐标系 → 动力学 → 假设 → 优化问题（Problem 1）；III-A 先讲传统 MPC 的两个局限，再给集成框架的三条特征和优化层问题（Problem 2）；III-B 把 RL 部分写成 MDP：可观测的三类信息 → 归一化 → 状态表示 → 动作 → 奖励 → 网络 → SAC 更新 → 流程图；III-C 约束处理（Definition）；III-D 性质（Theorem + Proof + 汇总表）+ Algorithm 1；IV 训练过程（与基线的训练曲线）→ 典型算例 → 多基线 Monte Carlo 对比 → 鲁棒性；V 结论（带关键数字与展望）。
- **同样用 SAC，** 网络与更新公式的写法可直接对照。
- 对比：哈工大 Yang 2025 场景最近但方法与实验过于简单；北航 Tian 2024 约束写法最规范但主体是 ADP 理论。二者降为局部参照。

### 不照搬的地方
- 它的 III-D 给出递归可行与渐近稳定"保证"；我们只写结构性 Proposition（E1、E2、推论、重定时可行候选不减）与 Lemma（扫掠窗口），并在每条后写清"不蕴含什么"（facts/05 §5.6）。
- 它的 Monte Carlo 只报均值与成功率；我们按 EXPERIMENTS 的配对统计（种子分层、区间、翻转数）写，比它更严格。
- 它在结论里写具体百分比；我们结论里的数字全部等正式结果，先用 \todo{}。
- 它的引言较长且偏应用背景；我们引言最后写，按死区迟滞稿的"两个问题 → 两个机制 → 贡献"压紧。

### 逐节对照模板（左：Ren 2026 的写法；右：本文对应内容与事实来源）

| Ren 2026 | 它怎么写 | 本文对应小节 | 本文内容（事实来源） | 局部参照 |
|---|---|---|---|---|
| Abstract | 问题→框架（RL 嵌入 MPC）→RL 给什么、MPC 做什么→约束处理→性质→MC 与基线验证 | Abstract | 快速翻滚预捕获的决策–执行失配→ECC（执行反馈进评价器 + 训练/执行共用的机会认证）+ RWG→六自由度真值验证（U/E/N）→结果 \todo{} | — |
| I Introduction | 背景→现有方法两类各自局限→MPC 的参考与时域问题→RL 应用与端到端顾虑→learning-based MPC 分类→"把学习只当辅助部件"的不足→本文三大特点→贡献 1)2)3)（每条"Unlike/In contrast to…"）→组织段 | I Introduction（最后写） | 背景（BACKGROUND §1.1–1.2）→翻滚目标 MPC 路线及其默认"可等待"（Dong 2022/2024、Xu 2026）→RL/MPC 互补（Reiter §5）→RL–MPC 分类与本文位置（Reiter §6/§8）→两个问题（BACKGROUND §1.4–1.5）→ECC/RWG→贡献 3 条（BACKGROUND §1.7）→组织段 | 死区迟滞稿引言三段式；Tian 2024 "two critical problems" |
| II（开头段） | 一句话定义任务→引入两个坐标系，逐轴定义，配 Fig. 1 | II-A Coordinate frames and dynamics | ECI、目标/追踪器本体系、目标 LVLH（初始条件用）；端口与捕获点在目标本体系（facts/01 §1.2） | Lian 2026 坐标系 (i)–(iv) |
| II 动力学 (1)–(8) | 非线性相对运动→Assumption 1→线性化/离散化 | II-A（续） | SE(3) 刚体动力学：平动（中心引力 + J2 + 二阶质量矩修正）、转动（欧拉方程 + 重力梯度）、RK45、0.1 s 零阶保持（facts/01 §1.3）；记号按 Yang 2025；**Assumption 1（信息假设）**：执行层用恒速姿态预报与各向同性惯量估计，相对状态完全可测（facts/01 §1.3、facts/04 §4.6） | Yang 2025 (1)–(13) |
| II Problem 1 (9)–(12) | 把任务写成一个带约束的优化问题，变量与约束逐项解释 | II-B Constraints and capture；II-C Mission and Problem 1 | 七类真值约束（表格 + 口部区域、制动包络定义）、捕获条件（facts/01 §1.5–1.6）；任务：期限、预算、代价、真实回报与终止（§1.7–1.8）；**Problem 1**：在真值动力学与约束下，以观测为输入的策略使真实回报期望最大（§1.9）；一段"难点"：窗口周期接近期限 | Tian 2024 走廊/视场；Xu 2026 (48)–(50)；Dong 2024 接近速度 |
| III 开头 | 一句话预告 A/B/C/D 四部分 | III/IV 开头 | 预告：框架与执行模式→RWG→执行层→ECC | — |
| III-A Integration of RL and MPC | 先列传统 MPC 的"两个局限"→提出框架→"三条关键特征"带小标题编号段→优化层问题（Problem 2）及其凸性 | III-A Hierarchical framework and multi-rate execution | 先说明短时域约束优化缺长期视野、学习策略难以把握约束执行的后果→分层结构与 ECC 两通道（facts/02 §2.1）→编号特征段：1) Event-driven macro steps；2) Execution modes S/H/E/D（表）；3) Information partition o / z_E（facts/02 §2.2–2.5）；时间尺度表 | Reiter §6.1.3、§8.2.2 定位句 |
| （无对应；III-C 的 Definition 写法） | Definition + 解析式 + 示意图 | III-B Sweep-cone geometry（**Lemma 1**） | 自旋坐标系、λ/β/Δλ/φ、球面余弦式、窗口半宽 Δφ、几何机会集与"几何≠可执行"（facts/03 §3.1） | Dong 2022 时变 LOS 锥 |
| III-A Problem 2 的写法 | 优化层的代价与约束写成显式问题 | III-C RWG stage guidance；III-D Dynamic entry and entry re-timing（**Proposition 1**）；III-E Execution-layer planners and near-field MPC | 意图→目标速度、局部运动学、限幅与安全投影（facts/03 §3.2–3.3）；入口判定与重定时、可行候选不减（§3.4–3.5）；阶段规划器、入口规划器（SOCP 写成显式问题）、远场跟踪律、末段下降、近场 MPC（facts/04） | Lian 2026 SOC 约束写法 |
| III-B Determination of MTS via RL | 任务写成 MDP→"可观测的三类信息"→统一归一化系数→状态表示与公式→动作定义→奖励（分段式）→网络结构图→SAC 三个损失→流程图 Fig. 3 | IV-A MDP formulation, observation, execution feedback and reward | 宏步 MDP 与折扣 γ_d（facts/02 §2.4）→策略观测 o 的 18 项"按类别"列出（距离与速度 / 几何与相位 / 任务与燃料 / 制动裕度），统一按物理尺度归一化（facts/02 §2.5）→动作 a=(a_p,a_z)→真实回报与势函数塑形（facts/01 §1.8、facts/06 §6.2）→网络与 SAC 更新（facts/06 §6.1–6.3） | Yang 2025 观测归一化动机 |
| III-C Convexification（Definition 1 + 步骤图） | 先给定义，再分步骤构造，配 2-D 示意 | IV-B Mode prediction（Definition）；IV-C Certification（Definition + **Proposition 2**） | 模式预测"问什么、怎么算、输出、它不是什么"（facts/05 §5.2）；可认证候选集三条件、认证映射、E1（§5.3）；F5(a) 式 3×3 候选示意 | Leung 2024 / Lu 2025 对照"最小修改" |
| III-D Recursive feasibility and stability（Theorem + case 分析 + Table I + Algorithm 1） | 定理→分情形证明→汇总表→算法框 | IV-D Training–execution consistent learning（**Proposition 3**、**Corollary 1**）；IV-E Algorithm and overall diagram | 认证诱导转移核、存提议动作、E2 与推论、部分可观与函数逼近说明（facts/05 §5.4）；Algorithm 1（facts/05 §5.5 伪代码）；方法总图（§5.5）；证明放附录 | AC4MPC 命题–证明–Remark；Reiter §8.3.2 |
| IV 开头段 | 预告各小节要证明什么 + 运行平台 | V 开头 | 预告 RQ1–RQ4 与对应小节（EXPERIMENTS §1） | — |
| IV-A Training process | 环境参数表、初始条件采样式、训练曲线与基线对比、解释收敛原因 | V-A Setup, baselines and training | T1 环境/训练/信息划分/各方法设置；初始条件与任务分布（facts/01 §1.4、§1.7）；对比方法与消融定义（facts/07）；F1 学习曲线 \todo{} | Zheng 2026 参数总表；Zeng 2026 消融句式 |
| IV-B Typical cases | 选代表案例，画轨迹 + 关键时程，并标出决策时刻 | V-B Representative case | F2–F3：按事先规则选定的物理案例，目标本体系轨迹、阶段/认证标记、约束裕度时程（EXPERIMENTS §7）\todo{} | Xu 2026 Fig. 6–9 |
| IV-C Comparative experiments | 先介绍各基线"各有所长"→统一 MC→成功率/计算时间/燃耗→解释为何无人 100% | V-C Main results on urgent and economy missions | F4/T2：U 结局构成与回报配对差、E 共同成功配对的时间与 Δv、N 汇总（EXPERIMENTS §5）\todo{} | Ramezani 2026 Table 4 |
| （它没有部件消融） | — | V-D Attribution and ablations | F6：2×2 归因、执行时认证开/关与仅执行时认证、w/o 旋转引导、重定时开/关 \todo{} | Zeng 2026 §4.4 |
| IV-D Robustness analysis | 加入扰动后重复同一批 MC | V-E Mechanism and operating boundary | F5 认证机制（η / N_geo 分层）、F7 燃料与自旋率分层；在线开销 vs 16 s 决策周期 \todo{} | Yang 2025 Fig. 6；Zeng 2026 开销 |
| V Conclusion | 方法一句话→各模块作用→关键数字→应用前景→未来工作 | VI Conclusion | 方法与机制总结→结果 \todo{}→局限（编号）→未来工作 | Xu 2026 Limitations |
| 附录/参考文献 | IEEE 格式 | Appendix | 命题证明；超参数表；（可选）开发期负结果 T3 | Yang 2025 附录 |

### 句型与段落组织（从 Ren 2026 归纳，用我们的话写，不照抄原句）
- **问题陈述段：** 一句话定义任务 → "To define the variables and constraints involved in this problem, …"式的引出坐标系 → 公式 → 一条 Assumption 说明适用条件。
- **方法动机段：** 先承认传统做法的两个局限（各一句，First/Second），再一句"Therefore, we propose …"。
- **框架特征段：** "The proposed framework is characterized by the following features." + 带粗体小标题的 1) 2) 3)，每条两三句：做什么 → 为什么 → 带来什么。
- **MDP 段：** "The … task is formulated as a Markov decision process" → 智能体接收什么、输出什么、得到什么 → 指出效果取决于"状态表示、网络表达、奖励与目标的一致"三点 → 再逐一展开。
- **观测段：** 先说"可以观测到三类信息：1)… 2)… 3)…"，再给统一的归一化系数，最后给完整状态表示的公式。
- **奖励段：** 先说奖励的目标，再给分段式公式，再逐项解释系数。
- **流程图段：** 一句话说明图中决策与训练的闭环，正文与 Algorithm 1 互相引用。
- **实验段：** 每小节开头一句话说"本节要展示什么"；对比实验先说明各基线的来源与长处，再给统一的 MC 设置；结果段先讲最重要的指标，再解释原因。
- **结论段：** 一句话复述方法 → 各模块作用 → 关键结果（等正式数字）→ 局限与未来工作。

### 篇幅参考
Ren 2026 正文 16 页（参考文献 50 篇以上），图 10、表 4、算法 1、定理 1。我们预计：图 7–9（框架图、RWG 几何、F1、F2–F3、F4、F5、F6、F7 视篇幅合并）、表 3（T1、T2、对比方法设置）、算法 1、Lemma 1、Proposition 1–3、Corollary 1（重定时 = Prop. 1，E1 = Prop. 2，E2 = Prop. 3）；参考文献 40–50 篇。

## 1 总览：每篇文献在本文中的角色

| 文献（文件名） | 期刊 / 年 | 角色 | 主要用于本文哪里 |
|---|---|---|---|
| 哈工大 Yang 2025 | TAES 61(5), 2025 | 建模来源 + 场景先例 + 版式 | II 系统建模（SE(3) 动力学、追踪器参数）；V TD3/SAC 基线、按角速度分层；TAES 版式 |
| 北航 Tian 2024 | TAES 60(6), 2024 | 约束表述 + 工况对比 | II 约束（走廊、视场、饱和分类）；I 中"现有工作目标转速约 1°/s" |
| 北航编队 Ren 2026 | TAES 62, 2026 | **主参考模板**；RL–MPC 接口先例 | I/相关工作"RL 给 MPC 调度"；MPC 两个局限；III 框架特征写法；V MC 对比与鲁棒性组织 |
| 南航 Xu 2026 | AST 173, 2026 | 双层 MPC 对照 + 动机 | I（Limitation 1：快速翻滚下短时域预测失效）；相关工作"上层提议、下层过滤"；II 走廊/视场写法；VI 局限写法 |
| 上海交大 Gong 2026 | AST 176, 2026 | SE(3)+学习 MPC；表格版式 | II 建模动机一句；V 主表指标版式（终端误差 + 时间 + 控制量 + 求解时间） |
| Ramezani 2026 | Acta Astronaut. 238, 2026 | 最接近的航天 RL–MPC 应用 | 相关工作；V 成功率定义、终端误差、燃料预算曲线 |
| Lian 2026 | ASR 77, 2026 | pre-capture、SOCP 约束 | III 执行层 SOCP 规划的写法依据；"capture window"术语 |
| Dong 2022 | Acta Astronaut. 200, 2022 | 翻滚目标两阶段 MPC | I/相关工作核心对照；III 扫掠锥几何的文献依托（时变 LOS 锥、body cone/space cone） |
| Dong 2024 | ASR 74, 2024 | 翻滚目标鲁棒 MPC | I（目标未来行为难预测、惯量未知）；工况先例（约 2.7°/s）；局限中对照其 EKF |
| Dong 2025（南理工） | Acta Astronaut. 235, 2025 | 背景 | I"捕获只有有限时间窗口" |
| Wang 2026（NUDT） | AST 170, 2026 | 背景 | I 翻滚目标接近通常表述为单次最优控制/轨迹规划；soft docking 定义 |
| Lu 2026（交大） | AST 168, 2026 | 航天安全 RL | 相关工作（CMDP 在学习层处理约束）；V 回报与约束并列报告 |
| Zheng 2026 | Acta Astronaut. 249, 2026 | 事件触发 RL–MPC | 相关工作（RL 管时机）；V 参数总表、评价指标小节、敏感性 |
| Tammam 2025 | CEP 156, 2025 | 分层 DRL 航天 | 相关工作一句 |
| Adjei 2026 | AST 178, 2026 | SAC+MPC 分层（航空） | 相关工作"长时域最优 vs 约束权限" |
| Reiter 综述 2026 | Annu. Rev. Control 61, 2026 | 定位与理论依据 | I RL/MPC 互补；本文分类定位（hierarchical / MPC as part of environment）；E2 动机（§8.3.2 后处理滤波器） |
| AC4MPC | IEEE TCST, 2025 | RL–MPC 代表 + 命题写法 | 相关工作（critic 作终端代价）；IV 命题–证明–Remark 组织 |
| ecc26_rollout | ECC 2026 | RL–MPC 代表 | 相关工作（学习给名义解、优化精修） |
| Leung 2024 | SCL 185, 2024 | 参考整形 | 相关工作（对上层命令做最小修改）——见待定事项 1 |
| Tartaglione 2024 | Automatica 166, 2024 | 参考整形 | 同上；"Problem 1 / Problem 2"分层陈述写法 |
| Wu 2024 | IEEE 期刊（TODO） | MPC 价值估计 | 相关工作（预测用于价值目标；ECC 不这样做） |
| Zeng 2026 | Neurocomputing 701, 2026 | 实验组织模板 | V "w/o 部件"消融句式；在线开销对比决策周期；种子带曲线 |
| Lu 2025 | EAAI 161, 2025 | 安全层 + 混合学习 | 相关工作；E2 中"存提议动作"的对照；V 屏蔽开/关组织 |
| Duan & Zou 2025 | Autom. Constr. 174, 2025 | 跨领域分层安全 | 相关工作一句 |
| 死区迟滞（用户首篇） | 中文期刊稿 | 组织形式 | 引言"两个问题→机制→贡献"；实验章节顺序；不引用 |

## 2 按本文章节的取材地图

**I Introduction（最后写）**
- 在轨服务与预捕获背景：Tian 2024（OOS 代表任务）、Lian 2026（pre-capture 需要窄走廊与持续指向以形成 capture window）、Xu 2026、Dong 2025（翻滚非合作目标、捕获时间窗口有限）。
- 快速翻滚的难点与既有 MPC 路线：Dong 2022（两阶段：进时变 LOS 区 → 周期跟踪端口）、Dong 2024（tube MPC + EKF，目标未来行为难预测）、Xu 2026（双层 MPC；Limitation 1 自述"翻滚速率相对预测时域须适中"）、Wang 2026（翻滚目标接近通常表述为单次轨迹优化）。对照点：这些工作多数默认"可以等待下一次对准"，没有期限与燃料预算下的窗口稀缺。
- 工况对比：Tian 2024 约 1°/s；Dong 2024 约 2.7°/s 但无期限/预算；本文 2.1–2.7°/s 且端口回访周期接近任务期限。
- MPC 与 RL 的能力边界：Reiter 综述 §5（性质近乎正交）、AC4MPC 引言（RL 近全局低精度 vs MPC 局部高精度）、Ren 2026（MPC 两个局限：参考难得、有限时域只得局部最优）、Xu 2026。
- 学习在航天控制：Yang 2025（SE(3) 上 PPO/TD3/SAC 端到端）、Tammam 2025（分层 DRL）、Lu 2026（安全 RL）、Ren 2026（端到端 RL 的可靠性顾虑）。
- RL–MPC 结合与分类：Reiter 综述 §6（三大类、五种参数化架构）、Gong 2026（学习代价权重）、Ramezani 2026（MPC 轨迹进奖励）、Ren 2026 / Zheng 2026（RL 决定机动或重优化时机）、Adjei 2026（按阶段分配权限）、AC4MPC / ecc26（critic 作终端代价、学习给名义解）、Wu 2024（预测用于价值估计）。
- 决策–执行失配的文献依据：Reiter 综述 §8.3.2（学习时不可见的后处理滤波器使策略不能期望闭环最优）；安全/参考整形层对上层命令做最小修改：Lu 2025、Zeng 2026、Leung 2024、Tartaglione 2024、Duan 2025——共同点是修改发生在学习之外或只为可行性/安全，不回馈"机会是否存在"。
- 组织：沿用死区迟滞稿与 Tian 2024 的"现有工作 → 两个问题 → 对应机制 → 贡献"写法；贡献句式参考 Yang 2025（每条以对比前人收尾）。

**II Problem Formulation / System Modeling**
- 坐标系逐条定义：Lian 2026 (i)–(iv)、Xu 2026 Definition 4。
- SE(3) 六自由度动力学：直接按 Yang 2025 (1)–(13) 的记号（T、ξ、ad*、Ad、引力梯度、J2、RK45）；"建模沿用、不作贡献"。动机一句可并引 Gong 2026（耦合建模三分法）。
- 追踪器参数来源：Yang 2025（106 kg、惯量）；推力/力矩上界为本文设定。
- 约束：分类与记号参照 Tian 2024（motion constraints：接近走廊、视场；input saturation）；端口锥写法参照 Xu 2026 (48)–(50)；接近速度约束参照 Dong 2024 §2.2.2；捕获条件 / soft docking 动机参照 Wang 2026。
- 任务（期限、预算、代价）与 MDP：本文自有；MDP 记号参照 Reiter 综述 §2；"Control objective"独立一句参照 Tian 2024。

**III 框架与 RWG**
- 分层定位句：Reiter 综述 §6.1.3（hierarchical architecture）+ §8.2.2（MPC as part of the environment）。
- 框架特征的编号段写法：Ren 2026 III-A（"1) Receding Horizon Implementation: …"）。
- 扫掠锥 Lemma：几何为本文推导；文献依托 Dong 2022（翻滚目标时变 LOS 锥、body cone 绕 space cone）。
- 执行层 SOCP 规划与制动包络：Lian 2026（走廊/推力作为二阶锥约束，"重在建模与约束集成而非新求解算法"）。
- 远场跟踪律 + 近场 MPC：MPC 写法参照 Xu 2026 Controller 1/2 的格式（但不写稳定性定理）。

**IV ECC**
- 命题组织：AC4MPC（Assumption → Lemma/Proposition → Proof → Remark，结论后加适用范围限定句）、Leung 2024（Proposition + 简短证明）。证明放附录参照 Tartaglione 2024。
- E2 动机：Reiter 综述 §8.3.2；与 Lu 2025 的"混合学习（投影动作/原动作）"对照说明为何存提议动作。
- 与最小修改类方法的差别：Leung 2024 / Tartaglione 2024 / Lu 2025 / Zeng 2026。
- 算法框：Yang 2025 附录伪代码、Lu 2026 §4.3"learning and deployment"。

**V Experiments（结果 \todo{}）**
- 设置与公平性：参数总表参照 Zheng 2026 Table 1、Yang 2025 附录超参表；公平比较说明句参照 Tian 2024。
- 基线：TD3 与 SAC 对照参照 Yang 2025；纯 MPC 作传统基线参照死区迟滞稿 3.4、Ramezani 2026（LQG-MPC）。
- 指标：成功率（全部阈值同时满足）、终端六自由度误差、控制量参照 Ramezani 2026 Table 4、Yang 2025 Table III、Gong 2026 Table 4；评价指标先单列一小节参照 Zheng 2026 §4.2。
- 消融：句式参照 Zeng 2026 §4.4（"without X, which retains … but disables …, thereby evaluating …"）；组件逐一加入的组织参照 Zheng 2026 §4.5。
- 执行时认证开/关、介入案例：Lu 2025。
- 在线开销对比决策周期：Zeng 2026（每次决策耗时 vs 决策间隔）、Gong 2026（平均求解时间列）。
- 边界：按目标角速度分层参照 Yang 2025 Fig. 6；燃料预算参照 Ramezani 2026 Fig. 15。
- 案例时程：Xu 2026 Fig. 6–9（分阶段子图 + 时间线叙述）、Yang 2025 Figs. 7–9。
- 章节顺序参照死区迟滞稿：性能对比 → 消融与机制验证 → 与传统 MPC 对比 → 边界。

**VI Conclusion / Limitations**
- 编号局限写法：Xu 2026 §5（"Limitation 1 (…): …"）；可对照 Dong 2024 的在线惯量估计说明本文执行层只用恒速预报与各向同性惯量估计。

## 3 逐篇笔记（期刊、结构、问题表述、方法呈现、实验组织、可借鉴写法、与本文的关系）


### 哈工大 = Yang, Yang, Hu, Dong, TAES 61(5), Oct 2025, pp.13677–13688(?), DOI 10.1109/TAES.2025.3581155
- 题目：Spacecraft Autonomous Rendezvous Control With Rotation Target on SE(3): Deep Reinforcement Learning Methods
- 结构：I Intro（文献脉络：对偶四元数→SE(3)→DRL 轨道/姿态→6DOF DRL [Oestreich PPO, Hu, Yang MBRL]→不足：未计摄动/不确定→传统控制→燃料；贡献 3 条 + 组织段）；II Preliminaries and Problem Formulation（A Lie group SE(3) 基础 (1)–(3)；B 动力学 (4)–(13)，ECI，引力梯度力矩、J2、RK45，相对位形 T_r = T_t^{-1}T_c、ξ_r = ξ_c − Ad ξ_t；C RL 基础 MDP/Bellman）；III Control Law Design（A 环境 SE3SAREnv 与奖励：state screw 对数映射归一化 (23)–(27)，奖励 = 塑形 + 积分补偿 + 终局 ±40，Remark 1；B RL 算法：PPO/TD3/SAC 公式）；IV Simulation（参数：目标 225 kg，追踪器 106 kg、J_c diag(20.3,19.9,23.7)+交叉项——与我们追踪器参数同源；推力 ±1 N、力矩 ±0.6 N·m；d_max=30 m、T_max=200 s；A SMC 基线 + 成功率–角速度倍率曲线（每点 500 次 MC）；B 训练曲线；C 1000 次 MC、终端误差表、轨迹长度/控制代价分布）；V Discussion；VI Conclusion；Appendix 伪代码 + 超参数表。
- 与本文的关系（可作来源）：
  - 我们的 SE(3) 建模直接沿用它：齐次矩阵 T、se(3) 扭量 ξ=(ω,v)、Ṫ=Tξ^、Iξ̇=ad*_ξ Iξ+χ_g+χ_c+χ_d、引力梯度力矩与二阶质量矩修正、J2、RK45、相对位形与伴随映射。系统建模节"按 [Yang 2025] 的 SE(3) 表述"写，明确不作贡献。
  - 追踪器质量/惯量、目标质量、30 m 最大距离、0.1 s 步长同源 → 写"chaser parameters follow [Yang 2025]"。（目标惯量不同：我们 diag(30,39,48)，它 42.7/40.3/37.9）
  - 观测归一化"按物理尺度"可引其 state screw 归一化动机（数据范围不均）。
  - 引言脉络：DRL 用于 6DOF 交会（Oestreich、Hu、Yang MBRL）可转引；它代表"端到端 DRL 直接输出力/力矩"，是我们"RL 只给阶段意图、执行层带约束"的对照位置。
  - 实验：PPO/TD3/SAC 对比 + 成功率–角速度曲线 → 我们 TD3 基线、按自旋率分层的依据。
- 写法可借鉴：贡献点"1) 2) 3)"每条以对比前人收尾（"Compared to the models in [19]–[21]…"）；"The rest of this article is organized as follows"；Remark 用于解释奖励设计；伪代码与超参数放附录。
- 注意：它的期刊格式即 TAES 正式版式（单栏摘要、罗马数字节号、A/B/C 小节、Remark/Algorithm、附录）。

### 南航 = Xu, Kang, Li, Wu, Zhou, Hao, Wu, Aerospace Science and Technology 173 (2026) 111572, DOI 10.1016/j.ast.2025.111572（= BACKGROUND 中的"Xu 等 2026"）
- 题目：Dual-layer model predictive control for autonomous capture of failed targets under complex obstacle conditions
- 结构：1 Intro（在轨服务动机→APF 局部极小→最优控制处理避障→MPC 在交会中的发展 [Richards, Hartley 分阶段 MPC, Zhu LOS, Iskender 变 LOS]→CBF 优势→MPC+CBF 冲突→本文；贡献 3 条 + 组织段）；2 Problem Formulation（2.1 预备定义 CLF/CBF Definition 1–3；2.2 动力学：CW 型相对轨道 + MRP 相对姿态 + 耦合 12 维；Definition 4 坐标系；2.3 Problem statement：期望位置 r_d = R_ot r_dt 随目标转动，误差定义）；3 Controller design（3.1 MPC-CLF "Controller 1"；3.2 MPC-CBF "Controller 2" 作为安全滤波；3.3 死锁检测（Definition 5 + 事件触发函数）；3.4 其他约束：安全接近走廊 SAC（端口锥）、视场（矩形锥）线性化；3.5 可行性/稳定性/安全 Assumption 1–4 + Theorem 1–2 + 证明）；4 Simulation（参数表 Table 1/2、单案例分阶段 6 子图、距离曲线、位置姿态曲线、力/力矩、视场轨迹、11 个初始位置误差表、消融：无死锁机制 vs 有、单层 vs 双层）；5 Limitations and future work（Limitation 1：LTI 预测模型隐含"翻滚速率相对预测时域适中"）。
- 与本文的关系：
  - 引言/相关工作：代表"上层提议、下层过滤 + 死锁检测"的双层 MPC；README 规定不得把分层结构声称为新颖——我们写"不同在于上层由学习得到、面向长期任务决策，且工况为快速翻滚（窗口周期接近任务期限）"。
  - 其 Limitation 1 是我们的动机句来源："短时域预测在快速翻滚下失去长期视野"可引用它的自述局限。
  - 约束建模写法：端口锥（SAC）用 r_p 与夹角 β 的不等式、视场为锥约束、期望位置随目标转动 r_d = R_ot r_dt → 我们的约束表 / 捕获条件可仿照"Definition + 不等式"写法。
  - "安全接近走廊 safe approach corridor"术语可参照；我们 facts 用 approach cone (corridor)。
- 写法可借鉴：Problem statement 小节独立；"Controller 1/2"式的方法框；每个约束一小节（3.4.1/3.4.2）；Limitations 单列编号；结果段"Fig. X(a) shows… at approximately 25 s"的时间线叙述。
- 不借鉴：其稳定性定理（我们不主张保证）；单一初始条件 + 11 点表的实验规模（我们是配对 MC）。

### 北航 = Tian, Shi, Shao, Hu, Yang, Zhu, TAES 60(6), Dec 2024, pp.7838–7850(约), DOI 10.1109/TAES.2024.3419763
- 题目：Spacecraft Proximity Operations Under Motion and Input Constraints: A Learning-Based Robust Optimal Control Approach
- 结构：I Intro（OOS 计划 Restore-L/Orbital Express/ClearSpace-1/ETS-VII→6DOF 位姿控制→MPC 与 RL/ADP 是最优控制两大类→ADP 面临"多约束与鲁棒"两问题→约束分类：运动约束（接近路径、视场）+ 输入饱和→贡献 2 条 + Notations）；II Preliminaries（坐标系 N/L/S/T；A 相对位置动力学，含 J2 与 Coriolis/切向/径向项解释；接近走廊 α=arccos(ρ̂·R n1)<α_m；B 相对姿态，视场 β<β_m；C 推力器配置；D Problem Formulation：耦合来源三条 + Remark 1（对偶四元数 vs 分离-集成建模）+ "Control objective"一段）；III Main Results（A 有限时间扰动观测器 Lemma 1 + Proof；B ADP 学习控制：性能指标含障碍函数、HJB、NN 逼近、Theorem 1 有界性）；IV Numerical Simulations（Molniya 轨道参数表；A 性能验证；B 对比 PD+、APF，调参使调节时间相同以保证公平；C 脉冲调制下的实际场景）；V Conclusion。
- 关键事实：目标角速度 [0.01,−0.01,0.01] rad/s（≈0.0173 rad/s ≈ 0.99°/s），追踪器 100 kg、J diag(22,20,23)、推力 10 N，接近走廊 15°、视场 30°，单一初始条件（约 155 m 外）。
- 与本文的关系：
  - 约束写法来源：接近走廊与视场的"夹角 < 半角"表述、约束分类"motion constraints + input saturation"——我们 Section II 约束表直接沿用这一分类与记号风格。
  - 背景句：OOS 代表任务列表、"6-DOF pose control problem"的表述。
  - 工况对比（README 规定）：它的目标角速度约 1°/s 且无速度上限 → 写"existing constrained proximity designs typically consider targets tumbling at about 1°/s [Tian 2024]，本文 2.1–2.7°/s，端口回访周期接近任务期限"。
  - Remark 1 关于耦合建模的论证可作为我们"建模沿用 SE(3)、不作贡献"一句的旁证。
- 写法可借鉴：Introduction 中"two critical problems, namely, X and Y"的提出方式；"Control objective:"独立一句；公平对比的说明句（"For a fair comparison, all parameters are tuned to…"）。

### 北航编队 = Ren, Zhong, Gui, TAES vol.62 (2026), pp.818–833, DOI 10.1109/TAES.2025.3624705
- 题目：Learning-Based Model Predictive Control for Cooperative Spacecraft Swarm Reconfiguration
- 结构：I Intro（集群背景→规模化轨迹优化计算量大→脉冲机动 + APF 固定 Δt→MPC 依赖参考轨迹、计算慢→RL 在空间任务的应用→端到端 RL 稳定可靠性存疑→learning-based MPC 三类（学动力学/学控制器设计/安全保证）→"仅把学习当 MPC 的辅助部件"之不足→三大优势 + 贡献 3 条 + 组织段）；II Preliminaries and Problem Formulation（ECI/LVLH、CW、脉冲模型、问题定义）；III Learning-Based MPC Control Design（A Integration of RL and MPC：先列传统 MPC 的"two critical limitations"（参考难得、有限时域只得局部最优），再给框架三特征；B RL 决定机动时间序列 MTS（RL 给 MPC 提供调度/时机）；C 避碰约束凸化（可达集）；D 递归可行与稳定 Theorem 1 + Table I 汇总定理）；IV Simulation（A 训练过程；B 典型算例；C 与 APF、SCP-MPC、SCP 三基线比较：500 次 MC，成功率/计算时间/燃耗；D 鲁棒性分析：重复 MC）；V Conclusion。
- 与本文的关系：
  - README：借鉴"RL 给 MPC 提供调度"的接口模式，不借鉴其脉冲模型。与我们相同点：RL 决定"何时、以何节奏"，优化器决定"怎么飞"。可在相关工作写"RL selects maneuver timing for an MPC [Ren 2026]；本文 RL 给阶段意图且接口带执行反馈"。
  - 引言可借其"MPC 两个局限"表述，和 Reiter 综述一起支撑"短时域优化缺长期视野"。
  - 其"learning-based MPC 三类"分类可与 Reiter 综述的分类对照，用于定位本文属于"学习任务参考 / 高层调度"。
  - "end-to-end RL 稳定可靠性存疑"一句是我们不用端到端 RL 的依据之一。
- 写法可借鉴："The proposed algorithm offers three major advantages. First,… Second,… Third,…"；框架特征"1) Receding Horizon Implementation: …"带粗体小标题的编号段；对比实验先交代每个基线"各有所长"再比较，解释为何无方法 100% 成功（时间不足）——与我们超时失败的叙述相似。

### 上海交大 = Gong, Mu, Liu, Chu, Yi, Aerospace Science and Technology 176 (2026) 111966, DOI 10.1016/j.ast.2026.111966
- 题目：A3C-based neural-cost model predictive control on SE(3) for autonomous spacecraft rendezvous and docking
- 结构：1 Intro（SPO/SOM 分类→近距离非线性与耦合→三种耦合建模 SICDM / 对偶四元数 / SE(3)（"85% 研究集中于前两者"）→SE(3) 上 SMC/TSMC 的不足→MPC 优点与"性能依赖代价函数设计"→学习+MPC 三类 [MPC 引导策略搜索 / NN 策略供 MPC / Learning-based MPC] + 第四类学习代价函数→不足→本文）；2 SE(3) 建模（2.2 刚体耦合动力学，2.3 相对耦合动力学）；3 NCMPC on SE(3)（3.2 A3C 训练框架改进 MPC）；4 Simulation（单案例；初始位置/姿态误差扫描的终端误差热图 Fig.10–11；Table 4 NCMPC/MPC/MLP 三方对比：终端误差、调节时间、控制能量积分、平均 MPC 求解时间）；5 Conclusion and future works。
- 与本文的关系：
  - README：只借"三方对比表的格式"；其方法（在线调 MPC 代价权重）已被排除，不作为我们的候选做法。
  - Table 4 的指标组织（终端四项误差 + 时间 + 控制量 + 每步求解时间）→ 我们 T2 主表 / 物理可信度指标的版式参考；"平均 MPC 求解时间"一列对应我们在线开销。
  - 引言中耦合建模三分法与 SE(3) 优点的论述 → 系统建模节一句话动机可引它与 [Yang 2025]。
  - 学习+MPC 分类（引用 Hewing 等）可与 Reiter 综述互证。
- 写法可借鉴：分类综述时编号"1) 2) 3)"，每类一句优点一句缺点；"However, most of the above methods…"收束相关工作。

### Ramezani = Ramezani, Alandihallaj, Yalçın, Olivares Mendez, Hein, Acta Astronautica 238 (2026) 690–705, DOI 10.1016/j.actaastro.2025.09.046
- 题目：Fuel-aware autonomous docking using RL-augmented MPC rewards for on-orbit refueling
- 结构：1 Intro（在轨加注意义→燃料晃动建模文献→MPC 局限：计算量、依赖精确模型→RL 优势与"训练代价大"→与已有良好控制器结合可加速训练→PPO-MPC 到 SAC-MPC；贡献 4 条（圆点）+ 组织段）；2 动力学模型（含 Zero-G Lab 平台、OpenFOAM 晃动）；3 MPC 框架；4 RL 方法与集成框架（4.2 SAC 网络，4.3 奖励函数：把 MPC 最优轨迹写进奖励）；5 实验（实物 3DoF + 6DoF 仿真；Monte Carlo；Table 4：PPO / SAC / PPO-MPC / SAC-MPC / LQG-MPC 五方法，成功率 + 终端位置/速度/姿态/角速度误差（均值±标准差）+ 归一化控制量 NCE；Fig.15 成功率随燃料量变化）；6 Conclusion。
- 与本文的关系：
  - README 定位：最接近的航天应用（RL 增强 MPC 对接）。相关工作写"RL 与 MPC 在奖励层耦合 [Ramezani 2026]"；我们的耦合在"意图–执行接口"，并把执行信息回馈给评价器与动作。
  - 实验组织借鉴（EXPERIMENTS §0 已对应）：成功率定义为全部阈值同时满足、终端六自由度误差、控制量按每次试验归一化、成功率随燃料量曲线 → 我们 E 任务燃料预算与 F7 边界图。
  - 名称"SAC-MPC"作为一类方法的通称有先例，我们的基线命名 SAC–MPC 与此一致。
- 写法可借鉴：贡献用圆点、每条以动词开头"We integrate… / We employ… / We extend… / We validate…"；表格"均值 ± 标准差"；NCE 的"公平比较"解释句。
- 注意：它没有约束安全的真值判定，也无配对统计；不引用其数字。

### Lian = Lian, Wang, Yang, Zhao, Advances in Space Research 77 (2026) 6084–6100, DOI 10.1016/j.asr.2025.12.114
- 题目：Autonomous SOCP–MPC for space debris pre-capture rendezvous and pointing
- 结构：1 Intro（ADR 背景→pre-capture 需要窄走廊 + 持续指向以提供"capture window"→DG 与 MPC 文献→凸规划编码 LOS 走廊、SOC 约束 [Fear & Lightsey 2024, Ortolano & Geller 2021, Malladi 2019]→缺口→本文强调"控制导向建模与约束集成，而非新数值算法"）；2 Dynamics modeling（Assumption 1–3、四个坐标系逐条定义、CW 状态空间）；3 SOCP–MPC scheme（3.1 离散 6DOF 预测模型；3.2–3.4 Type-1/2/3 SOC：代价改写为旋转锥、走廊锥、指向锥；3.5 时变锥收紧与算法）；4 Numerical simulations（4.1 场景与配置；4.2 与 DG–PD 基线（有/无人工引导）比较；4.3 参数研究与权衡：Nominal/Fast/Slow）；5 Conclusions（贡献列在结论中）；附录 A 目标机动、附录 B 基线策略。
- 与本文的关系：
  - 术语 pre-capture、"capture window"、走廊与指向作为二阶锥约束 → 我们阶段规划器与入口规划器都是 SOCP（推力 2-范数、制动包络弦线），可引它与 Fear & Lightsey 2024 支撑"以凸锥约束表述接近走廊与推力上界"。
  - 其"强调建模与约束集成而非新求解算法"的表态，可借作我们执行层段落的定位句（执行层不作为创新）。
  - 参数研究"Nominal/Fast/Slow"展示时长–入锥时机权衡 → 对应我们紧急/经济两类任务偏好的叙述。
- 写法可借鉴：坐标系 (i)(ii)(iii)(iv) 逐条定义；Assumption 编号；基线细节放附录；结论中列贡献。

### Wang (NUDT) = Wang, Chen, Zhou, Shan, Zhao, Bai, Aerospace Science and Technology 170 (2026) 111518, DOI 10.1016/j.ast.2025.111518
- 题目：Manifold-based trajectory planning for safe proximity and soft docking to a tumbling target by modular multi-spacecraft system
- 结构：1 Intro（模块化多航天器→"两个关键挑战"（能量协调、翻滚带来的安全风险）→soft docking 定义（终端相对位置/速度在容差内、径向速度尽量小）→直接法/间接法优缺点→"三大挑战"→流形方法"四个优点"→贡献）；2 Problem formulation（2.1 动力学，2.2 轨迹优化模型）；3 流形方法（测地线定义）；4 旋转 3D-Dubins 初值（4.1 翻滚目标捕获点运动分析）；5 势函数度量；6 实验（6.1 与 MPC-SCP 对比、6.2 插值影响、6.4 气浮台半物理）；7 Conclusions。
- 与本文的关系：主要作背景/相关工作。
  - "safe proximity + soft docking to a tumbling target 本质是带边界值、避碰、软对接约束的最优控制问题"——可作为引言中"翻滚目标接近通常被表述为单次最优控制/轨迹规划问题"的代表引文之一（另配 Dong 2024/2025）。
  - soft docking 的定义句（终端相对位置与速度在容差内、径向速度小）可支撑我们捕获条件与轴向制动包络的动机。
  - "捕获点随翻滚目标运动"的分析（4.1）与我们"运动的捕获接口"表述一致。
- 写法可借鉴：引言"This process faces two key challenges: First,… Second,…"；方法优点"(1)…(2)…(3)…(4)…"。

### Dong 2022 = Dong, Luo, Limon, Acta Astronautica 200 (2022) 176–187
- 题目：A novel stable and safe model predictive control framework for autonomous rendezvous and docking with a tumbling target
- 结构：1 Intro；2 Problem formulation（2.1 控制目标与相对运动、2.2 目标姿态动力学（"body cone rolling on the space cone"，章动角）、2.3 约束建模：饱和、速度、避碰、时变 LOS 锥）；3 两阶段交会策略（第一阶段到时变安全区 Γ_s，附带跟踪路点；第二阶段在 LOS 锥内到端口）；4 控制器（4.1 第一阶段：近最优、确定收敛、终端约束保证递归可行；4.2 第二阶段：periodic MPC for tracking，参考为端口轨迹，规划与控制合一；Remark 1）；5 Case study（经典 MPC 对比、所提控制器；MATLAB + YALMIP + Mosek，报告 CPU 时间）。
- 与本文的关系：
  - 最接近的"翻滚目标多阶段 MPC"代表：先进时变 LOS 区，再跟踪端口周期轨迹。我们同样是"阶段飞行 → 入口 → 下降"，区别：我们有期限与燃料预算，进入时机由学习策略决定；他们的收敛结论是渐近收敛到周期轨迹，隐含"可以等待"。→ 引言/相关工作的核心对照句。
  - 时变 LOS 锥、"body cone rolls on space cone"的翻滚描述 → 我们 RWG 扫掠锥几何的文献依托（扫掠锥是恒速预报下端口方向绕自旋轴的圆锥）。
- 写法：两阶段图示（Fig. 3）＋分阶段控制器小节；计算平台说明句。

### Dong 2024 = Dong, Luo, Ni, Advances in Space Research 74 (2024) 2270–2287
- 题目：A robust model predictive control unified framework for autonomous rendezvous and docking with a tumbling target
- 结构：1 Intro（tube MPC 文献 Mammarella 2018、Buckner 2018、Oestreich 2023、Specht 2023→"future behavior of a tumbling target cannot be accurately predicted"）；2 Docking dynamics and problem formulation（2.1 动力学、2.2 工程约束（2.2.2 接近速度约束）、2.4 不确定性分析、2.5 名义模型、2.6 约束收紧）；3 EKF 估计目标惯量与状态 + TMPT（tube MPC for tracking）；4 Simulation（Scenario 1 不翻滚、Scenario 2 翻滚：无估计 vs 统一框架）；5 Conclusions。
- 关键事实：目标初始转速 (0.27, 1.96, 1.90) °/s（模约 2.7 °/s）、章动角 60°、惯量 diag(100,100,60)、初始位置 (8,15,16) m、keep-out 半径 1 m；3DOF 平移对接。
- 与本文的关系：
  - 同量级转速的翻滚目标先例（约 2.7 °/s）→ 我们 2.1–2.7 °/s 不是夸大工况；但其任务没有期限/燃料预算与"窗口稀缺"。
  - "翻滚目标未来行为难以准确预测、惯量未知" → 我们"执行层用各向同性惯量估计、恒速预报"的动机与诚实说明（我们不做在线估计，可在局限中对照其 EKF）。
  - 接近速度约束（2.2.2）→ 我们制动包络的相关先例。
- 写法：Scenario 1/2 对照组织；"Uncertainty analysis"小节。

### Dong 2025 = Z. Dong, L. Wang, Zhu, Liu, Guo, Guo（南理工），Acta Astronautica 235 (2025) 463–474
- 题目：Motion planning of free-flying space robots for tracking relative motion tumbling targets by sequential optimization via reinforcement learning
- 结构：1 Intro（NCSO、"only a limited time window available for capture"）；2 Problem statement；3 SOPT（3.1 基座重定向/稳定/主动视觉跟踪；3.2 序贯优化层：DDPG、训练细节）；4 Simulation（4.1 性能比较、4.2 强度测试（目标角速度扩到 [1,2] rad/s 级）、4.3 Discussion）；5 Conclusions。
- 与本文的关系：背景支撑为主——"非合作目标翻滚、捕获只有有限时间窗口"的表述来源；"把高维规划分解给多个智能体以降低 RL 复杂度"可作为"分层/分解降低学习难度"的旁证。不作方法对照（空间机械臂、9DOF）。

### Lu 2026 = W. Lu, Zhuang, Geng, Shen, Mu, Wu, Razoumny, Razoumny, Aerospace Science and Technology 168 (2026) 111238
- 题目：Safe reinforcement learning based agile satellite attitude control for non-cooperative target tracking with operation constraints and actuator faults
- 结构：1 Intro（贡献圆点 3 条）；2 Related work（2.1 非合作目标主动跟踪、2.2 RL 在航天控制）；3 Preliminary（3.1 任务分解为三阶段：故障辨识、姿态调整、目标跟踪；3.2 运作约束（禁区、角速度限制）；3.3 执行器故障）；4 Proposed method（4.1 故障辨识网络；4.2 安全 RL：4.2.1 CMDP、4.2.2 Worst-case SAC；4.3 伪代码"learning and deployment"）；5 Experiment（5.2 对比：SAC、SAC-Lag、SAC-PID、TD3-PID、Reward Penalty、WCSAC，表中报告每个案例的 reward / cost（违反量）/ time；5.3 消融：故障辨识方法）；6 Conclusion。
- 与本文的关系：
  - 航天中的安全 RL 代表（CMDP + 代价评价器）——相关工作中"安全 RL 通过约束 MDP 或代价评价器在学习层处理约束"一类；我们不同：约束由执行层（规划器 + MPC）硬处理、安全由真值判定，学习层只决定意图。
  - 实验中"同时报告回报与约束代价、按案例列表"的组织 → 我们 T2 中"结局构成 + 真实回报"并列。
  - "任务分解为阶段"也是我们分层的旁证（不夸大）。
- 注意：它声称"establishes safety guarantees"，我们不作此类主张。

### Zheng 2026 = Zheng, Chen, Zhou, Cheng（川大）, Acta Astronautica 249 (2026) 1277–1298
- 题目：Gaussian-disturbance-aware deep reinforcement learning for event-triggered MPC in satellite attitude reorientation
- 结构：1 Intro（贡献 (1)(2)(3)，每条"Unlike conventional… / Compared with existing…"）；2 System modeling and problem formulation（2.1 姿态模型、2.2 问题、2.3 DRL-ET-MPC 框架）；3 方法（3.1 TD3 触发策略、3.2 Mamba 时序、3.3 KF）；4 Simulation（4.1 设置 Table 1 全部超参与物理参数合表、4.2 六项评价指标、4.3/4.4 两个算例、4.5 消融 TD3/MTD3/KTD3/KMTD3、4.6 惩罚权重 λ 敏感性）；5 Conclusion。
- 与本文的关系：
  - RL 决定"何时让 MPC 重新优化"的事件触发 RL–MPC 接口——与我们"事件驱动宏步（16 s 或模式事件）"同属"RL 管时机、MPC 管执行"的范式，可在相关工作并列 [Ren 2026, Zheng 2026]：RL 调度 MPC 的时机/频率。
  - 消融命名"组件逐一加入"（TD3 → +M → +K → 完整）与我们 2×2 归因对应；"敏感性分析"小节形式可用于附录（预测时域/候选密度只评估敏感性）。
  - 评价指标先列出再比较（4.2 Evaluation metrics 小节）——我们实验章节可设"Metrics and statistics"小节。
- 写法：一张参数总表（训练 + 物理 + 求解器）→ 我们 T1。

### Tammam = Tammam, Aouf, Control Engineering Practice 156 (2025) 106213
- 题目：Hierarchical Deep Reinforcement Learning for cubesat guidance and control
- 结构：1 Intro；2 Literature review（2.2 RL、2.3 姿态测试平台）；3 RL formalism（DDPG/TD3）；4 Problem statement（4.1 姿态动力学、4.3 环境扰动）；5 Methodology（5.2 奖励塑形）；6 训练与仿真（6.2 HDRL：Hierarchical Actor–Critic，对比组合式与分布式 TD3）；7 实物平台（mock CubeSat、气浮台、机械臂交会模拟器）；8 Conclusion。
- 与本文的关系：背景/相关工作——"分层 RL 用于航天制导与控制"的先例：高层子目标、低层动作均由学习得到；我们是"高层学习、低层为固定的规划与 MPC"，区别在低层不学习且带硬约束。用一句话引用即可。

### Reiter 综述 = Reiter, Hoffmann, Reinhardt, Messerer, Baumgärtner, Sawant, Bödecker, Diehl, Gros, Annual Reviews in Control 61 (2026) 101045, DOI 10.1016/j.arcontrol.2026.101045（两份 PDF 内容相同："RL-MPC综述.pdf" 与 "Synthesis…pdf"）
- 题目：Synthesis of model predictive control and reinforcement learning: Survey and classification
- 结构：1 Intro（1.1 范围、1.2 相关综述、1.3 记号、1.4 概览）；2 Problem setting（MDP）；3 RL（DP、TD、PG、DQN/DDPG/PPO/SAC）；4 MPC（问题形式、约束、不确定性、数值方法、次优性）；5 Comparison of MPC and RL（状态空间、模型、随机性、模型失配、稳定性、约束 5.6、在线计算、离线工程、泛化、实践表现；Table 2/3）；6 Combination approaches（三大类：MPC as expert actor / MPC within the deployed policy / MPC as part of the critic；6.1 参数化 MPC 架构：integrated / algorithmic / hierarchical / parallel / parameterized）；7 MPC as expert actor；8 MPC within the deployed policy（8.1 aligned learning、8.2 closed-loop learning（8.2.2 MPC as part of the environment）、8.3 pre/post-processing（8.3.1 参考生成、8.3.2 后处理/安全滤波器）、8.4 RL 引导 MPC 优化）；9 MPC as a critic；10 理论（MPC–MDP 等价、折扣近似、10.3 闭环最优的状态转移模型）；11 软件；12 结论。
- 与本文的关系（核心定位来源）：
  - 定位句：本文属于 6.1.3 hierarchical architecture（NN 输出参数/参考给优化层，φ=φ_θ(s) 在求解前计算）+ 8.2.2 "MPC as part of the environment"（评价器评估的是给优化层的参数，而非 MPC 实际输出）。BACKGROUND 1.3"学习任务参考"一类即此。
  - E2 的关键引证：8.3.2 指出后处理滤波器在学习时不可见，因此策略不能期望闭环最优（"Since the trained policy is not aware of the filter, its performance cannot be expected to be closed-loop optimal"）→ 我们"认证在训练与执行中共用"的动机，引用 Section 8.3（facts/05 已写明）。
  - 5.6：安全 RL 的三类约束表述（硬约束、机会约束、累积代价）及"safe RL 要么保守要么违约"→ 支撑"把约束留给 MPC/规划器"的选择。
  - 5 节总论：MPC 源于多变量约束控制、RL 追求闭环性能，二者性质近乎正交 → 引言第二段的权威依据。
  - 10.3：模型预测精度 ≠ 闭环最优 → 可在讨论中说明"认证依赖执行层模型的预测，其价值由真值后果检验"。
- 写法：分类图（Fig. 2/3）的组织方式可借来画我们的定位小图（可选）；术语 "optimization layer"、"closed-loop optimality"。

### AC4MPC = Reiter, Ghezzi, Baumgärtner, Hoffmann, McAllister, Diehl, IEEE Transactions on Control Systems Technology（2025 在线，卷期页 TODO），DOI 10.1109/TCST.2025.3620521
- 题目：AC4MPC: Actor-Critic Reinforcement Learning for Guiding Model Predictive Control
- 结构：I Intro（MPC 优点与计算/终端代价难题；RL 低精度近全局 vs MPC 高精度局部、二者性质"nearly orthogonal"；贡献 1)–4)；C Outline）；II Preliminaries（Assumption 1）；III Actor and Critic Models for Nonlinear MPC（Algorithm 1；Assumption 2、Lemma 1、Theorem 1：闭环折扣代价不超过 RL actor 代价 + 误差项；Remarks 1–3；Corollary 1；Lemma 2、Theorem 2）；IV 多重打靶与 RTI（Algorithm 2/3；B Evaluation）；V Experiments（A NN 嵌入 MPC 的实现问题、B 玩具例、C 超车场景）；VI Conclusion。
- 与本文的关系：
  - README：评价器作终端代价、并行双解给"不差于"的界——两者在我们问题中都不可用（学到的终端代价实测有害；双解需 1.5 倍预算），这种不可行性可作对比讨论（只能作"开发期观察"，不当正式结果）。
  - 相关工作："critic 作为 MPC 终端价值 [Reiter AC4MPC]"是"MPC as part of the critic / 学价值"一类代表；我们的 RL 不改 MPC 代价，而是给执行层意图。
  - 写法借鉴（命题组织）：Assumption → Lemma → Theorem → Proof → Remark 的 TCST 写法；"These results do not require globally optimal solutions"这种对结论适用范围的明确限定句——我们命题后面写"该性质不蕴含…"可仿此。
  - 引言句："RL typically yields policies with low accuracy but close to global solutions, in contrast to MPC, which finds high-accuracy local solutions"——可意译引用作 RL/MPC 互补的表述。

### ecc26_rollout = Ghezzi, Reiter, Baumgärtner, Bemporad, Diehl, 2026 European Control Conference (ECC), Reykjavík, July 7–10, 2026（页码 TODO）
- 题目：Rollout Then Optimize: A One-Step Newton Refinement of Learned Policies for Nonlinear Model Predictive Control
- 内容：学习策略给名义轨迹（rollout），MPC 用一步 Newton（Riccati）在其邻域改进；给出近似误差界，一步 Newton 使次优性按策略逼近误差二次下降；四旋翼约束跟踪验证，约半计算量接近完全收敛 MPC。
- 与本文的关系：README："学习提供好的名义解，优化在其邻域内改进"→ 相关工作中"学习提供初值/名义解、优化精修"一类（与 AC4MPC 同组）。也可作为"认证在有限候选上做最小改动"的远亲类比——但我们的修正是离散候选上的执行机会认证，不是 Newton 步，不要类比过度。
- 写法：短会议文，Intro 的"MPC 与 RL 计算负担分布不同（在线 vs 离线）"表述可借。

### Leung & Kolmanovsky = Systems & Control Letters 185 (2024) 105735
- 题目：Feasibility governor for MPC with disturbance preview information
- 内容/结构：1 Intro；2 Problem setting；3 预览 MPC；4 Feasibility governor（FG 对给定参考命令做"最小修改"，使 PMPC 预测终端状态落在鲁棒控制不变集内，从而可用更短时域保持递归可行）；Proposition 1、Theorem 1–2（ISS、参考有限时间收敛）；自适应巡航算例。
- 与本文的关系：README：参考整形一类的相关工作。可在相关工作写"reference/feasibility governors minimally modify the commanded reference so that a short-horizon MPC remains feasible [Leung 2024, Tartaglione 2024]"；我们的认证也"对上层命令做最小修改"，但判据是执行层模型下的"关键窗口内触发模式事件且燃料够用"，目的在于任务机会而非可行性/不变性，且无保证 → 用于说清"相似之处与差别"。
- 注意：facts/00 规定"调节器/governor 是开发期放弃的方案，论文不提"——指我们自己开发期的"最小干预调节器"方案。引用参考调节器文献作相关工作是可以的，但不要把 ECC 写成 governor，也不要写我们试过 governor。（需和用户确认是否在相关工作提 governor 一类，见待定事项）
- 写法：Proposition → Proof 的紧凑写法（证明简短或"analogous to"）。

### Tartaglione = Tartaglione, Nicotra, Naldi, Garone, Automatica 166 (2024) 111696
- 题目：A constrained control framework for unmanned aerial vehicles based on Explicit Reference Governor
- 内容/结构：1 Intro；2 Preliminaries；3 问题陈述（Problem 1 Control / Problem 2 Navigation：把约束控制拆为"稳定"与"约束执行"两层，各层问题比原问题简单）；4 Control layer（4.1 外环；Proposition 1、Theorem 1 + 附录证明）；5 Navigation layer（ERG：动态安全裕度 + 导航场）；6 Numerical validation；7 Conclusions。
- 与本文的关系：同上（参考整形一类）。另可借"把问题拆成两层后每层问题更简单"的论证语气，以及"Problem 1 / Problem 2"分别陈述两层目标的写法——我们可把"高层决策问题"与"执行层问题"分开陈述（但执行层不作贡献）。

### Wu 2024 = Q. Wu, K. Liu, L. Chen, "Model predictive control-based value estimation for efficient reinforcement learning"，IEEE 期刊 Feature Article: Deep Reinforcement Learning（PDF 为 arXiv:2310.16646v2 版本，期刊名/卷期/DOI 均 TODO，需核对正式出版信息）
- 内容：用数据驱动的环境模型做多步预测（MPC 式）来估计价值函数、优化策略，提高样本效率；经典基准与无人机动态避障验证。
- 与本文的关系：README：MPC 价值估计一类相关工作。我们与之相反：执行层预测"只用于认证动作，不生成价值目标、不改 SAC 损失"（facts/05 §5.4）——相关工作中写"MPC-based value expansion [Wu 2024] uses model rollouts to form value targets；ECC does not"。开发期反事实价值标签失败（DEAD_ENDS）可作讨论材料。

### Zeng 2026 = Zeng, Ni, Li, Zhang, Liu, Hu, Sun, Leng, Neurocomputing 701 (2026) 134631
- 题目：Physics-informed residual reinforcement learning via expert prior knowledge for safe and efficient autonomous merging
- 结构：1 Intro（贡献 1.–4.）；2 相关工作；3 Methodology（MPC 先验 + 有界残差 RL；风险自适应混合执行 HEM（在线调节残差权限 + MPC 回退）；解耦双评价器（任务回报与安全代价分开）；课程学习）；4 实验（4.3 对比：5 个基线 DQN/PPO/SAC/PPO-L/…，每法 5 种子 × 1000 回合，低/中/高三档交通密度；4.3.1 训练曲线均值±标准差带；在线开销：每次策略更新 MPC 先验 11.60 ms + 推理 0.42 ms，最大 17.46 ms ≪ 0.5 s 决策周期；4.3.3 观测噪声鲁棒性；4.4 消融：full / w/o HEM / w/o DC / w/o EP / w/o DC and EP / MPC，表中 SR/CR/FR 三档）；5 结论。
- 与本文的关系：
  - 实验组织模板（EXPERIMENTS §0 已对应）："w/o 部件"消融命名 → 我们 ECC w/o certification / execution feedback / rotation guidance；在线开销"每步耗时 vs 决策周期"→ 我们"认证耗时 vs 16 s 决策周期"；种子带的训练曲线 → F1。
  - 方法上：执行层在线调节学习动作的"权限"（HEM）与我们"认证在执行时修正意图"形式相近；区别：它是风险启发式、训练外的软安全；我们的认证判据是执行层模型下的模式事件 + 关键窗口 + 燃料，训练与执行共用。解耦双评价器与我们"评价器读执行反馈"都是"评价器结构"上的改动，可并列一句。
- 写法：消融逐项解释"(2) without HEM, which retains… but disables…, thereby evaluating…"——这一句式非常适合我们写三个消融的定义。

### Lu 2025 = P. Lu, S. Zhang, F. Tan, F. Zhang, Y. Feng, B. Hu, Engineering Applications of Artificial Intelligence 161 (2025) 112108
- 题目：An uncertainty-aware safe-evolving reinforcement learning algorithm for decision-making and control in highway autonomous driving
- 内容：内部稳定（策略更新高概率单调改进）+ 外部稳定（以基准策略为参照不差于它）+ 动作投影与混合学习（对学习策略做"最小修改"以保证安全，并同时从安全动作与原动作学习）；高速公路仿真。实验：屏蔽开/关、训练期累计不安全事件、介入案例。
- 与本文的关系：
  - 相关工作"安全层对学习动作做最小修改（projection / shield）"一类；它显式讨论了"学习时用投影后动作还是原动作"的混合学习——与我们 E2"经验池存提议的 a、回报来自认证后执行"直接相关，可作对照：我们选择存提议动作，使评价器学习"提出 a 时系统真实发生什么"。
  - 实验组织（EXPERIMENTS §0）：同一模型执行时认证开/关、训练期累计超时与不安全、F5 介入案例。
- 不借：其"provable safety"主张。

### Duan & Zou = Automation in Construction 174 (2025) 106130
- 题目：Safety-constrained Deep Reinforcement Learning control for human–robot collaboration in construction
- 内容：神经网络策略作"大脑"负责复杂决策，参考调节器层作"脊髓"负责快速安全响应；对力矩、关节角、速度、距离给出安全保证分析；0% 碰撞率。
- 与本文的关系：相关工作中"学习决策层 + 下层安全/约束执行层"的跨领域例子（一句话带过）；"大脑–脊髓"类比可不用。

### Adjei 2026 = Adjei, Xie, Li, Ren, Spitas, Aerospace Science and Technology 178 (2026) 113226
- 题目：Hierarchical model predictive control-guided reinforcement learning for energy management of hybrid-electric distributed propulsion aircraft
- 内容/结构：2 飞机与推进建模；3 问题表述（3.1 系统动力学、3.3 分层 MPC 层、3.5 优先经验回放）；4 结果（4.2 混合 MPC-SAC 能量管理性能）。架构：起飞爬升等安全关键阶段由 MPC 主导，巡航由 SAC 做长时域能量最优分配，按飞行阶段连续加权融合，SAC 退化时自动回退到 MPC。
- 与本文的关系：航空领域"SAC 负责长时域最优、MPC 负责约束与安全关键阶段"的分层先例 → 引言/相关工作中"long-horizon optimality vs constraint-aware authority"的平衡表述可引；同为 SAC + MPC 分层，区别：它按阶段混合权限，我们的耦合是执行反馈与执行机会认证。
- 贡献写法：圆点 + 粗体小标题（"• Hierarchical MPC–RL architecture: …"）。

### 死区迟滞（用户第一篇，王昱等，中文期刊稿 2026）——只作组织形式与中文行文参考，不引用
- 结构：0 引言（背景：巨型星座、碎片风险→编队优势与任务→难点：J2 漂移、碎片、燃料有限→现有研究两类：优化规划 / 反馈跟踪，各举 2–3 例并指出不足→DRL 在航天控制的应用按"轨道机动 / 安全避障 / 编队与燃料"分组综述→"仍存在两个主要问题：其一…其二…"→"针对上述两类核心问题，本文提出…主要贡献包括 1) 2) 3)"）；1 问题描述（1.1 任务描述、1.2 J2 相对动力学、1.3 强化学习问题建模）；2 分层分布式控制框架（总体图 + 2.2 底层轨迹执行、2.3 死区迟滞机制、2.4 奖励函数设计、2.5 稳定性分析）；3 仿真实验与分析（3.1 场景与训练配置、3.2 控制性能和收敛性对比、3.3 消融实验与机制验证、3.4 与传统 MPC 对比、3.5 策略泛化性验证）；4 结论（"主要结论如下"分条）。
- 可借鉴：引言的"两个主要问题 → 对应机制 → 贡献"三段式，与我们 BACKGROUND 的"理论问题 / 场景问题 → ECC / RWG"一一对应；实验章节"性能对比 → 消融与机制验证 → 与传统 MPC 对比 → 边界/泛化"的顺序可直接用；"稳定性分析"小节的位置（方法末尾）对应我们的性质小节。
- 同一课题组思路的连续性：策略输出期望边界条件、底层插值 + PD 跟踪、把底层实际推力反馈给奖励——与本文"学习给阶段意图、执行层跟踪、执行信息回馈学习"一脉相承（只作内部理解，不写进论文）。

## 4 全文架构示范论文：已定

主参考为 Ren 2026（北航编队，TAES），逐节对照见文首"★ 主参考模板"。局部参照：建模与场景——Yang 2025、Tian 2024；命题组织——AC4MPC；实验——Zeng 2026（消融、开销）、Ramezani 2026（指标、燃料）；引言组织——死区迟滞稿。

## 5 待定事项（需要你决定）

1. **参考调节器类文献是否在相关工作出现。** facts/00 写"调节器 / governor 是开发期放弃的方案，论文不提"。我的理解是"不提我们自己试过的调节器方案"，外部文献（Leung 2024、Tartaglione 2024、Duan 2025）作为"对上层命令做最小修改"的一类相关工作仍可引用，并说明 ECC 的判据与目的不同。若你希望完全回避这一类，我就只保留 Lu 2025 / Zeng 2026 作为"执行侧修改学习动作"的对照。
2. **Wu 2024 的正式出版信息。** PDF 是 arXiv v2（版头为 IEEE 期刊 Feature Article），期刊名、卷期、DOI 需要你确认或给出正式版本，否则 BibTeX 标 TODO。
3. **AC4MPC 与 ecc26 的卷期页码**：前者为 TCST 在线优先版、后者为 ECC 2026 会议，页码标 TODO。
4. **死区迟滞稿**：按 README 不引用；如导师希望引用（同一课题组前序工作），需要你明确。
