# 论文资料与写作取材指南

仓库里的文档大多是研发和工程记录，写法细、术语多。写论文时只从中**取事实**，表达按学术论文的简洁写法另写：不出现配置名、文件路径、提交号、工作包编号，也不出现"冻结""逐位等价"这类工程说法，不展开实现细节与研发过程。名称一律用 `BASELINE.md` 第 0 节的论文名（ECC、RWG、SAC–MPC、TD3–MPC、ECC w/o 认证 / 执行反馈 / 旋转引导）。

## 一、主要依据（论文内容来自这里）

| 文件 | 用于 | 注意 |
|---|---|---|
| `METHOD.md` 第 1–3 节、附录 A | 方法章节：符号、多速率执行系统、RWG、ECC、命题、约束与奖励、算法；"不主张"清单 | 附录 B（实现对照）是工程信息，不写进论文 |
| `EXPERIMENTS.md` 第 1–3、5–7 节 | 实验章节：研究问题、任务条件、训练组与对照、终点与统计、稀缺解释轴、图表规划 | 第 0 节是同类论文的组织方式参考；第 4 节运行协议、第 8 节数据接口只取必要事实（种子、步数、样本量） |
| `BASELINE.md` 第 0–3 节 | 命名、背景、两个问题（理论问题 → ECC，场景问题 → RWG）、贡献 | 第 4 节以后是统筹计划，过时部分以 `EXPERIMENTS.md` 为准 |
| `References/` 与 `References/README.md` | 相关工作与写作规范参考 | README 写明每篇用来做什么、不要声称什么 |
| 协作分支汇报稿 `wang24.pdf` | 背景与问题的表述要和它一致 | — |

## 二、只取事实、不取表达

| 文件 | 可用的事实 |
|---|---|
| `FACT_MAP.md`、`AUDIT.md` | 论文表述与实现是否一致；纠正过的旧说法（例如远场平移不是 MPC、训练不能写"逐位可复现"） |
| `docs/HISTORY.md`、`docs/DEAD_ENDS.md` | 为何选这个方法、试过哪些方案没成功（讨论或附录用，不写研发流水账） |
| `evidence/README.md` | 开发期结果的位置（只能作"开发期"出现，不并入正式结果） |
| `docs/archive/method_development/C1_THEORY.md` | 扫掠锥几何的推导细节（RWG 的理论支撑） |
| 协作分支 `archive/RESEARCH_EVOLUTION_AND_COUPLING_DESIGN_20261012.md` | 研究动机的来龙去脉 |

## 三、写作时不看（工程记录或已过时，容易误导）

- `CLAUDE.md`、`README.md` 的代码表、`docs/REPRODUCE.md`、注册表 JSON、协作分支的执行书与交付；
- `docs/archive/method_development/` 中除 C1_THEORY 以外的规格：其中的 EACH、U1、C1、U2、`ECC_deploy`、调节器等都是开发期名称或已放弃的方案；
- 各证据目录的原始文件。

## 正式结果

正式实验交付后在 `evidence/FORMAL/`：`FORMAL_READOUT.json`（主终点、次终点、归因、机制统计）、`EPISODES.csv`（逐回合）、`FIGURES/`（F1–F7、T1–T3、RWG 几何图）。交付前论文中的结果位置用占位符，不填估计值。

## 版本

- 方法行为基准：科学 `e0a61753`；
- 正式实验代码：科学 `e8474e21`（标签 `archive/formal-code-20261114`）。
