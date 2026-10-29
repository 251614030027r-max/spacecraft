# Git 资产总表（2026-10-18）

**原则：**
- 历史从不重写；
- 只保留三条分支；
- 冻结状态是科学分支上的提交，必要时由下层另打 `archive/*` 标签；
- 证据只以单个文件加入 `evidence/<实验ID>/`，并附 SHA256SUMS。

## 1. 分支（只保留三条）

| 分支 | 作用 | 说明 |
|---|---|---|
| `claude/sac-mpc-coupling-design-ns7g6i` | 唯一的科学分支：代码、测试、证据、当前文档 | 2026-10-18 快进到 T1-O 最新提交，见第 2 节 |
| `collab/spacecraft` | 协作：只改 `docs/collaboration/`（执行单、交接、审查、总账、项目状态） | 该分支上的其他文件是旧副本，**不得从它运行代码** |
| `main` | 实验全部结束后统一并入 | 停在 `619e491`（2026-09-04，"Add files via upload"），有 30 个早期独有提交，不动 |

## 2. 已并入并删除的实验分支（2026-10-18）

三条分支是一条直线：`1d432c8 ⊂ 166244b ⊂ 672ba79 ⊂ d3902a9`。科学分支快进到 `d3902a9` 之后，它们的全部提交都在科学分支上可达，因此删除不会丢失任何提交。

| 已删除分支 | 头提交 | 内容 | 状态 |
|---|---|---|---|
| `experiment/t1_l_option_relative_20261015` | `166244b` | T1-L 三臂：B2 / N0 / NΔ，`evidence/T1_L_PRECHECK` | 训练完成（交付 collab `68680eb`） |
| `experiment/t1_l_s_tcpf_20261016` | `672ba79` | T1-L-S：区间 Bellman、训练器 A / B，`evidence/T1_L_S_PRECHECK` | 正式训练取消 |
| `experiment/t1_o_tcpf_20261017` | `d3902a9` | T1-O：档案、物理筛选、V2 Gate A–E 工具与证据 | 停于 Gate C |

**远端删除：** 由下层在本机执行（沙盒代理不允许删除远端分支），2026-10-18 已执行并核验只剩三条分支：

```powershell
git push origin --delete experiment/t1_l_option_relative_20261015 experiment/t1_l_s_tcpf_20261016 experiment/t1_o_tcpf_20261017
```

**删除前核对：** 下面三条命令都必须输出 `ok`：

```powershell
git fetch origin
foreach ($b in "t1_l_option_relative_20261015","t1_l_s_tcpf_20261016","t1_o_tcpf_20261017") {
  git merge-base --is-ancestor origin/experiment/$b origin/claude/sac-mpc-coupling-design-ns7g6i; if ($LASTEXITCODE -eq 0) {"ok $b"} else {"NOT MERGED $b"}
}
```

## 3. 标签（全部保留，不新增）

| 标签 | 指向 | 内容 |
|---|---|---|
| `archive/pre-cleanup-20261010` | `813e27f` | 2026-10-10 整理前的完整科学分支（旧路径、旧文档、`logs/` 索引） |
| `archive/review/v3e-60k-value-r2-20260930` | `dc6682d` | E01 证据 |
| `archive/review/v3e-stage-c-20261002` | `b383cd2` | E04 证据 |
| `archive/review/stopping-20261006-audit` | `6c5cb97` | E06 证据 |
| `archive/review/stopping-value-check-20261006` | `2d8a581` | E09 证据 |
| `archive/v2-evaluation-20260924` | `736edbc` | 旧开发分支 |
| `archive/v3-t0b-20260925` | `f50eb78` | 旧开发分支 |
| `v3e-60k-value-r2-20260930` | `dc6682d` | 与 E01 同一提交（早期标签） |

## 4. 科学分支上的证据目录（`evidence/`）

| 目录 | 内容 |
|---|---|
| `E01_v3e_60k` … `E08_label_check` | F01 线（V3e、B / C 阶段、停止头正式审查、标签检查） |
| `historical` | 整理前的历史证据 |
| `T1_R0`、`T1_R1_pass1`、`T1_R1_pass2` | T1 R0 / R1 规则行（开发块） |
| `T1_M1_validationB` | M1 反事实验收 B（门控阈值来源） |
| `T1_dev_rule_295000`、`T1_dev_pure_295000` | T1 开发块规则基准与 Pure MPC 行 |
| `T1_pretrain_checks` | 第一轮训练前核查 |
| `T1_V2_S0` | 接口 v2、M1 种子 0（32k 平台期停止） |
| `T1_ROOTCAUSE_S1` | 根因诊断第一阶段 |
| `T1_PLAN_EXEC_GAP_R1` | 规划–执行差距（探索性） |
| `T1_BETA_B0`、`T1_BETA_B1` | β 参考整形（B1 NO-GO） |
| `T1_S1_SCREEN` | 场景筛选 S1 |
| `T1_FIX_PRECHECK`、`T1_ENTRY_SPEED_PRECHECK` | v4（入口速度、critic 修复）技术准入 |
| `T1_L_PRECHECK` | T1-L 三臂技术准入 |
| `T1_L_S_PRECHECK` | T1-L-S 技术准入（训练已取消） |
| `T1O_SCREEN` | T1-O P1 / P2 物理筛选（两档均 FAIL，原判定保留） |
| `T1O_V2` | V2 诊断 `DIAGNOSE.md` 与 Gate A（PASS，λ_f = 20） |

**不在科学分支上的大包**（collab 分支 `docs/collaboration/lower/handoffs/`）：

| 包 | 内容 | collab 提交 |
|---|---|---|
| `t1_v4_s0_20261015` | v4 交付 | `2242f1b` |
| `t1_l_option_20261015` | T1-L 40k 交付 | `68680eb` |
| `t1o_v2_gates_20261017` | V2 Gate B / C 交付 | `be5727d` |

对应的上层审查都在 collab `docs/collaboration/reviews/`。

## 5. 本机产物（不在 git 中，下层机器）

| 位置 | 内容 |
|---|---|
2026-10-18 下层完成本机资产归并：只保留一个科学工程，旧克隆保全后退役。

| 位置 | 内容 |
|---|---|
| `D:/py/DRL2` | 唯一科学工程，检出 `4795910`（不是权威科学头 `7c2d513`；下一轮按新冻结 SHA 同步）；模型与日志在 `logs/`、评估在 `eval/`（T1 v4 在 `logs/t1/`，T1-L 在 `logs/t1_l_option_20261015/`，T1-O V2 在 `logs/t1o_v2/`） |
| `D:/py/DRL2/local_artifacts/资产归并_20261018/` | 版本保全库：源码快照、Git 元数据、定位索引（不是另一套运行工程） |
| `D:/py/DRL2_T1`、`DRL2_T1_L`、`DRL2_T1_L_S`、`DRL2_T1_O` 等 | 已保全后退役，不要重建 |
| `D:/py/DRL` | 第一篇论文的独立工程，不动 |
| `C:/Users/35884/Documents/Spacecraft/过程文件/` | 过程记录；归并映射在 `资产归并_20261018/记录/`；交付在 `Spacecraft/上层交付/` |

本地资产地图：collab `lower/handoffs/cleanup_inventory_20261018/LOCAL_ASSET_MAP.md`（`4f3d305`）；2026-10-10 版在 `cleanup_inventory_20261010/`。历史证据里的旧路径不是现址，按映射查找并核验 SHA。模型 zip 一律只记 SHA，不上传。

## 6. 引用过的关键科学提交（都在科学分支上可达）

| 提交 | 含义 |
|---|---|
| `1d432c8` | T1 v4（接口 v4、G★ v0.6、critic 修复） |
| `166244b` | T1-L 三臂冻结 |
| `672ba79` | T1-L-S 冻结（训练取消） |
| `d7c5c1c` | T1-O 物理筛选预注册 |
| `e8c0f08` | T1-O 物理筛选证据 |
| `89b779a` | T1-O V2 预注册（Gate A–E 判据、开局） |
| `f315440` | V2 Gate A 证据 |
| `d3902a9` | V2 本机门控执行用提交 |


## 2026-10-29 下层轻量核验：B1/B2（不删除证据）

按collab执行书6bb1d66，核验科学基线`8761ff7dddb924a9ceb41e39287c4303a45b7f5b`的evidence/模型及ZIP，共40项；所有原ZIP完整保留，作为既有二进制例外登记，不回写历史、不补删。目录就近SHA清单及下列总清单记录路径、Git blob、大小与SHA256。以后新交付模型只登记SHA，不再加入模型ZIP。

以后证据内的过程资料目录统一使用`脚本/`、`记录/`、`输入/`三个名称，不再新增其他中文过程目录名；本次不改动既有目录与各轮自包含证据/复用SAC副本。

C1约0.5GB旧训练日志与C2协作旧交付ZIP均未获逐项确认，本次均不删除。`archive/science-pre-cleanup-20261029`是C1的前置步骤，本次B不删科学资产，未创建该可选标签。

ZIP总清单：[`MODEL_ZIP_EXCEPTIONS_20261029.json`](MODEL_ZIP_EXCEPTIONS_20261029.json)。本节仅B1/B2核验；T2提交证据总表由上层另补。
