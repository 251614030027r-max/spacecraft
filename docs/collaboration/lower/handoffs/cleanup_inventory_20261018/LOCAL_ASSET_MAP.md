# 本地资产地图（2026-10-18）

状态：本地归并完成，科学实验停止；本清单供科学分支 docs/archive/GIT_ASSETS_20261018.md 第5节核对。原始模型、日志、交付ZIP内容保持不变；位置变化通过本清单和逐文件CSV解析，不改写原始manifest或交付证据。

## 当前入口

- 唯一科学工程 D:/py/DRL2；本机HEAD仍4795910102cf9fbf1dfec74b77e9449623a7b51e，未切换源码。fetch后的权威科学远端为7c2d5138aeccc976349a690f49db45e4680cfb4a；CURRENT中的scientific_commit表示权威版本，并非本机HEAD。
- 模型/日志现址：D:/py/DRL2/logs；评估：D:/py/DRL2/eval。源码/Git元数据及独有附属文件：D:/py/DRL2/local_artifacts/资产归并_20261018。
- 原主工程34743个环境/IDE/日志文件已有保护指纹；.venv、PyCharm设置和第一篇论文D:/py/DRL保留。不同版本源码只存快照，没有混入主源码。
- 精确模型定位用MODEL_AND_ARCHIVE_LOCATION_INDEX.csv；全部8867条源资产定位用PAYLOAD_LOCATION_MAP.csv（路径、SHA、字节、复制/去重）。登记表、MODEL_SHA256和交付回执引用的文件按此追溯，不因旧路径失效判定资产缺失。
- 最新T1O_V2_GATES_LOCAL.zip SHA256：8e039d0445d70f905134c7831c548d2c05a57918ef83a68a479701de25724c82；原包未改。

## 退役目录（先归档核验再移除）

| 原绝对路径 | 内容/实验ID | 现址与动作 |
|---|---|---|
| `D:\py\DRL2_T1` | T1 R0/R1、v2、根因诊断、v4；原跟踪源码及Git历史、独有模型/日志 | 科学数据归入主工程logs/eval；源码与元数据快照见下表；旧目录退役，精确文件见CSV |
| `C:\Users\35884\Documents\Spacecraft\v3-interface-worktree` | V3接口历史；原跟踪源码及Git历史、独有模型/日志 | 科学数据归入主工程logs/eval；源码与元数据快照见下表；旧目录退役，精确文件见CSV |
| `C:\Users\35884\Documents\Spacecraft\v3-t0b-worktree` | V3-T0B；原跟踪源码及Git历史、独有模型/日志 | 科学数据归入主工程logs/eval；源码与元数据快照见下表；旧目录退役，精确文件见CSV |
| `C:\Users\35884\Documents\Spacecraft\过程文件\远端交付\审查工作树` | E01历史审查；原跟踪源码及Git历史、独有模型/日志 | 科学数据归入主工程logs/eval；源码与元数据快照见下表；旧目录退役，精确文件见CSV |
| `C:\Users\35884\Documents\Spacecraft\过程文件\阶段B2\扫描工作树_a714c61` | 阶段B2扫描；原跟踪源码及Git历史、独有模型/日志 | 科学数据归入主工程logs/eval；源码与元数据快照见下表；旧目录退役，精确文件见CSV |
| `D:\py\DRL2_T1_L` | T1-L；原跟踪源码及Git历史、独有模型/日志 | 科学数据归入主工程logs/eval；源码与元数据快照见下表；旧目录退役，精确文件见CSV |
| `D:\py\DRL2_T1_L_S` | T1-L-S（正式长训取消）；原跟踪源码及Git历史、独有模型/日志 | 科学数据归入主工程logs/eval；源码与元数据快照见下表；旧目录退役，精确文件见CSV |
| `D:\py\DRL2_T1_O` | T1-O V2；原跟踪源码及Git历史、独有模型/日志 | 科学数据归入主工程logs/eval；源码与元数据快照见下表；旧目录退役，精确文件见CSV |
| `D:\py\spacecraft` | F01临时核验；原跟踪源码及Git历史、独有模型/日志 | 科学数据归入主工程logs/eval；源码与元数据快照见下表；旧目录退役，精确文件见CSV |
| `D:\py\DRL2_T1_L_S_incomplete_20261016` | T1-L-S（正式长训取消）；原跟踪源码及Git历史、独有模型/日志 | 科学数据归入主工程logs/eval；源码与元数据快照见下表；旧目录退役，精确文件见CSV |
| `D:/py/DRL2_v3e` | V3e主目录junction | 仅删除链接，目标D:/py/DRL2未递归删除 |

## Spacecraft目录移动（旧前缀→新前缀）

| 原路径 | 新路径 | 内容/实验ID |
|---|---|---|
| `C:\Users\35884\Documents\Spacecraft\过程文件\T1v4首轮配对训练_20261015\测试临时` | `C:\Users\35884\Documents\Spacecraft\过程文件\资产归并_20261018\隔离缓存\历史T1v4_测试临时残留\测试临时` | T1-v4测试临时缓存；优先于下表普通前缀匹配 |
| `C:\Users\35884\Documents\Spacecraft\过程文件\编排` | `C:\Users\35884\Documents\Spacecraft\过程文件\历史\编排` | 编排（历史主题，具体ID见原记录）；原过程记录/脚本/证据，历史只读 |
| `C:\Users\35884\Documents\Spacecraft\过程文件\工程整理` | `C:\Users\35884\Documents\Spacecraft\过程文件\历史\工程整理` | 工程整理（历史主题，具体ID见原记录）；原过程记录/脚本/证据，历史只读 |
| `C:\Users\35884\Documents\Spacecraft\过程文件\汇报图表_20261009` | `C:\Users\35884\Documents\Spacecraft\过程文件\历史\汇报图表_20261009` | 汇报图表_20261009（历史主题，具体ID见原记录）；原过程记录/脚本/证据，历史只读 |
| `C:\Users\35884\Documents\Spacecraft\过程文件\价值规则检验` | `C:\Users\35884\Documents\Spacecraft\过程文件\历史\价值规则检验` | 价值规则检验（历史主题，具体ID见原记录）；原过程记录/脚本/证据，历史只读 |
| `C:\Users\35884\Documents\Spacecraft\过程文件\阶段基准图_20261009` | `C:\Users\35884\Documents\Spacecraft\过程文件\历史\阶段基准图_20261009` | 阶段基准图_20261009（历史主题，具体ID见原记录）；原过程记录/脚本/证据，历史只读 |
| `C:\Users\35884\Documents\Spacecraft\过程文件\阶段B1` | `C:\Users\35884\Documents\Spacecraft\过程文件\历史\阶段B1` | 阶段B1（历史主题，具体ID见原记录）；原过程记录/脚本/证据，历史只读 |
| `C:\Users\35884\Documents\Spacecraft\过程文件\阶段B2` | `C:\Users\35884\Documents\Spacecraft\过程文件\历史\阶段B2` | 阶段B2（历史主题，具体ID见原记录）；原过程记录/脚本/证据，历史只读 |
| `C:\Users\35884\Documents\Spacecraft\过程文件\阶段C` | `C:\Users\35884\Documents\Spacecraft\过程文件\历史\阶段C` | 阶段C（历史主题，具体ID见原记录）；原过程记录/脚本/证据，历史只读 |
| `C:\Users\35884\Documents\Spacecraft\过程文件\空间问题分层控制框架` | `C:\Users\35884\Documents\Spacecraft\过程文件\历史\空间问题分层控制框架` | 空间问题分层控制框架（历史主题，具体ID见原记录）；原过程记录/脚本/证据，历史只读 |
| `C:\Users\35884\Documents\Spacecraft\过程文件\历史过程` | `C:\Users\35884\Documents\Spacecraft\过程文件\历史\历史过程` | 历史过程（历史主题，具体ID见原记录）；原过程记录/脚本/证据，历史只读 |
| `C:\Users\35884\Documents\Spacecraft\过程文件\停止头` | `C:\Users\35884\Documents\Spacecraft\过程文件\历史\停止头` | 停止头（历史主题，具体ID见原记录）；原过程记录/脚本/证据，历史只读 |
| `C:\Users\35884\Documents\Spacecraft\过程文件\远端交付` | `C:\Users\35884\Documents\Spacecraft\过程文件\历史\远端交付` | 远端交付（历史主题，具体ID见原记录）；原过程记录/脚本/证据，历史只读 |
| `C:\Users\35884\Documents\Spacecraft\过程文件\整理_20261010` | `C:\Users\35884\Documents\Spacecraft\过程文件\历史\整理_20261010` | 整理_20261010（历史主题，具体ID见原记录）；原过程记录/脚本/证据，历史只读 |
| `C:\Users\35884\Documents\Spacecraft\过程文件\最终主线` | `C:\Users\35884\Documents\Spacecraft\过程文件\历史\最终主线` | 最终主线（历史主题，具体ID见原记录）；原过程记录/脚本/证据，历史只读 |
| `C:\Users\35884\Documents\Spacecraft\过程文件\最终主线重训` | `C:\Users\35884\Documents\Spacecraft\过程文件\历史\最终主线重训` | 最终主线重训（历史主题，具体ID见原记录）；原过程记录/脚本/证据，历史只读 |
| `C:\Users\35884\Documents\Spacecraft\过程文件\DRL与T1训练对照_20261015` | `C:\Users\35884\Documents\Spacecraft\过程文件\历史\DRL与T1训练对照_20261015` | DRL与T1训练对照_20261015（历史主题，具体ID见原记录）；原过程记录/脚本/证据，历史只读 |
| `C:\Users\35884\Documents\Spacecraft\过程文件\T1-L-S区间目标支线_20261016` | `C:\Users\35884\Documents\Spacecraft\过程文件\历史\T1-L-S区间目标支线_20261016` | T1-L-S；原过程记录/脚本/证据，历史只读 |
| `C:\Users\35884\Documents\Spacecraft\过程文件\T1-L三臂支线_20261015` | `C:\Users\35884\Documents\Spacecraft\过程文件\历史\T1-L三臂支线_20261015` | T1-L；原过程记录/脚本/证据，历史只读 |
| `C:\Users\35884\Documents\Spacecraft\过程文件\T1根因诊断_20261013` | `C:\Users\35884\Documents\Spacecraft\过程文件\历史\T1根因诊断_20261013` | T1_ROOTCAUSE_STAGE1；原过程记录/脚本/证据，历史只读 |
| `C:\Users\35884\Documents\Spacecraft\过程文件\T1首轮_20261012` | `C:\Users\35884\Documents\Spacecraft\过程文件\历史\T1首轮_20261012` | T1首轮_20261012（历史主题，具体ID见原记录）；原过程记录/脚本/证据，历史只读 |
| `C:\Users\35884\Documents\Spacecraft\过程文件\T1O_V2本机门控_20261017` | `C:\Users\35884\Documents\Spacecraft\过程文件\历史\T1O_V2本机门控_20261017` | T1-O V2；原过程记录/脚本/证据，历史只读 |
| `C:\Users\35884\Documents\Spacecraft\过程文件\T1v2种子0验证_20261013` | `C:\Users\35884\Documents\Spacecraft\过程文件\历史\T1v2种子0验证_20261013` | T1-v2；原过程记录/脚本/证据，历史只读 |
| `C:\Users\35884\Documents\Spacecraft\过程文件\T1v4首轮配对训练_20261015` | `C:\Users\35884\Documents\Spacecraft\过程文件\历史\T1v4首轮配对训练_20261015` | T1-v4；原过程记录/脚本/证据，历史只读 |
| `C:\Users\35884\Documents\Spacecraft\过程文件\V3e60k` | `C:\Users\35884\Documents\Spacecraft\过程文件\历史\V3e60k` | V3e60k（历史主题，具体ID见原记录）；原过程记录/脚本/证据，历史只读 |
| `C:\Users\35884\Documents\Spacecraft\过程文件\V3e价值第二轮` | `C:\Users\35884\Documents\Spacecraft\过程文件\历史\V3e价值第二轮` | V3e价值第二轮（历史主题，具体ID见原记录）；原过程记录/脚本/证据，历史只读 |
| `C:\Users\35884\Documents\Spacecraft\分析.md` | `C:\Users\35884\Documents\Spacecraft\历史记录\研究讨论_20261010\分析.md` | 分析.md（历史主题，具体ID见原记录）；原过程记录/脚本/证据，历史只读 |
| `C:\Users\35884\Documents\Spacecraft\清单.md` | `C:\Users\35884\Documents\Spacecraft\历史记录\研究讨论_20261010\清单.md` | 清单.md（历史主题，具体ID见原记录）；原过程记录/脚本/证据，历史只读 |
| `C:\Users\35884\Documents\Spacecraft\v3-interface-worktree/.pytest_cache` | `C:\Users\35884\Documents\Spacecraft\过程文件\资产归并_20261018\隔离缓存\v3-interface-worktree_缓存残留/.pytest_cache` | 旧测试缓存ACL拒绝访问，原样隔离，不是活跃仓库 |
| `C:\Users\35884\Documents\Spacecraft\v3-t0b-worktree/.pytest_cache` | `C:\Users\35884\Documents\Spacecraft\过程文件\资产归并_20261018\隔离缓存\v3-t0b-worktree_缓存残留/.pytest_cache` | 旧测试缓存ACL拒绝访问，原样隔离，不是活跃仓库 |

移动核验范围：可读取文件逐项SHA核对；权限拒绝的pytest缓存通过同卷父目录改名保留，不能声称其内容已读取或SHA验证。SPACECRAFT_MOVE_RECEIPT.json列出限制；T1-v4测试缓存使用上表更长前缀覆盖。可再生缓存随已保全副本退役，独有科学资产不删。

## 冻结源码与Git历史归档（仅本机）

| 本机快照路径 | SHA256 | 文件数 |
|---|---|---|
| `D:\py\DRL2\local_artifacts\资产归并_20261018\版本快照\DRL2_git_metadata.zip` | `44e38ad4b4b876cdb2580b8f985f21fb2bc08a763fc0775e55b6b7111470b0f4` | 100 |
| `D:\py\DRL2\local_artifacts\资产归并_20261018\版本快照\DRL2_checkout.zip` | `6d578133147c82b3461069d02365b2cc97a0de2aece586ed694c918bb189f31d` | 411 |
| `D:\py\DRL2\local_artifacts\资产归并_20261018\版本快照\DRL2_T1_checkout.zip` | `2f2d4e1f489cafce8e2e21a299e2b59cedd3abdab25d5c9524e84f67fb115e57` | 665 |
| `D:\py\DRL2\local_artifacts\资产归并_20261018\版本快照\DRL2_T1_L_checkout.zip` | `63d2a8a43f2de308d053d3e9575022162c08f4818a646fd060c5f649d2e388a2` | 682 |
| `D:\py\DRL2\local_artifacts\资产归并_20261018\版本快照\DRL2_T1_L_git_metadata.zip` | `b4af7f68b89a67a479c9f32e77db4aa886f46f0b0380b1d85047813aae147b95` | 27 |
| `D:\py\DRL2\local_artifacts\资产归并_20261018\版本快照\DRL2_T1_L_S_checkout.zip` | `d5fd5b2827e4de518766a46a6e13857e65fa3f59f737d33df48d94e6b21a9097` | 694 |
| `D:\py\DRL2\local_artifacts\资产归并_20261018\版本快照\DRL2_T1_L_S_git_metadata.zip` | `e19833eab210a64ab067b920c59337c270c8f39cc812ed98eb19495a99ea60d9` | 27 |
| `D:\py\DRL2\local_artifacts\资产归并_20261018\版本快照\DRL2_T1_O_checkout.zip` | `158be7b4732d6a8ad0f67cc1e4c6c261b14818320f8f382339d57562173f1de9` | 725 |
| `D:\py\DRL2\local_artifacts\资产归并_20261018\版本快照\DRL2_T1_O_git_metadata.zip` | `483a3828fd47e5c342dea68af8bfbd243515d93c75acb64eef457ddf7fd28350` | 27 |
| `D:\py\DRL2\local_artifacts\资产归并_20261018\版本快照\spacecraft_checkout.zip` | `29d4f59065c772d56b71d20b76a3c66d2e776cf528a7c2198dd5921f528f34ff` | 411 |
| `D:\py\DRL2\local_artifacts\资产归并_20261018\版本快照\spacecraft_git_metadata.zip` | `eb1b7ef49fce89658d51dda2e1da4fa301ebbdfae0c455a33c5314d9d9b54979` | 40 |
| `D:\py\DRL2\local_artifacts\资产归并_20261018\版本快照\v3-interface-worktree_checkout.zip` | `abe9b1cee631d289609ec01af6f9e10fc94cd4d053d2b3cebcb4c0b4edfb3043` | 358 |
| `D:\py\DRL2\local_artifacts\资产归并_20261018\版本快照\v3-t0b-worktree_checkout.zip` | `2371a732cf365cccf45d51c0831cb5d93bc70a4fafafd9adc3e6e780b189bf2a` | 393 |
| `D:\py\DRL2\local_artifacts\资产归并_20261018\版本快照\审查工作树_checkout.zip` | `38a530891bfaa0d843f37e48fd5d5e5cb3e364f7c8d0de96e22bf740c3f32734` | 475 |
| `D:\py\DRL2\local_artifacts\资产归并_20261018\版本快照\扫描工作树_a714c61_checkout.zip` | `b5eb0d7ce973bfac66632e737833c52e61850f7c4bc32106ce383bb28aae71c2` | 473 |
| `D:\py\DRL2\local_artifacts\资产归并_20261018\版本快照\incomplete_clone_20261016.zip` | `8739c76e681f900923b900c9df0ef75cf421d39cabb54650c4b9ad19b6a76d85` | 0 |

每份快照附逐成员manifest；快照不上传模型或Git元数据大包。原实验冻结版本可取证恢复，但不构成运行授权。

## Git收尾与现场状态

三条experiment分支均先通过merge-base祖先检查才删除；只保留main、claude/sac-mpc-coupling-design-ns7g6i、collab/spacecraft。没有新建、删除、移动tag或强推。存续DRL2与collab已fetch --prune（禁用tag抓取/清理）。DRL2_T1、DRL2_T1_L、DRL2_T1_O等目录在最新Git收尾指令到达前已依前一条资产归并授权退役，故该项记NOT_APPLICABLE_RETIRED，不重建目录或伪造fetch回执。主检出仍保持原分支/HEAD；无实验运行授权。

自动跟进保持PAUSED；本轮未运行训练、评估或测试。CURRENT旧对象完整放入previous_phase_snapshot，新顶层为空闲；last_delivery仍be5727d。

## 本机支撑清单（不上传）

以下CSV/JSON均位于 `C:/Users/35884/Documents/Spacecraft/过程文件/资产归并_20261018/记录/`，并镜像必要定位索引于主工程local_artifacts。

- `PAYLOAD_LOCATION_MAP.csv` SHA256 `8a58817d5ae3a1fec0fa07d53e47eada24fbb4cbeceb69a5840c795aacdbdfe1`
- `MODEL_AND_ARCHIVE_LOCATION_INDEX.csv` SHA256 `fe8e061688df8588b63e20e72ed793e547e81a0d71dbeffc2af379a153b1d4e6`
- `SPACECRAFT_DIRECTORY_MAP.csv` SHA256 `d032b7858f68a1824fd01e4c2084707d009391022b550ef3e6789100b2eb66ce`
- `SPACECRAFT_MOVE_RECEIPT.json` SHA256 `ec9a1c9b3e05b5b867d465923900c4185c3f172427e5bac35d7892bffdddc985`
- `SNAPSHOTS.json` SHA256 `daf0e4d690b08e99006a451e0b4cb0f9b0830af54bd73683b1c317ad27384799`
- `RETIRED_DIRECTORIES.json` SHA256 `80b8041b76d92b6575e93f139df41c738d6c377f224246b0bf1f2dcc4be28dfa`
- `RETIREMENT_RECEIPT.json` SHA256 `a5928488ab198eb3319b4e0047fd51e513782a4c4e873a078830755f74d82554`
- `QUARANTINED_CACHES.json` SHA256 `dcf43bbb86b9b4921fb1107b2739e40bcfd88c237ae1bd58725ead1cb597a275`
- `T1V4_TEMP_PATH_MAP.json` SHA256 `a6b6c8b747043b9ea83bdd07243744bffe37c2a48e2eeb107c1366f0b4d81afd`
- `FINAL_RECEIPT.json` SHA256 `ed684b887080a424c5024041d8e905cf78bda8461ca3ac571b1dca8a9abdfe63`
- `REMOTE_BRANCH_ANCESTRY_CHECKS.txt` SHA256 `f46e562a2e865eedd391f7a4299ae3d8afbc6bef69d7739fa45fbbd1805eec91`
- `REMOTE_HEADS_AFTER.txt` SHA256 `3e2e444a35cd03272416a43b3e4e4516b581373566af8b2e99f591acc3bf2ff9`
