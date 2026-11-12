# 执行书：RUN_CLEANUP（工程收尾：归档标签、Windows 核验、本机资产整理）

**状态：用户 2026-11-12 批准。**
- 用户原话："正式实验之前瘦身，本机只保留最近 3、4 轮即可，开题 PPT 或者一些中途工作的脚本都不需要……旧无用数据记录等无用需要的你直接删除……别删错了以及影响性能"。
- **本执行书不训练正式模型，不访问正式块 300000–999999，不用正式种子 20–22。**

## 0. 背景与冻结

上层已在科学分支完成代码与文档收尾（科学 `21730075`，collab `本次 collab 整理提交（见 PROJECT_STATE）`）：

- **代码：** 只保留正式实验用到的 45 个模块。环境、学习器、驱动、入口删去了全部历史分支（反事实标签、偏好、引导、校准、探询进 critic、证书图、调节器、稀缺任务等）。
- **行为：**
  - 执行器与物理模块与 `e0a61753` 逐字节相同（`t1_env.py` 只多只读审计）；
  - 上层沙箱核验：冻结路径对 `fcd8d2ee` 逐位等价（`evidence/CLEANUP_EQUIV/`）。
- **测试：** 只保留覆盖现行代码的测试，没有预期失败。
- **证据与文档：** 历史测试、T1 / F01 / E 系列与早期 T2 证据、旧阶段文档移出工作树（`docs/archive/PATH_MAP_20261112.csv`），原样在 `fcd8d2ee`。
- **collab：** 历史内容移入 `docs/collaboration/archive/`。

本执行书由下层完成三件事：打归档标签、在 Windows 上复核等价性与测试、整理本机资产。

## A. 归档标签（第一步；annotated tag，已存在的同名标签只核对不移动）

| 标签 | 目标 | 含义 |
|---|---|---|
| `archive/pre-cleanup-20261112` | 科学 `fcd8d2eec2d78faf43ae6522772589d6494cb7e2` | 收尾前完整科学树（全部历史代码、测试、证据） |
| `archive/collab-pre-cleanup-20261112` | collab `d6df275b`（下层用 `git rev-parse` 补全） | 收尾前完整 collab 树 |
| `archive/each-pretest-20261106` | 科学 `fd789126`（补全） | EACH 预实验冻结 |
| `archive/ecc-method-20261110` | 科学 `e0a617532008dac02e2d50f7e859aca613e03e5e` | ECC 方法行为冻结 |
| `archive/formal-code-20261112` | 科学 `21730075ecee9d9bbd06f34febf981121db87e68` | 正式实验代码 |

每个标签核对：
- 目标可从远端分支到达；
- `git ls-remote --tags` 回读一致。

回执写入交付 `TAGS.json`。

## B. Windows 核验（`D:\py\DRL2`）

1. **检出：** `git fetch`，检出 `21730075`，工作区干净（`git status --porcelain` 为空；未跟踪文件先按 C3 处理）。
2. **全量测试：** `python -B -m pytest -q`，退出码必须为 0。记录通过数（沙箱为 `158` passed、0 xfailed）。退出码与日志写入 `PYTEST.json` / `pytest.log`。
3. **逐位等价**（脚本在 `evidence/CLEANUP_EQUIV/scripts/`，用法见其 README）。参照树用 `git worktree add <临时目录> fcd8d2ee`，比较对象是 `21730075`：
   - (a) **环境：** 7 种情形 × 开局 294020、294031，逐步比较观测哈希、回报、事件、模式、时间、Δv、认证记录、只读诊断、终局统计。7 种情形为 ECC 紧急、无认证紧急、无旋转引导经济、刚性重规划、自然、R 预算 2.0、低燃料 0.3 耗尽。
   - (b) **训练：** 6 个配置 × 种子 10 × 40 决策 × 单进程，开正式阶段的只读日志。比较各检查点参数张量、保留字段上的回合记录与训练统计、`worker_order`。
   - (c) **评估：** 用 `fcd8d2ee` 训练的 (b) 中 `ECC_CERT_LOOP` 模型，在两边代码下评估 294300–294301（紧急、带审计），逐决策比较。这一步同时验证旧模型能被新代码加载。
   - (d) **B 步模型：** 用 `eval_equiv.py <根> <B 步 ECC_CERT_LOOP 种子 10 的运行目录> <输出> 294800,294801`，在 `fcd8d2ee` 与 `21730075` 两边各跑一次，再用 `compare_eval.py` 比较。在保留字段上须逐决策一致，被删的只有 `decisions[*].model`、`govern` 等历史键与耗时字段。与 `evidence/FORMAL_PREP/B_MODEL_LOG_EQUIVALENCE.json` 的结论应一致。
4. **入口冒烟：**
   - 种子 10，`--smoke-steps 40`，6 个配置各训练一次（2 进程）；
   - 对冒烟模型各跑一次 `eval --openings 294300-294301`（紧急、经济、资源条件均需显式开发开局），以及登记的开关 `--no-certify`、`--rigid`、`--certify`、验证检查点；
   - `screen` 跑 pure、gstar、rally（各两开局）；
   - `traj --entry ECC=<冒烟 ECC 运行目录> --entry pure=pure --openings 294300 --condition urgent`、`geoscan --openings 294020 --ranges 10` 各跑通一次；
   - `check`、`resolve`、`formal_readout formal / episodes`、`formal_figures --smoke` 各跑通一次。
   - 冒烟输出目录不放在 `过程文件\FORMAL\`。
5. **判定：**
   - 2、3(a)–(d)、4 全部通过，才算核验通过；
   - 任一不一致即停止，把差异的第一处（情形、步、字段、两边值）写进交付，不自行修代码。
   - 跨机器只比较同一台机器上两份代码的结果，不比较沙箱的比特。

## C. 本机资产整理（核验通过后进行）

1. **盘点：** 对 `C:\Users\35884\Documents\Spacecraft\` 全部目录做一次盘点，格式同 11-03 的 `INVENTORY.json`：
   - 目录、文件数、大小；
   - 模型 ZIP 的 SHA-256；
   - 训练 manifest 的关键字段；
   - 是否已有对应的 git 证据目录。

   写入交付 `INVENTORY_20261112.json` / `.md`。
2. **只保留最近几轮**（用户：最近 3、4 轮），保留以下目录的全部内容：
   - `FORMAL_PREP`（含预筛与 DRY）；
   - GOV_B（B 步，方法选型所用模型）；
   - GOV_A；
   - RUN_ECC 开发轮。

   **删除前**，对每个将删目录：
   - 确认其汇总证据已在 git（`fcd8d2ee` 的 `evidence/` 或 collab 交付）；
   - 把每个模型 ZIP 与关键原始文件的路径、大小、SHA-256 追加到 `DELETED_FILES_20261112.jsonl`；
   - 再整目录删除。

   **其余删除范围：**
   - 更早各轮（T1、F01、T2.2–T2.19、EACH 预实验、U2、U2C）的过程目录；
   - 旧发布临时 git index；
   - 重复的中期发布快照；
   - 旧行政故障记录副本；
   - 旧日志压缩包；
   - 开题 PPT 与中途工作脚本目录。

   删除清单写入 `DELETED_FILES_20261112.jsonl`。
3. **科学检出：** `D:\py\DRL2` 中未跟踪且被忽略的大文件（`logs/`、`eval/`、`models/`、旧 venv 之外的缓存等）先登记，再删除；保留当前 venv。
4. **说明文件：** `项目说明.md` 与 `AGENTS.md` 按科学分支新的 `README.md` 与 `CLAUDE.md` 更新。原文备份，并登记前后的 SHA-256。
5. **正式实验预备：**
   - `过程文件\FORMAL\` 不存在；
   - 磁盘剩余 ≥ 5 GB；
   - 没有残留的 Python 进程与计划任务；
   - 结果写入 `READY.json`。
6. **CURRENT.json：** 更新为"收尾完成，等待 RUN_FORMAL 授权"，并写明本机保留的目录。

## 交付

| 位置 | 内容 |
|---|---|
| collab `docs/collaboration/lower/handoffs/cleanup_20261112/` | `RESULT.md`，以及 `TAGS.json`、`PYTEST.json`、`INVENTORY_20261112.json` / `.md`、`DELETED_FILES_20261112.jsonl`（或其 SHA 与本机路径，若过大）、`READY.json` |
| 科学分支 `evidence/CLEANUP_EQUIV/windows/` | B3 的四组比较结果 JSON 与 B4 冒烟汇总 |

上层复核通过后，把 RUN_FORMAL 交给用户授权。

**转发给下层的一句话：**

> 读 collab `docs/collaboration/upper/run_orders/RUN_CLEANUP.md`。按 A（归档标签）→ B（在 `D:\py\DRL2` 检出 `21730075`，全量 pytest 退出码 0，用 `evidence/CLEANUP_EQUIV/scripts/` 对 `fcd8d2ee` 做环境 / 训练 / 评估 / B 步模型四组逐位比较，入口冒烟）→ C（盘点后本机只保留 FORMAL_PREP、GOV_B、GOV_A、RUN_ECC 四轮，其余登记 SHA 后删除）顺序执行；任何不一致即停，不改代码；按"交付"推送。不训练正式模型，不碰 300000 以上。
