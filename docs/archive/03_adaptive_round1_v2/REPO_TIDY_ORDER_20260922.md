# 仓库整理单（给下层）

*2026-09-22。基线 `45866cc`。*

**目的**：在开 V2 第一轮训练之前把仓库整理到"新窗口能看懂、老证据不断链"的状态。
**全部零机时，不跑训练，不跑探针。** 做完停下。

---

## 0. 先说三条**不要做**的，理由都是硬的

### ❌ 不要重写 git 历史

`.git` 已经 **3.4 GB**。用 `filter-branch` / `filter-repo` 能瘦下来，**但不要做**：

- 这个项目的地基是「**任何进入决策的数字必须指向仓库里的 artifact**」。历史里那些
  模型 zip、评估 JSON、tensorboard 事件**就是**那些 artifact 的出处。重写历史会改写
  所有 commit 哈希，**9 份文档里的 SHA-256 与提交引用全部作废**。
- 3.4 GB 是已经付掉的代价，瘦身收益是一次性的克隆速度；代价是审计链断掉，不划算。

**结论：体积问题止损，不回溯。**

### ❌ 不要把 `docs/` 或 `logs/` 挪进 `archive/`

实测过了，挪动会断链：

| | 数量 |
|---|---|
| 文档里引用的 `logs/...` 路径 | **111 个** |
| 文档之间互相引用的 `docs/*.md` 文件名 | **53 处** |
| 带 SHA-256 的文档 | **9 份** |

挪一次要改 164 处引用，改漏一处就是一条断掉的证据链。**得不偿失。**

### ❌ 不要删任何 `logs/` 或 `eval/` 下已跟踪的产物

同上，它们是被文档按路径 + 哈希引用的。

---

## 1. 整理的正确形式：一张地图，不是搬家

`logs/` 有 **85 个子目录**、`docs/` 有 **79 个 md**。真正的问题不是它们在哪，
而是**新窗口不知道哪些是活的**。所以整理 = 写一份权威索引。

### 任务 1.1：`docs/INDEX.md`（新建）

把 79 份文档分成三类，每份一行说明。分类只写三档：

- **LIVE** —— 当前有效，必读；
- **EVIDENCE** —— 已结案但被引用（含 SHA-256 的那 9 份必须在这一档）；
- **CLOSED** —— 历史研究线（`single_phase` / Waypoint / 24D / A1–A3 感知 / T0–T12 等），
  **保留不动，仅标注"已关闭，勿据此重开方向"**。

LIVE 一档应该只有十几份。按我的判断至少包括：

```
CLAUDE.md                                          入口
docs/ADAPTIVE_ROUND1_CLOSEOUT_20260921.md          第一轮收尾与根因终判
docs/ROUND1_EXTERNAL_REVIEW_20260922.md            外部审查件
docs/V3_LITERATURE_AND_COUPLING_DIRECTION_20260922.md  耦合方向与三篇核心文献
docs/V2_INTERFACE_IMPLEMENTATION_20260922.md       V2 实现与验证
docs/V2_RATE_LIMIT_DECISION_20260922.md            米限速上界裁定 0.40
docs/ADAPTIVE_MAINLINE_RUNSHEET.md                 运行单
docs/PURE_MPC_ROW_VERIFIED.md                      Pure MPC 基线行
docs/REPRODUCIBILITY.md / EVIDENCE_INDEX.md        复现与证据索引
```

其余 09-21/09-22 的诊断文档归 **EVIDENCE**。

### 任务 1.2：`logs/INDEX.md`（新建）

85 个目录按线分组，每组一行。**必须标出哪些是活的**：

- **活的（勿动）**：`logs/adp_rf_26241{0,1,2}/` —— 第一轮的模型、5k/10k checkpoint、
  训练曲线、manifest，是 `ADAPTIVE_ROUND1_CLOSEOUT` 按哈希引用的证据；
- **已关闭但被引用**：`sac_mpc_v2_arrival_*`、`sac_mpc_v3_bidirectional_*`、
  `p0_horizons_*`、`p1_*`、`p2_*`、`attitude_probe_*`、`compute_h35` 等（T6–T12 那条线）；
- **历史线**：`fullmission_sac_*`、`gatefree_sac_*`、`phase2_mission_*`（共约 40+ 个目录），
  标注"`single_phase` 历史线，已关闭"。

### 任务 1.3：补全 `EVIDENCE_INDEX.md` 与 `docs/README.md`

这两份里**一条 09-21/09-22 的文档都没有**（我上一单已提过，尚未做）。把 13 份补进去。

---

## 2. 止损：让以后不再长大

### 任务 2.1：扩 `.gitignore`

现在只挡了 `*.pt/*.pth/*.ckpt` 和几个具体目录。补上通用规则：

```
# 训练产物：默认不进版本库，发布证据时用显式 force-add
*.zip
events.out.tfevents.*
**/tensorboard/
**/checkpoints/
*.json.partial
```

**注意**：`logs/adp_rf_*` 的模型和 checkpoint **已经在库里了**，加 ignore 不会移除它们
（`git rm --cached` 才会，**不要做**——它们是引用中的证据）。这条规则只管**将来**。

### 任务 2.2：在 `CLAUDE.md` 的「How the work is run」里加一句政策

> 训练产物（模型 zip、checkpoint、tensorboard 事件、评估 JSON）默认被 `.gitignore`
> 排除。**发布证据时用显式 `git add -f` 单独加入，并在文档里记下路径 + SHA-256。**
> 不要整目录提交。

这条是本轮 3.4 GB 的真正成因，写进规则才不会重演。

---

## 3. 本地磁盘（你那台机器，和 git 无关）

`logs/` 工作树 1.7 GB、`References/` 51 MB。**本地想省空间，安全的做法只有一条**：

- 删除**未跟踪**的临时产物（`*.json.partial`、`__pycache__`、旧的 venv、
  `local_artifacts/` 里你确认不再需要的 patch）。
- **已跟踪的一律不动。** 想省空间就 `git clone --depth 1` 开一个浅克隆干活，
  原库留着当证据仓。

---

## 4. 交回

一句话 + 三个文件：

> `docs/INDEX.md`、`logs/INDEX.md` 已建；`EVIDENCE_INDEX.md` / `docs/README.md` 已补 13 条；
> `.gitignore` 已扩；`CLAUDE.md` 已加产物政策；未重写历史、未挪动、未删除任何已跟踪文件；
> 已推送 `<sha>`。

---

## 5. 禁止事项

- **不训练。** 训练由上层和用户定了再开。
- **不跑测试、不跑探针、不跑评估。** 本单纯文件整理。
- **不重写历史、不挪动、不删除已跟踪文件**（第 0 节）。
- **不改** reward / SAC / MPC / 任务参数 / V2 接口 / 任何测试。
- **不动**三个 V1 的 strict xfail（它们钉的 `arrival_condition` 路径仍然活着）。

---

## 附：上层刚提交的东西（你 pull 就有）

`experiments/probe_channel_liveness.py` —— **训练前的活性检查**，V1 缺的就是这一环。
随机策略跑 40 次决策，读三个量：整局有效参考改变率、后半段是否塌、不同参考点数。

实测（种子 262000 / 262001 / 262005）：

| | 有效改变率 | 后半段 | 40 次决策的不同参考点 |
|---|---:|---:|---:|
| V1 `arrival_condition` | 0.026 / 0.025 / 0.050 | **0.000 ×3** | **2 / 2 / 3** |
| V2 `task_state_v2` | **1.000 ×3** | **1.000 ×3** | **40 / 40 / 40** |

**V1 三条判据全不过，V2 三条全过。** 以后任何改了接口的训练轮次之前先跑它，几分钟。
