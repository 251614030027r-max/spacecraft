# 收尾单（给下层）：清干净，然后停下等耦合设计

*2026-09-22。基线 `e8cfa01`。*

**耦合设计讨论在上层进行，你不参与。你只做下面四件，全部零机时、不训练、不做实验。**
做完停下。

---

## 已经不需要你做的事（上层已完成，`git pull` 即可看到）

| | 状态 |
|---|---|
| 米限速上界裁定 | **已定 `v2_reference_step_max_m = 0.40`**，理由与我那个错判据的复盘写在 `docs/V2_RATE_LIMIT_DECISION_20260922.md`。**不要再扫 0.60 / 0.80** |
| `CLAUDE.md` 顶部状态 | 已刷新到 2026-09-22（此前停在 09-19，还写着"正在重训"） |
| `env/hybrid_env.py` 里 V1 的棘轮理由注释 | 已标注**作废、不得用于自适应任务**（`8/8 → 0/8` 在本任务复现不出来） |
| 上层全套回归 | **283 passed, 3 xfailed** |

---

## 你要做的四件

### 1. 在你的机器上跑一次全套回归

你上一轮按预注册规则停在四档全不满足，**没有跑全套 282/283**。现在上界已定，跑一次确认。

```
python -B -m pytest -q
```

**期望：`283 passed, 3 xfailed`。** 不一致就停下报上来，不要自己修。

### 2. 保留那三个 V1 的 strict xfail，不要清理

`tests/test_coupling_diagnostics.py` 里的 T-A / T-B / T-C 钉的是 V1 `arrival_condition`
路径的缺陷。**那条代码路径仍然存活**（Pure MPC 对照臂、历史评估复现都走它），
所以这三个钉子**必须留着**。

V2 用的是 `task_state_v2`，不继承那个实现——**所以它们不会因为 V2 修好而变绿，
这是预期行为，不是遗留垃圾。** 谁来做清理都不要动它们。

### 3. 把新文档补进两个索引

`docs/EVIDENCE_INDEX.md`（62 行）与 `docs/README.md`（17 行）里
**一条 09-21 / 09-22 的文档都没有**。补上这些：

```
docs/ADAPTIVE_ROUND1_CLOSEOUT_20260921.md          第一轮收尾与根因终判（含 20 个产物 SHA-256）
docs/COUPLING_DEFECT_DIAGNOSIS_ORDER_20260921.md   预注册诊断协议
docs/COUPLING_DEFECT_DIAGNOSIS_LOG_20260921.md     D0/D1 逐步审计
docs/ROUND1_EXTERNAL_REVIEW_20260922.md            外部审查件
docs/V2_EXECUTION_ORDER_20260922.md                V2 执行单
docs/V2_EXECUTION_SIGNAL_AUDIT_20260922.md         离线信号审计
docs/V2_INTERFACE_IMPLEMENTATION_20260922.md       V2 实现与验证
docs/V2_REVIEW_RATE_LIMIT_20260922.md              V2 独立复核
docs/V2_FIX_ORDER_RATE_LIMIT_20260922.md           限速修复单
docs/V2_RATE_LIMIT_SWEEP_STOP_20260922.md          四档扫描停止报告
docs/V2_RATE_LIMIT_DECISION_20260922.md            上界裁定（0.40 m）与判据复盘
docs/V3_LITERATURE_AND_COUPLING_DIRECTION_20260922.md  耦合方向与三篇核心文献
docs/DECK_CLEARING_ORDER_20260922.md               本文
```

**纯索引补录，不要改这些文档的内容。**

### 4. 在诊断日志里写一行结案

在 `docs/COUPLING_DEFECT_DIAGNOSIS_LOG_20260921.md` 末尾追加一段，写明：

- **D2 取消**（不是待授权）——D1 已测出有效介入恰好为 0，且 75% 的锁存由策略自身造成；
- **D5 由用户主动停止**，仅 r=0.10 的 Pure MPC 一行完成（36/48，与标准翻滚相同）；
- **诊断阶段结案**，根因 R5 单一，R2/R3/R4 证伪，R1 次要 23–25%；
- 你那条"预注册介入率未命中"是**上层预注册不精确**（该率由种子 262040 / 262039 独家贡献，
  都不在 262000–262007 里），**不是机理理解有误**。

这一段是为了让后来的窗口不再去等 D2 的授权、不再重开根因。

---

## 禁止事项

- **不训练。** 训练方案上层要先和用户讨论定。
- **不再扫上界**（0.60 / 0.80 都不要）。
- **不接 governor**，不增加任何在线额外 MPC 求解。
- **不改** reward / SAC 超参 / MPC / 任务参数 / 执行器 / 控制周期。
- **不重构 V2 架构**——已复核通过。
- **不动** T1–T7，**不清理** T-A/T-B/T-C。
- **不做**任何新探针、新消融、新扫描。诊断阶段已结案。

---

## 交回

一句话即可：

> 全套 `N passed, M xfailed`；索引已补 13 条；诊断日志已结案；工作树干净、已推送 `<sha>`。

**然后停下。**
