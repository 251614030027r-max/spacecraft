# R2_EQUIV：第二轮协调与第一稿 ECC_LE 执行的逐位核对（2026-11-19，上层沙箱）

**对象：** `docs/final_method/ROUND2.md` 第 2 节的逐位判据。

| 侧 | 代码 | 设置 |
|---|---|---|
| 参照 | 科学 `72a8d41`（第一稿） | `certify=True`、`long_entry=True`、强制紧急任务、48 维观测 |
| 新 | 科学 `5fb3d18d`（第二轮实现） | `coordination=True`（修正、直接入口均开）、`potential='lead24'`、17 维观测 |

`5fb3d18d` 之后，势函数又修订了两次（`nominal_entry_time`、`time_to_entry`，`ROUND2.md` 第 3 节的修订记录）。这些函数只在 `potential='opportunity'` 时被调用，本核对用 `lead24`，被比较的代码路径没有改变。

**做法：**
- 初始条件 294020、294031、294302–294305，各跑两条意图序列：
  - 均匀随机意图；
  - 慢速意图 a_p ∈ [−1, −0.2]，用于逼出末次经过；
- 每一步记录并逐行比较：
  - 观测（新侧 17 项，参照侧取对应项）；
  - 塑形回报与真实回报；
  - 事件、模式、时间、Δv、保持、阶段规划状态、规划 Δv；
  - 协调记录：原意图的预测、9 个候选的预测、执行意图、是否启动直接入口；
  - 审计记录、终局余量、回合统计、重定时记录。

**结果（`sandbox/COMPARISON.json`）：**
- 12 / 12 个回合逐行一致；
- 31 次协调决策：保持 15、修正 13、启动直接入口 3；
- 回合结局：11 次捕获、1 次超时（294020 慢速序列）。

**脚本：**

```
python -B scripts/env_equiv.py ROOT OUT.json 294020 294031 294302      # 两份检出各跑一次（ROOT 为检出目录）
python -B scripts/compare.py REF1.json REF2.json -- NEW1.json NEW2.json
```

原始记录：`sandbox/ref_*.json`、`sandbox/new_*.json`。
