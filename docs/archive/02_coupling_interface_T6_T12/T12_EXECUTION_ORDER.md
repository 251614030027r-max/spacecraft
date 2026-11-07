# T12 执行单:闸门 B 已通过,开训

写于 2026-09-13。**下层读这一份就够。** 裁决依据在 `docs/T11_GATE_B_RULING.md`。

---

## 0. 状态

- **闸门 A 通过**(`docs/T10_GATE_A_RULING.md`),**闸门 B 通过**(`docs/T11_GATE_B_RULING.md`)。
- 闸门 B 那次 STOP 是**我的判据写错了**,不是下层的问题:我写成"常数追平基线就停",
  而基线本身就是常数 T=0,那是恒真的。正确判据是**最好的常数对逐状态**:
  **32/48 对 40/48,差 8 个种子**,通过。工具已修,不会再读反。
- 已完成的 156 局 + 192 局**都不重跑**。MPC 和任务**保持冻结**。

**这一轮是训练轮。**

---

## 1. S0:落补丁

```
git am -3 < T12_delivery.patch
python -B -m pytest -q
```

**期望 228 passed**(224 + 棘轮 4 项)。下层当前是 `b7deb71` / 224 passed,说明补丁还没应用。

补丁内容:闸门 A 裁决、闸门 B 裁决、**S8 提交棘轮 + 观测补状态**、
`summarize_headroom` 判据修正、`max_consecutive_zero_wrench_steps` 字段。

---

## 2. ⚠ 一个会静默毁掉整轮的参数:`--horizon 35`

`train_hybrid` 的 `--horizon` **默认是 20**,而 T10/T11 的基线、扫描、headroom
**全部跑在 h35**。训练用默认值就会训出一个和主表基线不同时域的策略,**整轮作废**。

**所有训练必须显式写 `--horizon 35`。**

关于时域与保持步数不等(35 对 20)的说明,免得有人当成 bug:
`CLAUDE.md` 陷阱 4 说"包装器只在 `horizon_steps == external_reference_hold_steps`
时透明"。本线路上 **horizon=35、hold=20**,不满足那个条件。但这不是疏漏:

- T10/T11 的**每一个**数字——48 种子基线、六组提交扫描、192 局反向代价——
  都是在这个组合下测的,内部完全一致;
- 更硬的一条:这个组合下的 `commit_now` h35 结果是
  **9/12、88.5 s、167.2 N·s**,与仓库用**另一条参考路径**(`reference_source="fixed"`,
  不经包装器)独立产出的 `docs/PURE_MPC_ROW_VERIFIED.md` h35 行**逐项相同**。
  两条路径同值,说明常值参考下这个组合没有引入差异。

所以:**沿用 h35,不要改成 20,也不要把 `decision_period_steps` 调到 35**
(那会把决策周期变成 3.5 s,headroom 扫描全部作废)。

---

## 3. S9:训练,六个运行,可并行

两条臂,各三个种子,**全部从零**,60000 决策,h35。

**本文接口(`arrival_condition`,带棘轮):**

```
for SEED in 262400 262401 262402 ; do
  python -B -m train.train_hybrid --steps 60000 --seed $SEED --horizon 35 \
    --parametrization arrival_condition --no-execution-feedback \
    --run-name sac_mpc_arrival_$SEED --log-root logs/t12_train &
done
```

**普通航点对照(`radial_local`):**

```
for SEED in 262500 262501 262502 ; do
  python -B -m train.train_hybrid --steps 60000 --seed $SEED --horizon 35 \
    --parametrization radial_local --no-execution-feedback \
    --run-name sac_mpc_radial_$SEED --log-root logs/t12_train &
done
```

**逐条注意:**

- `--no-execution-feedback` **必须写**。`--execution-feedback` 是默认值,
  而反馈通道本轮不进核心(T7 点估计不利,且旧反馈信号受错误门控污染)。
- `--horizon 35` **必须写**(见 §2)。
- 棘轮默认开启,`monotone_commit` 会写进 manifest。**不要**加 `--no-monotone-commit`——
  那是日后的消融臂,不是主行。
- 观测维数(冒烟实测,不是推算):`arrival_condition` **32 维**
  = 24 基础 + 6 目标姿态 + 1 剩余时间 + 1 棘轮;`radial_local` **31 维**(无棘轮那一维)。
  动作维数 2 和 4。这是预期的,不是错配;评估时照 §4 的开关加载。
  (`train_hybrid` 无条件开启 phase/time 观测,所以不是裸 24/25 维。)
- **训练可以并行**;评估的并行授权见 T11 §1(结果型可并行,耗时必须串行单进程)。
- 任一种子全程零成功**照报**,不改参数、不加第四轮、不换种子。

---

## 4. ⚠ S10:评估的三个开关,一个都不能用默认值

`evaluate_hybrid_policy.py` 有三个默认值和本轮训练**不一致**:

| 开关 | 脚本默认 | 本轮训练实际 | 用默认的后果 |
|---|---|---|---|
| `--horizon` | 20 | **35** | 时域错配,和主表基线不可比 |
| `--parametrization` | `radial_local` | **`arrival_condition`** | 动作维数错 |
| `--phase-time-observation` | 关 | **开** | 观测 31 对 32 维,**直接拒绝加载** |

**不用猜。** 每个运行的 `manifest.json` 里现在直接写着一行可复制的
**`evaluation_flags`**,照抄即可。本轮两条臂分别是:

```
# arrival_condition（32D 观测 / 2D 动作）
--horizon 35 --parametrization arrival_condition --phase-time-observation

# radial_local（31D 观测 / 4D 动作）
--horizon 35 --parametrization radial_local --phase-time-observation
```

`--execution-feedback` **不要加**(本轮训练没开,脚本默认也是关,一致)。

完整命令(48 个种子,`deterministic=True`,可并行):

```
python -B -m experiments.evaluate_hybrid_policy \
  --model logs/t12_train/sac_mpc_arrival_262400/final_model.zip \
  --episodes 48 --seed 262000 \
  --horizon 35 --parametrization arrival_condition --phase-time-observation \
  --output logs/t12_eval/arrival_262400.json
```

已在上层沙箱实跑验证:训练 → `final_model.zip` → 上面这组开关加载成功、
通过维数检查、正常跑回合。

每一行必须带:完成数、成功平均时间、**成功平均力冲量**、真值违约率、
非法穿越总数、`qp_infeasible_steps`、`max_consecutive_zero_wrench_steps`。
最后两列是闸门 A 那条残留的可见性要求,**永远不要删**。

## 5. 主表(本轮三行)

| 行 | 状态 |
|---|---|
| **Pure MPC(＝`commit_now`)** | **已有:32/48、78.825 s、187.721 N·s、非法穿越 18、不可行 0** |
| 普通航点 SAC-MPC(`radial_local`) | 本轮训练,三种子,报分布 |
| **本文 `arrival_condition`** | 本轮训练,三种子,报分布 |

**Pure SAC 本轮不做**,等耦合线干净了再单独补,所以是三行表。

**compute 那一列单独串行单进程跑**,不跟并行批混。

---

## 6. 必须一起带走的三条

1. **逐状态 40/48 是 oracle,不是控制器。** T 是知道结果后按种子挑的,它是状态相关规则的
   **性能上界**,**永远不能作为主表的一行**。训练出来的策略能逼近多少,正是本轮要测的。
2. **T8 的负结果**:旧下层上,最好的训练种子输给固定设定点(7/12 对 8/12,配对燃料
   244.7 对 174.5 N·s,贵 40%)。它是旧缺陷控制器的历史证据,但在新结果出来之前,
   **目前唯一测过的正面对比仍然是负的**。
3. **燃料**:逐状态在 32 个基线完成的种子上一律选 T=0,所以那 32 个上它就是固定设定点;
   均燃料 187.7 → 230.9 N·s(+23%)**完全来自新增的 8 个完成**。被救种子最省燃料的成功
   轨迹 225.2–718.1 N·s,约为基线成功均值的 1.20–3.83 倍(这套 48 种子数据;
   上层 12 种子那套的 2.4–5.9 倍是另一网格,不要混用)。

---

## 7. 一件事务性说明

**原始日志推送被自动审批拒绝**:不要为它停推进。大体积 stdout/stderr 不必入 Git,
汇总 JSON 和报告进去就够。

(Pure SAC 行已由上层决定**本轮不做**,等耦合线干净后再单独补,不占本轮机时。)

---

## 8. 不要做的事

- 不要改 MPC(冻结)、不要加球面预览、不要动松弛上限或回退。
- 不要动任何冻结任务参数(几何、约束、时限、奖励、`omega_T`)。
- 不要复用 T7/T8 的 checkpoint(旧下层上训的,作废)。
- 不要用 `--horizon` 的默认值。
- 不要漏 `--no-execution-feedback`(训练侧)。
- 评估侧不要用 `--horizon` / `--parametrization` / `--phase-time-observation` 的默认值,
  照 manifest 的 `evaluation_flags` 抄。
- 不要在耗时测量里并行。
- 不要为了让 Hybrid 好看去削弱 Pure MPC。
- 中途不报常规进度;只在**任一种子出现 NaN、无故退出、或六个运行全部结束**时报。
