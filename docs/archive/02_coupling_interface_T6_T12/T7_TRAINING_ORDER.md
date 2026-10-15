# T7 执行单：打补丁、验证、开训练（2026-09-11）

> 读者：下层执行窗口（`D:\py\DRL2`）。本文自包含。
> 上游交付：`T6_delivery.patch`（4 条提交，基线 `3e5694c`）+ `docs/T6_COUPLING_INTERFACE.md`。
> **本单授权范围：S1–S4。除此之外不做别的。**

---

## 0. 背景一句话

T6 的耦合接口已定稿并通过验证。上层每 2 s 输出 2 维「到达条件」动作，
下层每个决策周期回传 3 维执行摘要。MPC、动力学、任务、奖励**一行未改**。
现在要做的是打上补丁、核验、开训练。

---

## S1：放置并应用补丁

**补丁放在 `D:\py\DRL2` 仓库根目录**（与 `.git` 同级），不要放进子目录。

补丁基于 `3e5694c`，而本地已在 `4eeaeee`（领先 6 条）。这 6 条也动过
`env/hybrid_env.py`（新增 info 统计），补丁同样动这个文件，**大概率需要三方合并**，
所以直接用 `-3`：

```powershell
cd D:\py\DRL2
git status --short          # 必须干净，有改动先提交或 stash
git am -3 T6_delivery.patch
```

**若出现冲突**：冲突只会在 `env/hybrid_env.py` 与 `train/train_hybrid.py`。
两边的改动是**互不重叠的增量**——本地那 6 条加的是 info 字段，补丁加的是
新参数化、反馈通道和观测维度。按「两边都保留」解，然后：

```powershell
git add <解决后的文件>
git am --continue
```

**不要**用 `git am --skip` 跳过任何一条，也不要 `git checkout --theirs/--ours` 整文件覆盖，
那会丢掉一侧的改动。

**若 `git am` 反复失败**，改用旁支再合并，不要硬扛：

```powershell
git checkout -b t6-import 3e5694c
git am T6_delivery.patch
git checkout claude/sac-mpc-coupling-design-ns7g6i
git merge t6-import
```

补丁含 4 条提交：3 条纯文档（执行计划、T0 修订令、T6 接口稿）+ 1 条代码
（`feat: freeze the T6 arrival-condition coupling interface`）。

---

## S2：核验（不通过就停下报告，不要开训练）

```powershell
python -B -m pytest -q
```

**预期 211 passed。** 若数字不同，先报告再动。

然后确认接口装好了：

```powershell
python -B -c "from env.hybrid_env import PrecaptureHybridConfig as C; c=C(waypoint_parametrization='arrival_condition'); print(c.action_dimension)"
```

**预期输出 `2`。**

### 可选但推荐：能力下界复核

补丁把「零动作即经典控制器」做成了可执行事实，值得在本机再确认一次。
上游已在 12 个种子、每个前 300 个控制步上测过：`a = (+1, ·)` 经耦合包装器发出的
指令，与直接构造 `reference_source="fixed"` 的 MPC **逐位相同**（`max|Δwrench| = 0`）。
本机复核用 `tests/test_hybrid_env.py` 里的
`test_arrival_condition_commit_end_names_the_desired_pose` 即可，S2 的全量回归已覆盖。

---

## S3：开训练（六个跑，两版本 × 三种子）

**必须同训练步数、每次从零、不载检查点。** 版本之间唯一的差别是那一条反向通道。

```powershell
# V3：完整耦合（接近策略 + 下层执行反馈）
python -B -m train.train_hybrid --steps 60000 --seed 262300 --parametrization arrival_condition --run-name sac_mpc_v3_bidirectional_262300
python -B -m train.train_hybrid --steps 60000 --seed 262301 --parametrization arrival_condition --run-name sac_mpc_v3_bidirectional_262301
python -B -m train.train_hybrid --steps 60000 --seed 262302 --parametrization arrival_condition --run-name sac_mpc_v3_bidirectional_262302

# V2：单向消融（其余全同，只摘掉反向通道）
python -B -m train.train_hybrid --steps 60000 --seed 262200 --parametrization arrival_condition --no-execution-feedback --run-name sac_mpc_v2_arrival_262200
python -B -m train.train_hybrid --steps 60000 --seed 262201 --parametrization arrival_condition --no-execution-feedback --run-name sac_mpc_v2_arrival_262201
python -B -m train.train_hybrid --steps 60000 --seed 262202 --parametrization arrival_condition --no-execution-feedback --run-name sac_mpc_v2_arrival_262202
```

要点：

- 训练**可以并行**跑多个进程。**但任何并行运行的耗时都不得用作实时性结论**，
  实时只认串行单进程评估。
- 每个 run 的 manifest 会记录 `coupling_direction` 字段，用它区分 V2/V3，不要靠 run 名。
- 中途不要改任何参数。若某个种子崩了，**重跑该种子**，不要换种子。

---

## S4：推送积压提交

本地 `4eeaeee` 领先远端 6 条，那 6 条里有 **T1 的逐时域 episode 产物**，
主表的 Pure MPC 行（各时域的完成率/时间/燃料/最差真值裕度）要从那批数据里出，
上游拿不到就少一行。

```powershell
git push -u origin claude/sac-mpc-coupling-design-ns7g6i
```

推完把这两样报回来：

1. 训练日志目录名（六个）
2. 确认远端已含 T1 产物

---

## 明确不做

- 不动 MPC、动力学、任务定义、奖励、约束
- 不动 `env/hybrid_env.py` 的默认参数
- 不加第三个动作通道、不加姿态相关改动
- 不给「正确进入时机」或「等待」任何额外奖励
- 不用中途切换控制器制造性能
- 不从检查点续训，不复用现有 25k 检查点填任何一行

## 什么情况停下来报告

- S1 的 `git am` 用两种办法都过不去
- S2 的回归不是 211 passed，或接口维数不是 2
- 训练中出现 NaN、崩溃、或某个种子反复起不来
- 任何看起来需要改任务/奖励/MPC 才能推进的情况

---

## 附：这次接口为什么长这样（供理解，不需执行）

三条都有实测支撑：

- **切入时刻**是唯一救回过持续失败工况的变量，所以它是主通道。
- **保持半径**新增救援为 0，但在配对格上省燃料 20.6%，所以留作第二通道。
- **横向调整** 26 格全败，**不给通道**。

新接口在 262005 上已复核：旧接口最快 271.2 s 完成，
新接口用分级暂存后 **122.1 s** 完成，且可行格明显增多但仍是挑剔的
（不是所有格都通过）——挑剔性正是「这里存在一个值得学的决策」的证据。

262006 已单独结案：离线可行性证书能在 217.5 s 完成它、七类约束零真值违约、
视场裕度 0.693 rad，**所以它可解**；但四类位置决策（时刻/半径/速率/横向）
共 90 余格全败，说明它的失败是**指向失败**而非路径选择失败。
这是一个独立的候选单因子（姿态参考），**本轮不做**，需单独授权。
