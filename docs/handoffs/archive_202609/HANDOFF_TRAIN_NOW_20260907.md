# 开训指令（含完整交付包）

包：`precapture_coupling_complete_20260907.zip`。**取代此前所有版本**，前几版没覆盖也不影响。
基线 `1fe5549`。`pytest -q` = **188 passed**。

---

## 0. 一句话

耦合环境、训练入口、评估脚本、可行性判定全部就绪。**先跑一版看提升，然后再在耦合上做创新。**

---

## 1. 先做（约 20 分钟）

**S-0 应用与自检**

解压覆盖 → `.\.venv\Scripts\python.exe -B -m pytest -q`（你当前 181 + 新增 7）。
两个新 MPC 配置项默认值都是旧行为（`precapture_attitude_reference="frozen"`、
`external_reference_frame="inertial"`），耦合是全新文件，**既有路径与既有数字一个都不动**。
`controllers/mpc/config.py`、`controller.py` 是 `1fe5549` 整文件；覆盖前确认它们在
`1fe5549..e212b4a` 之间没被改过，改过就用包内 `UPPER_DELIVERY.patch`。

**S-1 建对照行（串行、单进程，约 30 分钟）**

```powershell
.\.venv\Scripts\python.exe -B -m experiments.evaluate_hybrid_policy `
  --episodes 12 --seed 262000 --control desired_pose `
  --output logs\hybrid\control_desired_pose_262000.json
```

这就是**耦合方法必须打败的那一行**。上层沙箱在 `1fe5549` 上得到 8/12；
你带 F1 之后可能不同，**以你的数为准**。
这一条务必串行单进程跑——算力列只从这里取。

---

## 2. 开训（每种子约 15 小时）

```powershell
foreach ($s in 262100,262101,262102) {
  .\.venv\Scripts\python.exe -B -m train.train_hybrid `
    --steps 60000 --seed $s --run-name sac_mpc_hybrid_$s
}
```

- **每次从零**：全新 actor/critic/replay，不从 checkpoint 起。
- **训练可以并行**（三个种子同时开）。不碰红线，因为**实时算力只从 S-1 和 S-3 的串行评估里报**。
- 一个决策步 = 20 次 MPC 求解 + 20 步 RK45，实测 0.90 s。
- checkpoint 每 5000 决策步存一次。

**中途只看一件事，不要调参**：`logs\hybrid\sac_mpc_hybrid_*\train.monitor.csv` 里的
`completed` 列。到 20000 步时如果三个种子的完成率都还是 0，**停下来告诉我**，
不要自己改超参——那说明设计有问题，不是调参问题。

---

## 3. 出结果（串行、单进程）

```powershell
foreach ($s in 262100,262101,262102) {
  .\.venv\Scripts\python.exe -B -m experiments.evaluate_hybrid_policy `
    --episodes 12 --seed 262000 --model logs\hybrid\sac_mpc_hybrid_$s\final_model.zip `
    --output logs\hybrid\eval_$s.json
}
```

**报三个种子的分布，不报最好的那个。**每份 JSON 里的 `main_table` 块就是论文表格的四列。

读的时候对着两个数：
- **必须赢过 S-1 那一行**（上层沙箱是 8/12、116–142 s）
- **11/12 是"只选入口时机"能买到的天花板**（上层扫描出来的）
- 超过 11/12，说明 policy 找到了扫描没找到的东西

---

## 4. 顺带确认的两件事

**N-A（重要，可与训练并行做）** 在 `e212b4a` 上复算 12 种子扫描：

```powershell
foreach ($h in 20,50) { foreach ($s in 262000..262011) {
  .\.venv\Scripts\python.exe -B -m experiments.diagnose_precapture_reference `
    --seed $s --horizon $h --guidance fixed --output logs\gap\gap_h${h}_$s.json
}}
```
只报 `completed` / `terminal_region_active` / `illegal_terminal_entry_count` / `violation_steps`。
**要回答：带 F1 之后还剩几个失败，是不是仍然全是入口门失败。**这决定缺口的最终大小。

**N-B** 把 `external_reference_frame="target"` 作为单因子正式化：独立 commit + manifest。

---

## 5. 上层这轮测到的，供你对照

- **缺口**：12 种子定点 MPC，h20 8/12、h50 1/12；**15 次失败全是入口门失败、零违约**，
  失败后追踪星仍停在目标点上（离目标中位数 0.000 m）。
- **接口**：3D 航点通道在惯性定向下 0/3，改目标体系后 3/3 且**逐位无损**；
  oracle 走修好的通道 0/9 → 3/3，**S5 那批结论作废重做**。
- **可行性**：定点 MPC 失败的 4 个种子里，光选入口时机就救回 3 个（天花板 11/12）；
  seed 262011 **只有 60 s 那一档成**，40 s 和 80 s 都不成——窗口很窄。
- **实时性**：h20 耦合控制器串行 p95 = **0.47× 预算**，在预算内（h50 不在）。
  绝对毫秒不跨机器可比，报比值时把平台一起写。
- **姿态参考**：三种公式实测，现有 `frozen` 最好，我原来的前馈假设被证伪。

## 6. 红线（未松动）

- 违约按 RK45 真值 + 真实几何判。
- **实时性只认串行单进程**——训练可并行，评估必须串行。
- 不看结果调参；12 种子连号全报，入口时机扫描整格全报（含无解的 262006）。
- 下判断 ≥3 训练种子报分布，不报最好 seed。
- 每次训练从零。
- 训练中途不调超参；异常就停下来报告。
