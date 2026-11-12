# 图表规格：每张图的数据来源、坐标、分组与统计口径

> **非最终论文算法（2026-11-10）：** 本文件是研发记录。论文方法以 `docs/paper/METHOD.md` 为唯一源，对照见 `docs/paper/FACT_MAP.md`。

所有图由 `python -B -m experiments.final_figures --stage S --root DIR --output DIR [--geoscan G] [--traj T]` 从原始日志一键重建。
- 图例与颜色取自 `METHOD_REGISTRY.json`；
- 脚本先运行 `experiments.final.resolve`：不同科学 SHA、不同预算、配置与注册表不符的输入，一律拒绝；
- 每张图右下角带阶段水印（`dryrun` / `pretest` / `formal`），旁边写出同名 JSON（图中每个数字）；汇总在 `TABLES.json`。

| 图 | 内容 | 数据来源 | 横轴 / 纵轴 | 分组 / 统计 |
|---|---|---|---|---|
| `fig_training` | (a) 训练回合真实回报；(b) 训练回合紧急捕获 | `B/TRAIN_*/episodes.json`（`return_true`、`event`、`mission`、`steps`） | 存储决策数 / 最近 50 个回合（紧急：最近 50 个紧急回合）的平均 | 每种子一条细线；臂均值粗线；阴影为种子范围（不是置信区间）。曲线 AUC 与 `experiments.t2_curves` 相同。标注为行为策略统计 |
| `fig_final` | (a) 捕获率；(b) 紧急捕获率；(c) 平均真实回报 | `C/<CONFIG>_s<seed>/*.json`、`S/<policy>/*.json`（确定性评估） | 方法 / 比例或回报 | 柱为种子均值，点为各种子；附与 EACH 的逐开局配对差（种子平均）、中位数、去掉前 2 名后的均值，以及双方都捕获时的时间与 Δv（只在共同捕获开局上比较） |
| `fig_window`（图 C） | 窗口承诺对刚性承诺：(a) 入口丢失；(b) 捕获率；(c) 逐开局配对回报差 | `C/EACH_s*`、`C/EACH_s*_rigid`、`S/<policy>`、`S/<policy>_rigid` | 策略 / 计数、比例、回报差 | 同一开局、同一策略、同一下层控制，只有承诺方式不同。散点为开局，横线为均值；附双方都捕获时的时间与 Δv |
| `fig_geometry`（图 A） | 扫掠锥入口几何与条件 G★ 可行性 | `experiments.final geoscan`（冻结的恒角速度预报与 G★ 规划器；不读任何评估回报） | 端口相位（°）/ 纬向误差 Δλ（°），每个距离一栏 | 色块：最早可行到达时刻；虚线：θ_min = 35° 名义几何边界；箭头：等待、迎击、伴飞的名义一步位移。标注 β、自旋速率、零相对速度、估计惯量 |
| `fig_trajectory`（图 B） | 真实轨迹：(a) 目标本体系三维轨迹；(b) 距离随时间变化，底色为 EACH 的模式（stage / hold / bridge / descent）；(c) 在图 A 平面上的路径 | `experiments.final traj`（每 2 s 记录一次真值） | — | 选例规则事先固定：图用开局块 294200–294211 的前 3 个开局，不按结果挑选 |

**统计口径（正式实验）：**
- 开局是配对单位，种子是训练重复；报告种子均值、范围与逐开局配对差；
- 时间与 Δv 只在双方都捕获的开局上比较，同时单列只有一方捕获的开局数，避免幸存者偏差；
- 失败开局的选例规则事先登记；
- 训练曲线只来自真实训练回合日志，不用检查点评估连线。
