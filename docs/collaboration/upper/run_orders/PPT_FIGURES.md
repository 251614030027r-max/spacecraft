# 出图计划：PPT_FIGURES（汇报稿 wang23 改版；论文层面展示）

**状态：用户 11-05 要求。** 只出主流图（训练曲线、柱状对比、箱线图、配对柱、可行区热力图、轨迹图）；图中只出现论文层面的名称（EACH、U1、C1、SAC、纯 MPC），不出现工程细节。

**脚本与已出的图：** 科学 `6ae0bbd`，`experiments/ppt_figures.py`。开发期这一组已由上层出好：`evidence/ppt_dev/`（PNG 300 dpi、PDF、每张图对应的 JSON）。

**下层只需做一件事（预实验交付之后，不得打断正在跑的预实验）：**

```powershell
cd D:\py\DRL2; git fetch origin claude/sac-mpc-coupling-design-ns7g6i; git checkout --detach 6ae0bbd3d568583218ff0d847e59bd4a712ab3c5
& "D:\py\DRL2\.venv\Scripts\python.exe" -B -m experiments.ppt_figures pretest --root "C:\Users\35884\Documents\Spacecraft\过程文件\final_pretest" --output "C:\Users\35884\Documents\Spacecraft\过程文件\final_pretest\ppt"
```

产出与开发期同名的 P1–P5（预实验版）、`ablation/` 下的消融图，以及窗口配对图 `P4_window_commitment_pretest`。轨迹图直接用预实验自带的 `figures/fig_trajectory`。产物提交到科学分支 `evidence/final_pretest/ppt/`。

## 与汇报稿的对应（wang23 改版）

| 汇报稿页 | 改动 | 用图 |
|---|---|---|
| 1.4 研究目标 | 目标 2 改为"执行约束知识与真实价值学习的非对称耦合（U1）"，不再写反事实 | — |
| 2.2 方法总体框架 | 双向耦合总图：向下 C1（扫掠锥阶段意图 → G★ 窗口入口 → MPC），向上 U1（执行证书只进训练期 critic）；部署时只有策略网络 | 示意图，PPT 内绘制（结构见 `docs/final_method/METHOD_MINIMAL.md` 总图） |
| 2.5–2.6（原反事实两页） | 合并为一页 U1：critic 读执行证书，TD 只用真实转移；三个命题（无偏、表示能力不降、误差隔离） | P6（作为"为什么这样设计"的证据） |
| 2.7 C1 | 扫掠锥机会集 + 窗口承诺 | P5、P4 |
| 3.1 仿真设置 | 紧急期限改为 140–200 s；对比组：纯 MPC、SAC、EACH（加 G★、截止规则作参照） | — |
| 3.2 主结果 | 学习效率与最终性能 | P1、P2（可加 P3） |
| 3.3 机制验证 | U1：EACH 对 SAC、对等宽 critic；C1：对无几何、窗口配对 | 预实验 `ablation/`、`P4_window_commitment_pretest` |
| 3.4 典型轨迹 | 一个开局的三维轨迹 + 距离与模式 | 预实验 `figures/fig_trajectory` |

**口径：** 开发期图的页脚标注"开发验证"，预实验图标注"预实验"。正式实验之前，两者都不当作论文最终数字。
