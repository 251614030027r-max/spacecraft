# 实验注册表

`METHOD_REGISTRY.json` 是第二轮实验唯一的配置来源：训练配置（算法、协调 / 修正 / 直接入口开关、势函数、旋转引导、窗口重定时）、只评估开关、固定基线（纯 MPC）和各阶段（种子、步数、评估开局）。入口 `experiments/final.py` 只启动这里登记的内容。

- 规格：`ROUND2.md`（清理、方法收敛、奖励）；`dev` 阶段用于反馈机制开发，变体随其规格登记；正式实验阶段在正式实验设计中加入。
- 方法：`docs/paper/facts/`；命令：`docs/engineering/REPRODUCE.md`。
- 第一稿（第一轮）注册表：`docs/archive/registry/METHOD_REGISTRY_ROUND1_20261119.json`；开发期注册表：`docs/archive/registry/METHOD_REGISTRY_20261111.json`。
- 目录名沿用 `final_method` 是因为运行的 manifest 记录注册表路径与 SHA-256。
