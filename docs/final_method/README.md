# 实验注册表

`METHOD_REGISTRY.json` 是正式实验唯一的配置来源：6 个训练配置（显示名、arm、接口、窗口、旋转引导、认证、actor / critic 视图）、固定策略参照、`formal` 阶段（种子、预算、检查点、条件与开局块、验证曲线、只评估开关、评估清单）和 `formal_prescreen` 阶段（预算预筛，已完成：2.0 m/s，2026-11-13 起作为经济任务在训练与评估中的预算，紧急任务 3.0 m/s）。入口 `experiments/final.py` 只启动这里登记的内容，图表从这里取显示名。

- 方法：`docs/paper/METHOD.md`；实验设计：`docs/paper/EXPERIMENTS.md`；命令：`docs/REPRODUCE.md`。
- 开发期注册表（EACH、U2、ECC 各轮、调节器）：`docs/archive/registry/METHOD_REGISTRY_20261111.json`；开发期方法规格：`docs/archive/method_development/`。
- 目录名沿用 `final_method` 是因为正式运行的 manifest 记录注册表路径与 SHA-256。
