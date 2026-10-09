# Spacecraft 上下层协作入口

> **所有新窗口第一份要读的是 [HANDOFF.md](HANDOFF.md)**（三窗口共用交接：背景、现状、各窗口职责与纪律）。每个窗口结束前更新自己那一节。

固定分支：`collab/spacecraft`。本目录是任务交换与交付索引，正式实验只用各执行单冻结的科学提交；同步本分支不会同步实验工程。

|用途|入口|主要维护者|
|---|---|---|
|**项目当前状态（唯一入口，只记事实）**|[PROJECT_STATE.md](PROJECT_STATE.md)|上层|
|实验总账（全部实验的状态、证据、论文角色）|[registry/](registry/README.md)|上层预填，下层补本地列|
|当前阶段、代码版本与停止边界|[CURRENT.json](CURRENT.json)|下层核验后更新|
|上层执行单、预注册与审批记录|[upper/README.md](upper/README.md)|上层|
|下层已完成工作、交付与证据|[lower/README.md](lower/README.md)|下层|
|已完成结果的审查|[reviews/README.md](reviews/README.md)|上下层分别写署名报告|
|协作规则、接续方式|[WORKFLOW.md](WORKFLOW.md)|用户决策优先|

**当前状态的唯一入口：[PROJECT_STATE.md](PROJECT_STATE.md)**，包括正在运行的工作、资产位置、Git 结构和未完成事项。实验总账见 [registry/](registry/README.md)。本段之下的旧描述与历史执行单仅作历史资料。

上层以后把执行单提交到本目录upper，下层把完成状态与交付入口提交到lower并更新CURRENT；每次工作先同步这个分支。用户只需指示“读协作入口并推进已授权任务”，无需转发长文或下载再上传结果。

完整大ZIP、模型和日志原件在本地，Git上传够审查的报告、结构化结果、必要逐开局证据与哈希。历史审查使用固定提交链接，不重复复制，不改写已冻结报告。
