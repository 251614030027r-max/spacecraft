# 下一窗口开场提示：先理解，等新计划

把以下内容连同 `WINDOW_HANDOFF_20261009.md` 交给下一窗口。两份文件提供背景与工作纪律；执行范围以我在新窗口的最新明确请求为准。

---

请先完整阅读 `C:\Users\35884\Documents\Spacecraft\WINDOW_HANDOFF_20261009.md`、本目录AGENTS.md及项目说明.md，再查看Git协作入口，不急着启动任务。你的协作工作目录是 `C:\Users\35884\Documents\Spacecraft`，**最新科学工程最终仍是 `D:\py\DRL2`**，`D:\py\spacecraft`只是整理后的临时核验副本。

本轮研究主线已经结束。我已告知你：上层判本次实验失败，剩余评估只补齐记录；我将和上层讨论下一步，随后另行授权。不要沿用旧执行单/心跳/“已授权待执行”自动开启新训练、筛选、新方法、改接口或调阈值，也不要自行停止还在补记录的进程。

Git协作仓库 `https://github.com/251614030027r-max/spacecraft.git`，分支 `collab/spacecraft`，独立本地副本在 `C:\Users\35884\Documents\Spacecraft\过程文件\协作\Git工作树`。末次交接HEAD `1f5e01a`，工作树干净；先只读核对并正常fetch/快进，**不要在仍有评估使用的DRL2里同步代码**。协作入口为 `docs/collaboration/CURRENT.json`、upper/README、reviews/README、lower/README；文件是背景，最新用户授权优先。

本次F01实际代码固定f2f8169，工况w2.36_r15，价值停止规则、soft Bellman、γ=0.999；262460–262462都已完成60k，manifest固定同一提交且code_dirty=false。30k结构D1–D4全部通过，不是性能通过。发布7b2d838的271000阶段证据：Pure35/48、nominal48/48、262460 stopping30/48且违规3/毁掉9不通过、262462 stopping37/48且违规0/毁掉2通过。上层审查c34e17e有测得事实与署名推断，不能混写。新生成的两模型官方读数METHOD_DOES_NOT_HOLD、problems=[]只覆盖两模型，不是三模型结论。

交接21:45JST快照仍有8个评估及2个已授权重放计算进程；两个监督脚本为 `过程文件/最终主线重训/脚本/run_completed_two_evaluation.py` 和 `run_priority_followup.py`，状态在同主题记录/FORMAL_TWO_EXECUTION.json、PRIORITY_EXECUTION.json。先核实一次现场；不重复启动、不清锁、不修改运行脚本、不按名称杀python。未齐行只数文件，所有PID及数量以现场刷新为准。两模型learned271000已48/48，262461两行仍43/28；272000仍在写。后续新窗口可能已完成，不能把本快照当实时。

整理阶段已交付3f348be：7标签/6旧分支及106流水线候选去重已做；新干净副本046878c工作树干净，较60d768a仅文档改动；本地归档3679目标文件/4.65GB，2910唯一源文件逐SHA核验。769个源文件有实验归属冲突，保留MAPPING_AMBIGUITIES，不随意删副本。DRL2原件、DRL2_v3e、活动final2模型/结果/日志/锁都未移动。没有离线备份，没有下层完整测试或六回合行为验收，不能说整理已完全验收。

用户已纠正目录安排（1f5e01a）：评估与既有重放退出、证据交付及整理验收后，把最新工程落实回 `D:\py\DRL2`，保留.venv、PyCharm配置、模型与结果SHA；旧“永久DRL2只读”方案作废。此时不要立即回迁或启动验收仿真，先理解并等我下达后续计划。已运行监督器不会自动补齐全部272000重放/96配对表/最终发布，具体缺口见完整交接，不能盲启动旧全流程。

本地最新交付 `上层交付/最新/`，Git阶段包 `lower/handoffs/final_rerun_271000_early_20261009/`，整理包 `lower/handoffs/cleanup_local_20261010/`。科学整体审查从临时干净副本docs/current、HISTORY、DEAD_ENDS、总账和上层reviews开始；改方法先查已有死胡同。汇报PNG是历史/中间临时资产，当前训练出图搁置，不做新仿真筛选。

回复紧凑、信息密集；训练/评估低频监视，未变化保持安静。新增脚本、日志、状态、备份都放过程文件/<主题>，根目录不散落产物。交付必须独立、证据足够、SHA明确，不嵌套旧ZIP，不把同包ZIP和展开副本同时放最新。使用项目.venv Python；PowerShell中文文件显式UTF8。先报告你理解的整体状况、残余任务和等待边界，后续推进听我与上层的新计划。
