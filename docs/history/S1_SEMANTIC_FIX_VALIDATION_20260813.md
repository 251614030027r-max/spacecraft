# S1 semantic-fix validation — 2026-08-13

上层指导文件的判断与当前源码/rollout一致：0.50 m/s硬终止、仅在0.45–0.50 m/s出现的窄warning、持续绝对state cost及无界quadratic progress共同形成了错误终止经济。本轮没有改变Gate、初态分布、truth、tumble、Phase-II约束或SAC本体。

实现仅改变Phase-I语义：0.50 m/s不再终止；0.25 m/s起的soft speed cost为`-((v-0.25)/0.25)^2`，1.00 m/s仅作catastrophic guard；Phase-I state penalty固定为0；四类归一化误差使用`x²/(1+x²)`组成最大2.0的bounded potential，progress使用`2*(V_prev-0.997*V_next)`。Phase-II保留原reward路径。

exact-truth可达性验证使用简单受限target-frame位置/姿态PD，严格经过环境±5 N、±0.6 N·m action接口；固定seed 262000–262019 Gate 20/20，平均到达60.13 s，最大速度0.19814 m/s，平均动作饱和率0.01521。Gate入口位置误差均约0.993–1.000 m，姿态误差约0.0003–0.0010 rad，速度低于0.20 m/s，证明当前任务与sampler无需调整。

reward preference使用seed 262000–262002，对每个相同初态运行Gate-PD、zero和高速失败策略，并按SAC gamma 0.997累计environment return。Gate discounted return为2.55–3.05，高速失败为-234.46至-316.54，逐seed最小`G_Gate-G_high-speed`余量237.01；zero为-177.37至-336.64。基本排序明确通过。

代码回归：与本次语义/接口直接相关的39项pytest正常退出并通过；全量82项连续三次运行均显示全部用例到100%且无失败，但Windows会话中的CVXPY/pytest进程在测试结束后未退出，分别在180/360 s工具上限触发超时，因此不虚构正式`82 passed`退出码。canonical训练入口validate-only正常退出通过。

验证原始数据保存在`logs/phase2_mission_s1_semanticfix_validation/phase1_semantic_validation_20_seed262000.json`。下一动作是用户从零启动250k S1；50k只做结构性捷径检查，不作为最终能力结论。
