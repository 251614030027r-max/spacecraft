# 阶段汇报基准图（临时、待审查）

TEMPORARY_REVIEW_ONLY / NOT_PAPER_PERFORMANCE。执行单85bc7a8，五项共七张图；仅已有资产，无新实验，训练不动。历史正式与新方法中间案例分开注明，不能混成当前方法已成立。

- [1 · V3e历史学习层训练](fig1_v3e_training.png)
- [2 · 原工况正式完成数/能耗/时间](fig2_formal_comparison.png)
- [3 · 新工况Pure与nominal](fig3_baseline_time_energy.png)
- [4 · 事后交接时机价值](fig4_hindsight_timing.png)
- [5a · 已有50k案例轨迹](fig_trajectory_3d.png)
- [5b · 案例闭环与捕获阈值](fig_state_response.png)
- [5c · 案例价值触发](fig_handoff_value.png)

[核对数字](CHECKS.md) · [科学支撑与限制](REVIEW.md) · [实际绘图CSV](plot_data/) · [源文件及SHA](INPUTS_SHA256.json) · [脚本复用](REUSE.md) · [全文件SHA](FILES_SHA256.json)。

按真实learned违规1/0/1绘制已获用户明确授权。当前final2混合训练曲线剔除，旧图仅历史留存。后续更好的正式资产可替换本入口，不能抹掉不利证据。

上层已在c4fecf2同步更正图2真值违规：Pure0、stopping0/0/0、learned1/0/1，与本包及用户明确确认一致；RUN_ORDER.md保存该更正版。

2026-10-09格式更新：按用户最新指令仅保留高清PNG，移除PDF/SVG；科学数据、脚本及审核证据保留。归档执行单中的矢量格式要求由本次用户指令取代。
