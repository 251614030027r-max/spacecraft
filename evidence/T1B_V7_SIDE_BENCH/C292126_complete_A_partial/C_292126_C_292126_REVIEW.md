# 292126专项审查（只读）

冻结54c2587；评估伪影=False；seed1可放行=True。

TCPF记录的学习决策：[{"time_s": 0.0, "candidates_s": [100.0, 120.0, 140.0], "selected_index": 1, "has_plan": false}, {"time_s": 80.00000417232513, "candidates_s": [50.0, 70.0], "selected_index": 0, "has_plan": false}]

实际后续非最早学习决策数量=0。不得把假设写成恢复策略泛化事实。

旧固定教师扫描=-137.08880943392987 / timeout；V7接口固定首选中间+教师后续重放=45.60450548864637 / success。历史接口/后续执行差异需与本轮同接口证据分开。

两臂10k完整原始轨迹来源SHA、复现实测、每个决策的5k/10k TCPF Q1/Q2/概率/Actor和事件、预算曲线均随本目录提供；宏步原始曲线来自float32观测反归一化，复现曲线是实际time_s/_delta_v。
