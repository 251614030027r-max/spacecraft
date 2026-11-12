# V7 SIDE BENCH 阶段读出

2026-10-21T12:54:12.899522+09:00；phase=complete。A使用54c2587，不切换活动训练检出；B仅在全部54c2587训练与读出退出后切6fa8be8，已验证仅新增离线脚本。待完成项不外推，不用于本轮判定或调参。

确认块292500–292531：真值已出现32/32个开局，模型32/32。学习空间按任务分类见SIDE_ASSESSMENT，逐开局见CONFIRM_LEARNING_SPACE.csv；只有完整32例才能报该块整体结论。

标签准确度：{"candidate_event_comparisons": 93, "event_mismatch": 0, "success_failure_flips": 0, "all_pairs": 90, "direction_n_abs_truth_delta_gt05": 67, "direction_agreement": 63, "absolute_pair_error_median": 0.13993142089557153, "absolute_pair_error_p95": 2.269787133227497, "quantile": "linear interpolation at (n-1)*p; shared-state pairs are not independent samples"}。模型与真值均固定教师后续；线上策略后续可能不同，不等于当前SAC严格Q。

|块|方法|检查点|完成|捕获|平均原回报|首选与真值最佳一致|
|---|---|---:|---:|---:|---:|---:|
|confirm_292500|e_sac|5000|32/32|25|20.959474267|10|
|confirm_292500|e_sac|10000|32/32|25|20.959474267|10|
|confirm_292500|e_tcpf|5000|32/32|23|10.135096654|11|
|confirm_292500|e_tcpf|10000|32/32|23|10.097273232|12|
|indep_293100|e_sac|5000|32/32|24|16.683096591|13|
|indep_293100|e_sac|10000|32/32|24|16.683096591|13|
|indep_293100|e_tcpf|5000|32/32|23|12.013127082|17|
|indep_293100|e_tcpf|10000|32/32|23|10.079709772|16|

独立293100–293131仅在10k后报告，不参与M/R/种子1判定。离线B结果见bench_*.json：每个模型三种损失各3000次，三对梯度余弦中位及10/90分位；不新增真实样本、不改在线模型。5k模型明确使用训练结束的最终Replay，是离线诊断而非5k同期表现；Replay保存于callback，可能较模型时钟差最后4条在线转移（含专家前缀1000条），不拼接、不恢复训练。各bench_PHASE回执保留。

必要历史：V6单seed10k主块SAC36.538164132/TCPF36.590001135、均7/8，扩展均16.220946479、6/8；经济型首选回最早，未满足种子1。旧开发块的最佳教师首选收益约95%集中292140，因此新块与剔除该例的读出必要。不同块不能直接比较均值来归因。技术通过、梯度非零或离线训练曲线不冒充在线科研收益。


# 292126专项审查（只读）

冻结54c2587；评估伪影=False；seed1可放行=True。

TCPF记录的学习决策：[{"time_s": 0.0, "candidates_s": [100.0, 120.0, 140.0], "selected_index": 1, "has_plan": false}, {"time_s": 80.00000417232513, "candidates_s": [50.0, 70.0], "selected_index": 0, "has_plan": false}]

实际后续非最早学习决策数量=0。不得把假设写成恢复策略泛化事实。

旧固定教师扫描=-137.08880943392987 / timeout；V7接口固定首选中间+教师后续重放=45.60450548864637 / success。历史接口/后续执行差异需与本轮同接口证据分开。

两臂10k完整原始轨迹来源SHA、复现实测、每个决策的5k/10k TCPF Q1/Q2/概率/Actor和事件、预算曲线均随本目录提供；宏步原始曲线来自float32观测反归一化，复现曲线是实际time_s/_delta_v。


293100独立块模型标签审计（仅完整块报告总体结论）：{"model_openings": 32, "truth_openings": 32, "independent_block_complete": true, "nontrivial_pairs": 68, "direction_matches": 68, "absolute_pair_error_median": 0.08292050487258962, "absolute_pair_error_p95": 1.2029445855876897, "event_mismatches": 1, "success_failure_flips": 0, "only_report_no_tuning": true}

# 292126 归因核实

{
  "deterministic_eval_reproduced": true,
  "tcpf_recovery_non_earliest_count": 0,
  "same_interface_teacher_reproduces_tcpf": true,
  "conclusion": "Within V7, same first middle intent followed by earliest recovery reproduces TCPF; no evidence of learned non-earliest recovery benefit.",
  "historical_scan_is_not_matched_V7_comparison": true,
  "remaining_uncertainty": "Historical fixed-teacher reference has not been independently rerun under its historical interface; do not assign all differences to a specific cause."
}

10k首审Q最佳为最早、Actor为中间；5k首审Q与Actor均为中间。80秒恢复处两模型Q与Actor均为最早。Q/Actor不同步风险仍存在，不能据成功宣称稳定学习。


# V7 D：失败与有益偏离状态的Q–Actor审计

冻结54c2587；seed1按用户指令废弃，未改变算法。D在A全部完成后读取冻结模型，不更新权重。

分类按候选真值回报的最佳集合（并列容差1e-8）判定top1，完整min-Q排序、候选逐项数值另附。‘两者都错’明确指MPC和Q，三个正确性布尔字段保留，避免含混。对照态允许‘三者均对’，缺恢复态匹配参照必须标证据不足，不把初态真值套到恢复态。固定教师后续真值是诊断参照，并非当前策略严格Q。

本机纠正：292526首选最早且成功，条件失败案例不成立；293122实际V7轨迹初审有三个候选[100,120,140]，非单候选。具体有无后续学习决策以D_DECISION_STATES为准。

10k TCPF初审分类：{"Q 对、Actor 错": 1, "MPC 对、Q 错": 4, "三者均对（对照）": 2}

|开局|MPC最佳|Q最佳|Actor|分类|
|---|---|---|---|---|
|292522|1|1|2|Q 对、Actor 错|
|292530|1|2|2|MPC 对、Q 错|
|293122|0|2|1|MPC 对、Q 错|
|292505|2|0|1|MPC 对、Q 错|
|292513|1|2|2|MPC 对、Q 错|
|292515|1|1|1|三者均对（对照）|
|292524|2|2|2|三者均对（对照）|

