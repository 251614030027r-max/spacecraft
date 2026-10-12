# T1种子0的5k工程健康检查

科学提交f36ac77；检查时间2026-10-12T23:02:59.494106+00:00；六项工程门：全部通过。只判工程健康，不以成绩决定启动。

资源速度阈值2.824797s/决策；M0=2.303847、M1=2.778156。无ABORTED，两模型参数有限，当前决策均超过5k；两个5k检查点重载动作有限；两个开发评估各16回合；完整行快照存在终止事件及决策增长。

M1辅助目标证据：{"updates": 3999, "nonzero_updates": 2000, "first_nonzero_decision": 3000, "max_aux_loss": 3.9033150672912598, "valid_decisions_through5000": 2629}。

## 描述性跟踪（不作门槛）

训练快照取检查时完整行，已超过5k，计数见JSON；不能把该快照当作恰好5k训练统计。5k开发确定性评估两行均3/16，非独立正式性能。成绩不改变冻结方案。

### M0

快照决策8259、回合74。开局择时（分位0/25/50/75/100）：{"a0": [-0.999, -0.401, 0.859, 1.0, 1.0], "tau_request_s": [0.1, 47.9, 148.7, 160.0, 160.0], "tau_effective_s": [50.4, 100.0, 148.7, 160.0, 160.0], "distinct_effective_tau": 36, "earliest_requests": 4}。
计划状态占比：{"optimal": 0.7247850829398232, "hold": 0.12095895386850708, "descent": 0.10376558905436493, "kept": 0.05036929410340235, "optimal_inaccurate": 0.0001210800339024095}；顺延占比0.174234。失败类型：{"distance": 29}；终止阶段：{"timeout": {"far_field": 19, "near_field": 4, "entry": 1}, "unsafe": {"far_field": 29}, "success": {"near_field": 21}}。

### M1

快照决策6870、回合60。开局择时（分位0/25/50/75/100）：{"a0": [-0.999, -0.452, 0.702, 0.997, 1.0], "tau_request_s": [0.1, 43.8, 136.1, 159.8, 160.0], "tau_effective_s": [34.1, 89.9, 136.6, 159.8, 160.0], "distinct_effective_tau": 33, "earliest_requests": 4}。
计划状态占比：{"optimal": 0.6864628820960699, "hold": 0.1438136826783115, "descent": 0.11703056768558952, "kept": 0.05254730713245997, "optimal_inaccurate": 0.0001455604075691412}；顺延占比0.226201。失败类型：{"distance": 20}；终止阶段：{"timeout": {"far_field": 16, "near_field": 4}, "unsafe": {"far_field": 20}, "success": {"near_field": 20}}。
有效辅助率按决策类型：{"request_vs_earliest": {"n": 511, "open": 14, "open_share": 0.0273972602739726, "reasons": {"alt_saturation": 435, "alt_not_tracking": 389, "same_effective_tau": 383, "real_not_tracking": 381, "e_p": 282, "e_o": 227, "e_r": 174, "alt_mode_switch": 46, "real_mode_switch": 46, "real_terminated": 10, "near_horizon": 3}}, "replan_vs_keep": {"n": 4722, "open": 3050, "open_share": 0.6459127488352393, "reasons": {"same_effective_tau": 919, "alt_mode_switch": 362, "real_mode_switch": 362, "alt_saturation": 324, "real_not_tracking": 152, "e_r": 120, "e_o": 95, "e_p": 64, "real_terminated": 18, "real_slack": 16, "alt_slack": 14, "near_horizon": 10}}, "keep_vs_shift": {"n": 311, "open": 236, "open_share": 0.7588424437299035, "reasons": {"real_mode_switch": 32, "alt_mode_switch": 31, "alt_saturation": 22, "same_effective_tau": 21, "e_r": 11, "alt_not_tracking": 9, "e_o": 6, "e_p": 4, "near_horizon": 1}}, "earliest_vs_later": {"n": 472, "open": 9, "open_share": 0.019067796610169493, "reasons": {"real_not_tracking": 436, "same_effective_tau": 433, "alt_not_tracking": 433, "alt_saturation": 355, "e_p": 288, "e_o": 219, "e_r": 180, "alt_mode_switch": 36, "real_mode_switch": 36, "real_terminated": 7, "near_horizon": 3, "alt_slack": 2, "real_slack": 2}}, "not_representable": {"n": 50, "open": 0, "open_share": 0.0, "reasons": {"not_representable": 50}}, "descent_mode": {"n": 804, "open": 0, "open_share": 0.0, "reasons": {"descent_mode": 804}}}。保持计划与改变/开始计划分别列出：{"all_decisions": 0.4816593886462882, "policy_keeps_plan": 0.7588424437299035, "policy_changes_or_starts_plan": 0.5386503067484663}。不能把途中计划管理改善写成开局择时的学习优势。

全部原始完整行快照（JSON数组）、stats、5k逐开局评估、心跳与M1检查点辅助统计在health_5k_seed0/；模型只列SHA不上传。N=2，健康门通过只是允许后续队列，种子1/2仍须等现有一整对训练空位；不停止健康现有训练。
