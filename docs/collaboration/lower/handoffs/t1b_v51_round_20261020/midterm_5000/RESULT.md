# T1-B V5.1 actual research readout

RUNNING; 2026-10-20T15:47:39.293505+09:00; scientific freeze b6f5c8a22728b6ca3970d6201525cd4047cba8d8.

用户最新范围：资源优先seed0两臂各20k效果验证；seed1仅作为有证据需要时的条件复核，不自动排队；seed2取消，不做三种子最终出图。预留一个物理核（逻辑CPU0/1）、BelowNormal。每臂仍4环境，算法/奖励/安全/任务/Gstar/MPC/初值/步数全冻结。实际活动运行与排队种子见STATE/TRAINING_STATUS及RESOURCE_ADMISSION。

|运行|持久化真实步|critic更新|Actor更新|M|
|---|---:|---:|---:|---|
|e_sac_s0|19000|18496|6165|True|
|e_tcpf_s0|5500|4996|1665|True|

|seed|checkpoint|arm|block|complete|captures|mean original discounted return|non-earliest first|no learning first|
|---|---:|---|---|---:|---:|---:|---:|---:|
|0|2000|e_sac|292140–147|8/8|7|36.538164132|0|0|
|0|2000|e_sac|292120–127|8/8|6|16.220946479|0|1|
|0|2000|e_tcpf|292140–147|8/8|7|36.538164132|0|0|
|0|2000|e_tcpf|292120–127|8/8|6|16.220946479|0|1|
|0|5000|e_sac|292140–147|8/8|7|36.538164132|0|0|
|0|5000|e_sac|292120–127|8/8|6|16.220946479|0|1|
|0|5000|e_tcpf|292140–147|8/8|7|36.538164132|0|0|
|0|5000|e_tcpf|292120–127|8/8|6|16.220946479|0|1|
|0|10000|e_sac|292140–147|8/8|7|36.538164132|0|0|
|0|10000|e_sac|292120–127|8/8|6|16.220946479|0|1|

V5.1 common0k main: 8/8 completed, captures6, mean=13.715624173; partial blocks are not full estimates.

V5.1 common0k expanded: 8/8 completed, captures6, mean=18.001071055; partial blocks are not full estimates.

必要历史背景：V4同任务seed0主块0k6/8、13.715624173，2k和5k两臂均7/8、36.538164132；扩展0k6/8、18.001071055，2k/5k两臂均6/8、16.220946479，逐例相同。V4完整失败证据提交2c7283b，不把V4模型当V5.1模型。固定教师后续真值主块最早7/8约36.54、最佳首选8/8约60.87；扩展6/8约16.38和6/8约18.70。固定教师后续真值不等于当前SAC严格Q；强Gstar历史主块7/8、37.436580681，扩展7/8、39.961325235仅作背景而非同条件消融。

逐状态Q1/Q2/概率/梯度/熵，15个三候选初态Q最佳与真值最佳一致数、经济/急迫首选分布及每档Q斜率见POLICY_AND_GATE_AUDIT。每seed、每stage、同开局TCPF−SAC和三seed均值见THREE_SEED_COMPARISON；本轮至多两个种子，三种子均值字段不适用且保持null，不以此要求加训。单seed仅说明本次效果，不能替代最终统计结论；技术/M/非零梯度不等于科研成功。

Full-horizon MPC配对原始payload、真实区间回报与事实模型回报误差、拒绝原因、两槽接受率及是否两侧终止见MPC_FULL_HORIZON_AUDIT。门控实际冻结为同终止事件、终止步差<=10、回报误差<=5；当两侧终止时反馈Y为模型成对回报差，不含critic自举。CF_LABEL文件仍是冻结检查点对持久化配对的复算，不冒充历史每次更新原标签。开发初态的模型排序验证原始证据在科学b6f5c8a的evidence/T1B_V5_PRECHECK，最佳一致11/15、两两方向31/38；它是预检而非本轮训练独立增益。
