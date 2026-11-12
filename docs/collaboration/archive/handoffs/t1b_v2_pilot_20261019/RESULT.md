# T1_B_V2_PILOT_20261019 实测交付

STOPPED / 按门槛或技术失败停止，未调参重跑。时间2026-10-19T18:57:25.958260+09:00。科学冻结a1698acefd0f3f4cf7298438bd842ef63ae0aee0；seed0，E-SAC/E-TCPF各10k开发试跑，无多种子GO。

必要检查：pytest12/12通过（记录/pytest.log）；CONTRACT.passed=true，初值动作最大差0，alpha=.5、epsilon=.10；共享专家完整SHA bf3c24f9f8db7eb156a5ab42835f6faeefee2332efbd59ba7c35ed733f5ffe80，没有重采专家或BC。固定任务/奖励/D/B/G★/MPC/网络/训练器保持上层冻结版本。

|臂|真实全局步|critic更新|Actor更新|A2|
|---|---:|---:|---:|---|
|e_sac|5000|4496|1498|PASS so far|
|e_tcpf|2600|2096|698|PASS so far|

阶段stopped_A1_failed；停止原因{"type": "A1_failed", "arm": "e_sac", "mark": 5000, "evidence": {"pass": false, "original8_entropy_gt_005": 2, "actor_gradient_fraction_of_0k": 0.768436168660113, "baseline_observations": "Original8 gate; expanded8 reported separately; same actual V1 reset observations, unchanged frozen task/support."}, "retune_or_restart_permitted": false}。原生子进程、命令、门槛预登记和结果计数见ASSESSMENT。A1原8首审状态至少4个熵>.05，Actor梯度至少0k的1%；2k/5k任一臂失败即停。A2暖身结束后完整1k窗口非最早≥5%，执行窗口已在启动前固定501–1500、1501–2500等；尾部不足1k只报告，不冒充完整窗口。没有完整窗口/检查点时为PENDING而非通过。


停止时步数若只有heartbeat/完整日志而没有finished.json，仅为最后持久化可核验值；强制停止后未保存的精确优化次数不补造。

|检查点|臂|开发块|完成/8|捕获|平均原折扣回报|
|---|---|---|---:|---:|---:|
|2000|e_sac|292140–147|8/8|7|36.754444891|
|2000|e_sac|292120–127|8/8|6|16.176670965|
|2000|e_tcpf|292140–147|8/8|7|35.582056910|
|2000|e_tcpf|292120–127|8/8|6|15.505362613|

A3当前判定：PENDING；按两块总是最早原回报36.535394500 /16.215477761比较，并要求某开局首次非最早，不用挑选成功回合fuel替代全样本回报。B1需有效pair>0、提前/推迟两侧均出现、on阶段全部探针（含0）的反馈梯度占比中位2–10%；B2只如实报告选择差、与真值排序一致率、逐例配对回报差，不设方向门槛。详细数值见POLICY_AND_GATE_AUDIT与MPC_COUPLING_AUDIT。

必要历史对照：V1两臂2k/5k/10k原块均7/8、35.582056910，扩展10k均6/8、15.505362613；原始cold两块4/8、−34.446131473与6/8、17.132155286；强全程G★均7/8，回报37.436580681与39.961325235。不能把接口/专家收益计作RL或MPC独立收益。原块上层24候选真值采用固定V1教师后续，最佳首次8/8、约60.87，最早7/8、约36.50；扩展24例本机真值已完成24/24，全部候选/失败/skip均保留，不把这种教师后续回报称为当前SAC严格Q。

最终科学要求未由工程检查或非零梯度自动成立；本轮为单seed开发判断，不称跨seed统计成功。A1/A2失败不调参重跑，A3失败不扩种子，技术失败只允许按单核对原始stderr后从零_restart1，禁止续训。当前无多种子或新的正式训练授权。原始科学产物不搬动，来源C:\Users\35884\Documents\Spacecraft\过程文件\T1B_V2_PILOT_20261019；RAW_EVIDENCE_V2.zip每项SHA核验，未完成项明确缺失，不生成替代数据。