# T1根因诊断第一阶段：本机执行回执

执行单collab@7b82bab；本机执行日期2026-10-14（日本时间）。状态：RAW_RESULTS_READY。本轮只交付执行事实与原始数据，P1–P3解读由上层完成。

- 科学检出：ca82c637c0110e96910d87e3302f4fca18e03215；env/train/experiments/t1_dev_eval.py相对78eb63c逐字节无差异。原DRL2@4795910不动，未训练、未改代码或参数，未访问正式块。
- Python D:/py/DRL2/.venv/Scripts/python.exe -B；OMP/MKL/OPENBLAS线程均1。指定测试7项通过，完整命令、stdout/stderr和耗时附console_and_receipts/。
- 检查点0k/20k/30k SHA与前轮MODEL_SHA256逐项一致，详见_diag/model_sha.txt及MODEL_SHA_CHECK.json；模型不打包、不上传。
  - m1_0_decisions.zip: `d6d60f39a9dc82dcbbbf883f82a342e33c8e4941de69ce3324cd2b6f3396b0e1`
  - m1_20000_decisions.zip: `6fbce2d9b7a6a7fa5c928f123b81b98739dd2d6b990de366519a03bdba578ce2`
  - m1_30000_decisions.zip: `e62f9b95575fc760479d23dbb61ac9c0dabfa6913fef4cad82375172d5f7aec5`
- 30k A/295001试跑242.664403s、150决策，1.617762686s/决策，9500决策估算4.269096 CPU·h，门限6.5，通过。
- 原策略A抽查：{"30k": {"event": true, "return_bitwise": true, "time": true, "delta_v": true}, "20k": {"event": true, "return_bitwise": true, "time": true, "delta_v": true}, "0k": {"event": true, "return_bitwise": true, "time": true, "delta_v": true}}。未抽查标记不补写通过。
- 实际输出：P1 59个JSON；P2/P3 10个状态JSON。执行单成本口径为各输出wall_s之和/3600，实测6.739974小时；该数含并行争用/冷启动，为任务指定的统计口径，不是操作系统累计CPU时间。命令级墙钟另附回执。
- 成本记录异常：上述累计wall_s口径超过6.5小时；初始单回合估算通过成本门后仅执行了固定第一阶段，未扩展任何样本。累计争用墙钟与OS累计CPU时不可混称。
- replay_check.match=false：无；存在false也照单保留交付，没有改变状态选择。
- verify_branch原始结果（事件、结束时间、折扣回报、Δv的分支值/直接续跑值/逐位相等标记）：
  - 20k_295001_k2: 逐位相等{"event": true, "time_s": true, "discounted_return_from_k": false, "delta_v_m_s": false}；绝对差{"time_s": 0.0, "discounted_return_from_k": 0.0010698200652257128, "delta_v_m_s": 0.00023235481816974435}。完整原始值见EXECUTION_SUMMARY.json。
  - 30k_295001_k2: 逐位相等{"event": true, "time_s": true, "discounted_return_from_k": false, "delta_v_m_s": false}；绝对差{"time_s": 0.0, "discounted_return_from_k": 0.0005957205359550244, "delta_v_m_s": 0.00013747044452516732}。完整原始值见EXECUTION_SUMMARY.json。
- verify_branch中逐位不等的浮点项不改写为通过、不做原因归因；原始差值完整保留。未运行额外状态、随机后续策略、16开局扩展或受控训练。
- 异常/未完成：无执行报错；分支与重放结果按上述原样列出；最终读出只用已授权范围。

独立ZIP包含整个_diag（p1、p2、diag_readout、model_sha）、控制台与回执、同机复用A/E原始JSON及SHA256SUMS。模型只列SHA。原日志/模型未删除或移动，旧轮交付在本机历史中保留。本轮不把P0已证实事实、机制假设与P1–P3结果混写为新根因结论。
