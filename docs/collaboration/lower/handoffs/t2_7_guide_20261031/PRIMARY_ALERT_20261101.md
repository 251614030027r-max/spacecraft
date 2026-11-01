# T2.7 即时主比较问题提示（先供上层判断，评估不停）

2026-11-01。用户要求立即推送已有证据，不等待辅助评估完毕。科学冻结2208a51不变。完整EACC+引导384例与复用SAC384例主比较已齐；未修改的原生t27工具输出的主比较预览为 **harmful**。这不是全轮交付完成声明，SAC+引导辅助评估继续。

|原生门槛|结果|
|---|---|
|C0_complete|true|
|M_mechanism|true|
|E1_efficiency|false|
|U_urgent|true|
|E2_final|true|
|S_safety|false|

|种子|四点平均EACC+引导−SAC|上一轮无引导EACC−SAC|4000步差|四点unsafe EACC+引导/SAC|紧急捕获累计差|
|---|---:|---:|---:|---|---:|
|0|−5.742510|−8.212150|+0.005701|4/0|0|
|1|+10.307117|+19.685933|+8.782994|1/0|+6|

效率未过：种子0为负，且两种子平均约+2.28低于+5。安全未过：种子0四检查点累计4次unsafe，超过SAC+1；这是评估次数，不能冒称4个独立场景。4000步无unsafe不抹掉早期记录。

优先核查seed0的unsafe开局与检查点分布。SAC+引导尚未评估完整，不能提前判定问题来自引导本身或标签增量，更不能立即改机制。继续完成既定C_guide_only/C_labels/C_vs_eacc，不增加实验、不挑检查点。本次主比较原始逐例JSON、原生预览、四条训练的manifest/finished/episodes/model_stats/heartbeat及SHA256SUMS正在独立推送至科学分支evidence/T2_7_GUIDE/interim_20261101/；协作即时完整报告放interim_eval_20261101/。原始TensorBoard及外部30秒滚动日志留在本机。
