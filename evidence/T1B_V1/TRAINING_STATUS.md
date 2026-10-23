# T1-B V1真实训练状态

2026-10-19T15:31:26.867101+09:00

|臂|实际在线转移|critic更新|Actor更新|状态|heartbeat PID|
|---|---:|---:|---:|---|---:|
|e_sac|5000|4500|1500|finished|21096|
|e_tcpf|5000|4500|1500|finished|22436|
|e_sac continuation|10000|9500|3166|finished|16068|
|e_tcpf continuation|10000|9500|3166|finished|13124|

专家数据另计1000真实转移，两臂同字节Replay，采样目标20%，不计入在线5k；无BC或预训练。共同learning_starts=500、每在线真实转移1次critic、每3次转移1次Actor；128×128，gamma=.999，alpha=.005×gap，Cosine LR1e−4→1e−5。检查点callback在当次4转移写入/更新之前保存，与final另列。阶段 matched_10k_complete_readout_required；原生进程命令和启动/退出在过程STATE及EXIT回执。
