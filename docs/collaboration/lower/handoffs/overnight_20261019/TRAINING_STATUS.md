# 真实训练状态

快照UTC：2026-10-18T20:26:29.326256+00:00。DEV技术冒烟不计为正式训练。

已实际启动同seed对照；以以下真实步数和检查点为准。

|臂|真实步数|critic更新|Actor更新|真实解释器PID|参数有限|有效模型区间/真实封账区间|
|---|---:|---:|---:|---:|---|---|
|r3|700|0|0|10376|True|0/261|
|tcpf|380|0|0|25520|True|63/137|

相同真实步数的固定开发评估（292120–292127；正式300000–300047未访问）：

|臂|检查点|已完成/8|捕获|平均折扣原回报|
|---|---:|---:|---:|---:|
|r3|0|8|6|17.132155|
|tcpf|0|0|UNFINISHED|UNFINISHED|
|r3|2000|0|UNFINISHED|UNFINISHED|
|tcpf|2000|0|UNFINISHED|UNFINISHED|
|r3|5000|0|UNFINISHED|UNFINISHED|
|tcpf|5000|0|UNFINISHED|UNFINISHED|
|r3|10000|0|UNFINISHED|UNFINISHED|
|tcpf|10000|0|UNFINISHED|UNFINISHED|

2k为共同暖身终点，尚无正式梯度学习；两臂0k/2k权重应一致。0k→5k/10k才检验训练增长，TCPF−R3才检验反馈独立收益。单种子、开发集结果不能称最终三种子或正式独立优势。

原生运行状态：记录/PAIRED_TRAINING_STATUS.json；评估状态：记录/CHECKPOINT_EVALUATION.json；全部原始逐步、终止事件及检查点保留。
