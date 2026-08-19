# S1 stable run analysis — 2026-08-13

`phase2_mission_s1_stable_seed260813` manifest为`completed`，requested/actual均250,000；final model、250k checkpoint/evaluation、monitor、diagnostics和TensorBoard齐全。因此它不是可续训的240k中断产物。

50k至250k五次周期evaluation的Gate acquisition均为0。固定30-seed 260900–260929：Gate 0/30，speed failure 29/30，premature entry 1/30；minimum Gate distance均值9.246 m、中位数9.222 m、最好3.583 m；动作饱和率均值0.110。

数值稳定化有效：probe Q1/Q2最终10.47/10.54，actor std 0.262，探针饱和0.240；TensorBoard最终critic loss3.56、actor loss-17.19。失败不再来自critic发散。

reward分解显示终止捷径：初始Gate距离均值10.644 m，终止Gate距离均值10.921 m；累计progress均值-8.51、state cost-23.03、speed warning仅-0.193、失败event固定-10。策略在Gate尚远时把速度推到0.50 m/s边界，以提前结束避免继续累计state cost。

下一配置保持SAC优化和任务不变，仅把state weight降至0.005、speed warning weight提高到0.20、Phase-I failure event调整为-15。下一运行必须从零开始：`phase2_mission_s1_rewardfix_seed260813`，250k，seed 260813；不得续训本次actor或进入S2。
