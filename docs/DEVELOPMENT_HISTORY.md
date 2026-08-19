# 开发与实验演进

## 1. 早期复现与高保真truth

工程首先整理论文事实、偏差和六自由度SE(3)环境，建立J2、RK45、相对状态、作用力/力矩映射及paper-related复现边界。早期Phase-I训练产生过局部模型，但未通过当前任务和独立指标，现只在`legacy/`保留规范、选定manifest和评估证据。

## 2. 初始Phase-2受约束任务（2026-08-12）

引入corridor、FOV、速度包络和MPC约束，尝试V2.0/V2.1、bounded observation、actor initialization、dense cost、dt reward、gamma和warmup等路线。大量run提前停止或Gate为0，说明一次同时改动多个机制无法给出可信归因。该阶段日志保留在`logs/phase2_sac_v2_*`，详细对账见`history/EXPERIMENT_RECONCILIATION.md`。

## 3. 两阶段mission与语义修正（2026-08-13）

将任务明确分为Phase-I Gate acquisition和Phase-II terminal approach。发现原0.50 m/s Phase-I termination形成错误shortcut，随后移除普通速度终止，只保留1.0 m/s catastrophic guard；Phase-I采用bounded、discount-consistent potential shaping，避免persistent绝对状态惩罚。语义修正消除了接口错误，但标准任务仍未学习Gate。

## 4. 干净Pure SAC控制实验（2026-08-14）

退出adaptive curriculum，固定完整canonical sampler并fresh训练。依次做严格单因素：automatic entropy、local progress、20°方向锥、target/body平动坐标对齐。reward或entropy改动没有解决学习失败；body-frame observation首次让策略形成约1 m级Gate-directed approach，确认平动状态与body-force action的坐标一致性是有效因素。

## 5. Bellman稳定性（2026-08-15）

仅将tau从0.005降至0.001。Q由旧实验约110降至约1、critic loss由约277降至约0.08量级，早期接近能力维持更久，但Gate仍为0并在250k退化。结论是moving target速度影响数值稳定，但数值稳定本身不能产生制动和联合获取。

## 6. S1-v2 acquisition任务（2026-08-15）

旧S1实际上要求宽分布下完整六自由度regulation。S1-v2将初值改为10–14 m、15°方向锥、20°姿态、低速低角速，Gate定义为1.5 m位置球、0.30 m/s总速度和45°FOV，不再硬约束姿态/角速度。v2实际速度实验在400k独立20 seeds达到10/20位置球和16/20明显接近，但仅3/20低速，明确定位主动制动缺失；500k遗忘。

## 7. 显式速度误差观测（2026-08-15至16）

仅把24D中的实际速度替换为与reward一致的active velocity tracking error。200k固定评估首次Gate 1/10，独立20 seeds低速率提升到19/20，但位置球从旧v2的10/20降至0/20；训练Gate数量从14增至33，后400k–500k又归零。速度目标显式化有效但不足，空间推进与制动信用出现明显权衡。

## 当前问题

当前已经排除或显著缓解：任务接口错误、0.50 m/s termination shortcut、课程分布错位、target/body平动坐标错位和快速target update导致的严重Q膨胀。剩余问题是位置推进、速度跟踪和到达保持的联合信用结构，以及前期能力的中后期遗忘。后续实验应继续坚持一次只改一个可解释因素。
