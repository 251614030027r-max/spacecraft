# DRL2：最终纯 SAC Level-C 基线

D:\py\DRL2 是唯一活动代码、模型和日志目录；D:\py\DRL 始终只读。

## 当前状态

纯 SAC 基线已经冻结为一个确定性部署模型：

D:\py\DRL2\models\sac_level_c_v4_300k_s20260817\checkpoints\sac_200000_steps.zip

它由 fresh L0 → L2 → Level-C 逐级 SAC 适配获得。Level-C 阶段使用空经验池，但加载了共同 L2 父模型的 Actor、Critic、目标网络和优化器状态，因此应称为“共享父模型的课程适配纯 SAC 基线”，不是 Level-C 从零训练复现。

## 冻结任务

- 六自由度耦合 SE(3) 刚体动力学、J2、重力梯度和 RK45；无追踪器外扰、无质量惯量不确定性。
- 初始距离2–10 m，球壳体积均匀；初始姿态轴角0–75 deg。
- 惯性相对速度每轴±0.125 m/s；相对角速度每轴±0.03 rad/s；目标翻滚倍率0.50。
- 执行器为每轴5 N / 0.6 N·m；时域200 s；距离失败边界30 m。
- 12维观测为 SE(3) 对数位姿与相对 twist，冻结 L0 scale 和 softsign 语义。
- 奖励冻结为 se3_physical_storage_v4；三个核心项为储能耗散、储能时间积分、归一化控制努力。
- 正式工程成功阈值：姿态15 deg、Lie代数平移误差0.5 m、相对线速度0.1 m/s、相对角速度0.03 rad/s，连续保持5步。

## 最终验收

预先固定的推荐模型在全新统一种子块20275000–20275099上取得：联合成功93%、连续5步完成93%、位置成功95%、距离失败4%、超时3%，独立通过交付门槛。它与此前另一组独立100回合合并后为：联合成功95.5%、位置成功97%、距离失败2.5%、超时2%。

三个适配分支在同一新测试集上联合成功93% / 83% / 82%，但距离失败4% / 15% / 5%，均值8%，说明训练适配过程仍有种子敏感性；该结果不影响已冻结推荐模型通过，但不得宣称任意训练种子都满足相同安全门槛。

10 deg姿态诊断仅7%联合成功、97%位置成功、1%距离失败、92%超时，表明该交付物是15 deg粗交会基线，不是精密姿态捕获或对接基线。

## 交付证据

- 最终验收汇总：D:\py\DRL2\logs\final_baseline_review\final_baseline_acceptance.json
- 统一100回合：D:\py\DRL2\logs\final_baseline_review\common100_s20275000
- 10 deg诊断：D:\py\DRL2\logs\final_baseline_review\recommended_10deg_100_s20275100.json
- 完整说明：D:\py\DRL2\docs\FINAL_PURE_SAC_BASELINE.md
- 审查报告：D:\py\DRL2\docs\FINAL_BASELINE_REVIEW.md

当前模型无需重新训练。后续创新算法必须在完全相同的Level-C任务、15 deg正式口径和固定评估种子下比较，并同时报告多训练种子结果。
