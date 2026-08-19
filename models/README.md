# 保留模型说明

本目录只保留四个用于复核关键结论的模型文件：

- `phase2_mission_s1v2_acquisition_bodyobs_autoent_tau001_seed260815/checkpoints/sac_400000_steps.zip`：空间接近最佳；
- 同目录`final_model.zip`：v2后期遗忘对照；
- `phase2_mission_s1v2_velerror_bodyobs_autoent_tau001_seed260815/checkpoints/sac_200000_steps.zip`：速度控制最佳；
- 同目录`final_model.zip`：v3后期遗忘对照。

这些模型都没有通过独立20-seed Gate acquisition，不能称为合格S1，也不能用于S2初始化。旧任务和其他单因素模型已迁至`legacy/model_history/`，结论与解释证据由`logs/`及`docs/history/`保留。
