# 项目总览

## 研究目标

目标是建立可审计的高保真翻滚非合作目标预捕获基线，并判断标准Pure SAC在两阶段任务中能否学习：先在目标坐标系外侧获取安全Gate，再在视场、走廊、速度和执行器约束下完成终端接近。当前工作不是宣称复现成功，而是逐步排除任务语义、坐标表达、奖励shortcut和数值稳定性问题，定位剩余学习瓶颈。

## 系统结构

```text
dynamics/        SE(3)、相对状态、刚体动力学、J2、RK45
env/             sampler、24D观测、reward、termination、两阶段环境
controllers/mpc/ 约束MPC配置、线性化、预测、代价与约束
train/           Pure SAC配置、训练入口、诊断和固定种子评估
eval/            manifest驱动的deterministic策略评估
experiments/     MPC评估入口
tests/           动力学、环境、任务、reward、MPC和训练回归
logs/            manifest、monitor、TensorBoard、诊断和周期评估
models/          仅保留四个有解释价值的代表模型
legacy/          早期复现材料，不参与当前import
```

## 可信边界

- 高保真truth、Gate和Phase-II约束均有独立低成本验证；
- 每轮正式训练保存完整config、软件版本、固定seeds和数值诊断；
- 训练中的偶发成功不能代替deterministic独立seed成功；
- 当前所有deterministic独立20-seed Gate结果仍为0，因此没有合格模型；
- 代表checkpoint用于比较控制结构，不用于S2初始化；
- 论文和早期材料只作为相关基线，任何偏离均不冒充paper-literal复现。

## 当前最重要发现

body-frame平动表达解决了明显的state/action坐标错位；tau从0.005降到0.001显著抑制了旧实验中Q/loss膨胀；S1-v2任务重定义首次产生广泛Gate-directed approach；把实际速度改为desired-velocity tracking error又显著提升低速率。但空间推进和速度控制目前呈此消彼长，且两种表示都在中后期遗忘。这是后续审查最需要聚焦的问题。
