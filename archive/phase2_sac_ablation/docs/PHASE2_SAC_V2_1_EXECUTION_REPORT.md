# Phase-2 SAC V2.1执行报告与有效交接

> 日期：2026-08-12  
> 活动工程：`D:\py\DRL2`（唯一写入工程）  
> 上层指导：`C:\Users\35884\Documents\Spacecraft\Phase2_SAC_V2_1工程诊断与改进指导.md`  
> 结论边界：本报告记录开发级根因诊断，不把失败模型标为有效SAC基线，也不把6条MPC轨迹称为正式统计验收。

## 1. 交付结论

V2.1已经把Phase-2 SAC失败从“任务、观测、探索、奖励或算法均可能”缩小到一条可复核链路：**nominal和新Stage-A均由constrained MPC证明可行；23D直接任务观测对MPC动作的轨迹级监督拟合优于旧16D，但单独不足以闭环；首次训练还发现并修复了一个会让观测越界并触发critic瞬时爆炸的真实代码缺陷；修复后SAC前期数值稳定并学到少量接近行为，但replay安全/near-goal覆盖持续不足，随后Q值、critic loss和熵系数逐步漂移，150k仍零完成。** 因此当前不再支持继续Pure SAC长跑或随机调参；下一最小路线是复用已保存MPC数据做透明的demo replay prefill或actor初始化，再做短SAC微调。

## 2. 指导文档审查结果

上层文档对四个结构疑点的判断与活动代码一致：旧Phase-2确为16D `SE(3) log + margins`，Stage-A初始速度仍由0.35m/s任务硬限主导，SAC在20k前使用全范围随机动作，约束奖励仅越界后出现且distance failure固定−100。文档可作为路线基准，但Gate 0同时改变observation、Stage-A动力学、learning-starts和reward，而“每轮单一核心变量”与此有张力。本次执行将Gate 0定义为恢复可学习接口和诊断能力的必要组合改造，并通过旧/新observation同teacher监督对照单独估计表示影响；不把一次训练的成败归因于任一单独修改。修订结论已追加回原指导文档第15节。

## 3. 活动代码改动

- `env/observation.py`：新增显式schema `phase2_v2_1_23d`。通道为姿态误差3、raw target-frame位置误差3、raw target-frame位置率3、相对角速度3、目标角速度3、corridor方向2、FOV方向2、四类signed margin。旧16D builder保留，仅用于诊断对照。
- `env/task.py`：公共化任务轴正交平面基，保证sampler和observation方向约定一致。
- `env/scenarios.py`：显式区分任务硬约束和训练初态动能包络，新增initial total speed、relative angular-rate、interior radial fraction、minimum FOV margin参数。
- `env/se3_rendezvous_env.py`：Stage-A采用2–4m、35deg、tumble 0.18、0.15m/s、每轴0.018rad/s、无near-boundary、radial fraction≤0.65、FOV interior≥5deg；Phase-2 observation space变为23维；distance-failure penalty重标为−20；manifest通过环境dataclass显式记录全部值。
- `env/reward.py`：violation-only quadratic hinge改为归一化、有界、连续的softplus proximity cost，安全集接近边界时已经可见；gamma未改。
- `train/configs.py`：只对Phase-2把`learning_starts`从20k降为5k；最终action space、执行器上限和其余SAC参数不变。
- `train/callbacks.py`：新增固定probe actor/Q诊断、replay安全/near-goal/动作/终止分布；固定评估新增最小距离、到达时间、首次违规、最小margins、动作范数/饱和和行为分类。
- `train/train.py`：新增`--phase2-stage-a-only`和200k硬上限；Stage-A固定评估每50k运行，连续3个0 completion且distance failure>50%的窗口自动停止。
- `experiments/diagnose_phase2_v2_1.py`：2×10k初态、observation和reward分布统计。
- `experiments/diagnose_mpc_observation_learning.py`：MPC分层teacher数据、旧16D/新23D同网络轨迹留出回归和少量闭环克隆诊断。

## 4. Gate 0验证

完整测试最终为`67 passed in 27.09s`。新增行为测试覆盖23D方向符号、目标角速度响应、desired pose误差归零、Stage-A动能/interior envelope、proximity cost连续有界，以及port平面后横向±1000m逃逸状态的全部观测有限且绝对值小于10。

`logs/phase2_v2_1_gate0_stats.json`记录Stage-A与nominal各10k样本。Stage-A 10k全部truth-feasible，实际总速度最大0.14410m/s、角速度分量最大0.017999rad/s、radial fraction最大0.64998、FOV margin最小0.08799rad、corridor margin最小1.23298m；nominal 10k也全部truth-feasible。目标角速度3通道在单一固定tumble任务内近常数，这是固定目标初态定义的结果；不同tumble配置和逐步目标传播仍会改变这些通道。

奖励初态统计显示Stage-A约束接近项大多远小于task cost，符合interior基础学习任务；nominal近边界样本的FOV/corridor/speed proximity项显著增大但仍有界，没有单项高两个数量级。

## 5. MPC分层与observation可学习性诊断

证据目录：`logs/phase2_v2_1_mpc_observation_diagnostic/`。Stage-A、nominal interior、nominal near-boundary各2条，共6条、4642控制步；6/6 completion、6/6 constraint-success、0 fallback。完成步数514–1029，完整command中位约0.307–0.308s，再次确认采样器可行，但不重复100-seed正式MPC验收。

相同三层256-ReLU MLP、按整条轨迹留出而不是随机打散控制步：旧16D hold-out MSE/MAE为0.02219/0.06811，新23D为0.02073/0.05483，MAE改善约19.5%；新表示在大部分动作轴上更优，但teacher饱和区域MAE仍约0.372。两种回归策略各在3个未参与teacher采集的Stage-A seed闭环测试，均3/3 distance failure且constraint failure。由此可得：新表示降低了监督映射难度，但不是闭环充分条件；不能因为开环误差较小就把BC直接定为正式方法。

## 6. 两次150k训练及真实代码缺陷

### 6.1 V2.1首次运行：无效算法证据、有效工程诊断

运行：`phase2_sac_v2_1_stagea_200k_seed260812`，实际150k自动停止。固定评估50k/100k/150k均0/10 completion、10/10 distance failure，near-goal replay始终0，probe twin-Q分歧扩大到约100–147。

但stdout显示约8k时critic loss已达10^29–10^30、actor loss约10^14。复核发现23D corridor方向用横向位移除以`max(port_axial*tan(angle), eps)`，一旦策略越过port平面，分母退化为机器epsilon；该方向又未通过softsign，observation可以达到百万甚至更高，违背环境声明的`[-10,10]`。因此此运行不能用于评价SAC本身，只能证明observation builder存在越界bug。

修复为：corridor几何分母使用`max(abs(radius), 0.25m)`，方向特征再次通过softsign；新增极端逃逸回归测试。修复前后同seed的早期TensorBoard形成直接因果对照：缺陷轮约8k critic loss为10^29量级，修复轮约13.7k critic loss仅0.00566、actor loss−0.435、ent_coef 0.00214。

### 6.2 V2.1b修正版：有效Pure SAC能力判别

运行：`phase2_sac_v2_1b_boundedobs_stagea_150k_seed260812`，实际150k，约58.4分钟，按连续3窗条件自动停止。固定10回合结果：

| step | completion | distance failure | 平均最小位置误差 | 平均首次违规 | 行为分类 |
|---:|---:|---:|---:|---:|---|
| 50k | 0/10 | 10/10 | 3.049m | 9.62s | 1 approached-then-failed，9 constraint-loss |
| 100k | 0/10 | 10/10 | 2.936m | 7.51s | 2 approached-then-failed，8 constraint-loss |
| 150k | 0/10 | 10/10 | 2.960m | 7.35s | 1 approached-then-failed，9 constraint-loss |

固定probe/replay趋势：

| step | replay全约束满足 | replay `<1m` | Q1/Q2均值 | twin-Q分歧 | actor饱和分量 |
|---:|---:|---:|---:|---:|---:|
| 25k | 21.04% | 0 | 2.77 / 2.70 | 0.098 | 5.73% |
| 50k | 17.90% | 0.073% | 18.54 / 18.73 | 0.535 | 6.77% |
| 75k | 16.80% | 0.098% | 98.73 / 98.37 | 3.56 | 7.29% |
| 100k | 15.72% | 0.146% | 748.82 / 745.49 | 21.00 | 16.15% |
| 125k | 14.36% | 0.073% | 3111.12 / 3105.79 | 68.36 | 13.54% |
| 150k | 13.92% | 0.220% | 7840.33 / 7798.37 | 119.86 | 5.73% |

修正版不是瞬时数值爆炸：前期双Q一致且loss正常，策略确实获得少量接近能力。但训练数据越来越由违规/远离任务状态主导，安全覆盖下降，near-goal覆盖始终低于0.22%；Q值整体上漂而非早期twin-Q符号分裂，约138k stdout的critic loss已到10^5、actor loss约−10^4、ent_coef约8–11。策略学到的是更激进接近，而不是约束恢复、制动和保持。

## 7. 根因判定与下一最小路线

当前证据支持的优先级为：

1. 已修复主工程缺陷：新observation越界是首次V2.1瞬时爆炸的直接主因。
2. 剩余主瓶颈：从零探索/replay覆盖不足，安全样本和near-goal/制动/保持样本过少；随训练发生Q尺度、critic loss和熵系数漂移。
3. 次要但真实因素：23D表示比16D更易拟合teacher，但不足以单独闭环；constraint proximity没有阻止策略牺牲安全换接近，后续需在有合理数据覆盖后再评估其权重/归一化，而不是立即试参。
4. 当前被排除为主因：任务或Stage-A物理不可行、执行器不足、初态仍高动能/近边界、actor完全零动作、单纯动作饱和。

下一步只做一个高信息增益对比：复用`teacher_dataset.npz`，以透明manifest记录demo来源和比例，优先测试demo replay prefill（包含安全接近、制动、保持transition）或最小actor初始化，然后用不超过50–100k的Stage-A SAC微调。判据是早期replay安全/near-goal覆盖是否显著高于本报告、固定probe Q是否保持有界、以及固定Stage-A是否出现非零completion/constraint-success。若有示范仍发生同类Q漂移，再单独处理reward/Q-target归一化；若Stage-A建立闭环，再进入nominal两阶段。当前两个SAC模型均仅为诊断产物，不应进入候选筛选或正式评估。

## 8. 关键证据路径

- 指导修订：`C:\Users\35884\Documents\Spacecraft\Phase2_SAC_V2_1工程诊断与改进指导.md`
- Gate 0统计：`logs/phase2_v2_1_gate0_stats.json`
- MPC监督数据与结果：`logs/phase2_v2_1_mpc_observation_diagnostic/teacher_dataset.npz`、`result.json`
- 观测缺陷训练：`logs/phase2_sac_v2_1_stagea_200k_seed260812/`
- 修正版训练：`logs/phase2_sac_v2_1b_boundedobs_stagea_150k_seed260812/`
- 修正版检查点/诊断模型：`models/phase2_sac_v2_1b_boundedobs_stagea_150k_seed260812/`（不得标为有效模型）

上层轻量审查包只复制核心源码、测试、文档和关键JSON/NPZ/TensorBoard事件，不复制约36MB的失败模型检查点。完整模型与原始monitor仍在活动工程内，可按上述路径复核。

## 9. V2.2 MPC actor初始化追加判别（2026-08-12）

在V2.1交接后继续执行了最小示范分叉，没有伪造teacher数据中不存在的`next_obs/reward/done`，因此没有实施不真实的replay prefill；新增`train/behavior_initialization.py`，把4642条MPC样本按完整轨迹分训练/留出，直接初始化当前SB3 SAC四层256 actor的均值网络。普通Pure SAC配置保持不变；仅示范模式显式使用`learning_starts=1000`并初始化`log_std=-2`（std约0.135），避免actor先验在5k全随机warm-up和高初始方差下名存实亡。完整测试增加至68项。

初始化本身成功：49 epoch早停，train MSE/MAE为0.00385/0.03130，整轨迹hold-out MSE/MAE为0.01571/0.04668，饱和teacher样本MAE为0.308。运行`phase2_sac_v2_2_mpc_actor_init_stagea_100k_seed260812`随后进行Stage-A SAC微调。25k时Q1/Q2为4.10/4.07、分歧0.209，尚数值稳定，但replay全约束满足仅18.2%、`<1m`仅0.024%，fixed-probe动作饱和分量28.1%。50k固定评估为0/10 completion、10/10 distance failure、平均最小位置误差3.405m、平均首次违规4.91s，10/10均为`constraint_loss_without_close_approach`；相同seed Pure SAC修正版50k对照为3.049m、9.62s并有1个`approached_then_failed`。50k replay安全比例进一步降到16.1%，Q仍一致但已升至约39.7。

因此actor-only BC初始化不仅没有改善闭环，50k行为还劣于Pure SAC对照。进程在50k门槛后主动停止，保留`behavior_initialized_model.zip`、`sac_50000_steps.zip`、manifest、TensorBoard和固定诊断；未跑满100k，未启动第二seed。该结果说明teacher均值先验在没有示范critic/replay支持或持续BC约束时会被在线SAC迅速覆盖。下一步已不再是小型工程修复，而是方法选择：需要上层决定是否投入重新采集完整MPC transition以做真实demo replay、在在线更新中保留BC正则，或直接转向显式SAC–MPC约束交互。当前不应继续训练或把actor初始化模型作为候选。
