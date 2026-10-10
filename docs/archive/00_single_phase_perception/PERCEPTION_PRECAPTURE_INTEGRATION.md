# 感知接入 precapture:让"非合作"名副其实的单因子分阶段方案

写于 2026-09-13,上层沙箱,`41474cf`。**这是设计提案,不是结果。** 目的是把
"补上感知、明确非合作"变成一条不混淆、可归因、有文献支撑的执行路线,供 T12 结果
出来后拍板。**未动任何代码、未跑训练、未碰冻结件。**

---

## 1. 缺口(仓库已证实)

整条 precapture / 耦合线跑在 **truth 目标状态**上:

- `env/phase2_env.py:130 precapture_planning_environment_config()` 不带
  `perception` 参数,直接用 `PrecaptureTaskConfig()`,无相机、无 EKF。
- `env/hybrid_env.py` 的 T6 接口**直接读 `self.env.target_state` 真值**:构造
  arrival-condition 航点(`:395`、`:506`)、拼上层观测(`:361`)、并据此喂 MPC 参考
  (`reference_source="external_local"`,`:245`)。

所以当前"非合作"只体现为**目标翻滚快 + 不提供路径参考**;严格意义的非合作——
**目标状态不给真值、要靠机载传感自己估**——在耦合线里是缺的。这对论文声称的
"non-cooperative target" 是实质缺口,不是出图收尾。

## 2. 可复用件(已验证,勿重造)

A1 已经把这套感知链写好并验收过,只是接在旧 `single_phase` 上:

- 相机:`env/perception.py` `PerceptionConfig`(五特征针孔,1024²,f=430px,50° 半视场,
  30 m,1px 噪声,10 Hz)。
- 滤波:`estimation/relative_ekf.py` `RelativeStateEKF`(误差序 `[dθ,dp,dω,dv]`,
  `predict`/`update` 齐全)。
- 接法:`env/phase2_env.py:101 phase2_perception_environment_config()` 证明
  `perception=None` 保持原路径 bitwise 不变、单因子只改观测生成。
- `SE3RendezvousConfig` **已带 `perception` 字段**;A1 验收 **149 passed** + 30 s
  smoke(末端 0.057 rad / 0.85 m / 0.012 rad·s⁻¹ / 0.21 m·s⁻¹,仅 smoke,非性能结论)。

## 3. 为什么之前没加,以及怎么解决

没加的原因(用户原话):**感知和耦合一起加,失败无法归因是哪个造成的。** 这正是
本项目"严格一个可解释因子"规则要防的事。解决办法不是新发明,就是**沿用项目自己的
单因子分阶段法**——A1 当年就是把感知作为独立 foundation 阶段先验证,A2 才使用它。
把同一模式搬到 precapture:感知先独立立住,再进耦合。

## 4. 三阶段方案(每阶段恰好一个可归因因子)

**P1 — 开环 EKF 验证(感知因子,零学习、零耦合)。**
用现成的 Pure MPC / 脚本上层在 truth 上驱动轨迹,旁挂相机+EKF,量:5 特征可见性、
估计误差随距离/翻滚相位的分布、协方差是否收敛、有无测量中断。回答唯一一个问题:
**在 15–20 m、0.041 rad·s⁻¹ 快翻滚、入窗几何这个体制下,感知能不能立住。** 短诊断,
非训练,下层可跑。若这里就崩,后面都不用谈,且崩因明确是感知。

**P2 — Pure MPC on estimate(感知为唯一因子,仍无学习)。**
把 EKF 估计喂给 MPC 与参考,和**已完成的 Pure MPC on truth**(`PURE_MPC_ROW_VERIFIED.md`:
h35 9/12;新门控口径 32/48)对比。若 Pure MPC 在估计下仍守约束 → 感知不是瓶颈,而
**这一行本身就是一条合法主表基线:"非合作(估计状态)Pure MPC"**。这一步就能让论文的
非合作声明落地,即使 P3 不做。

**P3 — 耦合 on estimate(只有 P1、P2 通过才做)。**
此时估计已被独立验证,任何失败可归因到**耦合**而非感知。三种子从零,单因子。

## 5. 必须显式决定的一个设计点:真值入口

A1 的原则是"**只换观测**",truth 仍供 dynamics / reward / termination / 几何判违。
这条要保留——违规判定必须始终在 RK45 truth 几何上做(见 CLAUDE.md Traps)。

但 precapture 不同于 single_phase:**T6 接口和 MPC 直接读 `self.env.target_state` 真值**
(§1)。所以"真非合作"要把这些真值读换成 EKF 估计,而这**动到了冻结的 T6 输入**。
因此这不是一个 flag,是一个被声明的新 stage。建议:

- 以**新的 env config / wrapper**(estimated-target-state source)承载,`perception=None`
  路径与今天 bitwise 相同,把"估计入接口"作为**被声明的单因子**,绝不偷偷改 T6 语义。
- 违规与完成判定的真值几何**不变**;改的只是"控制器看到的目标状态来源"。
- T6 映射语义(a=(+1,·) 的固定设定点地板、棘轮)不变,只是其输入从 truth 换成估计——
  这一步的 bitwise 地板需要在估计源下重新定义并测(估计≠真值,地板不再是零差,而是
  "估计源下的 Pure MPC 等价")。

**这一点是整个接入的成败关键,建议由你先拍板"估计入 T6"是否算可接受的新因子,再动手。**

## 6. 冻结不动

下层 MPC 权重 / 门控 / 走廊面数、T6 映射语义、正在跑的 T12,全部不动。感知只加在
观测与目标状态来源;truth 路径保持不变、可回退。

## 7. 文献支撑

- `References/视觉抓捕.pdf`、`References/视觉综述.pdf`:非合作目标视觉相对导航 /
  视觉抓捕的建模背书;A1 的针孔相机 + 相对 EKF 已是其一种具体实现。
- `References/Ramezani 等 - 2026 - Fuel-aware autonomous docking using RL-augmented MPC …`:
  与"燃料是最终判据"直接对齐,可用于把**估计误差下的燃料代价**作为报告轴。
- **诚实标注:** 本沙箱无 poppler / 数值栈,未能提取上述 PDF 正文;此处按参考库中的
  角色引用,**未杜撰其内容**。精确引述与页码待一个带 venv 的步骤补齐(或由你指认)。

## 8. 成本与建议顺序

P1 是短开环诊断(下层可跑,非训练);P2 是评估(结果型可并行,compute 须串行单进程);
P3 才是训练(3 种子从零)。建议:**等 T12 结果**,再按 P1→P2→P3 推进。若 T12 显示学习
层有稳定正贡献,则一路做到 P3;若 T12 未显出稳定正贡献,则**先不做 P3**,只补 P1/P2
把"非合作 Pure MPC"这一行做实——论文的非合作声明即有着落,且不与耦合结论纠缠。
