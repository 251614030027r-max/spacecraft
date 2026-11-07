# 快速翻滚非合作目标捕获:真实需求与"快速"是否合理(文献支撑)

写于 2026-09-18,上层沙箱。回答导师问题:**快速翻滚目标捕获是不是真需求?我们的翻滚率合不合理?**
结论用于论文 motivation,数字均指向公开文献(见文末链接),不是拍脑袋。

## 1. 需求是真的:ADR / OOS 的头号对象就是翻滚的大型非合作体

- 主动清除碎片(ADR)和在轨服务(OOS)公认最难、最优先的目标,是**失效大卫星和火箭末级**——
  它们**没有专门抓取点**,因残余角动量、能量耗散和重力梯度力矩而**持续翻滚**,姿态状态未知。
- 具体任务:ESA **e.DEORBIT / ClearSpace** 明确面向 Envisat 这类翻滚目标;文献里有专门针对
  **Ariane 火箭末级**的"先消旋再机械臂抓取"方案。**"捕获前必须处理目标翻滚"是这些任务的设计前提。**

## 2. 我们的翻滚率是 Envisat 量级,现实、可辩护

- **Envisat**(~8 吨,失效):地面雷达/光学/SLR 观测,自旋从早期 ~0.1°/s 逐步升到 **2.9–3.5°/s**
  量级(2013 年测得峰值 ~2.9°/s、自旋周期 ~124 s;另有分析给出平均 ~3.5°/s)。模型预测未来
  可达 **~5°/s**。
- 我们的设定:**nominal 2.36°/s(0.0412 rad/s,周期 152 s)、标定用到 3.54°/s(0.0618 rad/s)**——
  **正好落在 Envisat 实测/预测区间内**,是"大型失效目标"的真实量级,不是虚构的高速。
- 能力门槛佐证任务难度真实:纯**贴附式**方案只适用于 **<2°/s**;而先进自主交会对接(RVD)系统
  的设计目标可到 **~25°/s**。我们的 2.4–3.5°/s 正处在**"简单方案已经不够、需要主动同步/控制"**的
  区间——即这个问题**值得做控制方法**。

## 3. 论文该怎么框"快速"(关键,别被审稿打)

- **不要吹"绝对高速"**:小碎片能自旋几十~上百°/s,我们 2.4–3.5°/s 绝对值只算"中等偏上"。
- **正确措辞 = "相对控制/预测时域而言快"**:在我们 3–15 m 近距离,`Λ=ωr/v_max>1`,站不住、必须
  持续共旋,且**超出参考文献适用域**(北航 ~1°/s 且无速度约束;南航明确假设"tumble moderate
  relative to prediction horizon"的 LTI 预测)。这个"Λ>1、LTI 预测假设失效"的立论**站得住**,
  也正是我们任务里"同步—进入"长期权衡产生的物理根源。
- 一句话 motivation:**"面向 Envisat 量级(~2–3.5°/s)失效大目标的预捕获,近距离共旋不可避免
  (Λ>1),已有 LTI 预测方法的适用假设失效——由此产生持续同步与机会式进入之间的长期决策问题。"**

## 4. 可引用来源(供正文/参考文献落地时再逐条核对)

- ESA e.DEORBIT 任务论文(SDC7):https://conference.sdo.esoc.esa.int/proceedings/sdc7/paper/1053/SDC7-paper1053.pdf
- Envisat 姿态运动调查(ESA eDeorbit workshop):https://indico.esa.int/event/46/attachments/1989/2333/ESA_eDeorbitWorkshop-_Envisat_attitude.pdf
- Envisat 转动状态 epoch 分析(Acta Astronautica / ScienceDirect):https://www.sciencedirect.com/science/article/abs/pii/S0273117720306360
- Envisat 长期转动运动分析与观测对比(ResearchGate):https://www.researchgate.net/publication/328159681
- Envisat 转动运动时序分析(SDC7 paper 437):https://conference.sdo.esoc.esa.int/proceedings/sdc7/paper/437/SDC7-paper437.pdf
- 空间碎片翻滚运动评估(ESA/ESOC 研究报告):https://nebula.esa.int/sites/default/files/neb_tec_studies/2745/public/GT17-152GR_EX.pdf
- Ariane 火箭末级主动消旋 + 机械抓取方案(ResearchGate):https://www.researchgate.net/publication/277714567
- 非合作翻滚目标交会(ISSFD 2015,Benninghoff):https://issfd.org/2015/files/downloads/papers/007_Benninghoff.pdf
- 火箭末级碎片鲁棒捕获与离轨(Stanford ASL, AeroConf17):https://stanfordasl.github.io/wp-content/papercite-data/pdf/Bylard.MacPherson.Hockman.ea.AeroConf17.pdf
- 翻滚空间碎片捕获策略(Acta Astronautica / ScienceDirect):https://www.sciencedirect.com/science/article/abs/pii/S0094576510002365

**注:** 上述为检索命中,正文引用前应逐篇核对确切数字与出处(尤其 Envisat 各文给的峰值/平均值
略有差异:2.9°/s@2013、~3.5°/s 平均、模型预测 ~5°/s)。方向性结论(Envisat 量级、需主动处理、
我们的率现实)是稳的。
