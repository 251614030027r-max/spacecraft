# Phase-2 mission P0 validation — 2026-08-13

## 执行边界

本验证只确认两阶段mission没有明显不可行或接口错误；没有调Gate、尺度或约束参数，没有运行S1长训练、正式策略评估或完整两阶段Pure MPC验收。解释器为bundled CPython 3.12.13，并加载工程`.venv/Lib/site-packages`。

## 结果

- 主工程回归：`75 passed in 21.73s`。
- truth reset/step：`phase1_pretrain`和`full_mission`各8个seed，共16次；24D observation和reward均有限，初始中心距离实测12.2648–16.7286 m，零动作首步均未异常终止。
- Gate：8个seed在精确Gate状态执行一个truth step，全部单向切换到`mission_phase=1`，无terminal constraint failure；最小corridor lateral margin 4.5514 m、FOV margin 0.8727 rad、closing-speed margin 0.2813 m/s。
- terminal MPC：canonical N=50、CLARABEL在Gate状态返回`optimal`，未使用zero fallback；solve time 0.5163 s，predicted minimum normalized truth margin 0.3685，输出满足物理输入界限。
- S1入口：`python -B -m train.train ... --mode phase1_pretrain --validate-only`通过Gymnasium环境与SAC配置检查，未创建模型或训练日志。

## 判定

P0达到停止条件，可以直接进入S1。上述数据是低成本接口验证，不是任务成功率、MPC实时性或论文性能结论；N=50 solve time明显高于0.1 s控制周期，但按既定边界留待P3/P4基于实际terminal运行再决定是否做小规模horizon比较。
