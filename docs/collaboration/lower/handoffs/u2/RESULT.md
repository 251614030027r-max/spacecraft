# U2 训练验证，2 种子 × 2000 决策，评估开局 294148–294195

SAC 与 EACH 为预实验模型。科学冻结 e7acaa340cbff41a2e618c200cd7bdeb0438118f，执行书 collab 385a99ab。PROBE 与 EACH_PROBE 各用种子 10、11，每条 2 个异步 worker，Actor 24 维，critic 分别 51、75 维，评估关闭探询。

- 自检全部通过。104 项测试通过，四个复用模型 SHA256 一致，EACH/SAC 各两个开局重评逐项一致。两臂 40 步冒烟和每臂两个冒烟评估完成，manifest、探询记录及日志占用恢复验收通过。见 pytest.log、REUSE、DRY/MANIFESTS.txt、PREFLIGHT_VERIFIED.json。
- 上层离线预检 stop=false、defects=false，见 STAGE_A_SUMMARY.json。四条训练均完成，八组各 48 例，原生 CHECK 完整。四条 TensorBoard 的真实回报回合数和末步与原生记录一致（402、403、419、376 回合）。
- 11 月 6 日 18:13:49 JST 开训，训练 7.72 小时，总墙钟 9.15 小时，11 月 7 日 03:23 JST 结束。

| 96 对评估 | 平均差 | s10 / s11 | 捕获数 | 原生判定 |
|---|---:|---:|---:|---|
| PROBE − SAC | +1.34 | +7.67 / −4.99 | 68 / 67 | no_gain |
| EACH_PROBE − EACH | −4.22 | +6.15 / −14.60 | 71 / 73 | kept |
| PROBE − EACH | −9.27 | +2.55 / −21.09 | 68 / 73 | 参照 |

主判据 G1、G2、G3 为 false，G4、G5 为 true。主对照去掉每种子最大的两个正差后为 −5.41。EACH_PROBE 满足保留容忍线，未达互补标准。八组最终评估均无 unsafe。按执行书结束 U2 同类调参迭代，正式实验另行授权。

最终训练日志共三次 unsafe。EACH_PROBE s11 在第 939 步（176.1 s），PROBE s10 在第 1419 步（121.4 s），中期简报之后又有 EACH_PROBE s10 在第 1972 步（186.6 s）。原生记录未含开局号、事件模式、距离或具体约束。详见 TRAIN_UNSAFE_FINAL.json，G5 按最终评估计算。

完整原始证据位于科学分支 evidence/U2_TRAIN，逐文件 SHA256 见 SHA256SUMS，模型 SHA256 见 REGISTRY_RESOLVED.json。中期简报固定引用 collab 873731ca。

科学证据提交 83bc7f6af19edfc7d9e6baf38a8ab1c918b2b135。425 个 Git 文件已逐字节核验。
