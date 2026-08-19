# S1 body-observation SAC tau=0.001 单因素验证 — 2026-08-15

上一轮`phase2_mission_s1_cone20_bodyobs_autoent_seed260814`证明body-frame平动observation是有效representation因素：独立20-seed在50k平均实际靠近0.998 m、7/20靠近超过1 m，而target-frame 100k仅0.453 m、3/20；训练中曾在约50,270步到达Gate距离0.289 m。但该能力随后消失，final平均仅靠近0.201 m且Gate仍为0。同期alpha约0.0054→0.189、probe Q约2.6→110、critic loss约0.09→277，符合前期学会趋近、后期value/Bellman drift同步遗忘。

本轮严格只修改Pure SAC Polyak target更新：

```text
tau: 0.005 -> 0.001
```

cone20、`phase2_mission_v2_body_translation_24d`、body-frame force、任务、reward/termination、normalization、automatic entropy、网络及其余训练参数全部冻结；actor、critic、replay必须fresh。

启动前证据：当前与上一轮正式manifest的hyperparameters逐字段比较，唯一差异为tau；全量回归89 passed。5200-step fresh CUDA smoke completed，manifest tau=0.001、schema和两个frame正确、train/eval config相同、actor init null、critic fresh、replay 0；alpha 0.005→0.004975，observation、actor/critic参数及sample action/log-prob/Q全部finite。smoke临时models/logs已清理，证据JSON保存在工作区。

唯一正式命令：

```powershell
python -B -m train.train --steps 250000 --seed 260814 --run-name phase2_mission_s1_cone20_bodyobs_autoent_tau001_seed260814 --mode phase1_pretrain --checkpoint-freq 50000 --eval-freq 50000 --eval-episodes 10 --eval-seed 20288000 --device cuda
```

除NaN/Inf或明确程序异常外不提前停止。结束后按相同fixed seeds及独立20 seeds与tau=0.005直接比较，核心裁决是50k附近的Gate-directed approach能否保持到150k–250k，并结合critic loss、Q、alpha、actor loss与最近点速度/姿态/角速度判断。
