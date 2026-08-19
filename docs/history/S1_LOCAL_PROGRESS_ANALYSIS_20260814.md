# S1 bounded local-progress 250k analysis — 2026-08-14

## 实验完整性

Run `phase2_mission_s1_fullcanonical_localprogress_autoent_seed260814`状态completed，实际/计划250000/250000；actor、critic fresh，replay 0；training/evaluation config相同，curriculum/adaptive false；automatic entropy仍为`auto_0.005`、target entropy -6.0。相对上一轮auto-ent，唯一任务变量为Phase-I progress从`k_p(C_t-gamma*C_t+1)`改为`k_p(C_t-C_t+1)`。

## 周期评估与训练结果

50k/100k/150k/200k/250k deterministic evaluation均Gate 0/10，minimum Gate mean依次为8.089、8.032、8.076、9.192、8.733 m，best依次为2.775、3.379、3.008、6.636、6.636 m。训练共745 episodes，Gate 0；speed 584、premature-entry 32、distance 130。仅1个训练episode位置进入1 m，但没有联合满足速度、姿态和角速度。

相同20-seed独立对照：上一轮auto-ent 100k/final minimum Gate mean为8.9116/9.3149 m；local-progress 100k/final为7.9749/9.4592 m，四者Gate均0/20。Local progress在100k带来约0.94 m mean improvement，但到final完全遗忘且略差于auto-ent final。

## 行为结构

100k代表轨迹中seed 267207将Gate距离从6.972降至2.717 m、横向误差从5.545降至2.180 m，但最近点速度0.655 m/s、无制动，姿态从0.024恶化至0.843 rad，角速度0.038 rad/s。Seed 267212将距离从7.219降至3.467 m、横向从6.750降至3.373 m，但最近点速度0.511 m/s、无制动，姿态从0.610恶化至1.129 rad，角速度0.047 rad/s。其余代表轨迹多以初始状态为最近点。Final代表轨迹进一步退化，仅1/5有超过1 m的位置改善，最近点速度仍0.639 m/s且无制动。

## 数值趋势

Alpha从0.005演化至约0.0738；diagnostics probe Q1 mean从-0.13增至83.14；TensorBoard critic loss从50k约0.087增至250k约105.57，actor loss约-112.71。上一轮auto-ent 250k alpha约0.0268、Q1约40.44、critic loss约11.97。动作饱和仍较低，因此失败不是简单actor全饱和，而是局部位置趋近没有转化为联合Gate控制，并伴随中后期value尺度增长和行为遗忘。

## 裁决

Local progress假设仅得到有限支持：它增强了早期位置方向的信用信号，但策略表现为高速接近、无制动并牺牲姿态，没有形成Gate acquisition；中后期改善消失。按上层情况C处理：停止继续修改reward，不进入S2，不续训，不保留100k/final作为可用模型。下一方向交由上层审查state/action representation alignment，尤其是target-frame translational observation与body-frame force action之间的映射负担。
