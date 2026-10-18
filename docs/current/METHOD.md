# 方法：SAC–G★–MPC 训练内耦合（2026-10-18）

**当前没有冻结的正式方法。** 下面是已实现并通过技术核验的方法构件，以及它们在已有实验中的结论。下一任务版本冻结前，不得把任何构件表述为"已证明有效"。

## 1. 职责划分

| 层 | 职责 | 代码 |
|---|---|---|
| SAC | 在审议点选择任务意图：首次延迟 / 在役平移、κ、v_in | `train/t1_option_sac.py`、`train/t1_tcpf.py` |
| G★ | 把意图变成可执行轨迹；计划保持、恢复、期限保护。最早可行时刻只作锚点和保底，不替 SAC 优化回报 | `env/t1_gstar.py` |
| MPC | 共享 h35 约束跟踪、真值守护、2 s 影子预测 | `controllers/mpc/` |
| Pure MPC | 无 G★ 的架构参照，冻结不动 | `experiments/t1_gstar_probe.py --row pure` |

## 2. 学习器构件

**`OptionSAC`：阶段感知 SAC**
- 审议样本的后继值取 soft value，α = 0.005 × 审议间隔；continue 样本的后继值取 continue 动作的 Q。
- Actor 只在审议态训练。
- 共用 critic 修复：MC 回写（λ_mc = 0.03）与保持带等价（λ_eq = 0.1）。

**MPC 相对项 NΔ：**
- 只在审议步作用，使用真实一步。
- 损失：Huber(ΔQ − δ)，其中 δ = (r̂ − r) + γ(V(ẑ′) − V(z′))。
- 权重在 3000 步时按梯度一次标定到 TD 梯度的 5%，之后每次更新上限 10%。

**`TCPFSAC`：审议区间 Bellman**
- 审议样本：y = R_k + γ^d(1 − D)V̄(z_{k+d})；continue 样本仍用一步目标。
- 区间未封账的审议样本被掩码。
- `IntervalReplayBuffer` 按环境封账；被环形覆盖的区间直接丢弃，不回填。

**`uniform_eligibility`：T1-O 2×2 四臂的统一样本资格**
- 四臂 O-R1 / O-R3 / O-Δ1 / O-TCPF 的 TD、MC、等价项、MPC 相对项与 Actor，都只使用 continue 样本和已封账的审议样本。
- 四臂只在"审议样本的目标"（一步或区间）和"是否启用 MPC 相对项"上不同。

**训练器：**

| 档 | 设置 | 来源 |
|---|---|---|
| A | UTD 4、恒定 lr 1e-4、batch 256、buffer 3e5、256×3、τ 0.001、learning_starts 1000 | 旧 T1-L |
| B | UTD 1、Actor 每 3 条全局真实转移更新一次、余弦 1e-4 → 1e-5 | 已实现，未使用 |

**训练入口：** `train/train_t1_option.py`。
- 臂：b2 / n0 / ndelta（T1-L）、sb2 / sn0 / sndelta（T1-L-S，已取消）、or1 / or3 / od1 / otcpf（T1-O，需 `--profile --fuel-cost`）。

## 3. 已有实验对方法的结论

详见 `docs/HISTORY.md` 与 `docs/DEAD_ENDS.md`。

| 实验 | 事实 |
|---|---|
| 2 s 逐步决策（T1 v2、v4，T1-L B2） | 后期退化："later"推迟占比上升，Q − G 系统性高估（D26） |
| 6 s 审议 + continue（T1-L N0 / NΔ，单种子 40k） | 稳定，相对 0k 学习成立，但低于最强固定规则；NΔ − N0 未证明 |
| 任务层面 | T1-L 与 T1-O V2 两次独立检验，状态相关、可由观测兑现的决策余量都不足（D27、D29） |

**本线未检验的部分：** TCPF（区间 Bellman + MPC 反馈）的训练效果从未在有学习空间的任务上检验过。T1-O V2 在训练前就按准入规则停止了。

## 4. MPC 反事实可信度工具（Gate E，已实现、未运行）

`experiments/t1o_cf_check.py`：
- 在审议点从同一快照用两个影子各飞 2 s 真值分支，比较预测奖励差与真值奖励差的符号和误差；
- 在标定集上设 e_r 门限（p95），再在独立验证集上核验。

注意：旧门限 e_r = 0.06 是在 λ_f = 5 下标定的，换燃耗权重后不能沿用。
