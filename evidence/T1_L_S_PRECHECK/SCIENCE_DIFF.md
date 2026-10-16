# T1-L-S 相对 166244b 的改动清单

`git diff --stat 166244b..<冻结提交>`（证据目录本身不计；代码提交 `1af9690`，外加冻结提交中的一处工具修正）：

| 文件 | 行（+/−） | 改动 |
|---|---|---|
| `train/t1_option_sac.py` | 27 | 抽出 4 个钩子：`_td_targets`、`_td_loss`（带掩码）、`_actor_update_due`、`_review_batch`。默认实现与 166244b 逐位相同（跨版本参数哈希一致） |
| `train/t1_tcpf.py` | +215 | 新增：`CosineLearningRate`、`IntervalReplayBuffer`、`TCPFSAC` |
| `train/train_t1_option.py` | 72 | 新增 `sb2`、`sn0`、`sndelta` 三臂与 `--trainer A/B`，按 UTD × n_envs 设置 Critic 更新数；写出 `tcpf_stats.json`、`interval_audit.json`；manifest 记录训练器与学习器类。旧三臂只允许训练器 A |
| `train/train_t1.py` | +2 | 检查点额外写出 `*_tcpf_stats.json`（若存在） |
| `env/t1_l_task.py` | 16 | 新增 T1-R 评估档案（15–19 m，2.4–3.0°/s）；T1-L 档案不变 |
| `experiments/t1_l_eval.py` | 27 | 新增 `--task t1l/t1r`；按 manifest 中的学习器类加载 |
| `experiments/t1_l_readout.py` | 41 | 配对差的 sd 和 95% t 区间、Δv/时间差、区间审计、S 臂配对、`--eval-subdir` |
| `experiments/t1_l_s_trainer_choice.py` | +113 | 预注册的训练器选择规则 |
| `experiments/t1_r1_openings.py`、`experiments/t1_gstar_probe.py` | 各 2 | `--task` 接受 `t1r` |
| `experiments/t1_pretrain_checks.py` | +4 | 冻结提交中补充：`reload` 按 manifest 用 `TCPFSAC` 加载 T1-L-S 检查点（此前会退回 M1SAC/SAC）；在三个冒烟运行上核验通过 |
| `experiments/t1l_dev64_openings_298000.json` | +1098 | 64 个扩展开发开局（SHA-256 `e21831dfb59fc526a2ee73ca63fc5fce5889e00cd5dd9223d93e1dddcf149f22`） |
| `experiments/t1r_openings64_299000.json` | +1098 | 64 个 T1-R 开局（SHA-256 `4fc4bb820c314380d12fadd7cd022896edf132db1c5f7c87e339c67d398d8223`） |
| `tests/test_t1_l_s.py` | +279 | 16 项新测试 |

**未改动：**
- 环境动力学
- 奖励
- G★ / MPC（`env/t1_env.py`、`env/t1_gstar.py`、`env/t1_task.py`、`env/t1_task_review.py`、`env/t1_option_cf.py`、`env/t1_counterfactual.py`）
- Pure MPC
- 166244b 的 16 个开发开局
- 旧三臂的训练路径
