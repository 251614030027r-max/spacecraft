# V2 离线执行信号审计

本审计不改变任务、训练、reward 或 MPC，只在已有 nominal 种子轨迹的代表状态上读取单步求解信号。

## 预注册协议

- 状态：种子 `[262000, 262006, 262012, 262018, 262024, 262030, 262036, 262042]`，每个种子决策点 `[0, 15, 30]`，共 24 点。
- 候选：距离前进/后撤各小步与大步，承诺增加/减少各小步与大步，加不动基准，共 9 个。
- 重复：每个候选独立 reset 后求解 3 次。
- 判据：至少 70% 状态上至少一个方向随幅度单调，且候选跨度大于同状态重复波动。

## 结果

| 信号 | 类型 | 有信息状态 | 比例 | 通过 |
|---|---|---:|---:|---|
| `maximum_successful_slack` | 便宜 | 24/24 | 100.0% | 是 |
| `total_successful_slack` | 便宜 | 24/24 | 100.0% | 是 |
| `objective` | 便宜 | 24/24 | 100.0% | 是 |
| `actuator_usage` | 便宜 | 23/24 | 95.8% | 是 |
| `predicted_minimum_margin` | 离线 oracle | 24/24 | 100.0% | 是 |

## 三选一结论

有便宜信号满足预注册判据：`maximum_successful_slack`、`total_successful_slack`、`objective`、`actuator_usage`。governor 有离线信号地基，但 V2 第一版仍不接入。

V2 第一版仍使用固定不对称限速；本审计不授权接入 governor。完整逐状态数据见 `eval/adp/v2_execution_signal_audit.json`。
