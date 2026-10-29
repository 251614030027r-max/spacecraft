# T2.4 09:00期限实际进度交付

生成 2026-10-26T08:31:37.772857+09:00；科学 `e13c8b1d52dafe7b90fac3eaaa3651294d253b0f`；执行书collab `57e4ba0`。本机26项测试通过，外部日志适配器单独验收；冻结科学文件不改，任务/奖励/候选K8/β0.5/执行层/步数/3worker不改。仅两组MVE seed0/1各4000，SAC复用T2.3，种子1SAC3k补评估；不跑后续分支。

## 日志和来源
用户明确授权外部日志监视；脚本随附，滚动JSON约30秒，TensorBoard逐回合真实/训练回报、成功率、原生critic损失/min-Q及模型反馈探针。日志验收含相同数据原生更新权重逐位一致、关闭日志不改变RNG状态、真实48决策异步冒烟事件生成和增长；机械数据不参与科研结果。异步调度不承诺跨次训练轨迹逐位相同。Q1/Q2分别值冻结探针未提供，使用实际q_pi_mean（min-Q），不额外前向、不伪造。
SAC逐文件SHA映射和旧模型SHA随附；复用SAC旧结果可能缺window_recoveries字段，明确计缺失，不能当恢复次数0。

## 判定与曲线
**未完成即如实标记未完成：本快照不是最终判定，不缩减步数/开局，不挑选检查点。**
```json
{
  "phase": "training",
  "rolling": {
    "e_mve_s0": {
      "jst": "2026-10-26T08:31:05.249458+09:00",
      "steps": 2860,
      "updates": 5322,
      "episodes": 633,
      "wall_s": 28722.375,
      "recent_decisions_per_s": 0.06591088847881411,
      "overall_decisions_per_s": 0.09957393843649767,
      "eta_s_estimate": 17296.07999999018,
      "snapshot_not_prediction": true,
      "replay": {
        "transitions": 2860,
        "with_candidates": 2231,
        "candidates": 17246,
        "entry_candidates": 3663,
        "reason": {
          "off": 63,
          "entry_calibration": 441,
          "stage": 1778,
          "not_comparable": 83,
          "entry": 495
        }
      },
      "model_probes": 213,
      "pending_truth_records": 0,
      "final": false
    },
    "e_mve_s1": {
      "jst": "2026-10-26T08:31:04.333633+09:00",
      "steps": 2777,
      "updates": 5156,
      "episodes": 655,
      "wall_s": 28721.467999999993,
      "recent_decisions_per_s": 0.0,
      "overall_decisions_per_s": 0.09668725846464396,
      "eta_s_estimate": null,
      "snapshot_not_prediction": true,
      "replay": {
        "transitions": 2777,
        "with_candidates": 2144,
        "candidates": 16458,
        "entry_candidates": 3699,
        "reason": {
          "off": 45,
          "entry_calibration": 454,
          "stage": 1692,
          "entry": 491,
          "not_comparable": 95
        }
      },
      "model_probes": 207,
      "pending_truth_records": 0,
      "final": false
    }
  },
  "eval_completed": 192,
  "audit_count": 6,
  "formal_complete": false
}
```
尚缺最终评估/判定则明确MISSING；已授权计算按原范围继续，完成后另交final。

## A机制与历史对照
本轮六开局2910000–2910005审计，已执行意图本阶段进/未进入口两组和raw/anchored误差在MECHANISM_READOUT.json（比较缺终止匹配的候选不强填误差）。原生A1_LABEL_AUDIT完整原件随附；若deadline尚未完成则在快照明确缺项。
T2.3固定对照science74945c71d5cdf883ab26aea812deeeb8d98f5e58：seed0 MVE4k42.364999、22/24、unsafe1；SAC4k54.476604、23/24、unsafe0；配对−12.111604，紧急−19.136867，去最大两例−22.569731；原生coupling_failure。T2.4 C1阈值严格42.365，其余按原生t24，不事后更改。

## 机制指标与计算
MECHANISM_READOUT包含每1000步探针的entry_bias_now/entry_rel_mean/loss_share/grad_cos/Q/龄期分布及window恢复逐事件原件，缺失明确保留；候选误差估计尚未证明覆盖真误差，不宣称理论界成立。实际训练墙钟、worker_order、各run吞吐见finished；并行墙钟不相加。00点启动、09点目标按实测核查，deadline进度不能冒充最终判定。

## 交付边界
只有事实与原生判定，不自动调参或追加分支；等待研究层与用户下一授权。模型ZIP/Replay和TensorBoard二进制事件只本机保留并记摘要/来源，Git提供必要432原始评估、6审计、原生读出/判定、训练JSON、外部日志源及验收证据。网络失败只补发布不重跑科学。
