# T2 阶段A完整交付与B运行中（2026-10-22 06时）

冻结31074206045cf62dba3d560e4442db4b00eea05a；执行书a3135ef。9项本机单测通过。24个固定开局294000–294023×9策略216/216完整，无ERROR，A三门全部通过。原始逐回合JSON和表随本轮提供。

A实际墙钟约101分钟。逐开局最佳相对最佳恒定策略余量+38.8352，剔除前两大仍+23.3881；三种策略各占5/24、5/24、4/24。相对rally规则余量+78.5474，剔除前两大+65.8236，但这只是固定策略能力空间，不是SAC已经学到的性能。

B于03:36自动开始，seed0双臂各5环境、各3000阶段决策，0/1000/2000/3000全读出。06:00时两个1000步检查点均未落盘；进程CPU持续增长、无Traceback，只有一次CLARABEL解精度警告。原生仅在1000步写heartbeat，不能从心跳空白断言死锁，也不能捏造当前步数。

本机吞吐显著慢于沙盒预估，先前7–8时完整交付预计无法保证。先交可独立判定的完整A；B继续原冻结设置，不减步数，不改算法，不额外训练，不因慢而重新启动。B完整曲线/模型项使用与两臂差尚未获得，不能声称学习或耦合有效。

## A原生门槛与全部数值
```json
{
  "openings": [
    294000,
    294001,
    294002,
    294003,
    294004,
    294005,
    294006,
    294007,
    294008,
    294009,
    294010,
    294011,
    294012,
    294013,
    294014,
    294015,
    294016,
    294017,
    294018,
    294019,
    294020,
    294021,
    294022,
    294023
  ],
  "policies": {
    "c_-5_0": {
      "n": 24,
      "mean_return": -128.343101150556,
      "capture": 0,
      "unsafe": 0,
      "mean_delta_v_success": null,
      "mean_time_success": null,
      "events": {
        "timeout": 24
      }
    },
    "c_0_0": {
      "n": 24,
      "mean_return": -47.50189717357792,
      "capture": 11,
      "unsafe": 0,
      "mean_delta_v_success": 1.3096907443775385,
      "mean_time_success": 176.7,
      "events": {
        "timeout": 13,
        "success": 11
      }
    },
    "c_0_5": {
      "n": 24,
      "mean_return": -44.146484792629565,
      "capture": 12,
      "unsafe": 0,
      "mean_delta_v_success": 1.5542774416308471,
      "mean_time_success": 190.51666666666665,
      "events": {
        "timeout": 12,
        "success": 12
      }
    },
    "c_10_0": {
      "n": 24,
      "mean_return": 16.858279513987753,
      "capture": 19,
      "unsafe": 0,
      "mean_delta_v_success": 1.3734159177556755,
      "mean_time_success": 131.1,
      "events": {
        "success": 19,
        "timeout": 5
      }
    },
    "c_10_10": {
      "n": 24,
      "mean_return": -1.3836019886520532,
      "capture": 18,
      "unsafe": 0,
      "mean_delta_v_success": 1.695535898977753,
      "mean_time_success": 150.7111111111111,
      "events": {
        "timeout": 5,
        "success": 18,
        "resource_exhausted": 1
      }
    },
    "c_5_0": {
      "n": 24,
      "mean_return": -8.343602219325135,
      "capture": 16,
      "unsafe": 0,
      "mean_delta_v_success": 1.3671586110904053,
      "mean_time_success": 142.04375,
      "events": {
        "timeout": 8,
        "success": 16
      }
    },
    "c_5_5": {
      "n": 24,
      "mean_return": 4.19243438859783,
      "capture": 18,
      "unsafe": 0,
      "mean_delta_v_success": 1.5453296166522354,
      "mean_time_success": 154.80555555555554,
      "events": {
        "timeout": 6,
        "success": 18
      }
    },
    "rally": {
      "n": 24,
      "mean_return": -22.85393832149659,
      "capture": 14,
      "unsafe": 0,
      "mean_delta_v_success": 1.3275381547691854,
      "mean_time_success": 135.20714285714286,
      "events": {
        "timeout": 10,
        "success": 14
      }
    },
    "gstar_reference_outside_action_space": {
      "n": 24,
      "mean_return": 59.55588065249387,
      "capture": 24,
      "unsafe": 0,
      "mean_delta_v_success": 1.424031166607666,
      "mean_time_success": 127.32083333333333,
      "events": {
        "success": 24
      }
    }
  },
  "best_constant_policy": "c_10_0",
  "per_opening_best_mean": 55.69346428213097,
  "gain_vs_best_constant": {
    "mean": 38.8351847681432,
    "without_top2": 23.388089460711043,
    "sorted": [
      210.29564329121027,
      207.21082300858365,
      206.8020554526995,
      202.67430561647967,
      31.49876606653423,
      17.83506859529038,
      9.342447269216358,
      6.290249027926123,
      6.1379125033385264,
      6.1349746657281585,
      5.937140860869718,
      4.643765101962785,
      4.402042680489245,
      3.5274393977952343,
      3.2817531260177617,
      2.7839290887986863,
      2.6329766254024634,
      0.33874945700621595,
      0.27439260008794975,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ]
  },
  "best_policy_counts": {
    "c_10_0": 5,
    "c_5_0": 5,
    "rally": 3,
    "c_5_5": 2,
    "c_0_0": 4,
    "c_-5_0": 1,
    "c_0_5": 3,
    "c_10_10": 1
  },
  "gain_vs_rally_rule_report_only": {
    "mean": 78.54740260362755,
    "without_top2": 65.8235884446324,
    "openings_beaten_by_2": [
      294000,
      294001,
      294003,
      294006,
      294008,
      294009,
      294010,
      294011,
      294014,
      294015,
      294016,
      294017,
      294018,
      294019,
      294020,
      294021,
      294022,
      294023
    ]
  },
  "held_or_rejected_fraction": 0.04853723404255319,
  "errors": [],
  "gates": {
    "A1_mechanics": true,
    "A2_state_dependence": true,
    "A3_spread": true
  },
  "passed": true
}
```

模型ZIP不入Git，活动HEAD/index保持冻结。B训练过程中无科学模块或参数修改。
