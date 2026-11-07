# T1-B V8 阶段 I（R1）实测交付

冻结 9183da2；collab 执行单 7e0d5a8。只执行阶段 I，未启动阶段 II。G0–G3：{"G0": true, "G1": true, "G2": true, "G3": false}。

实际完成 risk 36/64，random 36/64；配对交集 36/64。截止原因：85min_or_total_wallclock_cutoff。没有补跑或挑选；G2/G3 仅在完成交集上计算。少量完成开局不能视为总体性能已验证。

G0 人工核对：{"passed": true, "reviewer": "Codex manual review of raw pytest and contract logs", "pytest_passed": 17, "warnings": 3, "pytest_exit": 0, "contract_exit": 0, "contract_passed": true, "G0_requires_smoke_also": true, "warning_details": "Torch tensor scalar conversion and NumPy array-to-scalar deprecation; no failures", "sha256": {"pytest": "76fbe10a7af1ad7990651165b92c4fcbb207b263b1869c43367d7ee35cf16a81", "contract": "1b50d1b358917108411953ec34e2d1ec9ab2e995972ea595fa00ad7262615e19"}, "reviewed_jst": "2026-10-21T17:25:07.804310"}；原生 G0_smoke=True。pytest/契约和冒烟共同决定最终 G0。

Research review only; no Stage II authorized. NO-GO under registered gates.

阶段 I 对 ECPG 训练期独立贡献 P4 没有证据；冒烟只证明机械链路。等待研究层复核和用户批准，不自动长训。

## 固定阈值风险行为（q_theta=0，其他阈值仅报告）
```json
{
  "auc": 0.8993197278911564,
  "behaviour": {
    "states": 64,
    "threshold": 0.0,
    "classes": {
      "rescue": {
        "states": 1,
        "triggered": 1,
        "by_deviation": 0,
        "by_risk": 1,
        "q_min_earliest": {
          "p0": -12.391685485839844,
          "p50": -12.391685485839844,
          "p100": -12.391685485839844
        },
        "openings": [
          293117
        ]
      },
      "all_fail": {
        "states": 14,
        "triggered": 11,
        "by_deviation": 0,
        "by_risk": 11,
        "q_min_earliest": {
          "p0": -149.248046875,
          "p50": -94.09261703491211,
          "p100": 59.23582077026367
        },
        "openings": [
          292502,
          292504,
          292505,
          292507,
          292512,
          292518,
          292523,
          293101,
          293112,
          293113,
          293115,
          293116,
          293120,
          293124
        ]
      },
      "better_alternative": {
        "states": 19,
        "triggered": 5,
        "by_deviation": 0,
        "by_risk": 5,
        "q_min_earliest": {
          "p0": -57.76460647583008,
          "p50": 56.7065315246582,
          "p100": 72.1897964477539
        },
        "openings": [
          292508,
          292513,
          292514,
          292515,
          292519,
          292522,
          292524,
          292529,
          292531,
          293100,
          293104,
          293107,
          293110,
          293114,
          293118,
          293121,
          293123,
          293125,
          293128
        ]
      },
      "earliest_best": {
        "states": 30,
        "triggered": 6,
        "by_deviation": 0,
        "by_risk": 6,
        "q_min_earliest": {
          "p0": -71.51851654052734,
          "p50": 57.04055213928223,
          "p100": 90.79307556152344
        },
        "openings": [
          292500,
          292501,
          292503,
          292506,
          292509,
          292510,
          292511,
          292516,
          292517,
          292520,
          292521,
          292525,
          292526,
          292527,
          292528,
          292530,
          293102,
          293103,
          293105,
          293106,
          293108,
          293109,
          293111,
          293119,
          293122,
          293126,
          293127,
          293129,
          293130,
          293131
        ]
      }
    },
    "triggered_openings": [
      {
        "opening": 292504,
        "reason": "risk",
        "class": "all_fail",
        "proposal": 0,
        "q_min": [
          -122.4,
          -133.26,
          -142.76
        ],
        "truth_events": [
          "resource_exhausted",
          "resource_exhausted",
          "resource_exhausted"
        ],
        "truth_returns": [
          -132.33,
          -132.2,
          -132.1
        ]
      },
      {
        "opening": 292505,
        "reason": "risk",
        "class": "all_fail",
        "proposal": 0,
        "q_min": [
          -107.48,
          -123.93,
          -135.06
        ],
        "truth_events": [
          "resource_exhausted",
          "resource_exhausted",
          "resource_exhausted"
        ],
        "truth_returns": [
          -132.77,
          -132.74,
          -132.7
        ]
      },
      {
        "opening": 292507,
        "reason": "risk",
        "class": "all_fail",
        "proposal": 0,
        "q_min": [
          -133.14,
          -145.71,
          -156.45
        ],
        "truth_events": [
          "resource_exhausted",
          "resource_exhausted",
          "resource_exhausted"
        ],
        "truth_returns": [
          -126.73,
          -126.75,
          -126.72
        ]
      },
      {
        "opening": 292509,
        "reason": "risk",
        "class": "earliest_best",
        "proposal": 0,
        "q_min": [
          -8.54,
          -51.71
        ],
        "truth_events": [
          "success",
          "resource_exhausted"
        ],
        "truth_returns": [
          58.44,
          -131.63
        ]
      },
      {
        "opening": 292512,
        "reason": "risk",
        "class": "all_fail",
        "proposal": 0,
        "q_min": [
          -149.25,
          -162.05
        ],
        "truth_events": [
          "timeout",
          "timeout"
        ],
        "truth_returns": [
          -130.22,
          -128.29
        ]
      },
      {
        "opening": 292518,
        "reason": "risk",
        "class": "all_fail",
        "proposal": 0,
        "q_min": [
          -127.32,
          -142.4,
          -153.91
        ],
        "truth_events": [
          "resource_exhausted",
          "resource_exhausted",
          "resource_exhausted"
        ],
        "truth_returns": [
          -127.09,
          -126.9,
          -126.93
        ]
      },
      {
        "opening": 292519,
        "reason": "risk",
        "class": "better_alternative",
        "proposal": 0,
        "q_min": [
          -57.76,
          -72.21,
          -86.65
        ],
        "truth_events": [
          "success",
          "success",
          "success"
        ],
        "truth_returns": [
          67.18,
          70.55,
          71.26
        ]
      },
      {
        "opening": 292521,
        "reason": "risk",
        "class": "earliest_best",
        "proposal": 0,
        "q_min": [
          -18.61,
          -30.72,
          -42.33
        ],
        "truth_events": [
          "success",
          "success",
          "timeout"
        ],
        "truth_returns": [
          54.64,
          50.59,
          -130.84
        ]
      },
      {
        "opening": 292522,
        "reason": "risk",
        "class": "better_alternative",
        "proposal": 0,
        "q_min": [
          -52.37,
          -59.2,
          -65.55
        ],
        "truth_events": [
          "success",
          "success",
          "timeout"
        ],
        "truth_returns": [
          59.85,
          62.81,
          -118.79
        ]
      },
      {
        "opening": 292524,
        "reason": "risk",
        "class": "better_alternative",
        "proposal": 0,
        "q_min": [
          -0.81,
          -7.99,
          -15.08
        ],
        "truth_events": [
          "success",
          "success",
          "success"
        ],
        "truth_returns": [
          58.4,
          60.28,
          62.94
        ]
      },
      {
        "opening": 292527,
        "reason": "risk",
        "class": "earliest_best",
        "proposal": 0,
        "q_min": [
          -3.92,
          -18.58,
          -33.56
        ],
        "truth_events": [
          "success",
          "success",
          "success"
        ],
        "truth_returns": [
          57.2,
          56.16,
          56.15
        ]
      },
      {
        "opening": 292529,
        "reason": "risk",
        "class": "better_alternative",
        "proposal": 0,
        "q_min": [
          -16.11,
          -25.23,
          -35.16
        ],
        "truth_events": [
          "success",
          "success",
          "success"
        ],
        "truth_returns": [
          61.71,
          62.23,
          66.42
        ]
      },
      {
        "opening": 292530,
        "reason": "risk",
        "class": "earliest_best",
        "proposal": 0,
        "q_min": [
          -71.52,
          -85.97,
          -100.1
        ],
        "truth_events": [
          "success",
          "success",
          "timeout"
        ],
        "truth_returns": [
          56.86,
          57.17,
          -128.79
        ]
      },
      {
        "opening": 293100,
        "reason": "risk",
        "class": "better_alternative",
        "proposal": 0,
        "q_min": [
          -48.73,
          -57.23,
          -67.63
        ],
        "truth_events": [
          "success",
          "success",
          "success"
        ],
        "truth_returns": [
          56.15,
          56.82,
          62.79
        ]
      },
      {
        "opening": 293101,
        "reason": "risk",
        "class": "all_fail",
        "proposal": 0,
        "q_min": [
          -127.82,
          -139.41,
          -147.73
        ],
        "truth_events": [
          "resource_exhausted",
          "resource_exhausted",
          "resource_exhausted"
        ],
        "truth_returns": [
          -130.29,
          -130.16,
          -130.1
        ]
      },
      {
        "opening": 293112,
        "reason": "risk",
        "class": "all_fail",
        "proposal": 0,
        "q_min": [
          -80.7,
          -91.38,
          -102.1
        ],
        "truth_events": [
          "resource_exhausted",
          "resource_exhausted",
          "resource_exhausted"
        ],
        "truth_returns": [
          -131.71,
          -131.57,
          -131.52
        ]
      },
      {
        "opening": 293113,
        "reason": "risk",
        "class": "all_fail",
        "proposal": 0,
        "q_min": [
          -16.48,
          -26.05,
          -34.53
        ],
        "truth_events": [
          "resource_exhausted",
          "timeout",
          "timeout"
        ],
        "truth_returns": [
          -140.05,
          -138.57,
          -130.13
        ]
      },
      {
        "opening": 293115,
        "reason": "risk",
        "class": "all_fail",
        "proposal": 0,
        "q_min": [
          -114.42,
          -127.14
        ],
        "truth_events": [
          "timeout",
          "timeout"
        ],
        "truth_returns": [
          -132.68,
          -134.4
        ]
      },
      {
        "opening": 293116,
        "reason": "risk",
        "class": "all_fail",
        "proposal": 0,
        "q_min": [
          -76.78,
          -86.1,
          -95.42
        ],
        "truth_events": [
          "resource_exhausted",
          "resource_exhausted",
          "resource_exhausted"
        ],
        "truth_returns": [
          -130.45,
          -130.37,
          -130.43
        ]
      },
      {
        "opening": 293117,
        "reason": "risk",
        "class": "rescue",
        "proposal": 0,
        "q_min": [
          -12.39,
          -22.15,
          -32.73
        ],
        "truth_events": [
          "resource_exhausted",
          "unsafe",
          "success"
        ],
        "truth_returns": [
          -131.69,
          -177.64,
          53.12
        ]
      },
      {
        "opening": 293120,
        "reason": "risk",
        "class": "all_fail",
        "proposal": 0,
        "q_min": [
          -35.61,
          -55.28,
          -76.63
        ],
        "truth_events": [
          "timeout",
          "timeout",
          "resource_exhausted"
        ],
        "truth_returns": [
          -129.93,
          -136.17,
          -136.86
        ]
      },
      {
        "opening": 293122,
        "reason": "risk",
        "class": "earliest_best",
        "proposal": 0,
        "q_min": [
          -33.71,
          -45.7,
          -58.73
        ],
        "truth_events": [
          "success",
          "timeout",
          "timeout"
        ],
        "truth_returns": [
          58.5,
          -123.98,
          -118.54
        ]
      },
      {
        "opening": 293126,
        "reason": "risk",
        "class": "earliest_best",
        "proposal": 0,
        "q_min": [
          -27.84,
          -44.11,
          -60.34
        ],
        "truth_events": [
          "success",
          "success",
          "resource_exhausted"
        ],
        "truth_returns": [
          65.33,
          61.05,
          -127.84
        ]
      }
    ],
    "triggered": 23,
    "trigger_rate": 0.359375,
    "earliest_failure_recall": 0.8,
    "triggered_share_on_earliest_failures": 0.5217391304347826,
    "rescue_triggered": {
      "293117": true
    },
    "other_thresholds_report_only": {
      "-40.0": {
        "trigger_rate": 0.203125,
        "earliest_failure_recall": 0.6
      },
      "-20.0": {
        "trigger_rate": 0.25,
        "earliest_failure_recall": 0.6666666666666666
      },
      "0.0": {
        "trigger_rate": 0.359375,
        "earliest_failure_recall": 0.8
      },
      "20.0": {
        "trigger_rate": 0.421875,
        "earliest_failure_recall": 0.8666666666666667
      },
      "40.0": {
        "trigger_rate": 0.46875,
        "earliest_failure_recall": 0.9333333333333333
      }
    }
  },
  "offline_rho_for_random_arm": 0.359375
}
```

阈值作用请结合上面的四类状态、触发原因与开局清单判断；不按本次读数调整阈值。真值参照采用固定教师后续，并非当前策略严格 Q。

## 同开局闭环、计算与延迟
```json
{
  "risk_arm": {
    "openings": 36,
    "return": 21.60405812491174,
    "baseline_off_return": 16.285968284258004,
    "paired_delta_vs_off": 5.3180898406537365,
    "capture": 28,
    "baseline_capture": 27,
    "unsafe": 0,
    "baseline_unsafe": 0,
    "learning_decisions": 67,
    "triggered_decisions": 19,
    "trigger_rate": 0.2835820895522388,
    "trigger_reasons": {
      "deviation": 0,
      "risk": 19,
      "random": 0,
      "all": 0
    },
    "second_or_later_decisions_triggered": 9,
    "model_branches_per_episode": 1.5,
    "changed_choice_after_verification": 3,
    "latency_s_triggered_decision_parallel_branches": {
      "p50": 194.6769278049469,
      "p95": 343.9529369831085,
      "p100": 359.3257977962494
    },
    "lead_time_s_executed_intent": {
      "p0": 60.0,
      "p50": 80.0,
      "p100": 140.0
    },
    "latency_over_lead": {
      "p50": 1.8120580764917227,
      "p95": 2.7728441730370887,
      "p100": 2.8520403444766997
    },
    "events": {
      "292500": "success",
      "292501": "success",
      "292502": "resource_exhausted",
      "292503": "success",
      "293100": "success",
      "293101": "resource_exhausted",
      "293102": "success",
      "293103": "success",
      "293104": "success",
      "293105": "success",
      "293106": "success",
      "293107": "success",
      "293108": "success",
      "293109": "success",
      "293110": "success",
      "293111": "success",
      "293112": "resource_exhausted",
      "293113": "resource_exhausted",
      "293114": "success",
      "293115": "timeout",
      "293116": "resource_exhausted",
      "293117": "success",
      "293118": "success",
      "293119": "success",
      "293120": "timeout",
      "293121": "success",
      "293122": "success",
      "293123": "success",
      "293124": "timeout",
      "293125": "success",
      "293126": "success",
      "293127": "success",
      "293128": "success",
      "293129": "success",
      "293130": "success",
      "293131": "success"
    }
  },
  "random_arm": {
    "openings": 36,
    "return": 21.85459061468209,
    "baseline_off_return": 16.285968284258004,
    "paired_delta_vs_off": 5.568622330424081,
    "capture": 28,
    "baseline_capture": 27,
    "unsafe": 0,
    "baseline_unsafe": 0,
    "learning_decisions": 67,
    "triggered_decisions": 26,
    "trigger_rate": 0.3880597014925373,
    "trigger_reasons": {
      "deviation": 0,
      "risk": 0,
      "random": 26,
      "all": 0
    },
    "second_or_later_decisions_triggered": 14,
    "model_branches_per_episode": 2.0277777777777777,
    "changed_choice_after_verification": 5,
    "latency_s_triggered_decision_parallel_branches": {
      "p50": 155.62482368946075,
      "p95": 316.0100459456444,
      "p100": 350.2916622161865
    },
    "lead_time_s_executed_intent": {
      "p0": 25.0,
      "p50": 80.0,
      "p100": 140.0
    },
    "latency_over_lead": {
      "p50": 2.3493508378664654,
      "p95": 3.5510735425353053,
      "p100": 3.85865004658699
    },
    "events": {
      "292500": "success",
      "292501": "success",
      "292502": "resource_exhausted",
      "292503": "success",
      "293100": "success",
      "293101": "resource_exhausted",
      "293102": "success",
      "293103": "success",
      "293104": "success",
      "293105": "success",
      "293106": "success",
      "293107": "success",
      "293108": "success",
      "293109": "success",
      "293110": "success",
      "293111": "success",
      "293112": "resource_exhausted",
      "293113": "resource_exhausted",
      "293114": "success",
      "293115": "timeout",
      "293116": "resource_exhausted",
      "293117": "success",
      "293118": "success",
      "293119": "success",
      "293120": "timeout",
      "293121": "success",
      "293122": "success",
      "293123": "success",
      "293124": "timeout",
      "293125": "success",
      "293126": "success",
      "293127": "success",
      "293128": "success",
      "293129": "success",
      "293130": "success",
      "293131": "success"
    },
    "paired_delta_risk_minus_random": -0.2505324897703465
  },
  "serial_branch_cost": {
    "branches": 188,
    "wall_s_one_branch_serial": {
      "p50": 160.6348127500023,
      "p95": 225.9597208499989,
      "p100": 1616.9673239999975
    }
  },
  "smoke": {
    "preference_complete": 2,
    "ce_probes": 10,
    "ce_finite": true
  },
  "gates": {
    "G0_smoke": true,
    "G1_auc": true,
    "G1_rescue_triggered": true,
    "G1_failure_recall": true,
    "G1_rate": true,
    "G2_return": true,
    "G2_capture_unsafe": true,
    "G2_rate": true,
    "G3_paired_openings": 36,
    "G3_branch_ratio_random_over_risk": 1.3518518518518519,
    "G3_equal_compute": false,
    "G3_learned_beats_random": false
  }
}
```

延迟是在两臂各 2 回合、每回合 3 分支并行负载下测得，机器 12 逻辑核；冒烟是否重叠以各 LAUNCH/EXIT 为准。单分支串行成本来自既有冻结模型扫描，不能视为本轮独占硬件基准。不声称实时可用。
未完成回合未由冻结代码持久化的分支耗费/延迟为未知，不填补；G3 只比较完成配对的每回合分支数。

逐回合原始结果、实际配对、原始命令、源资产 SHA 与计时在本轮证据中；逐例表见 ACTUAL_EPISODES.csv。

固定阈值的实际作用：23 个被触发初态中，最早候选失败类占 12 个，最早已成功类仍有 11 个进入核验。各类别触发数为 rescue:1/1，all_fail:11/14，better_alternative:5/19，earliest_best:6/30；主要触发类别与原因据上述原始分解判读，不将 AUC 等同闭环收益。

risk_arm 完成交集捕获率：28/36 = 0.7777777777777778；平均折扣原回报 21.60405812491174，每回合分支数 1.5，决策触发率 0.2835820895522388。
random_arm 完成交集捕获率：28/36 = 0.7777777777777778；平均折扣原回报 21.85459061468209，每回合分支数 2.0277777777777777，决策触发率 0.3880597014925373。
