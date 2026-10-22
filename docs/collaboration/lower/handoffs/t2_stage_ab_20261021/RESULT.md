# T2 阶段 A/B 本机交付（2026-10-22）

科学冻结 `31074206045cf62dba3d560e4442db4b00eea05a`；执行单 collab `a3135ef`。开发开局固定 294000–294023，seed0，9固定策略；仅 A 全门通过后两臂各3000次阶段决策。未启动正式多种子长训。

实际停止原因：registered_A_B_complete。A 完成 216/216；B 完成 192/192（A失败则未授权B）。所有 ERROR/缺项保留，不以排除失败后的统计冒充完整结果。

## A 三门与规则余量
```json
{
  "gates": {
    "A1_mechanics": true,
    "A2_state_dependence": true,
    "A3_spread": true
  },
  "passed": true,
  "held_or_rejected_fraction": 0.04853723404255319,
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
  "errors": []
}
```

完整逐开局9策略表见 A_EPISODES.csv；原始 A_screen/*.json 随本轮提供。rally 属动作空间内的强规则，gstar 只作空间外参照；不混入可学习策略最优值。

## B 学习曲线与同任务配对

| 臂 | 阶段决策 | 完成开局 | 平均原回报 | 捕获 | unsafe | 成功Δv | 成功用时 |
|---|---:|---:|---:|---:|---:|---:|---:|
| e_sac | 0 | 24 | -51.14908425762051 | 11 | 0 | 1.2869952641745466 | 148.32727272727274 |
| e_sac | 1000 | 24 | -5.340011296967423 | 17 | 0 | 1.4464318697374867 | 132.36470588235295 |
| e_sac | 2000 | 24 | -2.027685354583383 | 17 | 0 | 1.4252196482908954 | 134.08235294117648 |
| e_sac | 3000 | 24 | 12.896978768424802 | 19 | 1 | 1.5349682525444777 | 141.4105263157895 |
| e_mve | 0 | 24 | -51.14908425762051 | 11 | 0 | 1.2869952641745466 | 148.32727272727274 |
| e_mve | 1000 | 24 | 3.9006749872453192 | 18 | 0 | 1.5142985370997541 | 138.21111111111114 |
| e_mve | 2000 | 24 | 5.994534773723108 | 18 | 0 | 1.464346958358233 | 139.98333333333332 |
| e_mve | 3000 | 24 | 23.421727291070482 | 20 | 0 | 1.5162611125748815 | 142.85000000000002 |

B逐例见 B_EPISODES.csv；两臂逐开局差见 PAIRED_EPISODES.csv，逐检查点差见 ARM_COMPARISON.csv。所有检查点完整保留，不挑最好点。
```json
{
  "reading_only": {
    "e_sac": {
      "gain_3000_vs_0": 64.04606302604532,
      "capture_0": 11,
      "capture_3000": 19,
      "reading_learning_curve_gain20_capture_nondecrease": true,
      "not_a_stop_gate": true
    },
    "e_mve": {
      "gain_3000_vs_0": 74.57081154869098,
      "capture_0": 11,
      "capture_3000": 20,
      "reading_learning_curve_gain20_capture_nondecrease": true,
      "not_a_stop_gate": true
    }
  },
  "arm_differences": [
    {
      "step": 0,
      "mve_minus_sac_mean": 0.0,
      "capture_sac": 11,
      "capture_mve": 11,
      "reading_mve_gain5": false,
      "not_a_stop_gate": true
    },
    {
      "step": 1000,
      "mve_minus_sac_mean": 9.240686284212742,
      "capture_sac": 17,
      "capture_mve": 18,
      "reading_mve_gain5": true,
      "not_a_stop_gate": true
    },
    {
      "step": 2000,
      "mve_minus_sac_mean": 8.02222012830649,
      "capture_sac": 17,
      "capture_mve": 18,
      "reading_mve_gain5": true,
      "not_a_stop_gate": true
    },
    {
      "step": 3000,
      "mve_minus_sac_mean": 10.52474852264568,
      "capture_sac": 19,
      "capture_mve": 20,
      "reading_mve_gain5": true,
      "not_a_stop_gate": true
    }
  ],
  "baselines": {
    "rally_rule": {
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
    "neutral_c_0_0": {
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
    "per_opening_best_fixed": 55.69346428213097
  },
  "model_use": {
    "TRAIN_e_mve": {
      "finished": {
        "steps": 3000,
        "updates": 2240,
        "replay": {
          "transitions": 3000,
          "with_candidates": 1378,
          "candidates": 4126
        }
      },
      "probes": 90,
      "probes_with_model_rows": 86,
      "lambda_eff_mean": 0.6335756237700239,
      "model_td_last": 47.24671936035156
    },
    "TRAIN_e_sac": {
      "finished": {
        "steps": 3000,
        "updates": 2240,
        "replay": {
          "transitions": 3000,
          "with_candidates": 1257,
          "candidates": 3759
        }
      },
      "probes": 90,
      "probes_with_model_rows": 0,
      "lambda_eff_mean": 0.0,
      "model_td_last": null
    }
  }
}
```

3000比0k高≥20且捕获不降、MVE在2k或3k比SAC高≥5均为研究层预期读法，不是停止门。单种子试验不足以证明正式统计结论。模型项使用/非零梯度不等于独立收益；必须结合逐例与配对曲线。

## 实际墙钟与处理记录
```json
{
  "started_jst": "2026-10-22T01:54:49.564448+09:00",
  "A_start_jst": "2026-10-22T01:55:20.258045+09:00",
  "A_end_jst": "2026-10-22T03:36:24.436865+09:00",
  "B_start_jst": "2026-10-22T03:36:54.470793+09:00",
  "B_training_end_jst": "2026-10-22T16:25:49.683892+09:00",
  "B_eval_end_jst": "2026-10-22T17:35:56.066752+09:00",
  "restarts": {},
  "stop_reason": "registered_A_B_complete"
}
```

驱动采用同一冻结 main 的受保护入口，避免 Windows spawn 重复执行入口；科学函数、参数和阈值未改。数值库单线程、低优先级、0xFFC预留桌面物理核；A及评估workers10，两臂各5环境。真实训练崩溃才允许同seed从零一次；若有重试，原始stderr与失败资产路径见 ASSESSMENT。原始运行命令/退出码在记录中。

模型ZIP仅本机保留，路径及SHA见 LOCAL_MODEL_ASSETS.json；不会推入Git。必要原始JSON、pytest、逻辑核与SHA256SUMS随交付，未复制历史整包。

实际阶段墙钟（秒）：{"A": 6064.17882, "B_train": 46135.213099, "B_eval_post_train_tail": 4206.38286, "B_eval_calendar_with_overlap": 27462.414193}
带候选转移比例：{"e_sac": 0.419, "e_mve": 0.4593333333333333}

只读吞吐抽样诊断（不修改科学环境；抽样不足以证明完整性能归因）：
```json
{
  "jst": "2026-10-22T07:00:20.1712246+09:00",
  "kind": "read_only_nonblocking_stack_sample",
  "tool": "py-spy 0.4.2 standalone wheel extracted under process records; science venv unmodified",
  "learner_pids": [
    3320,
    25024
  ],
  "observations": [
    "Both learners sampled waiting in SubprocVecEnv.step_wait / multiprocessing pipe recv",
    "Two successfully sampled busy environments were in MPC reference trajectory generation and prediction rollout",
    "Several other workers sampled idle awaiting the next command",
    "Worker CPU counters continued increasing; no Traceback in either training log"
  ],
  "interpretation": "Samples support expensive MPC/truth environment steps and synchronous slow-worker waiting; not evidence of optimizer failure or proof of deadlock",
  "limitations": [
    "Nonblocking reads sometimes failed with OS error299; preserved locally, not training failures",
    "Scalar locals can be torn/inaccurate; not used as physics or training-step evidence",
    "No exact num_timesteps established before first1000 checkpoint"
  ],
  "scientific_changes": false,
  "process_restart": false
}
```
