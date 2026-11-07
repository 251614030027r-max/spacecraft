# T2.6 时间压力测试：A

生成2026-10-27T16:28:33.044115+09:00；科学`f4d660aa820a2e3b08c8be310aeac7a55d641318`；执行书collab`c1a962c`。本机测试结果见PYTEST_VERIFIED（预期41）。仅改变紧急任务截止时间区间，其余算法仍t2-learner-2.5.1；任务分配/分位、经济任务、动力学、执行器、动作、奖励和约束冻结。

## A 原生筛查与选档
三档各9策略×24，共648例无ERROR。5个阶段动作空间内固定策略、纯MPC、内部G★参照、标准任务SAC/EACC s0的4k探针。p170是执行书要求的本机同提交同期对照，未重训标准任务。
```json
{
  "profiles": {
    "p170": {
      "openings": 24,
      "urgent_openings": 15,
      "deadlines_urgent": [
        200.2,
        172.4,
        186.0,
        200.0,
        183.2,
        200.7,
        181.2,
        189.0,
        177.6,
        191.6,
        186.0,
        183.6,
        208.9,
        181.9,
        196.9
      ],
      "policies": {
        "c_10_-5": {
          "n": 24,
          "mean_return": -1.6095204166378576,
          "capture": 17,
          "unsafe": 0,
          "mean_delta_v_success": 1.3643018863577892,
          "mean_time_success": 133.74117647058821,
          "events": {
            "success": 17,
            "timeout": 7
          },
          "urgent_capture": 8
        },
        "c_10_0": {
          "n": 24,
          "mean_return": 17.22068186827077,
          "capture": 19,
          "unsafe": 0,
          "mean_delta_v_success": 1.3591462693690168,
          "mean_time_success": 128.38947368421054,
          "events": {
            "success": 19,
            "timeout": 5
          },
          "urgent_capture": 10
        },
        "c_10_5": {
          "n": 24,
          "mean_return": 16.986973286046613,
          "capture": 19,
          "unsafe": 0,
          "mean_delta_v_success": 1.4500048626286208,
          "mean_time_success": 134.02105263157898,
          "events": {
            "timeout": 5,
            "success": 19
          },
          "urgent_capture": 10
        },
        "c_6_5": {
          "n": 24,
          "mean_return": 7.976136130153204,
          "capture": 18,
          "unsafe": 0,
          "mean_delta_v_success": 1.4242525449869552,
          "mean_time_success": 145.1777777777778,
          "events": {
            "timeout": 6,
            "success": 18
          },
          "urgent_capture": 9
        },
        "rally": {
          "n": 24,
          "mean_return": 10.915012889967263,
          "capture": 18,
          "unsafe": 0,
          "mean_delta_v_success": 1.3797683818813284,
          "mean_time_success": 143.52777777777774,
          "events": {
            "timeout": 6,
            "success": 18
          },
          "urgent_capture": 9
        },
        "gstar": {
          "n": 24,
          "mean_return": 61.95053018126341,
          "capture": 24,
          "unsafe": 0,
          "mean_delta_v_success": 1.3556059755334129,
          "mean_time_success": 118.52499999999999,
          "events": {
            "success": 24
          },
          "urgent_capture": 15
        },
        "pure": {
          "n": 24,
          "mean_return": -48.91966696337395,
          "capture": 14,
          "unsafe": 4,
          "mean_delta_v_success": 1.8913229681494534,
          "mean_time_success": 154.79999999999998,
          "events": {
            "success": 14,
            "resource_exhausted": 5,
            "unsafe": 4,
            "timeout": 1
          },
          "urgent_capture": 10
        }
      },
      "learned": {
        "mve_s0_4000": {
          "n": 24,
          "mean_return": 54.57618756322554,
          "capture": 23,
          "unsafe": 0,
          "mean_delta_v_success": 1.3010467064966178,
          "mean_time_success": 125.51304347826088,
          "events": {
            "success": 23,
            "timeout": 1
          },
          "urgent_capture": 14
        },
        "sac_s0_4000": {
          "n": 24,
          "mean_return": 54.47660355899104,
          "capture": 23,
          "unsafe": 0,
          "mean_delta_v_success": 1.3028055172793735,
          "mean_time_success": 127.01739130434781,
          "events": {
            "success": 23,
            "timeout": 1
          },
          "urgent_capture": 14
        }
      },
      "uniform_best": "c_10_0",
      "reach_in_space": 24,
      "reach_any": 24,
      "urgent_reach_any": 15,
      "unreachable": [],
      "headroom_capture": 5,
      "decision_value": 46.67146862114408,
      "decision_value_without_top1": 39.17281459202578,
      "unsafe_in_space": 0,
      "unsafe_learned": 0,
      "learned_urgent_capture": 28,
      "gstar_unreached_in_time": []
    },
    "p155": {
      "openings": 24,
      "urgent_openings": 15,
      "deadlines_urgent": [
        185.2,
        157.4,
        171.0,
        185.0,
        168.2,
        185.7,
        166.2,
        174.0,
        162.6,
        176.6,
        171.0,
        168.6,
        193.9,
        166.9,
        181.9
      ],
      "policies": {
        "c_10_-5": {
          "n": 24,
          "mean_return": -8.852267863532001,
          "capture": 16,
          "unsafe": 0,
          "mean_delta_v_success": 1.3332679020998155,
          "mean_time_success": 131.39375,
          "events": {
            "success": 16,
            "timeout": 8
          },
          "urgent_capture": 7
        },
        "c_10_0": {
          "n": 24,
          "mean_return": 1.0199498435453147,
          "capture": 17,
          "unsafe": 0,
          "mean_delta_v_success": 1.3103783064926278,
          "mean_time_success": 123.43529411764706,
          "events": {
            "success": 17,
            "timeout": 7
          },
          "urgent_capture": 8
        },
        "c_10_5": {
          "n": 24,
          "mean_return": 0.886250142390613,
          "capture": 17,
          "unsafe": 0,
          "mean_delta_v_success": 1.3807379036747032,
          "mean_time_success": 128.0705882352941,
          "events": {
            "timeout": 7,
            "success": 17
          },
          "urgent_capture": 8
        },
        "c_6_5": {
          "n": 24,
          "mean_return": -16.3045653711045,
          "capture": 15,
          "unsafe": 0,
          "mean_delta_v_success": 1.3694521970554916,
          "mean_time_success": 139.14,
          "events": {
            "timeout": 9,
            "success": 15
          },
          "urgent_capture": 6
        },
        "rally": {
          "n": 24,
          "mean_return": 3.565168926436499,
          "capture": 17,
          "unsafe": 0,
          "mean_delta_v_success": 1.341940459018442,
          "mean_time_success": 140.31176470588233,
          "events": {
            "timeout": 7,
            "success": 17
          },
          "urgent_capture": 8
        },
        "gstar": {
          "n": 24,
          "mean_return": 53.800193703662295,
          "capture": 23,
          "unsafe": 0,
          "mean_delta_v_success": 1.3367758461698664,
          "mean_time_success": 116.51304347826088,
          "events": {
            "success": 23,
            "timeout": 1
          },
          "urgent_capture": 14
        },
        "pure": {
          "n": 24,
          "mean_return": -57.11535786252265,
          "capture": 13,
          "unsafe": 4,
          "mean_delta_v_success": 1.8098295732062972,
          "mean_time_success": 151.13076923076923,
          "events": {
            "success": 13,
            "resource_exhausted": 5,
            "unsafe": 4,
            "timeout": 2
          },
          "urgent_capture": 9
        }
      },
      "learned": {
        "mve_s0_4000": {
          "n": 24,
          "mean_return": 46.49563097915067,
          "capture": 22,
          "unsafe": 0,
          "mean_delta_v_success": 1.2868050788170062,
          "mean_time_success": 123.80454545454546,
          "events": {
            "success": 22,
            "timeout": 2
          },
          "urgent_capture": 13
        },
        "sac_s0_4000": {
          "n": 24,
          "mean_return": 54.32821535139802,
          "capture": 23,
          "unsafe": 0,
          "mean_delta_v_success": 1.3242741413218906,
          "mean_time_success": 128.35217391304346,
          "events": {
            "success": 23,
            "timeout": 1
          },
          "urgent_capture": 14
        }
      },
      "uniform_best": "rally",
      "reach_in_space": 21,
      "reach_any": 23,
      "urgent_reach_any": 14,
      "unreachable": [
        294017
      ],
      "headroom_capture": 4,
      "decision_value": 36.703685710117796,
      "decision_value_without_top1": 29.28023531952632,
      "unsafe_in_space": 0,
      "unsafe_learned": 0,
      "learned_urgent_capture": 27,
      "gstar_unreached_in_time": [
        294003
      ],
      "criteria": {
        "F1_reference": true,
        "F2_reachable": true,
        "F3_safe": true,
        "D1_pressure": false
      }
    },
    "p140": {
      "openings": 24,
      "urgent_openings": 15,
      "deadlines_urgent": [
        170.2,
        142.4,
        156.0,
        170.0,
        153.2,
        170.7,
        151.2,
        159.0,
        147.6,
        161.6,
        156.0,
        153.6,
        178.9,
        151.9,
        166.9
      ],
      "policies": {
        "c_10_-5": {
          "n": 24,
          "mean_return": -16.11284448058616,
          "capture": 15,
          "unsafe": 0,
          "mean_delta_v_success": 1.3281200456346292,
          "mean_time_success": 130.52,
          "events": {
            "success": 15,
            "timeout": 9
          },
          "urgent_capture": 6
        },
        "c_10_0": {
          "n": 24,
          "mean_return": -14.327474645142402,
          "capture": 15,
          "unsafe": 0,
          "mean_delta_v_success": 1.2328449617569606,
          "mean_time_success": 118.64,
          "events": {
            "success": 15,
            "timeout": 9
          },
          "urgent_capture": 6
        },
        "c_10_5": {
          "n": 24,
          "mean_return": -14.832426652236776,
          "capture": 15,
          "unsafe": 0,
          "mean_delta_v_success": 1.368969938518403,
          "mean_time_success": 123.94666666666667,
          "events": {
            "timeout": 9,
            "success": 15
          },
          "urgent_capture": 6
        },
        "c_6_5": {
          "n": 24,
          "mean_return": -31.872075587811988,
          "capture": 13,
          "unsafe": 0,
          "mean_delta_v_success": 1.3112934398410583,
          "mean_time_success": 135.84615384615384,
          "events": {
            "timeout": 11,
            "success": 13
          },
          "urgent_capture": 4
        },
        "rally": {
          "n": 24,
          "mean_return": -44.75161961943268,
          "capture": 11,
          "unsafe": 0,
          "mean_delta_v_success": 1.2755588728373997,
          "mean_time_success": 137.56363636363633,
          "events": {
            "timeout": 13,
            "success": 11
          },
          "urgent_capture": 2
        },
        "gstar": {
          "n": 24,
          "mean_return": 43.048826064516014,
          "capture": 21,
          "unsafe": 0,
          "mean_delta_v_success": 1.2617321188702433,
          "mean_time_success": 112.48095238095236,
          "events": {
            "success": 21,
            "no_plan": 1,
            "timeout": 2
          },
          "urgent_capture": 12
        },
        "pure": {
          "n": 24,
          "mean_return": -73.44071047091094,
          "capture": 11,
          "unsafe": 4,
          "mean_delta_v_success": 1.7053795029233079,
          "mean_time_success": 149.11818181818182,
          "events": {
            "success": 11,
            "resource_exhausted": 3,
            "unsafe": 4,
            "timeout": 6
          },
          "urgent_capture": 7
        }
      },
      "learned": {
        "mve_s0_4000": {
          "n": 24,
          "mean_return": 38.30216679653197,
          "capture": 21,
          "unsafe": 0,
          "mean_delta_v_success": 1.282680822541179,
          "mean_time_success": 122.00952380952383,
          "events": {
            "success": 21,
            "timeout": 3
          },
          "urgent_capture": 12
        },
        "sac_s0_4000": {
          "n": 24,
          "mean_return": 12.23876651323008,
          "capture": 18,
          "unsafe": 0,
          "mean_delta_v_success": 1.2486135963375515,
          "mean_time_success": 122.42222222222222,
          "events": {
            "success": 18,
            "timeout": 6
          },
          "urgent_capture": 9
        }
      },
      "uniform_best": "c_10_0",
      "reach_in_space": 21,
      "reach_any": 22,
      "urgent_reach_any": 13,
      "unreachable": [
        294003,
        294017
      ],
      "headroom_capture": 6,
      "decision_value": 54.40268508219399,
      "decision_value_without_top1": 47.596265504012266,
      "unsafe_in_space": 0,
      "unsafe_learned": 0,
      "learned_urgent_capture": 21,
      "gstar_unreached_in_time": [
        294003,
        294013,
        294018
      ],
      "criteria": {
        "F1_reference": false,
        "F2_reachable": true,
        "F3_safe": true,
        "D1_pressure": true
      }
    }
  },
  "selected": null,
  "tightest_feasible": "p155",
  "order": [
    "p170",
    "p155",
    "p140"
  ]
}
```
F1–F3/D1和selected/tightest_feasible以原生文件为准；没有合格档就停，不因收益或方便手选档。F2用阶段动作空间内固定策略或学习探针的可达并集，不用G★替代可达性。

## 后续范围
原生selected为空则不启动B/C，报告tightest_feasible后停止；若非空，本次A推送成功且日志门通过后，已授权自动启动两臂×两种子各3000、各2worker，再7点×24共672评估与原生t26判定，不再另请批准。

## 标准任务固定对照与日志边界
```json
{
  "science_publication": "62d2d72570b0c214d2c194a9106b237554d39805",
  "collab_publication": "e9cf679be63cd19e21f4db264892a357c1b58af7",
  "standard_4k": {
    "e_mve_s0": 54.57618756322554,
    "e_mve_s1": 53.03017341449411,
    "e_sac_s0": 54.47660355899104,
    "capture_each": 23,
    "n_each": 24,
    "unsafe_each": 0
  },
  "verdict": "mixed",
  "paired_checkpoint_mean_s0": 3.9426856623783,
  "paired_checkpoint_mean_s1": -0.81791126572573,
  "first_model_row_book_reference": [
    512,
    537
  ],
  "sources": [
    {
      "path": "C:\\Users\\35884\\Documents\\Spacecraft\\过程文件\\T2_5_1_VALIDATE_20261026\\B_READOUT.json",
      "sha256": "e2d69f48bdb0b960568b6d936732f9898a0c0b8bad48efaa530025598d762178"
    },
    {
      "path": "C:\\Users\\35884\\Documents\\Spacecraft\\过程文件\\T2_5_1_VALIDATE_20261026\\DECISION.json",
      "sha256": "d23bd38c1357930d9ffda0289b586b760599e1ac1269cfceedfd4542c42db5ba"
    },
    {
      "path": "C:\\Users\\35884\\Documents\\Spacecraft\\过程文件\\T2_5_1_VALIDATE_20261026\\TRAIN_e_mve_s0\\model_stats.json",
      "sha256": "0dd73a4f9c2e4c6fd8bdc07f0cfdcb6595dd9ae91e34f08ba86cc32b3ee23f22"
    },
    {
      "path": "C:\\Users\\35884\\Documents\\Spacecraft\\过程文件\\T2_5_1_VALIDATE_20261026\\TRAIN_e_mve_s1\\model_stats.json",
      "sha256": "88ac7ce3ac9fb8f621d4404a22cb87f64a6c039de7b1234272d66d2b38c7454e"
    }
  ]
}
```
滚动日志约30秒准确决策/更新/回合/吞吐（ETA另标估计），逐回合真实和训练回报、成功率及实际原生critic/min-Q和模型反馈。独立Q1/Q2未提供则MISSING，不额外forward/backward/RNG。外部适配器新增urgent_deadline参数只原值透传；入场双臂实际pressure smoke及native update/RNG验证见LOGGING_VERIFIED。SAC无候选/反馈模型项关闭，不伪造模型探针。
本轮Windows后台与Codex生命周期解耦，全部隐藏无密码无提权；实际墙钟见finished和LAUNCH/EXIT，不把并行墙钟相加。任何重启/错误原件保全，训练真崩溃同配置从零最多一次，纯日志/发布失败不重训。阶段齐备才判定，缺项MISSING；无后续分支或正式长训。
模型ZIP/Replay/TB二进制本机保留不入Git；发布必要原始A648与C672（若已完成）、4run训练5日志、原生读出/判定、机制及激活步数、滚动JSON/TB解析/外部源码验收/pytest/源SHA/异常与真实墙钟，普通push保护活动HEAD/index。

尾部加速偏离：原p170只读筛查2路成为瓶颈后，核对PID/创建时间并SHA保全全部完成JSON，受控停止该只读池；当时未完成案例由动态池按相同参数补齐。无训练停止或重启、没有更改样本和选择规则；完整记录见A_TAIL_PARALLEL_HANDOFF。
