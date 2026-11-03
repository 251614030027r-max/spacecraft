# T2.11 EACC-A 本机事实交付

冻结科学 `ec394efcaaeadbb5305f4f9e289addb58517dca7`，执行书collab `b64fb92`；pytest68项通过、原生退出码0，日志门槛已在本SHA通过。四条从零训练e_acc/e_sac×种子0/1，各2000步、各2异步worker，obs=reduced_priv；共享CPU0–11，OMP/MKL/OPENBLAS=1。actor_view24；方法critic48，基线critic24。版本env0.9.0、learner3.0。

执行书标题≤10h与具体条款不一致：本轮按正文8.5h停训、10.75h后不开始新评估、12h硬上限执行，并在启动前向用户说明。外部驱动只读日志、精确1000机制快照和独立时间保护；科学源文件未改。

## 原生判定

```json
{
  "checkpoints": [
    1000,
    2000
  ],
  "runs": {
    "mve": [
      "e_acc_s0",
      "e_acc_s1"
    ],
    "sac": [
      "e_sac_s0",
      "e_sac_s1"
    ]
  },
  "criteria": {
    "C_complete": true,
    "L_learning": true,
    "R_reliability": true,
    "S_safety": true
  },
  "values": {
    "auc": {
      "mve": {
        "e_acc_s0": 22.8900124562753,
        "e_acc_s1": 28.69282915738603
      },
      "sac": {
        "e_sac_s0": -14.794000531995133,
        "e_sac_s1": -15.680390830728822
      }
    },
    "arm_mean": {
      "mve": 25.791420806830665,
      "sac": -15.237195681361978
    },
    "arm_diff": 41.02861648819264,
    "worst": {
      "mve": 22.8900124562753,
      "sac": -15.680390830728822
    },
    "eacc_above_sac_mean": 2,
    "unsafe": {
      "mve": 0,
      "sac": 0
    }
  },
  "reported": {
    "mve": {
      "500": {
        "mean_return": [
          31.70361013796636,
          25.68532464473525
        ],
        "urgent_capture": [
          19,
          18
        ],
        "economy_capture": [
          23,
          23
        ],
        "unsafe": [
          0,
          0
        ],
        "delta_v_success": [
          1.3578008527380614,
          1.407748259503126
        ],
        "time_success": [
          130.73095238095237,
          134.2878048780488
        ]
      },
      "1000": {
        "mean_return": [
          17.30615541764641,
          19.254261817021646
        ],
        "urgent_capture": [
          15,
          17
        ],
        "economy_capture": [
          23,
          22
        ],
        "unsafe": [
          0,
          0
        ],
        "delta_v_success": [
          1.2648284663064846,
          1.3386577817647756
        ],
        "time_success": [
          127.21052631578948,
          130.3923076923077
        ]
      },
      "1500": {
        "mean_return": [
          29.252797049059012,
          36.937623827056925
        ],
        "urgent_capture": [
          18,
          20
        ],
        "economy_capture": [
          23,
          23
        ],
        "unsafe": [
          0,
          0
        ],
        "delta_v_success": [
          1.3022582654851653,
          1.3428676237539303
        ],
        "time_success": [
          127.83170731707318,
          129.63488372093025
        ]
      },
      "2000": {
        "mean_return": [
          28.473869494904196,
          38.13139649775041
        ],
        "urgent_capture": [
          18,
          20
        ],
        "economy_capture": [
          23,
          23
        ],
        "unsafe": [
          0,
          0
        ],
        "delta_v_success": [
          1.3131040765894113,
          1.3114451930783262
        ],
        "time_success": [
          126.28536585365856,
          133.02325581395348
        ]
      }
    },
    "sac": {
      "500": {
        "mean_return": [
          -11.47954778812919,
          -22.4637142839082
        ],
        "urgent_capture": [
          10,
          7
        ],
        "economy_capture": [
          23,
          23
        ],
        "unsafe": [
          0,
          0
        ],
        "delta_v_success": [
          1.560751547769186,
          1.5013363405280855
        ],
        "time_success": [
          147.26363636363632,
          144.45666666666665
        ]
      },
      "1000": {
        "mean_return": [
          -24.9998221704215,
          -15.405692178999857
        ],
        "urgent_capture": [
          6,
          8
        ],
        "economy_capture": [
          23,
          23
        ],
        "unsafe": [
          0,
          0
        ],
        "delta_v_success": [
          1.4897982974977384,
          1.4463140736034077
        ],
        "time_success": [
          146.08620689655172,
          142.38387096774193
        ]
      },
      "1500": {
        "mean_return": [
          -13.572635592141927,
          -13.622909430875794
        ],
        "urgent_capture": [
          9,
          8
        ],
        "economy_capture": [
          23,
          23
        ],
        "unsafe": [
          0,
          0
        ],
        "delta_v_success": [
          1.5363088373619402,
          1.3486639305868764
        ],
        "time_success": [
          147.59375,
          138.07096774193548
        ]
      },
      "2000": {
        "mean_return": [
          -4.588178893568767,
          -15.955089482457785
        ],
        "urgent_capture": [
          11,
          8
        ],
        "economy_capture": [
          23,
          23
        ],
        "unsafe": [
          0,
          0
        ],
        "delta_v_success": [
          1.500622279583602,
          1.4206572161657145
        ],
        "time_success": [
          143.14117647058825,
          142.93870967741933
        ]
      }
    },
    "mechanism": {
      "TRAIN_e_acc_s0": {
        "two_stage_share": 0.7804146139646257,
        "decisive_share": 0.164983766682354,
        "c_mean": 0.1129437076088823,
        "grad_cos": 0.4364665200312932,
        "loss_share": 0.4006557087981288,
        "increment_abs_mean": 1.1395533699312328,
        "w_mean": 0.9425801325727392,
        "w2_mean": 0.9371485967694977,
        "pi_gap_sigma_median": 0.044379879295090095,
        "stage2_rows": 1241.7777777777778,
        "stage2_entry_share": 0.2114117863001647,
        "stage1_cutoff_share": 0.0,
        "stage2_cutoff_share": 0.019965562733741088,
        "pref_pairs": null,
        "pref_ordered": null,
        "pref_loss": null,
        "pref_share": null,
        "cal_acc": 0.9353086419753085,
        "acc_trusted": 1.0,
        "acc_rows": 1.0864197530864197,
        "acc_q_ordered": 1.0,
        "acc_q_share": 0.42161683455923055,
        "acc_bc": 0.0010798600558088497,
        "acc_actor_dist": 0.02214844535496509,
        "guided_share": 0.0
      },
      "TRAIN_e_acc_s1": {
        "two_stage_share": 0.8091643558831139,
        "decisive_share": 0.15393241697255475,
        "c_mean": 0.1809897552025907,
        "grad_cos": 0.31083450457524686,
        "loss_share": 0.42919894216466276,
        "increment_abs_mean": 2.0237904882725375,
        "w_mean": 0.9504111468056102,
        "w2_mean": 0.9597351823324038,
        "pi_gap_sigma_median": 0.03152089913226204,
        "stage2_rows": 1258.9135802469136,
        "stage2_entry_share": 0.19046501291018944,
        "stage1_cutoff_share": 0.0,
        "stage2_cutoff_share": 0.033783160627210214,
        "pref_pairs": null,
        "pref_ordered": null,
        "pref_loss": null,
        "pref_share": null,
        "cal_acc": 0.9329629629629632,
        "acc_trusted": 1.0,
        "acc_rows": 0.2962962962962963,
        "acc_q_ordered": 0.9731404943899675,
        "acc_q_share": 1.7373847714319461,
        "acc_bc": 0.01020204659367252,
        "acc_actor_dist": 0.047127012724310836,
        "guided_share": 0.0
      }
    }
  },
  "verdict": "pass"
}
```

## 千步止损检查

```json
{
  "native_quick_decision": {
    "checkpoints": [
      1000
    ],
    "runs": {
      "mve": [
        "e_acc_s0",
        "e_acc_s1"
      ],
      "sac": [
        "e_sac_s0",
        "e_sac_s1"
      ]
    },
    "criteria": {
      "C_complete": true,
      "L_learning": true,
      "R_reliability": true,
      "S_safety": true
    },
    "values": {
      "auc": {
        "mve": {
          "e_acc_s0": 12.531621371502254,
          "e_acc_s1": 17.864966016195847
        },
        "sac": {
          "e_sac_s0": -41.64901158163533,
          "e_sac_s1": -38.21887048562933
        }
      },
      "arm_mean": {
        "mve": 15.19829369384905,
        "sac": -39.93394103363233
      },
      "arm_diff": 55.13223472748138,
      "worst": {
        "mve": 12.531621371502254,
        "sac": -41.64901158163533
      },
      "eacc_above_sac_mean": 2,
      "unsafe": {
        "mve": 0,
        "sac": 0
      }
    },
    "reported": {
      "mve": {
        "1000": {
          "mean_return": [
            12.531621371502254,
            17.864966016195847
          ],
          "urgent_capture": [
            9,
            11
          ],
          "economy_capture": [
            9,
            8
          ],
          "unsafe": [
            0,
            0
          ],
          "delta_v_success": [
            1.243309724341794,
            1.3253125099885852
          ],
          "time_success": [
            124.42777777777776,
            130.85789473684213
          ]
        }
      },
      "sac": {
        "1000": {
          "mean_return": [
            -41.64901158163533,
            -38.21887048562933
          ],
          "urgent_capture": [
            3,
            3
          ],
          "economy_capture": [
            9,
            9
          ],
          "unsafe": [
            0,
            0
          ],
          "delta_v_success": [
            1.4248606191977562,
            1.3170009819393675
          ],
          "time_success": [
            138.58333333333334,
            125.89166666666665
          ]
        }
      }
    },
    "verdict": "pass"
  },
  "exact1000_label_evidence": {
    "TRAIN_e_acc_s0": {
      "steps": 1000,
      "updates": 1602,
      "replay": {
        "transitions": 1000,
        "with_candidates": 701,
        "candidates": 5608,
        "entry_candidates": 930,
        "reason": {
          "off": 11,
          "entry_calibration": 245,
          "stage": 578,
          "entry": 123,
          "not_comparable": 43
        },
        "pref_decisions": 744,
        "cutoff_candidates": 0,
        "two_stage": 4449,
        "two_stage_entry": 847,
        "two_stage_cutoff": 136
      },
      "checkpoint": "C:\\Users\\35884\\Documents\\Spacecraft\\过程文件\\T2_11_EACCA_20261102\\B\\TRAIN_e_acc_s0\\checkpoints\\step_1000"
    },
    "TRAIN_e_acc_s1": {
      "steps": 1000,
      "updates": 1602,
      "replay": {
        "transitions": 1000,
        "with_candidates": 676,
        "candidates": 5408,
        "entry_candidates": 827,
        "reason": {
          "off": 11,
          "entry_calibration": 271,
          "stage": 564,
          "entry": 112,
          "not_comparable": 42
        },
        "pref_decisions": 718,
        "cutoff_candidates": 0,
        "two_stage": 4473,
        "two_stage_entry": 762,
        "two_stage_cutoff": 200
      },
      "checkpoint": "C:\\Users\\35884\\Documents\\Spacecraft\\过程文件\\T2_11_EACCA_20261102\\B\\TRAIN_e_acc_s1\\checkpoints\\step_1000"
    },
    "TRAIN_e_sac_s0": {
      "steps": 1000,
      "updates": 1602,
      "replay": {
        "transitions": 1000,
        "with_candidates": 0,
        "candidates": 0,
        "entry_candidates": 0,
        "reason": {
          "off": 1000
        }
      },
      "checkpoint": "C:\\Users\\35884\\Documents\\Spacecraft\\过程文件\\T2_11_EACCA_20261102\\B\\TRAIN_e_sac_s0\\checkpoints\\step_1000"
    },
    "TRAIN_e_sac_s1": {
      "steps": 1000,
      "updates": 1602,
      "replay": {
        "transitions": 1000,
        "with_candidates": 0,
        "candidates": 0,
        "entry_candidates": 0,
        "reason": {
          "off": 1000
        }
      },
      "checkpoint": "C:\\Users\\35884\\Documents\\Spacecraft\\过程文件\\T2_11_EACCA_20261102\\B\\TRAIN_e_sac_s1\\checkpoints\\step_1000"
    }
  }
}
```

机制止损读取精确step1000保存时的只读replay计数，避免快速评估期间更新到1500后的heartbeat冒充1000证据。快速评估用24开局，主评估48开局，二者分开。

## 两臂各检查点与运行墙钟

```json
{
  "e_acc_s0": {
    "500": {
      "n": 48,
      "mean_return": 31.70361013796636,
      "capture": 42,
      "unsafe": 0,
      "mean_delta_v_success": 1.3578008527380614,
      "mean_time_success": 130.73095238095237,
      "events": {
        "success": 42,
        "timeout": 6
      },
      "urgent": {
        "n": 25,
        "mean_return": 25.660563992709445,
        "capture": 19,
        "unsafe": 0,
        "mean_delta_v_success": 1.2168666044797272,
        "mean_time_success": 121.99999999999997,
        "events": {
          "success": 19,
          "timeout": 6
        }
      },
      "economy": {
        "n": 23,
        "mean_return": 38.27213855672388,
        "capture": 23,
        "unsafe": 0,
        "mean_delta_v_success": 1.4742247969514681,
        "mean_time_success": 137.94347826086957,
        "events": {
          "success": 23
        }
      }
    },
    "1000": {
      "n": 48,
      "mean_return": 17.30615541764641,
      "capture": 38,
      "unsafe": 0,
      "mean_delta_v_success": 1.2648284663064846,
      "mean_time_success": 127.21052631578948,
      "events": {
        "success": 38,
        "timeout": 10
      },
      "urgent": {
        "n": 25,
        "mean_return": -6.928343223436818,
        "capture": 15,
        "unsafe": 0,
        "mean_delta_v_success": 1.146535515698008,
        "mean_time_success": 117.33999999999999,
        "events": {
          "success": 15,
          "timeout": 10
        }
      },
      "economy": {
        "n": 23,
        "mean_return": 43.64800176664992,
        "capture": 23,
        "unsafe": 0,
        "mean_delta_v_success": 1.341976042790274,
        "mean_time_success": 133.64782608695648,
        "events": {
          "success": 23
        }
      }
    },
    "1500": {
      "n": 48,
      "mean_return": 29.252797049059012,
      "capture": 41,
      "unsafe": 0,
      "mean_delta_v_success": 1.3022582654851653,
      "mean_time_success": 127.83170731707318,
      "events": {
        "success": 41,
        "timeout": 7
      },
      "urgent": {
        "n": 25,
        "mean_return": 18.398996409305717,
        "capture": 18,
        "unsafe": 0,
        "mean_delta_v_success": 1.1702595882192517,
        "mean_time_success": 116.9388888888889,
        "events": {
          "success": 18,
          "timeout": 7
        }
      },
      "economy": {
        "n": 23,
        "mean_return": 41.0504064400952,
        "capture": 23,
        "unsafe": 0,
        "mean_delta_v_success": 1.4055615781280548,
        "mean_time_success": 136.35652173913041,
        "events": {
          "success": 23
        }
      }
    },
    "2000": {
      "n": 48,
      "mean_return": 28.473869494904196,
      "capture": 41,
      "unsafe": 0,
      "mean_delta_v_success": 1.3131040765894113,
      "mean_time_success": 126.28536585365856,
      "events": {
        "timeout": 7,
        "success": 41
      },
      "urgent": {
        "n": 25,
        "mean_return": 17.710816594503576,
        "capture": 18,
        "unsafe": 0,
        "mean_delta_v_success": 1.165388897915898,
        "mean_time_success": 116.5,
        "events": {
          "timeout": 7,
          "success": 18
        }
      },
      "economy": {
        "n": 23,
        "mean_return": 40.17284003881791,
        "capture": 23,
        "unsafe": 0,
        "mean_delta_v_success": 1.428707259899117,
        "mean_time_success": 133.94347826086957,
        "events": {
          "success": 23
        }
      }
    }
  },
  "e_acc_s1": {
    "500": {
      "n": 48,
      "mean_return": 25.68532464473525,
      "capture": 41,
      "unsafe": 0,
      "mean_delta_v_success": 1.407748259503126,
      "mean_time_success": 134.2878048780488,
      "events": {
        "success": 41,
        "timeout": 7
      },
      "urgent": {
        "n": 25,
        "mean_return": 18.148264976922956,
        "capture": 18,
        "unsafe": 0,
        "mean_delta_v_success": 1.1885668112629786,
        "mean_time_success": 117.34444444444443,
        "events": {
          "success": 18,
          "timeout": 7
        }
      },
      "economy": {
        "n": 23,
        "mean_return": 33.87778080540079,
        "capture": 23,
        "unsafe": 0,
        "mean_delta_v_success": 1.5792815668215023,
        "mean_time_success": 147.54782608695652,
        "events": {
          "success": 23
        }
      }
    },
    "1000": {
      "n": 48,
      "mean_return": 19.254261817021646,
      "capture": 39,
      "unsafe": 0,
      "mean_delta_v_success": 1.3386577817647756,
      "mean_time_success": 130.3923076923077,
      "events": {
        "success": 39,
        "resource_exhausted": 1,
        "timeout": 8
      },
      "urgent": {
        "n": 25,
        "mean_return": 9.87194079943368,
        "capture": 17,
        "unsafe": 0,
        "mean_delta_v_success": 1.2504810465384368,
        "mean_time_success": 125.87647058823529,
        "events": {
          "success": 17,
          "timeout": 8
        }
      },
      "economy": {
        "n": 23,
        "mean_return": 29.45243683613901,
        "capture": 22,
        "unsafe": 0,
        "mean_delta_v_success": 1.4067943498942184,
        "mean_time_success": 133.88181818181815,
        "events": {
          "success": 22,
          "resource_exhausted": 1
        }
      }
    },
    "1500": {
      "n": 48,
      "mean_return": 36.937623827056925,
      "capture": 43,
      "unsafe": 0,
      "mean_delta_v_success": 1.3428676237539303,
      "mean_time_success": 129.63488372093025,
      "events": {
        "success": 43,
        "timeout": 5
      },
      "urgent": {
        "n": 25,
        "mean_return": 33.68113539372962,
        "capture": 20,
        "unsafe": 0,
        "mean_delta_v_success": 1.2536398396403146,
        "mean_time_success": 123.21,
        "events": {
          "success": 20,
          "timeout": 5
        }
      },
      "economy": {
        "n": 23,
        "mean_return": 40.47728516763008,
        "capture": 23,
        "unsafe": 0,
        "mean_delta_v_success": 1.4204570012440312,
        "mean_time_success": 135.2217391304348,
        "events": {
          "success": 23
        }
      }
    },
    "2000": {
      "n": 48,
      "mean_return": 38.13139649775041,
      "capture": 43,
      "unsafe": 0,
      "mean_delta_v_success": 1.3114451930783262,
      "mean_time_success": 133.02325581395348,
      "events": {
        "success": 43,
        "timeout": 5
      },
      "urgent": {
        "n": 25,
        "mean_return": 34.12473815713254,
        "capture": 20,
        "unsafe": 0,
        "mean_delta_v_success": 1.247440387573034,
        "mean_time_success": 123.30499999999999,
        "events": {
          "success": 20,
          "timeout": 5
        }
      },
      "economy": {
        "n": 23,
        "mean_return": 42.48645991146548,
        "capture": 23,
        "unsafe": 0,
        "mean_delta_v_success": 1.3671015456916238,
        "mean_time_success": 141.47391304347823,
        "events": {
          "success": 23
        }
      }
    }
  },
  "e_sac_s0": {
    "500": {
      "n": 48,
      "mean_return": -11.47954778812919,
      "capture": 33,
      "unsafe": 0,
      "mean_delta_v_success": 1.560751547769186,
      "mean_time_success": 147.26363636363632,
      "events": {
        "timeout": 15,
        "success": 33
      },
      "urgent": {
        "n": 25,
        "mean_return": -51.942181840144706,
        "capture": 10,
        "unsafe": 0,
        "mean_delta_v_success": 1.4430057380718537,
        "mean_time_success": 138.59999999999997,
        "events": {
          "timeout": 15,
          "success": 10
        }
      },
      "economy": {
        "n": 23,
        "mean_return": 32.50157618145289,
        "capture": 23,
        "unsafe": 0,
        "mean_delta_v_success": 1.6119453780723736,
        "mean_time_success": 151.0304347826087,
        "events": {
          "success": 23
        }
      }
    },
    "1000": {
      "n": 48,
      "mean_return": -24.9998221704215,
      "capture": 29,
      "unsafe": 0,
      "mean_delta_v_success": 1.4897982974977384,
      "mean_time_success": 146.08620689655172,
      "events": {
        "timeout": 19,
        "success": 29
      },
      "urgent": {
        "n": 25,
        "mean_return": -79.90406438557787,
        "capture": 6,
        "unsafe": 0,
        "mean_delta_v_success": 1.2312923773191302,
        "mean_time_success": 124.96666666666665,
        "events": {
          "timeout": 19,
          "success": 6
        }
      },
      "economy": {
        "n": 23,
        "mean_return": 34.67870197648758,
        "capture": 23,
        "unsafe": 0,
        "mean_delta_v_success": 1.5572346245008541,
        "mean_time_success": 151.595652173913,
        "events": {
          "success": 23
        }
      }
    },
    "1500": {
      "n": 48,
      "mean_return": -13.572635592141927,
      "capture": 32,
      "unsafe": 0,
      "mean_delta_v_success": 1.5363088373619402,
      "mean_time_success": 147.59375,
      "events": {
        "timeout": 16,
        "success": 32
      },
      "urgent": {
        "n": 25,
        "mean_return": -56.9216217511972,
        "capture": 9,
        "unsafe": 0,
        "mean_delta_v_success": 1.4118648532788909,
        "mean_time_success": 134.54444444444442,
        "events": {
          "timeout": 16,
          "success": 9
        }
      },
      "economy": {
        "n": 23,
        "mean_return": 33.5458276242225,
        "capture": 23,
        "unsafe": 0,
        "mean_delta_v_success": 1.5850043093944384,
        "mean_time_success": 152.70000000000002,
        "events": {
          "success": 23
        }
      }
    },
    "2000": {
      "n": 48,
      "mean_return": -4.588178893568767,
      "capture": 34,
      "unsafe": 0,
      "mean_delta_v_success": 1.500622279583602,
      "mean_time_success": 143.14117647058825,
      "events": {
        "timeout": 14,
        "success": 34
      },
      "urgent": {
        "n": 25,
        "mean_return": -40.34969229260632,
        "capture": 11,
        "unsafe": 0,
        "mean_delta_v_success": 1.360268228767174,
        "mean_time_success": 128.06363636363633,
        "events": {
          "timeout": 14,
          "success": 11
        }
      },
      "economy": {
        "n": 23,
        "mean_return": 34.28303132277642,
        "capture": 23,
        "unsafe": 0,
        "mean_delta_v_success": 1.567748129974068,
        "mean_time_success": 150.3521739130435,
        "events": {
          "success": 23
        }
      }
    }
  },
  "e_sac_s1": {
    "500": {
      "n": 48,
      "mean_return": -22.4637142839082,
      "capture": 30,
      "unsafe": 0,
      "mean_delta_v_success": 1.5013363405280855,
      "mean_time_success": 144.45666666666665,
      "events": {
        "timeout": 18,
        "success": 30
      },
      "urgent": {
        "n": 25,
        "mean_return": -72.28972053880562,
        "capture": 7,
        "unsafe": 0,
        "mean_delta_v_success": 1.077208344558036,
        "mean_time_success": 111.74285714285715,
        "events": {
          "timeout": 18,
          "success": 7
        }
      },
      "economy": {
        "n": 23,
        "mean_return": 31.69498816706726,
        "capture": 23,
        "unsafe": 0,
        "mean_delta_v_success": 1.6304187740841882,
        "mean_time_success": 154.41304347826082,
        "events": {
          "success": 23
        }
      }
    },
    "1000": {
      "n": 48,
      "mean_return": -15.405692178999857,
      "capture": 31,
      "unsafe": 0,
      "mean_delta_v_success": 1.4463140736034077,
      "mean_time_success": 142.38387096774193,
      "events": {
        "timeout": 17,
        "success": 31
      },
      "urgent": {
        "n": 25,
        "mean_return": -60.8914045463458,
        "capture": 8,
        "unsafe": 0,
        "mean_delta_v_success": 1.0807854385464926,
        "mean_time_success": 116.675,
        "events": {
          "timeout": 17,
          "success": 8
        }
      },
      "economy": {
        "n": 23,
        "mean_return": 34.03529952463703,
        "capture": 23,
        "unsafe": 0,
        "mean_delta_v_success": 1.573454468405813,
        "mean_time_success": 151.32608695652172,
        "events": {
          "success": 23
        }
      }
    },
    "1500": {
      "n": 48,
      "mean_return": -13.622909430875794,
      "capture": 31,
      "unsafe": 0,
      "mean_delta_v_success": 1.3486639305868764,
      "mean_time_success": 138.07096774193548,
      "events": {
        "timeout": 17,
        "success": 31
      },
      "urgent": {
        "n": 25,
        "mean_return": -62.917806430951096,
        "capture": 8,
        "unsafe": 0,
        "mean_delta_v_success": 1.1186461938108312,
        "mean_time_success": 118.9125,
        "events": {
          "timeout": 17,
          "success": 8
        }
      },
      "economy": {
        "n": 23,
        "mean_return": 39.95850035181474,
        "capture": 23,
        "unsafe": 0,
        "mean_delta_v_success": 1.4286700999002835,
        "mean_time_success": 144.73478260869567,
        "events": {
          "success": 23
        }
      }
    },
    "2000": {
      "n": 48,
      "mean_return": -15.955089482457785,
      "capture": 31,
      "unsafe": 0,
      "mean_delta_v_success": 1.4206572161657145,
      "mean_time_success": 142.93870967741933,
      "events": {
        "timeout": 17,
        "success": 31
      },
      "urgent": {
        "n": 25,
        "mean_return": -63.51822754080577,
        "capture": 8,
        "unsafe": 0,
        "mean_delta_v_success": 1.106513566898927,
        "mean_time_success": 114.15,
        "events": {
          "timeout": 17,
          "success": 8
        }
      },
      "economy": {
        "n": 23,
        "mean_return": 35.7439736244422,
        "capture": 23,
        "unsafe": 0,
        "mean_delta_v_success": 1.5299245724324226,
        "mean_time_success": 152.95217391304345,
        "events": {
          "success": 23
        }
      }
    }
  }
}
```

- TRAIN_e_acc_s0: steps=2000, finished=True, wall_s=26579.959578037262, seconds/decision=13.28997978901863; calibration={'n': 1741, 'accuracy': 0.945, 'trusted': True, 'w': [0.4552628436839191, -0.19294023726238854, 0.1344147526749624, 1.3330980611812495, 1.3654513195919054, -0.03809440380086747, -0.12839303603155197]}.
- TRAIN_e_acc_s1: steps=2000, finished=True, wall_s=26734.947432756424, seconds/decision=13.367473716378212; calibration={'n': 1715, 'accuracy': 0.93, 'trusted': True, 'w': [0.23831543830823954, 0.020232595334474274, 0.5015755484706658, 1.2189180468135632, 1.197410550453684, -0.04661715639182315, -0.38308569379765345]}.
- TRAIN_e_sac_s0: steps=2000, finished=True, wall_s=21019.519411563873, seconds/decision=10.509759705781937; calibration=None.
- TRAIN_e_sac_s1: steps=2000, finished=True, wall_s=19498.541127443314, seconds/decision=9.749270563721657; calibration=None.

## 时间线、缺项及解释边界

```text
start 11/02/2026 23:36:09
quick check at 4.59836755294444 h: arm_diff=55.13223472748138 labels_off=0
training end 7.43263728744444 h
eval step_2000 done 8.13224292777778 h
eval step_1000 done 8.81361103063889 h
eval step_1500 done 9.50685273513889 h
eval step_500 done 10.2582097140278 h
all done 10.2627761507222 h; decision checkpoints 1000,2000

```

缺项：[]。原生C/Q所有ERROR文件随原始证据保留，不剔除失败开局。独立watchdog停止回执位于本轮记录/STOP_*.json；中途停止仍按既定已有检查点交付，未完成处如实报告。未为通过门槛改变科学配置。

图B信息：原生calibration准确率/可信状态、model_stats内acc_rows和完整KEYSCORE_T211随证据提供；未产生的读数列作缺项，不捏造连续actor动作的真值遗憾。两种子按开发验证解释，不作正式泛化或因果证明。接口代价参考为T2.8 engineered SAC臂AUC25.784897473486076（固定原始科学证据3d215b6），与本轮reduced SAC按原生AUC对照，仅报告事实。

科学证据 `5e42c47032d4e6811451a6794f6254fe9fde2063` 的 `evidence/T2_11_EACCA/` 包含原生读出判定、逐例C/Q、四条训练5份原生元数据、模型SHA和日志；逐Git blob已核验原始字节SHA，无换行过滤。本机完整过程 `C:\Users\35884\Documents\Spacecraft\过程文件\T2_11_EACCA_20261102` 保存30秒进度、TensorBoard事件及模型。
