# T2.8 夜间执行事实交付

冻结科学 `24a342f85001fa2aff4d9db3b52de04558992ff2`；执行书 collab `d5cfab1`。两臂各种子0/1、每条从零2000步、2个异步worker、无引导；开发开局294000–294047，紧急期限140–200秒。按臂比较，2个种子仅作探索性证据。原脚本已按用户随后明确授权移除8.5/10.75/12小时自动截止，UTF-8 BOM保留；其余科学命令配置不变。外部只读日志和调度驱动未修改科学工作树。

## 原生 t28 判定与各项取值

```json
{
  "checkpoints": [
    1000,
    2000
  ],
  "runs": {
    "mve": [
      "e_mve_s0",
      "e_mve_s1"
    ],
    "sac": [
      "e_sac_s0",
      "e_sac_s1"
    ]
  },
  "criteria": {
    "C_complete": true,
    "L_learning": false,
    "R_reliability": false,
    "S_safety": true
  },
  "values": {
    "auc": {
      "mve": {
        "e_mve_s0": 34.161506794401205,
        "e_mve_s1": 7.353935104938072
      },
      "sac": {
        "e_sac_s0": 33.69228568671646,
        "e_sac_s1": 17.877509260255692
      }
    },
    "arm_mean": {
      "mve": 20.75772094966964,
      "sac": 25.784897473486076
    },
    "arm_diff": -5.027176523816436,
    "worst": {
      "mve": 7.353935104938072,
      "sac": 17.877509260255692
    },
    "eacc_above_sac_mean": 1,
    "unsafe": {
      "mve": 0,
      "sac": 0
    }
  },
  "reported": {
    "mve": {
      "500": {
        "mean_return": [
          41.129568286357504,
          -21.554233213375298
        ],
        "urgent_capture": [
          21,
          8
        ],
        "economy_capture": [
          23,
          23
        ],
        "unsafe": [
          0,
          1
        ],
        "delta_v_success": [
          1.3299252194032594,
          1.5788851816626037
        ],
        "time_success": [
          127.9568181818182,
          150.23870967741937
        ]
      },
      "1000": {
        "mean_return": [
          29.748055770581445,
          3.1212985802753983
        ],
        "urgent_capture": [
          18,
          12
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
          1.2880839071295929,
          1.3545540265455494
        ],
        "time_success": [
          125.18048780487806,
          128.54857142857145
        ]
      },
      "1500": {
        "mean_return": [
          38.546395608204335,
          14.96689067678804
        ],
        "urgent_capture": [
          20,
          15
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
          1.274277236246651,
          1.3673649392299116
        ],
        "time_success": [
          126.13023255813957,
          135.1263157894737
        ]
      },
      "2000": {
        "mean_return": [
          38.574957818220966,
          11.586571629600746
        ],
        "urgent_capture": [
          20,
          15
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
          1.2757653301745246,
          1.4875640265346504
        ],
        "time_success": [
          126.70232558139534,
          144.06052631578947
        ]
      }
    },
    "sac": {
      "500": {
        "mean_return": [
          23.678975657899226,
          -8.219408439363773
        ],
        "urgent_capture": [
          18,
          12
        ],
        "economy_capture": [
          22,
          21
        ],
        "unsafe": [
          0,
          0
        ],
        "delta_v_success": [
          1.2964166035687403,
          1.3618491623007534
        ],
        "time_success": [
          126.83250000000001,
          134.7030303030303
        ]
      },
      "1000": {
        "mean_return": [
          28.231773876546487,
          9.07279756554892
        ],
        "urgent_capture": [
          19,
          13
        ],
        "economy_capture": [
          22,
          23
        ],
        "unsafe": [
          0,
          0
        ],
        "delta_v_success": [
          1.3035349655853496,
          1.3015107151374508
        ],
        "time_success": [
          127.49756097560977,
          129.6472222222222
        ]
      },
      "1500": {
        "mean_return": [
          41.538039683137576,
          10.060232057688632
        ],
        "urgent_capture": [
          21,
          13
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
          1.318608448364902,
          1.2602362459476075
        ],
        "time_success": [
          128.7840909090909,
          131.33055555555558
        ]
      },
      "2000": {
        "mean_return": [
          39.15279749688643,
          26.682220954962464
        ],
        "urgent_capture": [
          20,
          17
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
          1.2643198264979052,
          1.2562166605066696
        ],
        "time_success": [
          130.39069767441862,
          132.5075
        ]
      }
    },
    "mechanism": {
      "TRAIN_e_mve_s0": {
        "two_stage_share": 0.7713923277305352,
        "decisive_share": 0.1806865695082111,
        "c_mean": 0.14756830735706988,
        "grad_cos": -0.10259564335103191,
        "loss_share": 0.5004057317954553,
        "increment_abs_mean": 1.734211316815129,
        "w_mean": 0.9440065254399805,
        "w2_mean": 0.9672557520277706,
        "pi_gap_sigma_median": 0.02793502032664823,
        "stage2_rows": 1136.0987654320988,
        "stage2_entry_share": 0.23434635168976253,
        "stage1_cutoff_share": 0.0,
        "stage2_cutoff_share": 0.012302461827610746,
        "guided_share": 0.0
      },
      "TRAIN_e_mve_s1": {
        "two_stage_share": 0.807577005101337,
        "decisive_share": 0.16693666880881344,
        "c_mean": 0.13968846119112438,
        "grad_cos": 0.3609987166488454,
        "loss_share": 0.4679333741315865,
        "increment_abs_mean": 1.3310511458067247,
        "w_mean": 0.9450250378361454,
        "w2_mean": 0.9389718682677658,
        "pi_gap_sigma_median": 0.03697254223588072,
        "stage2_rows": 1166.320987654321,
        "stage2_entry_share": 0.20653121246967787,
        "stage1_cutoff_share": 0.0,
        "stage2_cutoff_share": 0.03500105218133993,
        "guided_share": 0.0
      }
    }
  },
  "verdict": "harmful"
}
```

## 评估覆盖与各检查点读数

```json
{
  "result_json_files": 768,
  "errors": [],
  "arms": {
    "e_mve_s0": {
      "500": {
        "n": 48,
        "mean_return": 41.129568286357504,
        "capture": 44,
        "unsafe": 0,
        "mean_delta_v_success": 1.3299252194032594,
        "mean_time_success": 127.9568181818182,
        "events": {
          "success": 44,
          "timeout": 4
        },
        "urgent": {
          "n": 25,
          "mean_return": 42.70700461958752,
          "capture": 21,
          "unsafe": 0,
          "mean_delta_v_success": 1.2014911290538195,
          "mean_time_success": 120.3857142857143,
          "events": {
            "success": 21,
            "timeout": 4
          }
        },
        "economy": {
          "n": 23,
          "mean_return": 39.41496357632488,
          "capture": 23,
          "unsafe": 0,
          "mean_delta_v_success": 1.447191127983183,
          "mean_time_success": 134.8695652173913,
          "events": {
            "success": 23
          }
        }
      },
      "1000": {
        "n": 48,
        "mean_return": 29.748055770581445,
        "capture": 41,
        "unsafe": 0,
        "mean_delta_v_success": 1.2880839071295929,
        "mean_time_success": 125.18048780487806,
        "events": {
          "success": 41,
          "timeout": 7
        },
        "urgent": {
          "n": 25,
          "mean_return": 19.600818763514248,
          "capture": 18,
          "unsafe": 0,
          "mean_delta_v_success": 1.127480299434146,
          "mean_time_success": 114.46111111111112,
          "events": {
            "success": 18,
            "timeout": 7
          }
        },
        "economy": {
          "n": 23,
          "mean_return": 40.777661213045775,
          "capture": 23,
          "unsafe": 0,
          "mean_delta_v_success": 1.4137736870651596,
          "mean_time_success": 133.5695652173913,
          "events": {
            "success": 23
          }
        }
      },
      "1500": {
        "n": 48,
        "mean_return": 38.546395608204335,
        "capture": 43,
        "unsafe": 0,
        "mean_delta_v_success": 1.274277236246651,
        "mean_time_success": 126.13023255813957,
        "events": {
          "success": 43,
          "timeout": 5
        },
        "urgent": {
          "n": 25,
          "mean_return": 35.71486976096236,
          "capture": 20,
          "unsafe": 0,
          "mean_delta_v_success": 1.1388876143924875,
          "mean_time_success": 116.18499999999999,
          "events": {
            "success": 20,
            "timeout": 5
          }
        },
        "economy": {
          "n": 23,
          "mean_return": 41.62414109433694,
          "capture": 23,
          "unsafe": 0,
          "mean_delta_v_success": 1.3920073422067933,
          "mean_time_success": 134.7782608695652,
          "events": {
            "success": 23
          }
        }
      },
      "2000": {
        "n": 48,
        "mean_return": 38.574957818220966,
        "capture": 43,
        "unsafe": 0,
        "mean_delta_v_success": 1.2757653301745246,
        "mean_time_success": 126.70232558139534,
        "events": {
          "success": 43,
          "timeout": 5
        },
        "urgent": {
          "n": 25,
          "mean_return": 35.286129547916374,
          "capture": 20,
          "unsafe": 0,
          "mean_delta_v_success": 1.1576863805960997,
          "mean_time_success": 116.43999999999998,
          "events": {
            "success": 20,
            "timeout": 5
          }
        },
        "economy": {
          "n": 23,
          "mean_return": 42.14977115550855,
          "capture": 23,
          "unsafe": 0,
          "mean_delta_v_success": 1.3784426776340246,
          "mean_time_success": 135.62608695652173,
          "events": {
            "success": 23
          }
        }
      }
    },
    "e_mve_s1": {
      "500": {
        "n": 48,
        "mean_return": -21.554233213375298,
        "capture": 31,
        "unsafe": 1,
        "mean_delta_v_success": 1.5788851816626037,
        "mean_time_success": 150.23870967741937,
        "events": {
          "timeout": 16,
          "success": 31,
          "unsafe": 1
        },
        "urgent": {
          "n": 25,
          "mean_return": -69.52603836079135,
          "capture": 8,
          "unsafe": 1,
          "mean_delta_v_success": 1.3531230938731178,
          "mean_time_success": 134.4625,
          "events": {
            "timeout": 16,
            "success": 8,
            "unsafe": 1
          }
        },
        "economy": {
          "n": 23,
          "mean_return": 30.589033251207358,
          "capture": 23,
          "unsafe": 0,
          "mean_delta_v_success": 1.6574111252415549,
          "mean_time_success": 155.72608695652173,
          "events": {
            "success": 23
          }
        }
      },
      "1000": {
        "n": 48,
        "mean_return": 3.1212985802753983,
        "capture": 35,
        "unsafe": 0,
        "mean_delta_v_success": 1.3545540265455494,
        "mean_time_success": 128.54857142857145,
        "events": {
          "success": 35,
          "timeout": 13
        },
        "urgent": {
          "n": 25,
          "mean_return": -32.10821124329721,
          "capture": 12,
          "unsafe": 0,
          "mean_delta_v_success": 1.2697067710404548,
          "mean_time_success": 122.61666666666666,
          "events": {
            "success": 12,
            "timeout": 13
          }
        },
        "economy": {
          "n": 23,
          "mean_return": 41.414244040680416,
          "capture": 23,
          "unsafe": 0,
          "mean_delta_v_success": 1.3988221598525548,
          "mean_time_success": 131.64347826086959,
          "events": {
            "success": 23
          }
        }
      },
      "1500": {
        "n": 48,
        "mean_return": 14.96689067678804,
        "capture": 38,
        "unsafe": 0,
        "mean_delta_v_success": 1.3673649392299116,
        "mean_time_success": 135.1263157894737,
        "events": {
          "success": 38,
          "timeout": 10
        },
        "urgent": {
          "n": 25,
          "mean_return": -6.53674614823276,
          "capture": 15,
          "unsafe": 0,
          "mean_delta_v_success": 1.2094435230351837,
          "mean_time_success": 124.18,
          "events": {
            "success": 15,
            "timeout": 10
          }
        },
        "economy": {
          "n": 23,
          "mean_return": 38.34040896485413,
          "capture": 23,
          "unsafe": 0,
          "mean_delta_v_success": 1.4703571671829947,
          "mean_time_success": 142.26521739130433,
          "events": {
            "success": 23
          }
        }
      },
      "2000": {
        "n": 48,
        "mean_return": 11.586571629600746,
        "capture": 38,
        "unsafe": 0,
        "mean_delta_v_success": 1.4875640265346504,
        "mean_time_success": 144.06052631578947,
        "events": {
          "success": 38,
          "timeout": 10
        },
        "urgent": {
          "n": 25,
          "mean_return": -7.227688342860596,
          "capture": 15,
          "unsafe": 0,
          "mean_delta_v_success": 1.2816549452083659,
          "mean_time_success": 128.12666666666664,
          "events": {
            "success": 15,
            "timeout": 10
          }
        },
        "economy": {
          "n": 23,
          "mean_return": 32.036854208363074,
          "capture": 23,
          "unsafe": 0,
          "mean_delta_v_success": 1.6218525578344014,
          "mean_time_success": 154.4521739130435,
          "events": {
            "success": 23
          }
        }
      }
    },
    "e_sac_s0": {
      "500": {
        "n": 48,
        "mean_return": 23.678975657899226,
        "capture": 40,
        "unsafe": 0,
        "mean_delta_v_success": 1.2964166035687403,
        "mean_time_success": 126.83250000000001,
        "events": {
          "timeout": 7,
          "success": 40,
          "resource_exhausted": 1
        },
        "urgent": {
          "n": 25,
          "mean_return": 18.587943689118376,
          "capture": 18,
          "unsafe": 0,
          "mean_delta_v_success": 1.1537549109111165,
          "mean_time_success": 115.47222222222223,
          "events": {
            "timeout": 7,
            "success": 18
          }
        },
        "economy": {
          "n": 23,
          "mean_return": 29.212706058747987,
          "capture": 22,
          "unsafe": 0,
          "mean_delta_v_success": 1.413139806652251,
          "mean_time_success": 136.1272727272727,
          "events": {
            "success": 22,
            "resource_exhausted": 1
          }
        }
      },
      "1000": {
        "n": 48,
        "mean_return": 28.231773876546487,
        "capture": 41,
        "unsafe": 0,
        "mean_delta_v_success": 1.3035349655853496,
        "mean_time_success": 127.49756097560977,
        "events": {
          "success": 41,
          "timeout": 7
        },
        "urgent": {
          "n": 25,
          "mean_return": 27.81881780516949,
          "capture": 19,
          "unsafe": 0,
          "mean_delta_v_success": 1.1373646611889872,
          "mean_time_success": 115.26315789473684,
          "events": {
            "success": 19,
            "timeout": 6
          }
        },
        "economy": {
          "n": 23,
          "mean_return": 28.680639171521484,
          "capture": 22,
          "unsafe": 0,
          "mean_delta_v_success": 1.4470456830185716,
          "mean_time_success": 138.06363636363633,
          "events": {
            "success": 22,
            "timeout": 1
          }
        }
      },
      "1500": {
        "n": 48,
        "mean_return": 41.538039683137576,
        "capture": 44,
        "unsafe": 0,
        "mean_delta_v_success": 1.318608448364902,
        "mean_time_success": 128.7840909090909,
        "events": {
          "success": 44,
          "timeout": 4
        },
        "urgent": {
          "n": 25,
          "mean_return": 43.08552168832301,
          "capture": 21,
          "unsafe": 0,
          "mean_delta_v_success": 1.1918318234580436,
          "mean_time_success": 118.16666666666669,
          "events": {
            "success": 21,
            "timeout": 4
          }
        },
        "economy": {
          "n": 23,
          "mean_return": 39.85599402532732,
          "capture": 23,
          "unsafe": 0,
          "mean_delta_v_success": 1.4343610189320342,
          "mean_time_success": 138.47826086956522,
          "events": {
            "success": 23
          }
        }
      },
      "2000": {
        "n": 48,
        "mean_return": 39.15279749688643,
        "capture": 43,
        "unsafe": 0,
        "mean_delta_v_success": 1.2643198264979052,
        "mean_time_success": 130.39069767441862,
        "events": {
          "success": 43,
          "timeout": 5
        },
        "urgent": {
          "n": 25,
          "mean_return": 34.762369728449855,
          "capture": 20,
          "unsafe": 0,
          "mean_delta_v_success": 1.186891422769623,
          "mean_time_success": 118.81999999999998,
          "events": {
            "success": 20,
            "timeout": 5
          }
        },
        "economy": {
          "n": 23,
          "mean_return": 43.92500159301313,
          "capture": 23,
          "unsafe": 0,
          "mean_delta_v_success": 1.33164887321815,
          "mean_time_success": 140.4521739130435,
          "events": {
            "success": 23
          }
        }
      }
    },
    "e_sac_s1": {
      "500": {
        "n": 48,
        "mean_return": -8.219408439363773,
        "capture": 33,
        "unsafe": 0,
        "mean_delta_v_success": 1.3618491623007534,
        "mean_time_success": 134.7030303030303,
        "events": {
          "success": 33,
          "timeout": 15
        },
        "urgent": {
          "n": 25,
          "mean_return": -28.607194039122376,
          "capture": 12,
          "unsafe": 0,
          "mean_delta_v_success": 1.0686543461792068,
          "mean_time_success": 116.09166666666664,
          "events": {
            "success": 12,
            "timeout": 13
          }
        },
        "economy": {
          "n": 23,
          "mean_return": 13.941228082112968,
          "capture": 21,
          "unsafe": 0,
          "mean_delta_v_success": 1.529389057227351,
          "mean_time_success": 145.33809523809526,
          "events": {
            "success": 21,
            "timeout": 2
          }
        }
      },
      "1000": {
        "n": 48,
        "mean_return": 9.07279756554892,
        "capture": 36,
        "unsafe": 0,
        "mean_delta_v_success": 1.3015107151374508,
        "mean_time_success": 129.6472222222222,
        "events": {
          "success": 36,
          "timeout": 12
        },
        "urgent": {
          "n": 25,
          "mean_return": -20.495545828266906,
          "capture": 13,
          "unsafe": 0,
          "mean_delta_v_success": 1.1260353050979797,
          "mean_time_success": 114.86923076923077,
          "events": {
            "success": 13,
            "timeout": 12
          }
        },
        "economy": {
          "n": 23,
          "mean_return": 41.21230125447917,
          "capture": 23,
          "unsafe": 0,
          "mean_delta_v_success": 1.4006924686380209,
          "mean_time_success": 137.99999999999997,
          "events": {
            "success": 23
          }
        }
      },
      "1500": {
        "n": 48,
        "mean_return": 10.060232057688632,
        "capture": 36,
        "unsafe": 0,
        "mean_delta_v_success": 1.2602362459476075,
        "mean_time_success": 131.33055555555558,
        "events": {
          "success": 36,
          "timeout": 12
        },
        "urgent": {
          "n": 25,
          "mean_return": -19.860413207215235,
          "capture": 13,
          "unsafe": 0,
          "mean_delta_v_success": 1.073941659834597,
          "mean_time_success": 116.34615384615384,
          "events": {
            "success": 13,
            "timeout": 12
          }
        },
        "economy": {
          "n": 23,
          "mean_return": 42.58267256301891,
          "capture": 23,
          "unsafe": 0,
          "mean_delta_v_success": 1.3655331859245268,
          "mean_time_success": 139.8,
          "events": {
            "success": 23
          }
        }
      },
      "2000": {
        "n": 48,
        "mean_return": 26.682220954962464,
        "capture": 40,
        "unsafe": 0,
        "mean_delta_v_success": 1.2562166605066696,
        "mean_time_success": 132.5075,
        "events": {
          "success": 40,
          "timeout": 8
        },
        "urgent": {
          "n": 25,
          "mean_return": 11.794279841927143,
          "capture": 17,
          "unsafe": 0,
          "mean_delta_v_success": 1.1191827450010174,
          "mean_time_success": 119.99411764705881,
          "events": {
            "success": 17,
            "timeout": 8
          }
        },
        "economy": {
          "n": 23,
          "mean_return": 42.86476564304435,
          "capture": 23,
          "unsafe": 0,
          "mean_delta_v_success": 1.3575025980543258,
          "mean_time_success": 141.75652173913045,
          "events": {
            "success": 23
          }
        }
      }
    }
  }
}
```

## 实际训练及每决策墙钟

- TRAIN_e_mve_s0: steps=2000, finished=True, wall_s=26843.988698244095, seconds/decision=13.421994349122047.
- TRAIN_e_mve_s0.err.log: steps=0, finished=False, wall_s=None, seconds/decision=None.
- TRAIN_e_mve_s0.out.log: steps=0, finished=False, wall_s=None, seconds/decision=None.
- TRAIN_e_mve_s1: steps=2000, finished=True, wall_s=23746.444969654083, seconds/decision=11.873222484827041.
- TRAIN_e_mve_s1.err.log: steps=0, finished=False, wall_s=None, seconds/decision=None.
- TRAIN_e_mve_s1.out.log: steps=0, finished=False, wall_s=None, seconds/decision=None.
- TRAIN_e_sac_s0: steps=2000, finished=True, wall_s=25926.183851718903, seconds/decision=12.963091925859452.
- TRAIN_e_sac_s0.err.log: steps=0, finished=False, wall_s=None, seconds/decision=None.
- TRAIN_e_sac_s0.out.log: steps=0, finished=False, wall_s=None, seconds/decision=None.
- TRAIN_e_sac_s1: steps=2000, finished=True, wall_s=21585.80560541153, seconds/decision=10.792902802705765.
- TRAIN_e_sac_s1.err.log: steps=0, finished=False, wall_s=None, seconds/decision=None.
- TRAIN_e_sac_s1.out.log: steps=0, finished=False, wall_s=None, seconds/decision=None.

## night.log 时间线

```text
start=11/01/2026 23:05:16 W=2 EW=10
P3 done 0.00350665325 h, checkpoints 32
training end 7.50465438869444 h
eval step_2000 done 7.68561390402778 h
eval step_1000 done 7.86789381238889 h
eval step_1500 done 8.05376475 h
eval step_500 done 8.2430760695 h
all done 8.24330716119444 h; decision checkpoints 1000,2000

2026-11-02T07:20:10.669168+09:00 eval label repair: preserve genuine first-arm results, remaining registered evaluations only; no training restart
2026-11-02T07:51:14.967253+09:00 eval step_2000 complete after label repair
2026-11-02T08:21:49.829782+09:00 eval step_1000 complete after label repair
2026-11-02T08:51:54.920828+09:00 eval step_1500 complete after label repair
2026-11-02T09:25:01.012937+09:00 eval step_500 complete after label repair

```

## 异常、缺项及证据

{}

启动行政故障：原执行书PowerShell循环变量 $r 覆盖路径 $R（变量名不区分大小写），原night立即报路径错误，P3与正式训练均未启动；接续行政控制器随后因原night退出而未找到PID。已保全attempt1_path_and_handoff，仅改循环变量为$row和驱动退出检查，以单一recovery_controller接续；原60测试与日志验收保留，没有重启正式训练或热改观察器。

P3原始分数与汇总随附；旧模型在修复后关键状态上离线打分，不能作为当前执行器训练结果。原生READOUT/DECISION、逐开局C、四条训练原生元数据及SHA256SUMS供独立核验；模型ZIP、TensorBoard原始事件和30秒进度保留本机 `C:\Users\35884\Documents\Spacecraft\过程文件\T2_8_NIGHT_20261101`，只上传执行书明确要求的证据。用户随后取消自动硬截止，目标为2026-11-02 09:00 JST前交付；未按时钟截停。实际未完成样本如实保留。没有追加实验或改判据。

评估驱动补充：标签里的`${r}`未随循环变量改名，仍展开为根路径，导致首臂结果写到绝对路径且后续臂命中同一首臂缓存。没有将这些缓存充作其他臂结果；先让在途原生评估自然完成，再按每份JSON内checkpoint身份核验，仅复用真实e_mve_s0结果，并只改policy标签元数据（所有科学字段逐项精确相等）。错误标签原始JSON、原SHA/修正SHA映射在EVAL_LABEL_REPAIR/随交付。其余臂只补算原本未真正执行的既定case。为赶09:00交付目标，剩余评估worker由10调至12（执行书明确允许的资源调整），训练配置未变，无科学进程强停。复用原生case数：192。
