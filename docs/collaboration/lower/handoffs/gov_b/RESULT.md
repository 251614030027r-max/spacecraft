# RUN_GOV_B ECC 最终轮开发选型

执行书 618196da，A 步冻结 bc2d02509cd89000fba13a00bfeea1b11e908543，B 步冻结 e0a617532008dac02e2d50f7e859aca613e03e5e，A/B 科学证据合并提交 d738839e172ba9e33dfc1fbe52127a2fa7e735eb。
A 步 b_coordination = cert，B 步在环配置为 ECC_CERT_LOOP。SAC_C、EACH_C、ECC_CERT_LOOP 各种子 10、11，每条从零训练 3000 决策，2 个异步 worker，六条并行共享 CPU 0–11。
训练紧急与经济比例 3:1。评估只用 3000 步模型，10 个 worker。紧急 294800–294895，经济 294900–294947，自然 295000–295047，共 1920 例。
pytest 114 passed（退出码 0），三组 40 决策冒烟、评估及 manifest 校验通过。后台训练日志的快照锁定验收通过，锁定期间逐回合与 TensorBoard 增长，释放后快照恢复。六条训练的 TensorBoard 回合数、末条步数、逐回报及模型 SHA256 均与原生记录一致。
墙钟记录为 all done 21.8188909844167 h。训练异常记录 166 条，unsafe 回合 1 例。
原生 final_method = in_loop，final_config = ECC_CERT_LOOP。正式实验等待单独授权。
原生 CHECK 中六个 EACH_C 调节器组缺失符合执行书。已逐例核验其余 30 组，原生 B 判定 complete。

A 步逐例及清单见 evidence/GOV_A/，清单 SHA256 4cd6c93f1d3cd104f48bac65f5a01ede45bf18e39aaf8049e2c66a84d0df3edb。
B 步逐例及清单见 evidence/GOV_B/，清单 SHA256 a4f84c0bbcbbbb5f9027ad8b8a39a3a196dbf2975caa79f69c9afdde051d3a28。

## 原生判定与各档数值

```json
{
 "coordination": "cert",
 "evaluations": {
  "urgent": {
   "SAC_C_s10": 96,
   "SAC_C_s11": 96,
   "EACH_C_s10": 96,
   "EACH_C_s11": 96,
   "DEPLOY_s10": 96,
   "DEPLOY_s11": 96,
   "LOOP_s10": 96,
   "LOOP_s11": 96,
   "LOOP_MONITOR_s10": 96,
   "LOOP_MONITOR_s11": 96,
   "ECC_DEPLOY_OLD_s10": 96,
   "ECC_DEPLOY_OLD_s11": 96
  },
  "economy": {
   "SAC_C_s10": 48,
   "SAC_C_s11": 48,
   "EACH_C_s10": 48,
   "EACH_C_s11": 48,
   "DEPLOY_s10": 48,
   "DEPLOY_s11": 48,
   "LOOP_s10": 48,
   "LOOP_s11": 48,
   "LOOP_MONITOR_s10": 48,
   "LOOP_MONITOR_s11": 48,
   "ECC_DEPLOY_OLD_s10": 48,
   "ECC_DEPLOY_OLD_s11": 48
  },
  "natural": {
   "SAC_C_s10": 48,
   "SAC_C_s11": 48,
   "EACH_C_s10": 48,
   "EACH_C_s11": 48,
   "DEPLOY_s10": 48,
   "DEPLOY_s11": 48,
   "LOOP_s10": 48,
   "LOOP_s11": 48,
   "LOOP_MONITOR_s10": 48,
   "LOOP_MONITOR_s11": 48,
   "ECC_DEPLOY_OLD_s10": 48,
   "ECC_DEPLOY_OLD_s11": 48
  }
 },
 "complete": true,
 "rung1_in_loop": {
  "urgent": {
   "pairs": 192,
   "mean": 35.95574700067454,
   "median": 0.2356583220288755,
   "without_top2_per_seed": 32.18898521960214,
   "per_seed": {
    "10": {
     "n": 96,
     "mean": 43.70386422500436,
     "median": 0.29378775859223794,
     "without_top2": 40.02247801312789
    },
    "11": {
     "n": 96,
     "mean": 28.207629776344728,
     "median": 0.17520945522596065,
     "without_top2": 24.35549242607639
    }
   },
   "capture": [
    145,
    111
   ],
   "unsafe_first": 0,
   "flips": {
    "only_first_captures": 37,
    "only_second_captures": 3
   },
   "both_captured": {
    "n": 108,
    "time_s": [
     126.94351851851853,
     127.34537037037036
    ],
    "delta_v": [
     1.2382488715808297,
     1.2604042638802995
    ]
   },
   "gates": {
    "G1": true,
    "G2": true,
    "G3": true,
    "G4": true,
    "G5": true
   }
  },
  "natural": {
   "pairs": 96,
   "mean": 29.39315509120333,
   "median": 0.08645363076203694,
   "without_top2_per_seed": 21.45433565793763,
   "per_seed": {
    "10": {
     "n": 48,
     "mean": 30.67377235866127,
     "median": 0.23307662410489627,
     "without_top2": 22.62922617462957
    },
    "11": {
     "n": 48,
     "mean": 28.112537823745384,
     "median": -0.2892903682571557,
     "without_top2": 20.279445141245695
    }
   },
   "capture": [
    88,
    74
   ],
   "unsafe_first": 0,
   "flips": {
    "only_first_captures": 14,
    "only_second_captures": 0
   },
   "both_captured": {
    "n": 74,
    "time_s": [
     143.60135135135135,
     150.21621621621622
    ],
    "delta_v": [
     1.4549641980124814,
     1.4479983395552551
    ]
   },
   "not_inferior": true
  },
  "economy_report": {
   "pairs": 96,
   "mean": 3.4080336566284157,
   "median": -0.2214548156002536,
   "without_top2_per_seed": 1.3715438868913268,
   "per_seed": {
    "10": {
     "n": 48,
     "mean": 4.567617193007574,
     "median": 0.4263997791751457,
     "without_top2": 2.3941946655204647
    },
    "11": {
     "n": 48,
     "mean": 2.248450120249259,
     "median": -1.652580440674857,
     "without_top2": 0.34889310826218767
    }
   },
   "capture": [
    96,
    96
   ],
   "unsafe_first": 0,
   "flips": {
    "only_first_captures": 0,
    "only_second_captures": 0
   },
   "both_captured": {
    "n": 96,
    "time_s": [
     135.2125,
     151.89895833333333
    ],
    "delta_v": [
     1.3041899041211205,
     1.3810475163701639
    ]
   }
  },
  "pass": true
 },
 "rung2_deploy": {
  "urgent": {
   "pairs": 192,
   "mean": 32.7116203769474,
   "median": -0.11435944563888256,
   "without_top2_per_seed": 28.86436005415516,
   "per_seed": {
    "10": {
     "n": 96,
     "mean": 39.22605687277856,
     "median": -0.11435944563888256,
     "without_top2": 35.45671205889228
    },
    "11": {
     "n": 96,
     "mean": 26.197183881116246,
     "median": -0.08533798984836238,
     "without_top2": 22.27200804941803
    }
   },
   "capture": [
    142,
    111
   ],
   "unsafe_first": 0,
   "flips": {
    "only_first_captures": 35,
    "only_second_captures": 4
   },
   "both_captured": {
    "n": 107,
    "time_s": [
     127.08317757009345,
     127.2233644859813
    ],
    "delta_v": [
     1.2864932215577765,
     1.257781189436982
    ]
   },
   "gates": {
    "G1": true,
    "G2": true,
    "G3": true,
    "G4": true,
    "G5": true
   }
  },
  "natural": {
   "pairs": 96,
   "mean": 26.657558176239103,
   "median": -0.6212814523790051,
   "without_top2_per_seed": 18.585681253642495,
   "per_seed": {
    "10": {
     "n": 48,
     "mean": 24.99411171146707,
     "median": -0.5173242775460096,
     "without_top2": 16.691346590273632
    },
    "11": {
     "n": 48,
     "mean": 28.321004641011132,
     "median": -0.8372669599978124,
     "without_top2": 20.480015917011357
    }
   },
   "capture": [
    87,
    74
   ],
   "unsafe_first": 0,
   "flips": {
    "only_first_captures": 13,
    "only_second_captures": 0
   },
   "both_captured": {
    "n": 74,
    "time_s": [
     143.61486486486484,
     150.21621621621622
    ],
    "delta_v": [
     1.4803038441190004,
     1.4479983395552551
    ]
   },
   "not_inferior": true
  },
  "economy_report": {
   "pairs": 96,
   "mean": 2.5572495524481575,
   "median": -0.8462492121637588,
   "without_top2_per_seed": 0.5508177416618879,
   "per_seed": {
    "10": {
     "n": 48,
     "mean": -0.25680850650001424,
     "median": -1.066505907421469,
     "without_top2": -2.4984953682488515
    },
    "11": {
     "n": 48,
     "mean": 5.37130761139633,
     "median": -0.3371691276264741,
     "without_top2": 3.6001308515726267
    }
   },
   "capture": [
    96,
    96
   ],
   "unsafe_first": 0,
   "flips": {
    "only_first_captures": 0,
    "only_second_captures": 0
   },
   "both_captured": {
    "n": 96,
    "time_s": [
     135.55833333333334,
     151.89895833333333
    ],
    "delta_v": [
     1.3252865900589599,
     1.3810475163701639
    ]
   }
  },
  "pass": true
 },
 "final_method": "in_loop",
 "final_config": "ECC_CERT_LOOP",
 "dose": {
  "<160": {
   "EACH_C": {
    "pairs": 62,
    "mean": 13.081715573994154,
    "median": -0.5091612218497801,
    "without_top2_per_seed": -0.3755125877593045,
    "per_seed": {
     "10": {
      "n": 31,
      "mean": 21.032108598701846,
      "median": -0.10517221219089379,
      "without_top2": 7.972206708830106
     },
     "11": {
      "n": 31,
      "mean": 5.1313225492864625,
      "median": -0.8350048749555015,
      "without_top2": -8.723231884348717
     }
    },
    "capture": [
     28,
     24
    ],
    "unsafe_first": 0,
    "flips": {
     "only_first_captures": 8,
     "only_second_captures": 4
    },
    "both_captured": {
     "n": 20,
     "time_s": [
      116.06499999999998,
      114.13
     ],
     "delta_v": [
      1.1675597998000018,
      1.1466849772789571
     ]
    }
   },
   "DEPLOY": {
    "pairs": 62,
    "mean": 19.42181635358136,
    "median": -0.7313694653563161,
    "without_top2_per_seed": 6.401836521454607,
    "per_seed": {
     "10": {
      "n": 31,
      "mean": 27.67174735152892,
      "median": -0.10517221219089379,
      "without_top2": 15.069751582541805
     },
     "11": {
      "n": 31,
      "mean": 11.171885355633803,
      "median": -1.0545194256960428,
      "without_top2": -2.2660785396325895
     }
    },
    "capture": [
     30,
     24
    ],
    "unsafe_first": 0,
    "flips": {
     "only_first_captures": 9,
     "only_second_captures": 3
    },
    "both_captured": {
     "n": 21,
     "time_s": [
      116.97142857142858,
      115.0952380952381
     ],
     "delta_v": [
      1.1835072148699306,
      1.1516378439190047
     ]
    }
   },
   "LOOP": {
    "pairs": 62,
    "mean": 22.484271648243432,
    "median": -0.21817883114102443,
    "without_top2_per_seed": 9.827636502302997,
    "per_seed": {
     "10": {
      "n": 31,
      "mean": 34.13202440894153,
      "median": -0.0010235055170682017,
      "without_top2": 22.038609089143932
     },
     "11": {
      "n": 31,
      "mean": 10.836518887545342,
      "median": -1.0234056745267708,
      "without_top2": -2.38333608453793
     }
    },
    "capture": [
     31,
     24
    ],
    "unsafe_first": 0,
    "flips": {
     "only_first_captures": 9,
     "only_second_captures": 2
    },
    "both_captured": {
     "n": 22,
     "time_s": [
      116.4,
      116.24545454545455
     ],
     "delta_v": [
      1.1421163470990907,
      1.1693394523461085
     ]
    }
   },
   "ECC_DEPLOY_OLD": {
    "pairs": 62,
    "mean": 19.42181635358136,
    "median": -0.7313694653563161,
    "without_top2_per_seed": 6.401836521454607,
    "per_seed": {
     "10": {
      "n": 31,
      "mean": 27.67174735152892,
      "median": -0.10517221219089379,
      "without_top2": 15.069751582541805
     },
     "11": {
      "n": 31,
      "mean": 11.171885355633803,
      "median": -1.0545194256960428,
      "without_top2": -2.2660785396325895
     }
    },
    "capture": [
     30,
     24
    ],
    "unsafe_first": 0,
    "flips": {
     "only_first_captures": 9,
     "only_second_captures": 3
    },
    "both_captured": {
     "n": 21,
     "time_s": [
      116.97142857142858,
      115.0952380952381
     ],
     "delta_v": [
      1.1835072148699306,
      1.1516378439190047
     ]
    }
   },
   "summary": {
    "SAC_C": {
     "n": 62,
     "capture": 24,
     "unsafe": 1,
     "timeout": 37,
     "urgent_n": 62,
     "urgent_capture": 24,
     "mean_return": -48.98552493711592,
     "median_return": -123.63616384552292,
     "entries_lost": 4
    },
    "EACH_C": {
     "n": 62,
     "capture": 28,
     "unsafe": 0,
     "timeout": 34,
     "urgent_n": 62,
     "urgent_capture": 28,
     "mean_return": -35.90380936312175,
     "median_return": -124.17151453687717,
     "entries_lost": 4
    },
    "DEPLOY": {
     "n": 62,
     "capture": 30,
     "unsafe": 0,
     "timeout": 32,
     "urgent_n": 62,
     "urgent_capture": 30,
     "mean_return": -29.563708583534545,
     "median_return": -123.75815394029723,
     "entries_lost": 4
    },
    "LOOP": {
     "n": 62,
     "capture": 31,
     "unsafe": 0,
     "timeout": 31,
     "urgent_n": 62,
     "urgent_capture": 31,
     "mean_return": -26.501253288872466,
     "median_return": -27.250622388031573,
     "entries_lost": 5
    }
   },
   "same_model_deploy_vs_monitor": {
    "pairs": 62,
    "mean": 6.340100779587207,
    "median": 0.0,
    "without_top2_per_seed": -0.3538744745214306,
    "per_seed": {
     "10": {
      "n": 31,
      "mean": 6.639638752827072,
      "median": 0.0,
      "without_top2": -0.07263592221045495
     },
     "11": {
      "n": 31,
      "mean": 6.040562806347341,
      "median": 0.0,
      "without_top2": -0.6351130268324062
     }
    },
    "capture": [
     30,
     28
    ],
    "unsafe_first": 0,
    "flips": {
     "only_first_captures": 2,
     "only_second_captures": 0
    },
    "both_captured": {
     "n": 28,
     "time_s": [
      115.7892857142857,
      115.61785714285713
     ],
     "delta_v": [
      1.1824504258514188,
      1.1778544060352396
     ]
    }
   },
   "loop_vs_loop_monitor": {
    "pairs": 62,
    "mean": 2.658440758240914,
    "median": 0.0,
    "without_top2_per_seed": -0.6987485580660358,
    "per_seed": {
     "10": {
      "n": 31,
      "mean": -1.284464973410772,
      "median": 0.0,
      "without_top2": -1.3730487646804805
     },
     "11": {
      "n": 31,
      "mean": 6.601346489892601,
      "median": 0.0,
      "without_top2": -0.024448351451590904
     }
    },
    "capture": [
     31,
     30
    ],
    "unsafe_first": 0,
    "flips": {
     "only_first_captures": 1,
     "only_second_captures": 0
    },
    "both_captured": {
     "n": 30,
     "time_s": [
      116.89000000000001,
      116.86333333333333
     ],
     "delta_v": [
      1.1621827573652503,
      1.1588328503231822
     ]
    }
   }
  },
  "160-180": {
   "EACH_C": {
    "pairs": 70,
    "mean": 54.7027658414645,
    "median": 0.1947018538812202,
    "without_top2_per_seed": 45.12822037836355,
    "per_seed": {
     "10": {
      "n": 35,
      "mean": 58.19883277779296,
      "median": 0.18859784186615514,
      "without_top2": 48.71481755254844
     },
     "11": {
      "n": 35,
      "mean": 51.20669890513603,
      "median": 0.28161032482897497,
      "without_top2": 41.54162320417866
     }
    },
    "capture": [
     57,
     38
    ],
    "unsafe_first": 0,
    "flips": {
     "only_first_captures": 19,
     "only_second_captures": 0
    },
    "both_captured": {
     "n": 38,
     "time_s": [
      125.46842105263163,
      124.5184210526316
     ],
     "delta_v": [
      1.2673545100059764,
      1.2153857485676038
     ]
    }
   },
   "DEPLOY": {
    "pairs": 70,
    "mean": 51.74525433836938,
    "median": 0.06339002801447435,
    "without_top2_per_seed": 41.99146575386874,
    "per_seed": {
     "10": {
      "n": 35,
      "mean": 58.47263489327371,
      "median": 0.18859784186615514,
      "without_top2": 49.00521373563409
     },
     "11": {
      "n": 35,
      "mean": 45.01787378346505,
      "median": 0.040943050473913445,
      "without_top2": 34.977717772103375
     }
    },
    "capture": [
     56,
     38
    ],
    "unsafe_first": 0,
    "flips": {
     "only_first_captures": 18,
     "only_second_captures": 0
    },
    "both_captured": {
     "n": 38,
     "time_s": [
      124.08947368421056,
      124.5184210526316
     ],
     "delta_v": [
      1.2544506258809667,
      1.2153857485676038
     ]
    }
   },
   "LOOP": {
    "pairs": 70,
    "mean": 54.86246881394257,
    "median": 0.5543848796519484,
    "without_top2_per_seed": 45.334895117481324,
    "per_seed": {
     "10": {
      "n": 35,
      "mean": 59.03978136567358,
      "median": 0.3573302700400234,
      "without_top2": 49.59511451679626
     },
     "11": {
      "n": 35,
      "mean": 50.685156262211535,
      "median": 0.6707152385627211,
      "without_top2": 41.07467571816637
     }
    },
    "capture": [
     57,
     38
    ],
    "unsafe_first": 0,
    "flips": {
     "only_first_captures": 19,
     "only_second_captures": 0
    },
    "both_captured": {
     "n": 38,
     "time_s": [
      124.3815789473684,
      124.5184210526316
     ],
     "delta_v": [
      1.1931957860190907,
      1.2153857485676038
     ]
    }
   },
   "ECC_DEPLOY_OLD": {
    "pairs": 70,
    "mean": 51.74525433836938,
    "median": 0.06339002801447435,
    "without_top2_per_seed": 41.99146575386874,
    "per_seed": {
     "10": {
      "n": 35,
      "mean": 58.47263489327371,
      "median": 0.18859784186615514,
      "without_top2": 49.00521373563409
     },
     "11": {
      "n": 35,
      "mean": 45.01787378346505,
      "median": 0.040943050473913445,
      "without_top2": 34.977717772103375
     }
    },
    "capture": [
     56,
     38
    ],
    "unsafe_first": 0,
    "flips": {
     "only_first_captures": 18,
     "only_second_captures": 0
    },
    "both_captured": {
     "n": 38,
     "time_s": [
      124.08947368421056,
      124.5184210526316
     ],
     "delta_v": [
      1.2544506258809667,
      1.2153857485676038
     ]
    }
   },
   "summary": {
    "SAC_C": {
     "n": 70,
     "capture": 38,
     "unsafe": 0,
     "timeout": 32,
     "urgent_n": 70,
     "urgent_capture": 38,
     "mean_return": -18.47538316840729,
     "median_return": 68.9948822045603,
     "entries_lost": 8
    },
    "EACH_C": {
     "n": 70,
     "capture": 57,
     "unsafe": 0,
     "timeout": 13,
     "urgent_n": 70,
     "urgent_capture": 57,
     "mean_return": 36.227382673057214,
     "median_return": 73.3632798218194,
     "entries_lost": 9
    },
    "DEPLOY": {
     "n": 70,
     "capture": 56,
     "unsafe": 0,
     "timeout": 14,
     "urgent_n": 70,
     "urgent_capture": 56,
     "mean_return": 33.26987116996209,
     "median_return": 73.56983999617066,
     "entries_lost": 10
    },
    "LOOP": {
     "n": 70,
     "capture": 57,
     "unsafe": 0,
     "timeout": 13,
     "urgent_n": 70,
     "urgent_capture": 57,
     "mean_return": 36.38708564553527,
     "median_return": 73.99815715174267,
     "entries_lost": 13
    }
   },
   "same_model_deploy_vs_monitor": {
    "pairs": 70,
    "mean": -2.9575115030951142,
    "median": 0.0,
    "without_top2_per_seed": -3.3329268205232507,
    "per_seed": {
     "10": {
      "n": 35,
      "mean": 0.2738021154807545,
      "median": 0.0,
      "without_top2": -0.028329935041238593
     },
     "11": {
      "n": 35,
      "mean": -6.188825121670983,
      "median": 0.0,
      "without_top2": -6.637523706005263
     }
    },
    "capture": [
     56,
     57
    ],
    "unsafe_first": 0,
    "flips": {
     "only_first_captures": 0,
     "only_second_captures": 1
    },
    "both_captured": {
     "n": 56,
     "time_s": [
      126.1625,
      127.4375
     ],
     "delta_v": [
      1.267495987217491,
      1.275268319107448
     ]
    }
   },
   "loop_vs_loop_monitor": {
    "pairs": 70,
    "mean": 22.999681499524684,
    "median": 0.0,
    "without_top2_per_seed": 11.97291263961758,
    "per_seed": {
     "10": {
      "n": 35,
      "mean": 16.81773158304714,
      "median": 0.0,
      "without_top2": 5.781116663552561
     },
     "11": {
      "n": 35,
      "mean": 29.181631416002237,
      "median": 0.0,
      "without_top2": 18.164708615682596
     }
    },
    "capture": [
     57,
     49
    ],
    "unsafe_first": 0,
    "flips": {
     "only_first_captures": 8,
     "only_second_captures": 0
    },
    "both_captured": {
     "n": 49,
     "time_s": [
      124.71428571428571,
      124.7734693877551
     ],
     "delta_v": [
      1.2169184378011295,
      1.2218206870174566
     ]
    }
   }
  },
  ">=180": {
   "EACH_C": {
    "pairs": 60,
    "mean": 24.08537137182805,
    "median": -0.015305366924145858,
    "without_top2_per_seed": 10.99592062766587,
    "per_seed": {
     "10": {
      "n": 30,
      "mean": 28.59866404870237,
      "median": -0.11565745077022882,
      "without_top2": 15.36262885716731
     },
     "11": {
      "n": 30,
      "mean": 19.572078694953728,
      "median": 0.2890407124194141,
      "without_top2": 6.629212398164427
     }
    },
    "capture": [
     56,
     49
    ],
    "unsafe_first": 0,
    "flips": {
     "only_first_captures": 8,
     "only_second_captures": 1
    },
    "both_captured": {
     "n": 48,
     "time_s": [
      134.70416666666665,
      134.67083333333332
     ],
     "delta_v": [
      1.3636270483559698,
      1.337781960456021
     ]
    }
   },
   "DEPLOY": {
    "pairs": 60,
    "mean": 24.238511579433332,
    "median": -0.020932348888294428,
    "without_top2_per_seed": 11.072335089656276,
    "per_seed": {
     "10": {
      "n": 30,
      "mean": 28.71116902082549,
      "median": -0.1776789270672623,
      "without_top2": 15.396421560880741
     },
     "11": {
      "n": 30,
      "mean": 19.765854138041174,
      "median": 0.20807319988818307,
      "without_top2": 6.748248618431808
     }
    },
    "capture": [
     56,
     49
    ],
    "unsafe_first": 0,
    "flips": {
     "only_first_captures": 8,
     "only_second_captures": 1
    },
    "both_captured": {
     "n": 48,
     "time_s": [
      133.87708333333333,
      134.67083333333332
     ],
     "delta_v": [
      1.3569166543945161,
      1.337781960456021
     ]
    }
   },
   "LOOP": {
    "pairs": 60,
    "mean": 27.81842941604066,
    "median": 0.5652997297913345,
    "without_top2_per_seed": 14.872941448233073,
    "per_seed": {
     "10": {
      "n": 30,
      "mean": 35.70286203748853,
      "median": 0.5286056367106156,
      "without_top2": 22.866782636341537
     },
     "11": {
      "n": 30,
      "mean": 19.933996794592815,
      "median": 0.5652997297913345,
      "without_top2": 6.879100260124601
     }
    },
    "capture": [
     57,
     49
    ],
    "unsafe_first": 0,
    "flips": {
     "only_first_captures": 9,
     "only_second_captures": 1
    },
    "both_captured": {
     "n": 48,
     "time_s": [
      133.80416666666667,
      134.67083333333332
     ],
     "delta_v": [
      1.3179766380380034,
      1.337781960456021
     ]
    }
   },
   "ECC_DEPLOY_OLD": {
    "pairs": 60,
    "mean": 24.238511579433332,
    "median": -0.020932348888294428,
    "without_top2_per_seed": 11.072335089656276,
    "per_seed": {
     "10": {
      "n": 30,
      "mean": 28.71116902082549,
      "median": -0.1776789270672623,
      "without_top2": 15.396421560880741
     },
     "11": {
      "n": 30,
      "mean": 19.765854138041174,
      "median": 0.20807319988818307,
      "without_top2": 6.748248618431808
     }
    },
    "capture": [
     56,
     49
    ],
    "unsafe_first": 0,
    "flips": {
     "only_first_captures": 8,
     "only_second_captures": 1
    },
    "both_captured": {
     "n": 48,
     "time_s": [
      133.87708333333333,
      134.67083333333332
     ],
     "delta_v": [
      1.3569166543945161,
      1.337781960456021
     ]
    }
   },
   "summary": {
    "SAC_C": {
     "n": 60,
     "capture": 49,
     "unsafe": 0,
     "timeout": 11,
     "urgent_n": 60,
     "urgent_capture": 49,
     "mean_return": 35.319217811205974,
     "median_return": 71.9877885144647,
     "entries_lost": 2
    },
    "EACH_C": {
     "n": 60,
     "capture": 56,
     "unsafe": 0,
     "timeout": 4,
     "urgent_n": 60,
     "urgent_capture": 56,
     "mean_return": 59.40458918303402,
     "median_return": 73.3482092116923,
     "entries_lost": 0
    },
    "DEPLOY": {
     "n": 60,
     "capture": 56,
     "unsafe": 0,
     "timeout": 4,
     "urgent_n": 60,
     "urgent_capture": 56,
     "mean_return": 59.55772939063931,
     "median_return": 73.59177835440747,
     "entries_lost": 0
    },
    "LOOP": {
     "n": 60,
     "capture": 57,
     "unsafe": 0,
     "timeout": 3,
     "urgent_n": 60,
     "urgent_capture": 57,
     "mean_return": 63.13764722724664,
     "median_return": 73.56806186363599,
     "entries_lost": 2
    }
   },
   "same_model_deploy_vs_monitor": {
    "pairs": 60,
    "mean": 0.15314020760528196,
    "median": 0.0,
    "without_top2_per_seed": -0.09324934225356848,
    "per_seed": {
     "10": {
      "n": 30,
      "mean": 0.1125049721231169,
      "median": 0.0,
      "without_top2": -0.02963536624911559
     },
     "11": {
      "n": 30,
      "mean": 0.19377544308744704,
      "median": 0.0,
      "without_top2": -0.15686331825802138
     }
    },
    "capture": [
     56,
     56
    ],
    "unsafe_first": 0,
    "flips": {
     "only_first_captures": 0,
     "only_second_captures": 0
    },
    "both_captured": {
     "n": 56,
     "time_s": [
      132.78928571428574,
      134.0375
     ],
     "delta_v": [
      1.3408420375169876,
      1.3492186310090364
     ]
    }
   },
   "loop_vs_loop_monitor": {
    "pairs": 60,
    "mean": 13.96161107015646,
    "median": 0.0,
    "without_top2_per_seed": 7.167116974033921,
    "per_seed": {
     "10": {
      "n": 30,
      "mean": 0.1542478442591772,
      "median": 0.0,
      "without_top2": -0.060068528460284654
     },
     "11": {
      "n": 30,
      "mean": 27.768974296053738,
      "median": 0.0,
      "without_top2": 14.394302476528127
     }
    },
    "capture": [
     57,
     53
    ],
    "unsafe_first": 0,
    "flips": {
     "only_first_captures": 4,
     "only_second_captures": 0
    },
    "both_captured": {
     "n": 53,
     "time_s": [
      134.20754716981133,
      134.79811320754715
     ],
     "delta_v": [
      1.3110110691055747,
      1.3128933892572472
     ]
    }
   }
  },
  "economy": {
   "EACH_C": {
    "pairs": 96,
    "mean": 2.57846840687537,
    "median": -0.8462492121637588,
    "without_top2_per_seed": 0.5729591549772401,
    "per_seed": {
     "10": {
      "n": 48,
      "mean": -0.25680850650001424,
      "median": -1.066505907421469,
      "without_top2": -2.4984953682488515
     },
     "11": {
      "n": 48,
      "mean": 5.413745320250754,
      "median": -0.3371691276264741,
      "without_top2": 3.644413678203331
     }
    },
    "capture": [
     96,
     96
    ],
    "unsafe_first": 0,
    "flips": {
     "only_first_captures": 0,
     "only_second_captures": 0
    },
    "both_captured": {
     "n": 96,
     "time_s": [
      135.60520833333334,
      151.89895833333333
     ],
     "delta_v": [
      1.3247326811982796,
      1.3810475163701639
     ]
    }
   },
   "DEPLOY": {
    "pairs": 96,
    "mean": 2.5572495524481575,
    "median": -0.8462492121637588,
    "without_top2_per_seed": 0.5508177416618879,
    "per_seed": {
     "10": {
      "n": 48,
      "mean": -0.25680850650001424,
      "median": -1.066505907421469,
      "without_top2": -2.4984953682488515
     },
     "11": {
      "n": 48,
      "mean": 5.37130761139633,
      "median": -0.3371691276264741,
      "without_top2": 3.6001308515726267
     }
    },
    "capture": [
     96,
     96
    ],
    "unsafe_first": 0,
    "flips": {
     "only_first_captures": 0,
     "only_second_captures": 0
    },
    "both_captured": {
     "n": 96,
     "time_s": [
      135.55833333333334,
      151.89895833333333
     ],
     "delta_v": [
      1.3252865900589599,
      1.3810475163701639
     ]
    }
   },
   "LOOP": {
    "pairs": 96,
    "mean": 3.4080336566284157,
    "median": -0.2214548156002536,
    "without_top2_per_seed": 1.3715438868913268,
    "per_seed": {
     "10": {
      "n": 48,
      "mean": 4.567617193007574,
      "median": 0.4263997791751457,
      "without_top2": 2.3941946655204647
     },
     "11": {
      "n": 48,
      "mean": 2.248450120249259,
      "median": -1.652580440674857,
      "without_top2": 0.34889310826218767
     }
    },
    "capture": [
     96,
     96
    ],
    "unsafe_first": 0,
    "flips": {
     "only_first_captures": 0,
     "only_second_captures": 0
    },
    "both_captured": {
     "n": 96,
     "time_s": [
      135.2125,
      151.89895833333333
     ],
     "delta_v": [
      1.3041899041211205,
      1.3810475163701639
     ]
    }
   },
   "ECC_DEPLOY_OLD": {
    "pairs": 96,
    "mean": 2.5572495524481575,
    "median": -0.8462492121637588,
    "without_top2_per_seed": 0.5508177416618879,
    "per_seed": {
     "10": {
      "n": 48,
      "mean": -0.25680850650001424,
      "median": -1.066505907421469,
      "without_top2": -2.4984953682488515
     },
     "11": {
      "n": 48,
      "mean": 5.37130761139633,
      "median": -0.3371691276264741,
      "without_top2": 3.6001308515726267
     }
    },
    "capture": [
     96,
     96
    ],
    "unsafe_first": 0,
    "flips": {
     "only_first_captures": 0,
     "only_second_captures": 0
    },
    "both_captured": {
     "n": 96,
     "time_s": [
      135.55833333333334,
      151.89895833333333
     ],
     "delta_v": [
      1.3252865900589599,
      1.3810475163701639
     ]
    }
   },
   "summary": {
    "SAC_C": {
     "n": 96,
     "capture": 96,
     "unsafe": 0,
     "timeout": 0,
     "urgent_n": 0,
     "urgent_capture": 0,
     "mean_return": 41.72012017852678,
     "median_return": 45.53667914377864,
     "entries_lost": 0
    },
    "EACH_C": {
     "n": 96,
     "capture": 96,
     "unsafe": 0,
     "timeout": 0,
     "urgent_n": 0,
     "urgent_capture": 0,
     "mean_return": 44.29858858540214,
     "median_return": 45.6326560401821,
     "entries_lost": 1
    },
    "DEPLOY": {
     "n": 96,
     "capture": 96,
     "unsafe": 0,
     "timeout": 0,
     "urgent_n": 0,
     "urgent_capture": 0,
     "mean_return": 44.27736973097493,
     "median_return": 45.6326560401821,
     "entries_lost": 1
    },
    "LOOP": {
     "n": 96,
     "capture": 96,
     "unsafe": 0,
     "timeout": 0,
     "urgent_n": 0,
     "urgent_capture": 0,
     "mean_return": 45.128153835155196,
     "median_return": 47.62635226093769,
     "entries_lost": 0
    }
   },
   "same_model_deploy_vs_monitor": {
    "pairs": 96,
    "mean": -0.021218854427212424,
    "median": 0.0,
    "without_top2_per_seed": -0.022141413315352094,
    "per_seed": {
     "10": {
      "n": 48,
      "mean": 0.0,
      "median": 0.0,
      "without_top2": 0.0
     },
     "11": {
      "n": 48,
      "mean": -0.04243770885442485,
      "median": 0.0,
      "without_top2": -0.04428282663070419
     }
    },
    "capture": [
     96,
     96
    ],
    "unsafe_first": 0,
    "flips": {
     "only_first_captures": 0,
     "only_second_captures": 0
    },
    "both_captured": {
     "n": 96,
     "time_s": [
      135.55833333333334,
      135.60520833333334
     ],
     "delta_v": [
      1.3252865900589599,
      1.3247326811982796
     ]
    }
   },
   "loop_vs_loop_monitor": {
    "pairs": 96,
    "mean": 0.0,
    "median": 0.0,
    "without_top2_per_seed": 0.0,
    "per_seed": {
     "10": {
      "n": 48,
      "mean": 0.0,
      "median": 0.0,
      "without_top2": 0.0
     },
     "11": {
      "n": 48,
      "mean": 0.0,
      "median": 0.0,
      "without_top2": 0.0
     }
    },
    "capture": [
     96,
     96
    ],
    "unsafe_first": 0,
    "flips": {
     "only_first_captures": 0,
     "only_second_captures": 0
    },
    "both_captured": {
     "n": 96,
     "time_s": [
      135.2125,
      135.2125
     ],
     "delta_v": [
      1.3041899041211205,
      1.3041899041211205
     ]
    }
   }
  }
 },
 "mechanism": {
  "DEPLOY_urgent": {
   "pairs": 192,
   "observed_mean_gain": 1.0169211227815926,
   "prediction": 0.0,
   "destroy_rate": 0.3072916666666667,
   "replacements": 0,
   "realization": null,
   "persistence": null,
   "zero_band": 1.0416666666666667,
   "direction_consistent": true,
   "captures": [
    142,
    141
   ],
   "replace_per_episode": 0.0,
   "critical_monitor": {
    "decisions": 926,
    "critical_share": 0.5259179265658748,
    "replace_non_critical": 0,
    "replace_critical_share": 0.23408624229979466
   },
   "critical_gov": {
    "decisions": 0,
    "critical_share": null,
    "replace_non_critical": 0,
    "replace_critical_share": null
   }
  },
  "LOOP_urgent": {
   "pairs": 192,
   "observed_mean_gain": 13.606758834307564,
   "prediction": 0.0,
   "destroy_rate": 0.296875,
   "replacements": 0,
   "realization": null,
   "persistence": null,
   "zero_band": 1.0416666666666667,
   "direction_consistent": false,
   "captures": [
    145,
    132
   ],
   "replace_per_episode": 0.0,
   "critical_monitor": {
    "decisions": 1005,
    "critical_share": 0.4736318407960199,
    "replace_non_critical": 0,
    "replace_critical_share": 0.22268907563025211
   },
   "critical_gov": {
    "decisions": 0,
    "critical_share": null,
    "replace_non_critical": 0,
    "replace_critical_share": null
   }
  },
  "DEPLOY_economy": {
   "pairs": 96,
   "observed_mean_gain": -0.021218854427212424,
   "prediction": 0.0,
   "destroy_rate": 0.020833333333333332,
   "replacements": 0,
   "realization": null,
   "persistence": null,
   "zero_band": 2.0833333333333335,
   "direction_consistent": true,
   "captures": [
    96,
    96
   ],
   "replace_per_episode": 0.0,
   "critical_monitor": {
    "decisions": 336,
    "critical_share": 0.047619047619047616,
    "replace_non_critical": 0,
    "replace_critical_share": 0.375
   },
   "critical_gov": {
    "decisions": 0,
    "critical_share": null,
    "replace_non_critical": 0,
    "replace_critical_share": null
   }
  },
  "LOOP_economy": {
   "pairs": 96,
   "observed_mean_gain": 0.0,
   "prediction": 0.0,
   "destroy_rate": 0.020833333333333332,
   "replacements": 0,
   "realization": null,
   "persistence": null,
   "zero_band": 2.0833333333333335,
   "direction_consistent": true,
   "captures": [
    96,
    96
   ],
   "replace_per_episode": 0.0,
   "critical_monitor": {
    "decisions": 326,
    "critical_share": 0.03680981595092025,
    "replace_non_critical": 0,
    "replace_critical_share": 0.16666666666666666
   },
   "critical_gov": {
    "decisions": 0,
    "critical_share": null,
    "replace_non_critical": 0,
    "replace_critical_share": null
   }
  },
  "DEPLOY_natural": {
   "pairs": 96,
   "observed_mean_gain": 0.3567719241729345,
   "prediction": 0.0,
   "destroy_rate": 0.1875,
   "replacements": 0,
   "realization": null,
   "persistence": null,
   "zero_band": 2.0833333333333335,
   "direction_consistent": true,
   "captures": [
    87,
    87
   ],
   "replace_per_episode": 0.0,
   "critical_monitor": {
    "decisions": 427,
    "critical_share": 0.35362997658079626,
    "replace_non_critical": 0,
    "replace_critical_share": 0.2119205298013245
   },
   "critical_gov": {
    "decisions": 0,
    "critical_share": null,
    "replace_non_critical": 0,
    "replace_critical_share": null
   }
  },
  "LOOP_natural": {
   "pairs": 96,
   "observed_mean_gain": 14.588866213867226,
   "prediction": 0.0,
   "destroy_rate": 0.20833333333333334,
   "replacements": 0,
   "realization": null,
   "persistence": null,
   "zero_band": 2.0833333333333335,
   "direction_consistent": false,
   "captures": [
    88,
    81
   ],
   "replace_per_episode": 0.0,
   "critical_monitor": {
    "decisions": 453,
    "critical_share": 0.3068432671081678,
    "replace_non_critical": 0,
    "replace_critical_share": 0.2517985611510791
   },
   "critical_gov": {
    "decisions": 0,
    "critical_share": null,
    "replace_non_critical": 0,
    "replace_critical_share": null
   }
  }
 },
 "policy_behaviour": {
  "urgent": {
   "EACH_C": {
    "decisions": 926,
    "critical_share": 0.5259179265658748,
    "replace_non_critical": 0,
    "replace_critical_share": 0.23408624229979466,
    "destroy_rate": 0.3072916666666667
   },
   "LOOP_MONITOR": {
    "decisions": 1005,
    "critical_share": 0.4736318407960199,
    "replace_non_critical": 0,
    "replace_critical_share": 0.22268907563025211,
    "destroy_rate": 0.296875
   }
  },
  "economy": {
   "EACH_C": {
    "decisions": 336,
    "critical_share": 0.047619047619047616,
    "replace_non_critical": 0,
    "replace_critical_share": 0.375,
    "destroy_rate": 0.020833333333333332
   },
   "LOOP_MONITOR": {
    "decisions": 326,
    "critical_share": 0.03680981595092025,
    "replace_non_critical": 0,
    "replace_critical_share": 0.16666666666666666,
    "destroy_rate": 0.020833333333333332
   }
  },
  "natural": {
   "EACH_C": {
    "decisions": 427,
    "critical_share": 0.35362997658079626,
    "replace_non_critical": 0,
    "replace_critical_share": 0.2119205298013245,
    "destroy_rate": 0.1875
   },
   "LOOP_MONITOR": {
    "decisions": 453,
    "critical_share": 0.3068432671081678,
    "replace_non_critical": 0,
    "replace_critical_share": 0.2517985611510791,
    "destroy_rate": 0.20833333333333334
   }
  }
 },
 "gov_wall": {
  "DEPLOY": {
   "decisions": 0,
   "mean_s": null,
   "p95_s": null,
   "max_s": null
  },
  "LOOP": {
   "decisions": 0,
   "mean_s": null,
   "p95_s": null,
   "max_s": null
  }
 },
 "curves": {
  "SAC_C_s10": {
   "curve_auc": 4.042696972492543,
   "late_return": 15.970602385893644,
   "late_urgent_capture": 0.6875,
   "episodes": 566,
   "per_500": {
    "0": {
     "episodes": 66,
     "timeouts": 33,
     "unsafe": 0,
     "replacements": 0
    },
    "1": {
     "episodes": 108,
     "timeouts": 21,
     "unsafe": 0,
     "replacements": 0
    },
    "2": {
     "episodes": 97,
     "timeouts": 26,
     "unsafe": 0,
     "replacements": 0
    },
    "3": {
     "episodes": 98,
     "timeouts": 25,
     "unsafe": 0,
     "replacements": 0
    },
    "4": {
     "episodes": 93,
     "timeouts": 33,
     "unsafe": 0,
     "replacements": 0
    },
    "5": {
     "episodes": 104,
     "timeouts": 25,
     "unsafe": 0,
     "replacements": 0
    }
   }
  },
  "SAC_C_s11": {
   "curve_auc": -11.872361060035619,
   "late_return": -3.5832294121152444,
   "late_urgent_capture": 0.5396825396825397,
   "episodes": 497,
   "per_500": {
    "0": {
     "episodes": 58,
     "timeouts": 37,
     "unsafe": 0,
     "replacements": 0
    },
    "1": {
     "episodes": 86,
     "timeouts": 31,
     "unsafe": 0,
     "replacements": 0
    },
    "2": {
     "episodes": 93,
     "timeouts": 25,
     "unsafe": 0,
     "replacements": 0
    },
    "3": {
     "episodes": 76,
     "timeouts": 34,
     "unsafe": 0,
     "replacements": 0
    },
    "4": {
     "episodes": 97,
     "timeouts": 26,
     "unsafe": 0,
     "replacements": 0
    },
    "5": {
     "episodes": 87,
     "timeouts": 29,
     "unsafe": 0,
     "replacements": 0
    }
   }
  },
  "EACH_C_s10": {
   "curve_auc": 10.78403255900528,
   "late_return": 42.53032832615417,
   "late_urgent_capture": 0.8571428571428571,
   "episodes": 586,
   "per_500": {
    "0": {
     "episodes": 68,
     "timeouts": 34,
     "unsafe": 0,
     "replacements": 0
    },
    "1": {
     "episodes": 93,
     "timeouts": 26,
     "unsafe": 0,
     "replacements": 0
    },
    "2": {
     "episodes": 94,
     "timeouts": 27,
     "unsafe": 0,
     "replacements": 0
    },
    "3": {
     "episodes": 103,
     "timeouts": 24,
     "unsafe": 0,
     "replacements": 0
    },
    "4": {
     "episodes": 108,
     "timeouts": 18,
     "unsafe": 0,
     "replacements": 0
    },
    "5": {
     "episodes": 120,
     "timeouts": 13,
     "unsafe": 0,
     "replacements": 0
    }
   }
  },
  "EACH_C_s11": {
   "curve_auc": 12.80063224551416,
   "late_return": 21.16323409853643,
   "late_urgent_capture": 0.7088607594936709,
   "episodes": 592,
   "per_500": {
    "0": {
     "episodes": 64,
     "timeouts": 33,
     "unsafe": 0,
     "replacements": 0
    },
    "1": {
     "episodes": 107,
     "timeouts": 20,
     "unsafe": 0,
     "replacements": 0
    },
    "2": {
     "episodes": 105,
     "timeouts": 20,
     "unsafe": 0,
     "replacements": 0
    },
    "3": {
     "episodes": 111,
     "timeouts": 22,
     "unsafe": 0,
     "replacements": 0
    },
    "4": {
     "episodes": 100,
     "timeouts": 24,
     "unsafe": 0,
     "replacements": 0
    },
    "5": {
     "episodes": 105,
     "timeouts": 24,
     "unsafe": 0,
     "replacements": 0
    }
   }
  },
  "ECC_CERT_LOOP_s10": {
   "curve_auc": 33.05159576474877,
   "late_return": 38.294669780853894,
   "late_urgent_capture": 0.82,
   "episodes": 708,
   "per_500": {
    "0": {
     "episodes": 90,
     "timeouts": 23,
     "unsafe": 0,
     "replacements": 92
    },
    "1": {
     "episodes": 129,
     "timeouts": 15,
     "unsafe": 0,
     "replacements": 32
    },
    "2": {
     "episodes": 112,
     "timeouts": 22,
     "unsafe": 1,
     "replacements": 58
    },
    "3": {
     "episodes": 125,
     "timeouts": 20,
     "unsafe": 0,
     "replacements": 53
    },
    "4": {
     "episodes": 130,
     "timeouts": 14,
     "unsafe": 0,
     "replacements": 51
    },
    "5": {
     "episodes": 122,
     "timeouts": 18,
     "unsafe": 0,
     "replacements": 41
    }
   }
  },
  "ECC_CERT_LOOP_s11": {
   "curve_auc": 18.741547410491474,
   "late_return": 21.027315407152564,
   "late_urgent_capture": 0.7215189873417721,
   "episodes": 603,
   "per_500": {
    "0": {
     "episodes": 85,
     "timeouts": 27,
     "unsafe": 0,
     "replacements": 119
    },
    "1": {
     "episodes": 99,
     "timeouts": 22,
     "unsafe": 0,
     "replacements": 50
    },
    "2": {
     "episodes": 101,
     "timeouts": 21,
     "unsafe": 0,
     "replacements": 57
    },
    "3": {
     "episodes": 110,
     "timeouts": 20,
     "unsafe": 0,
     "replacements": 69
    },
    "4": {
     "episodes": 105,
     "timeouts": 22,
     "unsafe": 0,
     "replacements": 44
    },
    "5": {
     "episodes": 103,
     "timeouts": 22,
     "unsafe": 0,
     "replacements": 63
    }
   }
  }
 }
}
```

## 训练异常与 unsafe 回合

```json
{
  "timeline_anomalies": [
    "stall? SAC_C_s11 heartbeat older than 3 h at 5.38456921833333 h",
    "stall? SAC_C_s11 heartbeat older than 3 h at 5.46792403455556 h",
    "stall? SAC_C_s11 heartbeat older than 3 h at 5.55127848211111 h",
    "stall? SAC_C_s11 heartbeat older than 3 h at 5.63463219525 h",
    "stall? EACH_C_s10 heartbeat older than 3 h at 6.21805862133333 h",
    "stall? EACH_C_s10 heartbeat older than 3 h at 6.30140873227778 h",
    "stall? EACH_C_s10 heartbeat older than 3 h at 6.38475664561111 h",
    "stall? EACH_C_s11 heartbeat older than 3 h at 6.46809996655556 h",
    "stall? EACH_C_s10 heartbeat older than 3 h at 6.46810379516667 h",
    "stall? EACH_C_s11 heartbeat older than 3 h at 6.55144751180555 h",
    "stall? EACH_C_s10 heartbeat older than 3 h at 6.55145405766667 h",
    "stall? EACH_C_s11 heartbeat older than 3 h at 6.63480362475 h",
    "stall? EACH_C_s10 heartbeat older than 3 h at 6.63480745177778 h",
    "stall? SAC_C_s10 heartbeat older than 3 h at 6.71815030586111 h",
    "stall? EACH_C_s11 heartbeat older than 3 h at 6.71815301913889 h",
    "stall? EACH_C_s10 heartbeat older than 3 h at 6.71815740344444 h",
    "stall? SAC_C_s10 heartbeat older than 3 h at 6.80150130855556 h",
    "stall? EACH_C_s11 heartbeat older than 3 h at 6.80150458052778 h",
    "stall? EACH_C_s10 heartbeat older than 3 h at 6.80150966458333 h",
    "stall? SAC_C_s10 heartbeat older than 3 h at 6.88485272172222 h",
    "stall? EACH_C_s11 heartbeat older than 3 h at 6.88485599327778 h",
    "stall? EACH_C_s10 heartbeat older than 3 h at 6.88486101 h",
    "stall? SAC_C_s10 heartbeat older than 3 h at 6.96820549383333 h",
    "stall? EACH_C_s11 heartbeat older than 3 h at 6.96820814202778 h",
    "stall? EACH_C_s10 heartbeat older than 3 h at 6.96821315683333 h",
    "stall? SAC_C_s10 heartbeat older than 3 h at 7.05155408983333 h",
    "stall? EACH_C_s11 heartbeat older than 3 h at 7.05155799127778 h",
    "stall? SAC_C_s10 heartbeat older than 3 h at 7.13490450108333 h",
    "stall? EACH_C_s11 heartbeat older than 3 h at 7.13490721605555 h",
    "stall? ECC_CERT_LOOP_s11 heartbeat older than 3 h at 7.13491028311111 h",
    "stall? SAC_C_s10 heartbeat older than 3 h at 7.21825486844444 h",
    "stall? EACH_C_s11 heartbeat older than 3 h at 7.21825841575 h",
    "stall? ECC_CERT_LOOP_s11 heartbeat older than 3 h at 7.21826287588889 h",
    "stall? SAC_C_s10 heartbeat older than 3 h at 7.30161226622222 h",
    "stall? EACH_C_s11 heartbeat older than 3 h at 7.30161609386111 h",
    "stall? ECC_CERT_LOOP_s11 heartbeat older than 3 h at 7.30161880888889 h",
    "stall? SAC_C_s10 heartbeat older than 3 h at 7.38496476555556 h",
    "stall? EACH_C_s11 heartbeat older than 3 h at 7.38496811063889 h",
    "stall? ECC_CERT_LOOP_s11 heartbeat older than 3 h at 7.38497089655556 h",
    "stall? ECC_CERT_LOOP_s11 heartbeat older than 3 h at 7.46831757755556 h",
    "stall? ECC_CERT_LOOP_s11 heartbeat older than 3 h at 7.55166990202778 h",
    "stall? ECC_CERT_LOOP_s11 heartbeat older than 3 h at 7.63502205366667 h",
    "stall? ECC_CERT_LOOP_s11 heartbeat older than 3 h at 7.71837212619444 h",
    "stall? ECC_CERT_LOOP_s11 heartbeat older than 3 h at 7.80171755869444 h",
    "stall? ECC_CERT_LOOP_s11 heartbeat older than 3 h at 7.88506981066667 h",
    "stall? ECC_CERT_LOOP_s10 heartbeat older than 3 h at 7.96841580008333 h",
    "stall? ECC_CERT_LOOP_s11 heartbeat older than 3 h at 7.96842291697222 h",
    "stall? ECC_CERT_LOOP_s10 heartbeat older than 3 h at 8.05176735033333 h",
    "stall? ECC_CERT_LOOP_s11 heartbeat older than 3 h at 8.05177926994444 h",
    "stall? ECC_CERT_LOOP_s10 heartbeat older than 3 h at 8.13512281627778 h",
    "stall? ECC_CERT_LOOP_s11 heartbeat older than 3 h at 8.1351323845 h",
    "stall? ECC_CERT_LOOP_s10 heartbeat older than 3 h at 8.21847229438889 h",
    "stall? ECC_CERT_LOOP_s11 heartbeat older than 3 h at 8.21847997061111 h",
    "stall? ECC_CERT_LOOP_s10 heartbeat older than 3 h at 8.30182420525 h",
    "stall? ECC_CERT_LOOP_s11 heartbeat older than 3 h at 8.30182964041667 h",
    "stall? ECC_CERT_LOOP_s10 heartbeat older than 3 h at 8.38517610205555 h",
    "stall? ECC_CERT_LOOP_s11 heartbeat older than 3 h at 8.38518235772222 h",
    "stall? ECC_CERT_LOOP_s10 heartbeat older than 3 h at 8.46852762616667 h",
    "stall? ECC_CERT_LOOP_s11 heartbeat older than 3 h at 8.46853478766667 h",
    "stall? ECC_CERT_LOOP_s10 heartbeat older than 3 h at 8.55187874647222 h",
    "stall? ECC_CERT_LOOP_s11 heartbeat older than 3 h at 8.55188569269444 h",
    "stall? ECC_CERT_LOOP_s10 heartbeat older than 3 h at 8.63522781061111 h",
    "stall? ECC_CERT_LOOP_s11 heartbeat older than 3 h at 8.63523300452778 h",
    "stall? SAC_C_s11 heartbeat older than 3 h at 8.71857071677778 h",
    "stall? ECC_CERT_LOOP_s10 heartbeat older than 3 h at 8.71857533386111 h",
    "stall? SAC_C_s11 heartbeat older than 3 h at 8.80192110319444 h",
    "stall? ECC_CERT_LOOP_s10 heartbeat older than 3 h at 8.80192437522222 h",
    "stall? SAC_C_s11 heartbeat older than 3 h at 8.88527018127778 h",
    "stall? ECC_CERT_LOOP_s10 heartbeat older than 3 h at 8.88527254097222 h",
    "stall? SAC_C_s11 heartbeat older than 3 h at 8.96861880583333 h",
    "stall? ECC_CERT_LOOP_s10 heartbeat older than 3 h at 8.96862211930556 h",
    "stall? ECC_CERT_LOOP_s10 heartbeat older than 3 h at 9.05197054213889 h",
    "stall? ECC_CERT_LOOP_s10 heartbeat older than 3 h at 9.13531567858333 h",
    "stall? ECC_CERT_LOOP_s10 heartbeat older than 3 h at 9.21865550744444 h",
    "stall? ECC_CERT_LOOP_s10 heartbeat older than 3 h at 9.30199514363889 h",
    "stall? ECC_CERT_LOOP_s10 heartbeat older than 3 h at 9.38533641358333 h",
    "stall? ECC_CERT_LOOP_s10 heartbeat older than 3 h at 9.46867989266667 h",
    "stall? ECC_CERT_LOOP_s10 heartbeat older than 3 h at 9.55202074527778 h",
    "stall? ECC_CERT_LOOP_s10 heartbeat older than 3 h at 9.63536410505555 h",
    "stall? ECC_CERT_LOOP_s10 heartbeat older than 3 h at 9.7187072685 h",
    "stall? ECC_CERT_LOOP_s10 heartbeat older than 3 h at 9.80204734327778 h",
    "stall? EACH_C_s10 heartbeat older than 3 h at 10.0520705759722 h",
    "stall? EACH_C_s10 heartbeat older than 3 h at 10.1354116032778 h",
    "stall? EACH_C_s10 heartbeat older than 3 h at 10.2187520763056 h",
    "stall? EACH_C_s10 heartbeat older than 3 h at 10.3020951972778 h",
    "stall? EACH_C_s10 heartbeat older than 3 h at 10.3854374182778 h",
    "stall? SAC_C_s10 heartbeat older than 3 h at 10.4687766426667 h",
    "stall? EACH_C_s11 heartbeat older than 3 h at 10.4687787233611 h",
    "stall? EACH_C_s10 heartbeat older than 3 h at 10.4687828067222 h",
    "stall? SAC_C_s10 heartbeat older than 3 h at 10.5521364789167 h",
    "stall? EACH_C_s11 heartbeat older than 3 h at 10.55214303 h",
    "stall? EACH_C_s10 heartbeat older than 3 h at 10.5521533615556 h",
    "stall? SAC_C_s10 heartbeat older than 3 h at 10.6355051059167 h",
    "stall? EACH_C_s11 heartbeat older than 3 h at 10.6355113828611 h",
    "stall? EACH_C_s10 heartbeat older than 3 h at 10.6355249306667 h",
    "stall? SAC_C_s10 heartbeat older than 3 h at 10.7188726284444 h",
    "stall? EACH_C_s11 heartbeat older than 3 h at 10.7188786455833 h",
    "stall? EACH_C_s10 heartbeat older than 3 h at 10.7188881303611 h",
    "stall? SAC_C_s10 heartbeat older than 3 h at 10.8022358318611 h",
    "stall? EACH_C_s11 heartbeat older than 3 h at 10.8022401693611 h",
    "stall? EACH_C_s10 heartbeat older than 3 h at 10.8022464361111 h",
    "stall? SAC_C_s10 heartbeat older than 3 h at 10.8855979006944 h",
    "stall? EACH_C_s11 heartbeat older than 3 h at 10.8856044576111 h",
    "stall? EACH_C_s10 heartbeat older than 3 h at 10.8856125496944 h",
    "stall? SAC_C_s10 heartbeat older than 3 h at 10.9689608223056 h",
    "stall? EACH_C_s11 heartbeat older than 3 h at 10.9689641811667 h",
    "stall? EACH_C_s10 heartbeat older than 3 h at 10.9689697661111 h",
    "stall? SAC_C_s10 heartbeat older than 3 h at 11.0523165497222 h",
    "stall? EACH_C_s11 heartbeat older than 3 h at 11.0523215620556 h",
    "stall? EACH_C_s10 heartbeat older than 3 h at 11.0523268646111 h",
    "stall? SAC_C_s10 heartbeat older than 3 h at 11.1356796818611 h",
    "stall? EACH_C_s11 heartbeat older than 3 h at 11.1356870582222 h",
    "stall? EACH_C_s10 heartbeat older than 3 h at 11.1356981301667 h",
    "stall? SAC_C_s10 heartbeat older than 3 h at 11.2190563601667 h",
    "stall? EACH_C_s11 heartbeat older than 3 h at 11.2190626505556 h",
    "stall? EACH_C_s10 heartbeat older than 3 h at 11.2190729809167 h",
    "stall? SAC_C_s10 heartbeat older than 3 h at 11.30248555025 h",
    "stall? EACH_C_s11 heartbeat older than 3 h at 11.3024898696111 h",
    "stall? EACH_C_s10 heartbeat older than 3 h at 11.3024961551389 h",
    "stall? SAC_C_s10 heartbeat older than 3 h at 11.3858419671389 h",
    "stall? EACH_C_s11 heartbeat older than 3 h at 11.3858478370833 h",
    "stall? EACH_C_s10 heartbeat older than 3 h at 11.3858541134167 h",
    "stall? EACH_C_s11 heartbeat older than 3 h at 11.4692047448889 h",
    "stall? EACH_C_s10 heartbeat older than 3 h at 11.4692092119722 h",
    "stall? EACH_C_s11 heartbeat older than 3 h at 11.5525617744167 h",
    "stall? EACH_C_s10 heartbeat older than 3 h at 11.5525725435556 h",
    "stall? EACH_C_s11 heartbeat older than 3 h at 11.6359180347778 h",
    "stall? EACH_C_s10 heartbeat older than 3 h at 11.6359243378611 h",
    "stall? ECC_CERT_LOOP_s11 heartbeat older than 3 h at 11.7192686807222 h",
    "stall? EACH_C_s10 heartbeat older than 3 h at 11.71927184075 h",
    "stall? ECC_CERT_LOOP_s11 heartbeat older than 3 h at 11.8026153240556 h",
    "stall? EACH_C_s10 heartbeat older than 3 h at 11.8026203672778 h",
    "stall? ECC_CERT_LOOP_s11 heartbeat older than 3 h at 11.8859629017778 h",
    "stall? ECC_CERT_LOOP_s11 heartbeat older than 3 h at 11.9693053680278 h",
    "stall? ECC_CERT_LOOP_s11 heartbeat older than 3 h at 12.0526467503889 h",
    "stall? ECC_CERT_LOOP_s11 heartbeat older than 3 h at 12.1359928105833 h",
    "stall? ECC_CERT_LOOP_s11 heartbeat older than 3 h at 12.2193347235556 h",
    "stall? ECC_CERT_LOOP_s11 heartbeat older than 3 h at 12.30267910675 h",
    "stall? ECC_CERT_LOOP_s11 heartbeat older than 3 h at 12.3860210255833 h",
    "stall? ECC_CERT_LOOP_s11 heartbeat older than 3 h at 12.4693651193889 h",
    "stall? ECC_CERT_LOOP_s11 heartbeat older than 3 h at 12.5527050991389 h",
    "stall? ECC_CERT_LOOP_s11 heartbeat older than 3 h at 12.6360489621389 h",
    "stall? ECC_CERT_LOOP_s11 heartbeat older than 3 h at 12.71938996975 h",
    "stall? ECC_CERT_LOOP_s11 heartbeat older than 3 h at 12.8027303660556 h",
    "stall? ECC_CERT_LOOP_s10 heartbeat older than 3 h at 12.88606742625 h",
    "stall? ECC_CERT_LOOP_s11 heartbeat older than 3 h at 12.88606742625 h",
    "stall? ECC_CERT_LOOP_s10 heartbeat older than 3 h at 12.9694009937222 h",
    "stall? ECC_CERT_LOOP_s11 heartbeat older than 3 h at 12.9694009937222 h",
    "stall? ECC_CERT_LOOP_s10 heartbeat older than 3 h at 13.0527413915278 h",
    "stall? ECC_CERT_LOOP_s11 heartbeat older than 3 h at 13.0527430627222 h",
    "stall? ECC_CERT_LOOP_s10 heartbeat older than 3 h at 13.1360784395556 h",
    "stall? ECC_CERT_LOOP_s10 heartbeat older than 3 h at 13.2194131408056 h",
    "stall? ECC_CERT_LOOP_s10 heartbeat older than 3 h at 13.3027504073333 h",
    "stall? ECC_CERT_LOOP_s10 heartbeat older than 3 h at 13.3860863852222 h",
    "stall? ECC_CERT_LOOP_s10 heartbeat older than 3 h at 13.46942367825 h",
    "stall? ECC_CERT_LOOP_s10 heartbeat older than 3 h at 13.5527586911667 h",
    "stall? ECC_CERT_LOOP_s10 heartbeat older than 3 h at 13.6360964244722 h",
    "stall? ECC_CERT_LOOP_s10 heartbeat older than 3 h at 13.7194307733056 h",
    "stall? ECC_CERT_LOOP_s10 heartbeat older than 3 h at 13.802768591 h",
    "stall? ECC_CERT_LOOP_s10 heartbeat older than 3 h at 13.8861067849444 h",
    "stall? ECC_CERT_LOOP_s10 heartbeat older than 3 h at 13.9694479841667 h",
    "stall? ECC_CERT_LOOP_s10 heartbeat older than 3 h at 14.0527862045833 h",
    "stall? ECC_CERT_LOOP_s10 heartbeat older than 3 h at 14.1361246372222 h",
    "stall? ECC_CERT_LOOP_s10 heartbeat older than 3 h at 14.2194599885 h",
    "stall? ECC_CERT_LOOP_s10 heartbeat older than 3 h at 14.3027966890278 h",
    "stall? ECC_CERT_LOOP_s10 heartbeat older than 3 h at 14.3861334861667 h"
  ],
  "unsafe": {
    "SAC_C_s10": [],
    "SAC_C_s11": [],
    "EACH_C_s10": [],
    "EACH_C_s11": [],
    "ECC_CERT_LOOP_s10": [
      {
        "steps": 1170,
        "worker": 1,
        "event": "unsafe",
        "time_s": 128.1,
        "delta_v": 1.2403640570988124,
        "mission": {
          "deadline_s": 159.2,
          "urgent": true,
          "budget_m_s": 3.0,
          "time_cost_per_s": 0.1,
          "fuel_cost_per_m_s": 10.0
        },
        "entry_bias": null,
        "guided": 0,
        "return_true": -175.21364057098813,
        "return_shaped": -162.2025506787026,
        "exec_stats": {
          "stages": 8,
          "holds": 0,
          "rejected": 0,
          "bridge_attempts": 76,
          "rate_limited": 3,
          "entries_lost": 0,
          "window_recoveries": 0,
          "cert_interventions": 1
        }
      }
    ],
    "ECC_CERT_LOOP_s11": []
  }
}
```
