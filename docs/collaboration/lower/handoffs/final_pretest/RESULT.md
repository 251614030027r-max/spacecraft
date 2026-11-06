# 预实验，2 种子 × 2000 决策，评估开局 294100–294147

冻结科学版本 `fd789126cc822d3650e636ff2169c8757ff3bdb1`，协作执行书 `69616f11`。实验有效性 **有效**。

| 有效性检查 | 结果 | 证据 |
|---|---|---|
| 版本与配置可查 | 是 | REGISTRY_RESOLVED.json、ENV_LOCK.txt、检查点 SHA256 |
| 测试与全流程 dryrun | 是 | pytest.log、DRY/CHECK.json、DRY/MANIFESTS.txt、执行回执 |
| 十条训练完成 | 是 | 十份 B/TRAIN_*/finished.json |
| 20 组评估各 48 个开局 | 是 | CHECK.json、C/**、S/** 逐组开局身份 |
| 执行器统计可核验 | 是 | 训练 exec_stats，模型评估 decisions.mode，C/S 入口丢失与窗口恢复 |
| 图表可重建 | 是 | figures.log、五组原生图表、TABLES.json |

研究观察摘自原生 TABLES.json。以下为各模型与固定策略的原生逐种子统计、配对差及几何扫描统计。

```json
{
  "curve_auc_by_arm": {
    "DUP": [
      10.74924014949532,
      -3.8539369942567623
    ],
    "EACH": [
      12.793802371231141,
      16.834202367143288
    ],
    "NO_GEOM": [
      -54.767685216706745,
      -54.084699997766464
    ],
    "SAC_ENG": [
      21.89350597143087,
      6.182974536979467
    ],
    "SAC": [
      -4.930051774880031,
      12.656787493705309
    ]
  },
  "final": {
    "openings": "294100-294147",
    "entries": {
      "EACH": {
        "display": "EACH (U1+C1)",
        "per_seed": {
          "10": {
            "n": 48,
            "capture": 43,
            "urgent_n": 23,
            "urgent_capture": 18,
            "unsafe": 0,
            "timeout": 5,
            "resource_exhausted": 0,
            "mean_return": 35.5680079684663,
            "median_return": 50.9424852003784,
            "entries_lost": 1
          },
          "11": {
            "n": 48,
            "capture": 47,
            "urgent_n": 23,
            "urgent_capture": 22,
            "unsafe": 0,
            "timeout": 1,
            "resource_exhausted": 0,
            "mean_return": 55.77118001398404,
            "median_return": 63.76466193271352,
            "entries_lost": 1
          }
        }
      },
      "SAC": {
        "display": "SAC",
        "per_seed": {
          "10": {
            "n": 48,
            "capture": 36,
            "urgent_n": 23,
            "urgent_capture": 11,
            "unsafe": 0,
            "timeout": 12,
            "resource_exhausted": 0,
            "mean_return": 10.875692829280718,
            "median_return": 45.92203334262152,
            "entries_lost": 0
          },
          "11": {
            "n": 48,
            "capture": 44,
            "urgent_n": 23,
            "urgent_capture": 19,
            "unsafe": 0,
            "timeout": 4,
            "resource_exhausted": 0,
            "mean_return": 44.065282476614236,
            "median_return": 59.00148973201479,
            "entries_lost": 2
          }
        }
      },
      "SAC_ENG": {
        "display": "SAC + planner features",
        "per_seed": {
          "10": {
            "n": 48,
            "capture": 43,
            "urgent_n": 23,
            "urgent_capture": 18,
            "unsafe": 0,
            "timeout": 5,
            "resource_exhausted": 0,
            "mean_return": 38.19243929939277,
            "median_return": 56.57442357652903,
            "entries_lost": 2
          },
          "11": {
            "n": 48,
            "capture": 42,
            "urgent_n": 23,
            "urgent_capture": 17,
            "unsafe": 0,
            "timeout": 6,
            "resource_exhausted": 0,
            "mean_return": 33.27636009921475,
            "median_return": 54.23124369888586,
            "entries_lost": 0
          }
        }
      },
      "DUP": {
        "display": "w/o U1 (same width)",
        "per_seed": {
          "10": {
            "n": 48,
            "capture": 39,
            "urgent_n": 23,
            "urgent_capture": 14,
            "unsafe": 0,
            "timeout": 9,
            "resource_exhausted": 0,
            "mean_return": 20.792295235082975,
            "median_return": 48.18874158863618,
            "entries_lost": 2
          },
          "11": {
            "n": 48,
            "capture": 37,
            "urgent_n": 23,
            "urgent_capture": 12,
            "unsafe": 0,
            "timeout": 11,
            "resource_exhausted": 0,
            "mean_return": 9.44021322546691,
            "median_return": 39.516201468919654,
            "entries_lost": 1
          }
        }
      },
      "NO_GEOM": {
        "display": "w/o C1 geometry",
        "per_seed": {
          "10": {
            "n": 48,
            "capture": 20,
            "urgent_n": 23,
            "urgent_capture": 6,
            "unsafe": 0,
            "timeout": 28,
            "resource_exhausted": 0,
            "mean_return": -56.5914864491063,
            "median_return": -121.12366950531161,
            "entries_lost": 1
          },
          "11": {
            "n": 48,
            "capture": 21,
            "urgent_n": 23,
            "urgent_capture": 7,
            "unsafe": 0,
            "timeout": 27,
            "resource_exhausted": 0,
            "mean_return": -51.792968631567135,
            "median_return": -119.1397947462188,
            "entries_lost": 1
          }
        }
      },
      "pure": {
        "display": "Pure MPC",
        "per_seed": {
          "None": {
            "n": 48,
            "capture": 29,
            "urgent_n": 23,
            "urgent_capture": 11,
            "unsafe": 11,
            "timeout": 5,
            "resource_exhausted": 3,
            "mean_return": -53.378619357616394,
            "median_return": 6.890751027885848,
            "entries_lost": 0
          }
        }
      },
      "gstar": {
        "display": "Direct G*",
        "per_seed": {
          "None": {
            "n": 48,
            "capture": 45,
            "urgent_n": 23,
            "urgent_capture": 20,
            "unsafe": 0,
            "timeout": 2,
            "resource_exhausted": 0,
            "mean_return": 50.13785447458225,
            "median_return": 61.390651173467106,
            "entries_lost": 0
          }
        }
      },
      "rally": {
        "display": "Deadline rule",
        "per_seed": {
          "None": {
            "n": 48,
            "capture": 34,
            "urgent_n": 23,
            "urgent_capture": 9,
            "unsafe": 0,
            "timeout": 14,
            "resource_exhausted": 0,
            "mean_return": -1.070917412530677,
            "median_return": 32.47526357311021,
            "entries_lost": 1
          }
        }
      },
      "b_10": {
        "display": "Basic layering (a_p=1)",
        "per_seed": {
          "None": {
            "n": 48,
            "capture": 21,
            "urgent_n": 23,
            "urgent_capture": 7,
            "unsafe": 0,
            "timeout": 27,
            "resource_exhausted": 0,
            "mean_return": -54.90636335627715,
            "median_return": -122.29934262833947,
            "entries_lost": 0
          }
        }
      },
      "c_10_0": {
        "display": "Fixed intent (1, 0)",
        "per_seed": {
          "None": {
            "n": 48,
            "capture": 39,
            "urgent_n": 23,
            "urgent_capture": 14,
            "unsafe": 0,
            "timeout": 9,
            "resource_exhausted": 0,
            "mean_return": 18.404743288117846,
            "median_return": 48.281801940968776,
            "entries_lost": 0
          }
        }
      }
    },
    "paired_vs_EACH": {
      "SAC": {
        "n": 48,
        "mean_diff": 18.199106338277694,
        "median_diff": -0.695169079107572,
        "without_top2": 12.455763776794754,
        "both_success_first_seed": {
          "n": 36,
          "time_s": [
            133.0222222222222,
            130.1
          ],
          "delta_v": [
            1.3447880211309533,
            1.2109703394045859
          ],
          "only_first_captures": 7,
          "only_second_captures": 0
        }
      },
      "SAC_ENG": {
        "n": 48,
        "mean_diff": 9.93519429192141,
        "median_diff": -0.1361395760758768,
        "without_top2": 3.699841564502355,
        "both_success_first_seed": {
          "n": 42,
          "time_s": [
            131.9142857142857,
            128.1190476190476
          ],
          "delta_v": [
            1.329446166914229,
            1.253785338085829
          ],
          "only_first_captures": 1,
          "only_second_captures": 1
        }
      },
      "DUP": {
        "n": 48,
        "mean_diff": 30.553339760950227,
        "median_diff": 0.06805981628566826,
        "without_top2": 22.920938544472364,
        "both_success_first_seed": {
          "n": 39,
          "time_s": [
            135.02051282051283,
            135.3205128205128
          ],
          "delta_v": [
            1.363281414054371,
            1.2915875146134097
          ],
          "only_first_captures": 4,
          "only_second_captures": 0
        }
      },
      "NO_GEOM": {
        "n": 48,
        "mean_diff": 99.86182153156189,
        "median_diff": 96.51197778930515,
        "without_top2": 94.8088258182634,
        "both_success_first_seed": {
          "n": 20,
          "time_s": [
            129.89000000000001,
            147.60999999999999
          ],
          "delta_v": [
            1.3120267666517207,
            1.3033291382782326
          ],
          "only_first_captures": 23,
          "only_second_captures": 0
        }
      },
      "pure": {
        "n": 48,
        "mean_diff": 99.04821334884157,
        "median_diff": 32.03575645873194,
        "without_top2": 89.93959565139205,
        "both_success_first_seed": {
          "n": 27,
          "time_s": [
            136.11111111111111,
            156.38888888888889
          ],
          "delta_v": [
            1.3821304575954463,
            1.8216103805548853
          ],
          "only_first_captures": 16,
          "only_second_captures": 2
        }
      },
      "gstar": {
        "n": 48,
        "mean_diff": -4.468260483357075,
        "median_diff": -1.2949508875042355,
        "without_top2": -11.239530350741378,
        "both_success_first_seed": {
          "n": 42,
          "time_s": [
            132.4238095238095,
            114.15952380952383
          ],
          "delta_v": [
            1.3320170352633265,
            1.282919224057714
          ],
          "only_first_captures": 1,
          "only_second_captures": 3
        }
      },
      "rally": {
        "n": 48,
        "mean_diff": 46.74051140375585,
        "median_diff": 0.8936641938149847,
        "without_top2": 39.71558561321159,
        "both_success_first_seed": {
          "n": 34,
          "time_s": [
            137.6794117647059,
            153.0764705882353
          ],
          "delta_v": [
            1.3934024621607284,
            1.3607178036131824
          ],
          "only_first_captures": 9,
          "only_second_captures": 0
        }
      },
      "b_10": {
        "n": 48,
        "mean_diff": 100.57595734750232,
        "median_diff": 98.80189692928653,
        "without_top2": 95.75699981677441,
        "both_success_first_seed": {
          "n": 21,
          "time_s": [
            127.42857142857143,
            147.1571428571429
          ],
          "delta_v": [
            1.2880109083210498,
            1.4582321087904613
          ],
          "only_first_captures": 22,
          "only_second_captures": 0
        }
      },
      "c_10_0": {
        "n": 48,
        "mean_diff": 27.264850703107324,
        "median_diff": 0.03513263704821412,
        "without_top2": 19.354340640582752,
        "both_success_first_seed": {
          "n": 39,
          "time_s": [
            132.24615384615385,
            135.71282051282054
          ],
          "delta_v": [
            1.329851402033211,
            1.3263461637708924
          ],
          "only_first_captures": 4,
          "only_second_captures": 0
        }
      }
    }
  },
  "window": {
    "openings": "294100-294147",
    "pairs": {
      "EACH (U1+C1) s10": {
        "n": 48,
        "window": {
          "n": 48,
          "capture": 43,
          "urgent_n": 23,
          "urgent_capture": 18,
          "unsafe": 0,
          "timeout": 5,
          "resource_exhausted": 0,
          "mean_return": 35.5680079684663,
          "median_return": 50.9424852003784,
          "entries_lost": 1
        },
        "rigid": {
          "n": 48,
          "capture": 40,
          "urgent_n": 23,
          "urgent_capture": 17,
          "unsafe": 0,
          "timeout": 7,
          "resource_exhausted": 1,
          "mean_return": 19.55404127901332,
          "median_return": 47.63155940484588,
          "entries_lost": 50
        },
        "paired_return_mean": 16.013966689452975,
        "paired_return_median": 0.02050592730136458,
        "paired_positive": 15,
        "paired_negative": 2,
        "both_success": {
          "n": 40,
          "time_s": [
            130.3225,
            134.8
          ],
          "delta_v": [
            1.306114393546324,
            1.3677495193488713
          ],
          "only_first_captures": 3,
          "only_second_captures": 0
        }
      },
      "EACH (U1+C1) s11": {
        "n": 48,
        "window": {
          "n": 48,
          "capture": 47,
          "urgent_n": 23,
          "urgent_capture": 22,
          "unsafe": 0,
          "timeout": 1,
          "resource_exhausted": 0,
          "mean_return": 55.77118001398404,
          "median_return": 63.76466193271352,
          "entries_lost": 1
        },
        "rigid": {
          "n": 48,
          "capture": 46,
          "urgent_n": 23,
          "urgent_capture": 21,
          "unsafe": 0,
          "timeout": 2,
          "resource_exhausted": 0,
          "mean_return": 51.27702085338808,
          "median_return": 62.15022382819302,
          "entries_lost": 35
        },
        "paired_return_mean": 4.494159160595971,
        "paired_return_median": 0.0,
        "paired_positive": 8,
        "paired_negative": 1,
        "both_success": {
          "n": 46,
          "time_s": [
            128.67391304347828,
            129.8217391304348
          ],
          "delta_v": [
            1.2300891738344384,
            1.244411287503769
          ],
          "only_first_captures": 1,
          "only_second_captures": 0
        }
      },
      "Direct G*": {
        "n": 48,
        "window": {
          "n": 48,
          "capture": 45,
          "urgent_n": 23,
          "urgent_capture": 20,
          "unsafe": 0,
          "timeout": 2,
          "resource_exhausted": 0,
          "mean_return": 50.13785447458225,
          "median_return": 61.390651173467106,
          "entries_lost": 0
        },
        "rigid": {
          "n": 48,
          "capture": 46,
          "urgent_n": 23,
          "urgent_capture": 21,
          "unsafe": 0,
          "timeout": 1,
          "resource_exhausted": 0,
          "mean_return": 54.442181944746515,
          "median_return": 61.262410719309116,
          "entries_lost": 3
        },
        "paired_return_mean": -4.304327470164266,
        "paired_return_median": 0.0,
        "paired_positive": 12,
        "paired_negative": 8,
        "both_success": {
          "n": 45,
          "time_s": [
            112.82888888888887,
            114.61111111111109
          ],
          "delta_v": [
            1.263049239971641,
            1.2576664346300135
          ],
          "only_first_captures": 0,
          "only_second_captures": 1
        }
      },
      "Deadline rule": {
        "n": 48,
        "window": {
          "n": 48,
          "capture": 34,
          "urgent_n": 23,
          "urgent_capture": 9,
          "unsafe": 0,
          "timeout": 14,
          "resource_exhausted": 0,
          "mean_return": -1.070917412530677,
          "median_return": 32.47526357311021,
          "entries_lost": 1
        },
        "rigid": {
          "n": 48,
          "capture": 32,
          "urgent_n": 23,
          "urgent_capture": 9,
          "unsafe": 0,
          "timeout": 16,
          "resource_exhausted": 0,
          "mean_return": -9.568526170727457,
          "median_return": 31.900983436319287,
          "entries_lost": 45
        },
        "paired_return_mean": 8.497608758196778,
        "paired_return_median": 0.0,
        "paired_positive": 12,
        "paired_negative": 5,
        "both_success": {
          "n": 32,
          "time_s": [
            148.875,
            150.2
          ],
          "delta_v": [
            1.3430678440979391,
            1.3455537114234297
          ],
          "only_first_captures": 2,
          "only_second_captures": 0
        }
      },
      "Fixed intent (1, 0)": {
        "n": 48,
        "window": {
          "n": 48,
          "capture": 39,
          "urgent_n": 23,
          "urgent_capture": 14,
          "unsafe": 0,
          "timeout": 9,
          "resource_exhausted": 0,
          "mean_return": 18.404743288117846,
          "median_return": 48.281801940968776,
          "entries_lost": 0
        },
        "rigid": {
          "n": 48,
          "capture": 38,
          "urgent_n": 23,
          "urgent_capture": 14,
          "unsafe": 0,
          "timeout": 10,
          "resource_exhausted": 0,
          "mean_return": 12.834900500860085,
          "median_return": 44.36884503269475,
          "entries_lost": 35
        },
        "paired_return_mean": 5.569842787257766,
        "paired_return_median": 0.0,
        "paired_positive": 9,
        "paired_negative": 5,
        "both_success": {
          "n": 38,
          "time_s": [
            136.06578947368425,
            137.9473684210527
          ],
          "delta_v": [
            1.3274594291006587,
            1.3503501327631955
          ],
          "only_first_captures": 1,
          "only_second_captures": 0
        }
      }
    }
  },
  "geometry": {
    "opening": 294200,
    "beta_deg": 93.21875133822846,
    "rate_deg_s": 2.161664483065137,
    "ranges": {
      "8": {
        "gate_share": 0.22444444444444445,
        "feasible_share": 0.16666666666666666,
        "construction_error_deg": [
          4.263256414560601e-14,
          5.684341886080802e-14
        ]
      },
      "10": {
        "gate_share": 0.22444444444444445,
        "feasible_share": 0.14333333333333334,
        "construction_error_deg": [
          4.263256414560601e-14,
          5.684341886080802e-14
        ]
      },
      "12": {
        "gate_share": 0.22444444444444445,
        "feasible_share": 0.12333333333333334,
        "construction_error_deg": [
          4.263256414560601e-14,
          5.684341886080802e-14
        ]
      }
    }
  }
}
```

训练和评估墙钟、重启或停滞记录见原生时间线。Dryrun 训练以最多六条在途限制内存峰值，其余科学参数来自冻结注册表。

首次预检的外部 PowerShell 退出属性检查误报。12 条训练的原生 40 步完成回执和观测器终态均已核验，接续从尚未开始的 dryrun 评估执行，保留原测试和训练。详见 execution_records。

```text
start 11/05/2026 15:25:07
batch EACH,SAC,SAC_ENG launch ok at 0.0500629309722222 h
batch EACH,SAC,SAC_ENG end 8.5512278235 h
batch DUP,NO_GEOM launch ok at 8.60125319330555 h
batch DUP,NO_GEOM end 14.4348970300278 h
done eval C:\Users\35884\Documents\Spacecraft\过程文件\final_pretest\B\TRAIN_EACH_s10 at 14.6215101508333 h
done eval C:\Users\35884\Documents\Spacecraft\过程文件\final_pretest\B\TRAIN_EACH_s11 at 14.7916524601111 h
done eval C:\Users\35884\Documents\Spacecraft\过程文件\final_pretest\B\TRAIN_SAC_s10 at 14.9347486600833 h
done eval C:\Users\35884\Documents\Spacecraft\过程文件\final_pretest\B\TRAIN_SAC_s11 at 15.1052687301944 h
done eval C:\Users\35884\Documents\Spacecraft\过程文件\final_pretest\B\TRAIN_SAC_ENG_s10 at 15.2807335042778 h
done eval C:\Users\35884\Documents\Spacecraft\过程文件\final_pretest\B\TRAIN_SAC_ENG_s11 at 15.45420428825 h
done eval C:\Users\35884\Documents\Spacecraft\过程文件\final_pretest\B\TRAIN_DUP_s10 at 15.6162503586389 h
done eval C:\Users\35884\Documents\Spacecraft\过程文件\final_pretest\B\TRAIN_DUP_s11 at 15.781532263 h
done eval C:\Users\35884\Documents\Spacecraft\过程文件\final_pretest\B\TRAIN_NO_GEOM_s10 at 16.0191997229722 h
done eval C:\Users\35884\Documents\Spacecraft\过程文件\final_pretest\B\TRAIN_NO_GEOM_s11 at 16.2184709346111 h
done eval C:\Users\35884\Documents\Spacecraft\过程文件\final_pretest\B\TRAIN_EACH_s10 at 16.4321606316944 h
done eval C:\Users\35884\Documents\Spacecraft\过程文件\final_pretest\B\TRAIN_EACH_s11 at 16.6038431754722 h
done geoscan  at 16.6500563575833 h
done screen pure at 16.8979201044167 h
done screen gstar at 17.112964444 h
done screen rally at 17.2581482804167 h
done screen b_10 at 17.5455488088889 h
done screen c_10_0 at 17.7415095685833 h
done screen gstar at 17.9521741849722 h
done screen rally at 18.0937668269167 h
done screen c_10_0 at 18.29329994175 h
done traj  at 18.40827968575 h
all done 18.4118003829722 h

```

科学证据 `56841a04e0a5918f489097ebc5f05137c4d27690` 的 `evidence/final_pretest/`，1069 个 Git blob 经原字节核验。原模型和 TensorBoard 日志保留于 `C:\Users\35884\Documents\Spacecraft\过程文件\final_pretest`。
