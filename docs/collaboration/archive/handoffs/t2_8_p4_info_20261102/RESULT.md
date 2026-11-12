# T2.8 P4 事实交付

执行书 collab `a0b0b23`，detached科学 `8016b16df994ed2b206c3eea7096084494006288`；pytest 60 passed、退出码0。按原命令顺序执行：10进程真值分支，再对夜间四条训练500/1000/1500/2000的16个检查点打分，然后原生p4.py汇总。未训练，科学代码未修改。

覆盖：40开局2910000–2910039，紧急19、经济21；16检查点。obs_mismatch=67，nonterminal_branches=0。这两项若非0按执行书原样交付并注明，不改动原始结果。判定与解释由上层进行。

## 墙钟

```json
{
  "jst": "2026-11-02T19:48:38.197522+09:00",
  "stages_seconds": {
    "pytest": 120.06690764427185,
    "truth_branches": 17417.72495865822,
    "score": 30.035371780395508,
    "summary": 30.03070306777954
  },
  "total_seconds": 17597.857941150665
}
```

## 原生汇总

```json
{
  "decisions": 154,
  "phases": {
    "urgent_window": 52,
    "urgent_early": 34,
    "economy": 68
  },
  "decisive_pairs": {
    "urgent_window": 204,
    "urgent_early": 298,
    "economy": 367
  },
  "obs_mismatch": 67,
  "nonterminal_branches": 0,
  "critic_by_arm_phase": {
    "mve/urgent_window": {
      "checkpoints": 6,
      "accuracy_mean": 0.7058823529411765,
      "accuracy_min": 0.6519607843137255
    },
    "mve/urgent_early": {
      "checkpoints": 6,
      "accuracy_mean": 0.7628635346756152,
      "accuracy_min": 0.7315436241610739
    },
    "mve/economy": {
      "checkpoints": 6,
      "accuracy_mean": 0.6684831970935514,
      "accuracy_min": 0.6321525885558583
    },
    "sac/urgent_window": {
      "checkpoints": 6,
      "accuracy_mean": 0.6037581699346405,
      "accuracy_min": 0.5245098039215687
    },
    "sac/urgent_early": {
      "checkpoints": 6,
      "accuracy_mean": 0.7393736017897092,
      "accuracy_min": 0.714765100671141
    },
    "sac/economy": {
      "checkpoints": 6,
      "accuracy_mean": 0.6012715712988193,
      "accuracy_min": 0.5231607629427792
    }
  },
  "signals": {
    "label": {
      "selection": {
        "critic_right": 1439,
        "broken": 6,
        "net": 60,
        "pairs_urgent_early": 1116,
        "critic_wrong": 757,
        "fixed": 66,
        "pairs_urgent_window": 1080,
        "fix": 0.08718626155878467,
        "break": 0.004169562195969423,
        "pass": false
      },
      "heldout": {
        "critic_wrong": 978,
        "fixed": 134,
        "net": 134,
        "pairs_urgent_early": 2460,
        "critic_right": 2850,
        "broken": 0,
        "pairs_urgent_window": 1368,
        "fix": 0.13701431492842536,
        "break": 0.0,
        "pass": false
      },
      "checkpoints_net_positive": "12/12",
      "carries_missing_information": false
    },
    "opp": {
      "selection": {
        "critic_wrong": 1978,
        "fixed": 589,
        "net": 402,
        "pairs_economy": 2940,
        "critic_right": 3158,
        "broken": 187,
        "pairs_urgent_early": 1116,
        "pairs_urgent_window": 1080,
        "fix": 0.29777553083923153,
        "break": 0.05921469284357188,
        "pass": false
      },
      "heldout": {
        "critic_right": 3927,
        "broken": 69,
        "net": 225,
        "pairs_economy": 1464,
        "critic_wrong": 1365,
        "fixed": 294,
        "pairs_urgent_early": 2460,
        "pairs_urgent_window": 1368,
        "fix": 0.2153846153846154,
        "break": 0.01757066462948816,
        "pass": false
      },
      "checkpoints_net_positive": "12/12",
      "carries_missing_information": false
    }
  }
}
```

## 输入检查点SHA256

```json
{
  "e_mve_s0_500": "8d7216f0caff2f68242aa346ac14bab3ccdf45f01c65f8a7a163ae21f0296d38",
  "e_mve_s0_1000": "ceb957f58ea1a828f729dc730f8b50595aa189a4fccffe4c09e487590ae11169",
  "e_mve_s0_1500": "660acae641eb87fbca3687de326fc7a51604cfffc752e31e9d8f64578270dd3d",
  "e_mve_s0_2000": "aee66141e71933d766907e74497ad5ee3620126258b8419345afd3bd951a5323",
  "e_mve_s1_500": "906a79a3c08140fb372bbc671fa2a476b5e95f13feaf76d89dd380adcfd89957",
  "e_mve_s1_1000": "b70bce0cd4ede66998e8da1fa6aa4a74811f1bc8d872c65d348e09e80ed606bd",
  "e_mve_s1_1500": "c2b8ff11a3f3de9b2eb45ac8aa161a0d8dd308293668753679e71d96efabb394",
  "e_mve_s1_2000": "e0b324cc75ef43204bb48f4d7183ef436efb38a56192b5d3bb5c1e727edf076f",
  "e_sac_s0_500": "d7458ff380c40e57f085d1c11ff3e5d28d54e99dd9f8896672de0335ee2565f7",
  "e_sac_s0_1000": "ef72b2035c3e388f62dd3e81602dac2a4624f5734f25133b0d8f42a5f7ef20ee",
  "e_sac_s0_1500": "0fab5990c5af0b8a1c2fb03844ccb97b725ebc4dc01b561cad22f3e6a7d7d957",
  "e_sac_s0_2000": "de9be6f0c4c68a5fec4fdffcd1d68f9daee28f5710a29b22d2ea55147cd0ea0b",
  "e_sac_s1_500": "48e863908b12ebd951d7e9b7d393db41fada02d94d2fb7919e7bb7bacc92f61a",
  "e_sac_s1_1000": "ec7f1d1eaa4dbf5819a244f7c0832406b7b9f4a4082c035fdda8c62485bd8093",
  "e_sac_s1_1500": "f279eb9b39c3cc262fe97c61ae90ac1a118ee5f103c1e3b5d0688d9876586f54",
  "e_sac_s1_2000": "044972fdaa615dfbc1fc17f54741869d25b976d6319f57848f495a03c0c8f55f"
}
```

全部info逐开局、原始KEYSCORE_P4、P4_SUMMARY和四份日志及SHA256SUMS位于科学证据提交 `236fba1bef31618646dec2b9197e6b577612c144` 的 `evidence/T2_8_P4_INFO/`；Git blob已逐个核对原始SHA，未应用换行过滤。完整本机过程：`C:\Users\35884\Documents\Spacecraft\过程文件\T2_8_P4_INFO_20261102`。两个信号的原生汇总标志只按冻结p4.py报告，下层不另定门槛或结论。

资源调度：2026-11-02T16:29:08.440864+09:00按用户电脑空闲提速要求，将现有10个worker及行政父进程由低于正常提升到High（非Realtime），共享CPU0–11；不重启、不暂停、不改变科学配置或池内worker数。没有据此宣称实测速度提升。

## 第4节追加验证（用户授权，collab5358d0f）

第1–2节在8016b16完成并原样保全、退出后，才detached切换到`0dfdb8a8ad144a55635523612d245097ee541130`。按原两条命令运行40开局opp2和p4b，10进程，无训练；原P4_SUMMARY和原始info未改写。观测漂移≤1e-3的旧标签筛选由冻结p4b.py执行。下列为原生输出，不另作机制成效或论文创新判定。

墙钟：{"opp2_seconds": 600.1145343780518, "p4b_seconds": 30.018476009368896}

```json
{
  "decisions": 154,
  "phases": {
    "urgent_window": 52,
    "urgent_early": 34,
    "economy": 68
  },
  "decisive_pairs": {
    "urgent_window": 204,
    "urgent_early": 298,
    "economy": 367
  },
  "obs_mismatch": 67,
  "nonterminal_branches": 0,
  "alignment": {
    "label_used": 69,
    "label_dropped": 17,
    "opp2_used": 154,
    "opp2_dropped": 0,
    "label_drift_max": 0.04277169704437256,
    "label_drift_median": 2.2819403966423124e-06
  },
  "critic_by_arm_phase": {
    "mve/urgent_window": {
      "checkpoints": 6,
      "accuracy_mean": 0.7058823529411765,
      "accuracy_min": 0.6519607843137255
    },
    "mve/urgent_early": {
      "checkpoints": 6,
      "accuracy_mean": 0.7628635346756152,
      "accuracy_min": 0.7315436241610739
    },
    "mve/economy": {
      "checkpoints": 6,
      "accuracy_mean": 0.6684831970935514,
      "accuracy_min": 0.6321525885558583
    },
    "sac/urgent_window": {
      "checkpoints": 6,
      "accuracy_mean": 0.6037581699346405,
      "accuracy_min": 0.5245098039215687
    },
    "sac/urgent_early": {
      "checkpoints": 6,
      "accuracy_mean": 0.7393736017897092,
      "accuracy_min": 0.714765100671141
    },
    "sac/economy": {
      "checkpoints": 6,
      "accuracy_mean": 0.6012715712988193,
      "accuracy_min": 0.5231607629427792
    }
  },
  "signals": {
    "label": {
      "selection": {
        "critic_right": 1271,
        "broken": 0,
        "net": 17,
        "pairs_urgent_early": 1116,
        "critic_wrong": 613,
        "fixed": 17,
        "pairs_urgent_window": 768,
        "fix": 0.02773246329526917,
        "break": 0.0,
        "pass": false
      },
      "heldout": {
        "critic_wrong": 910,
        "fixed": 134,
        "net": 134,
        "pairs_urgent_early": 2460,
        "critic_right": 2786,
        "broken": 0,
        "pairs_urgent_window": 1236,
        "fix": 0.14725274725274726,
        "break": 0.0,
        "pass": false
      },
      "checkpoints_net_positive": "12/12",
      "carries_missing_information": false
    },
    "opp": {
      "selection": {
        "critic_wrong": 1978,
        "fixed": 589,
        "net": 402,
        "pairs_economy": 2940,
        "critic_right": 3158,
        "broken": 187,
        "pairs_urgent_early": 1116,
        "pairs_urgent_window": 1080,
        "fix": 0.29777553083923153,
        "break": 0.05921469284357188,
        "pass": false
      },
      "heldout": {
        "critic_right": 3927,
        "broken": 69,
        "net": 225,
        "pairs_economy": 1464,
        "critic_wrong": 1365,
        "fixed": 294,
        "pairs_urgent_early": 2460,
        "pairs_urgent_window": 1368,
        "fix": 0.2153846153846154,
        "break": 0.01757066462948816,
        "pass": false
      },
      "checkpoints_net_positive": "12/12",
      "carries_missing_information": false
    },
    "opp2": {
      "selection": {
        "critic_wrong": 1978,
        "fixed": 644,
        "net": 433,
        "pairs_economy": 2940,
        "critic_right": 3158,
        "broken": 211,
        "pairs_urgent_early": 1116,
        "pairs_urgent_window": 1080,
        "fix": 0.32558139534883723,
        "break": 0.06681443951868271,
        "pass": false
      },
      "heldout": {
        "critic_right": 3927,
        "broken": 142,
        "net": 197,
        "pairs_economy": 1464,
        "critic_wrong": 1365,
        "fixed": 339,
        "pairs_urgent_early": 2460,
        "pairs_urgent_window": 1368,
        "fix": 0.24835164835164836,
        "break": 0.03615991851285969,
        "pass": false
      },
      "checkpoints_net_positive": "12/12",
      "carries_missing_information": false
    }
  },
  "critic_per_checkpoint": {
    "e_mve_s0_1000": {
      "economy": {
        "right": 236,
        "wrong": 108,
        "tie": 23,
        "accuracy": 0.6430517711171662
      },
      "urgent_early": {
        "right": 223,
        "wrong": 64,
        "tie": 11,
        "accuracy": 0.7483221476510067
      },
      "urgent_window": {
        "right": 148,
        "wrong": 48,
        "tie": 8,
        "accuracy": 0.7254901960784313
      }
    },
    "e_mve_s0_1500": {
      "economy": {
        "right": 232,
        "wrong": 89,
        "tie": 46,
        "accuracy": 0.6321525885558583
      },
      "urgent_early": {
        "right": 232,
        "wrong": 61,
        "tie": 5,
        "accuracy": 0.7785234899328859
      },
      "urgent_window": {
        "right": 144,
        "wrong": 55,
        "tie": 5,
        "accuracy": 0.7058823529411765
      }
    },
    "e_mve_s0_2000": {
      "economy": {
        "right": 255,
        "wrong": 77,
        "tie": 35,
        "accuracy": 0.6948228882833788
      },
      "urgent_early": {
        "right": 242,
        "wrong": 53,
        "tie": 3,
        "accuracy": 0.8120805369127517
      },
      "urgent_window": {
        "right": 152,
        "wrong": 49,
        "tie": 3,
        "accuracy": 0.7450980392156863
      }
    },
    "e_mve_s0_500": {
      "economy": {
        "right": 193,
        "wrong": 145,
        "tie": 29,
        "accuracy": 0.5258855585831063
      },
      "urgent_early": {
        "right": 228,
        "wrong": 62,
        "tie": 8,
        "accuracy": 0.7651006711409396
      },
      "urgent_window": {
        "right": 115,
        "wrong": 80,
        "tie": 9,
        "accuracy": 0.5637254901960784
      }
    },
    "e_mve_s1_1000": {
      "economy": {
        "right": 247,
        "wrong": 94,
        "tie": 26,
        "accuracy": 0.6730245231607629
      },
      "urgent_early": {
        "right": 220,
        "wrong": 64,
        "tie": 14,
        "accuracy": 0.738255033557047
      },
      "urgent_window": {
        "right": 133,
        "wrong": 59,
        "tie": 12,
        "accuracy": 0.6519607843137255
      }
    },
    "e_mve_s1_1500": {
      "economy": {
        "right": 267,
        "wrong": 75,
        "tie": 25,
        "accuracy": 0.7275204359673024
      },
      "urgent_early": {
        "right": 218,
        "wrong": 68,
        "tie": 12,
        "accuracy": 0.7315436241610739
      },
      "urgent_window": {
        "right": 147,
        "wrong": 47,
        "tie": 10,
        "accuracy": 0.7205882352941176
      }
    },
    "e_mve_s1_2000": {
      "economy": {
        "right": 235,
        "wrong": 87,
        "tie": 45,
        "accuracy": 0.6403269754768393
      },
      "urgent_early": {
        "right": 229,
        "wrong": 65,
        "tie": 4,
        "accuracy": 0.7684563758389261
      },
      "urgent_window": {
        "right": 140,
        "wrong": 55,
        "tie": 9,
        "accuracy": 0.6862745098039216
      }
    },
    "e_mve_s1_500": {
      "economy": {
        "right": 197,
        "wrong": 107,
        "tie": 63,
        "accuracy": 0.5367847411444142
      },
      "urgent_early": {
        "right": 174,
        "wrong": 89,
        "tie": 35,
        "accuracy": 0.5838926174496645
      },
      "urgent_window": {
        "right": 91,
        "wrong": 93,
        "tie": 20,
        "accuracy": 0.44607843137254904
      }
    },
    "e_sac_s0_1000": {
      "economy": {
        "right": 192,
        "wrong": 133,
        "tie": 42,
        "accuracy": 0.5231607629427792
      },
      "urgent_early": {
        "right": 213,
        "wrong": 72,
        "tie": 13,
        "accuracy": 0.714765100671141
      },
      "urgent_window": {
        "right": 124,
        "wrong": 61,
        "tie": 19,
        "accuracy": 0.6078431372549019
      }
    },
    "e_sac_s0_1500": {
      "economy": {
        "right": 214,
        "wrong": 112,
        "tie": 41,
        "accuracy": 0.5831062670299727
      },
      "urgent_early": {
        "right": 222,
        "wrong": 63,
        "tie": 13,
        "accuracy": 0.7449664429530202
      },
      "urgent_window": {
        "right": 112,
        "wrong": 82,
        "tie": 10,
        "accuracy": 0.5490196078431373
      }
    },
    "e_sac_s0_2000": {
      "economy": {
        "right": 237,
        "wrong": 84,
        "tie": 46,
        "accuracy": 0.6457765667574932
      },
      "urgent_early": {
        "right": 231,
        "wrong": 60,
        "tie": 7,
        "accuracy": 0.7751677852348994
      },
      "urgent_window": {
        "right": 107,
        "wrong": 81,
        "tie": 16,
        "accuracy": 0.5245098039215687
      }
    },
    "e_sac_s0_500": {
      "economy": {
        "right": 173,
        "wrong": 176,
        "tie": 18,
        "accuracy": 0.4713896457765668
      },
      "urgent_early": {
        "right": 207,
        "wrong": 70,
        "tie": 21,
        "accuracy": 0.6946308724832215
      },
      "urgent_window": {
        "right": 118,
        "wrong": 77,
        "tie": 9,
        "accuracy": 0.5784313725490197
      }
    },
    "e_sac_s1_1000": {
      "economy": {
        "right": 240,
        "wrong": 103,
        "tie": 24,
        "accuracy": 0.6539509536784741
      },
      "urgent_early": {
        "right": 227,
        "wrong": 65,
        "tie": 6,
        "accuracy": 0.761744966442953
      },
      "urgent_window": {
        "right": 124,
        "wrong": 73,
        "tie": 7,
        "accuracy": 0.6078431372549019
      }
    },
    "e_sac_s1_1500": {
      "economy": {
        "right": 223,
        "wrong": 113,
        "tie": 31,
        "accuracy": 0.6076294277929155
      },
      "urgent_early": {
        "right": 215,
        "wrong": 75,
        "tie": 8,
        "accuracy": 0.7214765100671141
      },
      "urgent_window": {
        "right": 133,
        "wrong": 63,
        "tie": 8,
        "accuracy": 0.6519607843137255
      }
    },
    "e_sac_s1_2000": {
      "economy": {
        "right": 218,
        "wrong": 128,
        "tie": 21,
        "accuracy": 0.5940054495912807
      },
      "urgent_early": {
        "right": 214,
        "wrong": 80,
        "tie": 4,
        "accuracy": 0.7181208053691275
      },
      "urgent_window": {
        "right": 139,
        "wrong": 62,
        "tie": 3,
        "accuracy": 0.6813725490196079
      }
    },
    "e_sac_s1_500": {
      "economy": {
        "right": 147,
        "wrong": 164,
        "tie": 56,
        "accuracy": 0.40054495912806537
      },
      "urgent_early": {
        "right": 216,
        "wrong": 57,
        "tie": 25,
        "accuracy": 0.7248322147651006
      },
      "urgent_window": {
        "right": 145,
        "wrong": 44,
        "tie": 15,
        "accuracy": 0.7107843137254902
      }
    }
  },
  "critic_per_opening": {
    "2910000": {
      "mve": {
        "right": 2,
        "wrong": 16
      },
      "sac": {
        "right": 6,
        "wrong": 12
      }
    },
    "2910001": {
      "mve": {
        "right": 135,
        "wrong": 147
      },
      "sac": {
        "right": 121,
        "wrong": 161
      }
    },
    "2910002": {
      "mve": {
        "right": 7,
        "wrong": 5
      },
      "sac": {
        "right": 3,
        "wrong": 9
      }
    },
    "2910003": {
      "mve": {
        "right": 77,
        "wrong": 31
      },
      "sac": {
        "right": 72,
        "wrong": 36
      }
    },
    "2910004": {
      "mve": {
        "right": 158,
        "wrong": 106
      },
      "sac": {
        "right": 160,
        "wrong": 104
      }
    },
    "2910005": {
      "mve": {
        "right": 28,
        "wrong": 44
      },
      "sac": {
        "right": 36,
        "wrong": 36
      }
    },
    "2910007": {
      "mve": {
        "right": 187,
        "wrong": 119
      },
      "sac": {
        "right": 97,
        "wrong": 209
      }
    },
    "2910008": {
      "mve": {
        "right": 260,
        "wrong": 160
      },
      "sac": {
        "right": 220,
        "wrong": 200
      }
    },
    "2910009": {
      "mve": {
        "right": 93,
        "wrong": 3
      },
      "sac": {
        "right": 92,
        "wrong": 4
      }
    },
    "2910010": {
      "mve": {
        "right": 81,
        "wrong": 27
      },
      "sac": {
        "right": 71,
        "wrong": 37
      }
    },
    "2910011": {
      "mve": {
        "right": 143,
        "wrong": 115
      },
      "sac": {
        "right": 134,
        "wrong": 124
      }
    },
    "2910012": {
      "mve": {
        "right": 184,
        "wrong": 20
      },
      "sac": {
        "right": 161,
        "wrong": 43
      }
    },
    "2910013": {
      "mve": {
        "right": 25,
        "wrong": 41
      },
      "sac": {
        "right": 32,
        "wrong": 34
      }
    },
    "2910014": {
      "mve": {
        "right": 48,
        "wrong": 0
      },
      "sac": {
        "right": 46,
        "wrong": 2
      }
    },
    "2910015": {
      "mve": {
        "right": 54,
        "wrong": 12
      },
      "sac": {
        "right": 52,
        "wrong": 14
      }
    },
    "2910016": {
      "mve": {
        "right": 82,
        "wrong": 20
      },
      "sac": {
        "right": 65,
        "wrong": 37
      }
    },
    "2910017": {
      "mve": {
        "right": 6,
        "wrong": 0
      },
      "sac": {
        "right": 6,
        "wrong": 0
      }
    },
    "2910018": {
      "mve": {
        "right": 46,
        "wrong": 2
      },
      "sac": {
        "right": 33,
        "wrong": 15
      }
    },
    "2910019": {
      "mve": {
        "right": 71,
        "wrong": 13
      },
      "sac": {
        "right": 64,
        "wrong": 20
      }
    },
    "2910020": {
      "mve": {
        "right": 10,
        "wrong": 2
      },
      "sac": {
        "right": 5,
        "wrong": 7
      }
    },
    "2910024": {
      "mve": {
        "right": 100,
        "wrong": 26
      },
      "sac": {
        "right": 90,
        "wrong": 36
      }
    },
    "2910025": {
      "mve": {
        "right": 92,
        "wrong": 40
      },
      "sac": {
        "right": 88,
        "wrong": 44
      }
    },
    "2910026": {
      "mve": {
        "right": 25,
        "wrong": 41
      },
      "sac": {
        "right": 28,
        "wrong": 38
      }
    },
    "2910027": {
      "mve": {
        "right": 44,
        "wrong": 4
      },
      "sac": {
        "right": 42,
        "wrong": 6
      }
    },
    "2910028": {
      "mve": {
        "right": 94,
        "wrong": 26
      },
      "sac": {
        "right": 100,
        "wrong": 20
      }
    },
    "2910029": {
      "mve": {
        "right": 34,
        "wrong": 8
      },
      "sac": {
        "right": 36,
        "wrong": 6
      }
    },
    "2910030": {
      "mve": {
        "right": 156,
        "wrong": 30
      },
      "sac": {
        "right": 151,
        "wrong": 35
      }
    },
    "2910031": {
      "mve": {
        "right": 244,
        "wrong": 74
      },
      "sac": {
        "right": 238,
        "wrong": 80
      }
    },
    "2910032": {
      "mve": {
        "right": 401,
        "wrong": 133
      },
      "sac": {
        "right": 376,
        "wrong": 158
      }
    },
    "2910033": {
      "mve": {
        "right": 214,
        "wrong": 74
      },
      "sac": {
        "right": 194,
        "wrong": 94
      }
    },
    "2910034": {
      "mve": {
        "right": 126,
        "wrong": 24
      },
      "sac": {
        "right": 118,
        "wrong": 32
      }
    },
    "2910035": {
      "mve": {
        "right": 144,
        "wrong": 60
      },
      "sac": {
        "right": 139,
        "wrong": 65
      }
    },
    "2910037": {
      "mve": {
        "right": 19,
        "wrong": 5
      },
      "sac": {
        "right": 11,
        "wrong": 13
      }
    },
    "2910038": {
      "mve": {
        "right": 145,
        "wrong": 35
      },
      "sac": {
        "right": 150,
        "wrong": 30
      }
    },
    "2910039": {
      "mve": {
        "right": 165,
        "wrong": 51
      },
      "sac": {
        "right": 148,
        "wrong": 68
      }
    }
  },
  "net_per_checkpoint": {
    "label": {
      "e_mve_s0_1000": 8,
      "e_mve_s0_1500": 11,
      "e_mve_s0_2000": 7,
      "e_mve_s1_1000": 7,
      "e_mve_s1_1500": 8,
      "e_mve_s1_2000": 15,
      "e_sac_s0_1000": 9,
      "e_sac_s0_1500": 12,
      "e_sac_s0_2000": 19,
      "e_sac_s1_1000": 15,
      "e_sac_s1_1500": 21,
      "e_sac_s1_2000": 19
    },
    "opp": {
      "e_mve_s0_1000": 52,
      "e_mve_s0_1500": 47,
      "e_mve_s0_2000": 28,
      "e_mve_s1_1000": 53,
      "e_mve_s1_1500": 26,
      "e_mve_s1_2000": 39,
      "e_sac_s0_1000": 89,
      "e_sac_s0_1500": 63,
      "e_sac_s0_2000": 48,
      "e_sac_s1_1000": 52,
      "e_sac_s1_1500": 62,
      "e_sac_s1_2000": 68
    },
    "opp2": {
      "e_mve_s0_1000": 49,
      "e_mve_s0_1500": 46,
      "e_mve_s0_2000": 25,
      "e_mve_s1_1000": 51,
      "e_mve_s1_1500": 25,
      "e_mve_s1_2000": 37,
      "e_sac_s0_1000": 89,
      "e_sac_s0_1500": 66,
      "e_sac_s0_2000": 54,
      "e_sac_s1_1000": 55,
      "e_sac_s1_1500": 63,
      "e_sac_s1_2000": 70
    }
  }
}
```

合并科学证据提交：`27d59a5878b3181d1710c9c9ab1bf21d1b24bb0f`。原始P4证据提交：`236fba1bef31618646dec2b9197e6b577612c144`。
