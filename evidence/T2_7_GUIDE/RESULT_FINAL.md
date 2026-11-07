# T2.7 执行器引导2×2验证最终交付

生成2026-11-01T17:06:48.422771+09:00。科学冻结`2208a5176e88718c2c1db3e4a19c58853da49739`，执行书collab`8708f39`。原生主判定 **harmful**。四条从零训练、各4000步/2异步worker，guide_p0=0.5、guide_until=2000，环境0.5.0/学习器2.7；全部部署评估关闭引导，仅策略独立决策。

## 主判定与证据边界

```json
{
  "criteria": {
    "C0_complete": true,
    "M_mechanism": true,
    "E1_efficiency": false,
    "U_urgent": true,
    "E2_final": true,
    "S_safety": false
  },
  "values": {
    "paired_auc": {
      "0": -5.742509874095309,
      "1": 10.307117444964351
    },
    "paired_auc_urgent": {
      "0": -2.107338642351479,
      "1": 12.808658202250045
    },
    "urgent_capture_gain": {
      "0": 0,
      "1": 6
    },
    "paired_end": {
      "0": 0.005700593623631771,
      "1": 8.782993846688303
    },
    "unsafe": {
      "0": {
        "mve": 4,
        "sac": 0
      },
      "1": {
        "mve": 1,
        "sac": 0
      }
    },
    "mechanism": {
      "TRAIN_e_mve_s0": {
        "two_stage_share": 0.5571441291984794,
        "decisive_share": 0.1351503190503328,
        "c_mean": 0.13472728576526602,
        "grad_cos": 0.43350630070342394,
        "loss_share": 0.33436491440529065,
        "increment_abs_mean": 0.9614027807821377,
        "w_mean": 0.941841104218574,
        "w2_mean": 0.962529859602204,
        "pi_gap_sigma_median": 0.054854887688370164,
        "stage2_rows": 1031.390041493776,
        "stage2_entry_share": 0.24428371479155117,
        "stage1_cutoff_share": 0.0,
        "stage2_cutoff_share": 0.002553193985254608,
        "guided_share": 0.089
      },
      "TRAIN_e_mve_s1": {
        "two_stage_share": 0.5818707212523142,
        "decisive_share": 0.13171414122047265,
        "c_mean": 0.15100382821812175,
        "grad_cos": 0.3898650415424929,
        "loss_share": 0.4476164425620055,
        "increment_abs_mean": 1.4108384756131787,
        "w_mean": 0.9403145023895991,
        "w2_mean": 0.9585739794608469,
        "pi_gap_sigma_median": 0.05197367313244531,
        "stage2_rows": 1108.8174273858922,
        "stage2_entry_share": 0.22331401368765416,
        "stage1_cutoff_share": 0.0,
        "stage2_cutoff_share": 0.008658076108759849,
        "guided_share": 0.09375
      }
    }
  },
  "verdict": "harmful",
  "reported": {
    "late_s0": {
      "paired": {
        "3000": 3.2636210034547832,
        "3500": null,
        "4000": 0.005700593623631771
      },
      "capture": {
        "3000": [
          44,
          43
        ],
        "3500": null,
        "4000": [
          43,
          43
        ]
      },
      "unsafe": {
        "3000": [
          1,
          0
        ],
        "3500": null,
        "4000": [
          0,
          0
        ]
      },
      "return": {
        "mve": [
          41.56709431732819,
          null,
          38.9094808843579
        ],
        "sac": [
          38.303473313873404,
          null,
          38.90378029073427
        ]
      },
      "spread": {
        "mve": null,
        "sac": null
      }
    },
    "late_s1": {
      "paired": {
        "3000": 5.011754943742893,
        "3500": null,
        "4000": 8.782993846688303
      },
      "capture": {
        "3000": [
          42,
          41
        ],
        "3500": null,
        "4000": [
          42,
          41
        ]
      },
      "unsafe": {
        "3000": [
          1,
          0
        ],
        "3500": null,
        "4000": [
          0,
          0
        ]
      },
      "return": {
        "mve": [
          33.23019595625863,
          null,
          34.94189987473951
        ],
        "sac": [
          28.218441012515743,
          null,
          26.158906028051216
        ]
      },
      "spread": {
        "mve": null,
        "sac": null
      }
    },
    "urgent_bins_all_checkpoints": {
      "high": {
        "openings": 11,
        "capture_mve": 54,
        "capture_sac": 52,
        "paired_mean": 2.0048047949955397
      },
      "mid": {
        "openings": 7,
        "capture_mve": 43,
        "capture_sac": 43,
        "paired_mean": -0.804428822404375
      },
      "low": {
        "openings": 7,
        "capture_mve": 53,
        "capture_sac": 49,
        "paired_mean": 16.763520501515966
      }
    },
    "both_success": {
      "economy": {
        "checkpoints": 8,
        "delta_v": -0.046330633863404316,
        "time_s": -4.054118435911914
      },
      "urgent": {
        "checkpoints": 8,
        "delta_v": -0.022453352545002914,
        "time_s": -3.3161188510491924
      }
    }
  }
}
```

主判定仅EACC+引导对复用SAC；三个辅助配对仅报告，不另造通过条件。引导收益不能自动归给标签；标签独立增量以C_labels为准。两个种子、开发块结果不等于正式泛化证明。

## 四种配对曲线

|对照|种子|步数|均值|中位差|剔除最大两例|紧急|经济|捕获第一/第二|unsafe第一/第二|
|---|---|---:|---:|---:|---:|---:|---:|---|---|
|C_eval|0|1000|-18.974910|-0.210379|-29.270873|-11.480454|-27.121059|38/41|2/0|
|C_eval|0|2000|-7.264451|-0.049579|-12.549137|-2.448379|-12.499312|42/43|1/0|
|C_eval|0|3000|3.263621|0.259782|-1.152005|6.312860|-0.050769|44/43|1/0|
|C_eval|0|4000|0.005701|-0.194709|-0.708899|-0.813382|0.896008|43/43|0/0|
|C_eval|1|1000|15.419886|1.027637|6.591234|25.646928|4.303536|39/36|0/0|
|C_eval|1|2000|12.013835|0.938824|3.244151|15.529545|8.192412|41/39|0/0|
|C_eval|1|3000|5.011755|0.770378|-4.013348|7.504166|2.302612|42/41|1/0|
|C_eval|1|4000|8.782994|1.129071|-1.403652|2.553994|15.553646|42/41|0/0|
|C_guide_only|0|1000|-34.114579|-0.999806|-36.473124|-66.894040|1.515270|33/41|0/0|
|C_guide_only|0|2000|-40.297367|-0.261431|-42.572994|-77.498966|0.139154|34/43|1/0|
|C_guide_only|0|3000|-27.682312|0.146077|-29.239462|-54.650646|1.631095|37/43|1/0|
|C_guide_only|0|4000|-13.586528|-0.034315|-18.758631|-26.753019|0.724875|40/43|0/0|
|C_guide_only|1|1000|28.570198|1.157849|20.459279|50.043446|5.229712|42/36|0/0|
|C_guide_only|1|2000|15.748413|0.694901|7.392390|24.047060|6.728145|42/39|0/0|
|C_guide_only|1|3000|9.881528|0.734493|0.818441|17.766465|1.310944|43/41|0/0|
|C_guide_only|1|4000|16.616802|0.731992|6.518992|18.676915|14.377548|44/41|0/0|
|C_labels|0|1000|15.139669|-0.355212|6.341180|55.413586|-28.636329|38/33|2/0|
|C_labels|0|2000|33.032916|0.218726|25.163153|75.050588|-12.638466|42/34|1/1|
|C_labels|0|3000|30.945933|-0.163583|22.873205|60.963506|-1.681864|44/37|1/1|
|C_labels|0|4000|13.592229|0.036640|4.887304|25.939637|0.171133|43/40|0/0|
|C_labels|1|1000|-13.150313|-0.150063|-22.895935|-24.396518|-0.926176|39/42|0/0|
|C_labels|1|2000|-3.734578|0.045263|-12.916887|-8.517515|1.464267|41/42|0/0|
|C_labels|1|3000|-4.869773|-0.153166|-5.676247|-10.262298|0.991668|42/43|1/0|
|C_labels|1|4000|-7.833808|0.101060|-8.890781|-16.122922|1.176099|42/44|0/0|
|C_vs_eacc|0|1000|-5.194076|0.418188|-14.478931|14.196641|-26.270943|38/38|2/0|
|C_vs_eacc|0|2000|3.294527|0.326114|-5.688871|16.673185|-11.247493|42/41|1/1|
|C_vs_eacc|0|3000|7.276928|0.037185|-1.569647|15.597594|-1.767274|44/42|1/0|
|C_vs_eacc|0|4000|4.501182|-0.047207|-0.313873|8.004492|0.693236|43/42|0/0|
|C_vs_eacc|1|1000|-7.182426|0.310925|-8.450458|-16.032114|2.436801|39/41|0/0|
|C_vs_eacc|1|2000|-12.026065|0.082751|-13.542390|-25.349270|2.455679|41/44|0/0|
|C_vs_eacc|1|3000|-9.677780|-0.400971|-10.899018|-18.381970|-0.216704|42/44|1/0|
|C_vs_eacc|1|4000|-8.628990|0.116231|-9.329300|-16.244955|-0.350768|42/44|0/0|

C_eval=EACC+引导−SAC；C_guide_only=SAC+引导−SAC；C_labels=EACC+引导−SAC+引导；C_vs_eacc=EACC+引导−EACC。图/CURVES展示完整训练回报和四种配对曲线。原始逐开局结果、分档、双方成功燃料/时间随附，未选择性删样本。

## 三个问题的事实读数

完整方案跨种子可重复性按DECISION中E1/U/E2/S回答。引导作用与标签增量按如下四点均值各自报告：
- C_eval：{0: -5.742509874095309, 1: 10.307117444964351}
- C_guide_only：{0: -28.92019647189725, 1: 17.704235233439295}
- C_labels：{0: 23.177686597801937, 1: -7.3971177884749455}
- C_vs_eacc：{0: 2.469640072151599, 1: -9.378815177064949}

2000步撤除引导后的自主保持：3000/4000步部署仍没有引导，见完整四点曲线和原生reported.late_*；训练回合历史下列分段仅描述，跨边界完成的回合可能含前段guided动作，不把回合guided数误认为2000步后仍在引导。

```json
{
  "e_mve_s0": {
    "0-1000": {
      "episode_count": 211,
      "mean_true_return": 23.046393206833866,
      "mean_guided_per_completed_episode": 1.1943127962085307,
      "warning": "Episode ending in this interval may span earlier decisions; guided totals can carry over the2000 boundary.",
      "urgent": {
        "n": 114,
        "captures": 81,
        "capture_rate": 0.7105263157894737
      },
      "economy": {
        "n": 97,
        "captures": 95,
        "capture_rate": 0.979381443298969
      }
    },
    "1000-2000": {
      "episode_count": 247,
      "mean_true_return": 40.3587071849784,
      "mean_guided_per_completed_episode": 0.42105263157894735,
      "warning": "Episode ending in this interval may span earlier decisions; guided totals can carry over the2000 boundary.",
      "urgent": {
        "n": 124,
        "captures": 106,
        "capture_rate": 0.8548387096774194
      },
      "economy": {
        "n": 123,
        "captures": 122,
        "capture_rate": 0.991869918699187
      }
    },
    "2000-3000": {
      "episode_count": 246,
      "mean_true_return": 39.946049998643595,
      "mean_guided_per_completed_episode": 0,
      "warning": "Episode ending in this interval may span earlier decisions; guided totals can carry over the2000 boundary.",
      "urgent": {
        "n": 137,
        "captures": 113,
        "capture_rate": 0.8248175182481752
      },
      "economy": {
        "n": 109,
        "captures": 109,
        "capture_rate": 1
      }
    },
    "3000-4000": {
      "episode_count": 252,
      "mean_true_return": 40.47013425419967,
      "mean_guided_per_completed_episode": 0,
      "warning": "Episode ending in this interval may span earlier decisions; guided totals can carry over the2000 boundary.",
      "urgent": {
        "n": 121,
        "captures": 101,
        "capture_rate": 0.8347107438016529
      },
      "economy": {
        "n": 131,
        "captures": 129,
        "capture_rate": 0.9847328244274809
      }
    }
  },
  "e_mve_s1": {
    "0-1000": {
      "episode_count": 203,
      "mean_true_return": 15.202576736961372,
      "mean_guided_per_completed_episode": 1.3448275862068966,
      "warning": "Episode ending in this interval may span earlier decisions; guided totals can carry over the2000 boundary.",
      "urgent": {
        "n": 102,
        "captures": 63,
        "capture_rate": 0.6176470588235294
      },
      "economy": {
        "n": 101,
        "captures": 101,
        "capture_rate": 1
      }
    },
    "1000-2000": {
      "episode_count": 243,
      "mean_true_return": 30.605838348641733,
      "mean_guided_per_completed_episode": 0.41975308641975306,
      "warning": "Episode ending in this interval may span earlier decisions; guided totals can carry over the2000 boundary.",
      "urgent": {
        "n": 115,
        "captures": 85,
        "capture_rate": 0.7391304347826086
      },
      "economy": {
        "n": 128,
        "captures": 128,
        "capture_rate": 1
      }
    },
    "2000-3000": {
      "episode_count": 226,
      "mean_true_return": 29.606524937145526,
      "mean_guided_per_completed_episode": 0,
      "warning": "Episode ending in this interval may span earlier decisions; guided totals can carry over the2000 boundary.",
      "urgent": {
        "n": 110,
        "captures": 80,
        "capture_rate": 0.7272727272727273
      },
      "economy": {
        "n": 116,
        "captures": 116,
        "capture_rate": 1
      }
    },
    "3000-4000": {
      "episode_count": 260,
      "mean_true_return": 40.471006341466776,
      "mean_guided_per_completed_episode": 0,
      "warning": "Episode ending in this interval may span earlier decisions; guided totals can carry over the2000 boundary.",
      "urgent": {
        "n": 128,
        "captures": 106,
        "capture_rate": 0.828125
      },
      "economy": {
        "n": 132,
        "captures": 132,
        "capture_rate": 1
      }
    }
  },
  "e_sac_s0": {
    "0-1000": {
      "episode_count": 194,
      "mean_true_return": 8.53387418104901,
      "mean_guided_per_completed_episode": 1.2061855670103092,
      "warning": "Episode ending in this interval may span earlier decisions; guided totals can carry over the2000 boundary.",
      "urgent": {
        "n": 105,
        "captures": 63,
        "capture_rate": 0.6
      },
      "economy": {
        "n": 89,
        "captures": 88,
        "capture_rate": 0.9887640449438202
      }
    },
    "1000-2000": {
      "episode_count": 217,
      "mean_true_return": 20.32925234814356,
      "mean_guided_per_completed_episode": 0.3640552995391705,
      "warning": "Episode ending in this interval may span earlier decisions; guided totals can carry over the2000 boundary.",
      "urgent": {
        "n": 106,
        "captures": 70,
        "capture_rate": 0.660377358490566
      },
      "economy": {
        "n": 111,
        "captures": 111,
        "capture_rate": 1
      }
    },
    "2000-3000": {
      "episode_count": 209,
      "mean_true_return": 15.244703080051371,
      "mean_guided_per_completed_episode": 0,
      "warning": "Episode ending in this interval may span earlier decisions; guided totals can carry over the2000 boundary.",
      "urgent": {
        "n": 117,
        "captures": 73,
        "capture_rate": 0.6239316239316239
      },
      "economy": {
        "n": 92,
        "captures": 92,
        "capture_rate": 1
      }
    },
    "3000-4000": {
      "episode_count": 230,
      "mean_true_return": 32.21334368218248,
      "mean_guided_per_completed_episode": 0,
      "warning": "Episode ending in this interval may span earlier decisions; guided totals can carry over the2000 boundary.",
      "urgent": {
        "n": 121,
        "captures": 92,
        "capture_rate": 0.7603305785123967
      },
      "economy": {
        "n": 109,
        "captures": 109,
        "capture_rate": 1
      }
    }
  },
  "e_sac_s1": {
    "0-1000": {
      "episode_count": 219,
      "mean_true_return": 29.195103924586622,
      "mean_guided_per_completed_episode": 1.4337899543378996,
      "warning": "Episode ending in this interval may span earlier decisions; guided totals can carry over the2000 boundary.",
      "urgent": {
        "n": 111,
        "captures": 84,
        "capture_rate": 0.7567567567567568
      },
      "economy": {
        "n": 108,
        "captures": 107,
        "capture_rate": 0.9907407407407407
      }
    },
    "1000-2000": {
      "episode_count": 239,
      "mean_true_return": 34.75365586012422,
      "mean_guided_per_completed_episode": 0.3807531380753138,
      "warning": "Episode ending in this interval may span earlier decisions; guided totals can carry over the2000 boundary.",
      "urgent": {
        "n": 114,
        "captures": 90,
        "capture_rate": 0.7894736842105263
      },
      "economy": {
        "n": 125,
        "captures": 124,
        "capture_rate": 0.992
      }
    },
    "2000-3000": {
      "episode_count": 234,
      "mean_true_return": 38.690104786250984,
      "mean_guided_per_completed_episode": 0,
      "warning": "Episode ending in this interval may span earlier decisions; guided totals can carry over the2000 boundary.",
      "urgent": {
        "n": 111,
        "captures": 92,
        "capture_rate": 0.8288288288288288
      },
      "economy": {
        "n": 123,
        "captures": 122,
        "capture_rate": 0.991869918699187
      }
    },
    "3000-4000": {
      "episode_count": 252,
      "mean_true_return": 40.082939469385096,
      "mean_guided_per_completed_episode": 0,
      "warning": "Episode ending in this interval may span earlier decisions; guided totals can carry over the2000 boundary.",
      "urgent": {
        "n": 125,
        "captures": 104,
        "capture_rate": 0.832
      },
      "economy": {
        "n": 127,
        "captures": 127,
        "capture_rate": 1
      }
    }
  }
}
```

## 机制、压力分档、共同成功燃料/时间与墙钟

```json
{
  "training": {
    "TRAIN_e_mve_s0": {
      "finished": {
        "steps": 4000,
        "updates": 7602,
        "wall_s": 63751.569761514664,
        "replay": {
          "transitions": 4000,
          "with_candidates": 3399,
          "candidates": 26255,
          "entry_candidates": 5224,
          "reason": {
            "off": 54,
            "entry_calibration": 192,
            "stage": 2805,
            "entry": 710,
            "not_comparable": 239
          },
          "guided": 356,
          "cutoff_candidates": 0,
          "two_stage": 17652,
          "two_stage_entry": 4394,
          "two_stage_cutoff": 60
        }
      },
      "probes": 305,
      "probes_with_model_rows": 297,
      "first_model_step": 300,
      "w_mean": 0.9452881875262918,
      "abs_err_median": 0.33115386962890625,
      "sigma_last": 15.87801815348333,
      "model_td_last": 247.25230407714844,
      "q_pi_last_decile": 38.9169028600057,
      "loss_share": 0.33805526565466193,
      "grad_cos": 0.3563473682602247,
      "two_stage_share": 0.5699618534126755,
      "decisive_share": 0.13820144169107831,
      "c_mean": 0.14317362936157169,
      "c_low_share": 0.8677988847096761,
      "increment_abs_mean": 1.2663971488724655,
      "age_capped_share": 0.08353537510020614,
      "label_age_mean": 958.8436679005222,
      "entry_bias_now": -0.6782751309695945,
      "entry_share": 0.19999317664289315,
      "stage2_rows": 956.6801346801346,
      "w2_mean": 0.9655575884713067,
      "pi_gap_sigma_median": 0.04847125475207086,
      "stage2_entry_share": 0.24399903356426894,
      "stage1_cutoff_share": 0.0,
      "stage2_cutoff_share": 0.002375923018543495
    },
    "TRAIN_e_mve_s1": {
      "finished": {
        "steps": 4000,
        "updates": 7602,
        "wall_s": 63253.62050795555,
        "replay": {
          "transitions": 4000,
          "with_candidates": 3463,
          "candidates": 26786,
          "entry_candidates": 5420,
          "reason": {
            "off": 63,
            "entry_calibration": 190,
            "stage": 2817,
            "not_comparable": 195,
            "entry": 735
          },
          "guided": 375,
          "cutoff_candidates": 0,
          "two_stage": 18675,
          "two_stage_entry": 4511,
          "two_stage_cutoff": 208
        }
      },
      "probes": 305,
      "probes_with_model_rows": 297,
      "first_model_step": 300,
      "w_mean": 0.9410621909180072,
      "abs_err_median": 0.3751983642578125,
      "sigma_last": 14.800837896695496,
      "model_td_last": 158.9225616455078,
      "q_pi_last_decile": 24.70219841003418,
      "loss_share": 0.4526185128155953,
      "grad_cos": 0.3638414570937554,
      "two_stage_share": 0.6004935874337587,
      "decisive_share": 0.130676258719128,
      "c_mean": 0.16107465923835934,
      "c_low_share": 0.8515571179213347,
      "increment_abs_mean": 1.7794601048283305,
      "age_capped_share": 0.0913916808150878,
      "label_age_mean": 961.4106514670632,
      "entry_bias_now": -3.036903466118433,
      "entry_share": 0.19522062932401393,
      "stage2_rows": 1029.3468013468014,
      "w2_mean": 0.9581372489030112,
      "pi_gap_sigma_median": 0.046722735968176964,
      "stage2_entry_share": 0.21613829700834422,
      "stage1_cutoff_share": 0.0,
      "stage2_cutoff_share": 0.008179101379594518
    }
  },
  "pressure_and_both_success": {
    "C_eval": {
      "s0/step_1000": {
        "urgent_bins": {
          "high": {
            "n": 11,
            "mean": -26.91054619162288,
            "capture_mve": 6,
            "capture_sac": 7
          },
          "mid": {
            "n": 7,
            "mean": -30.382871273930498,
            "capture_mve": 5,
            "capture_sac": 6
          },
          "low": {
            "n": 7,
            "mean": 31.669252529540902,
            "capture_mve": 6,
            "capture_sac": 5
          }
        },
        "both_success": {
          "all": {
            "n": 36,
            "delta_v": 0.041983545375756735,
            "time_s": 1.3138888888888896
          },
          "economy": {
            "n": 21,
            "delta_v": 0.08676591560575761,
            "time_s": 4.6142857142857165
          },
          "urgent": {
            "n": 15,
            "delta_v": -0.02071177294624451,
            "time_s": -3.306666666666668
          }
        }
      },
      "s0/step_2000": {
        "urgent_bins": {
          "high": {
            "n": 11,
            "mean": 14.090567711978322,
            "capture_mve": 8,
            "capture_sac": 7
          },
          "mid": {
            "n": 7,
            "mean": -0.4828748148708034,
            "capture_mve": 6,
            "capture_sac": 6
          },
          "low": {
            "n": 7,
            "mean": -30.403655437542504,
            "capture_mve": 6,
            "capture_sac": 7
          }
        },
        "both_success": {
          "all": {
            "n": 41,
            "delta_v": -0.0005622783482237602,
            "time_s": 0.858536585365852
          },
          "economy": {
            "n": 22,
            "delta_v": 0.008822427499876,
            "time_s": 2.4545454545454537
          },
          "urgent": {
            "n": 19,
            "delta_v": -0.011428779856549798,
            "time_s": -0.9894736842105292
          }
        }
      },
      "s0/step_3000": {
        "urgent_bins": {
          "high": {
            "n": 11,
            "mean": 14.82846160382,
            "capture_mve": 8,
            "capture_sac": 7
          },
          "mid": {
            "n": 7,
            "mean": -0.72423931200871,
            "capture_mve": 6,
            "capture_sac": 6
          },
          "low": {
            "n": 7,
            "mean": -0.031700306082228735,
            "capture_mve": 7,
            "capture_sac": 7
          }
        },
        "both_success": {
          "all": {
            "n": 43,
            "delta_v": 0.004332943906027056,
            "time_s": -1.6162790697674405
          },
          "economy": {
            "n": 23,
            "delta_v": 0.0016605349451676986,
            "time_s": -0.7826086956521733
          },
          "urgent": {
            "n": 20,
            "delta_v": 0.007406214211015316,
            "time_s": -2.574999999999998
          }
        }
      },
      "s0/step_4000": {
        "urgent_bins": {
          "high": {
            "n": 11,
            "mean": -0.7208688127952133,
            "capture_mve": 7,
            "capture_sac": 7
          },
          "mid": {
            "n": 7,
            "mean": -0.8757232677870276,
            "capture_mve": 6,
            "capture_sac": 6
          },
          "low": {
            "n": 7,
            "mean": -0.8964197830806252,
            "capture_mve": 7,
            "capture_sac": 7
          }
        },
        "both_success": {
          "all": {
            "n": 43,
            "delta_v": 0.000967466876283664,
            "time_s": -0.7441860465116272
          },
          "economy": {
            "n": 23,
            "delta_v": -0.021911072598647326,
            "time_s": -0.9782608695652167
          },
          "urgent": {
            "n": 20,
            "delta_v": 0.0272777872724543,
            "time_s": -0.4749999999999993
          }
        }
      },
      "s1/step_1000": {
        "urgent_bins": {
          "high": {
            "n": 11,
            "mean": 17.80609713414176,
            "capture_mve": 5,
            "capture_sac": 4
          },
          "mid": {
            "n": 7,
            "mean": -0.06493720004851664,
            "capture_mve": 5,
            "capture_sac": 5
          },
          "low": {
            "n": 7,
            "mean": 63.680098842812775,
            "capture_mve": 6,
            "capture_sac": 4
          }
        },
        "both_success": {
          "all": {
            "n": 35,
            "delta_v": -0.05923891005705312,
            "time_s": -7.34857142857143
          },
          "economy": {
            "n": 23,
            "delta_v": -0.10265360756607265,
            "time_s": -9.869565217391305
          },
          "urgent": {
            "n": 12,
            "delta_v": 0.023972593501900924,
            "time_s": -2.516666666666673
          }
        }
      },
      "s1/step_2000": {
        "urgent_bins": {
          "high": {
            "n": 11,
            "mean": 15.79390512920427,
            "capture_mve": 6,
            "capture_sac": 5
          },
          "mid": {
            "n": 7,
            "mean": -1.1107700466561832,
            "capture_mve": 5,
            "capture_sac": 5
          },
          "low": {
            "n": 7,
            "mean": 31.7544365603136,
            "capture_mve": 7,
            "capture_sac": 6
          }
        },
        "both_success": {
          "all": {
            "n": 38,
            "delta_v": -0.12700020166897552,
            "time_s": -9.507894736842106
          },
          "economy": {
            "n": 23,
            "delta_v": -0.1976428984129483,
            "time_s": -14.334782608695651
          },
          "urgent": {
            "n": 15,
            "delta_v": -0.018681399994883933,
            "time_s": -2.1066666666666674
          }
        }
      },
      "s1/step_3000": {
        "urgent_bins": {
          "high": {
            "n": 11,
            "mean": 0.05963160963478293,
            "capture_mve": 7,
            "capture_sac": 7
          },
          "mid": {
            "n": 7,
            "mean": -4.666391494456685,
            "capture_mve": 5,
            "capture_sac": 5
          },
          "low": {
            "n": 7,
            "mean": 31.373278651428738,
            "capture_mve": 7,
            "capture_sac": 6
          }
        },
        "both_success": {
          "all": {
            "n": 40,
            "delta_v": -0.035676114599537456,
            "time_s": -4.902499999999999
          },
          "economy": {
            "n": 23,
            "delta_v": -0.05406747771181718,
            "time_s": -6.995652173913043
          },
          "urgent": {
            "n": 17,
            "delta_v": -0.010793682153511966,
            "time_s": -2.070588235294116
          }
        }
      },
      "s1/step_4000": {
        "urgent_bins": {
          "high": {
            "n": 11,
            "mean": -18.908809824396716,
            "capture_mve": 7,
            "capture_sac": 8
          },
          "mid": {
            "n": 7,
            "mean": 31.872376830523425,
            "capture_mve": 5,
            "capture_sac": 4
          },
          "low": {
            "n": 7,
            "mean": 6.962872954737053,
            "capture_mve": 7,
            "capture_sac": 7
          }
        },
        "both_success": {
          "all": {
            "n": 40,
            "delta_v": -0.1298908921450943,
            "time_s": -9.217500000000003
          },
          "economy": {
            "n": 22,
            "delta_v": -0.09161889266855038,
            "time_s": -6.5409090909090954
          },
          "urgent": {
            "n": 18,
            "delta_v": -0.17666778039420367,
            "time_s": -12.488888888888887
          }
        }
      }
    },
    "C_guide_only": {
      "s0/step_1000": {
        "urgent_bins": {
          "high": {
            "n": 11,
            "mean": -56.483848363129304,
            "capture_mve": 4,
            "capture_sac": 7
          },
          "mid": {
            "n": 7,
            "mean": -89.3023636492416,
            "capture_mve": 3,
            "capture_sac": 6
          },
          "low": {
            "n": 7,
            "mean": -60.84458780570428,
            "capture_mve": 3,
            "capture_sac": 5
          }
        },
        "both_success": {
          "all": {
            "n": 33,
            "delta_v": -0.013525246788002515,
            "time_s": -4.015151515151516
          },
          "economy": {
            "n": 23,
            "delta_v": -0.03510348939785431,
            "time_s": -5.5565217391304325
          },
          "urgent": {
            "n": 10,
            "delta_v": 0.0361047112146566,
            "time_s": -0.47000000000000314
          }
        }
      },
      "s0/step_2000": {
        "urgent_bins": {
          "high": {
            "n": 11,
            "mean": -55.58919029871609,
            "capture_mve": 4,
            "capture_sac": 7
          },
          "mid": {
            "n": 7,
            "mean": -67.81532013391889,
            "capture_mve": 4,
            "capture_sac": 6
          },
          "low": {
            "n": 7,
            "mean": -121.6122603070634,
            "capture_mve": 3,
            "capture_sac": 7
          }
        },
        "both_success": {
          "all": {
            "n": 34,
            "delta_v": -0.0058181210962684115,
            "time_s": 0.14411764705882368
          },
          "economy": {
            "n": 23,
            "delta_v": -0.0037853704337937597,
            "time_s": 0.6130434782608681
          },
          "urgent": {
            "n": 11,
            "delta_v": -0.010068417935988137,
            "time_s": -0.8363636363636328
          }
        }
      },
      "s0/step_3000": {
        "urgent_bins": {
          "high": {
            "n": 11,
            "mean": -42.34552745296181,
            "capture_mve": 5,
            "capture_sac": 7
          },
          "mid": {
            "n": 7,
            "mean": -33.61069745964424,
            "capture_mve": 5,
            "capture_sac": 6
          },
          "low": {
            "n": 7,
            "mean": -95.02720843871076,
            "capture_mve": 4,
            "capture_sac": 7
          }
        },
        "both_success": {
          "all": {
            "n": 37,
            "delta_v": 0.035698472079146566,
            "time_s": 3.164864864864866
          },
          "economy": {
            "n": 23,
            "delta_v": -0.0399121532956779,
            "time_s": -1.7304347826086943
          },
          "urgent": {
            "n": 14,
            "delta_v": 0.15991592805207247,
            "time_s": 11.207142857142857
          }
        }
      },
      "s0/step_4000": {
        "urgent_bins": {
          "high": {
            "n": 11,
            "mean": -38.355061880508565,
            "capture_mve": 5,
            "capture_sac": 7
          },
          "mid": {
            "n": 7,
            "mean": -31.446585215366408,
            "capture_mve": 5,
            "capture_sac": 6
          },
          "low": {
            "n": 7,
            "mean": -3.8276718679765005,
            "capture_mve": 7,
            "capture_sac": 7
          }
        },
        "both_success": {
          "all": {
            "n": 39,
            "delta_v": 0.02782607306356459,
            "time_s": 2.2820512820512815
          },
          "economy": {
            "n": 23,
            "delta_v": -0.017445779793960607,
            "time_s": -1.3521739130434798
          },
          "urgent": {
            "n": 16,
            "delta_v": 0.09290436154625707,
            "time_s": 7.506250000000001
          }
        }
      },
      "s1/step_1000": {
        "urgent_bins": {
          "high": {
            "n": 11,
            "mean": 55.22893241195123,
            "capture_mve": 7,
            "capture_sac": 4
          },
          "mid": {
            "n": 7,
            "mean": 30.775117644087043,
            "capture_mve": 6,
            "capture_sac": 5
          },
          "low": {
            "n": 7,
            "mean": 61.1631538693261,
            "capture_mve": 6,
            "capture_sac": 4
          }
        },
        "both_success": {
          "all": {
            "n": 36,
            "delta_v": -0.0866602518437994,
            "time_s": -5.572222222222225
          },
          "economy": {
            "n": 23,
            "delta_v": -0.12666235423114744,
            "time_s": -8.160869565217391
          },
          "urgent": {
            "n": 13,
            "delta_v": -0.01588730146618361,
            "time_s": -0.9923076923076949
          }
        }
      },
      "s1/step_2000": {
        "urgent_bins": {
          "high": {
            "n": 11,
            "mean": 34.70355299206294,
            "capture_mve": 7,
            "capture_sac": 5
          },
          "mid": {
            "n": 7,
            "mean": 29.545388123155615,
            "capture_mve": 6,
            "capture_sac": 5
          },
          "low": {
            "n": 7,
            "mean": 1.8028147154610212,
            "capture_mve": 6,
            "capture_sac": 6
          }
        },
        "both_success": {
          "all": {
            "n": 39,
            "delta_v": -0.11712559090793671,
            "time_s": -7.282051282051279
          },
          "economy": {
            "n": 23,
            "delta_v": -0.1632036193087655,
            "time_s": -9.999999999999996
          },
          "urgent": {
            "n": 16,
            "delta_v": -0.050888425081745264,
            "time_s": -3.375
          }
        }
      },
      "s1/step_3000": {
        "urgent_bins": {
          "high": {
            "n": 11,
            "mean": -0.07696070444672577,
            "capture_mve": 7,
            "capture_sac": 7
          },
          "mid": {
            "n": 7,
            "mean": 32.12590068987489,
            "capture_mve": 6,
            "capture_sac": 5
          },
          "low": {
            "n": 7,
            "mean": 31.446696914353424,
            "capture_mve": 7,
            "capture_sac": 6
          }
        },
        "both_success": {
          "all": {
            "n": 40,
            "delta_v": -0.023052282855449368,
            "time_s": 2.3725000000000023
          },
          "economy": {
            "n": 23,
            "delta_v": -0.03533447638243268,
            "time_s": 5.1217391304347855
          },
          "urgent": {
            "n": 17,
            "delta_v": -0.006435197495413129,
            "time_s": -1.3470588235294105
          }
        }
      },
      "s1/step_4000": {
        "urgent_bins": {
          "high": {
            "n": 11,
            "mean": -1.1797828788170959,
            "capture_mve": 8,
            "capture_sac": 8
          },
          "mid": {
            "n": 7,
            "mean": 62.08202538190597,
            "capture_mve": 6,
            "capture_sac": 4
          },
          "low": {
            "n": 7,
            "mean": 6.475188028576233,
            "capture_mve": 7,
            "capture_sac": 7
          }
        },
        "both_success": {
          "all": {
            "n": 41,
            "delta_v": -0.08390509256103976,
            "time_s": -7.797560975609757
          },
          "economy": {
            "n": 22,
            "delta_v": -0.05835900089253596,
            "time_s": -4.777272727272731
          },
          "urgent": {
            "n": 19,
            "delta_v": -0.11348477765088619,
            "time_s": -11.294736842105266
          }
        }
      }
    },
    "C_labels": {
      "s0/step_1000": {
        "urgent_bins": {
          "high": {
            "n": 11,
            "mean": 29.57330217150643,
            "capture_mve": 6,
            "capture_sac": 4
          },
          "mid": {
            "n": 7,
            "mean": 58.91949237531111,
            "capture_mve": 5,
            "capture_sac": 3
          },
          "low": {
            "n": 7,
            "mean": 92.51384033524518,
            "capture_mve": 6,
            "capture_sac": 3
          }
        },
        "both_success": {
          "all": {
            "n": 31,
            "delta_v": 0.08506337846499784,
            "time_s": 6.829032258064516
          },
          "economy": {
            "n": 21,
            "delta_v": 0.13522372556145903,
            "time_s": 10.547619047619047
          },
          "urgent": {
            "n": 10,
            "delta_v": -0.02027335043757069,
            "time_s": -0.9799999999999969
          }
        }
      },
      "s0/step_2000": {
        "urgent_bins": {
          "high": {
            "n": 11,
            "mean": 69.67975801069443,
            "capture_mve": 8,
            "capture_sac": 4
          },
          "mid": {
            "n": 7,
            "mean": 67.33244531904809,
            "capture_mve": 6,
            "capture_sac": 4
          },
          "low": {
            "n": 7,
            "mean": 91.20860486952087,
            "capture_mve": 6,
            "capture_sac": 3
          }
        },
        "both_success": {
          "all": {
            "n": 33,
            "delta_v": -0.00011953410536365073,
            "time_s": 1.5121212121212102
          },
          "economy": {
            "n": 22,
            "delta_v": 0.010532535976541035,
            "time_s": 1.9272727272727281
          },
          "urgent": {
            "n": 11,
            "delta_v": -0.02142367426917302,
            "time_s": 0.6818181818181741
          }
        }
      },
      "s0/step_3000": {
        "urgent_bins": {
          "high": {
            "n": 11,
            "mean": 57.173989056781814,
            "capture_mve": 8,
            "capture_sac": 5
          },
          "mid": {
            "n": 7,
            "mean": 32.88645814763553,
            "capture_mve": 6,
            "capture_sac": 5
          },
          "low": {
            "n": 7,
            "mean": 94.99550813262852,
            "capture_mve": 7,
            "capture_sac": 4
          }
        },
        "both_success": {
          "all": {
            "n": 37,
            "delta_v": -0.030949190275379592,
            "time_s": -4.156756756756757
          },
          "economy": {
            "n": 23,
            "delta_v": 0.0415726882408456,
            "time_s": 0.947826086956521
          },
          "urgent": {
            "n": 14,
            "delta_v": -0.1500922764091781,
            "time_s": -12.542857142857143
          }
        }
      },
      "s0/step_4000": {
        "urgent_bins": {
          "high": {
            "n": 11,
            "mean": 37.634193067713355,
            "capture_mve": 7,
            "capture_sac": 5
          },
          "mid": {
            "n": 7,
            "mean": 30.57086194757938,
            "capture_mve": 6,
            "capture_sac": 5
          },
          "low": {
            "n": 7,
            "mean": 2.931252084895875,
            "capture_mve": 7,
            "capture_sac": 7
          }
        },
        "both_success": {
          "all": {
            "n": 39,
            "delta_v": -0.024368714992844105,
            "time_s": -2.746153846153845
          },
          "economy": {
            "n": 23,
            "delta_v": -0.004465292804686718,
            "time_s": 0.3739130434782631
          },
          "urgent": {
            "n": 16,
            "delta_v": -0.052979884388320346,
            "time_s": -7.23125
          }
        }
      },
      "s1/step_1000": {
        "urgent_bins": {
          "high": {
            "n": 11,
            "mean": -37.422835277809476,
            "capture_mve": 5,
            "capture_sac": 7
          },
          "mid": {
            "n": 7,
            "mean": -30.840054844135558,
            "capture_mve": 5,
            "capture_sac": 6
          },
          "low": {
            "n": 7,
            "mean": 2.516944973486677,
            "capture_mve": 6,
            "capture_sac": 6
          }
        },
        "both_success": {
          "all": {
            "n": 37,
            "delta_v": 0.025124915225670338,
            "time_s": -1.456756756756758
          },
          "economy": {
            "n": 23,
            "delta_v": 0.02400874666507481,
            "time_s": -1.7086956521739123
          },
          "urgent": {
            "n": 14,
            "delta_v": 0.026958620718077284,
            "time_s": -1.0428571428571476
          }
        }
      },
      "s1/step_2000": {
        "urgent_bins": {
          "high": {
            "n": 11,
            "mean": -18.909647862858662,
            "capture_mve": 6,
            "capture_sac": 7
          },
          "mid": {
            "n": 7,
            "mean": -30.6561581698118,
            "capture_mve": 5,
            "capture_sac": 6
          },
          "low": {
            "n": 7,
            "mean": 29.95162184485258,
            "capture_mve": 7,
            "capture_sac": 6
          }
        },
        "both_success": {
          "all": {
            "n": 39,
            "delta_v": -0.002623059818962956,
            "time_s": -1.9076923076923094
          },
          "economy": {
            "n": 23,
            "delta_v": -0.034439279104182846,
            "time_s": -4.3347826086956545
          },
          "urgent": {
            "n": 16,
            "delta_v": 0.043112755403540634,
            "time_s": 1.581249999999999
          }
        }
      },
      "s1/step_3000": {
        "urgent_bins": {
          "high": {
            "n": 11,
            "mean": 0.1365923140815087,
            "capture_mve": 7,
            "capture_sac": 7
          },
          "mid": {
            "n": 7,
            "mean": -36.79229218433157,
            "capture_mve": 5,
            "capture_sac": 6
          },
          "low": {
            "n": 7,
            "mean": -0.0734182629246882,
            "capture_mve": 7,
            "capture_sac": 7
          }
        },
        "both_success": {
          "all": {
            "n": 42,
            "delta_v": -0.011830898436363424,
            "time_s": -6.8880952380952385
          },
          "economy": {
            "n": 23,
            "delta_v": -0.018733001329384494,
            "time_s": -12.117391304347825
          },
          "urgent": {
            "n": 19,
            "delta_v": -0.003475721250074758,
            "time_s": -0.557894736842105
          }
        }
      },
      "s1/step_4000": {
        "urgent_bins": {
          "high": {
            "n": 11,
            "mean": -17.72902694557962,
            "capture_mve": 7,
            "capture_sac": 8
          },
          "mid": {
            "n": 7,
            "mean": -30.20964855138255,
            "capture_mve": 5,
            "capture_sac": 6
          },
          "low": {
            "n": 7,
            "mean": 0.4876849261608202,
            "capture_mve": 7,
            "capture_sac": 7
          }
        },
        "both_success": {
          "all": {
            "n": 42,
            "delta_v": -0.04088766993445025,
            "time_s": -1.2714285714285716
          },
          "economy": {
            "n": 23,
            "delta_v": -0.028502468387204778,
            "time_s": -1.800000000000002
          },
          "urgent": {
            "n": 19,
            "delta_v": -0.05588028233374739,
            "time_s": -0.6315789473684188
          }
        }
      }
    },
    "C_vs_eacc": {
      "s0/step_1000": {
        "urgent_bins": {
          "high": {
            "n": 11,
            "mean": 30.653669729458354,
            "capture_mve": 6,
            "capture_sac": 4
          },
          "mid": {
            "n": 7,
            "mean": 1.3466386420858518,
            "capture_mve": 5,
            "capture_sac": 5
          },
          "low": {
            "n": 7,
            "mean": 1.185596748416062,
            "capture_mve": 6,
            "capture_sac": 6
          }
        },
        "both_success": {
          "all": {
            "n": 36,
            "delta_v": 0.005583373557954047,
            "time_s": 2.4694444444444446
          },
          "economy": {
            "n": 21,
            "delta_v": 0.06398055158424976,
            "time_s": 6.642857142857143
          },
          "urgent": {
            "n": 15,
            "delta_v": -0.07617267567885994,
            "time_s": -3.373333333333334
          }
        }
      },
      "s0/step_2000": {
        "urgent_bins": {
          "high": {
            "n": 11,
            "mean": 32.81566307377355,
            "capture_mve": 8,
            "capture_sac": 6
          },
          "mid": {
            "n": 7,
            "mean": 38.179284486110724,
            "capture_mve": 6,
            "capture_sac": 5
          },
          "low": {
            "n": 7,
            "mean": -30.199667217927576,
            "capture_mve": 6,
            "capture_sac": 7
          }
        },
        "both_success": {
          "all": {
            "n": 39,
            "delta_v": -0.0359013694941306,
            "time_s": -2.40769230769231
          },
          "economy": {
            "n": 22,
            "delta_v": -0.02248080646692535,
            "time_s": -2.6227272727272735
          },
          "urgent": {
            "n": 17,
            "delta_v": -0.0532691569411021,
            "time_s": -2.129411764705886
          }
        }
      },
      "s0/step_3000": {
        "urgent_bins": {
          "high": {
            "n": 11,
            "mean": 15.30368944307964,
            "capture_mve": 8,
            "capture_sac": 7
          },
          "mid": {
            "n": 7,
            "mean": 30.772014107427594,
            "capture_mve": 6,
            "capture_sac": 5
          },
          "low": {
            "n": 7,
            "mean": 0.8850245652488168,
            "capture_mve": 7,
            "capture_sac": 7
          }
        },
        "both_success": {
          "all": {
            "n": 42,
            "delta_v": 0.013518935845578464,
            "time_s": -1.5785714285714272
          },
          "economy": {
            "n": 23,
            "delta_v": 0.043953596586674484,
            "time_s": 0.45652173913043664
          },
          "urgent": {
            "n": 19,
            "delta_v": -0.023323021893643027,
            "time_s": -4.042105263157894
          }
        }
      },
      "s0/step_4000": {
        "urgent_bins": {
          "high": {
            "n": 11,
            "mean": 18.44227012426727,
            "capture_mve": 7,
            "capture_sac": 6
          },
          "mid": {
            "n": 7,
            "mean": -0.48698017235981744,
            "capture_mve": 6,
            "capture_sac": 6
          },
          "low": {
            "n": 7,
            "mean": 0.09374059979640086,
            "capture_mve": 7,
            "capture_sac": 7
          }
        },
        "both_success": {
          "all": {
            "n": 42,
            "delta_v": -0.01417148871560688,
            "time_s": -2.2309523809523824
          },
          "economy": {
            "n": 23,
            "delta_v": -0.016376559289105447,
            "time_s": -1.908695652173914
          },
          "urgent": {
            "n": 19,
            "delta_v": -0.011502192758213877,
            "time_s": -2.6210526315789493
          }
        }
      },
      "s1/step_1000": {
        "urgent_bins": {
          "high": {
            "n": 11,
            "mean": 1.4748355164034075,
            "capture_mve": 5,
            "capture_sac": 5
          },
          "mid": {
            "n": 7,
            "mean": -30.32991351057325,
            "capture_mve": 5,
            "capture_sac": 6
          },
          "low": {
            "n": 7,
            "mean": -29.245235233478443,
            "capture_mve": 6,
            "capture_sac": 7
          }
        },
        "both_success": {
          "all": {
            "n": 39,
            "delta_v": -0.021623427317299788,
            "time_s": -3.7794871794871803
          },
          "economy": {
            "n": 23,
            "delta_v": -0.05782872043042714,
            "time_s": -6.182608695652173
          },
          "urgent": {
            "n": 16,
            "delta_v": 0.0304216815328208,
            "time_s": -0.32500000000000373
          }
        }
      },
      "s1/step_2000": {
        "urgent_bins": {
          "high": {
            "n": 11,
            "mean": -37.876477836860836,
            "capture_mve": 6,
            "capture_sac": 8
          },
          "mid": {
            "n": 7,
            "mean": -31.097046698690317,
            "capture_mve": 5,
            "capture_sac": 6
          },
          "low": {
            "n": 7,
            "mean": 0.08412033326970507,
            "capture_mve": 7,
            "capture_sac": 7
          }
        },
        "both_success": {
          "all": {
            "n": 41,
            "delta_v": -0.031285284674004044,
            "time_s": -2.9560975609756093
          },
          "economy": {
            "n": 23,
            "delta_v": -0.05905284970853479,
            "time_s": -4.678260869565217
          },
          "urgent": {
            "n": 18,
            "delta_v": 0.004195492870118562,
            "time_s": -0.7555555555555544
          }
        }
      },
      "s1/step_3000": {
        "urgent_bins": {
          "high": {
            "n": 11,
            "mean": -17.891203579294526,
            "capture_mve": 7,
            "capture_sac": 8
          },
          "mid": {
            "n": 7,
            "mean": -37.37739426279205,
            "capture_mve": 5,
            "capture_sac": 6
          },
          "low": {
            "n": 7,
            "mean": -0.1577488453345483,
            "capture_mve": 7,
            "capture_sac": 7
          }
        },
        "both_success": {
          "all": {
            "n": 42,
            "delta_v": -0.005038943058117281,
            "time_s": -0.6309523809523799
          },
          "economy": {
            "n": 23,
            "delta_v": 0.006028470504531842,
            "time_s": -1.221739130434781
          },
          "urgent": {
            "n": 19,
            "delta_v": -0.018436338423429375,
            "time_s": 0.08421052631578992
          }
        }
      },
      "s1/step_4000": {
        "urgent_bins": {
          "high": {
            "n": 11,
            "mean": -17.95237046487979,
            "capture_mve": 7,
            "capture_sac": 8
          },
          "mid": {
            "n": 7,
            "mean": -30.626446321620385,
            "capture_mve": 5,
            "capture_sac": 6
          },
          "low": {
            "n": 7,
            "mean": 0.8196185153961741,
            "capture_mve": 7,
            "capture_sac": 7
          }
        },
        "both_success": {
          "all": {
            "n": 42,
            "delta_v": -0.01657431458757081,
            "time_s": 1.666666666666665
          },
          "economy": {
            "n": 23,
            "delta_v": 0.007047460306995521,
            "time_s": 3.443478260869561
          },
          "urgent": {
            "n": 19,
            "delta_v": -0.04516909472309847,
            "time_s": -0.48421052631578815
          }
        }
      }
    }
  }
}
```

## 对照与交付位置

SAC复用T2.6W，未重训/重评；无引导EACC复用T2.6W1，仅辅助参照。旧轮数值在四种配对读出中重述，原始复用JSON作为本轮判断必需证据随附，其余历史只固定引用及SHA。52项测试、真实guidance-on两臂日志冒烟、TensorBoard实际解析及增长验收、操作系统祖先链、完整训练meta/回合/模型探针均附。实际PyTorch线程数在每条训练live/TORCH_THREADS.json，启动时Normal优先级；11月1日11:50按用户要求将现有训练树提升High（非Realtime），不据此声称已证明加速。四条共享CPU0–11，OMP/MKL/OPENBLAS=1。

完整过程目录`C:\Users\35884\Documents\Spacecraft\过程文件\T2_7_GUIDE_20261031`；科学证据`evidence/T2_7_GUIDE/`。模型ZIP及原始TensorBoard事件留本机，仅记SHA。报告、原始结果与SHA256SUMS足够上层独立判定。下一轮实验或机制修改未授权。
