# T2.6W1 截止终止修复及动作排序核验

完成时间：2026-10-31T10:44:49.098199+09:00。原生判定：**no_gain**。执行书 collab `185765b`；训练/评估冻结 `e2d4c14bc4d3bbe385c44e65f3c3dcdbd41372d5`，机制审计 `91b346b664fe59e1f9c00af38f45e65735b27839`。主张仍为执行反馈独立改善SAC长期决策；实际结论以如下判定与证据为限。

## 版本、配置与覆盖

环境 t2-stage-0.4.3、学习器 t2-learner-2.6.1；EACC两种子各从零4000步、2个异步worker。紧急U(140,200)s、经济U(280,340)s各0.5。只改已冻结的截止/预算终止逻辑；未加预测深度、直接下降、入口偏差修正或候选差训练。每种子4检查点×48开局；新EACC384例、原样复用SAC384例，总768例，294000–294047，原始JSON随附。SAC无重训、无重评。系统参照为T2.6W A_ref 336例（固定位置/哈希见PROVENANCE）。

## 原生判定（不改门槛）

```json
{
  "criteria": {
    "C0_complete": true,
    "M_mechanism": true,
    "E1_efficiency": false,
    "U_urgent": false,
    "E2_final": true,
    "S_safety": true
  },
  "values": {
    "paired_auc": {
      "0": -8.212149946246909,
      "1": 19.6859326220293
    },
    "paired_auc_urgent": {
      "0": -15.725316426231014,
      "1": 31.810735224027155
    },
    "urgent_capture_gain": {
      "0": -7,
      "1": 15
    },
    "paired_end": {
      "0": -4.495481258563308,
      "1": 17.41198412251502
    },
    "unsafe": {
      "0": {
        "mve": 1,
        "sac": 0
      },
      "1": {
        "mve": 0,
        "sac": 0
      }
    },
    "mechanism": {
      "TRAIN_e_mve_s0": {
        "two_stage_share": 0.5656753733032249,
        "decisive_share": 0.13033673779351088,
        "c_mean": 0.16189146867307885,
        "grad_cos": 0.4169269673159865,
        "loss_share": 0.3015652249013988,
        "increment_abs_mean": 1.3464559192974044,
        "w_mean": 0.9540403136573886,
        "w2_mean": 0.9746724594678127,
        "pi_gap_sigma_median": 0.03794689552232438,
        "stage2_rows": 1007.9709543568465,
        "stage2_entry_share": 0.22727711678045914,
        "stage1_cutoff_share": 0.0,
        "stage2_cutoff_share": 0.010629325163168099
      },
      "TRAIN_e_mve_s1": {
        "two_stage_share": 0.571378009830872,
        "decisive_share": 0.13616070070949332,
        "c_mean": 0.18720038797845484,
        "grad_cos": 0.3306581377143376,
        "loss_share": 0.36410198634849117,
        "increment_abs_mean": 1.6159624140786928,
        "w_mean": 0.9444615507026926,
        "w2_mean": 0.973282090855832,
        "pi_gap_sigma_median": 0.04135997982935292,
        "stage2_rows": 988.9502074688796,
        "stage2_entry_share": 0.2352625396860091,
        "stage1_cutoff_share": 0.0,
        "stage2_cutoff_share": 0.0032648746655090746
      }
    }
  },
  "verdict": "no_gain",
  "reported": {
    "late_s0": {
      "paired": {
        "3000": -4.013307032438349,
        "3500": null,
        "4000": -4.495481258563308
      },
      "capture": {
        "3000": [
          42,
          43
        ],
        "3500": null,
        "4000": [
          42,
          43
        ]
      },
      "unsafe": {
        "3000": [
          0,
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
          34.290166281435056,
          null,
          34.40829903217095
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
        "3000": 14.689534818603207,
        "3500": null,
        "4000": 17.41198412251502
      },
      "capture": {
        "3000": [
          44,
          41
        ],
        "3500": null,
        "4000": [
          44,
          41
        ]
      },
      "unsafe": {
        "3000": [
          0,
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
          42.907975831118954,
          null,
          43.570890150566235
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
        "capture_mve": 52,
        "capture_sac": 52,
        "paired_mean": -1.116454705747845
      },
      "mid": {
        "openings": 7,
        "capture_mve": 45,
        "capture_sac": 43,
        "paired_mean": 6.64805164389708
      },
      "low": {
        "openings": 7,
        "capture_mve": 55,
        "capture_sac": 49,
        "paired_mean": 23.83033931834264
      }
    },
    "both_success": {
      "economy": {
        "checkpoints": 8,
        "delta_v": -0.04203414205464803,
        "time_s": -3.286042490118577
      },
      "urgent": {
        "checkpoints": 8,
        "delta_v": -0.0012193608670461147,
        "time_s": -1.4285119348852895
      }
    }
  }
}
```

## 配对曲线与历史对照

|种子|步数|本轮EACC−SAC|剔除最大两例|紧急|经济|捕获EACC/SAC|unsafe EACC/SAC|修复前EACC−SAC|
|---|---:|---:|---:|---:|---:|---|---|---:|
|0|1000|-13.780834|-23.800736|-25.677094|-0.850116|38/41|0/0|-20.067031|
|0|2000|-10.558978|-11.381788|-19.121563|-1.251819|41/43|1/0|-9.373589|
|0|3000|-4.013307|-9.084654|-9.284734|1.716505|42/43|0/0|-8.787404|
|0|4000|-4.495481|-4.971401|-8.817874|0.202772|42/43|0/0|-0.217215|
|1|1000|22.602311|14.017987|41.679042|1.866735|41/36|0/0|-6.352559|
|1|2000|24.039900|15.752718|40.878815|5.736732|44/39|0/0|-9.782688|
|1|3000|14.689535|5.830746|25.886136|2.519316|44/41|0/0|-21.853285|
|1|4000|17.411984|7.337912|18.798948|15.904414|44/41|0/0|-14.452282|

种子0四点均值：修复前-9.611310，本轮-8.212150；历史对照本轮重述，不能由此单独推断标签修复的因果贡献。

种子1四点均值：修复前-13.110203，本轮19.685933；历史对照本轮重述，不能由此单独推断标签修复的因果贡献。

完整训练曲线和四点评估见 `图/CURVES.png` / PDF；原始TensorBoard事件保留本机并列SHA256，训练逐回合曲线无补点、无插值。

## 机制、分档与墙钟

```json
{
  "training": {
    "TRAIN_e_mve_s0": {
      "finished": {
        "steps": 4000,
        "updates": 7602,
        "wall_s": 49910.33018541336,
        "replay": {
          "transitions": 4000,
          "with_candidates": 3385,
          "candidates": 25668,
          "entry_candidates": 5030,
          "reason": {
            "off": 49,
            "entry_calibration": 281,
            "stage": 2803,
            "entry": 665,
            "not_comparable": 202
          },
          "cutoff_candidates": 0,
          "two_stage": 17063,
          "two_stage_entry": 4133,
          "two_stage_cutoff": 213
        }
      },
      "probes": 305,
      "probes_with_model_rows": 297,
      "first_model_step": 300,
      "w_mean": 0.951988935069203,
      "abs_err_median": 0.3397560119628906,
      "sigma_last": 17.039026207704374,
      "model_td_last": 128.87867736816406,
      "q_pi_last_decile": 22.502199109395345,
      "loss_share": 0.3326457245168638,
      "grad_cos": 0.37773392379283904,
      "two_stage_share": 0.5788889941334374,
      "decisive_share": 0.1359434788325419,
      "c_mean": 0.1885753164086679,
      "c_low_share": 0.8258385040141918,
      "increment_abs_mean": 2.027255127124915,
      "age_capped_share": 0.07956693181297694,
      "label_age_mean": 927.0591044313578,
      "entry_bias_now": -2.696354841307449,
      "entry_share": 0.1901530440808948,
      "stage2_rows": 915.6161616161617,
      "w2_mean": 0.9671471650752957,
      "pi_gap_sigma_median": 0.03293069881980499,
      "stage2_entry_share": 0.23114604400063204,
      "stage1_cutoff_share": 0.0,
      "stage2_cutoff_share": 0.010321502591426852
    },
    "TRAIN_e_mve_s1": {
      "finished": {
        "steps": 4000,
        "updates": 7602,
        "wall_s": 51553.42220830917,
        "replay": {
          "transitions": 4000,
          "with_candidates": 3327,
          "candidates": 25579,
          "entry_candidates": 5248,
          "reason": {
            "off": 39,
            "entry_calibration": 297,
            "stage": 2743,
            "entry": 697,
            "not_comparable": 224
          },
          "cutoff_candidates": 0,
          "two_stage": 17149,
          "two_stage_entry": 4413,
          "two_stage_cutoff": 131
        }
      },
      "probes": 305,
      "probes_with_model_rows": 296,
      "first_model_step": 312,
      "w_mean": 0.9465789756662136,
      "abs_err_median": 0.2731513977050781,
      "sigma_last": 14.548119059804527,
      "model_td_last": 168.15701293945312,
      "q_pi_last_decile": 32.28206596374512,
      "loss_share": 0.4024971407303374,
      "grad_cos": 0.306657042788543,
      "two_stage_share": 0.5870950853395127,
      "decisive_share": 0.13267282127226526,
      "c_mean": 0.2085379626903985,
      "c_low_share": 0.8038943510351552,
      "increment_abs_mean": 2.0909703516879596,
      "age_capped_share": 0.07752458590422631,
      "label_age_mean": 906.1754277203534,
      "entry_bias_now": -1.9016913828183248,
      "entry_share": 0.2020953082404024,
      "stage2_rows": 888.3648648648649,
      "w2_mean": 0.9729893443149489,
      "pi_gap_sigma_median": 0.035824032983647475,
      "stage2_entry_share": 0.22553060119468216,
      "stage1_cutoff_share": 0.0,
      "stage2_cutoff_share": 0.002775942630840881
    }
  },
  "urgent_bins_and_both_success": {
    "s0/step_1000": {
      "urgent_bins": {
        "high": {
          "n": 11,
          "mean": -57.56421592108124,
          "capture_mve": 4,
          "capture_sac": 7
        },
        "mid": {
          "n": 7,
          "mean": -31.729509916016355,
          "capture_mve": 5,
          "capture_sac": 6
        },
        "low": {
          "n": 7,
          "mean": 30.483655781124842,
          "capture_mve": 6,
          "capture_sac": 5
        }
      },
      "both_success": {
        "all": {
          "n": 36,
          "delta_v": 0.03005794903184216,
          "time_s": -1.3972222222222213
        },
        "economy": {
          "n": 23,
          "delta_v": 0.022152902875856695,
          "time_s": -1.799999999999999
        },
        "urgent": {
          "n": 13,
          "delta_v": 0.04404379992320107,
          "time_s": -0.684615384615384
        }
      }
    },
    "s0/step_2000": {
      "urgent_bins": {
        "high": {
          "n": 11,
          "mean": -18.725095361795226,
          "capture_mve": 6,
          "capture_sac": 7
        },
        "mid": {
          "n": 7,
          "mean": -38.66215930098154,
          "capture_mve": 5,
          "capture_sac": 6
        },
        "low": {
          "n": 7,
          "mean": -0.2039882196149309,
          "capture_mve": 7,
          "capture_sac": 7
        }
      },
      "both_success": {
        "all": {
          "n": 41,
          "delta_v": 0.02371205482775145,
          "time_s": 3.041463414634147
        },
        "economy": {
          "n": 23,
          "delta_v": 0.028810701484219357,
          "time_s": 4.969565217391304
        },
        "urgent": {
          "n": 18,
          "delta_v": 0.017197117433375795,
          "time_s": 0.5777777777777788
        }
      }
    },
    "s0/step_3000": {
      "urgent_bins": {
        "high": {
          "n": 11,
          "mean": -0.47522783925963485,
          "capture_mve": 7,
          "capture_sac": 7
        },
        "mid": {
          "n": 7,
          "mean": -31.496253419436307,
          "capture_mve": 5,
          "capture_sac": 6
        },
        "low": {
          "n": 7,
          "mean": -0.9167248713310455,
          "capture_mve": 7,
          "capture_sac": 7
        }
      },
      "both_success": {
        "all": {
          "n": 41,
          "delta_v": -0.014951583304256827,
          "time_s": -0.014634146341463277
        },
        "economy": {
          "n": 23,
          "delta_v": -0.04229306164150677,
          "time_s": -1.23913043478261
        },
        "urgent": {
          "n": 18,
          "delta_v": 0.019984750126673667,
          "time_s": 1.5500000000000018
        }
      }
    },
    "s0/step_4000": {
      "urgent_bins": {
        "high": {
          "n": 11,
          "mean": -19.163138937062488,
          "capture_mve": 6,
          "capture_sac": 7
        },
        "mid": {
          "n": 7,
          "mean": -0.3887430954272101,
          "capture_mve": 6,
          "capture_sac": 6
        },
        "low": {
          "n": 7,
          "mean": -0.9901603828770261,
          "capture_mve": 7,
          "capture_sac": 7
        }
      },
      "both_success": {
        "all": {
          "n": 42,
          "delta_v": 0.015664746024963417,
          "time_s": 1.5690476190476208
        },
        "economy": {
          "n": 23,
          "delta_v": -0.005534513309541877,
          "time_s": 0.9304347826086972
        },
        "urgent": {
          "n": 19,
          "delta_v": 0.04132700732462772,
          "time_s": 2.342105263157897
        }
      }
    },
    "s1/step_1000": {
      "urgent_bins": {
        "high": {
          "n": 11,
          "mean": 16.331261617738352,
          "capture_mve": 5,
          "capture_sac": 4
        },
        "mid": {
          "n": 7,
          "mean": 30.26497631052473,
          "capture_mve": 6,
          "capture_sac": 5
        },
        "low": {
          "n": 7,
          "mean": 92.92533407629124,
          "capture_mve": 7,
          "capture_sac": 4
        }
      },
      "both_success": {
        "all": {
          "n": 36,
          "delta_v": -0.031451269034774354,
          "time_s": -2.9250000000000007
        },
        "economy": {
          "n": 23,
          "delta_v": -0.04482488713564547,
          "time_s": -3.6869565217391305
        },
        "urgent": {
          "n": 13,
          "delta_v": -0.0077902523947716105,
          "time_s": -1.576923076923079
        }
      }
    },
    "s1/step_2000": {
      "urgent_bins": {
        "high": {
          "n": 11,
          "mean": 53.670382966065105,
          "capture_mve": 8,
          "capture_sac": 5
        },
        "mid": {
          "n": 7,
          "mean": 29.98627665203414,
          "capture_mve": 6,
          "capture_sac": 5
        },
        "low": {
          "n": 7,
          "mean": 31.6703162270439,
          "capture_mve": 7,
          "capture_sac": 6
        }
      },
      "both_success": {
        "all": {
          "n": 39,
          "delta_v": -0.09112915890226111,
          "time_s": -5.97948717948718
        },
        "economy": {
          "n": 23,
          "delta_v": -0.13859004870441352,
          "time_s": -9.656521739130433
        },
        "urgent": {
          "n": 16,
          "delta_v": -0.022904129811666972,
          "time_s": -0.6937499999999996
        }
      }
    },
    "s1/step_3000": {
      "urgent_bins": {
        "high": {
          "n": 11,
          "mean": 17.95083518892931,
          "capture_mve": 8,
          "capture_sac": 7
        },
        "mid": {
          "n": 7,
          "mean": 32.71100276833537,
          "capture_mve": 6,
          "capture_sac": 5
        },
        "low": {
          "n": 7,
          "mean": 31.53102749676328,
          "capture_mve": 7,
          "capture_sac": 6
        }
      },
      "both_success": {
        "all": {
          "n": 41,
          "delta_v": -0.02530264677055369,
          "time_s": -4.121951219512195
        },
        "economy": {
          "n": 23,
          "delta_v": -0.06009594821634901,
          "time_s": -5.773913043478261
        },
        "urgent": {
          "n": 18,
          "delta_v": 0.01915546063240701,
          "time_s": -2.0111111111111097
        }
      }
    },
    "s1/step_4000": {
      "urgent_bins": {
        "high": {
          "n": 11,
          "mean": -0.9564393595169194,
          "capture_mve": 8,
          "capture_sac": 8
        },
        "mid": {
          "n": 7,
          "mean": 62.498823152143814,
          "capture_mve": 6,
          "capture_sac": 4
        },
        "low": {
          "n": 7,
          "mean": 6.143254439340879,
          "capture_mve": 7,
          "capture_sac": 7
        }
      },
      "both_success": {
        "all": {
          "n": 41,
          "delta_v": -0.10742356981975067,
          "time_s": -10.448780487804878
        },
        "economy": {
          "n": 22,
          "delta_v": -0.09589828178980365,
          "time_s": -10.031818181818183
        },
        "urgent": {
          "n": 19,
          "delta_v": -0.12076864017021563,
          "time_s": -10.93157894736842
        }
      }
    }
  }
}
```

## 动作价值排序审计

下表以原生D_RANK逐项读数为准。准确率中并列计0.5；关键对按捕获翻转或真值回报差≥50定义。top1分母为真值可区分决策。

|检查点|评分|关键对数|准确率|Kendall tau|top1|
|---|---|---:|---:|---:|---:|
|e_mve_s0_step_4000|critic|36|0.333333|0.000000|0.000000|
|e_mve_s0_step_4000|label1|36|0.500000|0.131313|0.333333|
|e_mve_s0_step_4000|label2|36|0.444444|0.161616|0.000000|
|e_mve_s0_step_4000|diff|36|0.472222|0.131313|0.000000|
|e_mve_s1_step_4000|critic|26|1.000000|0.148796|0.000000|
|e_mve_s1_step_4000|label1|26|0.769231|0.114996|0.000000|
|e_mve_s1_step_4000|label2|26|0.692308|0.122378|0.000000|
|e_mve_s1_step_4000|diff|26|0.769231|0.097902|0.000000|
|e_sac_s0_step_4000|critic|26|0.076923|-0.057239|0.000000|
|e_sac_s0_step_4000|label1|26|0.961538|0.116582|1.000000|
|e_sac_s0_step_4000|label2|26|0.923077|0.338384|0.500000|
|e_sac_s0_step_4000|diff|26|0.961538|0.107744|1.000000|
|e_sac_s1_step_4000|critic|44|0.659091|-0.151481|0.200000|
|e_sac_s1_step_4000|label1|44|0.772727|0.047035|0.600000|
|e_sac_s1_step_4000|label2|44|0.477273|0.043084|0.400000|
|e_sac_s1_step_4000|diff|44|0.795455|0.158249|0.600000|

审计解释和误差（不改变性能判定，不自动安排下一轮）：

```json
{
  "readings": {
    "e_mve_s0_step_4000": {
      "decisions": 15,
      "items": 91,
      "decisions_truth_separates": 3,
      "critic": {
        "decisive_pairs": 36,
        "right": 12,
        "wrong": 24,
        "tie": 0,
        "accuracy": 0.3333333333333333,
        "kendall_tau_mean": 1.766263902812749e-17,
        "top1_on_separating": 0.0
      },
      "label1": {
        "decisive_pairs": 36,
        "right": 18,
        "wrong": 18,
        "tie": 0,
        "accuracy": 0.5,
        "kendall_tau_mean": 0.1313131313131313,
        "top1_on_separating": 0.3333333333333333
      },
      "label2": {
        "decisive_pairs": 36,
        "right": 16,
        "wrong": 20,
        "tie": 0,
        "accuracy": 0.4444444444444444,
        "kendall_tau_mean": 0.16161616161616157,
        "top1_on_separating": 0.0
      },
      "diff": {
        "decisive_pairs": 36,
        "right": 17,
        "wrong": 19,
        "tie": 0,
        "accuracy": 0.4722222222222222,
        "kendall_tau_mean": 0.1313131313131313,
        "top1_on_separating": 0.0
      },
      "label1_abs_error": {
        "event": {
          "n": 8,
          "median": 0.014008137387364172
        },
        "bootstrap": {
          "n": 68,
          "median": 31.77786213921992
        }
      },
      "label2_abs_error": {
        "event": {
          "n": 8,
          "median": 0.014008137387364172
        },
        "bootstrap": {
          "n": 68,
          "median": 30.96788129134101
        }
      },
      "diff_abs_error": {
        "event": {
          "n": 8,
          "median": 3.550072842457091
        },
        "bootstrap": {
          "n": 68,
          "median": 39.80352688894558
        }
      }
    },
    "e_mve_s1_step_4000": {
      "decisions": 13,
      "items": 117,
      "decisions_truth_separates": 2,
      "critic": {
        "decisive_pairs": 26,
        "right": 26,
        "wrong": 0,
        "tie": 0,
        "accuracy": 1.0,
        "kendall_tau_mean": 0.1487956487956488,
        "top1_on_separating": 0.0
      },
      "label1": {
        "decisive_pairs": 26,
        "right": 20,
        "wrong": 6,
        "tie": 0,
        "accuracy": 0.7692307692307693,
        "kendall_tau_mean": 0.114996114996115,
        "top1_on_separating": 0.0
      },
      "label2": {
        "decisive_pairs": 26,
        "right": 18,
        "wrong": 8,
        "tie": 0,
        "accuracy": 0.6923076923076923,
        "kendall_tau_mean": 0.12237762237762236,
        "top1_on_separating": 0.0
      },
      "diff": {
        "decisive_pairs": 26,
        "right": 20,
        "wrong": 6,
        "tie": 0,
        "accuracy": 0.7692307692307693,
        "kendall_tau_mean": 0.09790209790209792,
        "top1_on_separating": 0.0
      },
      "label1_abs_error": {
        "event": {
          "n": 4,
          "median": 89.12436677611208
        },
        "bootstrap": {
          "n": 100,
          "median": 22.50039284436572
        }
      },
      "label2_abs_error": {
        "event": {
          "n": 4,
          "median": 89.12436677611208
        },
        "bootstrap": {
          "n": 100,
          "median": 31.223478242879246
        }
      },
      "diff_abs_error": {
        "event": {
          "n": 4,
          "median": 110.44189043491664
        },
        "bootstrap": {
          "n": 100,
          "median": 22.135049235712017
        }
      }
    },
    "e_sac_s0_step_4000": {
      "decisions": 13,
      "items": 109,
      "decisions_truth_separates": 2,
      "critic": {
        "decisive_pairs": 26,
        "right": 2,
        "wrong": 24,
        "tie": 0,
        "accuracy": 0.07692307692307693,
        "kendall_tau_mean": -0.05723905723905725,
        "top1_on_separating": 0.0
      },
      "label1": {
        "decisive_pairs": 26,
        "right": 25,
        "wrong": 1,
        "tie": 0,
        "accuracy": 0.9615384615384616,
        "kendall_tau_mean": 0.11658249158249158,
        "top1_on_separating": 1.0
      },
      "label2": {
        "decisive_pairs": 26,
        "right": 24,
        "wrong": 2,
        "tie": 0,
        "accuracy": 0.9230769230769231,
        "kendall_tau_mean": 0.3383838383838384,
        "top1_on_separating": 0.5
      },
      "diff": {
        "decisive_pairs": 26,
        "right": 25,
        "wrong": 1,
        "tie": 0,
        "accuracy": 0.9615384615384616,
        "kendall_tau_mean": 0.10774410774410775,
        "top1_on_separating": 1.0
      },
      "label1_abs_error": {
        "event": {
          "n": 0,
          "median": null
        },
        "bootstrap": {
          "n": 96,
          "median": 12.164594896994089
        }
      },
      "label2_abs_error": {
        "event": {
          "n": 0,
          "median": null
        },
        "bootstrap": {
          "n": 96,
          "median": 9.820045142220472
        }
      },
      "diff_abs_error": {
        "event": {
          "n": 0,
          "median": null
        },
        "bootstrap": {
          "n": 96,
          "median": 12.764067942607731
        }
      }
    },
    "e_sac_s1_step_4000": {
      "decisions": 21,
      "items": 161,
      "decisions_truth_separates": 5,
      "critic": {
        "decisive_pairs": 44,
        "right": 29,
        "wrong": 15,
        "tie": 0,
        "accuracy": 0.6590909090909091,
        "kendall_tau_mean": -0.15148079433793718,
        "top1_on_separating": 0.2
      },
      "label1": {
        "decisive_pairs": 44,
        "right": 34,
        "wrong": 10,
        "tie": 0,
        "accuracy": 0.7727272727272727,
        "kendall_tau_mean": 0.047034975606404184,
        "top1_on_separating": 0.6
      },
      "label2": {
        "decisive_pairs": 44,
        "right": 21,
        "wrong": 23,
        "tie": 0,
        "accuracy": 0.4772727272727273,
        "kendall_tau_mean": 0.04308390022675737,
        "top1_on_separating": 0.4
      },
      "diff": {
        "decisive_pairs": 44,
        "right": 35,
        "wrong": 9,
        "tie": 0,
        "accuracy": 0.7954545454545454,
        "kendall_tau_mean": 0.15824915824915825,
        "top1_on_separating": 0.6
      },
      "label1_abs_error": {
        "event": {
          "n": 15,
          "median": 3.1312479559421007
        },
        "bootstrap": {
          "n": 125,
          "median": 14.730758495410015
        }
      },
      "label2_abs_error": {
        "event": {
          "n": 15,
          "median": 3.1312479559421007
        },
        "bootstrap": {
          "n": 125,
          "median": 10.793271042896322
        }
      },
      "diff_abs_error": {
        "event": {
          "n": 15,
          "median": 11.959100071180643
        },
        "bootstrap": {
          "n": 125,
          "median": 16.119793270770785
        }
      }
    }
  },
  "decision_rules": "Pre-registered section3; diagnostic only; not the t26w verdict",
  "matches": [
    "两个种子的EACC critic关键对准确率均高于SAC；若性能未转正，下一步对象为Actor更新/探索。",
    "种子0：['label1', 'label2']准确率高于自身critic，符合核查融合方式的分支。",
    "种子0 label1：准确率<=0.5或多数关键对并列，符合核查候选信息构造的分支。",
    "种子0 label2：准确率<=0.5或多数关键对并列，符合核查候选信息构造的分支。"
  ],
  "seed0_bootstrap_error": {
    "absolute": {
      "n": 68,
      "median": 31.77786213921992
    },
    "difference": {
      "n": 68,
      "median": 39.80352688894558
    },
    "diff_over_label1": 1.2525552132665485
  },
  "seed1_bootstrap_error": {
    "absolute": {
      "n": 100,
      "median": 22.50039284436572
    },
    "difference": {
      "n": 100,
      "median": 22.135049235712017
    },
    "diff_over_label1": 0.9837627897796821
  },
  "limits": [
    "执行书没有给明显更小的数值门槛，报告误差和比率，不事后造门槛。",
    "各检查点审计沿各自策略访问状态，策略后续真值也各自不同；四个检查点汇总不是跨策略共同状态的配对因果估计。每个决策内部四种打分在同一候选组上对照相同真值。",
    "label2为审计工具定义的两阶段目标，非完整生产软Retrace融合目标。审计不训练、不实施候选差替换。",
    "多个预登记解释可以同时匹配；不足以唯一归因时保留不确定性，不强行归到一行。"
  ]
}
```

## 证据与交付边界

本轮原始评估、审计JSON、训练manifest/finished/episodes/model_stats/heartbeat、启动/日志验收、模型SHA256、原生读出和判定一并提供；SHA256SUMS验证本轮文件。历史基线不复制整套目录，仅保留固定引用、哈希及本轮复用所必需的SAC评估。模型ZIP和TensorBoard原始事件留本机。所有状态、日志、脚本在本轮过程目录；OS_ANCESTRY证明任务不依赖Codex进程。

过程目录：`C:\Users\35884\Documents\Spacecraft\过程文件\T2_6W1_CUTOFF_20261029`。科学交付路径：`evidence/T2_6W1_CUTOFF/`；collab交付：`docs/collaboration/lower/handoffs/t2_6w1_cutoff_20261029/`。训练/评估/审计均已结束；下一版方案及执行书由上层审读后决定，本轮不擅改机制。
