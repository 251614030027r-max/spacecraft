# RUN_ECC 开发轮交付

科学冻结 1cddbae26233b167d8c418614c467d21bf51b833，collab 执行书 ca4980b4。SAC_C、EACH_C、ECC 各种子 10、11，各从零训练 2000 步、2 异步 worker。新开局 294320–294367 共 480 例，含 ECC 关闭认证、EACH_C 部署认证的同模型配对。CPU 0–10 用于本轮，CPU 11 留桌面。

自检全部通过。pytest 退出码 0，三组 40 步冒烟和五组冒烟评估通过，原生 manifest、认证开关、TensorBoard 和快照锁定恢复已核验。正式 CHECK 完整，逐例核验 480 个开局、标签和认证开关。模型 SHA 见 REGISTRY_RESOLVED.json，TensorBoard 核验见 FINAL_TENSORBOARD_VERIFICATION.json。

原生 final_method 为 **ECC_deploy**。以下为冻结读出原文。

```
{
 "margin": 8.3,
 "evaluations": {
  "SAC_C_s10": 48,
  "SAC_C_s11": 48,
  "EACH_C_s10": 48,
  "EACH_C_s11": 48,
  "ECC_s10": 48,
  "ECC_s11": 48,
  "ECC_NOCERT_s10": 48,
  "ECC_NOCERT_s11": 48,
  "EACH_C_CERT_s10": 48,
  "EACH_C_CERT_s11": 48
 },
 "complete": true,
 "final_method": "ECC_deploy",
 "arms": {
  "SAC_C_s10": {
   "n": 48,
   "capture": 40,
   "unsafe": 0,
   "timeout": 8,
   "urgent_n": 21,
   "urgent_capture": 13,
   "mean_return": 25.611506838147836,
   "median_return": 51.23371417002471,
   "entries_lost": 2
  },
  "SAC_C_s11": {
   "n": 48,
   "capture": 38,
   "unsafe": 0,
   "timeout": 10,
   "urgent_n": 21,
   "urgent_capture": 11,
   "mean_return": 13.711464545086736,
   "median_return": 43.41302218200735,
   "entries_lost": 1
  },
  "EACH_C_s10": {
   "n": 48,
   "capture": 39,
   "unsafe": 0,
   "timeout": 9,
   "urgent_n": 21,
   "urgent_capture": 12,
   "mean_return": 18.54984952928987,
   "median_return": 46.316027464103726,
   "entries_lost": 0
  },
  "EACH_C_s11": {
   "n": 48,
   "capture": 42,
   "unsafe": 0,
   "timeout": 6,
   "urgent_n": 21,
   "urgent_capture": 15,
   "mean_return": 32.64491320174228,
   "median_return": 57.454835530966264,
   "entries_lost": 2
  },
  "ECC_s10": {
   "n": 48,
   "capture": 41,
   "unsafe": 1,
   "timeout": 6,
   "urgent_n": 21,
   "urgent_capture": 14,
   "mean_return": 27.060690021883925,
   "median_return": 48.90626464482391,
   "entries_lost": 1
  },
  "ECC_s11": {
   "n": 48,
   "capture": 40,
   "unsafe": 0,
   "timeout": 8,
   "urgent_n": 21,
   "urgent_capture": 13,
   "mean_return": 23.70312852885482,
   "median_return": 47.66235655548465,
   "entries_lost": 0
  },
  "ECC_NOCERT_s10": {
   "n": 48,
   "capture": 40,
   "unsafe": 0,
   "timeout": 8,
   "urgent_n": 21,
   "urgent_capture": 13,
   "mean_return": 23.454154872980396,
   "median_return": 48.370242901192825,
   "entries_lost": 1
  },
  "ECC_NOCERT_s11": {
   "n": 48,
   "capture": 40,
   "unsafe": 0,
   "timeout": 8,
   "urgent_n": 21,
   "urgent_capture": 13,
   "mean_return": 24.09093100798786,
   "median_return": 47.66235655548465,
   "entries_lost": 1
  },
  "EACH_C_CERT_s10": {
   "n": 48,
   "capture": 41,
   "unsafe": 0,
   "timeout": 7,
   "urgent_n": 21,
   "urgent_capture": 14,
   "mean_return": 27.08050910956528,
   "median_return": 50.08972227132617,
   "entries_lost": 0
  },
  "EACH_C_CERT_s11": {
   "n": 48,
   "capture": 43,
   "unsafe": 0,
   "timeout": 5,
   "urgent_n": 21,
   "urgent_capture": 16,
   "mean_return": 36.674542662088406,
   "median_return": 59.04847691108176,
   "entries_lost": 1
  }
 },
 "interventions": {
  "SAC_C_s10": 0,
  "SAC_C_s11": 0,
  "EACH_C_s10": 0,
  "EACH_C_s11": 0,
  "ECC_s10": 8,
  "ECC_s11": 12,
  "ECC_NOCERT_s10": 0,
  "ECC_NOCERT_s11": 0,
  "EACH_C_CERT_s10": 13,
  "EACH_C_CERT_s11": 10
 },
 "ecc_vs_sac": {
  "pairs": 96,
  "mean": 5.720423583752084,
  "median": -0.30900860877792624,
  "without_top2_per_seed": -2.890793394288364,
  "per_seed": {
   "10": {
    "n": 48,
    "mean": 1.4491831837360838,
    "median": -1.400101447825918,
    "without_top2": -7.361652169810448
   },
   "11": {
    "n": 48,
    "mean": 9.991663983768081,
    "median": 0.21568353317189803,
    "without_top2": 1.58006538123372
   }
  },
  "capture": [
   81,
   78
  ],
  "unsafe_first": 1,
  "flips": {
   "only_first_captures": 4,
   "only_second_captures": 1
  },
  "both_captured": {
   "n": 77,
   "time_s": [
    135.65454545454543,
    133.6168831168831
   ],
   "delta_v": [
    1.2992836635924288,
    1.3029539042373999
   ]
  },
  "gates": {
   "G1": false,
   "G2": false,
   "G3": true,
   "G4": true,
   "G5": false
  },
  "pass": false
 },
 "ecc_vs_each": {
  "pairs": 96,
  "mean": -0.2154720901467003,
  "median": -0.08639377674827386,
  "without_top2_per_seed": -7.221022681571385,
  "per_seed": {
   "10": {
    "n": 48,
    "mean": 8.510840492594056,
    "median": -0.5152602973642075,
    "without_top2": -0.2512634794640548
   },
   "11": {
    "n": 48,
    "mean": -8.941784672887456,
    "median": 0.39302816795862583,
    "without_top2": -14.19078188367872
   }
  },
  "capture": [
   81,
   81
  ],
  "unsafe_first": 1,
  "flips": {
   "only_first_captures": 3,
   "only_second_captures": 3
  },
  "both_captured": {
   "n": 78,
   "time_s": [
    136.7230769230769,
    133.27564102564102
   ],
   "delta_v": [
    1.3055024834066644,
    1.3177486794776674
   ]
  },
  "not_degraded": false
 },
 "fallback_each_cert_vs_sac": {
  "pairs": 96,
  "mean": 12.21604019420956,
  "median": -0.2413835190504301,
  "without_top2_per_seed": 3.7535735905055154,
  "per_seed": {
   "10": {
    "n": 48,
    "mean": 1.469002271417442,
    "median": -0.4251746782521053,
    "without_top2": -7.39529451201526
   },
   "11": {
    "n": 48,
    "mean": 22.963078117001672,
    "median": -0.08110747593074841,
    "without_top2": 14.90244169302629
   }
  },
  "capture": [
   84,
   78
  ],
  "unsafe_first": 0,
  "flips": {
   "only_first_captures": 8,
   "only_second_captures": 2
  },
  "both_captured": {
   "n": 76,
   "time_s": [
    132.15921052631577,
    133.82631578947368
   ],
   "delta_v": [
    1.30875700771411,
    1.3010359710811077
   ]
  },
  "gates": {
   "G1": true,
   "G2": true,
   "G3": true,
   "G4": true,
   "G5": true
  },
  "pass": true
 },
 "report_certification_same_model": {
  "pairs": 96,
  "mean": 1.6093663348852456,
  "median": 0.0,
  "without_top2_per_seed": -5.011657026274898,
  "per_seed": {
   "10": {
    "n": 48,
    "mean": 3.606535148903529,
    "median": 0.0,
    "without_top2": -5.147060968568212
   },
   "11": {
    "n": 48,
    "mean": -0.3878024791330379,
    "median": 0.0,
    "without_top2": -4.876253083981584
   }
  },
  "capture": [
   81,
   80
  ],
  "unsafe_first": 1,
  "flips": {
   "only_first_captures": 3,
   "only_second_captures": 2
  },
  "both_captured": {
   "n": 78,
   "time_s": [
    135.95128205128205,
    136.2551282051282
   ],
   "delta_v": [
    1.2984880759165058,
    1.3037843564194678
   ]
  }
 },
 "report_each_vs_sac": {
  "pairs": 96,
  "mean": 5.935895673898784,
  "median": -0.2413835190504301,
  "without_top2_per_seed": -0.6201803846737757,
  "per_seed": {
   "10": {
    "n": 48,
    "mean": -7.0616573088579715,
    "median": -0.4251746782521053,
    "without_top2": -11.937971721143091
   },
   "11": {
    "n": 48,
    "mean": 18.93344865665554,
    "median": -0.0827999922636451,
    "without_top2": 10.69761095179554
   }
  },
  "capture": [
   81,
   78
  ],
  "unsafe_first": 0,
  "flips": {
   "only_first_captures": 6,
   "only_second_captures": 3
  },
  "both_captured": {
   "n": 75,
   "time_s": [
    132.66266666666667,
    134.24533333333332
   ],
   "delta_v": [
    1.3121005028345951,
    1.3051122938873578
   ]
  }
 },
 "curves": {
  "SAC_C_s10": {
   "curve_auc": 13.222292489572885,
   "late_return": 30.942340025942787,
   "late_urgent_capture": 0.7037037037037037,
   "episodes": 408
  },
  "SAC_C_s11": {
   "curve_auc": -10.739096439794537,
   "late_return": 10.076285995993304,
   "late_urgent_capture": 0.5925925925925926,
   "episodes": 334
  },
  "EACH_C_s10": {
   "curve_auc": 5.7839789136818105,
   "late_return": 26.110464326125637,
   "late_urgent_capture": 0.7169811320754716,
   "episodes": 388
  },
  "EACH_C_s11": {
   "curve_auc": 14.252861658792796,
   "late_return": 32.461828125517115,
   "late_urgent_capture": 0.7777777777777778,
   "episodes": 391
  },
  "ECC_s10": {
   "curve_auc": 26.26204918224272,
   "late_return": 38.14858229619849,
   "late_urgent_capture": 0.8,
   "episodes": 451
  },
  "ECC_s11": {
   "curve_auc": 23.19008182350275,
   "late_return": 32.845075274875846,
   "late_urgent_capture": 0.7868852459016393,
   "episodes": 413
  }
 }
}
```

11 月 8 日 03:29 JST，Windows 更新重启中断最后一组评估。已完成的六条训练及 453 例评估全部保留，只补齐剩余 27 例，已有逐例 SHA256 逐一复核。自检曾因外部日志等待窗口过短停下，保全首轮记录后延长窗口，复验通过。

墙钟见 timeline.log，末条 all done after OS restart recovery at 2026-11-08T11:02:42.2996303+09:00。训练 unsafe 原生记录见 TRAIN_UNSAFE_FINAL.json，共 3 次，缺失字段按原样保留。

科学证据位于 evidence/ECC_DEV，提交 cc1a09548bd81e13b8d78f4f26f1640d6640b5d5，526 个文件已核对 Git 原字节。正式实验另行授权。
