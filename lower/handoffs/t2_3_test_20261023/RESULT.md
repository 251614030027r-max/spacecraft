# T2.3 B：种子0完整判定，种子1部分曲线补充

生成：2026-10-25T20:49:22.374018+09:00。科学冻结 `9fc96e82018a7358a9e866e9e90377bdcb5941e7`；本机23项测试通过。用户最新授权缩减为一个完整种子，原双种子结论与自动P/N/K/L分支不再执行；任务、奖励、参数、开局、步数及原生t23数值判据未改。

## 范围偏离与证据强度

{
  "authorization": "User latest stop seed1 by progress; retain seed0 original pair to4000 and seed1 partial1k/2k",
  "decision_jst": "2026-10-25T17:38:57.131294+09:00",
  "actual_seed1_last_logged": {
    "e_sac": 3000,
    "e_mve": 3000
  },
  "correction": "Upper example MVEseed1=2000 was outdated; actual MVE/SACseed1 both3000 at stop. Seed1 MVE reached3000 later than seed0. No3000/4000 evaluation existed or was read before selection.",
  "seed0_training_restarted": false,
  "seed1_assets_deleted": false,
  "seed1_episode_and_model_stats": "MISSING: frozen trainer writes at normal finish; seed1 intentionally stopped",
  "native_criteria": "unchanged t23 --final4000, only one complete seed0 in input; weaker than original two-complete-seed plan",
  "no_automatic_branch": true,
  "samples": {
    "seed0_full": 240,
    "seed1_partial": 96
  },
  "request": "Await upper review and new user approval after delivery; do not start P/N/K/L."
}

## 原生种子0判定
```json
{
  "seeds": [
    "0"
  ],
  "final": 4000,
  "criteria": {
    "C0_complete": true,
    "C1_beyond_t22": false,
    "C2_coupling_auc": false,
    "C3_end_level": false,
    "C4_converged": true,
    "C5_safety": true
  },
  "values": {
    "eacc_final": {
      "0": 42.36499930600538
    },
    "sac_final": {
      "0": 54.47660355899104
    },
    "paired_auc": {
      "0": -9.015146129419966
    },
    "paired_final": {
      "0": -12.111604252985662
    },
    "paired_final_urgent": {
      "0": -19.136867192946493
    },
    "paired_final_without_top2": {
      "0": -22.569731348268842
    },
    "last_step_change": {
      "e_mve_s0": 7.229779465293987,
      "e_sac_s0": -0.27923011171375833
    },
    "learning_gain": {
      "mve": 56.955111184856975,
      "sac": 69.06671543784263
    }
  },
  "verdict": "coupling_failure"
}
```

该判定只覆盖一个完整开发种子，不等于原计划两个完整种子的复现结论，不据此宣称主创新已成立。

## 固定开发开局学习曲线

|范围|臂|步数|任务|n|均回报|捕获|unsafe|
|---|---|---:|---|---:|---:|---:|---:|
|seed0判定|e_mve_s0|0|all|24|-14.590111878851593|15|0|
|seed0判定|e_mve_s0|0|urgent|15|-49.11830536798308|6|0|
|seed0判定|e_mve_s0|0|economy|9|42.95687726970089|9|0|
|seed0判定|e_mve_s0|1000|all|24|34.76142755137095|21|0|
|seed0判定|e_mve_s0|1000|urgent|15|29.568683338534658|12|0|
|seed0判定|e_mve_s0|1000|economy|9|43.41600123943142|9|0|
|seed0判定|e_mve_s0|2000|all|24|42.06363747433587|22|0|
|seed0判定|e_mve_s0|2000|urgent|15|43.96535680548393|13|0|
|seed0判定|e_mve_s0|2000|economy|9|38.894105255755754|9|0|
|seed0判定|e_mve_s0|3000|all|24|35.13521984071139|21|0|
|seed0判定|e_mve_s0|3000|urgent|15|30.06873600802959|12|0|
|seed0判定|e_mve_s0|3000|economy|9|43.579359561847745|9|0|
|seed0判定|e_mve_s0|4000|all|24|42.36499930600538|22|1|
|seed0判定|e_mve_s0|4000|urgent|15|41.13014961702924|13|1|
|seed0判定|e_mve_s0|4000|economy|9|44.42308212096561|9|0|
|seed0判定|e_sac_s0|0|all|24|-14.590111878851593|15|0|
|seed0判定|e_sac_s0|0|urgent|15|-49.11830536798308|6|0|
|seed0判定|e_sac_s0|0|economy|9|42.95687726970089|9|0|
|seed0判定|e_sac_s0|1000|all|24|36.48854557639367|21|0|
|seed0判定|e_sac_s0|1000|urgent|15|31.311230045404276|12|0|
|seed0判定|e_sac_s0|1000|economy|9|45.11740479470933|9|0|
|seed0判定|e_sac_s0|2000|all|24|44.66488588401395|22|0|
|seed0判定|e_sac_s0|2000|urgent|15|44.8866178570459|13|0|
|seed0判定|e_sac_s0|2000|economy|9|44.295332595627364|9|0|
|seed0判定|e_sac_s0|3000|all|24|54.755833670704796|23|0|
|seed0判定|e_sac_s0|3000|urgent|15|59.990403078883396|14|0|
|seed0判定|e_sac_s0|3000|economy|9|46.03155132374046|9|0|
|seed0判定|e_sac_s0|4000|all|24|54.47660355899104|23|0|
|seed0判定|e_sac_s0|4000|urgent|15|60.26701680997573|14|0|
|seed0判定|e_sac_s0|4000|economy|9|44.825914807349896|9|0|
|seed1补充|e_mve_s1|1000|all|24|25.750673096675147|20|0|
|seed1补充|e_mve_s1|1000|urgent|15|16.812045781125814|11|0|
|seed1补充|e_mve_s1|1000|economy|9|40.64838528925738|9|0|
|seed1补充|e_mve_s1|2000|all|24|36.617160871719115|21|0|
|seed1补充|e_mve_s1|2000|urgent|15|30.83326879039561|12|0|
|seed1补充|e_mve_s1|2000|economy|9|46.256981007258275|9|0|
|seed1补充|e_sac_s1|1000|all|24|26.432544287573393|20|0|
|seed1补充|e_sac_s1|1000|urgent|15|17.205131828890696|11|0|
|seed1补充|e_sac_s1|1000|economy|9|41.811565052044536|9|0|
|seed1补充|e_sac_s1|2000|all|24|43.5895344494727|22|0|
|seed1补充|e_sac_s1|2000|urgent|15|44.16671838737794|13|0|
|seed1补充|e_sac_s1|2000|economy|9|42.62756121963066|9|0|

## 同任务配对差（MVE−SAC，逐检查点/任务类别/剔除大收益）

种子0：
```json
{
  "s0/step_0": {
    "n": 24,
    "mean": 0.0,
    "without_top2": 0.0,
    "urgent": 0.0,
    "economy": 0.0,
    "positive": 0,
    "negative": 0,
    "capture_mve": 15,
    "capture_sac": 15,
    "unsafe_mve": 0,
    "unsafe_sac": 0
  },
  "s0/step_1000": {
    "n": 24,
    "mean": -1.7271180250227285,
    "without_top2": -11.480334006277085,
    "urgent": -1.742546706869617,
    "economy": -1.7014035552779125,
    "positive": 4,
    "negative": 20,
    "capture_mve": 21,
    "capture_sac": 21,
    "unsafe_mve": 0,
    "unsafe_sac": 0
  },
  "s0/step_2000": {
    "n": 24,
    "mean": -2.601248409678083,
    "without_top2": -12.537537413402838,
    "urgent": -0.9212610515619649,
    "economy": -5.401227339871613,
    "positive": 8,
    "negative": 16,
    "capture_mve": 22,
    "capture_sac": 22,
    "unsafe_mve": 0,
    "unsafe_sac": 0
  },
  "s0/step_3000": {
    "n": 24,
    "mean": -19.620613829993392,
    "without_top2": -22.108021417197236,
    "urgent": -29.921667070853797,
    "economy": -2.4521917618927094,
    "positive": 6,
    "negative": 18,
    "capture_mve": 21,
    "capture_sac": 23,
    "unsafe_mve": 0,
    "unsafe_sac": 0
  },
  "s0/step_4000": {
    "n": 24,
    "mean": -12.111604252985662,
    "without_top2": -22.569731348268842,
    "urgent": -19.136867192946493,
    "economy": -0.40283268638427927,
    "positive": 11,
    "negative": 13,
    "capture_mve": 22,
    "capture_sac": 23,
    "unsafe_mve": 1,
    "unsafe_sac": 0
  }
}
```
种子1仅1k/2k：
```json
{
  "s1/step_1000": {
    "n": 24,
    "mean": -0.6818711908982399,
    "without_top2": -10.351371476297611,
    "urgent": -0.39308604776488115,
    "economy": -1.1631797627871723,
    "positive": 10,
    "negative": 14,
    "capture_mve": 20,
    "capture_sac": 20,
    "unsafe_mve": 0,
    "unsafe_sac": 0
  },
  "s1/step_2000": {
    "n": 24,
    "mean": -6.9723735777535865,
    "without_top2": -8.937366439181991,
    "urgent": -13.333449596982314,
    "economy": 3.6294197876276306,
    "positive": 11,
    "negative": 13,
    "capture_mve": 21,
    "capture_sac": 22,
    "unsafe_mve": 0,
    "unsafe_sac": 0
  }
}
```

## 强基线与历史参照
本轮同窗口A筛选参考：
```json
{
  "rally_rule": {
    "n": 24,
    "mean_return": 10.915012889967263,
    "capture": 18,
    "unsafe": 0,
    "mean_delta_v_success": 1.3797683818813284,
    "mean_time_success": 143.52777777777774,
    "events": {
      "timeout": 6,
      "success": 18
    }
  },
  "basic_layering_best": null,
  "neutral_c_0_0": null,
  "pure_mpc": null,
  "best_constant": {
    "n": 24,
    "mean_return": 17.22068186827077,
    "capture": 19,
    "unsafe": 0,
    "mean_delta_v_success": 1.3591462693690168,
    "mean_time_success": 128.38947368421054,
    "events": {
      "success": 19,
      "timeout": 5
    }
  },
  "per_opening_best_fixed": 56.08790796722625
}
```
历史T2.2 seed1（fa54ed1、刚性执行层，science ad3e302818b7b7fb0c419cc4e4b792208ca0fc00）：SAC0k均−63.086319、9/24；3k54.315918、23/24、unsafe0；MVE3k13.393997、19/24、unsafe1，配对−40.921921。跨版本改进不能归为耦合独立收益，本轮只用同版本同种子配对归因。

## A机制与训练诊断
A1完整24开局，A2五策略×24=120配对，gstar仅域外参照。A原始证据固定于PUSH_A_RECEIPT对应science5cdbdec，当前交付重述两个完整读数：
```json
{
  "A1": {
    "raw": {
      "candidates": 769,
      "event_agreement": 0.988296488946684,
      "entry_tp": 270,
      "entry_fp": 8,
      "entry_fn": 0,
      "entry_label_err_mean": -23.269509560541017,
      "entry_label_abs_err_median": 21.95136537071067,
      "stage_reward_abs_err_median": 0.004812759518606491,
      "stage_pos_err_median_m": 0.0005201613317034126,
      "stage_pos_err_p90_m": 0.26666335019844906
    },
    "anchored": {
      "candidates": 769,
      "event_agreement": 0.9921976592977894,
      "entry_tp": 273,
      "entry_fp": 6,
      "entry_fn": 0,
      "entry_label_err_mean": -10.29466975216328,
      "entry_label_abs_err_median": 3.718283367578465,
      "stage_reward_abs_err_median": 0.003016188526252961,
      "stage_pos_err_median_m": 3.104794886234502e-05,
      "stage_pos_err_p90_m": 0.027307505662676146
    },
    "openings": 24,
    "errors": 0
  },
  "A2": {
    "pairs": 120,
    "rigid": {
      "n": 120,
      "mean_return": 10.205206909823628,
      "capture": 92,
      "unsafe": 0,
      "mean_delta_v_success": 1.4153167190662699,
      "mean_time_success": 137.4684782608696,
      "events": {
        "success": 92,
        "timeout": 27,
        "resource_exhausted": 1
      }
    },
    "window": {
      "n": 120,
      "mean_return": 19.290568130603354,
      "capture": 96,
      "unsafe": 0,
      "mean_delta_v_success": 1.3752482425262995,
      "mean_time_success": 132.85729166666667,
      "events": {
        "success": 96,
        "timeout": 24
      }
    },
    "paired_return": {
      "mean": 9.08536122077973,
      "without_top2": 4.901469752071885,
      "positive": 42,
      "negative": 30
    },
    "entries_lost": {
      "rigid": 110,
      "window": 3
    },
    "per_policy": {
      "c_10_-5": 1.1828418421511862,
      "c_10_0": 1.0601244532674297,
      "c_6_5": 13.80400851459026,
      "gstar": 1.0197652721418722,
      "rally": 28.360066021747894
    }
  }
}
```
种子0模型使用/权重/误差/Q等原生探针：
```json
{
  "TRAIN_e_mve_s0": {
    "finished": {
      "steps": 4000,
      "updates": 7602,
      "wall_s": 57392.00274705887,
      "replay": {
        "transitions": 4000,
        "with_candidates": 3314,
        "candidates": 25848,
        "entry_candidates": 9980,
        "reason": {
          "off": 63,
          "entry_calibration": 443,
          "stage": 2628,
          "entry": 732,
          "not_comparable": 134
        }
      }
    },
    "probes": 305,
    "probes_with_model_rows": 285,
    "w_mean": 0.821079376915045,
    "abs_err_median": 1.7386856079101562,
    "sigma_last": 15.48238208533425,
    "model_td_last": 205.16969299316406,
    "q_pi_last_decile": 36.604247856140134
  },
  "TRAIN_e_sac_s0": {
    "finished": {
      "steps": 4000,
      "updates": 7602,
      "wall_s": 56861.68946862221,
      "replay": {
        "transitions": 4000,
        "with_candidates": 0,
        "candidates": 0,
        "entry_candidates": 0,
        "reason": {
          "off": 4000
        }
      }
    },
    "probes": 305,
    "probes_with_model_rows": 0,
    "w_mean": null,
    "abs_err_median": null,
    "sigma_last": 13.439478830945994,
    "model_td_last": null,
    "q_pi_last_decile": 40.66578737894694
  }
}
```
两组model_stats/episodes/heartbeat原件随附；训练探索统计不替代确定性部署。训练累计回报与30秒滚动计数当前冻结版本未记录，明确MISSING，不补造。真实步数由原生heartbeat/finished提供，不把千步落盘点当瞬时准确计数。

## 证据与结束状态
本包含种子0完整240条及种子1部分96条原始评估JSON、原生B_READOUT/DECISION_B/partial读出、A两个读数、种子0所需诊断及模型SHA；模型ZIP/Replay不上传。评估先前已完成JSON原样复用，未按回报选种子或检查点。停止种子1丢失的内存统计不伪造。全部输出与实际墙钟可由finished/heartbeat、scope时间与日志回执核对；并行耗时不可简单相加。完成后停止，等待上层分析与用户新授权。
